#!/usr/bin/env python3
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─── CONFIG ────────────────────────────────────────────────────────────────
# base_path = 'output/flux.1-dev/image_reward/overall_score/20250507_125421/'  # flux full send run
base_path = 'output/sdxl-base/image_reward/overall_score/20250507_141459'  # sdxl full send run

json_path = os.path.join(base_path, 'score_output.json')
early_stop_timestep = 801
# ────────────────────────────────────────────────────────────────────────────

# Load the merged JSON
with open(json_path, 'r') as f:
    data = json.load(f)

# Group seeds by round
round_to_seeds = {}
for seed_str, seed_data in data.items():
    rnd = seed_data.get('round')
    if rnd is None:
        continue
    round_to_seeds.setdefault(rnd, []).append(seed_str)

rounds = sorted(round_to_seeds.keys())
early_times = []
full_times  = []

for rnd in rounds:
    seeds = round_to_seeds[rnd]
    sum_early = 0.0
    sum_full  = 0.0

    for seed in seeds:
        interm = data[seed]['intermediates']
        # 1) Early time
        t_early = interm.get(str(early_stop_timestep), {}).get('time_so_far')
        if t_early is not None:
            sum_early += t_early

        # 2) Full time at the seed's lowest timestep
        ts_keys = [int(t) for t in interm.keys()]
        if ts_keys:
            final_t = min(ts_keys)
            t_full = interm[str(final_t)]['time_so_far']
            sum_full += t_full

    early_times.append(sum_early)
    full_times.append(sum_full)

# ─── PLOT ───────────────────────────────────────────────────────────────────
x = np.arange(len(rounds))
width = 0.35
labels = [str(2**r) for r in rounds]  # number of noises = 2^round

plt.figure(figsize=(8,5))
plt.bar(x - width/2, early_times, width, label=f"Early stopped @ {early_stop_timestep}")
plt.bar(x + width/2, full_times,  width, label="Full rollout")

plt.xlabel("# Initial Noise Samples")
plt.ylabel("Wall-clock time (s)")
plt.title("Search Time (SDXL)")
plt.xticks(x, labels)
plt.legend()
plt.tight_layout()

out_png = os.path.join(base_path, "total_times.png")
plt.savefig(out_png, dpi=300)
print(f"Saved grouped time chart to {out_png}")
