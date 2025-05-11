
#!/usr/bin/env python3
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─── CONFIG ────────────────────────────────────────────────────────────────
# json_path = 'output/flux.1-dev/image_reward/overall_score/20250507_123802/score_output.json'
# json_path = 'output/flux.1-dev/image_reward/overall_score/20250507_125421/score_output.json'  # flux full send run
json_path = 'output/sdxl-base/image_reward/overall_score/20250507_141459/score_output.json'  # sdxl full send run

early_stop_timestep = 801
# ────────────────────────────────────────────────────────────────────────────

# Load data
with open(json_path, 'r') as f:
    data = json.load(f)

# Organize seeds by round
round_to_seeds = {}
for seed_str, seed_data in data.items():
    rnd = seed_data.get('round')
    if rnd is None:
        continue
    round_to_seeds.setdefault(rnd, []).append(seed_str)

rounds = sorted(round_to_seeds.keys())
early_final_scores = []
final_best_scores = []

for rnd in rounds:
    seeds = round_to_seeds[rnd]
    # Determine the final timestep (lowest numeric key) available in this round
    # (assuming all seeds share the same set of timesteps)
    # We'll find the min numeric key among all intermediates for any seed.
    all_ts = []
    for seed in seeds:
        ts = [int(t) for t in data[seed]['intermediates'].keys()]
        all_ts.extend(ts)
    final_timestep = min(all_ts)

    # 1) Early-best seed
    early_best_seed = None
    early_best_val  = -np.inf
    for seed in seeds:
        interm = data[seed]['intermediates']
        val = interm.get(str(early_stop_timestep), {}).get('overall_score', None)
        if val is not None and val > early_best_val:
            early_best_val  = val
            early_best_seed = seed

    # 2) Final-best seed
    final_best_seed = None
    final_best_val  = -np.inf
    for seed in seeds:
        interm = data[seed]['intermediates']
        val = interm.get(str(final_timestep), {}).get('overall_score', None)
        if val is not None and val > final_best_val:
            final_best_val  = val
            final_best_seed = seed

    # 3) Look up both seeds’ scores at the final timestep
    #    (if a seed is missing that timestep, we'll record NaN)
    if early_best_seed:
        early_final_scores.append(
            data[early_best_seed]['intermediates']
                                 .get(str(final_timestep), {})
                                 .get('overall_score', np.nan)
        )
    else:
        early_final_scores.append(np.nan)

    if final_best_seed:
        final_best_scores.append(
            data[final_best_seed]['intermediates']
                                 .get(str(final_timestep), {})
                                 .get('overall_score', np.nan)
        )
    else:
        final_best_scores.append(np.nan)

# Find the global minimum across both sets
all_scores = early_final_scores + final_best_scores
baseline = min(all_scores)

# Recompute heights relative to that baseline
early_heights = [s - baseline for s in early_final_scores]
full_heights  = [s - baseline for s in final_best_scores]


# ─── PLOT ───────────────────────────────────────────────────────────────────
x = np.arange(len(rounds))
n_samples = [2**r for r in rounds]
width = 0.35

x = np.arange(len(rounds))
n_samples = [2**r for r in rounds]
width = 0.35

plt.figure(figsize=(8,5))
plt.bar(x - width/2, early_heights, width, bottom=baseline,
        label=f"Early stopped @{early_stop_timestep}")
plt.bar(x + width/2, full_heights, width, bottom=baseline,
        label="Full rollout")

plt.xlabel("# Initial Noise Samples")
plt.ylabel("CLIP Score")
plt.title("CLIP Scores (SDXL)")
plt.xticks(x, n_samples)
plt.legend()
plt.tight_layout()

out_png = os.path.join(os.path.dirname(json_path), "early_vs_final_best.png")
plt.savefig(out_png, dpi=300)
print(f"Saved grouped bar chart to {out_png}")
