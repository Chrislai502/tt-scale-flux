#!/usr/bin/env python3
import os
import json
import numpy as np
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─── ARGPARSE ───────────────────────────────────────────────────────────────
default_base_path = 'output/sdxl-base/image_reward/overall_score/20250507_141459'  # overwritten by --base_path
parser = argparse.ArgumentParser(
    description='Generate CLIP and timing plots from diffusion score JSON'
)
parser.add_argument(
    '--base_path',
    type=str,
    default=default_base_path,
    help='Directory containing score_output.json'
)
parser.add_argument(
    '--early_stop_timestep',
    type=int,
    default=781,
    help='Timestep to use for early-stop metrics'
)
parser.add_argument(
    '--window_size',
    type=int,
    default=10,
    help='Window size for variance-based plateau detection'
)
parser.add_argument(
    '--var_threshold',
    type=float,
    default=2e-3,
    help='Variance threshold for plateau detection'
)
parser.add_argument(
    '--show_first_stopping_point',
    action='store_true',
    help='Highlight only the first plateau point'
)
parser.add_argument(
    '--show_all_stopping_points',
    action='store_true',
    help='Highlight all plateau points'
)

args = parser.parse_args()
base_path                 = args.base_path
json_path                 = os.path.join(base_path, 'score_output.json')
early_stop_timestep       = args.early_stop_timestep
window_size               = args.window_size
var_threshold             = args.var_threshold
show_first_stopping_point = args.show_first_stopping_point
show_all_stopping_points  = args.show_all_stopping_points

# ─── PREPARE OUTPUT ────────────────────────────────────────────────────────
plots_dir = os.path.join(base_path, 'plots')
os.makedirs(plots_dir, exist_ok=True)
with open(json_path, 'r') as f:
    data = json.load(f)

# ─── PLOT #1: Intermediate CLIP Scores ─────────────────────────────────────
plt.figure(figsize=(8, 5))
for seed_str, seed_data in data.items():
    interm = seed_data.get('intermediates', {})
    if not interm:
        continue

    timesteps = sorted(int(t) for t in interm.keys())
    scores    = np.array([interm[str(t)]['overall_score'] for t in timesteps])

    line, = plt.plot(timesteps, scores) #, label=seed_str)
    c = line.get_color()

    # rolling-window variance
    variances = np.array([
        np.var(scores[i:i+window_size])
        for i in range(len(scores) - window_size + 1)
    ])
    centers = np.where(variances < var_threshold)[0] + window_size//2

    if show_first_stopping_point and centers.size:
        idx = centers[-1]
        plt.scatter(
            timesteps[idx], scores[idx],
            marker='x', s=75, color=c, label='_nolegend_'
        )

    if show_all_stopping_points and centers.size:
        xs = [timesteps[i] for i in centers]
        ys = [scores[i] for i in centers]
        plt.scatter(xs, ys, marker='x', s=40,
                    color='gray', label='_nolegend_')

plt.xlabel('Timestep')
plt.ylabel('Overall Score')
plt.title('CLIP Intermediate Scores')
plt.gca().invert_xaxis()
plt.axvline(
    x=early_stop_timestep,
    color='red', linestyle='--', linewidth=2,
    label=f'Early stop @ {early_stop_timestep}'
)
plt.legend()
plt.tight_layout()
out1 = os.path.join(plots_dir, 'intermediate_scores.png')
plt.savefig(out1, dpi=300)
plt.close()

# ─── GROUP SEEDS BY ROUND ──────────────────────────────────────────────────
round_to_seeds = {}
for seed_str, seed_data in data.items():
    rnd = seed_data.get('round')
    if rnd is not None:
        round_to_seeds.setdefault(rnd, []).append(seed_str)
rounds = sorted(round_to_seeds.keys())
n_samples = [2**r for r in rounds]
x = np.arange(len(rounds))
width = 0.35

# ─── PLOT #2: Final CLIP Scores by Seed Selection ──────────────────────────
early_final_scores = []
full_final_scores  = []

for rnd in rounds:
    seeds = round_to_seeds[rnd]

    # pick best seed at early_stop_timestep
    best_seed_early, best_early_val = None, -np.inf
    for sd in seeds:
        v = data[sd]['intermediates'] \
                .get(str(early_stop_timestep), {}) \
                .get('overall_score')
        if v is not None and v > best_early_val:
            best_early_val, best_seed_early = v, sd

    # seed_early's own final timestep & score
    if best_seed_early is not None:
        ts_early = [int(t) for t in data[best_seed_early]['intermediates']]
        final_ts_early = min(ts_arly for ts_arly in ts_early)
        early_final_scores.append(
            data[best_seed_early]['intermediates'][str(final_ts_early)]['overall_score']
        )
    else:
        early_final_scores.append(np.nan)

    # pick best seed by its own final timestep
    best_seed_full, best_full_val = None, -np.inf
    for sd in seeds:
        ts = [int(t) for t in data[sd]['intermediates']]
        if not ts:
            continue
        seed_final = min(ts)
        v = data[sd]['intermediates'][str(seed_final)]['overall_score']
        if v > best_full_val:
            best_full_val, best_seed_full = v, sd

    if best_seed_full is not None:
        full_final_scores.append(best_full_val)
    else:
        full_final_scores.append(np.nan)

# normalize bottoms so negatives start at same baseline
baseline = min([s for s in (early_final_scores + full_final_scores) if not np.isnan(s)]) - 0.3
early_h = [s - baseline for s in early_final_scores]
full_h  = [s - baseline for s in full_final_scores]

plt.figure(figsize=(8,5))
plt.bar(x - width/2, early_h, width, bottom=baseline,
        label=f"Early stop @{early_stop_timestep}")
plt.bar(x + width/2, full_h,  width, bottom=baseline,
        label="Full rollout")

plt.xlabel("# Initial Noise Samples")
plt.ylabel("CLIP Score")
plt.title("CLIP Scores")
plt.xticks(x, n_samples)
plt.legend()
plt.tight_layout()
out2 = os.path.join(plots_dir, 'final_scores.png')
plt.savefig(out2, dpi=300)
plt.close()

# ─── PLOT #3: Total Generation Time per Round ──────────────────────────────
early_times = []
full_times  = []

for rnd in rounds:
    seeds = round_to_seeds[rnd]
    sum_e, sum_f = 0.0, 0.0
    for sd in seeds:
        interm = data[sd]['intermediates']
        # early-stop time
        t_e = interm.get(str(early_stop_timestep), {}).get('time_so_far', 0.0)
        sum_e += t_e
        # full time at seed's own final timestep
        ts = [int(t) for t in interm.keys()]
        if ts:
            f_t = min(ts)
            sum_f += interm[str(f_t)]['time_so_far']
    early_times.append(sum_e)
    full_times.append(sum_f)

plt.figure(figsize=(8,5))
plt.bar(x - width/2, early_times, width,
        label=f"Early stop @ {early_stop_timestep}")
plt.bar(x + width/2, full_times,  width,
        label="Full rollout")

plt.xlabel("# Initial Noise Samples")
plt.ylabel("Wall-clock Time (s)")
plt.title("Search Time")
plt.xticks(x, n_samples)
plt.legend()
plt.tight_layout()
out3 = os.path.join(plots_dir, 'total_times.png')
plt.savefig(out3, dpi=300)
plt.close()

print("Saved plots to:", plots_dir)
