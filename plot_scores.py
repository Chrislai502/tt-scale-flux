import os
import json
import numpy as np
import matplotlib

# Use Agg backend for SSH environments
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Configuration: adjust these as needed
# base_path = 'output/flux.1-dev/image_reward/overall_score/20250507_125421'  # flux full send run
base_path = 'output/sdxl-base/image_reward/overall_score/20250507_141459'  # sdxl full send run
json_path = os.path.join(base_path, 'score_output.json')

show_first_stopping_point = False   # whether to highlight the first plateau
show_all_stopping_points  = False   # whether to highlight all plateaus
window_size   = 10                  # number of consecutive timesteps in each window
var_threshold = 2e-3                # max variance in window to qualify as plateau

# Load data
with open(json_path, 'r') as f:
    data = json.load(f)

plt.figure(figsize=(8, 5))

for seed_str, seed_data in data.items():
    # grab the intermediates map
    intermediates = seed_data.get('intermediates', {})
    if not intermediates:
        continue

    # sorted timesteps & score array
    timesteps = sorted(int(t) for t in intermediates.keys())
    scores    = np.array([intermediates[str(t)]['overall_score'] for t in timesteps])

    # plot and remember the line color
    line, = plt.plot(timesteps, scores) #, label=f"Seed {seed_str} (round {seed_data.get('round')})")
    c = line.get_color()

    # rolling-window variances
    variances = np.array([
        np.var(scores[i:i+window_size])
        for i in range(len(scores) - window_size + 1)
    ])
    centers = np.where(variances < var_threshold)[0] + window_size // 2

    if show_first_stopping_point and len(centers):
        idx = centers[-1]
        plt.scatter(
            timesteps[idx], scores[idx],
            marker='x', s=75, color=c,
            label='_nolegend_'
        )

    if show_all_stopping_points and len(centers):
        xs = [timesteps[i] for i in centers]
        ys = [scores[i] for i in centers]
        plt.scatter(
            xs, ys,
            marker='x', s=40, color='gray',
            label='_nolegend_'
        )

plt.xlabel('Timestep')
plt.ylabel('Overall Score')
plt.title('CLIP Intermediate Scores (SDXL)')
plt.legend(loc='best', title=f"w={window_size}, var<{var_threshold}" if (show_first_stopping_point or show_all_stopping_points) else None)
plt.tight_layout()
plt.gca().invert_xaxis()

plt.axvline(x=801, color='red', linestyle='--', linewidth=2, label='Early Stop Timestep')

# Save to file
out_png = os.path.join(base_path, 'scores.png')
plt.savefig(out_png, dpi=300)
print(f"Saved plateau-annotated plot to {out_png}")
