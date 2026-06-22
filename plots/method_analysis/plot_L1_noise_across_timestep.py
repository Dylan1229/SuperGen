import json
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.cm as cm

# ---------------------------------------------------------------------------
# Configurable location of the noise-prediction profile JSONs.
# These JSONs are produced by a *profiled* generation run (not checked into the
# repo). Point NOISE_PROFILE_DIR at the directory that contains them, or set the
# NOISE_PROFILE_DIR environment variable to override without editing this file.
# Expected file used by this script: Across_timesteps_l1_distances_50steps.json
# ---------------------------------------------------------------------------
NOISE_PROFILE_DIR = os.environ.get(
    "NOISE_PROFILE_DIR",
    "/mnt/data/fanjiang/repo/SuperGen/plots/noise_prediction_profile/",
)

with open(os.path.join(NOISE_PROFILE_DIR, "Across_timesteps_l1_distances_50steps.json"), "r") as f:
    l1_distances = json.load(f)

plt.figure(figsize=(12, 7))

num_samples = len(l1_distances)

cmap = plt.get_cmap('tab10' if num_samples <= 10 else 'viridis')
colors = [cmap(i/max(1, num_samples-1)) for i in range(num_samples)]

for i, (tile_idx, distances) in enumerate(l1_distances.items()):
    distances.sort(key=lambda x: x[0])
    
    l1_vals = [d[1] for d in distances]
    
    step_indices = list(range(len(l1_vals)))
    
    color = colors[i % len(colors)]
    plt.plot(step_indices, l1_vals, marker='o', linestyle='-', label=f'Tile {tile_idx}', color=color)

plt.xlabel('Denoising Step Index')
plt.ylabel('L1 Distance')
plt.title('L1 Distance of Predicted Noise Between Consecutive Steps')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  
plt.grid(True)
plt.xlim(-0.5, 80.5)
plt.tight_layout()  
os.makedirs(os.path.join(NOISE_PROFILE_DIR, "plots"), exist_ok=True)
plt.savefig(os.path.join(NOISE_PROFILE_DIR, "plots", "Across_timesteps_l1_distances_50steps_1536_1024.png"), dpi=300, bbox_inches='tight')
plt.show()