import json
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.cm as cm
from matplotlib.colors import Normalize

# ---------------------------------------------------------------------------
# Configurable location of the noise-prediction profile JSONs.
# These JSONs are produced by a *profiled* generation run (not checked into the
# repo). Point NOISE_PROFILE_DIR at the directory that contains them, or set the
# NOISE_PROFILE_DIR environment variable to override without editing this file.
# Expected file used by this script: tile_to_tile_l1_distances_50steps.json
# ---------------------------------------------------------------------------
NOISE_PROFILE_DIR = os.environ.get(
    "NOISE_PROFILE_DIR",
    "/mnt/data/fanjiang/repo/SuperGen/plots/noise_prediction_profile/",
)

# Load tile-to-tile comparison data
with open(os.path.join(NOISE_PROFILE_DIR, "tile_to_tile_l1_distances_50steps.json"), "r") as f:
    tile_to_tile_data = json.load(f)

# Create output directory
output_dir = os.path.join(NOISE_PROFILE_DIR, "plots", "tile_to_tile_50steps_6samples")
os.makedirs(output_dir, exist_ok=True)

# Sort timesteps for consistent analysis
timesteps = sorted([int(ts) for ts in tile_to_tile_data.keys()])

# Get list of all unique tiles across all timesteps
all_tiles = set()
for ts in timesteps:
    ts_str = str(ts)
    for pair_data in tile_to_tile_data[ts_str]:
        all_tiles.add(str(pair_data['tile1']))
        all_tiles.add(str(pair_data['tile2']))
all_tiles = sorted(list(all_tiles))

# Helper function to parse tile position string to tuple
def parse_tile_pos(pos_str):
    return tuple(map(int, pos_str.strip('()').split(', ')))

# 1. Create heatmap visualization for specified timesteps
def create_heatmap(timestep):
    ts_str = str(timestep)
    if ts_str not in tile_to_tile_data:
        print(f"Timestep {timestep} not found in data")
        return
    
    # Get all unique tiles in this timestep
    tiles_in_timestep = set()
    for pair_data in tile_to_tile_data[ts_str]:
        tiles_in_timestep.add(str(pair_data['tile1']))
        tiles_in_timestep.add(str(pair_data['tile2']))
    tiles_in_timestep = sorted(list(tiles_in_timestep))
    
    # Create empty matrix for L1 distances
    n_tiles = len(tiles_in_timestep)
    heatmap_data = np.zeros((n_tiles, n_tiles))
    
    # Fill in the L1 distances
    tile_to_idx = {tile: i for i, tile in enumerate(tiles_in_timestep)}
    for pair_data in tile_to_tile_data[ts_str]:
        i = tile_to_idx[str(pair_data['tile1'])]
        j = tile_to_idx[str(pair_data['tile2'])]
        heatmap_data[i, j] = pair_data['l1_distance']
        heatmap_data[j, i] = pair_data['l1_distance']  # Mirror the matrix
    
    plt.figure(figsize=(10, 8))
    
    # Create heatmap using matplotlib instead of seaborn
    im = plt.imshow(heatmap_data, cmap='viridis')
    
    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label('L1 Distance')
    
    # Add labels and ticks
    plt.xticks(np.arange(n_tiles), tiles_in_timestep, rotation=45)
    plt.yticks(np.arange(n_tiles), tiles_in_timestep)
    
    # Add text annotations with values
    # for i in range(n_tiles):
    #     for j in range(n_tiles):
    #         text_color = 'white' if heatmap_data[i, j] > np.mean(heatmap_data) else 'black'
    #         plt.text(j, i, f"{heatmap_data[i, j]:.3f}", 
    #                  ha="center", va="center", color=text_color, fontsize=8)
    
    plt.title(f'L1 Distance Between Tiles at Timestep {timestep}')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/tile_to_tile_heatmap_timestep_{timestep}.png", dpi=300, bbox_inches='tight')
    plt.close()

# 2. Create scatter plot showing relationship between spatial distance and L1 distance
def create_spatial_l1_scatter():
    plt.figure(figsize=(12, 8))
    
    for ts in timesteps:
        ts_str = str(ts)
        if ts_str not in tile_to_tile_data:
            continue
            
        spatial_distances = []
        l1_distances = []
        
        for pair_data in tile_to_tile_data[ts_str]:
            spatial_distances.append(pair_data['spatial_distance'])
            l1_distances.append(pair_data['l1_distance'])
        
        plt.scatter(spatial_distances, l1_distances, alpha=0.7, label=f'Timestep {ts}')
    
    plt.xlabel('Spatial Distance Between Tiles')
    plt.ylabel('L1 Distance Between Noise Predictions')
    plt.title('Relationship Between Spatial Distance and L1 Distance')
    # plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/spatial_vs_l1_distance_scatter.png", dpi=300, bbox_inches='tight')
    plt.close()

# 3. Create grid visualization showing tile layout and compare L1 distances
def create_grid_visualization(timestep):
    ts_str = str(timestep)
    if ts_str not in tile_to_tile_data:
        print(f"Timestep {timestep} not found in data")
        return
    
    # Get all tiles and their positions
    tile_positions = {}
    for pair_data in tile_to_tile_data[ts_str]:
        tile_positions[str(pair_data['tile1'])] = pair_data['tile1_pos']
        tile_positions[str(pair_data['tile2'])] = pair_data['tile2_pos']
    
    # Extract grid dimensions
    all_positions = list(tile_positions.values())
    rows = set([pos[0] for pos in all_positions])
    cols = set([pos[1] for pos in all_positions])
    min_row, max_row = min(rows), max(rows)
    min_col, max_col = min(cols), max(cols)
    
    # Create the figure with proper layout for colorbar
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Draw the tiles
    for tile, pos in tile_positions.items():
        row, col = pos
        ax.plot(col, row, 'o', markersize=15, alpha=0.7)
        ax.text(col, row, tile, fontsize=8, ha='center', va='center')
    
    # Draw connections between tiles, with width based on L1 distance
    max_l1 = max([pair_data['l1_distance'] for pair_data in tile_to_tile_data[ts_str]])
    min_l1 = min([pair_data['l1_distance'] for pair_data in tile_to_tile_data[ts_str]])
    
    norm = Normalize(vmin=min_l1, vmax=max_l1)
    cmap = plt.get_cmap('viridis_r')  # Reversed so darker = closer
    
    for pair_data in tile_to_tile_data[ts_str]:
        tile1 = str(pair_data['tile1'])
        tile2 = str(pair_data['tile2'])
        l1_dist = pair_data['l1_distance']
        
        pos1 = tile_positions[tile1]
        pos2 = tile_positions[tile2]
        
        # Line width inversely proportional to L1 distance
        line_width = 5 * (1 - norm(l1_dist)) + 0.5
        line_color = cmap(norm(l1_dist))
        
        ax.plot([pos1[1], pos2[1]], [pos1[0], pos2[0]], 
                 linewidth=line_width, color=line_color, alpha=0.6)
    
    # Add colorbar - fixed to use the current axis
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label='L1 Distance (smaller = more similar)')
    
    ax.set_xlim(min_col-10, max_col+10)
    ax.set_ylim(min_row-10, max_row+10)
    ax.invert_yaxis()  # Invert Y axis to match typical grid layout
    ax.set_title(f'Tile Grid Visualization at Timestep {timestep}\nLines: thicker = more similar noise predictions')
    ax.grid(False)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/tile_grid_visualization_timestep_{timestep}.png", dpi=300, bbox_inches='tight')
    plt.close()

# 4. Average L1 distance vs timestep for all tile pairs
def plot_average_l1_by_timestep():
    avg_l1_by_timestep = []
    
    for ts in timesteps:
        ts_str = str(ts)
        if ts_str not in tile_to_tile_data or not tile_to_tile_data[ts_str]:
            continue
            
        avg_l1 = np.mean([pair_data['l1_distance'] for pair_data in tile_to_tile_data[ts_str]])
        avg_l1_by_timestep.append((ts, avg_l1))
    
    avg_l1_by_timestep.sort(key=lambda x: x[0])
    
    plt.figure(figsize=(12, 6))
    plt.plot([x[0] for x in avg_l1_by_timestep], 
             [x[1] for x in avg_l1_by_timestep], 
             marker='o', linestyle='-')
    plt.xlabel('Timestep')
    plt.ylabel('Average L1 Distance Between Tile Pairs')
    plt.title('Average Tile-to-Tile Noise Prediction Similarity Over Time')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/average_l1_by_timestep.png", dpi=300, bbox_inches='tight')
    plt.close()

# Execute visualizations
if __name__ == "__main__":
    print("Generating visualizations for tile-to-tile noise prediction analysis...")
    
    # Create heatmaps for representative timesteps
    for ts in timesteps:
        create_heatmap(ts)
        create_grid_visualization(ts)
    
    # Create scatter plot
    create_spatial_l1_scatter()
    
    # Create average L1 by timestep plot
    plot_average_l1_by_timestep()
    
    print(f"Visualizations saved to {output_dir}") 