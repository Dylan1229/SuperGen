import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from utils.distributed import DistributedManager
    
class TileStdTracker:
    def __init__(self, num_tiles: int, update_interval: int = 5, dist_manager: "DistributedManager" = None):
        self.num_tiles = num_tiles
        self.update_interval = update_interval
        self.dist_manager = dist_manager
        
        # Welford's algorithm state for each tile
        self.allocate()
        
        # Track last update step
        self.last_update_step = -1
        
        # For visualization profiling
        self.snapshot_steps = []
        self.tile_snapshots = {}
    
    def allocate(self):
        self.count = self.dist_manager.get_local_buffer("std_count")
        self.mean = self.dist_manager.get_local_buffer("std_mean")
        self.m2 = self.dist_manager.get_local_buffer("std_m2")
        self.std_values = self.dist_manager.get_local_buffer("std_values")

        self.count.zero_()
        self.mean.zero_()
        self.m2.zero_()
        self.std_values.zero_()

        self.count_global = self.dist_manager.get_global_buffer("std_count")
        self.mean_global = self.dist_manager.get_global_buffer("std_mean")
        self.m2_global = self.dist_manager.get_global_buffer("std_m2")
        self.std_values_global = self.dist_manager.get_global_buffer("std_values")
    
    def update(self, tile_idx: int, noise_pred: torch.Tensor) -> None:
        """Update std tracker for a specific tile using Welford's algorithm"""
        # Compute mean absolute value of noise prediction
        noise_mean = torch.mean(torch.abs(noise_pred)).item()

        offset = self.dist_manager.convert_tile_idx_to_local_offset(tile_idx)
        
        # Welford's online algorithm
        self.count[offset] += 1
        delta = noise_mean - self.mean[offset]
        self.mean[offset] += delta / self.count[offset]
        delta2 = noise_mean - self.mean[offset]
        self.m2[offset] += delta * delta2
        
        # Calculate std
        if self.count[offset] > 1:
            variance = self.m2[offset] / (self.count[offset] - 1)
            self.std_values[offset] = torch.sqrt(variance)
    
    def _get_all_std_values_as_list(self) -> List[float]:
        l = [0.0] * self.num_tiles
        for cur_rank, index_list in enumerate(self.dist_manager.global_indices):
            rank_value_list = self.std_values_global[cur_rank].tolist()
            for i, idx in enumerate(index_list):
                l[idx] = rank_value_list[i]
        return l
    
    def should_update_cache_thresholds(self, step: int) -> bool:
        """Check if cache thresholds should be updated"""
        return (step - self.last_update_step) >= self.update_interval
    
    def get_normalized_stds(self) -> List[float]:
        """Get normalized std values in [0,1] range"""
        std_values = self._get_all_std_values_as_list()

        if not any(std_values):
            return [0.0] * self.num_tiles
        
        std_array = np.array(std_values)
        min_std = np.min(std_array)
        max_std = np.max(std_array)
        
        if max_std == min_std:
            return [0.0] * self.num_tiles
        
        normalized = (std_array - min_std) / (max_std - min_std)
        return normalized.tolist()
    
    def compute_cache_thresholds(self, base_thresh: float, alpha: float = 1.0, mode: str = "original") -> List[float]:
        """
        Compute cache thresholds based on std values
        
        Args:
            base_thresh: Base threshold value
            alpha: Scaling factor for adjustment
            mode: "max_min_adjust" (max increase, min decrease), 
                  "top_selective" (only top 1-2 increase)
        """
        # All following process depends on `norm_stds`
        norm_stds = self.get_normalized_stds()
        thresholds = []
                
        if mode == "max_min_adjust":
            if not norm_stds or all(x == 0 for x in norm_stds):
                return [base_thresh] * self.num_tiles
                
            max_idx = np.argmax(norm_stds)
            min_idx = np.argmin(norm_stds)
            
            for i, std_val in enumerate(norm_stds):
                if i == max_idx:
                    thr = base_thresh * (1 + alpha)
                elif i == min_idx and max_idx != min_idx:
                    thr = base_thresh * (1 - alpha * 0.5)
                else:
                    thr = base_thresh
                thresholds.append(thr)
                
        elif mode == "top_selective":
            if not norm_stds or all(x == 0 for x in norm_stds):
                return [base_thresh] * self.num_tiles
                
            std_array = np.array(norm_stds)
            sorted_indices = np.argsort(std_array)[::-1] 
            
            for i, std_val in enumerate(norm_stds):
                if i == sorted_indices[0]:  
                    thr = base_thresh * (1 + alpha)
                elif len(sorted_indices) > 1 and i == sorted_indices[1]: 
                    thr = base_thresh * (1 + alpha * 0.5) 
                else:
                    thr = base_thresh
                thresholds.append(thr)
        
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        return thresholds
    
    def update_last_step(self, step: int) -> None:
        """Update last update step"""
        self.last_update_step = step
    