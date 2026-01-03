import torch
from typing import Tuple, Dict, Any, TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from .distributed import DistributedManager

logger = logging.getLogger(__name__)

def get_dimension_slices_and_sizes(begin, end, size):

    slices = []
    sizes = [] 
    current_pos = begin
    
    while current_pos < end:
        start_idx = current_pos % size # Start index in the current tile
        next_boundary = ((current_pos // size) + 1) * size # The start position of the next tile
        end_pos = min(end, next_boundary) # The end position of the current tile
        length = end_pos - current_pos
        end_idx = (start_idx + length) % size # End index in the current tile

        if end_idx > start_idx:
            slices.append(slice(start_idx, end_idx))
            sizes.append(end_idx - start_idx)
        else: 
            slices.append(slice(start_idx, size))
            sizes.append(size - start_idx)
            if end_idx > 0:
                slices.append(slice(0, end_idx))
                sizes.append(end_idx)
        current_pos = end_pos
    
    return slices, sizes

class TiledLatentTensor2D:
    def __init__(self, latent_tensor: torch.Tensor):
        self.torch_latent = latent_tensor.clone()
        assert self.torch_latent.ndim == 5
        self.batch_size = self.torch_latent.shape[0] 
        self.num_frames = self.torch_latent.shape[1]
        self.channels = self.torch_latent.shape[2]
        self.height = self.torch_latent.shape[3]
        self.width = self.torch_latent.shape[4]

    def get_shape(self):
        return self.torch_latent.shape
    
    def get_window_latent(self, top: int = None, bottom: int = None, left: int = None, right: int = None):
        if top is None:
            top = 0
        if bottom is None:
            bottom = self.height
        if left is None:
            left = 0
        if right is None:
            right = self.width
            
        # Ensure the indices are within the valid range
        assert 0 <= top < bottom <= self.height * 2, f"Invalid top {top} and bottom {bottom}"
        assert 0 <= left < right <= self.width * 2, f"Invalid left {left} and right {right}"
        
        height_slices, height_sizes = get_dimension_slices_and_sizes(top, bottom, self.height)
        width_slices, width_sizes = get_dimension_slices_and_sizes(left, right, self.width)

        # Get the parts of the latent tensor
        parts = []
        for h_slice in height_slices:
            row_parts = []
            for w_slice in width_slices:
                part = self.torch_latent[:, :, :, h_slice, w_slice]
                row_parts.append(part)
            row = torch.cat(row_parts, dim=4)
            parts.append(row)
        desired_latent = torch.cat(parts, dim=3)
        
        return desired_latent
    
    def set_window_latent(self, input_latent: torch.Tensor,

                          top: int = None,
                          bottom: int = None,
                          left: int = None,
                          right: int = None):
        if top is None:
            top = 0
        if bottom is None:
            bottom = self.height
        if left is None:
            left = 0
        if right is None:
            right = self.width

        assert 0 <= top < bottom <= self.height * 2, f"Invalid top {top} and bottom {bottom}"
        assert 0 <= left < right <= self.width * 2, f"Invalid left {left} and right {right}"
        assert bottom - top <= self.height, f"warp should not occur"
        assert right - left <= self.width, f"warp should not occur"

       # Calculate the target latent tensor
        target_height = bottom - top if bottom <= self.height else (self.height - top) + (bottom % self.height)
        target_width = right - left if right <= self.width else (self.width - left) + (right % self.width)

        width_slices, width_sizes = get_dimension_slices_and_sizes(left, right, self.width)
        height_slices, height_sizes = get_dimension_slices_and_sizes(top, bottom, self.height)

        target_height = sum(height_sizes)
        target_width = sum(width_sizes)
        # Check the shape of the input latent tensor
        assert input_latent.shape[3:] == (target_height, target_width), f"Input latent shape {input_latent.shape[3:]} does not match target window shape {(target_height, target_width)}"
        size_input = input_latent.dtype.itemsize
        size_latent = self.torch_latent.dtype.itemsize
        assert size_latent >= size_input, f"{size_input=} {size_latent=}"

        # Write the parts of the latent tensor
        h_start = 0
        for h_slice, h_size in zip(height_slices, height_sizes):
            w_start = 0
            for w_slice, w_size in zip(width_slices, width_sizes):
                input_part = input_latent[:, :, :, h_start:h_start+h_size, w_start:w_start+w_size]
                self.torch_latent[:, :, :, h_slice, w_slice] = input_part
                w_start += w_size
            h_start += h_size
    
    def zero_(self):
        self.torch_latent.zero_()

class TileNoiseAggregator2D:
    """
    A class to handle noise prediction accumulation with overlap support.
    Supports weighted averaging in overlapping regions.
    """
    def __init__(self, noise_shape, device, dtype, dist_manager: "DistributedManager", fusion_method="weighted_average"):
        """
        Args:
            noise_shape: Shape of the noise tensor (B, F, C, H, W)
            device: Device to store tensors on
            dtype: Data type for tensors
            fusion_method: Method to fuse overlapping predictions 
                          - "weighted_average": weighted average based on contribution count
                          - "simple_average": simple average
                          - "last_wins": last prediction overwrites (current behavior)
        """
        self.batch_size = noise_shape[0]
        self.num_frames = noise_shape[1] 
        self.channels = noise_shape[2]
        self.height = noise_shape[3]
        self.width = noise_shape[4]
        self.device = device
        self.dtype = dtype
        self.fusion_method = fusion_method

        self.dist_manager = dist_manager
        self.tile_shape = None
        
        # Initialize accumulation tensors
        self.noise_accumulator = torch.zeros(noise_shape, device=device, dtype=dtype)
        self.contribution_count = torch.zeros(noise_shape, device=device, dtype=torch.float32)
    
    def set_tile_shape(self, shape):
        self.tile_shape = shape
        
    def get_shape(self):
        return (self.batch_size, self.num_frames, self.channels, self.height, self.width)
    
    def reset(self):
        """Reset accumulator for next timestep"""
        self.noise_accumulator.zero_()
        self.contribution_count.zero_()
    
    def _check_and_slice(
        self,
        top: int,
        bottom: int,
        left: int,
        right: int,
    ):
        if top is None:
            top = 0
        if bottom is None:
            bottom = self.height
        if left is None:
            left = 0
        if right is None:
            right = self.width
            
        # Validate coordinates
        assert 0 <= top < bottom <= self.height * 2, f"Invalid top {top} and bottom {bottom}"
        assert 0 <= left < right <= self.width * 2, f"Invalid left {left} and right {right}"
        assert bottom - top <= self.height, f"warp should not occur"
        assert right - left <= self.width, f"warp should not occur"
        
        # Get slices for ring topology
        height_slices, height_sizes = get_dimension_slices_and_sizes(top, bottom, self.height)
        width_slices, width_sizes = get_dimension_slices_and_sizes(left, right, self.width)

        return height_slices, height_sizes, width_slices, width_sizes
        

    def set_tile_noise(self, noise_pred: torch.Tensor, 
                      top: int = None, bottom: int = None, 
                      left: int = None, right: int = None,):
        height_slices, height_sizes, width_slices, width_sizes = self._check_and_slice(top, bottom, left, right)
        
        # Verify input shape matches expected tile size
        target_height = sum(height_sizes)
        target_width = sum(width_sizes)
        assert noise_pred.shape[3:] == (target_height, target_width), \
            f"Noise shape {noise_pred.shape[3:]} does not match tile shape {(target_height, target_width)}"
        
        # Cast noise_pred to accumulator dtype for compatibility with different diffusers versions
        if noise_pred.dtype != self.noise_accumulator.dtype:
            noise_pred = noise_pred.to(self.noise_accumulator.dtype)
        
        h_start = 0
        for h_slice, h_size in zip(height_slices, height_sizes):
            w_start = 0
            for w_slice, w_size in zip(width_slices, width_sizes):
                noise_part = noise_pred[:, :, :, h_start:h_start+h_size, w_start:w_start+w_size]
                self.noise_accumulator[:, :, :, h_slice, w_slice] = noise_part
                w_start += w_size
            h_start += h_size
    
    
    def add_tile_noise(self, noise_pred: torch.Tensor, 
                      top: int = None, bottom: int = None, 
                      left: int = None, right: int = None,
                      weight: float = 1.0):
        """
        Add noise prediction for a specific tile with optional overlap handling
        
        Args:
            noise_pred: Predicted noise for the tile [B, F, C, tile_H, tile_W]
            top, bottom, left, right: Tile coordinates (supporting ring topology)
            weight: Weight for this tile's contribution (for weighted fusion)
        """
        height_slices, height_sizes, width_slices, width_sizes = self._check_and_slice(top, bottom, left, right)
        
        # Verify input shape matches expected tile size
        target_height = sum(height_sizes)
        target_width = sum(width_sizes)
        assert noise_pred.shape[3:] == (target_height, target_width), \
            f"Noise shape {noise_pred.shape[3:]} does not match tile shape {(target_height, target_width)}"
        
        # Cast noise_pred to accumulator dtype for compatibility with different diffusers versions
        if noise_pred.dtype != self.noise_accumulator.dtype:
            noise_pred = noise_pred.to(self.noise_accumulator.dtype)
        
        # Add noise prediction to accumulator
        h_start = 0
        for h_slice, h_size in zip(height_slices, height_sizes):
            w_start = 0
            for w_slice, w_size in zip(width_slices, width_sizes):
                noise_part = noise_pred[:, :, :, h_start:h_start+h_size, w_start:w_start+w_size]
                
                if self.fusion_method == "last_wins":
                    # Simply overwrite (current behavior)
                    self.noise_accumulator[:, :, :, h_slice, w_slice] = noise_part
                    self.contribution_count[:, :, :, h_slice, w_slice] = 1.0
                elif self.fusion_method == "weighted_average":
                    # Weighted accumulation
                    self.noise_accumulator[:, :, :, h_slice, w_slice] += noise_part * weight
                    self.contribution_count[:, :, :, h_slice, w_slice] += weight
                elif self.fusion_method == "simple_average":
                    # Simple accumulation for averaging
                    self.noise_accumulator[:, :, :, h_slice, w_slice] += noise_part
                    self.contribution_count[:, :, :, h_slice, w_slice] += 1.0
                    
                w_start += w_size
            h_start += h_size
    
    def get_tile_fused_noise(self, 
                      top: int = None, bottom: int = None, 
                      left: int = None, right: int = None,):
        height_slices, height_sizes, width_slices, width_sizes = self._check_and_slice(top, bottom, left, right)
        
        # Verify input shape matches expected tile size
        target_height = sum(height_sizes)
        target_width = sum(width_sizes)
        assert self.tile_shape[3:] == (target_height, target_width)
        
        parts = []
        for h_slice in height_slices:
            row_parts = []
            for w_slice in width_slices:
                noise_part = self.noise_accumulator[:, :, :, h_slice, w_slice]
                if self.fusion_method == "last_wins":
                    row_parts.append(noise_part)
                else:
                    contri_part = self.contribution_count[:, :, :, h_slice, w_slice]
                    safe_count = torch.clamp(contri_part, min=1e-8).to(dtype=self.dtype)
                    fused_noise = noise_part / safe_count
                    row_parts.append(fused_noise)
            row = torch.cat(row_parts, dim=4)
            parts.append(row)
        return torch.cat(parts, dim=3).to(dtype=self.dtype)

    def get_fused_noise(self):
        """
        Get the final fused noise prediction tensor
        
        Returns:
            Fused noise tensor [B, F, C, H, W]
        """
        raise DeprecationWarning
        if self.fusion_method == "last_wins":
            return self.noise_accumulator.clone()
        else:
            # For weighted or simple average, divide by contribution count
            # Avoid division by zero and maintain dtype consistency
            safe_count = torch.clamp(self.contribution_count, min=1e-8).to(dtype=self.dtype)
            fused_noise = self.noise_accumulator / safe_count
            return fused_noise.to(dtype=self.dtype)
            
    def get_coverage_map(self):
        """
        Get a map showing how many times each position was covered
        Useful for debugging overlap patterns
        """
        return self.contribution_count[:, 0, 0, :, :].clone()  # Just spatial coverage


class SlidingWindowConfig:
    """
    A class to manage sliding window configuration for tiled video processing.
    Provides automatic window size selection and step size calculation based on input resolution.
    """
    
    # Predefined window size mappings for common resolutions
    RESOLUTION_TO_WINDOW_SIZE = {
        (90, 160): (90, 160),    # 720p: 90, 160 (1 tile)
        (180, 320): (90, 160),    # 2K: 180, 320 (4 tiles)
        (270, 480): (90, 160),    # 4K: 270, 480 (9 tiles)
    }
    
    DEFAULT_WINDOW_SIZE = (64, 64)
    
    def __init__(self, height: int, width: int, loop_step: int = 8):
        """
        Initialize sliding window configuration.
        
        Args:
            height: Input height in latent space
            width: Input width in latent space
            loop_step: Loop step parameter for calculating step sizes
        """
        self.height = height
        self.width = width
        self.loop_step = loop_step
        
        # Auto-select window size based on resolution
        self.window_size = self._get_window_size(height, width)
        
        # Calculate derived parameters
        self.num_windows_h = height // self.window_size[0]
        self.num_windows_w = width // self.window_size[1]
        self.total_windows = self.num_windows_h * self.num_windows_w
        
        # Calculate step sizes
        self.step_size_h = self._calculate_step_size(self.window_size[0], self.num_windows_h, loop_step)
        self.step_size_w = self._calculate_step_size(self.window_size[1], self.num_windows_w, loop_step)
        
        # Log configuration
        logger.info(f"SlidingWindowConfig: window_size={self.window_size}, "
                   f"num_windows=({self.num_windows_h}, {self.num_windows_w}), "
                   f"step_sizes=({self.step_size_h}, {self.step_size_w})")
    
    @classmethod
    def _get_window_size(cls, height: int, width: int) -> Tuple[int, int]:
        """
        Get appropriate window size based on input resolution.
        
        Args:
            height: Input height
            width: Input width
            
        Returns:
            Tuple of (window_height, window_width)
        """
        resolution_key = (height, width)
        
        if resolution_key in cls.RESOLUTION_TO_WINDOW_SIZE:
            window_size = cls.RESOLUTION_TO_WINDOW_SIZE[resolution_key]
            logger.info(f"Using predefined window_size: {window_size} for latent space resolution {height}x{width}")
            return window_size
        else:
            logger.info(f"Using default window_size: {cls.DEFAULT_WINDOW_SIZE} for latent space resolution {height}x{width}")
            return cls.DEFAULT_WINDOW_SIZE
    
    @staticmethod
    def _calculate_step_size(window_dim: int, num_windows: int, loop_step: int) -> int:
        """
        Calculate step size for a dimension.
        
        Args:
            window_dim: Window size in this dimension
            num_windows: Number of windows in this dimension
            loop_step: Loop step parameter
            
        Returns:
            Step size for this dimension
        """
        if num_windows == 1:
            return 0
        return window_dim // loop_step
    
    def get_window_params(self) -> Dict[str, Any]:
        """
        Get all window parameters as a dictionary.
        
        Returns:
            Dictionary containing all window configuration parameters
        """
        return {
            'window_size': self.window_size,
            'num_windows_h': self.num_windows_h,
            'num_windows_w': self.num_windows_w,
            'total_windows': self.total_windows,
            'step_size_h': self.step_size_h,
            'step_size_w': self.step_size_w,
            'latent_step_size_h': self.step_size_h,  # Alias for backward compatibility
            'latent_step_size_w': self.step_size_w,  # Alias for backward compatibility
        }
    
    def add_resolution_mapping(self, height: int, width: int, window_size: Tuple[int, int]):
        """
        Add a new resolution to window size mapping.
        
        Args:
            height: Input height
            width: Input width
            window_size: Corresponding window size tuple
        """
        self.RESOLUTION_TO_WINDOW_SIZE[(height, width)] = window_size
        logger.info(f"Added resolution mapping: {height}x{width} -> {window_size}")
    
    def __repr__(self) -> str:
        return (f"SlidingWindowConfig(resolution={self.height}x{self.width}, "
                f"window_size={self.window_size}, "
                f"num_windows=({self.num_windows_h}, {self.num_windows_w}), "
                f"step_sizes=({self.step_size_h}, {self.step_size_w}))")


def create_sliding_window_config(height: int, width: int, loop_step: int = 8) -> SlidingWindowConfig:
    """
    Convenience function to create a sliding window configuration.
    
    Args:
        height: Input height in latent space
        width: Input width in latent space
        loop_step: Loop step parameter for calculating step sizes
        
    Returns:
        SlidingWindowConfig instance
    """
    return SlidingWindowConfig(height, width, loop_step)

