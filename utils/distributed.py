from typing import TYPE_CHECKING, List, Tuple, Dict, Optional

import torch
import torch.distributed as dist
import time
import math
import datetime

from .tile_utils import TiledLatentTensor2D, SlidingWindowConfig, TileNoiseAggregator2D
from .tile_std_tracker import TileStdTracker

class DistributedManager:
    def __init__(
        self,
        comm_method: str = "allgather",
        enable_cache: bool = False,
    ):
        if not dist.is_initialized():
            raise NotImplementedError("Should be initialized as distributed!")
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.comm_method = comm_method
        print(f"Communication method: {self.comm_method}")

        self.enable_cache = enable_cache
        if self.enable_cache:
            assert self.comm_method == "allgather"

        self.latent_ring2d = None
        self.image_latent_ring2d = None

        self.avg_workload: int = None

        self.buffer_list = None
        self.starts = []
        self.ends = []
        self.shapes = []
        self.buffer_index_dict = {}
        self.buffer_numel = 0
        self.idx_latent: int = None
        self.skipped_idx_list = [[] for _ in range(self.world_size)]

        self.ring2d_dict: Dict[str, TiledLatentTensor2D] = {}

        self.setup_done = False
        self.do_classifier_free_guidance = False


    def setup_config(
        self, 
        latent, 
        image_latent, 
        window_config: SlidingWindowConfig,
        noise_fusion_method: str,
        std_tracker_update_interval: int,
        do_classifier_free_guidance: bool,
        tile_noise_fuser: TileNoiseAggregator2D = None
    ):
        self.do_classifier_free_guidance = do_classifier_free_guidance
        # 1. Latents
        # Latent must use seperate memory space from buffers for allgather,
        # because when shifting, we should slice and reorganize all tiles.
        # Using only buffers and perfom all operations in-place isn't enough.
        assert self.latent_ring2d is None and self.image_latent_ring2d is None
        self.latent_ring2d = TiledLatentTensor2D(latent_tensor=latent)
        self.image_latent_ring2d = TiledLatentTensor2D(latent_tensor=image_latent)
        self.ring2d_dict[self.latent_name] = self.latent_ring2d

        # 2. Window config
        self.window_config = window_config
        window_params = window_config.get_window_params()
        window_size = window_params['window_size']
        num_windows_h = window_params['num_windows_h']
        num_windows_w = window_params['num_windows_w']
        total_windows = window_params['total_windows']
        latent_step_size_h = window_params['latent_step_size_h']
        latent_step_size_w = window_params['latent_step_size_w']

        self.window_height = window_size[0]
        self.window_width = window_size[1]
        self.loop_step = self.window_config.loop_step
        self.num_total_windows_h = num_windows_h
        self.num_total_windows_w = num_windows_w
        self.num_total_windows = total_windows
        self.latent_step_size_h = latent_step_size_h
        self.latent_step_size_w = latent_step_size_w

        self.latent_window_shape = (*latent.shape[:3], self.window_height, self.window_width)

        # 3. Initialize config
        self._get_global_indices()
        self.current_shift_step_h = 0
        self.current_shift_step_w = 0

        self.allocate_buffer(tile_noise_fuser)

        # 4. Noise aggregator
        if tile_noise_fuser is not None:
            # ! Sometimes the shape of tile noise fuser is not latent.shape,
            # ! we should initialize it outside.
            self.tile_noise_fuser = tile_noise_fuser
            tile_noise_fuser.set_tile_shape(self.latent_window_shape)
        else:
            self.tile_noise_fuser = TileNoiseAggregator2D(
                noise_shape=tuple(latent.shape),
                device=latent.device,
                dtype=latent.dtype,  # Use prompt_embeds.dtype for consistency
                dist_manager=self,
                fusion_method=noise_fusion_method,
            )
            self.tile_noise_fuser.set_tile_shape(self.latent_window_shape)


        # 5. std tracker
        self.std_tracker = None
        if self.enable_cache:
            self.std_tracker = TileStdTracker(
                self.num_total_windows, 
                std_tracker_update_interval,
                dist_manager=self,
            )
        
        self.setup_done = True
    
    
    def register_cache_ring2d(
        self,
        tsfmr_prev_input_ring2d = None,
        tsfmr_prev_output_ring2d = None,
        tsfmr_prev_prev_input_ring2d = None,
        tsfmr_prev_prev_output_ring2d = None,
        tsfmr_cache_residual_ring2d = None,
        tsfmr_accumulated_error_ring2d = None,
    ):
        assert not self.setup_done and self.enable_cache
        self.tsfmr_prev_input_ring2d = tsfmr_prev_input_ring2d
        self.ring2d_dict["tsfmr_prev_input_ring2d"] = self.tsfmr_prev_input_ring2d
        self.tsfmr_prev_output_ring2d = tsfmr_prev_output_ring2d
        self.ring2d_dict["tsfmr_prev_output_ring2d"] = self.tsfmr_prev_output_ring2d
        self.tsfmr_prev_prev_input_ring2d = tsfmr_prev_prev_input_ring2d
        self.ring2d_dict["tsfmr_prev_prev_input_ring2d"] = self.tsfmr_prev_prev_input_ring2d
        self.tsfmr_prev_prev_output_ring2d = tsfmr_prev_prev_output_ring2d
        self.ring2d_dict["tsfmr_prev_prev_output_ring2d"] = self.tsfmr_prev_prev_output_ring2d
        self.tsfmr_cache_residual_ring2d = tsfmr_cache_residual_ring2d
        self.ring2d_dict["tsfmr_cache_residual_ring2d"] = self.tsfmr_cache_residual_ring2d
        self.tsfmr_accumulated_error_ring2d = tsfmr_accumulated_error_ring2d
        self.ring2d_dict["tsfmr_accumulated_error_ring2d"] = self.tsfmr_accumulated_error_ring2d
        

    def register_buffer(self, name, shape) -> int:
        assert len(self.starts) == len(self.ends)
        self.starts.append(self.buffer_numel)
        numel = math.prod(shape)
        self.buffer_numel += numel
        self.ends.append(self.buffer_numel)
        self.shapes.append(shape)
        # return index
        idx = len(self.ends) - 1
        self.buffer_index_dict[name] = idx
        return idx


    def get_latents(self):
        return self.latent_ring2d.torch_latent

    def get_image_latents(self):
        return self.image_latent_ring2d.torch_latent

    
    def set_latents(self, t):
        assert t.shape == self.latent_ring2d.torch_latent.shape and t.dtype == self.latent_ring2d.torch_latent.dtype
        self.latent_ring2d.torch_latent = t


    def get_local_indices(self) -> List[int]:
        return self.local_indices
    

    def get_tile(self, index):
        """Get a tile based on current shift offest."""
        start_h, end_h, start_w, end_w = self.get_tile_boundary_for_idx(index)
        latents_for_view = self.latent_ring2d.get_window_latent(top=start_h, bottom=end_h, left=start_w, right=end_w)
        image_latents_for_view = self.image_latent_ring2d.get_window_latent(top=start_h, bottom=end_h, left=start_w, right=end_w)
        # print(f"{self.rank=} {index=} {latents_for_view.shape} {image_latents_for_view.shape}")
        return latents_for_view, image_latents_for_view
    

    def update_tile(self, index, tile_latents):
        assert tile_latents.shape == self.latent_window_shape
        start_h, end_h, start_w, end_w = self.get_tile_boundary_for_idx(index)
        # print(f"{self.rank=} update tile {index=} {start_h=} {end_h=} {start_w=} {end_w=}")
        self.latent_ring2d.set_window_latent(tile_latents, top=start_h, bottom=end_h, left=start_w, right=end_w)


    def get_tensor_in_buffer(self, name: str, tile_idx: int):
        """Get a tensor in the buffer given game and tile idx.
        The target is only local buffer, not global.

        If `tile_idx == -1`, then get all local tiles of this tensor.
        """
        buffer = self.get_local_buffer(name)
        if tile_idx == -1:
            return buffer
        assert tile_idx in self.local_indices
        offset = self.convert_tile_idx_to_local_offset(tile_idx)
        return buffer[offset]

    def set_tensor_in_buffer(self, name: str, tile_idx: int, t: torch.Tensor):
        """Set a tensor in the buffer to data in provided tensor.
        The target is only local buffer, not global.

        If `tile_idx == -1`, then set all local tiles of this tensor to t.
        """
        buffer = self.get_local_buffer(name)
        if tile_idx == -1:
            buffer[:] = t
        else:
            assert tile_idx in self.local_indices
            offset = self.convert_tile_idx_to_local_offset(tile_idx)
            if isinstance(t, torch.Tensor):
                assert buffer[offset].shape == t.shape
                buf_dtype_size = buffer.dtype.itemsize
                t_dtype_size = t.dtype.itemsize
                assert buf_dtype_size >= t_dtype_size, f"{buf_dtype_size=} {t_dtype_size=}"
            buffer[offset] = t
    
    def get_local_buffer(self, name, rank = None):
        if rank is None:
            rank = self.rank
        tensor_idx = self.buffer_index_dict[name]
        start, end, shape = self._get_tensor_metadata(tensor_idx)
        return self.buffer_list[rank][start:end].view(shape)
    

    def get_global_buffer(self, name):
        tensor_idx = self.buffer_index_dict[name]
        start, end, shape = self._get_tensor_metadata(tensor_idx)
        return [t[start:end].view(shape) for t in self.buffer_list]


    def shift(self):
        # TODO(MX): can be changed to interleaved shifting
        self.current_shift_step_h = (self.current_shift_step_h + 1) % self.loop_step
        self.current_shift_step_w = (self.current_shift_step_w + 1) % self.loop_step
    

    def allocate_buffer(self, tile_noise_fuser: TileNoiseAggregator2D = None):
        if self.comm_method != "allgather":
            return
        # Allocate buffer for padded collective communication
        # We should allocate extra space for dummy tiles
        latent_shape = (self.avg_workload, *self.latent_window_shape)
        name = self.latent_name
        self.idx_latent = self.register_buffer(name=name, shape=latent_shape)
        # Cache related
        if self.enable_cache:
            # TODO(MX): This is fixed to CFG method, make it more flexible if necessary
            # Cache related:
            cfg_factor = 2 if self.do_classifier_free_guidance else 1
            shape = (self.avg_workload, self.latent_window_shape[0] * cfg_factor, *self.latent_window_shape[1:])
            for name in self.ring2d_dict:
                if not "tsfmr" in name:
                    continue
                self.register_buffer(name, shape)
            name = "skipped"
            idx = self.register_buffer(name=name, shape=(self.avg_workload,))
            # std tracker related
            name = "std_count"
            idx = self.register_buffer(name=name, shape=(self.avg_workload,))
            name = "std_mean"
            idx = self.register_buffer(name=name, shape=(self.avg_workload,))
            name = "std_m2"
            idx = self.register_buffer(name=name, shape=(self.avg_workload,))
            name = "std_values"
            idx = self.register_buffer(name=name, shape=(self.avg_workload,))

        # Noise Related
        name = self.fused_noise_name
        noise_shape = latent_shape
        if tile_noise_fuser is not None:
            acc_shape = tile_noise_fuser.noise_accumulator.shape
            noise_shape = (self.avg_workload, *acc_shape[:3], *self.latent_window_shape[-2:])
        self.register_buffer(name=name, shape=noise_shape)

        buffer_dtype = self.latent_ring2d.torch_latent.dtype
        self.buffer_list = [
            torch.zeros(size=(self.buffer_numel,), dtype=buffer_dtype, 
                        device=self.latent_ring2d.torch_latent.device)
            for _ in range(self.world_size)
        ]
            
    def _get_local_all_buffer(self):
        return self.buffer_list[self.rank]
    
    def _get_global_all_buffer_as_list(self):
        return self.buffer_list
    
    def _get_local_buf_by_idx_range(self, start_idx, end_idx):
        start = self.starts[start_idx]
        end = self.ends[end_idx]
        return self.buffer_list[self.rank][start:end]
    
    def _get_global_buf_by_idx_range(self, start_idx, end_idx):
        start = self.starts[start_idx]
        end = self.ends[end_idx]
        return [rt[start:end] for rt in self.buffer_list]
    
    def _copy_tiles_to_buffer(self, start_idx: int, end_idx: int):
        """Copy latest ring latent to buffer, preparing for allgather."""
        # Since tensors used by caching are always updated in-place, 
        # we only need to update tiles here.
        # TODO(MX): We may use tile buffer in-place too.
        for name, ring2d in self.ring2d_dict.items():
            tensor_idx = self.buffer_index_dict[name]
            if start_idx > tensor_idx or end_idx < tensor_idx:
                # This tensor is not involved in this communication
                continue
            local_tiles = self._get_local_tiles(ring2d)
            local_buffer = self.get_local_buffer(name)
            for i, t in enumerate(local_tiles):
                # We should handle: 0, 1 | 2 <> | 3 <>
                assert local_buffer[i].shape == t.shape
                local_buffer[i].copy_(t)

    def _copy_buffer_to_tiles(self, start_idx: int, end_idx: int):
        for name, ring2d in self.ring2d_dict.items():
            tensor_idx = self.buffer_index_dict[name]
            if start_idx > tensor_idx or end_idx < tensor_idx:
                # This tensor is not involved in this communication
                continue
            start, end, shape = self._get_tensor_metadata(tensor_idx)
            for cur_rank, tile_list in enumerate(self.global_indices):
                if cur_rank == self.rank:
                    # No need to copy local tiles
                    continue
                tile_buffer = self.buffer_list[cur_rank][start:end].view(shape)
                for i, tidx in enumerate(tile_list):
                    window_position = self.get_tile_boundary_for_idx(tidx)
                    ring2d.set_window_latent(tile_buffer[i], *window_position)
    
    def communicate(self, name_list: Optional[List[str]] = None):
        # Since devices may have uneven number of tiles, like 9 tiles for 8 gpuk
        # or 4 tiles for 8 gpus, we should properly handle bounday conditions.
        if self.world_size == 1:
            return
        # torch.cuda.synchronize()
        start_time = time.perf_counter()
        # 0. Get tensor metadata
        if name_list is None or isinstance(name_list, list) and len(name_list) == 0:
            start_idx = None
            end_idx = None
        else:
            if not isinstance(name_list, list):
                name_list = [name_list]
            index_list = []
            for name in name_list:
                tensor_idx = self.buffer_index_dict[name]
                index_list.append(tensor_idx)
            index_list.sort()
            start_idx = index_list[0]
            end_idx = index_list[-1]

        # 1. Check idx range
        start_idx = 0 if start_idx is None else start_idx
        end_idx = len(self.starts) - 1 if end_idx is None else end_idx
        assert end_idx >= start_idx and start_idx >= 0 and end_idx < len(self.starts)

        # 2. Communicate
        if self.comm_method == "allgather":
            self._communicate_allgather_padded(start_idx, end_idx)
        elif self.comm_method == "gather_scatter_padded":
            self._communicate_gather_scatter_padded()
        elif self.comm_method == "gather_scatter_obj_list":
            self._communicate_gather_scatter_obj_list()
        end_time = time.perf_counter()
        # torch.cuda.synchronize()
        print(f"rank{self.rank} Communicate time consumption {end_time - start_time:.4f} seconds")

    def _communicate_allgather_padded(self, start_idx, end_idx):
        print(f"{self.rank=} {start_idx=} {end_idx=}")
        # latents are involved
        # 1. Copy latents to buffer
        self._copy_tiles_to_buffer(start_idx, end_idx)

        # 2. Allgather
        global_buf_list = self._get_global_buf_by_idx_range(start_idx, end_idx)
        local_buf = self._get_local_buf_by_idx_range(start_idx, end_idx)
        dist.all_gather(global_buf_list, local_buf)

        # 3. Put back to ring latent
        self._copy_buffer_to_tiles(start_idx, end_idx)

    def _communicate_gather_scatter_padded(self):
        # 0. Allocate buffer if not yet
        # 1. root process gather tensors
        # 2. root process scatter tensors
        # 3. write back to local complete tensor
        raise NotImplementedError

    def _communicate_gather_scatter_obj_list(self):
        # This method has not been adapted. So we raise exception here.
        raise RuntimeError("gather scatter is not supported right now.")
        # 1. root process gather tensors
        local_tiles = self._get_local_tiles()
        gather_list = [None] * self.world_size if self.is_first_rank else None
        dist.gather_object(local_tiles, gather_list, dst=self.first_rank)
        if self.is_first_rank:
            # Write tiles to local complete latent
            i = 0
            for tiles in gather_list:
                for tile in tiles:
                    self.update_tile(i, tile)
                    i += 1
        # We must shift after gathering original tiles and before scattering
        self._shift()
        # 2. root process scatter tensors
        output_list = [None]
        if self.is_first_rank:
            all_tiles = self._get_all_tiles()
            dist.scatter_object_list(output_list, all_tiles, src=self.first_rank)
        else:
            dist.scatter_object_list(output_list, None, src=self.first_rank)
        # 3. write back to local complete tensor
        output_list = output_list[0]
        assert self.num_local_windows == len(output_list)
        for index, tile in zip(self.local_indices, output_list):
            self.update_tile(index, tile)
            
    def _get_tensor_metadata(self, tensor_idx) -> Tuple[int, int, Tuple]:
        start = self.starts[tensor_idx]
        end = self.ends[tensor_idx]
        shape = self.shapes[tensor_idx]
        return start, end, shape

    
    def _get_local_tiles(self, ring2d: TiledLatentTensor2D):
        tiles = []
        for i in self.local_indices:
            window_position = self.get_tile_boundary_for_idx(i)
            t = ring2d.get_window_latent(*window_position)
            tiles.append(t)
        return tiles
    
    
    def _get_all_tiles(self):
        all_tiles = []
        for tile_idx_list in self.global_indices:
            t = []
            for i in tile_idx_list:
                l, il = self.get_tile(i)
                t.append(l)
            all_tiles.append(t)
        return all_tiles
    
    
    def _distribute_workload(self, active_idx_list, skipped_idx_list) -> int:
        """Distribute workload across all ranks.

        Args:
            active_idx_list : Active tile indices.
            skipped_idx_list : Inactive tile indices.

        Returns:
            int: The average number of active workload.
        """
        total_active_workload = len(active_idx_list)
        avg_active_workload = (total_active_workload + self.world_size - 1) // self.world_size
        # [0:dummy_tile_cnt] rank will have only <avg_workload-1> tiles
        # [dummy_tile_cnt:] rank will have <avg_workload> tiles
        dummy_tile_cnt = avg_active_workload * self.world_size - total_active_workload

        # 1. distribute active tiles
        active_indices: List[List[int]] = []
        i = 0
        for rank in range(self.world_size):
            start = i
            if rank < dummy_tile_cnt:
                end = start + (avg_active_workload - 1)
            else:
                end = start + avg_active_workload
            active_indices.append(active_idx_list[start:end])
            i = end

        # 2. distribute skipped tiles
        avg_global_workload = (self.num_total_windows + self.world_size - 1) // self.world_size
        dummy_global_tile_cnt = avg_global_workload * self.world_size - self.num_total_windows
        skipped_indices = []
        i = 0
        for rank in range(self.world_size):
            active_tile_cnt = len(active_indices[rank])
            start = i
            if rank < dummy_global_tile_cnt:
                end = start + (avg_global_workload - 1 - active_tile_cnt)
            else:
                end = start + (avg_global_workload - active_tile_cnt)
            skipped_indices.append(skipped_idx_list[start:end])
            i = end

        return active_indices, skipped_indices, avg_active_workload

    
    def _get_global_indices(self):
        indices, _, avg_workload = self._distribute_workload(
            active_idx_list=list(range(self.num_total_windows)), skipped_idx_list=[])
        self.avg_workload = avg_workload

        self.global_indices = indices
        self.local_indices: List = self.global_indices[self.rank]
        self.num_local_windows = len(self.local_indices)
        if self.num_local_windows == 0:
            print(f"WARNING, the total number of tiles is less than available devices. rank {self.rank} is idle.")

        print(f"{self.rank=} {indices=}")


    def _redistribute_buffer_content(self, old_global_indices, new_global_indices):
        """Copy data of new indices into LOCAL buffer."""
        # TODO(MX): Prefer to hold local tiles if possible.
        # The current version will redistribute all tiles regardless of local ones.
        old_mapping = {}
        for rank, idx_list in enumerate(old_global_indices):
            for off, idx in enumerate(idx_list):
                old_mapping[idx] = (rank, off)

        local_new_indices = new_global_indices[self.rank]
        for name in self.buffer_index_dict:
            if name in self.ring2d_dict:
                continue
            local_buffer = self.get_local_buffer(name)
            # we should backup the buffer to avoid overwritting data
            local_buffer_bkp = local_buffer.clone()
            for i, new_idx in enumerate(local_new_indices):
                rank, off = old_mapping[new_idx]
                if rank == self.rank:
                    local_buffer[i] = local_buffer_bkp[off]
                else:
                    new_t = self.get_local_buffer(name, rank)
                    local_buffer[i] = new_t[off]

    
    def redistribute_workload(self):
        if self.world_size == 1:
            return
        buffer_list = self.get_global_buffer("skipped")
        buffer_list = torch.cat(buffer_list, dim=0).tolist()
        assert len(buffer_list) == self.world_size * self.avg_workload
        # print(f"{self.rank=} {buffer_list=}")

        skipped_tile_idx_list = []
        active_tile_idx_list = []
        for rank, idx_list in enumerate(self.global_indices):
            idx_base = rank * self.avg_workload
            for k, tile_idx in enumerate(idx_list):
                skipped = buffer_list[idx_base + k] == 1.0
                if skipped:
                    skipped_tile_idx_list.append(tile_idx)
                else:
                    active_tile_idx_list.append(tile_idx)
        
        skipped_tile_idx_list.sort()
        active_tile_idx_list.sort()

        active_indices, skipped_indices, _ = self._distribute_workload(
            active_idx_list=active_tile_idx_list, skipped_idx_list=skipped_tile_idx_list)
        if len(active_indices[self.rank]) == 0:
            print(f"WARNING, rank {self.rank} is idle after work redistribution.")
        global_indices = [a + s for a,s in zip(active_indices, skipped_indices)]

        # TODO(MX): We only changed the tile index, but the buffer content doesn't
        # get redistributed yet! We should move them too!

        old_global_indices = self.global_indices
        self._redistribute_buffer_content(old_global_indices, global_indices)

        self.skipped_idx_list = skipped_indices
        self.global_indices = global_indices
        self.local_indices: List = self.global_indices[self.rank]
        self.num_local_windows = len(self.local_indices)
        if self.num_local_windows == 0:
            print(f"WARNING, the total number of tiles is less than available devices. rank {self.rank} is idle.")
        print(f"{self.rank=} After redistribution: {active_indices=} {global_indices=}")
    

    def get_tile_boundary_for_idx(self, index) -> Tuple[int, int, int, int]:
        """Tiles are numbered in row-major fashion.
        This function returns the boundary of the tile of this index based on the
        current shifting position.

        The start_h, end_h, start_w, end_w may be greater than the boundary limit!

        Args:
            index : tile index
        """
        row = index // self.num_total_windows_w
        col = index % self.num_total_windows_w

        shift_height_offset = self.current_shift_step_h * self.latent_step_size_h
        shift_width_offset = self.current_shift_step_w * self.latent_step_size_w
        start_h = row * self.window_height + shift_height_offset
        end_h = start_h + self.window_height
        start_w = col * self.window_width + shift_width_offset
        end_w = start_w + self.window_width

        return (start_h, end_h, start_w, end_w)

    
    def convert_tile_idx_to_local_offset(self, tile_idx: int) -> int:
        return self.local_indices.index(tile_idx)
    
    ######### Cache related functions #########
    
    def tile_is_skipped(self, tile_idx: int) -> bool:
        assert tile_idx in self.local_indices
        return tile_idx in self.skipped_idx_list[self.rank]
    
    def mark_tile_skipped(self, tile_idx):
        assert tile_idx in self.local_indices
        if tile_idx not in self.skipped_idx_list[self.rank]:
            self.skipped_idx_list[self.rank].append(tile_idx)
    
    
    ######### Std-tracker related functions #########
    
    def std_tracker_update(self, tile_idx, noise_pred, step):
        self.std_tracker.update(tile_idx, noise_pred)
    
    def std_tracker_get_new_threshold(self, base_thresh: float, alpha: float, mode: str):
        return self.std_tracker.compute_cache_thresholds(base_thresh, alpha, mode)

    def std_tracker_should_update(self, i: int) -> bool:
        return self.std_tracker.should_update_cache_thresholds(i)
    
    def std_tracker_update_last_step(self, i: int):
        return self.std_tracker.update_last_step(i)

    ######### Noise tracker related functions #########

    def allgather_fused_noise(self):
        if self.world_size == 1:
            return self.tile_noise_fuser.noise_accumulator
        start_time = time.perf_counter()
        name = self.fused_noise_name
        # 1. Copy noise to buffer tile by tile
        for tile_idx in self.local_indices:
            start_h, end_h, start_w, end_w = self.get_tile_boundary_for_idx(tile_idx)
            t = self.tile_noise_fuser.get_tile_fused_noise(top=start_h, bottom=end_h, left=start_w, right=end_w)
            self.set_tensor_in_buffer(name, tile_idx, t)

        # 2. Allgather
        local_buf = self.get_local_buffer(name)
        global_buf = self.get_global_buffer(name)
        dist.all_gather(global_buf, local_buf)

        # 3. Copy all noise tiles to the fuser to get a complete noise
        tensor_idx = self.buffer_index_dict[name]
        start, end, shape = self._get_tensor_metadata(tensor_idx)
        for cur_rank, tile_list in enumerate(self.global_indices):
            tile_buffer = self.buffer_list[cur_rank][start:end].view(shape)
            for i, tidx in enumerate(tile_list):
                start_h, end_h, start_w, end_w = self.get_tile_boundary_for_idx(tidx)
                self.tile_noise_fuser.set_tile_noise(tile_buffer[i], top=start_h, bottom=end_h, left=start_w, right=end_w)
        # torch.cuda.synchronize()
        end_time = time.perf_counter()
        cost = end_time - start_time
        # print(f"{self.rank=} Allgather noise time consumption {cost:.4f} seconds")
        return self.tile_noise_fuser.noise_accumulator
    
    def tile_noise_fuser_add(self, tile_idx, noise_pred, tile_weight):
        start_h, end_h, start_w, end_w = self.get_tile_boundary_for_idx(tile_idx)
        self.tile_noise_fuser.add_tile_noise(
            noise_pred,
            top=start_h,
            bottom=end_h,
            left=start_w,
            right=end_w,
            weight=tile_weight
        )
        
    def clear(self):
        """Clears states that should be reset at the beginning of each round."""
        # 1. Skipped idex
        if self.enable_cache:
            self.skipped_idx_list = [[] for _ in range(self.world_size)]
            self.set_tensor_in_buffer("skipped", -1, 0.0)
        # 2. Reset tile_noise_fuser
        self.tile_noise_fuser.reset()
    
    
    @property
    def is_first_rank(self):
        return self.rank == self.first_rank
    
    @property
    def first_rank(self):
        return 0
    
    @property
    def latent_name(self):
        return "latent"
    
    @property
    def fused_noise_name(self):
        return "fused_noise"