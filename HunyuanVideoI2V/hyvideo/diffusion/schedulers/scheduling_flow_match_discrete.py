# Copyright 2024 Stability AI, Katherine Crowson and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# Modified from diffusers==0.29.2
#
# ==============================================================================

from dataclasses import dataclass
from typing import Optional, Tuple, Union, List

import numpy as np
import torch

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput, logging
from diffusers.schedulers.scheduling_utils import SchedulerMixin


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class FlowMatchDiscreteSchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's `step` function output.

    Args:
        prev_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample `(x_{t-1})` of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
    """

    prev_sample: torch.FloatTensor


class FlowMatchDiscreteScheduler(SchedulerMixin, ConfigMixin):
    """
    Euler scheduler.

    This model inherits from [`SchedulerMixin`] and [`ConfigMixin`]. Check the superclass documentation for the generic
    methods the library implements for all schedulers such as loading and saving.

    Args:
        num_train_timesteps (`int`, defaults to 1000):
            The number of diffusion steps to train the model.
        timestep_spacing (`str`, defaults to `"linspace"`):
            The way the timesteps should be scaled. Refer to Table 2 of the [Common Diffusion Noise Schedules and
            Sample Steps are Flawed](https://huggingface.co/papers/2305.08891) for more information.
        shift (`float`, defaults to 1.0):
            The shift value for the timestep schedule.
        reverse (`bool`, defaults to `True`):
            Whether to reverse the timestep schedule.
    """

    _compatibles = []
    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 1.0,
        reverse: bool = True,
        solver: str = "euler",
        n_tokens: Optional[int] = None,
    ):
        sigmas = torch.linspace(1, 0, num_train_timesteps + 1)

        if not reverse:
            sigmas = sigmas.flip(0)

        self.sigmas = sigmas
        # the value fed to model
        self.timesteps = (sigmas[:-1] * num_train_timesteps).to(dtype=torch.float32)

        self._step_index = None
        self._begin_index = None
        
        # Store sigma arrays and step indices for each tile
        self._tile_sigmas = {}  # Dictionary to store sigmas for each tile
        self._tile_step_indices = {}  # Dictionary to store step indices for each tile
        self._current_tile_id = None

        self.supported_solver = ["euler"]
        if solver not in self.supported_solver:
            raise ValueError(
                f"Solver {solver} not supported. Supported solvers: {self.supported_solver}"
            )

    @property
    def step_index(self):
        """
        The index counter for current timestep. It will increase 1 after each scheduler step.
        """
        if self._current_tile_id is not None:
            return self._tile_step_indices.get(self._current_tile_id, self._step_index)
        return self._step_index

    @property
    def begin_index(self):
        """
        The index for the first timestep. It should be set from pipeline with `set_begin_index` method.
        """
        return self._begin_index

    # Copied from diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler.set_begin_index
    def set_begin_index(self, begin_index: int = 0):
        """
        Sets the begin index for the scheduler. This function should be run from pipeline before the inference.

        Args:
            begin_index (`int`):
                The begin index for the scheduler.
        """
        self._begin_index = begin_index

    def set_current_tile_id(self, tile_id: str):
        """
        Set the current tile ID and create sigma array and step index for this tile if it doesn't exist.
        
        Args:
            tile_id (`str`): Unique identifier for the current tile
        """
        self._current_tile_id = tile_id
        if tile_id not in self._tile_sigmas:
            self._tile_sigmas[tile_id] = self.sigmas.clone()
        if tile_id not in self._tile_step_indices:
            # Initialize step index for this tile
            if self._step_index is not None:
                self._tile_step_indices[tile_id] = self._step_index
            else:
                self._tile_step_indices[tile_id] = 0

    def get_tile_sigmas(self, tile_id: str = None):
        """
        Get the sigma array for a specific tile.
        
        Args:
            tile_id (`str`): Tile ID, defaults to current tile
            
        Returns:
            `torch.Tensor`: Sigma array for the tile
        """
        if tile_id is None:
            tile_id = self._current_tile_id
        return self._tile_sigmas.get(tile_id, self.sigmas)

    def get_tile_step_index(self, tile_id: str = None):
        """
        Get the step index for a specific tile.
        
        Args:
            tile_id (`str`): Tile ID, defaults to current tile
            
        Returns:
            `int`: Step index for the tile
        """
        if tile_id is None:
            tile_id = self._current_tile_id
        return self._tile_step_indices.get(tile_id, self._step_index)

    def reset_tile_tracking(self):
        """
        Reset all tile sigma arrays and step indices. Call this when starting a new timestep.
        """
        logger.info(f"self._tile_sigmas: {self._tile_sigmas}")
        logger.info(f"self._tile_step_indices: {self._tile_step_indices}")
        self._current_tile_id = None

    def _sigma_to_t(self, sigma):
        return sigma * self.config.num_train_timesteps

    def set_timesteps(
        self,
        num_inference_steps: int = None,
        device: Union[str, torch.device] = None,
        n_tokens: int = None,
        timesteps: Optional[torch.Tensor] = None,
        num_tiles: int = 1,  # Add num_tiles parameter
    ):
        """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).

        Args:
            num_inference_steps (`int`, *optional*):
                The number of diffusion steps used when generating samples with a pre-trained model.
            device (`str` or `torch.device`, *optional*):
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
            n_tokens (`int`, *optional*):
                Number of tokens in the input sequence.
            timesteps (`torch.Tensor`, *optional*):
                Custom timesteps to use. If provided, `num_inference_steps` is ignored.
            num_tiles (`int`, *optional*):
                Number of tiles being processed per timestep. Defaults to 1.
        """
        self.num_inference_steps = num_inference_steps

        if timesteps is not None:
            self.timesteps = timesteps.to(dtype=torch.float32, device=device)
            sigmas = timesteps.float() / self.config.num_train_timesteps
            if not self.config.reverse:
                sigmas = 1 - sigmas
            if num_inference_steps is not None and len(sigmas) == num_inference_steps:
                final_sigma = torch.tensor([0.0], device=sigmas.device, dtype=sigmas.dtype)
                sigmas = torch.cat([sigmas, final_sigma])
            # logger.info(f"timesteps in set_timesteps if timesteps is not None: {timesteps}")
            # logger.info(f"sigmas in set_timesteps if timesteps is not None: {sigmas}")

            # if num_inference_steps is None:
            #     raise ValueError("Either `timesteps` or `num_inference_steps` must be provided")
            # sigmas = torch.linspace(1, 0, num_inference_steps + 1)
            # sigmas = self.sd3_time_shift(sigmas)
            # if not self.config.reverse:
            #     sigmas = 1 - sigmas

            # self.timesteps = (sigmas[:-1] * self.config.num_train_timesteps).to(
            #     dtype=torch.float32, device=device
            # )

        else:
            # Use num_inference_steps to calculate sigmas
            if num_inference_steps is None:
                raise ValueError("Either `timesteps` or `num_inference_steps` must be provided")
            sigmas = torch.linspace(1, 0, num_inference_steps + 1)
            sigmas = self.sd3_time_shift(sigmas)

            if not self.config.reverse:
                sigmas = 1 - sigmas

            self.timesteps = (sigmas[:-1] * self.config.num_train_timesteps).to(
                dtype=torch.float32, device=device
            )
            # logger.info(f"timesteps in set_timesteps if timesteps is None: {self.timesteps}")
            # logger.info(f"sigmas in set_timesteps if timesteps is None: {sigmas}")
        self.sigmas = sigmas
        # Reset step index
        self._step_index = None

    def index_for_timestep(self, timestep, schedule_timesteps=None):
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps

        indices = (schedule_timesteps == timestep).nonzero()

        # The sigma index that is taken for the **very** first `step`
        # is always the second index (or the last index if there is only 1)
        # This way we can ensure we don't accidentally skip a sigma in
        # case we start in the middle of the denoising schedule (e.g. for image-to-image)
        pos = 1 if len(indices) > 1 else 0

        return indices[pos].item()

    def _init_step_index(self, timestep):
        if self.begin_index is None:
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self.timesteps.device)
            step_idx = self.index_for_timestep(timestep)
            if self._current_tile_id is not None:
                self._tile_step_indices[self._current_tile_id] = step_idx
            else:
                self._step_index = step_idx
        else:
            if self._current_tile_id is not None:
                self._tile_step_indices[self._current_tile_id] = self._begin_index
            else:
                self._step_index = self._begin_index

    def scale_model_input(
        self, sample: torch.Tensor, timestep: Optional[int] = None
    ) -> torch.Tensor:
        return sample

    def sd3_time_shift(self, t: torch.Tensor):
        return (self.config.shift * t) / (1 + (self.config.shift - 1) * t)
        
    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        sample: torch.FloatTensor,
        return_dict: bool = True,
    ) -> Union[FlowMatchDiscreteSchedulerOutput, Tuple]:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                A current instance of a sample created by the diffusion process.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            n_tokens (`int`, *optional*):
                Number of tokens in the input sequence.
            return_dict (`bool`):
                Whether or not to return a [`~schedulers.scheduling_euler_discrete.EulerDiscreteSchedulerOutput`] or
                tuple.

        Returns:
            [`~schedulers.scheduling_euler_discrete.EulerDiscreteSchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_euler_discrete.EulerDiscreteSchedulerOutput`] is
                returned, otherwise a tuple is returned where the first element is the sample tensor.
        """

        if (
            isinstance(timestep, int)
            or isinstance(timestep, torch.IntTensor)
            or isinstance(timestep, torch.LongTensor)
        ):
            raise ValueError(
                (
                    "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to"
                    " `EulerDiscreteScheduler.step()` is not supported. Make sure to pass"
                    " one of the `scheduler.timesteps` as a timestep."
                ),
            )

        if self.step_index is None:
            self._init_step_index(timestep)

        # Upcast to avoid precision issues when computing prev_sample
        sample = sample.to(torch.float32)

        dt = self.sigmas[self.step_index + 1] - self.sigmas[self.step_index]

        # logger.info(f"self._current_tile_id: {self._current_tile_id}, current_sigmas: {self.sigmas}, current_step_index: {self.step_index}, dt: {dt}")

        if self.config.solver == "euler":
            # logger.info(f"euler: ")
            prev_sample = sample + model_output.to(torch.float32) * dt
        else:
            raise ValueError(
                f"Solver {self.config.solver} not supported. Supported solvers: {self.supported_solver}"
            )

        # upon completion increase step index by one
        self._step_index += 1

        if not return_dict:
            return (prev_sample,)

        return FlowMatchDiscreteSchedulerOutput(prev_sample=prev_sample)
    # def step(
    #     self,
    #     model_output: torch.FloatTensor,
    #     timestep: Union[float, torch.FloatTensor],
    #     sample: torch.FloatTensor,
    #     return_dict: bool = True,
    # ) -> Union[FlowMatchDiscreteSchedulerOutput, Tuple]:
    #     """
    #     Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
    #     process from the learned model outputs (most often the predicted noise).

    #     Args:
    #         model_output (`torch.FloatTensor`):
    #             The direct output from learned diffusion model.
    #         timestep (`float`):
    #             The current discrete timestep in the diffusion chain.
    #         sample (`torch.FloatTensor`):
    #             A current instance of a sample created by the diffusion process.
    #         generator (`torch.Generator`, *optional*):
    #             A random number generator.
    #         n_tokens (`int`, *optional*):
    #             Number of tokens in the input sequence.
    #         return_dict (`bool`):
    #             Whether or not to return a [`~schedulers.scheduling_euler_discrete.EulerDiscreteSchedulerOutput`] or
    #             tuple.

    #     Returns:
    #         [`~schedulers.scheduling_euler_discrete.EulerDiscreteSchedulerOutput`] or `tuple`:
    #             If return_dict is `True`, [`~schedulers.scheduling_euler_discrete.EulerDiscreteSchedulerOutput`] is
    #             returned, otherwise a tuple is returned where the first element is the sample tensor.
    #     """

    #     if (
    #         isinstance(timestep, int)
    #         or isinstance(timestep, torch.IntTensor)
    #         or isinstance(timestep, torch.LongTensor)
    #     ):
    #         raise ValueError(
    #             (
    #                 "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to"
    #                 " `EulerDiscreteScheduler.step()` is not supported. Make sure to pass"
    #                 " one of the `scheduler.timesteps` as a timestep."
    #             ),
    #         )

    #     if self.step_index is None:
    #         self._init_step_index(timestep)

    #     # Get the appropriate sigma array for current tile
    #     if self._current_tile_id is not None:
    #         current_sigmas = self.get_tile_sigmas(self._current_tile_id)
    #         current_step_index = self.get_tile_step_index(self._current_tile_id)
    #     else:
    #         current_sigmas = self.sigmas
    #         current_step_index = self._step_index
            
    #     sample = sample.to(torch.float32)
            
    #     dt = current_sigmas[current_step_index + 1] - current_sigmas[current_step_index]

    #     logger.info(f"self._current_tile_id: {self._current_tile_id}, current_sigmas: {current_sigmas}, current_step_index: {current_step_index}, dt: {dt}")


    #     if self.config.solver == "euler":
    #         prev_sample = sample + model_output.to(torch.float32) * dt
    #     else:
    #         raise ValueError(
    #             f"Solver {self.config.solver} not supported. Supported solvers: {self.supported_solver}"
    #         )

    #     # upon completion increase step index by one for the current tile
    #     if self._current_tile_id is not None:
    #         self._tile_step_indices[self._current_tile_id] = current_step_index + 1
    #     else:
    #         self._step_index = current_step_index + 1

    #     if not return_dict:
    #         return (prev_sample,)

    #     return FlowMatchDiscreteSchedulerOutput(prev_sample=prev_sample)

    def __len__(self):
        return self.config.num_train_timesteps


    def scale_noise(
        self,
        sample: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        noise: Optional[torch.FloatTensor] = None,
        to_timestep: Optional[float] = None,
    ) -> torch.FloatTensor:
        """
        Forward process in flow-matching

        Args:
            sample (`torch.FloatTensor`):
                The input sample.
            timestep (`int`, *optional*):
                The current timestep in the diffusion chain.

        Returns:
            `torch.FloatTensor`:
                A scaled input sample.
        """
        # Make sure sigmas and timesteps have the same device and dtype as original_samples
        sigmas = self.sigmas.to(device=sample.device, dtype=sample.dtype)

        if sample.device.type == "mps" and torch.is_floating_point(timestep):
            # mps does not support float64
            schedule_timesteps = self.timesteps.to(sample.device, dtype=torch.float32)
            timestep = timestep.to(sample.device, dtype=torch.float32)
        else:
            schedule_timesteps = self.timesteps.to(sample.device)
            timestep = timestep.to(sample.device)

        # self.begin_index is None when scheduler is used for training, or pipeline does not implement set_begin_index
        if self.begin_index is None:
            step_indices = [self.index_for_timestep(t, schedule_timesteps) for t in timestep]
        elif self.step_index is not None:
            # add_noise is called after first denoising step (for inpainting)
            step_indices = [self.step_index] * timestep.shape[0]
        else:
            # add noise is called before first denoising step to create initial latent(img2img)
            step_indices = [self.begin_index] * timestep.shape[0]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < len(sample.shape):
            sigma = sigma.unsqueeze(-1)
        sigma = timestep
        sample = sigma[to_timestep] * noise + (1.0 - sigma[to_timestep]) * sample

        return sample

    def re_noise(
        self,
        sample: torch.FloatTensor,
        to_timestep_idx: int,
        noise: Optional[torch.FloatTensor] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    ) -> torch.FloatTensor:
        """
        Add noise to a denoised sample to move it back to a specific timestep in the denoising schedule.
        
        This function is used for re-noising in multi-stage generation pipelines where you want to
        partially denoise a sample and then continue denoising from an intermediate step.
        
        Args:
            sample (`torch.FloatTensor`):
                The denoised or partially denoised sample to add noise to.
            to_timestep_idx (`int`):
                The target timestep index to re-noise to. For example, if the scheduler has 50 steps
                and you want to re-noise back to step 45, pass to_timestep_idx=45.
            noise (`torch.FloatTensor`, *optional*):
                Pre-generated noise to add. If None, will generate random noise.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                Random number generator for reproducible noise generation. Can be a single generator
                or a list of generators (in which case the first one is used).
                
        Returns:
            `torch.FloatTensor`:
                The re-noised sample at the target timestep.
                
        Example:
            ```python
            # Denoise for 50 steps to get a clean sample
            clean_sample = denoise_fully(noisy_sample, num_steps=50)
            
            # Re-noise back to timestep 45 (5 steps before the end)
            renoised_sample = scheduler.re_noise(clean_sample, to_timestep_idx=45)
            
            # Continue denoising from step 45
            refined_sample = denoise_from_step(renoised_sample, start_step=45, num_steps=50)
            ```
        """
        # Handle generator - it can be a single generator or a list
        if generator is not None and isinstance(generator, list):
            # Use the first generator from the list
            noise_generator = generator[0]
        else:
            noise_generator = generator
        
        # Ensure generator is on the same device as the tensors
        if noise_generator is not None:
            if hasattr(noise_generator, 'device'):
                # If generator has a device attribute, ensure it matches the tensor device
                if noise_generator.device != sample.device:
                    noise_generator = torch.Generator(device=sample.device).manual_seed(noise_generator.initial_seed())
            else:
                # If generator doesn't have device attribute, create one on the correct device
                noise_generator = torch.Generator(device=sample.device).manual_seed(noise_generator.initial_seed())
        
        # Generate noise if not provided
        if noise is None:
            noise = torch.randn(
                sample.shape,
                device=sample.device,
                dtype=sample.dtype,
                generator=noise_generator
            )
        
        # Get the sigma value for the target timestep
        # Note: In flow matching, sigma represents the interpolation coefficient
        # between noise (sigma=1) and clean sample (sigma=0)
        sigmas = self.sigmas.to(device=sample.device, dtype=sample.dtype)
        
        # Ensure to_timestep_idx is within valid range
        if to_timestep_idx < 0 or to_timestep_idx >= len(self.timesteps):
            raise ValueError(
                f"to_timestep_idx must be between 0 and {len(self.timesteps)-1}, got {to_timestep_idx}"
            )
        
        # Get sigma for the target timestep
        sigma = sigmas[to_timestep_idx]
        logger.info(f"sigma in re_noise: {sigma}")
        # Expand sigma to match sample dimensions
        while len(sigma.shape) < len(sample.shape):
            sigma = sigma.unsqueeze(-1)
        
        # Re-noise the sample using flow matching formula:
        # x_t = sigma_t * noise + (1 - sigma_t) * x_0
        # where x_0 is the clean sample and x_t is the noisy sample at timestep t
        renoised_sample = sigma * noise + (1.0 - sigma) * sample
        
        return renoised_sample