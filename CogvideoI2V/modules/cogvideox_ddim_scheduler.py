from typing import Optional, Union
import torch
from diffusers import CogVideoXDDIMScheduler

class CustomCogVideoXDDIMScheduler(CogVideoXDDIMScheduler):
    """
    Custom DDIM scheduler that extends CogVideoXDDIMScheduler with additional re_noise functionality.
    Inherits all standard functionality from the parent class and only adds custom methods.
    """

    def re_noise(
        self,
        sample: torch.Tensor,
        from_timestep: Union[int, torch.IntTensor],
        to_timestep: Union[int, torch.IntTensor],
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Noises a sample from a less noisy timestep to a more noisy timestep. This is the reverse of a DDIM step.

        Args:
            sample (`torch.Tensor`): The sample at `from_timestep`.
            from_timestep (`Union[int, torch.IntTensor]`): The starting timestep (less noisy).
            to_timestep (`Union[int, torch.IntTensor]`): The target timestep (more noisy).
            generator (`torch.Generator`, *optional*): A random number generator.

        Returns:
            `torch.Tensor`: The noised sample at `to_timestep`.
        """
        # Convert to integers if needed
        if isinstance(from_timestep, torch.Tensor):
            from_timestep = from_timestep.item()
        if isinstance(to_timestep, torch.Tensor):
            to_timestep = to_timestep.item()
            
        if to_timestep <= from_timestep:
            raise ValueError("`to_timestep` must be greater than `from_timestep` to add noise.")

        self.alphas_cumprod = self.alphas_cumprod.to(device=sample.device)
        alphas_cumprod = self.alphas_cumprod.to(dtype=sample.dtype)

        alpha_prod_t_from = alphas_cumprod[from_timestep]
        alpha_prod_t_to = alphas_cumprod[to_timestep]

        # Formula from q-sampler (forward process) x_t = sqrt(alpha_t)x_0 + sqrt(1-alpha_t)epsilon
        # Derivation for x_t2 from x_t1: x_t2 = sqrt(alpha_bar_t2/alpha_bar_t1) * x_t1 + sqrt(1 - alpha_bar_t2/alpha_bar_t1) * epsilon
        c = torch.sqrt(alpha_prod_t_to / alpha_prod_t_from)
        s = torch.sqrt(1 - alpha_prod_t_to / alpha_prod_t_from)

        noise = torch.randn(sample.shape, generator=generator, device=sample.device, dtype=sample.dtype)

        # Match dimensions for broadcasting
        c = c.flatten()
        while len(c.shape) < len(sample.shape):
            c = c.unsqueeze(-1)
        s = s.flatten()
        while len(s.shape) < len(sample.shape):
            s = s.unsqueeze(-1)

        noised_sample = c * sample + s * noise

        return noised_sample
