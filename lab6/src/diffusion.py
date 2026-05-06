from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm


def linear_beta_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 1e-4, 0.999)


def extract(values: torch.Tensor, timesteps: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    out = values.gather(0, timesteps)
    return out.reshape(timesteps.shape[0], *((1,) * (len(x_shape) - 1)))


class GaussianDiffusion(nn.Module):
    def __init__(self, timesteps: int = 1000, beta_schedule: str = "linear") -> None:
        super().__init__()
        if beta_schedule == "linear":
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError("beta_schedule must be 'linear' or 'cosine'")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.num_timesteps = timesteps
        self.beta_schedule = beta_schedule
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance.clamp(min=1e-20))

    def q_sample(self, x_start: torch.Tensor, timesteps: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
        return (
            extract(self.sqrt_alphas_cumprod, timesteps, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, timesteps, x_start.shape) * noise
        )

    def p_losses(
        self,
        model: nn.Module,
        x_start: torch.Tensor,
        timesteps: torch.Tensor,
        labels: torch.Tensor,
        cfg_drop_prob: float = 0.0,
        loss_type: str = "huber",
    ) -> torch.Tensor:
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, timesteps=timesteps, noise=noise)
        if cfg_drop_prob > 0:
            keep = (torch.rand(labels.shape[0], device=labels.device) > cfg_drop_prob).float()[:, None]
            labels = labels * keep
        predicted_noise = model(x_noisy, timesteps, labels)
        if loss_type == "mse":
            return F.mse_loss(predicted_noise, noise)
        if loss_type == "huber":
            return F.smooth_l1_loss(predicted_noise, noise)
        raise ValueError("loss_type must be 'huber' or 'mse'")

    def predict_noise(self, model: nn.Module, x: torch.Tensor, timesteps: torch.Tensor, labels: torch.Tensor, cfg_scale: float) -> torch.Tensor:
        if cfg_scale == 1.0:
            return model(x, timesteps, labels)
        uncond = torch.zeros_like(labels)
        pred_uncond = model(x, timesteps, uncond)
        pred_cond = model(x, timesteps, labels)
        return pred_uncond + cfg_scale * (pred_cond - pred_uncond)

    @torch.no_grad()
    def p_sample(self, model: nn.Module, x: torch.Tensor, timesteps: torch.Tensor, labels: torch.Tensor, step_index: int, cfg_scale: float) -> torch.Tensor:
        predicted_noise = self.predict_noise(model, x, timesteps, labels, cfg_scale)
        betas_t = extract(self.betas, timesteps, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, timesteps, x.shape)
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, timesteps, x.shape)
        model_mean = sqrt_recip_alphas_t * (x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)
        if step_index == 0:
            return model_mean
        posterior_variance_t = extract(self.posterior_variance, timesteps, x.shape)
        return model_mean + torch.sqrt(posterior_variance_t) * torch.randn_like(x)

    @torch.no_grad()
    def p_sample_loop(
        self,
        model: nn.Module,
        labels: torch.Tensor,
        image_size: int = 64,
        channels: int = 3,
        cfg_scale: float = 1.0,
        return_intermediates: bool = False,
        num_intermediates: int = 8,
    ):
        device = labels.device
        image = torch.randn((labels.shape[0], channels, image_size, image_size), device=device)
        intermediates: List[torch.Tensor] = []
        capture = set(torch.linspace(self.num_timesteps - 1, 0, steps=num_intermediates).long().tolist())
        for i in tqdm(reversed(range(self.num_timesteps)), total=self.num_timesteps, desc="DDPM sampling"):
            t = torch.full((labels.shape[0],), i, device=device, dtype=torch.long)
            image = self.p_sample(model, image, t, labels, i, cfg_scale)
            if return_intermediates and i in capture:
                intermediates.append(image.detach().cpu())
        if return_intermediates:
            return image, intermediates
        return image

    @torch.no_grad()
    def ddim_sample(
        self,
        model: nn.Module,
        labels: torch.Tensor,
        image_size: int = 64,
        channels: int = 3,
        sample_steps: int = 100,
        cfg_scale: float = 1.0,
        eta: float = 0.0,
        return_intermediates: bool = False,
        num_intermediates: int = 8,
    ):
        device = labels.device
        sample_steps = min(sample_steps, self.num_timesteps)
        image = torch.randn((labels.shape[0], channels, image_size, image_size), device=device)
        times = torch.linspace(-1, self.num_timesteps - 1, steps=sample_steps + 1, device=device).long()
        times = list(reversed(times.tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))
        capture_indices = set(torch.linspace(0, len(time_pairs) - 1, steps=num_intermediates).long().tolist())
        intermediates: List[torch.Tensor] = []

        for idx, (time, time_next) in enumerate(tqdm(time_pairs, desc="DDIM sampling")):
            t = torch.full((labels.shape[0],), time, device=device, dtype=torch.long)
            pred_noise = self.predict_noise(model, image, t, labels, cfg_scale)
            alpha = self.alphas_cumprod[time]
            alpha_next = torch.tensor(1.0, device=device) if time_next < 0 else self.alphas_cumprod[time_next]
            x_start = (image - torch.sqrt(1 - alpha) * pred_noise) / torch.sqrt(alpha)
            x_start = x_start.clamp(-1.0, 1.0)
            if time_next < 0:
                image = x_start
            else:
                sigma = eta * torch.sqrt((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha))
                c = torch.sqrt((1 - alpha_next - sigma**2).clamp(min=0.0))
                image = torch.sqrt(alpha_next) * x_start + c * pred_noise + sigma * torch.randn_like(image)
            if return_intermediates and idx in capture_indices:
                intermediates.append(image.detach().cpu())
        if return_intermediates:
            return image, intermediates
        return image
