from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.generators.common import (
    ConditionalLatentBackbone3D,
    PosteriorBackboneConfig,
    repeat_batch,
)


@dataclass(frozen=True)
class DiffusionConfig:
    timesteps: int = 1000
    sampling_steps: int = 50
    ddim_eta: float = 0.0
    consistency_weight: float = 0.0


def cosine_alpha_bar(timesteps: int, offset: float = 0.008) -> torch.Tensor:
    steps = torch.linspace(0, timesteps, timesteps + 1, dtype=torch.float64)
    values = torch.cos(((steps / timesteps + offset) / (1 + offset)) * math.pi / 2) ** 2
    values = values / values[0]
    return values[1:].clamp(1e-8, 1.0).float()


class ConditionalLatentDiffusion(nn.Module):
    """Conditional latent diffusion reference with v-prediction and DDIM.

    This is the robust multi-step reference branch. It models a full-evidence
    latent target conditioned on the deterministic observed-evidence state.
    """

    method = "diffusion"

    def __init__(
        self,
        backbone_config: PosteriorBackboneConfig,
        config: DiffusionConfig = DiffusionConfig(),
    ) -> None:
        super().__init__()
        if config.timesteps < 2:
            raise ValueError("timesteps must be at least 2")
        self.backbone_config = backbone_config
        self.config = config
        self.model = ConditionalLatentBackbone3D(backbone_config)
        alpha_bar = cosine_alpha_bar(config.timesteps)
        self.register_buffer("alpha_bar", alpha_bar, persistent=True)

    def training_loss(
        self,
        target_latent: torch.Tensor,
        condition: torch.Tensor,
        modality_mask: Optional[torch.Tensor] = None,
        target_ids: Optional[torch.Tensor] = None,
        quality: Optional[torch.Tensor] = None,
        consistency_target: Optional[torch.Tensor] = None,
        consistency_mask: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> dict[str, torch.Tensor]:
        batch_size = target_latent.shape[0]
        timesteps = torch.randint(
            0,
            self.config.timesteps,
            (batch_size,),
            device=target_latent.device,
            generator=generator,
        )
        noise = torch.randn(
            target_latent.shape,
            device=target_latent.device,
            dtype=target_latent.dtype,
            generator=generator,
        )
        alpha = self.alpha_bar[timesteps].sqrt().to(target_latent.dtype)
        sigma = (1 - self.alpha_bar[timesteps]).sqrt().to(target_latent.dtype)
        alpha_view = alpha.view(batch_size, 1, 1, 1, 1)
        sigma_view = sigma.view(batch_size, 1, 1, 1, 1)
        noisy = alpha_view * target_latent + sigma_view * noise
        velocity_target = alpha_view * noise - sigma_view * target_latent
        velocity = self.model(
            noisy,
            condition,
            timesteps.float() / (self.config.timesteps - 1),
            modality_mask=modality_mask,
            target_ids=target_ids,
            quality=quality,
        )
        native_loss = F.mse_loss(velocity, velocity_target)
        predicted_x0 = alpha_view * noisy - sigma_view * velocity
        consistency_loss = target_latent.new_zeros(())
        if consistency_target is not None:
            difference = (predicted_x0 - consistency_target).pow(2)
            if consistency_mask is not None:
                mask = consistency_mask.to(difference.dtype).expand_as(difference)
                consistency_loss = (difference * mask).sum() / mask.sum().clamp_min(1)
            else:
                consistency_loss = difference.mean()
        loss = native_loss + self.config.consistency_weight * consistency_loss
        return {
            "loss": loss,
            "native_loss": native_loss.detach(),
            "consistency_loss": consistency_loss.detach(),
            "predicted_x0": predicted_x0,
            "timesteps": timesteps,
        }

    def forward(self, *args, **kwargs) -> dict[str, torch.Tensor]:
        return self.training_loss(*args, **kwargs)

    @torch.no_grad()
    def sample(
        self,
        condition: torch.Tensor,
        n_samples: int = 1,
        seed: int = 0,
        modality_mask: Optional[torch.Tensor] = None,
        target_ids: Optional[torch.Tensor] = None,
        quality: Optional[torch.Tensor] = None,
        sampling_steps: Optional[int] = None,
    ) -> torch.Tensor:
        if n_samples < 1:
            raise ValueError("n_samples must be positive")
        steps = sampling_steps or self.config.sampling_steps
        if not 1 <= steps <= self.config.timesteps:
            raise ValueError("sampling_steps must be in [1, timesteps]")
        batch_size, _, depth, height, width = condition.shape
        expanded_condition = repeat_batch(condition, n_samples)
        expanded_mask = repeat_batch(modality_mask, n_samples)
        expanded_targets = repeat_batch(target_ids, n_samples)
        expanded_quality = repeat_batch(quality, n_samples)
        rng = torch.Generator(device=condition.device)
        rng.manual_seed(seed)
        x = torch.randn(
            (
                batch_size * n_samples,
                self.backbone_config.latent_channels,
                depth,
                height,
                width,
            ),
            device=condition.device,
            dtype=condition.dtype,
            generator=rng,
        )
        schedule = torch.linspace(
            self.config.timesteps - 1,
            0,
            steps,
            device=condition.device,
        ).round().long().unique_consecutive()
        for index, timestep in enumerate(schedule):
            t = torch.full(
                (x.shape[0],),
                int(timestep.item()),
                device=x.device,
                dtype=torch.long,
            )
            alpha_bar_t = self.alpha_bar[t].to(x.dtype)
            alpha_t = alpha_bar_t.sqrt().view(-1, 1, 1, 1, 1)
            sigma_t = (1 - alpha_bar_t).sqrt().view(-1, 1, 1, 1, 1)
            velocity = self.model(
                x,
                expanded_condition,
                t.float() / (self.config.timesteps - 1),
                modality_mask=expanded_mask,
                target_ids=expanded_targets,
                quality=expanded_quality,
            )
            predicted_x0 = alpha_t * x - sigma_t * velocity
            predicted_noise = sigma_t * x + alpha_t * velocity
            if index + 1 == len(schedule):
                x = predicted_x0
                continue
            previous_t = schedule[index + 1]
            previous_alpha_bar = self.alpha_bar[previous_t].to(x.dtype)
            ddim_sigma = self.config.ddim_eta * torch.sqrt(
                ((1 - previous_alpha_bar) / (1 - alpha_bar_t))
                * (1 - alpha_bar_t / previous_alpha_bar).clamp_min(0)
            )
            direction_scale = (1 - previous_alpha_bar - ddim_sigma.square()).clamp_min(0).sqrt()
            noise = torch.randn(
                x.shape, device=x.device, dtype=x.dtype, generator=rng
            )
            x = (
                previous_alpha_bar.sqrt().view(-1, 1, 1, 1, 1) * predicted_x0
                + direction_scale.view(-1, 1, 1, 1, 1) * predicted_noise
                + ddim_sigma.view(-1, 1, 1, 1, 1) * noise
            )
        return x.reshape(
            batch_size,
            n_samples,
            self.backbone_config.latent_channels,
            depth,
            height,
            width,
        )

    @property
    def nfe(self) -> int:
        return self.config.sampling_steps
