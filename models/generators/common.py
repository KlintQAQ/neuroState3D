from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class PosteriorBackboneConfig:
    """Shared architecture contract for the two posterior engines."""

    latent_channels: int
    condition_channels: int
    hidden_channels: int = 64
    depth: int = 4
    num_modalities: int = 5
    num_targets: int = 5
    quality_channels: int = 0
    embedding_dim: int = 128


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.dimension = dimension

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = value.float().reshape(-1)
        half = self.dimension // 2
        if half == 0:
            return value[:, None]
        scale = -math.log(10000.0) / max(half - 1, 1)
        frequencies = torch.exp(
            torch.arange(half, device=value.device, dtype=value.dtype) * scale
        )
        angles = value[:, None] * frequencies[None]
        embedding = torch.cat([angles.sin(), angles.cos()], dim=1)
        if embedding.shape[1] < self.dimension:
            embedding = F.pad(embedding, (0, self.dimension - embedding.shape[1]))
        return embedding


class ConditionedResidualBlock3D(nn.Module):
    def __init__(self, channels: int, embedding_dim: int) -> None:
        super().__init__()
        groups = _group_count(channels)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.film = nn.Linear(embedding_dim, 2 * channels)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.film(embedding).chunk(2, dim=1)
        x = self.norm2(x)
        x = x * (1 + scale[:, :, None, None, None])
        x = x + shift[:, :, None, None, None]
        x = self.conv2(F.silu(x))
        return residual + x


class ConditionalLatentBackbone3D(nn.Module):
    """Compact 3D residual backbone shared by diffusion and drifting.

    The sample/noise tensor and observed-evidence condition stay separate in
    the public API. Modality availability, target identity, and optional
    spatial quality maps are explicit conditions rather than fake modalities.
    """

    def __init__(self, config: PosteriorBackboneConfig) -> None:
        super().__init__()
        self.config = config
        total_in = (
            config.latent_channels
            + config.condition_channels
            + config.quality_channels
        )
        self.input_projection = nn.Conv3d(
            total_in, config.hidden_channels, kernel_size=3, padding=1
        )
        self.time_embedding = nn.Sequential(
            SinusoidalEmbedding(config.embedding_dim),
            nn.Linear(config.embedding_dim, config.embedding_dim),
            nn.SiLU(),
            nn.Linear(config.embedding_dim, config.embedding_dim),
        )
        self.mask_embedding = nn.Sequential(
            nn.Linear(config.num_modalities, config.embedding_dim),
            nn.SiLU(),
            nn.Linear(config.embedding_dim, config.embedding_dim),
        )
        self.target_embedding = nn.Embedding(config.num_targets, config.embedding_dim)
        self.blocks = nn.ModuleList(
            [
                ConditionedResidualBlock3D(
                    config.hidden_channels, config.embedding_dim
                )
                for _ in range(config.depth)
            ]
        )
        self.output_norm = nn.GroupNorm(
            _group_count(config.hidden_channels), config.hidden_channels
        )
        self.output_projection = nn.Conv3d(
            config.hidden_channels, config.latent_channels, kernel_size=3, padding=1
        )
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(
        self,
        sample: torch.Tensor,
        condition: torch.Tensor,
        time: torch.Tensor,
        modality_mask: Optional[torch.Tensor] = None,
        target_ids: Optional[torch.Tensor] = None,
        quality: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self._validate_spatial_inputs(sample, condition, quality)
        batch_size = sample.shape[0]
        if modality_mask is None:
            modality_mask = sample.new_ones(
                (batch_size, self.config.num_modalities)
            )
        if modality_mask.shape != (batch_size, self.config.num_modalities):
            raise ValueError(
                "modality_mask must have shape "
                f"{(batch_size, self.config.num_modalities)}, got "
                f"{tuple(modality_mask.shape)}"
            )
        if target_ids is None:
            target_ids = torch.zeros(batch_size, dtype=torch.long, device=sample.device)
        if target_ids.shape != (batch_size,):
            raise ValueError(
                f"target_ids must have shape {(batch_size,)}, got {tuple(target_ids.shape)}"
            )

        inputs = [sample, condition]
        if self.config.quality_channels:
            if quality is None:
                quality = sample.new_zeros(
                    (batch_size, self.config.quality_channels, *sample.shape[2:])
                )
            inputs.append(quality)

        embedding = self.time_embedding(time)
        embedding = embedding + self.mask_embedding(modality_mask.to(sample.dtype))
        embedding = embedding + self.target_embedding(target_ids.long())
        x = self.input_projection(torch.cat(inputs, dim=1))
        for block in self.blocks:
            x = block(x, embedding)
        return self.output_projection(F.silu(self.output_norm(x)))

    def _validate_spatial_inputs(
        self,
        sample: torch.Tensor,
        condition: torch.Tensor,
        quality: Optional[torch.Tensor],
    ) -> None:
        if sample.ndim != 5 or condition.ndim != 5:
            raise ValueError("sample and condition must be [B,C,D,H,W]")
        if sample.shape[0] != condition.shape[0] or sample.shape[2:] != condition.shape[2:]:
            raise ValueError("sample and condition batch/spatial shapes must match")
        if sample.shape[1] != self.config.latent_channels:
            raise ValueError(
                f"Expected {self.config.latent_channels} latent channels, got {sample.shape[1]}"
            )
        if condition.shape[1] != self.config.condition_channels:
            raise ValueError(
                f"Expected {self.config.condition_channels} condition channels, got {condition.shape[1]}"
            )
        if quality is not None:
            expected = (sample.shape[0], self.config.quality_channels, *sample.shape[2:])
            if quality.shape != expected:
                raise ValueError(
                    f"Expected quality shape {expected}, got {tuple(quality.shape)}"
                )


def repeat_batch(tensor: Optional[torch.Tensor], repeats: int) -> Optional[torch.Tensor]:
    if tensor is None:
        return None
    return tensor.repeat_interleave(repeats, dim=0)
