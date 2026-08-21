from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets.brats_fusion_dataset import BRATS_MODALITIES
from models.evidence_fusion import ConvNormAct, ResidualConvBlock


@dataclass(frozen=True)
class VirtualModalityGeneratorConfig:
    modalities: Sequence[str] = BRATS_MODALITIES
    hidden_channels: int = 16
    state_embedding_dim: int = 8
    uncertainty_min: float = 1e-4


class VirtualModalityGenerator(nn.Module):
    """Lightweight 3D conditional generator for missing MRI modalities.

    The module predicts one target modality from the observed modality tensor,
    modality availability mask, and a target-modality id. It also predicts a
    voxel-wise uncertainty map, which is meant to control downstream fusion
    rather than letting synthetic evidence masquerade as real evidence.
    """

    def __init__(self, config: VirtualModalityGeneratorConfig | None = None) -> None:
        super().__init__()
        self.config = config or VirtualModalityGeneratorConfig()
        self.modalities = tuple(name.lower() for name in self.config.modalities)
        self.num_modalities = len(self.modalities)
        hidden = int(self.config.hidden_channels)
        state_dim = int(self.config.state_embedding_dim)

        self.target_embedding = nn.Embedding(self.num_modalities, state_dim)
        in_channels = self.num_modalities * 2 + state_dim
        self.stem = nn.Sequential(
            ConvNormAct(in_channels, hidden),
            ResidualConvBlock(hidden),
        )
        self.down1 = nn.Sequential(
            nn.Conv3d(hidden, hidden * 2, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(_group_count(hidden * 2), hidden * 2),
            nn.GELU(),
            ResidualConvBlock(hidden * 2),
        )
        self.down2 = nn.Sequential(
            nn.Conv3d(hidden * 2, hidden * 4, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(_group_count(hidden * 4), hidden * 4),
            nn.GELU(),
            ResidualConvBlock(hidden * 4),
        )
        self.bottleneck = nn.Sequential(
            ResidualConvBlock(hidden * 4),
            ResidualConvBlock(hidden * 4),
        )
        self.up1 = nn.Sequential(
            ConvNormAct(hidden * 4 + hidden * 2, hidden * 2),
            ResidualConvBlock(hidden * 2),
        )
        self.up2 = nn.Sequential(
            ConvNormAct(hidden * 2 + hidden, hidden),
            ResidualConvBlock(hidden),
        )
        self.out = nn.Conv3d(hidden, 2, kernel_size=1)

    def forward(
        self,
        images: torch.Tensor,
        modality_mask: torch.Tensor,
        target_index: int | torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if images.ndim != 5:
            raise ValueError(f"Expected images [B,M,D,H,W], got {tuple(images.shape)}")
        b, m, d, h, w = images.shape
        if m != self.num_modalities:
            raise ValueError(f"Expected {self.num_modalities} modalities, got {m}.")
        modality_mask = modality_mask.to(device=images.device, dtype=images.dtype)
        if modality_mask.shape != (b, m):
            raise ValueError(
                f"Expected modality_mask shape {(b, m)}, got {tuple(modality_mask.shape)}"
            )
        if torch.is_tensor(target_index):
            target_ids = target_index.to(device=images.device, dtype=torch.long)
            if target_ids.ndim == 0:
                target_ids = target_ids.view(1).expand(b)
        else:
            target_ids = torch.full(
                (b,),
                int(target_index),
                dtype=torch.long,
                device=images.device,
            )
        if target_ids.shape != (b,):
            raise ValueError(f"Expected target_index shape {(b,)}, got {tuple(target_ids.shape)}")

        masked_images = images * modality_mask.view(b, m, 1, 1, 1)
        mask_channels = modality_mask.view(b, m, 1, 1, 1).expand(b, m, d, h, w)
        target_condition = self.target_embedding(target_ids)
        target_condition = target_condition.view(b, -1, 1, 1, 1).expand(b, -1, d, h, w)
        x0 = self.stem(torch.cat([masked_images, mask_channels, target_condition], dim=1))
        x1 = self.down1(x0)
        x2 = self.bottleneck(self.down2(x1))
        x = F.interpolate(x2, size=x1.shape[2:], mode="trilinear", align_corners=False)
        x = self.up1(torch.cat([x, x1], dim=1))
        x = F.interpolate(x, size=x0.shape[2:], mode="trilinear", align_corners=False)
        x = self.up2(torch.cat([x, x0], dim=1))
        pred = self.out(x)
        synthetic = torch.tanh(pred[:, :1])
        uncertainty = F.softplus(pred[:, 1:2]) + float(self.config.uncertainty_min)
        confidence = torch.exp(-uncertainty)
        return {
            "synthetic": synthetic,
            "uncertainty": uncertainty,
            "confidence": confidence,
            "target_index": target_ids,
        }


def _group_count(channels: int) -> int:
    groups = min(8, channels)
    while groups > 1 and (channels % groups != 0 or channels // groups < 2):
        groups -= 1
    return groups


def observed_mask_without_target(
    batch_size: int,
    num_modalities: int,
    target_index: int,
    device: torch.device,
) -> torch.Tensor:
    mask = torch.ones(
        batch_size,
        num_modalities,
        dtype=torch.float32,
        device=device,
    )
    mask[:, int(target_index)] = 0.0
    return mask
