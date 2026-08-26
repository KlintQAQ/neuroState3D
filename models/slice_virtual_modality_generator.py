from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_count(channels: int) -> int:
    groups = min(8, channels)
    while groups > 1 and (channels % groups != 0 or channels // groups < 2):
        groups -= 1
    return groups


class ConvNormAct2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualConvBlock2d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvNormAct2d(channels, channels),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(_group_count(channels), channels),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


@dataclass(frozen=True)
class SliceVirtualModalityGeneratorConfig:
    in_modalities: int = 4
    hidden_channels: int = 32
    uncertainty_min: float = 1e-4


class SliceVirtualModalityGenerator(nn.Module):
    """2D axial-slice generator for missing MRI modalities.

    The model is intentionally small for local iteration. It predicts a target
    modality slice and a pixel-wise uncertainty map from observed source
    modality slices plus observed-mask channels.
    """

    def __init__(
        self,
        config: SliceVirtualModalityGeneratorConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or SliceVirtualModalityGeneratorConfig()
        hidden = int(self.config.hidden_channels)
        in_channels = int(self.config.in_modalities) * 2
        self.stem = nn.Sequential(
            ConvNormAct2d(in_channels, hidden),
            ResidualConvBlock2d(hidden),
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(hidden, hidden * 2, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(_group_count(hidden * 2), hidden * 2),
            nn.GELU(),
            ResidualConvBlock2d(hidden * 2),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(hidden * 2, hidden * 4, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(_group_count(hidden * 4), hidden * 4),
            nn.GELU(),
            ResidualConvBlock2d(hidden * 4),
        )
        self.bottleneck = nn.Sequential(
            ResidualConvBlock2d(hidden * 4),
            ResidualConvBlock2d(hidden * 4),
        )
        self.up1 = nn.Sequential(
            ConvNormAct2d(hidden * 4 + hidden * 2, hidden * 2),
            ResidualConvBlock2d(hidden * 2),
        )
        self.up2 = nn.Sequential(
            ConvNormAct2d(hidden * 2 + hidden, hidden),
            ResidualConvBlock2d(hidden),
        )
        self.out = nn.Conv2d(hidden, 2, kernel_size=1)

    def forward(
        self,
        slices: torch.Tensor,
        modality_mask: torch.Tensor,
        class_condition: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        _ = class_condition
        if slices.ndim != 4:
            raise ValueError(f"Expected slices [B,M,H,W], got {tuple(slices.shape)}")
        b, m, h, w = slices.shape
        if m != int(self.config.in_modalities):
            raise ValueError(f"Expected {self.config.in_modalities} modalities, got {m}")
        modality_mask = modality_mask.to(device=slices.device, dtype=slices.dtype)
        if tuple(modality_mask.shape) != (b, m):
            raise ValueError(
                f"Expected modality_mask {(b, m)}, got {tuple(modality_mask.shape)}"
            )
        masked = slices * modality_mask.view(b, m, 1, 1)
        mask_channels = modality_mask.view(b, m, 1, 1).expand(b, m, h, w)
        x0 = self.stem(torch.cat([masked, mask_channels], dim=1))
        x1 = self.down1(x0)
        x2 = self.bottleneck(self.down2(x1))
        x = F.interpolate(x2, size=x1.shape[2:], mode="bilinear", align_corners=False)
        x = self.up1(torch.cat([x, x1], dim=1))
        x = F.interpolate(x, size=x0.shape[2:], mode="bilinear", align_corners=False)
        x = self.up2(torch.cat([x, x0], dim=1))
        pred = self.out(x)
        synthetic = torch.tanh(pred[:, :1])
        uncertainty = F.softplus(pred[:, 1:2]) + float(self.config.uncertainty_min)
        confidence = torch.exp(-uncertainty)
        return {
            "synthetic": synthetic,
            "uncertainty": uncertainty,
            "confidence": confidence,
        }
