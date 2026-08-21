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


class ResidualBlock2d(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            ConvNormAct2d(channels, channels),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(_group_count(channels), channels),
        ]
        if dropout > 0:
            layers.insert(1, nn.Dropout2d(float(dropout)))
        self.block = nn.Sequential(*layers)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class DownBlock2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.GELU(),
            ResidualBlock2d(out_channels, dropout),
            ResidualBlock2d(out_channels, dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UpBlock2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            ConvNormAct2d(in_channels, out_channels),
            ResidualBlock2d(out_channels, dropout),
            ResidualBlock2d(out_channels, dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass(frozen=True)
class HighFidelityVirtualModalityGeneratorConfig:
    in_modalities: int = 4
    context_slices: int = 5
    hidden_channels: int = 32
    dropout: float = 0.0
    uncertainty_min: float = 1e-4


class HighFidelityVirtualModalityGenerator(nn.Module):
    """2.5D high-fidelity generator for a single missing MRI modality.

    One checkpoint is trained per target modality. The model receives adjacent
    slices from all modalities and a modality-availability mask, then predicts
    the center slice of the missing target modality plus uncertainty.
    """

    def __init__(
        self,
        config: HighFidelityVirtualModalityGeneratorConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or HighFidelityVirtualModalityGeneratorConfig()
        hidden = int(self.config.hidden_channels)
        in_modalities = int(self.config.in_modalities)
        context_slices = int(self.config.context_slices)
        if context_slices <= 0 or context_slices % 2 != 1:
            raise ValueError("context_slices must be a positive odd integer.")
        in_channels = in_modalities * context_slices + in_modalities
        dropout = float(self.config.dropout)
        self.stem = nn.Sequential(
            ConvNormAct2d(in_channels, hidden),
            ResidualBlock2d(hidden, dropout),
            ResidualBlock2d(hidden, dropout),
        )
        self.down1 = DownBlock2d(hidden, hidden * 2, dropout)
        self.down2 = DownBlock2d(hidden * 2, hidden * 4, dropout)
        self.down3 = DownBlock2d(hidden * 4, hidden * 8, dropout)
        self.bottleneck = nn.Sequential(
            ResidualBlock2d(hidden * 8, dropout),
            ResidualBlock2d(hidden * 8, dropout),
            ResidualBlock2d(hidden * 8, dropout),
        )
        self.up2 = UpBlock2d(hidden * 8 + hidden * 4, hidden * 4, dropout)
        self.up1 = UpBlock2d(hidden * 4 + hidden * 2, hidden * 2, dropout)
        self.up0 = UpBlock2d(hidden * 2 + hidden, hidden, dropout)
        self.detail = nn.Sequential(
            ConvNormAct2d(hidden + in_modalities, hidden),
            ResidualBlock2d(hidden, dropout),
            nn.Conv2d(hidden, 2, kernel_size=1),
        )

    def forward(
        self,
        context: torch.Tensor,
        modality_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if context.ndim == 4:
            context = context.unsqueeze(2)
        if context.ndim != 5:
            raise ValueError(
                f"Expected context [B,M,K,H,W], got {tuple(context.shape)}"
            )
        b, m, k, h, w = context.shape
        if m != int(self.config.in_modalities):
            raise ValueError(f"Expected {self.config.in_modalities} modalities, got {m}")
        if k != int(self.config.context_slices):
            raise ValueError(f"Expected {self.config.context_slices} context slices, got {k}")
        modality_mask = modality_mask.to(device=context.device, dtype=context.dtype)
        if tuple(modality_mask.shape) != (b, m):
            raise ValueError(
                f"Expected modality_mask {(b, m)}, got {tuple(modality_mask.shape)}"
            )

        masked = context * modality_mask.view(b, m, 1, 1, 1)
        flat_context = masked.reshape(b, m * k, h, w)
        center = masked[:, :, k // 2]
        mask_channels = modality_mask.view(b, m, 1, 1).expand(b, m, h, w)
        x0 = self.stem(torch.cat([flat_context, mask_channels], dim=1))
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.bottleneck(self.down3(x2))
        x = F.interpolate(x3, size=x2.shape[2:], mode="bilinear", align_corners=False)
        x = self.up2(torch.cat([x, x2], dim=1))
        x = F.interpolate(x, size=x1.shape[2:], mode="bilinear", align_corners=False)
        x = self.up1(torch.cat([x, x1], dim=1))
        x = F.interpolate(x, size=x0.shape[2:], mode="bilinear", align_corners=False)
        x = self.up0(torch.cat([x, x0], dim=1))
        pred = self.detail(torch.cat([x, center], dim=1))
        synthetic = torch.tanh(pred[:, :1])
        uncertainty = F.softplus(pred[:, 1:2]) + float(self.config.uncertainty_min)
        confidence = torch.exp(-uncertainty)
        return {
            "synthetic": synthetic,
            "uncertainty": uncertainty,
            "confidence": confidence,
        }
