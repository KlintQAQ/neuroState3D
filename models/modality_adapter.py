from __future__ import annotations

from typing import Mapping

import torch
import torch.nn as nn


MODALITY_ORDER = ("t1", "t2", "fa", "md", "alff")
DEFAULT_ADAPTER_TYPES = {
    "t1": "identity",
    "t2": "identity",
    "fa": "residual_conv",
    "md": "residual_conv",
    "alff": "residual_conv",
}
SUPPORTED_MODALITIES = MODALITY_ORDER


class IdentityAdapter(nn.Module):
    """No-op modality adapter.

    Input shape: ``[B, C, D, H, W]``
    Output shape: ``[B, C, D, H, W]``
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class ResidualConvAdapter(nn.Module):
    """Lightweight residual Conv3D modality adapter.

    Input shape: ``[B, C, D, H, W]``
    Output shape: ``[B, C, D, H, W]``
    """

    def __init__(
        self,
        channels: int = 1,
        hidden_channels: int = 8,
        norm: str = "instance",
    ) -> None:
        super().__init__()
        if norm == "instance":
            norm_layer: nn.Module = nn.InstanceNorm3d(hidden_channels, affine=True)
        elif norm == "batch":
            norm_layer = nn.BatchNorm3d(hidden_channels)
        elif norm == "none":
            norm_layer = nn.Identity()
        else:
            raise ValueError("norm must be one of: instance, batch, none")

        self.net = nn.Sequential(
            nn.Conv3d(channels, hidden_channels, kernel_size=3, padding=1),
            norm_layer,
            nn.GELU(),
            nn.Conv3d(hidden_channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


def build_modality_adapter(
    adapter_type: str,
    channels: int = 1,
    hidden_channels: int = 8,
    norm: str = "instance",
) -> nn.Module:
    if adapter_type == "identity":
        return IdentityAdapter()
    if adapter_type in {"residual_conv", "conv"}:
        return ResidualConvAdapter(
            channels=channels, hidden_channels=hidden_channels, norm=norm
        )
    raise ValueError(f"Unknown adapter type: {adapter_type}")


class ModalityAdapterBank(nn.Module):
    """Configurable adapter collection keyed by modality name.

    Input shape per modality: ``[B, C, D, H, W]``
    Output shape per modality: ``[B, C, D, H, W]``
    """

    def __init__(
        self,
        modalities: list[str],
        adapter_types: Mapping[str, str] | None = None,
        channels: int = 1,
        hidden_channels: int = 8,
        norm: str = "instance",
    ) -> None:
        super().__init__()
        self.modalities = [m.lower() for m in modalities]
        adapter_types = adapter_types or {}
        adapters = {}
        for modality in self.modalities:
            adapter_type = adapter_types.get(modality, "identity")
            adapters[modality] = build_modality_adapter(
                adapter_type,
                channels=channels,
                hidden_channels=hidden_channels,
                norm=norm,
            )
        self.adapters = nn.ModuleDict(adapters)

    def forward(self, x: torch.Tensor, modality: str) -> torch.Tensor:
        key = modality.lower()
        if key not in self.adapters:
            raise KeyError(f"Unknown modality '{modality}'. Known: {self.modalities}")
        return self.adapters[key](x)
