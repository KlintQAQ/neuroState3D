from __future__ import annotations

from typing import Mapping

import torch
import torch.nn as nn


class MeanFusion(nn.Module):
    """Mask-aware mean fusion baseline.

    Input shape:
        ``features``: ``[B, M, C, D, H, W]``
        ``mask``: ``[B, M]`` with 1 for observed modalities.

    Output shape:
        ``fused``: ``[B, C, D, H, W]``
        ``weights``: ``[B, M, 1, 1, 1, 1]``
    """

    def forward(
        self, features: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if features.ndim != 6:
            raise ValueError(
                f"Expected features [B,M,C,D,H,W], got {tuple(features.shape)}"
            )
        if mask.ndim != 2:
            raise ValueError(f"Expected mask [B,M], got {tuple(mask.shape)}")
        if features.shape[:2] != mask.shape:
            raise ValueError(
                f"Feature/modal mask mismatch: {tuple(features.shape[:2])} vs {tuple(mask.shape)}"
            )

        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        weights = (mask / denom).to(features.dtype).view(
            mask.shape[0], mask.shape[1], 1, 1, 1, 1
        )
        fused = (features * weights).sum(dim=1)
        return fused, weights


class ConcatFusion(nn.Module):
    """Fixed-slot concat fusion baseline with modality mask conditioning.

    Missing modality slots are zero-filled before projection, while the mask is
    appended as spatially broadcast channels so the model can distinguish
    missing features from genuine zero-valued features.

    Input shape:
        ``features``: ``[B, M, C, D, H, W]``
        ``mask``: ``[B, M]``

    Output shape:
        ``fused``: ``[B, out_channels, D, H, W]``
        ``weights``: ``None``
    """

    def __init__(
        self,
        num_modalities: int,
        in_channels: int,
        out_channels: int | None = None,
        hidden_channels: int | None = None,
    ) -> None:
        super().__init__()
        self.num_modalities = num_modalities
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        projection_in = num_modalities * in_channels + num_modalities
        hidden = hidden_channels or self.out_channels
        self.projection = nn.Sequential(
            nn.Conv3d(projection_in, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(hidden, self.out_channels, kernel_size=1),
        )

    def forward(
        self, features: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, None]:
        if features.ndim != 6:
            raise ValueError(
                f"Expected features [B,M,C,D,H,W], got {tuple(features.shape)}"
            )
        if mask.ndim != 2:
            raise ValueError(f"Expected mask [B,M], got {tuple(mask.shape)}")
        b, m, c, d, h, w = features.shape
        if m != self.num_modalities or c != self.in_channels:
            raise ValueError(
                "ConcatFusion was built for "
                f"M={self.num_modalities}, C={self.in_channels}; got M={m}, C={c}"
            )

        mask_float = mask.to(features.dtype)
        masked_features = features * mask_float.view(b, m, 1, 1, 1, 1)
        fixed_slots = masked_features.reshape(b, m * c, d, h, w)
        mask_channels = mask_float.view(b, m, 1, 1, 1).expand(b, m, d, h, w)
        fused = self.projection(torch.cat([fixed_slots, mask_channels], dim=1))
        return fused, None


def stack_modalities(
    feature_by_modality: Mapping[str, torch.Tensor],
    modality_mask: torch.Tensor,
    modality_order: list[str],
) -> torch.Tensor:
    """Build fixed-slot ``[B,M,C,D,H,W]`` features for fusion."""

    if not feature_by_modality:
        raise ValueError("feature_by_modality cannot be empty")
    reference = next(iter(feature_by_modality.values()))
    b, c, d, h, w = reference.shape
    slots = []
    for index, modality in enumerate(modality_order):
        if modality in feature_by_modality:
            feature = feature_by_modality[modality]
            if feature.shape != reference.shape:
                raise ValueError(
                    f"Feature shape mismatch for {modality}: "
                    f"{tuple(feature.shape)} vs {tuple(reference.shape)}"
                )
            slots.append(feature)
        else:
            slots.append(reference.new_zeros((b, c, d, h, w)))
    return torch.stack(slots, dim=1)


def build_fusion(
    fusion_type: str,
    num_modalities: int,
    in_channels: int,
    out_channels: int | None = None,
    hidden_channels: int | None = None,
) -> nn.Module:
    if fusion_type == "mean":
        return MeanFusion()
    if fusion_type == "concat":
        return ConcatFusion(
            num_modalities=num_modalities,
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
        )
    raise ValueError("fusion_type must be one of: mean, concat")
