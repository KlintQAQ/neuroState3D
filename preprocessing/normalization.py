from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from monai.transforms import (
    CenterSpatialCropd,
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    ScaleIntensityRangePercentilesd,
    Spacingd,
    SpatialPadd,
    ToTensord,
)


BRAINMVP_PREPROCESSING_AUDIT = [
    "LoadImaged",
    "EnsureChannelFirstd",
    "Orientationd(axcodes='RAS')",
    "Spacingd(pixdim=(1.0,1.0,1.0), mode='bilinear')",
    "CenterCropForegroundd in official pretraining / CropForegroundd in HCP smoke",
    "ScaleIntensityRangePercentilesd(lower=5, upper=95, b_min=0.0, b_max=1.0, clip=True, channel_wise=True)",
    "Second foreground crop",
    "Official pretraining: RandSpatialCropSamplesd(roi_size=96, random_center=True)",
    "HCP smoke: deterministic CenterSpatialCropd(roi_size=96) plus SpatialPadd",
    "ToTensord",
]


def tensor_stats(tensor: torch.Tensor | np.ndarray) -> dict[str, Any]:
    array = (
        tensor.detach().float().cpu().numpy()
        if torch.is_tensor(tensor)
        else np.asarray(tensor, dtype=np.float32)
    )
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        raise ValueError("No finite values available for statistics.")
    percentiles = np.percentile(finite, [1, 50, 99])
    return {
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "p1": float(percentiles[0]),
        "p50": float(percentiles[1]),
        "p99": float(percentiles[2]),
        "nan": bool(np.isnan(array).any()),
        "inf": bool(np.isinf(array).any()),
    }


def build_brainmvp_hcp_smoke_transform(
    roi_size: int = 96,
    key: str = "image",
) -> Compose:
    return Compose(
        [
            LoadImaged(keys=[key]),
            EnsureChannelFirstd(keys=[key]),
            Orientationd(keys=[key], axcodes="RAS"),
            Spacingd(keys=[key], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
            CropForegroundd(keys=[key], source_key=key),
            ScaleIntensityRangePercentilesd(
                keys=[key],
                lower=5,
                upper=95,
                b_min=0.0,
                b_max=1.0,
                clip=True,
                channel_wise=True,
            ),
            CropForegroundd(keys=[key], source_key=key),
            CenterSpatialCropd(keys=[key], roi_size=(roi_size, roi_size, roi_size)),
            SpatialPadd(keys=[key], spatial_size=(roi_size, roi_size, roi_size), mode="constant"),
            ToTensord(keys=[key]),
        ]
    )


def preprocess_nifti_for_brainmvp(
    path: str | Path,
    roi_size: int = 96,
) -> tuple[torch.Tensor, dict[str, Any]]:
    transform = build_brainmvp_hcp_smoke_transform(roi_size=roi_size)
    output = transform({"image": str(path)})
    tensor = output["image"].float()
    if tensor.ndim != 4:
        raise ValueError(f"Expected [C,D,H,W] after preprocessing, got {tuple(tensor.shape)}")
    metadata = {
        "preprocessing_audit": BRAINMVP_PREPROCESSING_AUDIT,
        "after_shape": list(tensor.shape),
        "after_stats": tensor_stats(tensor),
    }
    return tensor, metadata

