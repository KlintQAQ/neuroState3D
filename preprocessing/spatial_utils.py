from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np


@dataclass(frozen=True)
class NiftiAudit:
    path: str
    shape: list[int]
    dtype: str
    affine: list[list[float]]
    voxel_spacing: list[float]
    orientation_codes: list[str]
    qform_code: int
    sform_code: int
    qform: list[list[float]]
    sform: list[list[float]]
    stats: dict[str, Any]


def _finite_stats(data: np.ndarray) -> dict[str, Any]:
    values = np.asarray(data, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("NIfTI volume contains no finite values.")
    percentiles = np.percentile(finite, [1, 5, 50, 95, 99])
    return {
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "p1": float(percentiles[0]),
        "p5": float(percentiles[1]),
        "p50": float(percentiles[2]),
        "p95": float(percentiles[3]),
        "p99": float(percentiles[4]),
        "nan": bool(np.isnan(values).any()),
        "inf": bool(np.isinf(values).any()),
    }


def inspect_nifti(path: str | Path, load_data: bool = True) -> NiftiAudit:
    image = nib.load(str(path))
    header = image.header
    qform, qform_code = image.get_qform(coded=True)
    sform, sform_code = image.get_sform(coded=True)
    stats = _finite_stats(np.asanyarray(image.dataobj)) if load_data else {}
    return NiftiAudit(
        path=str(path),
        shape=[int(dim) for dim in image.shape],
        dtype=str(header.get_data_dtype()),
        affine=np.asarray(image.affine, dtype=float).tolist(),
        voxel_spacing=[float(value) for value in header.get_zooms()[:3]],
        orientation_codes=list(nib.aff2axcodes(image.affine)),
        qform_code=int(qform_code),
        sform_code=int(sform_code),
        qform=np.asarray(qform, dtype=float).tolist(),
        sform=np.asarray(sform, dtype=float).tolist(),
        stats=stats,
    )


def physical_points(shape: list[int], affine: np.ndarray) -> np.ndarray:
    max_index = np.asarray(shape[:3], dtype=float) - 1.0
    corners = np.array(
        [
            [0, 0, 0],
            [max_index[0], 0, 0],
            [0, max_index[1], 0],
            [0, 0, max_index[2]],
            [max_index[0], max_index[1], max_index[2]],
            max_index / 2.0,
        ],
        dtype=float,
    )
    hom = np.concatenate([corners, np.ones((corners.shape[0], 1))], axis=1)
    return (affine @ hom.T).T[:, :3]


def compare_nifti_space(
    left: NiftiAudit,
    right: NiftiAudit,
    affine_atol: float = 1e-4,
    spacing_atol: float = 1e-4,
    physical_atol: float = 1e-3,
) -> dict[str, Any]:
    left_affine = np.asarray(left.affine, dtype=float)
    right_affine = np.asarray(right.affine, dtype=float)
    shape_match = left.shape[:3] == right.shape[:3]
    spacing_match = np.allclose(
        np.asarray(left.voxel_spacing),
        np.asarray(right.voxel_spacing),
        atol=spacing_atol,
        rtol=0.0,
    )
    orientation_match = left.orientation_codes == right.orientation_codes
    affine_match = np.allclose(left_affine, right_affine, atol=affine_atol, rtol=0.0)
    physical_delta = np.abs(
        physical_points(left.shape, left_affine)
        - physical_points(right.shape, right_affine)
    )
    physical_match = bool(np.max(physical_delta) <= physical_atol)
    same_space: str
    if shape_match and spacing_match and orientation_match and affine_match and physical_match:
        same_space = "TRUE"
    elif not shape_match or not spacing_match or not orientation_match:
        same_space = "FALSE"
    else:
        same_space = "NEEDS_REVIEW"
    return {
        "T1_T2_SHAPE_MATCH": bool(shape_match),
        "T1_T2_SPACING_MATCH": bool(spacing_match),
        "T1_T2_ORIENTATION_MATCH": bool(orientation_match),
        "T1_T2_AFFINE_MATCH": bool(affine_match),
        "T1_T2_PHYSICAL_POINTS_MATCH": physical_match,
        "max_physical_coordinate_delta_mm": float(np.max(physical_delta)),
        "T1_T2_SAME_PHYSICAL_SPACE": same_space,
    }

