from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.brats_fusion_dataset import BRATS_MODALITIES, BRATS_REGIONS  # noqa: E402
from models.slice_virtual_modality_generator import (  # noqa: E402
    SliceVirtualModalityGenerator,
    SliceVirtualModalityGeneratorConfig,
)
from models.prompted_slice_virtual_modality_generator import (  # noqa: E402
    PromptedSliceVirtualModalityGenerator,
    PromptedSliceVirtualModalityGeneratorConfig,
)
from models.slice_drift_transport_generator import (  # noqa: E402
    SliceDriftTransportGenerator,
    SliceDriftTransportGeneratorConfig,
)
from scripts.smoke_evidence_fusion_brats import environment, git_commit  # noqa: E402
from utils.brats_metrics import dice_scores, segmentation_loss  # noqa: E402
from utils.torch_drift_loss import drift_loss  # noqa: E402

BRATS_DATASETS = ("GLI", "MEN", "PED")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a 2D slice-wise drifting generator for missing BraTS T1c."
    )
    parser.add_argument(
        "--manifest",
        default=str(
            ROOT
            / "data"
            / "manifests"
            / "BraTS2023_HF"
            / "brats_model_ready_processed.csv"
        ),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--target-modality", default="t1c", choices=BRATS_MODALITIES)
    parser.add_argument("--spatial-size", type=int, default=64)
    parser.add_argument("--max-subjects", type=int, default=512)
    parser.add_argument("--val-subjects", type=int, default=64)
    parser.add_argument("--slices-per-subject", type=int, default=4)
    parser.add_argument("--slice-context-radius", type=int, default=0)
    parser.add_argument("--slice-crop-size", type=int, default=0)
    parser.add_argument("--val-slice-crop-size", type=int, default=-1)
    parser.add_argument("--slice-crop-jitter", type=int, default=8)
    parser.add_argument(
        "--slice-crop-mode",
        default="none",
        choices=("none", "region_balanced"),
    )
    parser.add_argument(
        "--val-slice-crop-mode",
        default="",
        choices=("", "none", "region_balanced"),
    )
    parser.add_argument("--hard-slice-json", default="")
    parser.add_argument("--hard-slice-prob", type=float, default=0.0)
    parser.add_argument("--hard-slice-min-score", type=float, default=0.0)
    parser.add_argument("--hard-slice-top-k", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-train-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--model-kind", default="slice", choices=("slice", "prompted", "transport"))
    parser.add_argument(
        "--best-metric",
        default="mae",
        choices=("mae", "lesion_composite", "lesion_noharm_composite"),
    )
    parser.add_argument("--freeze-base-generator", action="store_true")
    parser.add_argument("--detail-only-refinement", action="store_true")
    parser.add_argument("--train-prompt-with-detail", action="store_true")
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--transport-steps", type=int, default=4)
    parser.add_argument("--transport-step-scale", type=float, default=1.0)
    parser.add_argument("--transport-velocity-scale", type=float, default=1.0)
    parser.add_argument("--transport-init-blur-kernel", type=int, default=5)
    parser.add_argument("--transport-velocity-weight", type=float, default=0.0)
    parser.add_argument("--transport-path-weight", type=float, default=0.0)
    parser.add_argument("--transport-monotonic-weight", type=float, default=0.0)
    parser.add_argument("--gated-refinement", action="store_true")
    parser.add_argument("--refinement-residual-scale", type=float, default=0.25)
    parser.add_argument("--gate-bias-init", type=float, default=-3.0)
    parser.add_argument("--refinement-acceptance-gate", action="store_true")
    parser.add_argument("--accept-bias-init", type=float, default=-2.0)
    parser.add_argument("--refinement-channels-multiplier", type=int, default=1)
    parser.add_argument("--refinement-blocks", type=int, default=1)
    parser.add_argument("--refinement-detail-features", action="store_true")
    parser.add_argument("--gate-supervision-weight", type=float, default=0.0)
    parser.add_argument("--gate-target-dilation", type=int, default=2)
    parser.add_argument("--residual-need-gate-weight", type=float, default=0.0)
    parser.add_argument("--residual-need-gate-threshold", type=float, default=0.05)
    parser.add_argument("--gate-sparsity-weight", type=float, default=0.0)
    parser.add_argument("--background-preserve-weight", type=float, default=0.0)
    parser.add_argument("--core-overfill-weight", type=float, default=0.0)
    parser.add_argument("--acceptance-supervision-weight", type=float, default=0.0)
    parser.add_argument("--acceptance-error-threshold", type=float, default=0.04)
    parser.add_argument("--refinement-residual-target-weight", type=float, default=0.0)
    parser.add_argument("--refinement-direction-weight", type=float, default=0.0)
    parser.add_argument("--refinement-residual-budget-weight", type=float, default=0.0)
    parser.add_argument("--refinement-lesion-budget-weight", type=float, default=0.0)
    parser.add_argument("--refinement-no-harm-weight", type=float, default=0.0)
    parser.add_argument("--refinement-lesion-no-harm-weight", type=float, default=0.0)
    parser.add_argument("--refinement-background-no-harm-weight", type=float, default=0.0)
    parser.add_argument("--refinement-no-harm-margin", type=float, default=0.0)
    parser.add_argument("--residual-scale", type=float, default=0.35)
    parser.add_argument("--lesion-residual-scale", type=float, default=0.45)
    parser.add_argument("--detail-residual-scale", type=float, default=0.20)
    parser.add_argument("--enhancement-residual-scale", type=float, default=0.0)
    parser.add_argument("--class-conditioned", action="store_true")
    parser.add_argument(
        "--output-activation",
        default="tanh",
        choices=("tanh", "hardtanh", "none"),
    )
    parser.add_argument("--positive-lesion-residual", action="store_true")
    parser.add_argument("--reset-lesion-residual-bias", type=float, default=None)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--recon-weight", type=float, default=1.0)
    parser.add_argument("--nll-weight", type=float, default=0.04)
    parser.add_argument("--gradient-weight", type=float, default=0.12)
    parser.add_argument("--multiscale-ssim-weight", type=float, default=0.0)
    parser.add_argument("--lesion-multiscale-ssim-weight", type=float, default=0.0)
    parser.add_argument("--laplacian-pyramid-weight", type=float, default=0.0)
    parser.add_argument("--lesion-laplacian-pyramid-weight", type=float, default=0.0)
    parser.add_argument("--fidelity-pyramid-levels", type=int, default=3)
    parser.add_argument("--drift-weight", type=float, default=0.12)
    parser.add_argument("--drift-patch-size", type=int, default=4)
    parser.add_argument("--drift-radii", nargs="+", type=float, default=[0.2, 0.05, 0.02])
    parser.add_argument("--window-drift-weight", type=float, default=0.0)
    parser.add_argument("--window-feature-weight", type=float, default=0.0)
    parser.add_argument("--window-size", type=int, default=9)
    parser.add_argument("--window-stride", type=int, default=8)
    parser.add_argument("--window-drift-radii", nargs="+", type=float, default=[0.004, 0.01, 0.04])
    parser.add_argument("--window-token-clamp", type=float, default=4.0)
    parser.add_argument("--micro-window-feature-weight", type=float, default=0.0)
    parser.add_argument("--micro-window-drift-weight", type=float, default=0.0)
    parser.add_argument("--micro-window-drift-radii", nargs="+", type=float, default=[0.006, 0.015, 0.04])
    parser.add_argument("--micro-window-drift-max-tokens", type=int, default=256)
    parser.add_argument("--micro-window-sizes", nargs="+", type=int, default=[3, 5, 7])
    parser.add_argument("--micro-window-stride", type=int, default=2)
    parser.add_argument("--medical-drift-weight", type=float, default=0.0)
    parser.add_argument("--medical-drift-window-sizes", nargs="+", type=int, default=[4, 8, 16])
    parser.add_argument("--medical-drift-radii", nargs="+", type=float, default=[0.02, 0.05, 0.2])
    parser.add_argument("--medical-drift-token-clamp", type=float, default=4.0)
    parser.add_argument("--medical-drift-bank-size", type=int, default=4096)
    parser.add_argument("--medical-drift-memory-tokens", type=int, default=128)
    parser.add_argument("--medical-drift-max-add-tokens", type=int, default=512)
    parser.add_argument("--medical-drift-max-current-tokens", type=int, default=256)
    parser.add_argument("--lesion-texture-weight", type=float, default=0.0)
    parser.add_argument("--region-moment-weight", type=float, default=0.0)
    parser.add_argument("--edge-weight", type=float, default=0.0)
    parser.add_argument("--enhancement-under-weight", type=float, default=0.0)
    parser.add_argument("--lesion-boundary-weight", type=float, default=0.0)
    parser.add_argument("--enhancement-contrast-weight", type=float, default=0.0)
    parser.add_argument("--lesion-residual-target-weight", type=float, default=0.0)
    parser.add_argument("--enhancement-residual-target-weight", type=float, default=0.0)
    parser.add_argument("--enhancement-leak-weight", type=float, default=0.0)
    parser.add_argument("--top-intensity-weight", type=float, default=0.0)
    parser.add_argument("--top-intensity-quantile", type=float, default=0.80)
    parser.add_argument("--prompt-weight", type=float, default=0.0)
    parser.add_argument("--prompt-balanced-bce-weight", type=float, default=0.0)
    parser.add_argument("--prompt-max-pos-weight", type=float, default=50.0)
    parser.add_argument("--prompt-et-weight", type=float, default=3.0)
    parser.add_argument("--prompt-tc-weight", type=float, default=2.0)
    parser.add_argument("--prompt-wt-weight", type=float, default=1.0)
    parser.add_argument("--support-threshold", type=float, default=1e-5)
    parser.add_argument("--background-weight", type=float, default=0.01)
    parser.add_argument("--focus-dilation", type=int, default=0)
    parser.add_argument("--focus-base", type=float, default=0.10)
    parser.add_argument("--focus-et", type=float, default=4.0)
    parser.add_argument("--focus-tc", type=float, default=2.0)
    parser.add_argument("--focus-wt", type=float, default=0.35)
    parser.add_argument("--resume-checkpoint", default="")
    parser.add_argument("--seed", type=int, default=46)
    parser.add_argument("--split-seed", type=int, default=-1)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "outputs" / "slice_virtual_modality_drifting"),
    )
    parser.add_argument(
        "--report-path",
        default=str(ROOT / "reports" / "slice_virtual_modality_drifting.json"),
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def brats_region_targets_np(seg: np.ndarray) -> np.ndarray:
    et = seg == 3
    tc = (seg == 1) | (seg == 3)
    wt = seg > 0
    return np.stack([et, tc, wt], axis=0).astype(np.float32)


def load_hard_slice_index(
    path: str | Path,
    min_score: float = 0.0,
    top_k: int = 0,
) -> tuple[dict[str, list[tuple[int, float]]], dict[str, Any]]:
    if not path:
        return {}, {"path": "", "subjects": 0, "slices": 0}
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"hard slice file not found: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        records = payload.get("records", payload.get("cases", []))
    elif isinstance(payload, list):
        records = payload
    else:
        raise ValueError(f"Unsupported hard slice payload type: {type(payload).__name__}")
    grouped: dict[str, dict[int, float]] = {}
    dropped = 0
    for item in records:
        if not isinstance(item, dict):
            dropped += 1
            continue
        subject_id = str(item.get("subject_id", "")).strip()
        raw_slice = item.get("slice_index", item.get("z", None))
        if not subject_id or raw_slice is None:
            dropped += 1
            continue
        score = float(
            item.get(
                "score",
                item.get(
                    "hard_score",
                    item.get(
                        "mae_ET",
                        item.get("mae_TC", item.get("mae_WT", item.get("mae", 0.0))),
                    ),
                ),
            )
            or 0.0
        )
        if score < float(min_score):
            dropped += 1
            continue
        slice_index = int(raw_slice)
        subject = grouped.setdefault(subject_id, {})
        subject[slice_index] = max(subject.get(slice_index, float("-inf")), score)
    top_k = int(top_k)
    hard_slices: dict[str, list[tuple[int, float]]] = {}
    for subject_id, slices in grouped.items():
        ordered = sorted(slices.items(), key=lambda item: item[1], reverse=True)
        if top_k > 0:
            ordered = ordered[:top_k]
        if ordered:
            hard_slices[subject_id] = [(int(z), float(score)) for z, score in ordered]
    total = sum(len(items) for items in hard_slices.values())
    return hard_slices, {
        "path": str(source),
        "subjects": len(hard_slices),
        "slices": total,
        "min_score": float(min_score),
        "top_k": top_k,
        "dropped_records": dropped,
    }


class BraTSSliceDataset(Dataset):
    def __init__(
        self,
        manifest_csv: str | Path,
        spatial_size: int,
        max_subjects: int | None,
        slices_per_subject: int,
        target_modality: str,
        seed: int,
        slice_context_radius: int = 0,
        slice_crop_size: int = 0,
        slice_crop_jitter: int = 8,
        slice_crop_mode: str = "none",
        hard_slices: dict[str, list[tuple[int, float]]] | None = None,
        hard_slice_prob: float = 0.0,
    ) -> None:
        rows = read_rows(manifest_csv)
        if max_subjects is not None:
            rows = rows[: int(max_subjects)]
        if not rows:
            raise ValueError(f"No rows selected from {manifest_csv}")
        self.rows = rows
        self.spatial_size = int(spatial_size)
        self.slices_per_subject = int(max(1, slices_per_subject))
        self.base_target_index = BRATS_MODALITIES.index(target_modality)
        self.slice_context_radius = int(max(0, slice_context_radius))
        self.context_depth = 2 * self.slice_context_radius + 1
        self.target_index = self.base_target_index * self.context_depth + self.slice_context_radius
        self.seed = int(seed)
        self.slice_crop_size = int(max(0, slice_crop_size))
        self.slice_crop_jitter = int(max(0, slice_crop_jitter))
        self.slice_crop_mode = slice_crop_mode
        self.hard_slices = hard_slices or {}
        self.hard_slice_prob = float(max(0.0, min(1.0, hard_slice_prob)))

    def __len__(self) -> int:
        return len(self.rows) * self.slices_per_subject

    def __getitem__(self, index: int) -> dict[str, Any]:
        row_index = index // self.slices_per_subject
        local_index = index % self.slices_per_subject
        row = self.rows[row_index]
        image = np.load(row["multimodal_path"]).astype(np.float32, copy=False)
        seg = np.load(row["seg_path"]).astype(np.int64, copy=False)
        rng = random.Random(self.seed + row_index * 1009 + local_index * 9176)
        z = self._sample_slice(seg, rng, row["subject_id"])
        image_slice = self._context_slices(image, z)
        target = torch.from_numpy(brats_region_targets_np(seg[:, :, z]))
        if self.slice_crop_mode == "region_balanced" and self.slice_crop_size > 0:
            image_slice, target = self._crop_slice_around_region(
                image_slice,
                target,
                rng,
                self.slice_crop_size,
                self.slice_crop_jitter,
            )
        if image_slice.shape[-2:] != (self.spatial_size, self.spatial_size):
            image_slice = F.interpolate(
                image_slice.unsqueeze(0),
                size=(self.spatial_size, self.spatial_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            target = F.interpolate(
                target.unsqueeze(0),
                size=(self.spatial_size, self.spatial_size),
                mode="nearest",
            ).squeeze(0)
        image_slice = torch.nan_to_num(image_slice, nan=0.0, posinf=1.0, neginf=-1.0)
        image_slice = image_slice.clamp(-3.0, 3.0)
        mask = torch.ones(len(BRATS_MODALITIES) * self.context_depth, dtype=torch.float32)
        start = self.base_target_index * self.context_depth
        mask[start : start + self.context_depth] = 0.0
        return {
            "image": image_slice,
            "target_regions": target,
            "observed_mask": mask,
            "target_index": self.target_index,
            "dataset": row["dataset"],
            "subject_id": row["subject_id"],
            "slice_index": z,
        }

    def _sample_slice(self, seg: np.ndarray, rng: random.Random, subject_id: str) -> int:
        hard_slices = self.hard_slices.get(subject_id)
        if hard_slices and rng.random() < self.hard_slice_prob:
            depth = int(seg.shape[-1])
            valid = [(z, score) for z, score in hard_slices if 0 <= int(z) < depth]
            if valid:
                weights = [max(float(score), 1e-4) for _, score in valid]
                total = sum(weights)
                cursor = rng.random() * total
                running = 0.0
                for (z, _), weight in zip(valid, weights):
                    running += weight
                    if running >= cursor:
                        return int(z)
                return int(valid[-1][0])
        # Prefer slices containing enhancing tumor or tumor core; fall back to any foreground.
        candidates = np.where(((seg == 3) | (seg == 1)).sum(axis=(0, 1)) > 0)[0]
        if candidates.size == 0:
            candidates = np.where((seg > 0).sum(axis=(0, 1)) > 0)[0]
        if candidates.size == 0:
            return rng.randrange(seg.shape[-1])
        return int(candidates[rng.randrange(candidates.size)])

    def _context_slices(self, image: np.ndarray, z: int) -> torch.Tensor:
        if self.slice_context_radius <= 0:
            return torch.from_numpy(image[:, :, :, z].copy())
        depth = image.shape[-1]
        slices = []
        for modality_index in range(image.shape[0]):
            for offset in range(-self.slice_context_radius, self.slice_context_radius + 1):
                zz = max(0, min(depth - 1, int(z) + offset))
                slices.append(image[modality_index, :, :, zz])
        return torch.from_numpy(np.stack(slices, axis=0).copy())

    @staticmethod
    def _crop_slice_around_region(
        image_slice: torch.Tensor,
        target_regions: torch.Tensor,
        rng: random.Random,
        crop_size: int,
        jitter: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _, height, width = image_slice.shape
        crop = int(max(1, min(crop_size, height, width)))
        if crop >= height and crop >= width:
            return image_slice, target_regions

        candidates = None
        for channel in (0, 1, 2):
            coords = torch.nonzero(target_regions[channel] > 0.5, as_tuple=False)
            if coords.numel() > 0:
                candidates = coords
                break
        if candidates is None:
            support = torch.nonzero(image_slice.abs().amax(dim=0) > 1e-5, as_tuple=False)
            candidates = support if support.numel() > 0 else torch.tensor([[height // 2, width // 2]])

        center = candidates[rng.randrange(int(candidates.shape[0]))].tolist()
        cy = int(center[0]) + rng.randint(-jitter, jitter)
        cx = int(center[1]) + rng.randint(-jitter, jitter)
        top = max(0, min(height - crop, cy - crop // 2))
        left = max(0, min(width - crop, cx - crop // 2))
        return (
            image_slice[:, top : top + crop, left : left + crop],
            target_regions[:, top : top + crop, left : left + crop],
        )


def subject_level_split_indices(
    dataset: BraTSSliceDataset,
    val_subjects: int,
    seed: int,
) -> tuple[list[int], list[int], dict[str, Any]]:
    row_count = len(dataset.rows)
    if row_count <= 1:
        indices = list(range(len(dataset)))
        return indices, indices, {"mode": "single_subject_fallback", "val_subjects": row_count}
    val_count = min(max(1, int(val_subjects)), max(1, row_count // 4))
    rows = list(range(row_count))
    rng = random.Random(int(seed))
    rng.shuffle(rows)
    val_rows = set(rows[:val_count])
    train_rows = [row for row in rows[val_count:]]
    if not train_rows:
        train_rows = [rows[-1]]
        val_rows = set(rows[:-1])
    train_indices = [
        row * dataset.slices_per_subject + local
        for row in train_rows
        for local in range(dataset.slices_per_subject)
    ]
    val_indices = [
        row * dataset.slices_per_subject + local
        for row in sorted(val_rows)
        for local in range(dataset.slices_per_subject)
    ]
    return train_indices, val_indices, {
        "mode": "subject_random",
        "seed": int(seed),
        "train_subjects": len(train_rows),
        "val_subjects": len(val_rows),
    }


def to_device(batch: dict[str, Any], device: str) -> dict[str, Any]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def class_condition_from_batch(batch: dict[str, Any], device: str) -> torch.Tensor | None:
    datasets = batch.get("dataset")
    if datasets is None:
        return None
    condition = torch.zeros(len(datasets), len(BRATS_DATASETS), device=device)
    for row_index, name in enumerate(datasets):
        if name in BRATS_DATASETS:
            condition[row_index, BRATS_DATASETS.index(name)] = 1.0
    return condition


def generator_forward(
    model: torch.nn.Module,
    image: torch.Tensor,
    observed_mask: torch.Tensor,
    batch: dict[str, Any],
) -> dict[str, torch.Tensor]:
    class_condition = (
        class_condition_from_batch(batch, image.device)
        if getattr(model.config, "class_conditioned", False)
        else None
    )
    return model(image, observed_mask, class_condition)


def region_focus(
    target_regions: torch.Tensor,
    base: float,
    et_weight: float,
    tc_weight: float,
    wt_weight: float,
    dilation: int = 0,
) -> torch.Tensor:
    regions = target_regions.float()
    if dilation > 0:
        kernel = int(dilation) * 2 + 1
        regions = F.max_pool2d(regions, kernel_size=kernel, stride=1, padding=int(dilation))
    et = regions[:, 0:1]
    tc = regions[:, 1:2]
    wt = regions[:, 2:3]
    return (
        float(base)
        + float(et_weight) * et
        + float(tc_weight) * tc
        + float(wt_weight) * wt
    ).clamp_min(max(float(base), 1e-4))


def weighted_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return (value * weight).sum() / weight.sum().clamp_min(1e-8)


def transport_velocity_loss(
    generated: dict[str, torch.Tensor],
    target: torch.Tensor,
    focus: torch.Tensor,
    step_scale: float,
) -> torch.Tensor:
    states = generated.get("drift_states")
    velocities = generated.get("drift_velocities")
    if states is None or velocities is None:
        return target.new_zeros(())
    steps = int(velocities.shape[1])
    if steps <= 0:
        return target.new_zeros(())
    scale = max(float(step_scale), 1e-6)
    loss = target.new_zeros(())
    for step in range(steps):
        current = states[:, step].detach()
        remaining = max(steps - step, 1)
        ideal_velocity = (target - current) / (float(remaining) * scale)
        error = F.smooth_l1_loss(
            velocities[:, step],
            ideal_velocity,
            reduction="none",
            beta=0.05,
        )
        loss = loss + weighted_mean(error, focus)
    return loss / float(steps)


def transport_path_loss(
    generated: dict[str, torch.Tensor],
    target: torch.Tensor,
    focus: torch.Tensor,
) -> torch.Tensor:
    states = generated.get("drift_states")
    if states is None or states.shape[1] <= 1:
        return target.new_zeros(())
    steps = int(states.shape[1] - 1)
    initial = states[:, 0].detach()
    loss = target.new_zeros(())
    for step in range(1, steps + 1):
        alpha = float(step) / float(steps)
        waypoint = initial + alpha * (target - initial)
        loss = loss + weighted_mean((states[:, step] - waypoint).abs(), focus)
    return loss / float(steps)


def transport_monotonic_loss(
    generated: dict[str, torch.Tensor],
    target: torch.Tensor,
    focus: torch.Tensor,
) -> torch.Tensor:
    states = generated.get("drift_states")
    if states is None or states.shape[1] <= 1:
        return target.new_zeros(())
    target_path = target.unsqueeze(1)
    previous_error = (states[:, :-1] - target_path).abs()
    next_error = (states[:, 1:] - target_path).abs()
    penalty = F.relu(next_error - previous_error.detach())
    return weighted_mean(penalty, focus.unsqueeze(1))


def observed_brain_support(
    image: torch.Tensor,
    observed_mask: torch.Tensor,
    threshold: float,
    dilation: int,
) -> torch.Tensor:
    mask = observed_mask.to(device=image.device, dtype=image.dtype).view(image.shape[0], image.shape[1], 1, 1)
    observed = (image.abs() * mask).amax(dim=1, keepdim=True)
    support = (observed > float(threshold)).float()
    if dilation > 0:
        kernel = int(dilation) * 2 + 1
        support = F.max_pool2d(support, kernel_size=kernel, stride=1, padding=int(dilation))
    return support


def gradient_difference_loss(
    synthetic: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    losses = []
    for axis in (2, 3):
        pred_grad = synthetic.diff(dim=axis)
        target_grad = target.diff(dim=axis)
        grad_weight = weight.narrow(axis, 0, weight.shape[axis] - 1)
        losses.append(weighted_mean((pred_grad - target_grad).abs(), grad_weight))
    return torch.stack(losses).mean()


def edge_magnitude_2d(image: torch.Tensor) -> torch.Tensor:
    dx = F.pad(image.diff(dim=3), (0, 1, 0, 0))
    dy = F.pad(image.diff(dim=2), (0, 0, 0, 1))
    return (dx.square() + dy.square() + 1e-8).sqrt()


def ssim_map_2d(
    synthetic: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 7,
    data_range: float = 2.0,
) -> torch.Tensor:
    window_size = max(3, int(window_size) | 1)
    padding = window_size // 2
    c1 = (0.01 * float(data_range)) ** 2
    c2 = (0.03 * float(data_range)) ** 2
    mu_x = F.avg_pool2d(
        synthetic,
        kernel_size=window_size,
        stride=1,
        padding=padding,
        count_include_pad=False,
    )
    mu_y = F.avg_pool2d(
        target,
        kernel_size=window_size,
        stride=1,
        padding=padding,
        count_include_pad=False,
    )
    mu_x2 = mu_x.square()
    mu_y2 = mu_y.square()
    sigma_x = F.avg_pool2d(
        synthetic.square(),
        kernel_size=window_size,
        stride=1,
        padding=padding,
        count_include_pad=False,
    ) - mu_x2
    sigma_y = F.avg_pool2d(
        target.square(),
        kernel_size=window_size,
        stride=1,
        padding=padding,
        count_include_pad=False,
    ) - mu_y2
    sigma_xy = F.avg_pool2d(
        synthetic * target,
        kernel_size=window_size,
        stride=1,
        padding=padding,
        count_include_pad=False,
    ) - mu_x * mu_y
    numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x2 + mu_y2 + c1) * (sigma_x + sigma_y + c2)
    return numerator / denominator.clamp_min(1e-8)


def multiscale_ssim_loss(
    synthetic: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    levels: int,
) -> torch.Tensor:
    losses = []
    current_synthetic = synthetic
    current_target = target
    current_weight = weight.to(device=synthetic.device, dtype=synthetic.dtype)
    for _ in range(max(1, int(levels))):
        if current_weight.shape[-2:] != current_synthetic.shape[-2:]:
            current_weight = F.interpolate(
                current_weight,
                size=current_synthetic.shape[-2:],
                mode="nearest",
            )
        if (current_weight > 0.0).any():
            dissimilarity = (1.0 - ssim_map_2d(current_synthetic, current_target).clamp(-1.0, 1.0)) * 0.5
            losses.append(weighted_mean(dissimilarity, current_weight.clamp_min(1e-4)))
        if min(current_synthetic.shape[-2:]) < 24:
            break
        current_synthetic = F.avg_pool2d(current_synthetic, kernel_size=2, stride=2)
        current_target = F.avg_pool2d(current_target, kernel_size=2, stride=2)
        current_weight = F.avg_pool2d(current_weight, kernel_size=2, stride=2)
    if not losses:
        return synthetic.new_zeros(())
    return torch.stack(losses).mean()


def laplacian_pyramid_fidelity_loss(
    synthetic: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    levels: int,
) -> torch.Tensor:
    losses = []
    current_synthetic = synthetic
    current_target = target
    current_weight = weight.to(device=synthetic.device, dtype=synthetic.dtype)
    for _ in range(max(1, int(levels))):
        if min(current_synthetic.shape[-2:]) < 16:
            break
        if current_weight.shape[-2:] != current_synthetic.shape[-2:]:
            current_weight = F.interpolate(
                current_weight,
                size=current_synthetic.shape[-2:],
                mode="nearest",
            )
        low_synthetic = F.avg_pool2d(current_synthetic, kernel_size=2, stride=2)
        low_target = F.avg_pool2d(current_target, kernel_size=2, stride=2)
        high_synthetic = current_synthetic - F.interpolate(
            low_synthetic,
            size=current_synthetic.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        high_target = current_target - F.interpolate(
            low_target,
            size=current_target.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        if (current_weight > 0.0).any():
            detail_error = F.smooth_l1_loss(
                high_synthetic,
                high_target.detach(),
                beta=0.015,
                reduction="none",
            )
            losses.append(weighted_mean(detail_error, current_weight.clamp_min(1e-4)))
        current_synthetic = low_synthetic
        current_target = low_target
        current_weight = F.avg_pool2d(current_weight, kernel_size=2, stride=2)
    if not losses:
        return synthetic.new_zeros(())
    return torch.stack(losses).mean()


def tumor_edge_loss(
    synthetic: torch.Tensor,
    target: torch.Tensor,
    target_regions: torch.Tensor,
    dilation: int,
) -> torch.Tensor:
    wt = target_regions[:, 2:3].float()
    if dilation > 0:
        kernel = int(dilation) * 2 + 1
        outer = F.max_pool2d(wt, kernel_size=kernel, stride=1, padding=int(dilation))
    else:
        outer = wt
    boundary = edge_magnitude_2d(wt).clamp(0.0, 1.0)
    boundary = F.max_pool2d(boundary, kernel_size=3, stride=1, padding=1)
    edge_weight = (0.25 * outer + 2.0 * boundary).clamp_min(1e-4)
    return weighted_mean(
        (edge_magnitude_2d(synthetic) - edge_magnitude_2d(target)).abs(),
        edge_weight,
    )


def region_moment_loss(
    synthetic: torch.Tensor,
    target: torch.Tensor,
    target_regions: torch.Tensor,
) -> torch.Tensor:
    terms = []
    for channel, weight in ((0, 3.0), (1, 2.0), (2, 1.0)):
        mask = target_regions[:, channel : channel + 1] > 0.5
        if not mask.any():
            continue
        pred_values = synthetic[mask]
        target_values = target[mask]
        terms.append(
            float(weight)
            * (
                (pred_values.mean() - target_values.mean()).abs()
                + (pred_values.std(unbiased=False) - target_values.std(unbiased=False)).abs()
            )
        )
    if not terms:
        return synthetic.new_zeros(())
    return torch.stack(terms).mean()


def enhancement_under_loss(
    synthetic: torch.Tensor,
    target: torch.Tensor,
    target_regions: torch.Tensor,
) -> torch.Tensor:
    et = target_regions[:, 0:1].float()
    if not (et > 0.5).any():
        return synthetic.new_zeros(())
    return weighted_mean(F.relu(target - synthetic), et.clamp_min(1e-4))


def lesion_boundary_band_loss(
    synthetic: torch.Tensor,
    target: torch.Tensor,
    target_regions: torch.Tensor,
    dilation: int,
) -> torch.Tensor:
    et = target_regions[:, 0:1].float()
    tc = target_regions[:, 1:2].float()
    lesion = ((et + tc) > 0.5).float()
    if not (lesion > 0.5).any():
        return synthetic.new_zeros(())
    radius = max(1, int(dilation) + 1)
    kernel = 2 * radius + 1
    outer = F.max_pool2d(lesion, kernel_size=kernel, stride=1, padding=radius)
    inner = -F.max_pool2d(-lesion, kernel_size=kernel, stride=1, padding=radius)
    band = (outer - inner).clamp(0.0, 1.0)
    if not (band > 0.0).any():
        return synthetic.new_zeros(())
    intensity = F.smooth_l1_loss(synthetic, target, reduction="none", beta=0.04)
    edge = (edge_magnitude_2d(synthetic) - edge_magnitude_2d(target)).abs()
    return weighted_mean(intensity + 0.5 * edge, band.clamp_min(1e-4))


def enhancement_contrast_loss(
    synthetic: torch.Tensor,
    target: torch.Tensor,
    target_regions: torch.Tensor,
    dilation: int,
) -> torch.Tensor:
    et = target_regions[:, 0:1].float()
    wt = target_regions[:, 2:3].float()
    if not (et > 0.5).any():
        return synthetic.new_zeros(())
    radius = max(1, int(dilation) + 2)
    kernel = 2 * radius + 1
    outer = F.max_pool2d(wt, kernel_size=kernel, stride=1, padding=radius)
    ring = (outer - wt).clamp(0.0, 1.0)
    terms = []
    for sample_index in range(synthetic.shape[0]):
        lesion_mask = et[sample_index : sample_index + 1] > 0.5
        ring_mask = ring[sample_index : sample_index + 1] > 0.5
        if not lesion_mask.any() or not ring_mask.any():
            continue
        pred_contrast = synthetic[sample_index : sample_index + 1][lesion_mask].mean() - synthetic[
            sample_index : sample_index + 1
        ][ring_mask].mean()
        target_contrast = target[sample_index : sample_index + 1][lesion_mask].mean() - target[
            sample_index : sample_index + 1
        ][ring_mask].mean()
        terms.append(F.smooth_l1_loss(pred_contrast, target_contrast.detach(), beta=0.04))
    if not terms:
        return synthetic.new_zeros(())
    return torch.stack(terms).mean()


def lesion_refinement_target(
    target_regions: torch.Tensor,
    dilation: int,
) -> torch.Tensor:
    et = target_regions[:, 0:1].float()
    tc = target_regions[:, 1:2].float()
    wt = target_regions[:, 2:3].float()
    target = (0.2 * wt + 0.55 * tc + et).clamp(0.0, 1.0)
    if dilation > 0:
        radius = int(dilation)
        kernel = 2 * radius + 1
        target = F.max_pool2d(target, kernel_size=kernel, stride=1, padding=radius)
    return target.clamp(0.0, 1.0)


def lesion_gate_supervision_loss(
    generated: dict[str, torch.Tensor],
    target_regions: torch.Tensor,
    dilation: int,
    max_pos_weight: float = 20.0,
) -> torch.Tensor:
    if "refinement_gate_logits" not in generated:
        return target_regions.new_zeros(())
    logits = generated["refinement_gate_logits"]
    target = lesion_refinement_target(target_regions, dilation).to(logits.dtype)
    positive_fraction = target.detach().mean().clamp_min(1e-4)
    pos_weight = ((1.0 - positive_fraction) / positive_fraction).clamp(
        min=1.0,
        max=float(max_pos_weight),
    )
    bce = F.binary_cross_entropy_with_logits(
        logits,
        target,
        reduction="none",
        pos_weight=pos_weight,
    )
    return bce.mean()


def residual_need_gate_supervision_loss(
    generated: dict[str, torch.Tensor],
    target: torch.Tensor,
    target_regions: torch.Tensor,
    dilation: int,
    error_threshold: float,
    max_pos_weight: float = 20.0,
) -> torch.Tensor:
    if "refinement_gate_logits" not in generated or "stage1_synthetic" not in generated:
        return target.new_zeros(())
    logits = generated["refinement_gate_logits"]
    stage1 = generated["stage1_synthetic"].detach()
    lesion = lesion_refinement_target(target_regions, dilation).to(
        device=target.device,
        dtype=target.dtype,
    )
    threshold = max(float(error_threshold), 1e-6)
    residual_need = (target - stage1).abs().detach()
    target_gate = (residual_need / threshold).clamp(0.0, 1.0) * lesion
    positive_fraction = target_gate.detach().mean().clamp_min(1e-4)
    pos_weight = ((1.0 - positive_fraction) / positive_fraction).clamp(
        min=1.0,
        max=float(max_pos_weight),
    )
    loss = F.binary_cross_entropy_with_logits(
        logits,
        target_gate,
        reduction="none",
        pos_weight=pos_weight,
    )
    return loss.mean()


def refinement_gate_sparsity_loss(
    generated: dict[str, torch.Tensor],
    target_regions: torch.Tensor,
    dilation: int,
) -> torch.Tensor:
    if "refinement_gate" not in generated:
        return target_regions.new_zeros(())
    gate = generated["refinement_gate"]
    target = lesion_refinement_target(target_regions, dilation).to(gate.dtype)
    background = (1.0 - target).clamp(0.0, 1.0)
    if not (background > 0.0).any():
        return gate.new_zeros(())
    return weighted_mean(gate, background.clamp_min(1e-4))


def background_preservation_loss(
    generated: dict[str, torch.Tensor],
    target_regions: torch.Tensor,
    dilation: int,
) -> torch.Tensor:
    if "stage1_synthetic" not in generated:
        return target_regions.new_zeros(())
    target = lesion_refinement_target(target_regions, dilation).to(generated["synthetic"].dtype)
    background = (1.0 - target).clamp(0.0, 1.0)
    if not (background > 0.0).any():
        return generated["synthetic"].new_zeros(())
    return weighted_mean(
        (generated["synthetic"] - generated["stage1_synthetic"].detach()).abs(),
        background.clamp_min(1e-4),
    )


def core_overfill_loss(
    synthetic: torch.Tensor,
    target: torch.Tensor,
    target_regions: torch.Tensor,
) -> torch.Tensor:
    tc = target_regions[:, 1:2].float()
    et = target_regions[:, 0:1].float()
    core_without_enhancement = (tc > 0.5) & (et <= 0.5)
    if not core_without_enhancement.any():
        return synthetic.new_zeros(())
    over = F.relu(synthetic - target)
    return weighted_mean(over, core_without_enhancement.float())


def refinement_residual_target_loss(
    generated: dict[str, torch.Tensor],
    target: torch.Tensor,
    target_regions: torch.Tensor,
    dilation: int,
    max_abs_residual: float,
) -> torch.Tensor:
    if "stage1_synthetic" not in generated:
        return target.new_zeros(())
    stage1 = generated["stage1_synthetic"].detach()
    applied_residual = generated["synthetic"] - stage1
    residual_target = (target - stage1).clamp(
        min=-float(max_abs_residual),
        max=float(max_abs_residual),
    )
    weights = lesion_refinement_target(target_regions, dilation).to(
        device=target.device,
        dtype=target.dtype,
    )
    if not (weights > 0.0).any():
        return target.new_zeros(())
    loss = F.smooth_l1_loss(
        applied_residual,
        residual_target.detach(),
        beta=0.03,
        reduction="none",
    )
    return weighted_mean(loss, weights.clamp_min(1e-4))


def refinement_direction_loss(
    generated: dict[str, torch.Tensor],
    target: torch.Tensor,
    target_regions: torch.Tensor,
    dilation: int,
) -> torch.Tensor:
    if "stage1_synthetic" not in generated:
        return target.new_zeros(())
    needed = (target - generated["stage1_synthetic"].detach()).detach()
    applied_residual = generated["synthetic"] - generated["stage1_synthetic"].detach()
    weights = lesion_refinement_target(target_regions, dilation).to(
        device=target.device,
        dtype=target.dtype,
    )
    weights = weights * needed.abs().clamp_min(1e-4)
    if not (weights > 0.0).any():
        return target.new_zeros(())
    wrong_direction = F.relu(-(applied_residual * needed))
    return weighted_mean(wrong_direction, weights)


def refinement_residual_budget_loss(
    generated: dict[str, torch.Tensor],
    target: torch.Tensor,
    target_regions: torch.Tensor,
    dilation: int,
    region: str,
) -> torch.Tensor:
    if "stage1_synthetic" not in generated:
        return target.new_zeros(())
    stage1 = generated["stage1_synthetic"].detach()
    applied = (generated["synthetic"] - stage1).abs()
    budget = (target - stage1).abs().detach()
    overshoot = F.relu(applied - budget)
    lesion = lesion_refinement_target(target_regions, dilation).to(
        device=target.device,
        dtype=target.dtype,
    )
    if region == "all":
        weights = torch.ones_like(overshoot)
    elif region == "lesion":
        weights = lesion
    elif region == "background":
        weights = (1.0 - lesion).clamp(0.0, 1.0)
    else:
        raise ValueError(f"Unknown residual-budget region: {region}")
    if not (weights > 0.0).any():
        return target.new_zeros(())
    return weighted_mean(overshoot, weights.clamp_min(1e-4))


def refinement_no_harm_loss(
    generated: dict[str, torch.Tensor],
    target: torch.Tensor,
    target_regions: torch.Tensor,
    dilation: int,
    margin: float,
    region: str,
) -> torch.Tensor:
    if "stage1_synthetic" not in generated:
        return target.new_zeros(())
    synthetic = generated["synthetic"]
    stage1 = generated["stage1_synthetic"].detach()
    final_error = (synthetic - target).abs()
    stage1_error = (stage1 - target).abs()
    harm = F.relu(final_error - stage1_error + float(margin))
    lesion = lesion_refinement_target(target_regions, dilation).to(
        device=target.device,
        dtype=target.dtype,
    )
    if region == "all":
        weights = torch.ones_like(harm)
    elif region == "lesion":
        weights = lesion
    elif region == "background":
        weights = (1.0 - lesion).clamp(0.0, 1.0)
    else:
        raise ValueError(f"Unknown no-harm region: {region}")
    if not (weights > 0.0).any():
        return target.new_zeros(())
    return weighted_mean(harm, weights.clamp_min(1e-4))


def refinement_acceptance_supervision_loss(
    generated: dict[str, torch.Tensor],
    target: torch.Tensor,
    target_regions: torch.Tensor,
    dilation: int,
    error_threshold: float,
    max_pos_weight: float = 20.0,
) -> torch.Tensor:
    if "refinement_acceptance_logits" not in generated or "stage1_synthetic" not in generated:
        return target.new_zeros(())
    logits = generated["refinement_acceptance_logits"]
    stage1_error = (generated["stage1_synthetic"].detach() - target).abs()
    lesion = lesion_refinement_target(target_regions, dilation).to(
        device=target.device,
        dtype=target.dtype,
    )
    threshold = max(float(error_threshold), 1e-6)
    target_acceptance = ((stage1_error - threshold) / threshold).clamp(0.0, 1.0) * lesion
    positive_fraction = target_acceptance.detach().mean().clamp_min(1e-4)
    pos_weight = ((1.0 - positive_fraction) / positive_fraction).clamp(
        min=1.0,
        max=float(max_pos_weight),
    )
    loss = F.binary_cross_entropy_with_logits(
        logits,
        target_acceptance.detach(),
        reduction="none",
        pos_weight=pos_weight,
    )
    return loss.mean()


def lesion_residual_target_loss(
    generated: dict[str, torch.Tensor],
    target: torch.Tensor,
    target_regions: torch.Tensor,
) -> torch.Tensor:
    if "lesion_residual" not in generated or "base_logits" not in generated:
        return target.new_zeros(())
    et = target_regions[:, 0:1].float()
    tc = target_regions[:, 1:2].float()
    lesion = (3.0 * et + tc).clamp(0.0, 4.0)
    if not (lesion > 0.5).any():
        return target.new_zeros(())
    needed_boost = (target - generated["base_logits"].detach()).clamp_min(0.0)
    predicted_boost = generated["lesion_residual"].clamp_min(0.0)
    loss = F.smooth_l1_loss(
        predicted_boost,
        needed_boost,
        reduction="none",
        beta=0.08,
    )
    return weighted_mean(loss, lesion.clamp_min(1e-4))


def enhancement_residual_target_loss(
    generated: dict[str, torch.Tensor],
    target: torch.Tensor,
    target_regions: torch.Tensor,
    quantile: float,
) -> torch.Tensor:
    if (
        "enhancement_residual" not in generated
        or "pre_enhancement_synthetic" not in generated
    ):
        return target.new_zeros(())
    et = target_regions[:, 0:1].float()
    tc = target_regions[:, 1:2].float()
    wt = target_regions[:, 2:3].float()
    tumor = wt > 0.5
    if not tumor.any():
        return target.new_zeros(())
    threshold = torch.quantile(target[tumor].detach().float(), float(quantile))
    core = tc > 0.5
    high_target = (core & (target >= threshold)) | (et > 0.5)
    if not high_target.any():
        return target.new_zeros(())
    needed = (target - generated["pre_enhancement_synthetic"].detach()).clamp_min(0.0)
    predicted = generated["enhancement_residual"]
    loss = F.smooth_l1_loss(
        predicted,
        needed,
        reduction="none",
        beta=0.05,
    )
    weight = (1.0 + 5.0 * et + 0.5 * tc) * high_target.float()
    return weighted_mean(loss, weight.clamp_min(1e-4))


def enhancement_leak_loss(
    generated: dict[str, torch.Tensor],
    target_regions: torch.Tensor,
) -> torch.Tensor:
    if "enhancement_residual" not in generated:
        return target_regions.new_zeros(())
    wt = target_regions[:, 2:3].float()
    outside = wt <= 0.5
    if not outside.any():
        return target_regions.new_zeros(())
    leak = F.relu(generated["enhancement_residual"])
    return weighted_mean(leak, outside.float())


def lesion_texture_loss(
    synthetic: torch.Tensor,
    target: torch.Tensor,
    target_regions: torch.Tensor,
    dilation: int,
) -> torch.Tensor:
    et = target_regions[:, 0:1].float()
    tc = target_regions[:, 1:2].float()
    wt = target_regions[:, 2:3].float()
    lesion = (4.0 * et + 1.5 * tc + 0.25 * wt).clamp_min(0.0)
    if dilation > 0:
        kernel = int(dilation) * 2 + 1
        lesion = F.max_pool2d(lesion, kernel_size=kernel, stride=1, padding=int(dilation))
    if not (lesion > 0.0).any():
        return synthetic.new_zeros(())
    pred_features = conv_medical_features_2d(synthetic)
    target_features = conv_medical_features_2d(target).detach()
    feature_error = F.smooth_l1_loss(
        pred_features,
        target_features,
        reduction="none",
        beta=0.05,
    ).mean(dim=1, keepdim=True)
    return weighted_mean(feature_error, lesion.clamp_min(1e-4))


def top_intensity_recall_loss(
    synthetic: torch.Tensor,
    target: torch.Tensor,
    target_regions: torch.Tensor,
    quantile: float,
) -> torch.Tensor:
    tumor = target_regions[:, 2:3] > 0.5
    if not tumor.any():
        return synthetic.new_zeros(())
    target_values = target[tumor].detach()
    threshold = torch.quantile(target_values.float(), float(quantile))
    top_mask = tumor & (target >= threshold)
    if not top_mask.any():
        return synthetic.new_zeros(())
    under = F.relu(target - synthetic)
    return weighted_mean(under + 0.25 * (synthetic - target).abs(), top_mask.float())


def prompt_balanced_bce_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    max_pos_weight: float,
) -> torch.Tensor:
    dims = tuple(range(2, target.ndim))
    positives = target.sum(dim=dims)
    total = float(np.prod(target.shape[2:]))
    negatives = torch.clamp(torch.as_tensor(total, device=target.device, dtype=target.dtype) - positives, min=1.0)
    pos_weight = (negatives / positives.clamp_min(1.0)).clamp(1.0, float(max_pos_weight))
    view_shape = (1, -1) + (1,) * (target.ndim - 2)
    loss = F.binary_cross_entropy_with_logits(
        logits,
        target,
        pos_weight=pos_weight.mean(dim=0).view(view_shape),
        reduction="none",
    )
    region_weights = torch.tensor(
        [4.0, 2.5, 1.0],
        dtype=loss.dtype,
        device=loss.device,
    ).view(view_shape)
    return (loss * region_weights).mean()


def patch_tokens_2d(slice_tensor: torch.Tensor, patch_size: int) -> torch.Tensor:
    if slice_tensor.ndim != 4:
        raise ValueError(f"Expected [B,C,H,W], got {tuple(slice_tensor.shape)}")
    spatial = tuple(int(item) for item in slice_tensor.shape[-2:])
    kernel = min(int(patch_size), *spatial)
    if kernel <= 1:
        mean = slice_tensor
        std = torch.zeros_like(slice_tensor)
        vmin = slice_tensor
        vmax = slice_tensor
    else:
        mean = F.avg_pool2d(slice_tensor, kernel_size=kernel, stride=kernel)
        second = F.avg_pool2d(slice_tensor.square(), kernel_size=kernel, stride=kernel)
        std = (second - mean.square()).clamp_min(1e-6).sqrt()
        vmax = F.max_pool2d(slice_tensor, kernel_size=kernel, stride=kernel)
        vmin = -F.max_pool2d(-slice_tensor, kernel_size=kernel, stride=kernel)
    return torch.cat([mean, std, vmax - vmin], dim=1).flatten(2).transpose(1, 2).contiguous()


def patch_weights_2d(weight: torch.Tensor, patch_size: int) -> torch.Tensor:
    spatial = tuple(int(item) for item in weight.shape[-2:])
    kernel = min(int(patch_size), *spatial)
    pooled = (
        weight
        if kernel <= 1
        else F.avg_pool2d(weight, kernel_size=kernel, stride=kernel)
    )
    tokens = pooled.flatten(2).squeeze(1)
    return (tokens / tokens.mean(dim=1, keepdim=True).clamp_min(1e-8)).clamp(0.25, 4.0)


def slice_drift_loss(
    synthetic: torch.Tensor,
    target: torch.Tensor,
    focus: torch.Tensor,
    patch_size: int,
    radii: Sequence[float],
) -> torch.Tensor:
    tokens = patch_tokens_2d(synthetic, patch_size)
    positive = patch_tokens_2d(target, patch_size)
    weights = patch_weights_2d(focus, patch_size).to(tokens.dtype)
    loss, _ = drift_loss(
        generated=tokens,
        positive=positive,
        weight_generated=weights,
        weight_positive=weights,
        radii=radii,
    )
    return loss


def conv_medical_features_2d(image: torch.Tensor) -> torch.Tensor:
    dtype = image.dtype
    device = image.device
    sobel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        dtype=dtype,
        device=device,
    ).view(1, 1, 3, 3) / 8.0
    sobel_y = sobel_x.transpose(2, 3)
    laplace = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        dtype=dtype,
        device=device,
    ).view(1, 1, 3, 3) / 4.0
    grad_x = F.conv2d(image, sobel_x, padding=1)
    grad_y = F.conv2d(image, sobel_y, padding=1)
    grad_mag = (grad_x.square() + grad_y.square() + 1e-8).sqrt()
    lap = F.conv2d(image, laplace, padding=1)
    local_mean = F.avg_pool2d(image, kernel_size=5, stride=1, padding=2)
    local_second = F.avg_pool2d(image.square(), kernel_size=5, stride=1, padding=2)
    local_std = (local_second - local_mean.square()).clamp_min(1e-6).sqrt()
    return torch.cat([image, grad_x, grad_y, grad_mag, lap, local_mean, local_std], dim=1)


def window_tokens_2d(features: torch.Tensor, window_size: int, stride: int) -> torch.Tensor:
    kernel = max(1, min(int(window_size), *features.shape[-2:]))
    stride = max(1, int(stride))
    padding = kernel // 2
    tokens = F.unfold(features, kernel_size=kernel, stride=stride, padding=padding)
    return tokens.transpose(1, 2).contiguous()


def normalize_window_tokens(tokens: torch.Tensor, clamp: float) -> torch.Tensor:
    stats_source = tokens.detach()
    mean = stats_source.mean(dim=(1, 2), keepdim=True)
    std = stats_source.std(dim=(1, 2), keepdim=True, unbiased=False).clamp_min(1e-3)
    normalized = (tokens - mean) / std
    return torch.nan_to_num(normalized, nan=0.0, posinf=float(clamp), neginf=-float(clamp)).clamp(
        -float(clamp),
        float(clamp),
    )


def normalize_token_triplet(
    generated: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor | None,
    clamp: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    pieces = [generated.detach(), positive.detach()]
    if negative is not None and negative.numel() > 0:
        pieces.append(negative.detach())
    stats_source = torch.cat(pieces, dim=1)
    mean = stats_source.mean(dim=(1, 2), keepdim=True)
    std = stats_source.std(dim=(1, 2), keepdim=True, unbiased=False).clamp_min(1e-3)

    def apply(tokens: torch.Tensor | None) -> torch.Tensor | None:
        if tokens is None:
            return None
        normalized = (tokens - mean) / std
        return torch.nan_to_num(
            normalized,
            nan=0.0,
            posinf=float(clamp),
            neginf=-float(clamp),
        ).clamp(-float(clamp), float(clamp))

    return apply(generated), apply(positive), apply(negative)


def window_weights_2d(focus: torch.Tensor, window_size: int, stride: int) -> torch.Tensor:
    kernel = max(1, min(int(window_size), *focus.shape[-2:]))
    stride = max(1, int(stride))
    padding = kernel // 2
    pooled = F.unfold(focus, kernel_size=kernel, stride=stride, padding=padding)
    weights = pooled.mean(dim=1)
    return (weights / weights.mean(dim=1, keepdim=True).clamp_min(1e-8)).clamp(0.25, 6.0)


def window_drift_loss(
    synthetic: torch.Tensor,
    target: torch.Tensor,
    focus: torch.Tensor,
    window_size: int,
    stride: int,
    radii: Sequence[float],
    token_clamp: float,
) -> torch.Tensor:
    generated_tokens = window_tokens_2d(
        conv_medical_features_2d(synthetic),
        window_size,
        stride,
    )
    positive_tokens = window_tokens_2d(
        conv_medical_features_2d(target),
        window_size,
        stride,
    )
    generated_tokens = normalize_window_tokens(generated_tokens, token_clamp)
    positive_tokens = normalize_window_tokens(positive_tokens, token_clamp)
    weights = window_weights_2d(focus, window_size, stride).to(generated_tokens.dtype)
    loss, _ = drift_loss(
        generated=generated_tokens,
        positive=positive_tokens,
        weight_generated=weights,
        weight_positive=weights,
        radii=radii,
    )
    return loss


def window_feature_alignment_loss(
    synthetic: torch.Tensor,
    target: torch.Tensor,
    focus: torch.Tensor,
    window_size: int,
    stride: int,
    token_clamp: float,
) -> torch.Tensor:
    generated_tokens = normalize_window_tokens(
        window_tokens_2d(conv_medical_features_2d(synthetic), window_size, stride),
        token_clamp,
    )
    positive_tokens = normalize_window_tokens(
        window_tokens_2d(conv_medical_features_2d(target), window_size, stride),
        token_clamp,
    )
    weights = window_weights_2d(focus, window_size, stride).to(generated_tokens.dtype)
    feature_error = F.smooth_l1_loss(
        generated_tokens,
        positive_tokens.detach(),
        reduction="none",
        beta=0.25,
    ).mean(dim=2)
    return weighted_mean(feature_error.unsqueeze(1), weights.unsqueeze(1))


def micro_window_feature_alignment_loss(
    synthetic: torch.Tensor,
    target: torch.Tensor,
    focus: torch.Tensor,
    window_sizes: Sequence[int],
    stride: int,
    token_clamp: float,
) -> torch.Tensor:
    synthetic_features = conv_medical_features_2d(synthetic)
    target_features = conv_medical_features_2d(target)
    losses = []
    for raw_size in window_sizes:
        size = max(1, int(raw_size))
        generated_tokens = normalize_window_tokens(
            window_tokens_2d(synthetic_features, size, stride),
            token_clamp,
        )
        positive_tokens = normalize_window_tokens(
            window_tokens_2d(target_features, size, stride),
            token_clamp,
        )
        weights = window_weights_2d(focus, size, stride).to(generated_tokens.dtype)
        feature_error = F.smooth_l1_loss(
            generated_tokens,
            positive_tokens.detach(),
            reduction="none",
            beta=0.18,
        ).mean(dim=2)
        losses.append(weighted_mean(feature_error.unsqueeze(1), weights.unsqueeze(1)))
    if not losses:
        return synthetic.new_zeros(())
    return torch.stack(losses).mean()


def micro_window_drift_alignment_loss(
    synthetic: torch.Tensor,
    target: torch.Tensor,
    focus: torch.Tensor,
    window_sizes: Sequence[int],
    stride: int,
    token_clamp: float,
    radii: Sequence[float],
    max_tokens: int,
) -> torch.Tensor:
    synthetic_features = conv_medical_features_2d(synthetic)
    target_features = conv_medical_features_2d(target)
    losses = []
    for raw_size in window_sizes:
        size = max(1, int(raw_size))
        generated_tokens = window_tokens_2d(synthetic_features, size, stride)
        positive_tokens = window_tokens_2d(target_features, size, stride)
        generated_tokens, positive_tokens, _ = normalize_token_triplet(
            generated_tokens,
            positive_tokens,
            None,
            token_clamp,
        )
        weights = window_weights_2d(focus, size, stride).to(generated_tokens.dtype)
        generated_tokens, positive_tokens, weights = subsample_pair_tokens_by_weight(
            generated_tokens,
            positive_tokens,
            weights,
            max_tokens,
        )
        loss, _ = drift_loss(
            generated=generated_tokens,
            positive=positive_tokens,
            weight_generated=weights,
            weight_positive=weights,
            radii=radii,
        )
        losses.append(loss)
    if not losses:
        return synthetic.new_zeros(())
    return torch.stack(losses).mean()


def compact_window_stats_tokens_2d(
    features: torch.Tensor,
    window_size: int,
    stride: int,
) -> torch.Tensor:
    kernel = max(1, min(int(window_size), *features.shape[-2:]))
    stride = max(1, int(stride))
    padding = kernel // 2
    patches = F.unfold(features, kernel_size=kernel, stride=stride, padding=padding)
    b, channels_times_area, num_windows = patches.shape
    area = kernel * kernel
    patches = patches.view(b, features.shape[1], area, num_windows)
    mean = patches.mean(dim=2)
    second = patches.square().mean(dim=2)
    std = (second - mean.square()).clamp_min(1e-6).sqrt()
    return torch.cat([mean, std], dim=1).transpose(1, 2).contiguous()


def weighted_global_stats_tokens_2d(
    features: torch.Tensor,
    focus: torch.Tensor,
) -> torch.Tensor:
    weight = focus.to(device=features.device, dtype=features.dtype).clamp_min(1e-4)
    denom = weight.sum(dim=(2, 3), keepdim=True).clamp_min(1e-6)
    mean = (features * weight).sum(dim=(2, 3), keepdim=True) / denom
    second = (features.square() * weight).sum(dim=(2, 3), keepdim=True) / denom
    std = (second - mean.square()).clamp_min(1e-6).sqrt()
    return torch.cat([mean.flatten(1), std.flatten(1)], dim=1).unsqueeze(1)


def medical_multilevel_descriptors_2d(
    image: torch.Tensor,
    focus: torch.Tensor,
    window_sizes: Sequence[int],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    features = conv_medical_features_2d(image)
    descriptors: dict[str, torch.Tensor] = {}
    weights: dict[str, torch.Tensor] = {}

    energy = features.square().mean(dim=(2, 3)).unsqueeze(1)
    descriptors["energy"] = energy
    weights["energy"] = image.new_ones((image.shape[0], 1))

    descriptors["global"] = weighted_global_stats_tokens_2d(features, focus)
    weights["global"] = image.new_ones((image.shape[0], 1))

    for raw_size in window_sizes:
        size = max(1, int(raw_size))
        stride = max(1, size // 2)
        name = f"spatial{size}"
        descriptors[name] = compact_window_stats_tokens_2d(features, size, stride)
        weights[name] = window_weights_2d(focus, size, stride).to(image.dtype)
    return descriptors, weights


def _gather_token_rows(tokens: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    feature_dim = tokens.shape[-1]
    expanded = indices.unsqueeze(-1).expand(-1, -1, feature_dim)
    return torch.gather(tokens, dim=1, index=expanded)


def subsample_pair_tokens_by_weight(
    generated: torch.Tensor,
    positive: torch.Tensor,
    weights: torch.Tensor,
    max_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_tokens = int(max(1, max_tokens))
    if generated.shape[1] <= max_tokens:
        return generated, positive, weights
    batch_size, token_count = weights.shape
    top_count = max(1, max_tokens // 2)
    random_count = max_tokens - top_count
    top_indices = torch.topk(weights, k=min(top_count, token_count), dim=1).indices
    if random_count > 0:
        probs = weights.clamp_min(1e-6)
        probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-6)
        random_indices = torch.multinomial(
            probs,
            num_samples=random_count,
            replacement=token_count < random_count,
        )
        indices = torch.cat([top_indices, random_indices], dim=1)
    else:
        indices = top_indices
    if indices.shape[1] < max_tokens:
        pad = indices[:, -1:].expand(batch_size, max_tokens - indices.shape[1])
        indices = torch.cat([indices, pad], dim=1)
    return (
        _gather_token_rows(generated, indices),
        _gather_token_rows(positive, indices),
        torch.gather(weights, dim=1, index=indices),
    )


class _TokenStore:
    def __init__(self, max_tokens: int, max_add_tokens: int) -> None:
        self.max_tokens = int(max(0, max_tokens))
        self.max_add_tokens = int(max(1, max_add_tokens))
        self._storage: dict[str, torch.Tensor] = {}

    def __len__(self) -> int:
        return sum(int(tokens.shape[0]) for tokens in self._storage.values())

    def add(self, key: str, tokens: torch.Tensor) -> None:
        if self.max_tokens <= 0:
            return
        flat = tokens.detach().flatten(0, 1).float()
        finite = torch.isfinite(flat).all(dim=1)
        flat = flat[finite]
        if flat.numel() == 0:
            return
        if flat.shape[0] > self.max_add_tokens:
            choice = torch.randperm(flat.shape[0], device=flat.device)[: self.max_add_tokens]
            flat = flat[choice]
        flat = flat.cpu()
        previous = self._storage.get(key)
        combined = flat if previous is None else torch.cat([previous, flat], dim=0)
        if combined.shape[0] > self.max_tokens:
            combined = combined[-self.max_tokens :]
        self._storage[key] = combined

    def sample(
        self,
        key: str,
        batch_size: int,
        count: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        count = int(max(0, count))
        tokens = self._storage.get(key)
        if count <= 0 or tokens is None or tokens.numel() == 0:
            return None
        indices = torch.randint(0, tokens.shape[0], (batch_size, count))
        return tokens[indices].to(device=device, dtype=dtype)

    def count(self, key: str) -> int:
        tokens = self._storage.get(key)
        return 0 if tokens is None else int(tokens.shape[0])


class MedicalDriftBank2D:
    def __init__(
        self,
        max_tokens: int,
        max_add_tokens: int,
        memory_tokens: int,
    ) -> None:
        self.positive = _TokenStore(max_tokens, max_add_tokens)
        self.negative = _TokenStore(max_tokens, max_add_tokens)
        self.memory_tokens = int(max(0, memory_tokens))

    @staticmethod
    def _key(target_modality: str, feature_name: str) -> str:
        return f"{target_modality}:{feature_name}"

    def sample_positive(
        self,
        target_modality: str,
        feature_name: str,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        return self.positive.sample(
            self._key(target_modality, feature_name),
            batch_size,
            self.memory_tokens,
            device,
            dtype,
        )

    def sample_negative(
        self,
        target_modality: str,
        feature_name: str,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        return self.negative.sample(
            self._key(target_modality, feature_name),
            batch_size,
            self.memory_tokens,
            device,
            dtype,
        )

    def update(
        self,
        target_modality: str,
        positive_descriptors: dict[str, torch.Tensor],
        negative_descriptors: dict[str, torch.Tensor],
    ) -> None:
        for name, tokens in positive_descriptors.items():
            self.positive.add(self._key(target_modality, name), tokens)
        for name, tokens in negative_descriptors.items():
            self.negative.add(self._key(target_modality, name), tokens)

    def counts(self, target_modality: str, feature_names: Sequence[str]) -> dict[str, int]:
        return {
            f"bank_pos_{name}": self.positive.count(self._key(target_modality, name))
            for name in feature_names
        } | {
            f"bank_neg_{name}": self.negative.count(self._key(target_modality, name))
            for name in feature_names
        }


def medical_multilevel_drift_loss(
    synthetic: torch.Tensor,
    target: torch.Tensor,
    focus: torch.Tensor,
    target_modality: str,
    bank: MedicalDriftBank2D,
    window_sizes: Sequence[int],
    radii: Sequence[float],
    token_clamp: float,
    max_current_tokens: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    generated_desc, generated_weights = medical_multilevel_descriptors_2d(
        synthetic,
        focus,
        window_sizes,
    )
    positive_desc, positive_weights = medical_multilevel_descriptors_2d(
        target,
        focus,
        window_sizes,
    )
    losses = []
    report: dict[str, float] = {}
    feature_names = list(generated_desc)
    for name in feature_names:
        generated_tokens, positive_tokens, generated_weight = subsample_pair_tokens_by_weight(
            generated_desc[name],
            positive_desc[name].detach(),
            generated_weights[name],
            max_current_tokens,
        )
        batch_size = generated_tokens.shape[0]
        positive_memory = bank.sample_positive(
            target_modality,
            name,
            batch_size,
            generated_tokens.device,
            generated_tokens.dtype,
        )
        negative_memory = bank.sample_negative(
            target_modality,
            name,
            batch_size,
            generated_tokens.device,
            generated_tokens.dtype,
        )

        positive_all = (
            positive_tokens
            if positive_memory is None
            else torch.cat([positive_tokens, positive_memory.detach()], dim=1)
        )
        positive_weight = generated_weight.to(generated_tokens.dtype)
        if positive_memory is not None:
            positive_weight = torch.cat(
                [
                    positive_weight,
                    torch.ones(
                        batch_size,
                        positive_memory.shape[1],
                        device=generated_tokens.device,
                        dtype=generated_tokens.dtype,
                    ),
                ],
                dim=1,
            )

        generated_norm, positive_norm, negative_norm = normalize_token_triplet(
            generated_tokens,
            positive_all,
            negative_memory,
            token_clamp,
        )
        loss, _ = drift_loss(
            generated=generated_norm,
            positive=positive_norm,
            negative=negative_norm,
            weight_generated=generated_weight.to(generated_tokens.dtype),
            weight_positive=positive_weight,
            weight_negative=(
                None
                if negative_norm is None
                else torch.ones(
                    batch_size,
                    negative_norm.shape[1],
                    device=generated_tokens.device,
                    dtype=generated_tokens.dtype,
                )
            ),
            radii=radii,
        )
        losses.append(loss)
        report[f"medical_drift_{name}"] = float(loss.detach().cpu().item())

    bank.update(target_modality, positive_desc, generated_desc)
    report.update(bank.counts(target_modality, feature_names))
    if not losses:
        return synthetic.new_zeros(()), report
    total = torch.stack(losses).mean()
    report["medical_drift_total"] = float(total.detach().cpu().item())
    return total, report


def loss_for_batch(
    model: SliceVirtualModalityGenerator | PromptedSliceVirtualModalityGenerator | SliceDriftTransportGenerator,
    batch: dict[str, Any],
    args: argparse.Namespace,
    medical_bank: MedicalDriftBank2D | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    image = batch["image"]
    target_index = int(batch["target_index"][0].item())
    target = image[:, target_index : target_index + 1]
    generated = generator_forward(model, image, batch["observed_mask"], batch)
    synthetic = generated["synthetic"]
    uncertainty = generated["uncertainty"]
    focus = region_focus(
        batch["target_regions"],
        args.focus_base,
        args.focus_et,
        args.focus_tc,
        args.focus_wt,
        args.focus_dilation,
    )
    support = observed_brain_support(
        image,
        batch["observed_mask"],
        args.support_threshold,
        args.focus_dilation,
    )
    focus = focus * support + float(args.background_weight) * (1.0 - support)
    abs_error = (synthetic - target).abs()
    recon = weighted_mean(abs_error, focus)
    nll = weighted_mean(abs_error / uncertainty + uncertainty.log(), focus)
    grad = (
        gradient_difference_loss(synthetic, target, focus)
        if float(args.gradient_weight) > 0.0
        else synthetic.new_zeros(())
    )
    lesion_focus = lesion_refinement_target(batch["target_regions"], args.gate_target_dilation).to(
        device=synthetic.device,
        dtype=synthetic.dtype,
    )
    multiscale_ssim = (
        multiscale_ssim_loss(synthetic, target, focus, args.fidelity_pyramid_levels)
        if float(args.multiscale_ssim_weight) > 0.0
        else synthetic.new_zeros(())
    )
    lesion_multiscale_ssim = (
        multiscale_ssim_loss(
            synthetic,
            target,
            lesion_focus.clamp_min(1e-4),
            args.fidelity_pyramid_levels,
        )
        if float(args.lesion_multiscale_ssim_weight) > 0.0 and (lesion_focus > 0.0).any()
        else synthetic.new_zeros(())
    )
    laplacian_pyramid = (
        laplacian_pyramid_fidelity_loss(synthetic, target, focus, args.fidelity_pyramid_levels)
        if float(args.laplacian_pyramid_weight) > 0.0
        else synthetic.new_zeros(())
    )
    lesion_laplacian_pyramid = (
        laplacian_pyramid_fidelity_loss(
            synthetic,
            target,
            lesion_focus.clamp_min(1e-4),
            args.fidelity_pyramid_levels,
        )
        if float(args.lesion_laplacian_pyramid_weight) > 0.0 and (lesion_focus > 0.0).any()
        else synthetic.new_zeros(())
    )
    drift = (
        slice_drift_loss(
            synthetic,
            target,
            focus,
            args.drift_patch_size,
            args.drift_radii,
        )
        if float(args.drift_weight) > 0.0
        else synthetic.new_zeros(())
    )
    window_drift = (
        window_drift_loss(
            synthetic,
            target,
            focus,
            args.window_size,
            args.window_stride,
            args.window_drift_radii,
            args.window_token_clamp,
        )
        if float(args.window_drift_weight) > 0.0
        else synthetic.new_zeros(())
    )
    window_feature = (
        window_feature_alignment_loss(
            synthetic,
            target,
            focus,
            args.window_size,
            args.window_stride,
            args.window_token_clamp,
        )
        if float(args.window_feature_weight) > 0.0
        else synthetic.new_zeros(())
    )
    micro_window_feature = (
        micro_window_feature_alignment_loss(
            synthetic,
            target,
            lesion_focus.clamp_min(1e-4),
            args.micro_window_sizes,
            args.micro_window_stride,
            args.window_token_clamp,
        )
        if float(args.micro_window_feature_weight) > 0.0 and (lesion_focus > 0.0).any()
        else synthetic.new_zeros(())
    )
    micro_window_drift = (
        micro_window_drift_alignment_loss(
            synthetic,
            target,
            lesion_focus.clamp_min(1e-4),
            args.micro_window_sizes,
            args.micro_window_stride,
            args.window_token_clamp,
            args.micro_window_drift_radii,
            args.micro_window_drift_max_tokens,
        )
        if float(args.micro_window_drift_weight) > 0.0 and (lesion_focus > 0.0).any()
        else synthetic.new_zeros(())
    )
    medical_drift_report: dict[str, float] = {}
    medical_drift = synthetic.new_zeros(())
    if float(args.medical_drift_weight) > 0.0:
        if medical_bank is None:
            raise RuntimeError("medical_drift_weight > 0 requires a MedicalDriftBank2D")
        medical_drift, medical_drift_report = medical_multilevel_drift_loss(
            synthetic,
            target,
            focus,
            args.target_modality,
            medical_bank,
            args.medical_drift_window_sizes,
            args.medical_drift_radii,
            args.medical_drift_token_clamp,
            args.medical_drift_max_current_tokens,
        )
    transport_velocity = (
        transport_velocity_loss(
            generated,
            target,
            focus,
            args.transport_step_scale,
        )
        if float(args.transport_velocity_weight) > 0.0
        else synthetic.new_zeros(())
    )
    transport_path = (
        transport_path_loss(generated, target, focus)
        if float(args.transport_path_weight) > 0.0
        else synthetic.new_zeros(())
    )
    transport_monotonic = (
        transport_monotonic_loss(generated, target, focus)
        if float(args.transport_monotonic_weight) > 0.0
        else synthetic.new_zeros(())
    )
    lesion_texture = (
        lesion_texture_loss(synthetic, target, batch["target_regions"], args.focus_dilation)
        if float(args.lesion_texture_weight) > 0.0
        else synthetic.new_zeros(())
    )
    region_moment = (
        region_moment_loss(synthetic, target, batch["target_regions"])
        if float(args.region_moment_weight) > 0.0
        else synthetic.new_zeros(())
    )
    edge = (
        tumor_edge_loss(synthetic, target, batch["target_regions"], args.focus_dilation)
        if float(args.edge_weight) > 0.0
        else synthetic.new_zeros(())
    )
    enhance_under = (
        enhancement_under_loss(synthetic, target, batch["target_regions"])
        if float(args.enhancement_under_weight) > 0.0
        else synthetic.new_zeros(())
    )
    lesion_boundary = (
        lesion_boundary_band_loss(synthetic, target, batch["target_regions"], args.focus_dilation)
        if float(args.lesion_boundary_weight) > 0.0
        else synthetic.new_zeros(())
    )
    enhancement_contrast = (
        enhancement_contrast_loss(synthetic, target, batch["target_regions"], args.focus_dilation)
        if float(args.enhancement_contrast_weight) > 0.0
        else synthetic.new_zeros(())
    )
    gate_supervision = (
        lesion_gate_supervision_loss(
            generated,
            batch["target_regions"],
            args.gate_target_dilation,
        )
        if float(args.gate_supervision_weight) > 0.0
        else synthetic.new_zeros(())
    )
    residual_need_gate = (
        residual_need_gate_supervision_loss(
            generated,
            target,
            batch["target_regions"],
            args.gate_target_dilation,
            args.residual_need_gate_threshold,
        )
        if float(args.residual_need_gate_weight) > 0.0
        else synthetic.new_zeros(())
    )
    gate_sparsity = (
        refinement_gate_sparsity_loss(
            generated,
            batch["target_regions"],
            args.gate_target_dilation,
        )
        if float(args.gate_sparsity_weight) > 0.0
        else synthetic.new_zeros(())
    )
    background_preserve = (
        background_preservation_loss(
            generated,
            batch["target_regions"],
            args.gate_target_dilation,
        )
        if float(args.background_preserve_weight) > 0.0
        else synthetic.new_zeros(())
    )
    core_overfill = (
        core_overfill_loss(synthetic, target, batch["target_regions"])
        if float(args.core_overfill_weight) > 0.0
        else synthetic.new_zeros(())
    )
    acceptance_supervision = (
        refinement_acceptance_supervision_loss(
            generated,
            target,
            batch["target_regions"],
            args.gate_target_dilation,
            args.acceptance_error_threshold,
        )
        if float(args.acceptance_supervision_weight) > 0.0
        else synthetic.new_zeros(())
    )
    refinement_residual_target = (
        refinement_residual_target_loss(
            generated,
            target,
            batch["target_regions"],
            args.gate_target_dilation,
            args.refinement_residual_scale,
        )
        if float(args.refinement_residual_target_weight) > 0.0
        else synthetic.new_zeros(())
    )
    refinement_direction = (
        refinement_direction_loss(
            generated,
            target,
            batch["target_regions"],
            args.gate_target_dilation,
        )
        if float(args.refinement_direction_weight) > 0.0
        else synthetic.new_zeros(())
    )
    refinement_residual_budget = (
        refinement_residual_budget_loss(
            generated,
            target,
            batch["target_regions"],
            args.gate_target_dilation,
            "all",
        )
        if float(args.refinement_residual_budget_weight) > 0.0
        else synthetic.new_zeros(())
    )
    refinement_lesion_budget = (
        refinement_residual_budget_loss(
            generated,
            target,
            batch["target_regions"],
            args.gate_target_dilation,
            "lesion",
        )
        if float(args.refinement_lesion_budget_weight) > 0.0
        else synthetic.new_zeros(())
    )
    refinement_no_harm = (
        refinement_no_harm_loss(
            generated,
            target,
            batch["target_regions"],
            args.gate_target_dilation,
            args.refinement_no_harm_margin,
            "all",
        )
        if float(args.refinement_no_harm_weight) > 0.0
        else synthetic.new_zeros(())
    )
    refinement_lesion_no_harm = (
        refinement_no_harm_loss(
            generated,
            target,
            batch["target_regions"],
            args.gate_target_dilation,
            args.refinement_no_harm_margin,
            "lesion",
        )
        if float(args.refinement_lesion_no_harm_weight) > 0.0
        else synthetic.new_zeros(())
    )
    refinement_background_no_harm = (
        refinement_no_harm_loss(
            generated,
            target,
            batch["target_regions"],
            args.gate_target_dilation,
            args.refinement_no_harm_margin,
            "background",
        )
        if float(args.refinement_background_no_harm_weight) > 0.0
        else synthetic.new_zeros(())
    )
    lesion_residual_target = (
        lesion_residual_target_loss(generated, target, batch["target_regions"])
        if float(args.lesion_residual_target_weight) > 0.0
        else synthetic.new_zeros(())
    )
    enhancement_residual_target = (
        enhancement_residual_target_loss(
            generated,
            target,
            batch["target_regions"],
            args.top_intensity_quantile,
        )
        if float(args.enhancement_residual_target_weight) > 0.0
        else synthetic.new_zeros(())
    )
    enhancement_leak = (
        enhancement_leak_loss(generated, batch["target_regions"])
        if float(args.enhancement_leak_weight) > 0.0
        else synthetic.new_zeros(())
    )
    top_intensity = (
        top_intensity_recall_loss(
            synthetic,
            target,
            batch["target_regions"],
            args.top_intensity_quantile,
        )
        if float(args.top_intensity_weight) > 0.0
        else synthetic.new_zeros(())
    )
    prompt = (
        segmentation_loss(
            generated["prompt_logits"],
            batch["target_regions"],
            region_weights=(args.prompt_et_weight, args.prompt_tc_weight, args.prompt_wt_weight),
        )
        if "prompt_logits" in generated and float(args.prompt_weight) > 0.0
        else synthetic.new_zeros(())
    )
    prompt_bce = (
        prompt_balanced_bce_loss(
            generated["prompt_logits"],
            batch["target_regions"],
            args.prompt_max_pos_weight,
        )
        if "prompt_logits" in generated and float(args.prompt_balanced_bce_weight) > 0.0
        else synthetic.new_zeros(())
    )
    loss = (
        float(args.recon_weight) * recon
        + float(args.nll_weight) * nll
        + float(args.gradient_weight) * grad
        + float(args.multiscale_ssim_weight) * multiscale_ssim
        + float(args.lesion_multiscale_ssim_weight) * lesion_multiscale_ssim
        + float(args.laplacian_pyramid_weight) * laplacian_pyramid
        + float(args.lesion_laplacian_pyramid_weight) * lesion_laplacian_pyramid
        + float(args.drift_weight) * drift
        + float(args.window_drift_weight) * window_drift
        + float(args.window_feature_weight) * window_feature
        + float(args.micro_window_feature_weight) * micro_window_feature
        + float(args.micro_window_drift_weight) * micro_window_drift
        + float(args.medical_drift_weight) * medical_drift
        + float(args.transport_velocity_weight) * transport_velocity
        + float(args.transport_path_weight) * transport_path
        + float(args.transport_monotonic_weight) * transport_monotonic
        + float(args.lesion_texture_weight) * lesion_texture
        + float(args.region_moment_weight) * region_moment
        + float(args.edge_weight) * edge
        + float(args.enhancement_under_weight) * enhance_under
        + float(args.lesion_boundary_weight) * lesion_boundary
        + float(args.enhancement_contrast_weight) * enhancement_contrast
        + float(args.gate_supervision_weight) * gate_supervision
        + float(args.residual_need_gate_weight) * residual_need_gate
        + float(args.gate_sparsity_weight) * gate_sparsity
        + float(args.background_preserve_weight) * background_preserve
        + float(args.core_overfill_weight) * core_overfill
        + float(args.acceptance_supervision_weight) * acceptance_supervision
        + float(args.refinement_residual_target_weight) * refinement_residual_target
        + float(args.refinement_direction_weight) * refinement_direction
        + float(args.refinement_residual_budget_weight) * refinement_residual_budget
        + float(args.refinement_lesion_budget_weight) * refinement_lesion_budget
        + float(args.refinement_no_harm_weight) * refinement_no_harm
        + float(args.refinement_lesion_no_harm_weight) * refinement_lesion_no_harm
        + float(args.refinement_background_no_harm_weight) * refinement_background_no_harm
        + float(args.lesion_residual_target_weight) * lesion_residual_target
        + float(args.enhancement_residual_target_weight) * enhancement_residual_target
        + float(args.enhancement_leak_weight) * enhancement_leak
        + float(args.top_intensity_weight) * top_intensity
        + float(args.prompt_weight) * prompt
        + float(args.prompt_balanced_bce_weight) * prompt_bce
    )
    return loss, {
        "loss": float(loss.detach().cpu().item()),
        "recon_l1": float(recon.detach().cpu().item()),
        "nll": float(nll.detach().cpu().item()),
        "gradient_l1": float(grad.detach().cpu().item()),
        "multiscale_ssim": float(multiscale_ssim.detach().cpu().item()),
        "lesion_multiscale_ssim": float(lesion_multiscale_ssim.detach().cpu().item()),
        "laplacian_pyramid": float(laplacian_pyramid.detach().cpu().item()),
        "lesion_laplacian_pyramid": float(lesion_laplacian_pyramid.detach().cpu().item()),
        "drift_loss": float(drift.detach().cpu().item()),
        "window_drift_loss": float(window_drift.detach().cpu().item()),
        "window_feature_loss": float(window_feature.detach().cpu().item()),
        "micro_window_feature_loss": float(micro_window_feature.detach().cpu().item()),
        "micro_window_drift_loss": float(micro_window_drift.detach().cpu().item()),
        "medical_drift_loss": float(medical_drift.detach().cpu().item()),
        "transport_velocity_loss": float(transport_velocity.detach().cpu().item()),
        "transport_path_loss": float(transport_path.detach().cpu().item()),
        "transport_monotonic_loss": float(transport_monotonic.detach().cpu().item()),
        "lesion_texture": float(lesion_texture.detach().cpu().item()),
        "region_moment": float(region_moment.detach().cpu().item()),
        "tumor_edge": float(edge.detach().cpu().item()),
        "enhancement_under": float(enhance_under.detach().cpu().item()),
        "lesion_boundary": float(lesion_boundary.detach().cpu().item()),
        "enhancement_contrast": float(enhancement_contrast.detach().cpu().item()),
        "gate_supervision": float(gate_supervision.detach().cpu().item()),
        "residual_need_gate": float(residual_need_gate.detach().cpu().item()),
        "gate_sparsity": float(gate_sparsity.detach().cpu().item()),
        "background_preserve": float(background_preserve.detach().cpu().item()),
        "core_overfill": float(core_overfill.detach().cpu().item()),
        "acceptance_supervision": float(acceptance_supervision.detach().cpu().item()),
        "refinement_residual_target": float(refinement_residual_target.detach().cpu().item()),
        "refinement_direction": float(refinement_direction.detach().cpu().item()),
        "refinement_residual_budget": float(refinement_residual_budget.detach().cpu().item()),
        "refinement_lesion_budget": float(refinement_lesion_budget.detach().cpu().item()),
        "refinement_no_harm": float(refinement_no_harm.detach().cpu().item()),
        "refinement_lesion_no_harm": float(refinement_lesion_no_harm.detach().cpu().item()),
        "refinement_background_no_harm": float(
            refinement_background_no_harm.detach().cpu().item()
        ),
        "lesion_residual_target": float(lesion_residual_target.detach().cpu().item()),
        "enhancement_residual_target": float(enhancement_residual_target.detach().cpu().item()),
        "enhancement_leak": float(enhancement_leak.detach().cpu().item()),
        "top_intensity": float(top_intensity.detach().cpu().item()),
        "prompt_loss": float(prompt.detach().cpu().item()),
        "prompt_balanced_bce": float(prompt_bce.detach().cpu().item()),
        "uncertainty_mean": float(uncertainty.detach().mean().cpu().item()),
        "confidence_mean": float(generated["confidence"].detach().mean().cpu().item()),
        "stage1_mae": float(
            (generated.get("stage1_synthetic", synthetic).detach() - target.detach())
            .abs()
            .mean()
            .cpu()
            .item()
        ),
        "gate_mean": float(
            generated.get("refinement_gate", synthetic.new_zeros(()))
            .detach()
            .mean()
            .cpu()
            .item()
        ),
        "acceptance_mean": float(
            generated.get("refinement_acceptance", synthetic.new_zeros(()))
            .detach()
            .mean()
            .cpu()
            .item()
        ),
    } | medical_drift_report


def pearson_correlation(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.detach().flatten().float()
    y = y.detach().flatten().float()
    x = x - x.mean()
    y = y - y.mean()
    denom = x.square().mean().sqrt() * y.square().mean().sqrt()
    if denom.item() <= 1e-8:
        return float("nan")
    return float(((x * y).mean() / denom).cpu().item())


@torch.no_grad()
def evaluate(
    model: SliceVirtualModalityGenerator | PromptedSliceVirtualModalityGenerator | SliceDriftTransportGenerator,
    loader: DataLoader,
    device: str,
) -> dict[str, float]:
    model.eval()
    rows = []
    for batch in loader:
        batch = to_device(batch, device)
        image = batch["image"]
        target_index = int(batch["target_index"][0].item())
        target = image[:, target_index : target_index + 1]
        generated = generator_forward(model, image, batch["observed_mask"], batch)
        synthetic = generated["synthetic"]
        error = (synthetic - target).abs()
        synthetic_features = conv_medical_features_2d(synthetic)
        target_features = conv_medical_features_2d(target)
        edge_error = (synthetic_features[:, 3:4] - target_features[:, 3:4]).abs()
        laplacian_error = (synthetic_features[:, 4:5] - target_features[:, 4:5]).abs()
        ssim_map = ssim_map_2d(synthetic, target)
        tumor = batch["target_regions"][:, 2:3] > 0.5
        if tumor.any():
            high_threshold = torch.quantile(target[tumor].detach().float(), 0.70)
            high_tumor = tumor & (target >= high_threshold)
        else:
            high_tumor = torch.zeros_like(tumor)
        high_tumor_under = F.relu(target - synthetic)
        mse = (synthetic - target).square().mean().clamp_min(1e-8)
        psnr = 20.0 * math.log10(2.0) - 10.0 * math.log10(float(mse.cpu().item()))
        row = {
            "mae": float(error.mean().cpu().item()),
            "mse": float(mse.cpu().item()),
            "psnr": psnr,
            "ssim": float(ssim_map.mean().cpu().item()),
            "edge_mae": float(edge_error.mean().cpu().item()),
            "laplacian_mae": float(laplacian_error.mean().cpu().item()),
            "confidence_mean": float(generated["confidence"].mean().cpu().item()),
            "uncertainty_mean": float(generated["uncertainty"].mean().cpu().item()),
            "uncertainty_error_corr": pearson_correlation(generated["uncertainty"], error),
            "high_tumor_mae": (
                float(error[high_tumor].mean().cpu().item()) if high_tumor.any() else float("nan")
            ),
            "high_tumor_under": (
                float(high_tumor_under[high_tumor].mean().cpu().item())
                if high_tumor.any()
                else float("nan")
            ),
        }
        if "stage1_synthetic" in generated:
            stage1_error = (generated["stage1_synthetic"] - target).abs()
            stage1_under = F.relu(target - generated["stage1_synthetic"])
            row["stage1_mae"] = float(stage1_error.mean().cpu().item())
            row["stage1_high_tumor_mae"] = (
                float(stage1_error[high_tumor].mean().cpu().item())
                if high_tumor.any()
                else float("nan")
            )
            row["stage1_high_tumor_under"] = (
                float(stage1_under[high_tumor].mean().cpu().item())
                if high_tumor.any()
                else float("nan")
            )
            row["high_tumor_mae_delta_from_stage1"] = (
                row["high_tumor_mae"] - row["stage1_high_tumor_mae"]
                if high_tumor.any()
                else float("nan")
            )
            row["high_tumor_under_delta_from_stage1"] = (
                row["high_tumor_under"] - row["stage1_high_tumor_under"]
                if high_tumor.any()
                else float("nan")
            )
            harm_map = (error > stage1_error + 1e-4).float()
            improvement_map = (error + 1e-4 < stage1_error).float()
            applied_delta = (synthetic - generated["stage1_synthetic"]).abs()
            overshoot_map = (applied_delta > stage1_error.detach() + 1e-4).float()
            row["refinement_harm_rate"] = float(harm_map.mean().cpu().item())
            row["refinement_improvement_rate"] = float(improvement_map.mean().cpu().item())
            row["refinement_overshoot_rate"] = float(overshoot_map.mean().cpu().item())
            row["delta_from_stage1"] = float(
                applied_delta.mean().cpu().item()
            )
            gate_target = lesion_refinement_target(batch["target_regions"], dilation=2).to(
                device=synthetic.device,
                dtype=synthetic.dtype,
            )
            background = (1.0 - gate_target).clamp(0.0, 1.0)
            lesion = gate_target.clamp(0.0, 1.0)
            row["background_delta_from_stage1"] = float(
                weighted_mean(
                    applied_delta,
                    background.clamp_min(1e-4),
                )
                .cpu()
                .item()
            )
            row["lesion_delta_from_stage1"] = float(
                weighted_mean(
                    applied_delta,
                    lesion.clamp_min(1e-4),
                )
                .cpu()
                .item()
            )
            row["lesion_harm_rate"] = float(
                weighted_mean(harm_map, lesion.clamp_min(1e-4)).cpu().item()
            )
            row["background_harm_rate"] = float(
                weighted_mean(harm_map, background.clamp_min(1e-4)).cpu().item()
            )
            row["lesion_overshoot_rate"] = float(
                weighted_mean(overshoot_map, lesion.clamp_min(1e-4)).cpu().item()
            )
            row["background_overshoot_rate"] = float(
                weighted_mean(overshoot_map, background.clamp_min(1e-4)).cpu().item()
            )
        if "refinement_gate" in generated:
            gate = generated["refinement_gate"]
            gate_target = lesion_refinement_target(batch["target_regions"], dilation=2).to(
                device=gate.device,
                dtype=gate.dtype,
            )
            background = (1.0 - gate_target).clamp(0.0, 1.0)
            lesion = gate_target.clamp(0.0, 1.0)
            row["gate_mean"] = float(gate.mean().cpu().item())
            row["gate_lesion_mean"] = float(
                weighted_mean(gate, lesion.clamp_min(1e-4)).cpu().item()
            )
            row["gate_background_mean"] = float(
                weighted_mean(gate, background.clamp_min(1e-4)).cpu().item()
            )
        if "refinement_acceptance" in generated:
            acceptance = generated["refinement_acceptance"]
            gate_target = lesion_refinement_target(batch["target_regions"], dilation=2).to(
                device=acceptance.device,
                dtype=acceptance.dtype,
            )
            background = (1.0 - gate_target).clamp(0.0, 1.0)
            lesion = gate_target.clamp(0.0, 1.0)
            row["acceptance_mean"] = float(acceptance.mean().cpu().item())
            row["acceptance_lesion_mean"] = float(
                weighted_mean(acceptance, lesion.clamp_min(1e-4)).cpu().item()
            )
            row["acceptance_background_mean"] = float(
                weighted_mean(acceptance, background.clamp_min(1e-4)).cpu().item()
            )
        if "prompt_logits" in generated:
            row.update(
                {
                    f"prompt_{key}": value
                    for key, value in dice_scores(
                        generated["prompt_logits"],
                        batch["target_regions"],
                        region_names=BRATS_REGIONS,
                    ).items()
                }
            )
        for index, region in enumerate(BRATS_REGIONS):
            mask = batch["target_regions"][:, index : index + 1] > 0.5
            row[f"mae_{region}"] = (
                float(error[mask].mean().cpu().item()) if mask.any() else float("nan")
            )
            if "stage1_synthetic" in generated:
                row[f"stage1_mae_{region}"] = (
                    float(stage1_error[mask].mean().cpu().item()) if mask.any() else float("nan")
                )
                row[f"mae_{region}_delta_from_stage1"] = (
                    row[f"mae_{region}"] - row[f"stage1_mae_{region}"]
                    if mask.any()
                    else float("nan")
                )
            row[f"ssim_{region}"] = (
                float(ssim_map[mask].mean().cpu().item()) if mask.any() else float("nan")
            )
            row[f"edge_mae_{region}"] = (
                float(edge_error[mask].mean().cpu().item()) if mask.any() else float("nan")
            )
            row[f"laplacian_mae_{region}"] = (
                float(laplacian_error[mask].mean().cpu().item()) if mask.any() else float("nan")
            )
        rows.append(row)
    keys = list(rows[0]) if rows else []
    return {
        key: float(np.mean([row[key] for row in rows if math.isfinite(row[key])]))
        for key in keys
    }


def eval_metric(metrics: dict[str, float], key: str, default: float = float("inf")) -> float:
    value = metrics.get(key, default)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(value):
        return float(default)
    return value


def selection_score(metrics: dict[str, float], mode: str) -> float:
    if mode == "mae":
        return eval_metric(metrics, "mae")
    if mode not in {"lesion_composite", "lesion_noharm_composite"}:
        raise ValueError(f"Unknown best metric: {mode}")
    mae = eval_metric(metrics, "mae")
    if not math.isfinite(mae):
        return float("inf")
    ssim_penalty = max(0.0, 1.0 - eval_metric(metrics, "ssim", 1.0))
    lesion_ssim_penalty = max(
        0.0,
        1.0
        - np.mean(
            [
                eval_metric(metrics, "ssim_ET", 1.0),
                eval_metric(metrics, "ssim_TC", 1.0),
                eval_metric(metrics, "ssim_WT", 1.0),
            ]
        ),
    )
    score = (
        0.35 * mae
        + 0.20 * eval_metric(metrics, "high_tumor_mae", mae)
        + 0.16 * eval_metric(metrics, "high_tumor_under", mae)
        + 0.12 * eval_metric(metrics, "mae_ET", mae)
        + 0.08 * eval_metric(metrics, "mae_TC", mae)
        + 0.04 * eval_metric(metrics, "mae_WT", mae)
        + 0.03 * eval_metric(metrics, "edge_mae", mae)
        + 0.02 * eval_metric(metrics, "laplacian_mae", mae)
        + 0.08 * eval_metric(metrics, "refinement_harm_rate", 0.0)
        + 0.08 * eval_metric(metrics, "lesion_harm_rate", 0.0)
        + 0.04 * eval_metric(metrics, "background_delta_from_stage1", 0.0)
        + 0.04 * ssim_penalty
        + 0.04 * float(lesion_ssim_penalty)
    )
    if mode == "lesion_noharm_composite":
        stage1_mae = eval_metric(metrics, "stage1_mae", mae)
        whole_regression = max(0.0, mae - stage1_mae)
        et_regression = max(0.0, eval_metric(metrics, "mae_ET_delta_from_stage1", 0.0))
        tc_regression = max(0.0, eval_metric(metrics, "mae_TC_delta_from_stage1", 0.0))
        wt_regression = max(0.0, eval_metric(metrics, "mae_WT_delta_from_stage1", 0.0))
        high_regression = max(
            0.0,
            eval_metric(metrics, "high_tumor_mae_delta_from_stage1", 0.0),
        )
        score = (
            score
            + 0.40 * whole_regression
            + 0.24 * high_regression
            + 0.18 * et_regression
            + 0.12 * tc_regression
            + 0.08 * wt_regression
            + 0.14 * eval_metric(metrics, "background_harm_rate", 0.0)
            + 0.10 * eval_metric(metrics, "background_delta_from_stage1", 0.0)
        )
    return float(score)


def train_one_epoch(
    model: SliceVirtualModalityGenerator | PromptedSliceVirtualModalityGenerator | SliceDriftTransportGenerator,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    args: argparse.Namespace,
    epoch: int,
    medical_bank: MedicalDriftBank2D | None = None,
) -> dict[str, Any]:
    model.train()
    losses = []
    started = time.perf_counter()
    last_log = started
    for step, batch in enumerate(loader):
        if args.max_train_steps > 0 and step >= args.max_train_steps:
            break
        batch = to_device(batch, device)
        loss, parts = loss_for_batch(model, batch, args, medical_bank)
        if not torch.isfinite(loss).item():
            raise RuntimeError(f"Non-finite loss at step {step}: {parts}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nonfinite_grad_values = sanitize_gradients(model)
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters(model), 1.0)
        if not torch.isfinite(grad_norm).item():
            optimizer.zero_grad(set_to_none=True)
            parts["grad_norm"] = float("nan")
            parts["skipped_nonfinite_grad"] = True
            parts["nonfinite_grad_values"] = nonfinite_grad_values
            losses.append(parts)
            continue
        optimizer.step()
        parts["grad_norm"] = float(grad_norm.detach().cpu().item())
        parts["skipped_nonfinite_grad"] = False
        parts["nonfinite_grad_values"] = nonfinite_grad_values
        losses.append(parts)
        if args.log_every > 0 and (step + 1) % args.log_every == 0:
            now = time.perf_counter()
            print(
                json.dumps(
                    {
                        "epoch": epoch,
                        "step": step + 1,
                        **parts,
                        "elapsed_sec": now - started,
                        "sec_since_last_log": now - last_log,
                    }
                ),
                flush=True,
            )
            last_log = now
    keys = [key for key in losses[0] if key != "grad_norm"] if losses else []
    return {
        "epoch": epoch,
        "steps": len(losses),
        **{f"mean_{key}": float(np.mean([item[key] for item in losses])) for key in keys},
        "first_loss": losses[0] if losses else None,
        "last_loss": losses[-1] if losses else None,
        "runtime_sec": time.perf_counter() - started,
    }


def build_model(
    args: argparse.Namespace,
) -> SliceVirtualModalityGenerator | PromptedSliceVirtualModalityGenerator | SliceDriftTransportGenerator:
    in_modalities = len(BRATS_MODALITIES) * (2 * int(args.slice_context_radius) + 1)
    if args.model_kind == "prompted":
        return PromptedSliceVirtualModalityGenerator(
            PromptedSliceVirtualModalityGeneratorConfig(
                hidden_channels=args.hidden_channels,
                in_modalities=in_modalities,
                prompt_channels=len(BRATS_REGIONS),
                residual_scale=args.residual_scale,
                lesion_residual_scale=args.lesion_residual_scale,
                detail_residual_scale=args.detail_residual_scale,
                enhancement_residual_scale=args.enhancement_residual_scale,
                class_channels=len(BRATS_DATASETS),
                class_conditioned=args.class_conditioned,
                output_activation=args.output_activation,
                positive_lesion_residual=args.positive_lesion_residual,
            )
        )
    if args.model_kind == "transport":
        return SliceDriftTransportGenerator(
            SliceDriftTransportGeneratorConfig(
                hidden_channels=args.hidden_channels,
                in_modalities=in_modalities,
                transport_steps=args.transport_steps,
                transport_step_scale=args.transport_step_scale,
                velocity_scale=args.transport_velocity_scale,
                init_blur_kernel=args.transport_init_blur_kernel,
                class_channels=len(BRATS_DATASETS),
                class_conditioned=args.class_conditioned,
                output_activation=args.output_activation,
                gated_refinement=args.gated_refinement,
                refinement_residual_scale=args.refinement_residual_scale,
                gate_bias_init=args.gate_bias_init,
                refinement_acceptance_gate=args.refinement_acceptance_gate,
                accept_bias_init=args.accept_bias_init,
                refinement_channels_multiplier=args.refinement_channels_multiplier,
                refinement_blocks=args.refinement_blocks,
                refinement_detail_features=args.refinement_detail_features,
            )
        )
    return SliceVirtualModalityGenerator(
        SliceVirtualModalityGeneratorConfig(
            hidden_channels=args.hidden_channels,
            in_modalities=in_modalities,
        )
    )


def remap_checkpoint_state(
    model: SliceVirtualModalityGenerator | PromptedSliceVirtualModalityGenerator | SliceDriftTransportGenerator,
    state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    mapped = dict(state)
    if isinstance(model, PromptedSliceVirtualModalityGenerator) and "out.weight" in mapped:
        out_weight = mapped.pop("out.weight")
        out_bias = mapped.pop("out.bias", None)
        mapped["base_out.weight"] = out_weight.clone()
        if out_bias is not None:
            mapped["base_out.bias"] = out_bias.clone()
    target_state = model.state_dict()

    def copy_modalities_with_center_context(
        expanded: torch.Tensor,
        source: torch.Tensor,
        source_start: int,
        target_start: int,
        old_modalities: int,
        new_modalities: int,
    ) -> None:
        if old_modalities <= 0 or new_modalities <= 0:
            return
        old_depth = max(1, old_modalities // len(BRATS_MODALITIES))
        new_depth = max(1, new_modalities // len(BRATS_MODALITIES))
        old_center = old_depth // 2
        new_center = new_depth // 2
        for modality_index in range(len(BRATS_MODALITIES)):
            old_channel = source_start + modality_index * old_depth + min(old_center, old_depth - 1)
            new_channel = target_start + modality_index * new_depth + min(new_center, new_depth - 1)
            if old_channel < source.shape[1] and new_channel < expanded.shape[1]:
                expanded[:, new_channel] = source[:, old_channel].to(
                    device=expanded.device,
                    dtype=expanded.dtype,
                )

    def expand_transport_stem_conv(key: str) -> bool:
        if key not in mapped or key not in target_state:
            return False
        source = mapped[key]
        target = target_state[key]
        if source.ndim != 4 or target.ndim != 4 or source.shape[0] != target.shape[0] or source.shape[2:] != target.shape[2:]:
            return False
        if source.shape[1] == target.shape[1]:
            return False
        target_class_channels = int(getattr(model.config, "class_channels", 0)) if bool(
            getattr(model.config, "class_conditioned", False)
        ) else 0
        source_class_channels = target_class_channels if (source.shape[1] - target_class_channels - 3) % 2 == 0 else 0
        old_modalities = (int(source.shape[1]) - source_class_channels - 3) // 2
        new_modalities = int(getattr(model.config, "in_modalities", len(BRATS_MODALITIES)))
        if old_modalities <= 0 or new_modalities <= 0:
            return False
        expanded = target.clone()
        expanded.zero_()
        expanded[:, 0] = source[:, 0]
        copy_modalities_with_center_context(expanded, source, 1, 1, old_modalities, new_modalities)
        copy_modalities_with_center_context(
            expanded,
            source,
            1 + old_modalities,
            1 + new_modalities,
            old_modalities,
            new_modalities,
        )
        old_tail = 1 + 2 * old_modalities
        new_tail = 1 + 2 * new_modalities
        tail_count = min(source.shape[1] - old_tail, target.shape[1] - new_tail)
        if tail_count > 0:
            expanded[:, new_tail : new_tail + tail_count] = source[
                :,
                old_tail : old_tail + tail_count,
            ].to(device=expanded.device, dtype=expanded.dtype)
        mapped[key] = expanded
        return True

    def expand_refinement_input_conv(key: str) -> bool:
        if key not in mapped or key not in target_state:
            return False
        source = mapped[key]
        target = target_state[key]
        if source.ndim != 4 or target.ndim != 4 or source.shape[0] != target.shape[0] or source.shape[2:] != target.shape[2:]:
            return False
        if source.shape[1] == target.shape[1]:
            return False
        target_class_channels = int(getattr(model.config, "class_channels", 0)) if bool(
            getattr(model.config, "class_conditioned", False)
        ) else 0
        source_class_channels = target_class_channels if (source.shape[1] - target_class_channels - 3) % 2 == 0 else 0
        old_modalities = (int(source.shape[1]) - source_class_channels - 3) // 2
        new_modalities = int(getattr(model.config, "in_modalities", len(BRATS_MODALITIES)))
        if old_modalities <= 0 or new_modalities <= 0:
            return False
        expanded = target.clone()
        expanded.zero_()
        copy_modalities_with_center_context(expanded, source, 0, 0, old_modalities, new_modalities)
        copy_modalities_with_center_context(
            expanded,
            source,
            old_modalities,
            new_modalities,
            old_modalities,
            new_modalities,
        )
        old_tail = 2 * old_modalities
        new_tail = 2 * new_modalities
        tail_count = min(source.shape[1] - old_tail, target.shape[1] - new_tail)
        if tail_count > 0:
            expanded[:, new_tail : new_tail + tail_count] = source[
                :,
                old_tail : old_tail + tail_count,
            ].to(device=expanded.device, dtype=expanded.dtype)
        mapped[key] = expanded
        return True

    def expand_first_conv(key: str, old_modalities: int, new_modalities: int) -> None:
        if key not in mapped or key not in target_state:
            return
        source = mapped[key]
        target = target_state[key]
        if (
            source.ndim != target.ndim
            or source.shape[0] != target.shape[0]
            or source.shape[2:] != target.shape[2:]
            or source.shape[1] == target.shape[1]
        ):
            return
        expanded = target.clone()
        expanded.zero_()
        old_width = int(source.shape[1])
        new_width = int(target.shape[1])
        if old_modalities > 0 and new_modalities > old_modalities and old_width >= old_modalities * 2:
            old_depth = old_width // (old_modalities * 2)
            new_depth = new_modalities // old_modalities
            old_center = old_depth // 2
            new_center = new_depth // 2
            for modality_index in range(old_modalities):
                old_image = modality_index * old_depth + old_center
                new_image = modality_index * new_depth + new_center
                expanded[:, new_image] = source[:, old_image].to(
                    device=expanded.device,
                    dtype=expanded.dtype,
                )
                old_mask = old_modalities * old_depth + modality_index * old_depth + old_center
                new_mask = new_modalities + modality_index * new_depth + new_center
                if old_mask < old_width and new_mask < new_width:
                    expanded[:, new_mask] = source[:, old_mask].to(
                        device=expanded.device,
                        dtype=expanded.dtype,
                    )
        else:
            expanded[:, : min(old_width, new_width)] = source[
                :,
                : min(old_width, new_width),
            ].to(device=expanded.device, dtype=expanded.dtype)
        mapped[key] = expanded

    new_modalities = int(getattr(model.config, "in_modalities", len(BRATS_MODALITIES)))
    if isinstance(model, SliceDriftTransportGenerator):
        if not expand_transport_stem_conv("stem.0.net.0.weight"):
            expand_first_conv("stem.0.net.0.weight", len(BRATS_MODALITIES), new_modalities)
        expand_refinement_input_conv("refinement_context.0.net.0.weight")
    else:
        expand_first_conv("stem.0.net.0.weight", len(BRATS_MODALITIES), new_modalities)
    expand_first_conv("detail_residual.0.net.0.weight", len(BRATS_MODALITIES), new_modalities)
    expand_first_conv("enhancement_residual.0.net.0.weight", len(BRATS_MODALITIES), new_modalities)
    if isinstance(model, SliceDriftTransportGenerator) and bool(
        getattr(model.config, "gated_refinement", False)
    ):
        final_context_index = max(1, int(getattr(model.config, "refinement_blocks", 1))) + 1
        old_prefix = "refinement_context.2."
        new_prefix = f"refinement_context.{final_context_index}."
        if final_context_index != 2:
            for key, tensor in list(mapped.items()):
                if key.startswith(old_prefix):
                    mapped.setdefault(new_prefix + key[len(old_prefix) :], tensor)
    return mapped


def compatible_checkpoint_state(
    model: torch.nn.Module,
    state: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]], list[dict[str, Any]]]:
    target_state = model.state_dict()
    compatible: dict[str, torch.Tensor] = {}
    shape_mismatch: list[dict[str, Any]] = []
    partial_shape_loads: list[dict[str, Any]] = []
    for key, tensor in state.items():
        target_tensor = target_state.get(key)
        if target_tensor is None:
            compatible[key] = tensor
            continue
        if tuple(tensor.shape) == tuple(target_tensor.shape):
            compatible[key] = tensor
            continue
        if key.startswith("refinement_") and tensor.ndim == target_tensor.ndim:
            widened = target_tensor.clone()
            slices = tuple(
                slice(0, min(int(source_dim), int(target_dim)))
                for source_dim, target_dim in zip(tensor.shape, target_tensor.shape)
            )
            widened[slices] = tensor[slices].to(dtype=target_tensor.dtype)
            compatible[key] = widened
            partial_shape_loads.append(
                {
                    "key": key,
                    "checkpoint_shape": list(tensor.shape),
                    "model_shape": list(target_tensor.shape),
                }
            )
            continue
        shape_mismatch.append(
            {
                "key": key,
                "checkpoint_shape": list(tensor.shape),
                "model_shape": list(target_tensor.shape),
            }
        )
    return compatible, shape_mismatch, partial_shape_loads


def freeze_prompted_base(
    model: SliceVirtualModalityGenerator | PromptedSliceVirtualModalityGenerator | SliceDriftTransportGenerator,
    detail_only: bool = False,
    train_prompt_with_detail: bool = False,
) -> dict[str, int]:
    if isinstance(model, SliceDriftTransportGenerator):
        if not bool(getattr(model.config, "gated_refinement", False)):
            return {
                "frozen_parameters": 0,
                "trainable_parameters": sum(p.numel() for p in model.parameters()),
            }
        trainable_prefixes = (
            "refinement_context.",
            "refinement_feature_project.",
            "refinement_gate_head.",
            "refinement_accept_head.",
            "refinement_residual_head.",
        )
        frozen = 0
        trainable = 0
        for name, parameter in model.named_parameters():
            keep_trainable = name.startswith(trainable_prefixes)
            parameter.requires_grad = keep_trainable
            if keep_trainable:
                trainable += parameter.numel()
            else:
                frozen += parameter.numel()
        return {"frozen_parameters": frozen, "trainable_parameters": trainable}
    if not isinstance(model, PromptedSliceVirtualModalityGenerator):
        return {"frozen_parameters": 0, "trainable_parameters": sum(p.numel() for p in model.parameters())}
    if detail_only:
        trainable_prefixes = ("detail_residual.", "enhancement_residual.")
        if train_prompt_with_detail:
            trainable_prefixes = ("prompt_head.", *trainable_prefixes)
    else:
        trainable_prefixes = (
            "up1.",
            "up2.",
            "prompt_head.",
            "refine.",
            "lesion_residual.",
            "detail_residual.",
            "enhancement_residual.",
            "synthetic_head.",
            "uncertainty_head.",
        )
    frozen = 0
    trainable = 0
    for name, parameter in model.named_parameters():
        keep_trainable = name.startswith(trainable_prefixes)
        parameter.requires_grad = keep_trainable
        if keep_trainable:
            trainable += parameter.numel()
        else:
            frozen += parameter.numel()
    return {"frozen_parameters": frozen, "trainable_parameters": trainable}


def trainable_parameters(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def reset_lesion_residual_head(
    model: SliceVirtualModalityGenerator | PromptedSliceVirtualModalityGenerator | SliceDriftTransportGenerator,
    bias: float | None,
) -> bool:
    if bias is None or not isinstance(model, PromptedSliceVirtualModalityGenerator):
        return False
    final = model.lesion_residual[-1]
    torch.nn.init.zeros_(final.weight)
    torch.nn.init.constant_(final.bias, float(bias))
    return True


def sanitize_gradients(model: torch.nn.Module) -> int:
    replaced = 0
    for parameter in trainable_parameters(model):
        if parameter.grad is None:
            continue
        finite = torch.isfinite(parameter.grad)
        bad = int((~finite).sum().detach().cpu().item())
        if bad:
            parameter.grad = torch.nan_to_num(parameter.grad, nan=0.0, posinf=0.0, neginf=0.0)
            replaced += bad
    return replaced


def main() -> int:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    set_seed(args.seed)
    max_subjects = None if args.max_subjects <= 0 else args.max_subjects
    hard_slices, hard_slice_report = load_hard_slice_index(
        args.hard_slice_json,
        args.hard_slice_min_score,
        args.hard_slice_top_k,
    )
    val_slice_crop_size = (
        int(args.slice_crop_size)
        if int(args.val_slice_crop_size) < 0
        else int(args.val_slice_crop_size)
    )
    val_slice_crop_mode = (
        str(args.slice_crop_mode)
        if not str(args.val_slice_crop_mode)
        else str(args.val_slice_crop_mode)
    )
    split_dataset = BraTSSliceDataset(
        args.manifest,
        spatial_size=args.spatial_size,
        max_subjects=max_subjects,
        slices_per_subject=args.slices_per_subject,
        target_modality=args.target_modality,
        seed=args.seed,
        slice_crop_size=val_slice_crop_size,
        slice_crop_jitter=args.slice_crop_jitter,
        slice_crop_mode=val_slice_crop_mode,
        slice_context_radius=args.slice_context_radius,
    )
    train_dataset = BraTSSliceDataset(
        args.manifest,
        spatial_size=args.spatial_size,
        max_subjects=max_subjects,
        slices_per_subject=args.slices_per_subject,
        target_modality=args.target_modality,
        seed=args.seed,
        slice_crop_size=args.slice_crop_size,
        slice_crop_jitter=args.slice_crop_jitter,
        slice_crop_mode=args.slice_crop_mode,
        slice_context_radius=args.slice_context_radius,
        hard_slices=hard_slices,
        hard_slice_prob=args.hard_slice_prob,
    )
    val_dataset = split_dataset
    split_seed = int(args.seed if int(args.split_seed) < 0 else args.split_seed)
    train_indices, val_indices, split_report = subject_level_split_indices(
        split_dataset,
        args.val_subjects,
        split_seed,
    )
    train_set = Subset(train_dataset, train_indices)
    val_set = Subset(val_dataset, val_indices)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = build_model(args).to(args.device)
    resume_report = None
    if args.resume_checkpoint:
        payload = torch.load(args.resume_checkpoint, map_location="cpu", weights_only=False)
        checkpoint_target = str(
            payload.get(
                "target_modality",
                payload.get("config", {}).get("target_modality", args.target_modality),
            )
        )
        if checkpoint_target != args.target_modality:
            raise ValueError(
                f"resume checkpoint target {checkpoint_target!r} != requested {args.target_modality!r}"
            )
        checkpoint_state = remap_checkpoint_state(model, payload["model"])
        checkpoint_state, shape_mismatch, partial_shape_loads = compatible_checkpoint_state(
            model,
            checkpoint_state,
        )
        missing, unexpected = model.load_state_dict(checkpoint_state, strict=False)
        resume_report = {
            "path": args.resume_checkpoint,
            "missing_keys": list(missing),
            "unexpected_keys": list(unexpected),
            "shape_mismatch_keys": shape_mismatch,
            "partial_shape_loads": partial_shape_loads,
            "source_model_kind": payload.get("config", {}).get("model_kind", "slice"),
            "target_model_kind": args.model_kind,
        }
    lesion_reset = reset_lesion_residual_head(model, args.reset_lesion_residual_bias)
    freeze_report = (
        freeze_prompted_base(
            model,
            args.detail_only_refinement,
            args.train_prompt_with_detail,
        )
        if args.freeze_base_generator
        else {
            "frozen_parameters": 0,
            "trainable_parameters": sum(p.numel() for p in model.parameters()),
        }
    )
    optimizer_parameters = trainable_parameters(model)
    if not optimizer_parameters:
        raise RuntimeError("No trainable parameters selected")
    optimizer = torch.optim.AdamW(optimizer_parameters, lr=args.lr, weight_decay=1e-4)
    medical_bank = (
        MedicalDriftBank2D(
            max_tokens=args.medical_drift_bank_size,
            max_add_tokens=args.medical_drift_max_add_tokens,
            memory_tokens=args.medical_drift_memory_tokens,
        )
        if float(args.medical_drift_weight) > 0.0
        else None
    )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "slice_virtual_modality_generator_last.pt"
    best_checkpoint_path = output_dir / "slice_virtual_modality_generator_best.pt"
    best_eval = None
    best_epoch = None
    best_score = float("inf")
    started = time.perf_counter()
    epoch_reports = []
    eval_reports = []
    for epoch in range(args.epochs):
        train_report = train_one_epoch(
            model,
            train_loader,
            optimizer,
            args.device,
            args,
            epoch,
            medical_bank,
        )
        eval_report = evaluate(model, val_loader, args.device)
        eval_score = selection_score(eval_report, args.best_metric)
        eval_report = dict(eval_report)
        eval_report["selection_score"] = float(eval_score)
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "best_metric": args.best_metric,
                    **eval_report,
                }
            ),
            flush=True,
        )
        epoch_reports.append(train_report)
        eval_reports.append({"epoch": epoch, "metrics": eval_report})
        if math.isfinite(float(eval_score)):
            if best_eval is None or float(eval_score) < best_score:
                best_score = float(eval_score)
                best_eval = dict(eval_report)
                best_epoch = int(epoch)
                torch.save(
                    {
                        "model": model.state_dict(),
                        "config": vars(args),
                        "target_modality": args.target_modality,
                        "modalities": BRATS_MODALITIES,
                        "best_epoch": best_epoch,
                        "best_eval": best_eval,
                        "best_metric": args.best_metric,
                        "best_score": best_score,
                    },
                    best_checkpoint_path,
                )
    torch.save(
        {
            "model": model.state_dict(),
            "config": vars(args),
            "target_modality": args.target_modality,
            "modalities": BRATS_MODALITIES,
        },
        checkpoint_path,
    )
    report = {
        "verdict": "SLICE VIRTUAL MODALITY DRIFTING: PASS",
        "git_commit": git_commit(),
        "environment": environment(args.device),
        "config": vars(args),
        "derived_config": {
            "val_slice_crop_size": val_slice_crop_size,
            "val_slice_crop_mode": val_slice_crop_mode,
        },
        "resume": resume_report,
        "lesion_residual_head_reset": lesion_reset,
        "freeze": freeze_report,
        "split": split_report,
        "hard_slices": hard_slice_report | {"sampling_prob": float(args.hard_slice_prob)},
        "train_slices": len(train_set),
        "val_slices": len(val_set),
        "epochs": epoch_reports,
        "eval_by_epoch": eval_reports,
        "final_eval": eval_reports[-1]["metrics"] if eval_reports else {},
        "best_epoch": best_epoch,
        "best_eval": best_eval or {},
        "best_metric": args.best_metric,
        "best_score": best_score if best_epoch is not None else None,
        "best_checkpoint_path": str(best_checkpoint_path) if best_epoch is not None else None,
        "checkpoint_path": str(checkpoint_path),
        "memory": {
            "peak_allocated_mb": (
                torch.cuda.max_memory_allocated() / (1024**2)
                if torch.cuda.is_available()
                else None
            ),
            "peak_reserved_mb": (
                torch.cuda.max_memory_reserved() / (1024**2)
                if torch.cuda.is_available()
                else None
            ),
        },
        "runtime_sec": time.perf_counter() - started,
    }
    out_path = Path(args.report_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
