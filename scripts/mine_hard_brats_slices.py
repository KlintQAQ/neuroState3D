from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.brats_fusion_dataset import BRATS_MODALITIES  # noqa: E402
from scripts.train_slice_virtual_modality_drifting import (  # noqa: E402
    BraTSSliceDataset,
    subject_level_split_indices,
)
from scripts.visualize_slice_virtual_modality_generation import (  # noqa: E402
    context_slice_tensor,
    generate_slice,
    load_model,
    resize_image,
    resize_seg,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mine hard BraTS slices for lesion-focused stage-2 T1c generation fine-tuning."
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
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--target-modality", default="t1c", choices=BRATS_MODALITIES)
    parser.add_argument("--spatial-size", type=int, default=128)
    parser.add_argument("--max-subjects", type=int, default=0)
    parser.add_argument("--val-subjects", type=int, default=235)
    parser.add_argument("--seed", type=int, default=46)
    parser.add_argument("--split-seed", type=int, default=4601)
    parser.add_argument("--candidate-slices-per-subject", type=int, default=6)
    parser.add_argument("--top-slices-per-subject", type=int, default=3)
    parser.add_argument("--max-mine-subjects", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument(
        "--output-json",
        default=str(ROOT / "reports" / "hard_slices" / "transport_t1c_hard_slices.json"),
    )
    return parser.parse_args()


def candidate_slices(seg: np.ndarray, count: int) -> list[int]:
    et = (seg == 3).sum(axis=(0, 1)).astype(np.float64)
    tc = (((seg == 1) | (seg == 3)).sum(axis=(0, 1))).astype(np.float64)
    wt = (seg > 0).sum(axis=(0, 1)).astype(np.float64)
    score = 4.0 * et + 1.8 * tc + 0.25 * wt
    candidates = np.where(score > 0)[0]
    if candidates.size == 0:
        foreground = np.where(wt > 0)[0]
        candidates = foreground if foreground.size > 0 else np.arange(seg.shape[-1])
    order = sorted(candidates.tolist(), key=lambda z: float(score[z]), reverse=True)
    return [int(z) for z in order[: max(1, int(count))]]


def edge_magnitude_np(image: np.ndarray) -> np.ndarray:
    dx = np.pad(np.diff(image, axis=1), ((0, 0), (0, 1)))
    dy = np.pad(np.diff(image, axis=0), ((0, 1), (0, 0)))
    return np.sqrt(dx * dx + dy * dy + 1e-8)


def boundary_band_np(mask: np.ndarray, radius: int = 2) -> np.ndarray:
    tensor = torch.from_numpy(mask.astype(np.float32))[None, None]
    kernel = 2 * int(radius) + 1
    outer = F.max_pool2d(tensor, kernel_size=kernel, stride=1, padding=int(radius))
    inner = -F.max_pool2d(-tensor, kernel_size=kernel, stride=1, padding=int(radius))
    return (outer - inner).squeeze().numpy() > 0.0


def score_slice(
    synthetic: np.ndarray,
    target: np.ndarray,
    seg_slice: np.ndarray,
) -> dict[str, float | None]:
    error = np.abs(synthetic - target)
    et = seg_slice == 3
    tc = (seg_slice == 1) | (seg_slice == 3)
    wt = seg_slice > 0
    full_mae = float(error.mean())
    et_mae = float(error[et].mean()) if et.any() else None
    tc_mae = float(error[tc].mean()) if tc.any() else None
    wt_mae = float(error[wt].mean()) if wt.any() else None
    lesion_mae = next(
        item for item in (et_mae, tc_mae, wt_mae, full_mae) if item is not None
    )

    tumor = wt
    if tumor.any():
        tumor_target = target[tumor]
        threshold = float(np.quantile(tumor_target, 0.70))
        high_mask = tumor & (target >= threshold)
        enhancement_under = (
            float(np.maximum(target[high_mask] - synthetic[high_mask], 0.0).mean())
            if high_mask.any()
            else 0.0
        )
    else:
        enhancement_under = 0.0

    boundary_source = tc if tc.any() else wt
    if boundary_source.any():
        band = boundary_band_np(boundary_source, radius=2)
        boundary_error = float(
            np.abs(edge_magnitude_np(synthetic) - edge_magnitude_np(target))[band].mean()
        )
    else:
        boundary_error = 0.0
    score = (
        0.55 * float(lesion_mae)
        + 0.25 * float(enhancement_under)
        + 0.15 * float(boundary_error)
        + 0.05 * full_mae
    )
    return {
        "score": float(score),
        "mae": full_mae,
        "mae_ET": et_mae,
        "mae_TC": tc_mae,
        "mae_WT": wt_mae,
        "enhancement_under": float(enhancement_under),
        "boundary_error": float(boundary_error),
    }


@torch.no_grad()
def main() -> int:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    model, target_modality = load_model(args.checkpoint, args.device, args.target_modality)
    target_index = BRATS_MODALITIES.index(target_modality)
    context_depth = max(1, int(model.config.in_modalities) // len(BRATS_MODALITIES))
    context_radius = context_depth // 2
    max_subjects = None if int(args.max_subjects) <= 0 else int(args.max_subjects)
    split_dataset = BraTSSliceDataset(
        args.manifest,
        spatial_size=args.spatial_size,
        max_subjects=max_subjects,
        slices_per_subject=1,
        target_modality=target_modality,
        seed=args.seed,
    )
    train_indices, _, split_report = subject_level_split_indices(
        split_dataset,
        args.val_subjects,
        args.split_seed,
    )
    train_rows = [split_dataset.rows[int(index)] for index in train_indices]
    if int(args.max_mine_subjects) > 0:
        rng = random.Random(int(args.seed))
        rng.shuffle(train_rows)
        train_rows = train_rows[: int(args.max_mine_subjects)]

    records: list[dict[str, Any]] = []
    for row_index, row in enumerate(train_rows):
        image = torch.from_numpy(np.load(row["multimodal_path"]).astype(np.float32, copy=False))
        seg = torch.from_numpy(np.load(row["seg_path"]).astype(np.int64, copy=False))
        image = resize_image(image, int(args.spatial_size))
        seg = resize_seg(seg, int(args.spatial_size))
        image_np = image.cpu().numpy()
        seg_np = seg.cpu().numpy()
        real_volume = image_np[target_index]
        subject_records = []
        for z in candidate_slices(seg_np, int(args.candidate_slices_per_subject)):
            context_tensor = context_slice_tensor(image_np, z, context_radius)
            synthetic, _, _, _ = generate_slice(
                model,
                context_tensor,
                target_index,
                str(row["dataset"]),
            )
            metrics = score_slice(synthetic, real_volume[:, :, z], seg_np[:, :, z])
            subject_records.append(
                {
                    "dataset": row["dataset"],
                    "subject_id": row["subject_id"],
                    "slice_index": int(z),
                    **metrics,
                }
            )
        subject_records.sort(key=lambda item: float(item["score"]), reverse=True)
        records.extend(subject_records[: max(1, int(args.top_slices_per_subject))])
        if int(args.log_every) > 0 and (row_index + 1) % int(args.log_every) == 0:
            print(
                json.dumps(
                    {
                        "mined_subjects": row_index + 1,
                        "records": len(records),
                        "latest_subject": row["subject_id"],
                    }
                ),
                flush=True,
            )
    records.sort(key=lambda item: float(item["score"]), reverse=True)
    output = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "target_modality": target_modality,
        "spatial_size": int(args.spatial_size),
        "split": split_report,
        "candidate_slices_per_subject": int(args.candidate_slices_per_subject),
        "top_slices_per_subject": int(args.top_slices_per_subject),
        "subjects_mined": len(train_rows),
        "records": records,
    }
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output_json": str(out_path),
                "subjects_mined": len(train_rows),
                "records": len(records),
                "top_records": records[:5],
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
