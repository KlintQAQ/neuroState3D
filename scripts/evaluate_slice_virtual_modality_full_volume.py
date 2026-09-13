from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.brats_fusion_dataset import BRATS_MODALITIES  # noqa: E402
from scripts.train_slice_virtual_modality_drifting import ssim_map_2d  # noqa: E402
from scripts.visualize_slice_virtual_modality_generation import (  # noqa: E402
    brats_region_targets_np,
    generate_volume,
    load_model,
    observed_brain_support_np,
    resize_image,
    resize_seg,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Patient-level full-volume evaluation for a slice generator.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--target-modality", default="", choices=("", *BRATS_MODALITIES))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--spatial-size", type=int, default=128)
    parser.add_argument("--max-subjects", type=int, default=0)
    parser.add_argument("--val-subjects", type=int, default=235)
    parser.add_argument("--split-seed", type=int, default=4601)
    parser.add_argument("--batch-slices", type=int, default=32)
    parser.add_argument("--posterior-samples", type=int, default=1)
    parser.add_argument("--support-threshold", type=float, default=1e-5)
    parser.add_argument("--support-dilation", type=int, default=2)
    parser.add_argument("--report-path", required=True)
    return parser.parse_args()


def read_rows(path: str) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return [row for row in csv.DictReader(handle) if row.get("status", "processed") == "processed"]


def correlation(x: np.ndarray, y: np.ndarray) -> float:
    x = x.astype(np.float64, copy=False).reshape(-1)
    y = y.astype(np.float64, copy=False).reshape(-1)
    x -= x.mean()
    y -= y.mean()
    denom = float(np.sqrt(np.mean(x * x) * np.mean(y * y)))
    return float(np.mean(x * y) / denom) if denom > 1e-12 else 0.0


@torch.no_grad()
def main() -> int:
    args = parse_args()
    model, target_modality = load_model(args.checkpoint, args.device, args.target_modality)
    target_index = BRATS_MODALITIES.index(target_modality)
    rows = read_rows(args.manifest)
    if int(args.max_subjects) > 0:
        rows = rows[: int(args.max_subjects)]
    rng = random.Random(int(args.split_seed))
    rng.shuffle(rows)
    val_count = min(max(1, int(args.val_subjects)), max(1, len(rows) // 4))
    rows = rows[:val_count]
    patient_metrics: list[dict[str, float | str]] = []

    for row in rows:
        image = torch.from_numpy(np.load(row["multimodal_path"]).astype(np.float32, copy=False))
        seg = torch.from_numpy(np.load(row["seg_path"]).astype(np.int64, copy=False))
        image = resize_image(image, int(args.spatial_size))
        seg = resize_seg(seg, int(args.spatial_size))
        synthetic, _, uncertainty, _, _, _, _ = generate_volume(
            model,
            image,
            target_index,
            int(args.batch_slices),
            str(row.get("dataset", "")),
            posterior_samples=int(args.posterior_samples),
        )
        image_np = image.cpu().numpy()
        seg_np = seg.cpu().numpy()
        target = image_np[target_index]
        support = observed_brain_support_np(
            image_np,
            target_index,
            float(args.support_threshold),
            int(args.support_dilation),
        )
        error = np.abs(synthetic - target)
        squared = (synthetic - target) ** 2
        valid = support if support.any() else np.ones_like(support, dtype=bool)
        mse = float(squared[valid].mean())
        ssim = ssim_map_2d(
            torch.from_numpy(synthetic.transpose(2, 0, 1))[:, None],
            torch.from_numpy(target.transpose(2, 0, 1))[:, None],
        ).squeeze(1).numpy().transpose(1, 2, 0)
        regions = brats_region_targets_np(seg_np).astype(bool)
        metrics: dict[str, float | str] = {
            "subject_id": str(row["subject_id"]),
            "mae": float(error[valid].mean()),
            "mse": mse,
            "psnr": float(10.0 * math.log10(4.0 / max(mse, 1e-12))),
            "ssim": float(ssim[valid].mean()),
            "uncertainty_mean": float(uncertainty[valid].mean()),
            "uncertainty_error_corr": correlation(uncertainty[valid], error[valid]),
        }
        for region_index, region_name in enumerate(("ET", "TC", "WT")):
            mask = regions[region_index] & valid
            metrics[f"mae_{region_name}"] = float(error[mask].mean()) if mask.any() else float("nan")
        patient_metrics.append(metrics)

    numeric_keys = [key for key in patient_metrics[0] if key != "subject_id"] if patient_metrics else []
    aggregate = {
        key: float(np.nanmean([float(row[key]) for row in patient_metrics]))
        for key in numeric_keys
    }
    report = {
        "evaluation": "full_volume_patient_level",
        "checkpoint": str(args.checkpoint),
        "target_modality": target_modality,
        "val_subjects": len(patient_metrics),
        "split_seed": int(args.split_seed),
        "posterior_samples": int(args.posterior_samples),
        "aggregate": aggregate,
        "patients": patient_metrics,
    }
    output = Path(args.report_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=True), encoding="utf-8")
    print(json.dumps(report["aggregate"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
