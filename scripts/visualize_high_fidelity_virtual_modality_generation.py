from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.brats_fusion_dataset import BRATS_MODALITIES  # noqa: E402
from models.high_fidelity_virtual_modality_generator import (  # noqa: E402
    HighFidelityVirtualModalityGenerator,
    HighFidelityVirtualModalityGeneratorConfig,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize high-fidelity generated missing MRI modalities."
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
    parser.add_argument("--target-modality", default="", choices=("", *BRATS_MODALITIES))
    parser.add_argument("--spatial-size", type=int, default=128)
    parser.add_argument("--num-cases", type=int, default=6)
    parser.add_argument("--batch-slices", type=int, default=8)
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "reports" / "visuals" / "high_fidelity_virtual_modality"),
    )
    return parser.parse_args()


def read_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return [row for row in csv.DictReader(handle) if row.get("status", "processed") == "processed"]


def load_model(
    checkpoint_path: str | Path,
    device: str,
    override_target: str,
) -> tuple[HighFidelityVirtualModalityGenerator, str, int]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = payload.get("config", {})
    target = override_target or str(
        payload.get("target_modality", config.get("target_modality", "t1c"))
    )
    context_slices = int(config.get("context_slices", 5))
    model = HighFidelityVirtualModalityGenerator(
        HighFidelityVirtualModalityGeneratorConfig(
            in_modalities=len(BRATS_MODALITIES),
            context_slices=context_slices,
            hidden_channels=int(config.get("hidden_channels", 32)),
            dropout=float(config.get("dropout", 0.0)),
        )
    )
    model.load_state_dict(payload["model"], strict=True)
    model.to(device).eval()
    return model, target, context_slices


def resize_image(image: torch.Tensor, spatial_size: int) -> torch.Tensor:
    if spatial_size <= 0 or tuple(image.shape[-3:]) == (spatial_size,) * 3:
        return image
    return F.interpolate(
        image.unsqueeze(0),
        size=(spatial_size, spatial_size, spatial_size),
        mode="trilinear",
        align_corners=False,
    ).squeeze(0)


def resize_seg(seg: torch.Tensor, spatial_size: int) -> torch.Tensor:
    if spatial_size <= 0 or tuple(seg.shape[-3:]) == (spatial_size,) * 3:
        return seg
    return F.interpolate(
        seg.unsqueeze(0).unsqueeze(0).float(),
        size=(spatial_size, spatial_size, spatial_size),
        mode="nearest",
    ).squeeze(0).squeeze(0).long()


@torch.no_grad()
def generate_volume(
    model: HighFidelityVirtualModalityGenerator,
    image: torch.Tensor,
    target_index: int,
    context_slices: int,
    batch_slices: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    device = next(model.parameters()).device
    image = image.to(device)
    mask = torch.ones(1, len(BRATS_MODALITIES), device=device)
    mask[:, target_index] = 0.0
    _, d, h, w = image.shape
    radius = context_slices // 2
    synthetic_slices = []
    confidence_slices = []
    uncertainty_slices = []
    for start in range(0, w, int(batch_slices)):
        end = min(start + int(batch_slices), w)
        contexts = []
        for z in range(start, end):
            indices = [
                min(max(z + offset, 0), w - 1)
                for offset in range(-radius, radius + 1)
            ]
            contexts.append(image[:, :, :, indices].permute(0, 3, 1, 2))
        batch_context = torch.stack(contexts, dim=0).contiguous()
        batch_mask = mask.expand(batch_context.shape[0], -1)
        output = model(batch_context, batch_mask)
        synthetic_slices.append(output["synthetic"].squeeze(1).cpu())
        confidence_slices.append(output["confidence"].squeeze(1).cpu())
        uncertainty_slices.append(output["uncertainty"].squeeze(1).cpu())
    synthetic = torch.cat(synthetic_slices, dim=0).numpy().transpose(1, 2, 0)
    confidence = torch.cat(confidence_slices, dim=0).numpy().transpose(1, 2, 0)
    uncertainty = torch.cat(uncertainty_slices, dim=0).numpy().transpose(1, 2, 0)
    return synthetic, confidence, uncertainty


def robust_limits(array: np.ndarray) -> tuple[float, float]:
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(finite, [1, 99])
    if hi <= lo:
        hi = lo + 1e-6
    return float(lo), float(hi)


def selected_slice(seg: np.ndarray) -> int:
    foreground = (seg > 0).sum(axis=(0, 1))
    if foreground.max() > 0:
        return int(foreground.argmax())
    return int(seg.shape[-1] // 2)


def show_slice(
    axis: plt.Axes,
    volume: np.ndarray,
    z: int,
    title: str,
    cmap: str = "gray",
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    if vmin is None or vmax is None:
        vmin, vmax = robust_limits(volume)
    axis.imshow(volume[:, :, z].T, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    axis.set_title(title, fontsize=9)
    axis.axis("off")


def psnr(mse: float) -> float:
    return 20.0 * math.log10(2.0) - 10.0 * math.log10(max(float(mse), 1e-8))


def choose_cases(rows: list[dict[str, str]], num_cases: int) -> list[dict[str, str]]:
    limits = defaultdict(lambda: max(1, num_cases // 3))
    for name in ("GLI", "MEN", "PED"):
        limits[name] = max(1, num_cases // 3)
    counts: dict[str, int] = defaultdict(int)
    chosen = []
    probe = np.linspace(0, len(rows) - 1, num=min(len(rows), max(200, num_cases * 80)), dtype=int)
    for index in probe:
        row = rows[int(index)]
        dataset = row["dataset"]
        if counts[dataset] >= limits[dataset]:
            continue
        seg = np.load(row["seg_path"])
        if int((seg > 0).sum()) < 500:
            continue
        chosen.append(row)
        counts[dataset] += 1
        if len(chosen) >= num_cases:
            break
    for row in rows:
        if len(chosen) >= num_cases:
            break
        if row not in chosen:
            chosen.append(row)
    return chosen[:num_cases]


@torch.no_grad()
def main() -> int:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    model, target_modality, context_slices = load_model(
        args.checkpoint,
        args.device,
        args.target_modality,
    )
    target_index = BRATS_MODALITIES.index(target_modality)
    rows = choose_cases(read_rows(args.manifest), int(args.num_cases))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    case_records = []
    summary_items = []
    for case_index, row in enumerate(rows):
        image = torch.from_numpy(np.load(row["multimodal_path"]).astype(np.float32, copy=False))
        seg = torch.from_numpy(np.load(row["seg_path"]).astype(np.int64, copy=False))
        image = resize_image(image, int(args.spatial_size))
        seg = resize_seg(seg, int(args.spatial_size))
        synthetic, confidence, uncertainty = generate_volume(
            model,
            image,
            target_index,
            context_slices,
            int(args.batch_slices),
        )
        image_np = image.cpu().numpy()
        seg_np = seg.cpu().numpy()
        real = image_np[target_index]
        error = np.abs(synthetic - real)
        mse = float(np.mean((synthetic - real) ** 2))
        z = selected_slice(seg_np)
        record = {
            "dataset": row["dataset"],
            "subject_id": row["subject_id"],
            "target_modality": target_modality,
            "slice_index": z,
            "mae": float(error.mean()),
            "mse": mse,
            "psnr": psnr(mse),
            "confidence_mean": float(confidence.mean()),
            "uncertainty_mean": float(uncertainty.mean()),
        }
        for label, region_mask in {
            "ET": seg_np == 3,
            "TC": (seg_np == 1) | (seg_np == 3),
            "WT": seg_np > 0,
        }.items():
            record[f"mae_{label}"] = (
                float(error[region_mask].mean()) if region_mask.any() else None
            )

        lo, hi = robust_limits(real)
        error_hi = float(np.percentile(error, 99))
        fig, axes = plt.subplots(2, 4, figsize=(14, 7), dpi=160)
        input_modalities = [
            (BRATS_MODALITIES[i].upper(), image_np[i])
            for i in range(len(BRATS_MODALITIES))
            if i != target_index
        ]
        panels: list[tuple[str, np.ndarray, str, float | None, float | None]] = [
            *[(f"{name} input", volume, "gray", None, None) for name, volume in input_modalities[:3]],
            (f"Real {target_modality.upper()}", real, "gray", lo, hi),
            (f"Generated {target_modality.upper()}", synthetic, "gray", lo, hi),
            ("Abs error", error, "magma", 0.0, error_hi),
            ("Confidence", confidence, "viridis", 0.0, 1.0),
            ("Seg label", seg_np, "tab10", 0.0, float(max(3, int(seg_np.max())))),
        ]
        for axis, (title, volume, cmap, vmin, vmax) in zip(axes.ravel(), panels):
            show_slice(axis, volume, z, title, cmap, vmin, vmax)
        fig.suptitle(
            f"{row['dataset']} {row['subject_id']} | missing {target_modality.upper()} "
            f"| slice {z} | MAE={record['mae']:.4f} | PSNR={record['psnr']:.2f}",
            fontsize=11,
        )
        fig.tight_layout()
        panel_path = output_dir / (
            f"case_{case_index:02d}_{row['dataset']}_{row['subject_id']}_"
            f"{target_modality}_compare.png"
        )
        fig.savefig(panel_path, bbox_inches="tight")
        plt.close(fig)
        record["panel_path"] = str(panel_path.resolve())
        case_records.append(record)
        summary_items.append((row, z, real, synthetic, error, lo, hi, error_hi))

    fig, axes = plt.subplots(len(summary_items), 3, figsize=(8.8, 2.6 * len(summary_items)), dpi=160)
    if len(summary_items) == 1:
        axes = np.expand_dims(axes, axis=0)
    for row_index, (row, z, real, synthetic, error, lo, hi, error_hi) in enumerate(summary_items):
        show_slice(axes[row_index, 0], real, z, f"{row['dataset']} {row['subject_id']} real", "gray", lo, hi)
        show_slice(axes[row_index, 1], synthetic, z, "generated", "gray", lo, hi)
        show_slice(axes[row_index, 2], error, z, "abs error", "magma", 0.0, error_hi)
    fig.suptitle(f"{args.spatial_size}x{args.spatial_size} comparison: real vs generated {target_modality.upper()}", fontsize=12)
    fig.tight_layout()
    summary_png = output_dir / f"summary_real_vs_generated_{target_modality}_{args.spatial_size}.png"
    fig.savefig(summary_png, bbox_inches="tight")
    plt.close(fig)

    report = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "target_modality": target_modality,
        "context_slices": context_slices,
        "spatial_size": int(args.spatial_size),
        "summary_png": str(summary_png.resolve()),
        "cases": case_records,
        "mean_mae": float(np.mean([item["mae"] for item in case_records])),
        "mean_psnr": float(np.mean([item["psnr"] for item in case_records])),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
