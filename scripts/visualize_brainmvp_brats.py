from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODALITIES = ("t1n", "t1c", "t2w", "t2f")
STAGES = ("stage1", "stage2", "stage3", "stage4")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize BrainMVP pretrained feature responses on a BraTS subject."
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
    parser.add_argument("--dataset", default="GLI")
    parser.add_argument("--subject-id", default=None)
    parser.add_argument("--subject-index", type=int, default=0)
    parser.add_argument(
        "--checkpoint",
        default=str(ROOT / "pretrained" / "BrainMVP_uniformer.pt"),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "reports" / "brats_brainmvp_visual"),
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def choose_row(
    rows: list[dict[str, str]],
    dataset: str,
    subject_id: str | None,
    subject_index: int,
) -> dict[str, str]:
    if subject_id is not None:
        for row in rows:
            if row["subject_id"] == subject_id:
                return row
        raise ValueError(f"Subject not found: {subject_id}")

    dataset_rows = [row for row in rows if row["dataset"].lower() == dataset.lower()]
    if not dataset_rows:
        raise ValueError(f"No rows found for dataset {dataset}")
    index = min(max(subject_index, 0), len(dataset_rows) - 1)
    return dataset_rows[index]


def robust01(array: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    values = array[mask] if mask is not None and np.any(mask) else array.reshape(-1)
    low, high = np.percentile(values, [1.0, 99.0])
    if float(high - low) < 1e-8:
        return np.zeros_like(array, dtype=np.float32)
    out = (array.astype(np.float32) - float(low)) / float(high - low)
    return np.clip(out, 0.0, 1.0)


def stage_activation(feature: torch.Tensor, target_shape: tuple[int, int, int]) -> np.ndarray:
    activation = feature.detach().abs().mean(dim=1, keepdim=True)
    activation = F.interpolate(
        activation,
        size=target_shape,
        mode="trilinear",
        align_corners=False,
    )
    array = activation[0, 0].float().cpu().numpy()
    return robust01(array)


def find_display_slice(seg: np.ndarray, foreground: np.ndarray) -> int:
    tumor_area = (seg > 0).sum(axis=(0, 1))
    if tumor_area.max() > 0:
        return int(tumor_area.argmax())
    foreground_area = foreground.sum(axis=(0, 1))
    if foreground_area.max() > 0:
        return int(foreground_area.argmax())
    return seg.shape[2] // 2


def encode_modalities(
    image: np.ndarray,
    checkpoint: str,
    device: str,
) -> dict[str, dict[str, np.ndarray]]:
    from models.brainmvp_encoder import BrainMVPEncoder

    encoder = BrainMVPEncoder(
        in_channels=1,
        checkpoint_path=checkpoint,
        freeze="freeze_all",
        error_on_low_coverage=True,
    ).to(device)
    encoder.eval()

    target_shape = tuple(int(item) for item in image.shape[1:])
    outputs: dict[str, dict[str, np.ndarray]] = {}
    with torch.no_grad():
        for index, modality in enumerate(MODALITIES):
            x = torch.from_numpy(image[index : index + 1]).unsqueeze(0).float().to(device)
            features = encoder(x)
            outputs[modality] = {
                stage: stage_activation(features[stage], target_shape)
                for stage in STAGES
            }
            del x, features
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return outputs


def activation_stats(
    activations: dict[str, dict[str, np.ndarray]],
    seg: np.ndarray,
    foreground: np.ndarray,
) -> list[dict[str, Any]]:
    tumor = seg > 0
    background = foreground & ~tumor
    rows: list[dict[str, Any]] = []
    for modality in MODALITIES:
        for stage in STAGES:
            activation = activations[modality][stage]
            tumor_mean = float(activation[tumor].mean()) if np.any(tumor) else 0.0
            background_mean = (
                float(activation[background].mean()) if np.any(background) else 0.0
            )
            rows.append(
                {
                    "modality": modality,
                    "stage": stage,
                    "tumor_mean_activation": tumor_mean,
                    "foreground_non_tumor_mean_activation": background_mean,
                    "tumor_to_foreground_ratio": tumor_mean / (background_mean + 1e-8),
                }
            )
    return rows


def render_modality_grid(
    image: np.ndarray,
    seg: np.ndarray,
    z_index: int,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(15, 4), constrained_layout=True)
    foreground = np.any(image != 0, axis=0)
    for index, modality in enumerate(MODALITIES):
        ax = axes[index]
        base = robust01(image[index], foreground)
        ax.imshow(base[:, :, z_index].T, cmap="gray", origin="lower")
        ax.imshow(
            np.ma.masked_where(seg[:, :, z_index].T == 0, seg[:, :, z_index].T),
            cmap="autumn",
            alpha=0.35,
            origin="lower",
            vmin=0,
            vmax=max(3, int(seg.max())),
        )
        if np.any(seg[:, :, z_index] > 0):
            ax.contour(seg[:, :, z_index].T > 0, colors="lime", linewidths=0.7, origin="lower")
        ax.set_title(modality)
        ax.axis("off")
    fig.suptitle(f"BraTS modalities with segmentation overlay, slice {z_index}")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def render_activation_grid(
    image: np.ndarray,
    seg: np.ndarray,
    activations: dict[str, dict[str, np.ndarray]],
    z_index: int,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(4, 4, figsize=(15, 15), constrained_layout=True)
    foreground = np.any(image != 0, axis=0)
    for row_index, modality in enumerate(MODALITIES):
        base = robust01(image[row_index], foreground)
        for col_index, stage in enumerate(STAGES):
            ax = axes[row_index, col_index]
            ax.imshow(base[:, :, z_index].T, cmap="gray", origin="lower")
            ax.imshow(
                activations[modality][stage][:, :, z_index].T,
                cmap="magma",
                alpha=0.55,
                origin="lower",
                vmin=0,
                vmax=1,
            )
            if np.any(seg[:, :, z_index] > 0):
                ax.contour(seg[:, :, z_index].T > 0, colors="cyan", linewidths=0.6, origin="lower")
            if row_index == 0:
                ax.set_title(stage)
            if col_index == 0:
                ax.set_ylabel(modality)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle("BrainMVP pretrained activation maps over BraTS MRI")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def render_ratio_heatmap(rows: list[dict[str, Any]], output_path: Path) -> None:
    ratios = np.zeros((len(MODALITIES), len(STAGES)), dtype=np.float32)
    for row in rows:
        ratios[MODALITIES.index(row["modality"]), STAGES.index(row["stage"])] = row[
            "tumor_to_foreground_ratio"
        ]

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    im = ax.imshow(ratios, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(STAGES)), labels=STAGES)
    ax.set_yticks(range(len(MODALITIES)), labels=MODALITIES)
    for i in range(ratios.shape[0]):
        for j in range(ratios.shape[1]):
            ax.text(j, i, f"{ratios[i, j]:.2f}x", ha="center", va="center", color="white")
    ax.set_title("Activation enrichment inside tumor mask")
    ax.set_xlabel("BrainMVP feature stage")
    ax.set_ylabel("Input modality")
    fig.colorbar(im, ax=ax, label="tumor / non-tumor foreground")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    row = choose_row(read_rows(Path(args.manifest)), args.dataset, args.subject_id, args.subject_index)
    image = np.load(row["multimodal_path"]).astype(np.float32)
    seg = np.load(row["seg_path"]).astype(np.uint8)
    foreground = np.any(image != 0, axis=0)
    z_index = find_display_slice(seg, foreground)

    activations = encode_modalities(image, args.checkpoint, args.device)
    stats = activation_stats(activations, seg, foreground)

    stem = f"{row['dataset']}_{row['subject_id']}"
    modality_png = output_dir / f"{stem}_modalities.png"
    activation_png = output_dir / f"{stem}_brainmvp_activations.png"
    ratio_png = output_dir / f"{stem}_activation_ratios.png"
    summary_json = output_dir / f"{stem}_summary.json"

    render_modality_grid(image, seg, z_index, modality_png)
    render_activation_grid(image, seg, activations, z_index, activation_png)
    render_ratio_heatmap(stats, ratio_png)

    summary = {
        "dataset": row["dataset"],
        "subject_id": row["subject_id"],
        "slice_index": z_index,
        "input_shape": list(image.shape),
        "seg_shape": list(seg.shape),
        "checkpoint": str(args.checkpoint),
        "device": args.device,
        "activation_stats": stats,
        "outputs": {
            "modalities_png": str(modality_png),
            "activation_png": str(activation_png),
            "ratio_png": str(ratio_png),
        },
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
