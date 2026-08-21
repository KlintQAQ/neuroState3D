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


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.brats_fusion_dataset import (  # noqa: E402
    BRATS_MODALITIES,
    BRATS_REGIONS,
    brats_region_targets,
    modality_mask_from_names,
)
from models.evidence_fusion import EvidenceFusionConfig, EvidenceReliableFusion  # noqa: E402
from utils.brats_metrics import dice_scores  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize NeuroState evidence-fusion BraTS predictions."
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
    parser.add_argument(
        "--trained-checkpoint",
        default=str(
            ROOT
            / "outputs"
            / "evidence_fusion_complete_local_96_e1"
            / "evidence_fusion_small_last.pt"
        ),
    )
    parser.add_argument(
        "--brainmvp-checkpoint",
        default=str(ROOT / "pretrained" / "BrainMVP_uniformer.pt"),
    )
    parser.add_argument(
        "--report",
        default=str(ROOT / "reports" / "evidence_fusion_complete_local_96_e1.json"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "reports" / "evidence_fusion_visual_complete_local_96"),
    )
    parser.add_argument("--spatial-size", type=int, default=96)
    parser.add_argument("--val-subjects", type=int, default=235)
    parser.add_argument("--num-cases", type=int, default=3)
    parser.add_argument("--hidden-channels", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def read_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def crop_bounds(center: np.ndarray, spatial: tuple[int, int, int], size: int) -> tuple[slice, slice, slice]:
    slices = []
    for axis, length in enumerate(spatial):
        low = int(center[axis]) - size // 2
        high = low + size
        if low < 0:
            low = 0
            high = size
        if high > length:
            high = length
            low = high - size
        slices.append(slice(low, high))
    return slices[0], slices[1], slices[2]


def tumor_center(seg: np.ndarray) -> np.ndarray:
    coords = np.argwhere(seg > 0)
    if coords.size == 0:
        return np.asarray(seg.shape) // 2
    return np.round(np.median(coords, axis=0)).astype(int)


def select_subjects(rows: list[dict[str, str]], val_subjects: int, num_cases: int) -> list[dict[str, str]]:
    val_rows = rows[-val_subjects:]
    ranked = []
    for row in val_rows:
        seg = np.load(row["seg_path"])
        et = int((seg == 3).sum())
        tc = int(((seg == 1) | (seg == 3)).sum())
        wt = int((seg > 0).sum())
        ranked.append((et, tc, wt, row))
    ranked.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    selected = [item[-1] for item in ranked[:num_cases]]
    if len(selected) < num_cases:
        selected.extend(item[-1] for item in ranked[len(selected) : num_cases])
    return selected


def make_mask(names: list[str], batch_size: int, device: torch.device) -> torch.Tensor:
    mask = modality_mask_from_names(names).to(device)
    return mask.view(1, -1).expand(batch_size, -1).clone()


def make_state(mask: torch.Tensor) -> torch.Tensor:
    state = torch.zeros(mask.shape, dtype=torch.long, device=mask.device)
    state[mask <= 0] = 1
    return state


def load_model(args: argparse.Namespace) -> EvidenceReliableFusion:
    payload = torch.load(args.trained_checkpoint, map_location="cpu", weights_only=False)
    config = payload.get("config", {})
    hidden_channels = int(config.get("hidden_channels", args.hidden_channels))
    model = EvidenceReliableFusion(
        EvidenceFusionConfig(
            checkpoint_path=args.brainmvp_checkpoint,
            encoder_freeze=config.get("encoder_freeze", "freeze_all"),
            feature_stage=config.get("feature_stage", "stage4"),
            feature_channels=int(config.get("feature_channels", 512)),
            hidden_channels=hidden_channels,
            enable_region_query_fusion=bool(
                config.get("enable_region_query_fusion", False)
            ),
            completion_gate_weight=float(config.get("completion_gate_weight", 0.0)),
            completion_gate_cap=float(config.get("completion_gate_cap", 1.0)),
            enable_virtual_t1c_confidence=bool(
                config.get("enable_virtual_t1c_confidence", False)
            ),
            virtual_t1c_min_gate_cap=float(
                config.get("virtual_t1c_min_gate_cap", 0.05)
            ),
            virtual_t1c_max_gate_cap=float(
                config.get("virtual_t1c_max_gate_cap", 0.45)
            ),
            virtual_t1c_disagreement_scale=float(
                config.get("virtual_t1c_disagreement_scale", 0.25)
            ),
            enable_high_res_branch=bool(config.get("enable_high_res_branch", False)),
            high_res_channels=int(config.get("high_res_channels", 16)),
        )
    )
    model.load_state_dict(payload["model"], strict=True)
    model.to(args.device)
    model.eval()
    return model


def normalize_image(image: np.ndarray) -> np.ndarray:
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return np.zeros_like(image, dtype=np.float32)
    lo, hi = np.percentile(finite, [1, 99])
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((image - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def regions_to_label(regions: np.ndarray) -> np.ndarray:
    label = np.zeros(regions.shape[1:], dtype=np.uint8)
    label[regions[2] > 0.5] = 1
    label[regions[1] > 0.5] = 2
    label[regions[0] > 0.5] = 3
    return label


def overlay_mask(ax: plt.Axes, base: np.ndarray, label: np.ndarray, title: str) -> None:
    ax.imshow(normalize_image(base), cmap="gray")
    rgba = np.zeros((*label.shape, 4), dtype=np.float32)
    colors = {
        1: (0.0, 0.75, 1.0, 0.42),  # WT
        2: (1.0, 0.85, 0.0, 0.48),  # TC
        3: (1.0, 0.05, 0.05, 0.56),  # ET
    }
    for value, color in colors.items():
        rgba[label == value] = color
    ax.imshow(rgba)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def dice_title(name: str, dice: dict[str, float]) -> str:
    return (
        f"{name}\n"
        f"ET {dice['dice_ET']:.2f} | TC {dice['dice_TC']:.2f} | "
        f"WT {dice['dice_WT']:.2f}"
    )


@torch.no_grad()
def predict_cases(
    model: EvidenceReliableFusion,
    image: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, dict[str, Any]]:
    cases = {
        "full": list(BRATS_MODALITIES),
        "remove_t1c": ["t1n", "t2w", "t2f"],
        "remove_t2f": ["t1n", "t1c", "t2w"],
    }
    outputs: dict[str, dict[str, Any]] = {}
    for name, modalities in cases.items():
        mask = make_mask(modalities, image.shape[0], image.device)
        output = model(image, modality_mask=mask, modality_state=make_state(mask))
        probs = torch.sigmoid(output["logits"])
        outputs[name] = {
            "probs": probs.detach().cpu().numpy()[0],
            "dice": dice_scores(output["logits"], target, region_names=BRATS_REGIONS),
        }
    return outputs


def visualize_subject(
    model: EvidenceReliableFusion,
    row: dict[str, str],
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any]:
    image = np.load(row["multimodal_path"]).astype(np.float32)
    seg = np.load(row["seg_path"]).astype(np.int64)
    center = tumor_center(seg)
    z_slice, y_slice, x_slice = crop_bounds(center, tuple(seg.shape), args.spatial_size)
    image_crop = image[:, z_slice, y_slice, x_slice]
    seg_crop = seg[z_slice, y_slice, x_slice]
    target = brats_region_targets(torch.from_numpy(seg_crop)).unsqueeze(0).to(args.device)
    image_tensor = torch.from_numpy(image_crop).unsqueeze(0).to(args.device)
    outputs = predict_cases(model, image_tensor, target)

    tumor_area = (seg_crop > 0).sum(axis=(1, 2))
    axial_index = int(tumor_area.argmax()) if tumor_area.max() > 0 else args.spatial_size // 2
    base = image_crop[3, axial_index]
    target_label = regions_to_label(target.detach().cpu().numpy()[0])

    fig, axes = plt.subplots(2, 4, figsize=(14, 7), constrained_layout=True)
    for col, modality in enumerate(BRATS_MODALITIES):
        axes[0, col].imshow(normalize_image(image_crop[col, axial_index]), cmap="gray")
        axes[0, col].set_title(modality, fontsize=10)
        axes[0, col].axis("off")

    overlay_mask(
        axes[1, 0],
        base,
        target_label[axial_index],
        "GT\ncyan WT | yellow TC | red ET",
    )
    for col, name in enumerate(("full", "remove_t1c", "remove_t2f"), start=1):
        label = regions_to_label(outputs[name]["probs"])
        overlay_mask(
            axes[1, col],
            base,
            label[axial_index],
            dice_title(name, outputs[name]["dice"]),
        )

    subject = f"{row['dataset']}_{row['subject_id']}"
    fig.suptitle(f"{subject} | tumor-centered axial slice {axial_index}", fontsize=12)
    output_path = output_dir / f"{subject}_fusion_predictions.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    return {
        "dataset": row["dataset"],
        "subject_id": row["subject_id"],
        "output_path": str(output_path),
        "crop_center_zyx": center.tolist(),
        "axial_slice_in_crop": axial_index,
        "dice": {name: item["dice"] for name, item in outputs.items()},
    }


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(args.manifest)
    selected = select_subjects(rows, args.val_subjects, args.num_cases)
    model = load_model(args)
    summaries = [
        visualize_subject(model, row, args, output_dir)
        for row in selected
    ]
    summary = {
        "trained_checkpoint": args.trained_checkpoint,
        "report": args.report,
        "spatial_size": args.spatial_size,
        "val_subjects": args.val_subjects,
        "cases": summaries,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
