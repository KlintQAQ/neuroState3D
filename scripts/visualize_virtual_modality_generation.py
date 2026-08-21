from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.brats_fusion_dataset import BRATS_MODALITIES, BraTSFusionDataset  # noqa: E402
from models.virtual_modality_generator import (  # noqa: E402
    VirtualModalityGenerator,
    VirtualModalityGeneratorConfig,
    observed_mask_without_target,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize synthetic missing-modality generation slices."
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
        "--checkpoint",
        default=str(
            ROOT
            / "outputs"
            / "virtual_modality_drifting_16_smoke"
            / "virtual_modality_generator_last.pt"
        ),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--spatial-size", type=int, default=16)
    parser.add_argument("--max-subjects", type=int, default=12)
    parser.add_argument("--val-subjects", type=int, default=3)
    parser.add_argument("--num-cases", type=int, default=3)
    parser.add_argument("--target-modality", default="", choices=("", *BRATS_MODALITIES))
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "reports" / "virtual_modality_generation_visual"),
    )
    return parser.parse_args()


def load_model(checkpoint_path: str, device: str, target_modality: str) -> tuple[VirtualModalityGenerator, str]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = payload.get("config", {})
    hidden = int(config.get("hidden_channels", 12))
    checkpoint_target = str(payload.get("target_modality", config.get("target_modality", "t1c")))
    target = target_modality or checkpoint_target
    model = VirtualModalityGenerator(
        VirtualModalityGeneratorConfig(hidden_channels=hidden)
    )
    model.load_state_dict(payload["model"], strict=True)
    model.to(device).eval()
    return model, target


def center_slice(seg: torch.Tensor) -> int:
    foreground = torch.nonzero(seg > 0, as_tuple=False)
    if foreground.numel() == 0:
        return int(seg.shape[0] // 2)
    return int(torch.median(foreground[:, 0].to(torch.float32)).item())


def normalize_panel(array: np.ndarray) -> np.ndarray:
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return array
    lo, hi = np.percentile(finite, [1, 99])
    if hi <= lo:
        return np.zeros_like(array)
    return np.clip((array - lo) / (hi - lo), 0.0, 1.0)


@torch.no_grad()
def main() -> int:
    args = parse_args()
    model, target_modality = load_model(args.checkpoint, args.device, args.target_modality)
    target_index = BRATS_MODALITIES.index(target_modality)
    dataset = BraTSFusionDataset(
        manifest_csv=args.manifest,
        spatial_size=args.spatial_size,
        max_subjects=None if args.max_subjects <= 0 else args.max_subjects,
        modality_mask_mode="all",
        foreground_crop_prob=1.0,
        crop_mode="region_balanced",
        seed=46,
    )
    val_count = min(args.val_subjects, max(1, len(dataset) // 4))
    indices = list(range(len(dataset)))
    val_set = Subset(dataset, indices[len(dataset) - val_count :])
    loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for case_index, batch in enumerate(loader):
        if case_index >= args.num_cases:
            break
        image = batch["image"].to(args.device)
        seg = batch["seg"][0]
        mask = observed_mask_without_target(
            image.shape[0],
            image.shape[1],
            target_index,
            image.device,
        ).to(image.dtype)
        output = model(image, mask, target_index)
        target = image[:, target_index : target_index + 1]
        synthetic = output["synthetic"]
        error = (synthetic - target).abs()
        uncertainty = output["uncertainty"]
        z = center_slice(seg)

        target_slice = target[0, 0, z].detach().cpu().numpy()
        synth_slice = synthetic[0, 0, z].detach().cpu().numpy()
        error_slice = error[0, 0, z].detach().cpu().numpy()
        uncert_slice = uncertainty[0, 0, z].detach().cpu().numpy()
        seg_slice = seg[z].detach().cpu().numpy()

        fig, axes = plt.subplots(1, 5, figsize=(14, 3.4), dpi=140)
        panels = [
            ("Real " + target_modality.upper(), normalize_panel(target_slice), "gray"),
            ("Synthetic " + target_modality.upper(), normalize_panel(synth_slice), "gray"),
            ("Absolute Error", normalize_panel(error_slice), "magma"),
            ("Uncertainty", normalize_panel(uncert_slice), "viridis"),
            ("Tumor Mask", seg_slice, "tab20"),
        ]
        for axis, (title, array, cmap) in zip(axes, panels):
            axis.imshow(np.rot90(array), cmap=cmap)
            axis.set_title(title, fontsize=9)
            axis.axis("off")
        subject = str(batch["subject_id"][0])
        fig.suptitle(f"{subject} | missing {target_modality.upper()} generation", fontsize=10)
        fig.tight_layout()
        out_path = output_dir / f"{case_index:02d}_{subject}_{target_modality}_generation.png"
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        written.append(str(out_path))

    summary = {
        "checkpoint": str(args.checkpoint),
        "target_modality": target_modality,
        "files": written,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
