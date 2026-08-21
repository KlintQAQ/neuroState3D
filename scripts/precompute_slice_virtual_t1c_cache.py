from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.brats_fusion_dataset import BRATS_MODALITIES  # noqa: E402
from models.slice_virtual_modality_generator import (  # noqa: E402
    SliceVirtualModalityGenerator,
    SliceVirtualModalityGeneratorConfig,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute 3D virtual T1c volumes from a 2D slice generator."
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
    parser.add_argument("--spatial-size", type=int, default=32)
    parser.add_argument("--batch-slices", type=int, default=32)
    parser.add_argument("--max-subjects", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--apply-brain-mask", action="store_true")
    parser.add_argument("--support-threshold", type=float, default=1e-5)
    parser.add_argument("--support-dilation", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument(
        "--output-root",
        default=str(ROOT / "data" / "model_ready" / "BraTS2023_HF_virtual_t1c_slice_32"),
    )
    parser.add_argument(
        "--output-manifest",
        default=str(
            ROOT
            / "data"
            / "manifests"
            / "BraTS2023_HF"
            / "brats_virtual_t1c_slice_32_cache.csv"
        ),
    )
    return parser.parse_args()


def read_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_model(path: str, device: str) -> tuple[SliceVirtualModalityGenerator, str]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = payload.get("config", {})
    model = SliceVirtualModalityGenerator(
        SliceVirtualModalityGeneratorConfig(
            in_modalities=len(BRATS_MODALITIES),
            hidden_channels=int(config.get("hidden_channels", 32)),
        )
    )
    model.load_state_dict(payload["model"], strict=True)
    model.to(device).eval()
    return model, str(payload.get("target_modality", config.get("target_modality", "t1c")))


def resize_volume(image: torch.Tensor, spatial_size: int) -> torch.Tensor:
    if tuple(image.shape[-3:]) == (spatial_size, spatial_size, spatial_size):
        return image
    return F.interpolate(
        image.unsqueeze(0),
        size=(spatial_size, spatial_size, spatial_size),
        mode="trilinear",
        align_corners=False,
    ).squeeze(0)


def save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array.astype(np.float32, copy=False))


def observed_brain_support(
    image: torch.Tensor,
    target_index: int,
    threshold: float,
    dilation: int,
) -> torch.Tensor:
    observed_indices = [index for index in range(image.shape[0]) if index != target_index]
    observed = image[observed_indices].abs().amax(dim=0, keepdim=True).unsqueeze(0)
    support = (observed > float(threshold)).float()
    if dilation > 0:
        support = F.max_pool3d(
            support,
            kernel_size=int(dilation) * 2 + 1,
            stride=1,
            padding=int(dilation),
        )
    return support.squeeze(0).squeeze(0) > 0.5


@torch.no_grad()
def main() -> int:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    model, target = load_model(args.checkpoint, args.device)
    if target != "t1c":
        raise ValueError(f"Expected T1c generator, got {target}")
    rows = read_rows(args.manifest)
    if args.max_subjects > 0:
        rows = rows[: int(args.max_subjects)]
    output_root = Path(args.output_root)
    manifest_rows = []
    started = time.perf_counter()
    target_index = BRATS_MODALITIES.index("t1c")
    for index, row in enumerate(rows):
        subject_dir = output_root / row["dataset"] / row["subject_id"]
        synthetic_path = subject_dir / "virtual_t1c.npy"
        uncertainty_path = subject_dir / "virtual_t1c_uncertainty.npy"
        confidence_path = subject_dir / "virtual_t1c_confidence.npy"
        if (
            args.skip_existing
            and synthetic_path.exists()
            and uncertainty_path.exists()
            and confidence_path.exists()
        ):
            stats = {
                "synthetic_mean": "",
                "synthetic_std": "",
                "uncertainty_mean": "",
                "confidence_mean": "",
            }
        else:
            image_np = np.load(row["multimodal_path"]).astype(np.float32, copy=False)
            image = torch.from_numpy(image_np).to(args.device)
            image = resize_volume(image, int(args.spatial_size))
            _, d, h, w = image.shape
            mask = torch.ones(1, len(BRATS_MODALITIES), device=args.device)
            mask[:, target_index] = 0.0
            synthetic_slices = []
            uncertainty_slices = []
            confidence_slices = []
            # Match training: slice = image[:, :, :, z] from [M, D, H, W].
            for start in range(0, w, int(args.batch_slices)):
                end = min(start + int(args.batch_slices), w)
                slices = image[:, :, :, start:end].permute(3, 0, 1, 2).contiguous()
                batch_mask = mask.expand(slices.shape[0], -1)
                generated = model(slices, batch_mask)
                synthetic_slices.append(generated["synthetic"].squeeze(1).detach().cpu())
                uncertainty_slices.append(generated["uncertainty"].squeeze(1).detach().cpu())
                confidence_slices.append(generated["confidence"].squeeze(1).detach().cpu())
            synthetic = torch.cat(synthetic_slices, dim=0).numpy()
            uncertainty = torch.cat(uncertainty_slices, dim=0).numpy()
            confidence = torch.cat(confidence_slices, dim=0).numpy()
            # Return to [D, H, W], matching the model-ready/cache convention.
            synthetic = np.transpose(synthetic, (1, 2, 0))
            uncertainty = np.transpose(uncertainty, (1, 2, 0))
            confidence = np.transpose(confidence, (1, 2, 0))
            if args.apply_brain_mask:
                support = observed_brain_support(
                    image,
                    target_index,
                    args.support_threshold,
                    args.support_dilation,
                ).detach().cpu().numpy()
                synthetic = synthetic * support.astype(np.float32)
                confidence = confidence * support.astype(np.float32)
            save_npy(synthetic_path, synthetic)
            save_npy(uncertainty_path, uncertainty)
            save_npy(confidence_path, confidence)
            stats = {
                "synthetic_mean": float(np.mean(synthetic)),
                "synthetic_std": float(np.std(synthetic)),
                "uncertainty_mean": float(np.mean(uncertainty)),
                "confidence_mean": float(np.mean(confidence)),
            }
            (subject_dir / "virtual_t1c_stats.json").write_text(
                json.dumps(stats, indent=2),
                encoding="utf-8",
            )
        manifest_rows.append(
            {
                "dataset": row["dataset"],
                "subject_id": row["subject_id"],
                "source_manifest": str(Path(args.manifest).resolve()),
                "target_modality": "t1c",
                "synthetic_image_mode": "slice_raw",
                "synthetic_path": str(synthetic_path),
                "uncertainty_path": str(uncertainty_path),
                "confidence_path": str(confidence_path),
                "confidence_mean": stats["confidence_mean"],
                "uncertainty_mean": stats["uncertainty_mean"],
            }
        )
        if args.log_every > 0 and (index + 1) % args.log_every == 0:
            print(
                json.dumps(
                    {
                        "processed": index + 1,
                        "total": len(rows),
                        "elapsed_sec": time.perf_counter() - started,
                    }
                ),
                flush=True,
            )
    out_manifest = Path(args.output_manifest)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    with out_manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    report = {
        "verdict": "SLICE VIRTUAL T1C CACHE: PASS",
        "subjects": len(manifest_rows),
        "output_root": str(output_root),
        "output_manifest": str(out_manifest),
        "runtime_sec": time.perf_counter() - started,
    }
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
