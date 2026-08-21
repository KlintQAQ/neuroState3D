from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.brats_fusion_dataset import BRATS_MODALITIES  # noqa: E402
from models.virtual_modality_generator import (  # noqa: E402
    VirtualModalityGenerator,
    VirtualModalityGeneratorConfig,
    observed_mask_without_target,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute virtual T1c tensors for cached missing-modality fusion."
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
        "--generator-checkpoint",
        default=str(
            ROOT
            / "outputs"
            / "virtual_modality_semantic_drift_t1c_32_quick"
            / "virtual_modality_generator_last.pt"
        ),
    )
    parser.add_argument("--target-modality", default="t1c", choices=BRATS_MODALITIES)
    parser.add_argument("--spatial-size", type=int, default=96)
    parser.add_argument("--max-subjects", type=int, default=0)
    parser.add_argument("--synthetic-image-mode", choices=("raw", "confidence_weighted"), default="confidence_weighted")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--output-root",
        default=str(ROOT / "data" / "model_ready" / "BraTS2023_HF_virtual_t1c_96"),
    )
    parser.add_argument(
        "--output-manifest",
        default=str(
            ROOT
            / "data"
            / "manifests"
            / "BraTS2023_HF"
            / "brats_virtual_t1c_96_cache.csv"
        ),
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args()


def read_manifest(path: str | Path, max_subjects: int) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("status", "processed") == "processed"]
    if max_subjects > 0:
        rows = rows[:max_subjects]
    if not rows:
        raise ValueError(f"No processed rows found in {path}")
    return rows


def load_generator(path: str, device: str) -> tuple[VirtualModalityGenerator, str]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = payload.get("config", {})
    hidden = int(config.get("hidden_channels", 12))
    target = str(payload.get("target_modality", config.get("target_modality", "t1c")))
    model = VirtualModalityGenerator(
        VirtualModalityGeneratorConfig(hidden_channels=hidden)
    )
    model.load_state_dict(payload["model"], strict=True)
    model.to(device).eval()
    return model, target


def resize_image(image: torch.Tensor, spatial_size: int) -> torch.Tensor:
    if spatial_size <= 0 or tuple(image.shape[-3:]) == (spatial_size,) * 3:
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


@torch.no_grad()
def main() -> int:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false.")
    rows = read_manifest(args.manifest, args.max_subjects)
    target_index = BRATS_MODALITIES.index(args.target_modality)
    generator, generator_target = load_generator(args.generator_checkpoint, args.device)
    if generator_target != args.target_modality:
        raise ValueError(
            f"Generator target {generator_target!r} != requested {args.target_modality!r}"
        )

    output_root = Path(args.output_root)
    manifest_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, row in enumerate(rows, start=1):
        dataset = row["dataset"]
        subject_id = row["subject_id"]
        subject_dir = output_root / dataset / subject_id
        synthetic_path = subject_dir / "virtual_t1c.npy"
        uncertainty_path = subject_dir / "virtual_t1c_uncertainty.npy"
        confidence_path = subject_dir / "virtual_t1c_confidence.npy"
        stats_path = subject_dir / "virtual_t1c_stats.json"
        if args.skip_existing and synthetic_path.exists() and uncertainty_path.exists() and confidence_path.exists():
            stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else {}
        else:
            image = torch.from_numpy(
                np.load(row["multimodal_path"]).astype(np.float32, copy=False)
            )
            image = resize_image(image, int(args.spatial_size)).unsqueeze(0).to(args.device)
            input_mask = observed_mask_without_target(
                image.shape[0],
                image.shape[1],
                target_index,
                image.device,
            ).to(image.dtype)
            generated = generator(image, input_mask, target_index)
            synthetic = generated["synthetic"]
            if args.synthetic_image_mode == "confidence_weighted":
                synthetic = synthetic * generated["confidence"]
            synthetic_np = synthetic.squeeze(0).squeeze(0).detach().cpu().numpy()
            uncertainty_np = generated["uncertainty"].squeeze(0).squeeze(0).detach().cpu().numpy()
            confidence_np = generated["confidence"].squeeze(0).squeeze(0).detach().cpu().numpy()
            save_npy(synthetic_path, synthetic_np)
            save_npy(uncertainty_path, uncertainty_np)
            save_npy(confidence_path, confidence_np)
            stats = {
                "dataset": dataset,
                "subject_id": subject_id,
                "target_modality": args.target_modality,
                "spatial_size": int(args.spatial_size),
                "synthetic_image_mode": args.synthetic_image_mode,
                "synthetic_mean": float(np.mean(synthetic_np)),
                "synthetic_std": float(np.std(synthetic_np)),
                "uncertainty_mean": float(np.mean(uncertainty_np)),
                "confidence_mean": float(np.mean(confidence_np)),
            }
            stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        manifest_rows.append(
            {
                "dataset": dataset,
                "subject_id": subject_id,
                "target_modality": args.target_modality,
                "spatial_size": int(args.spatial_size),
                "synthetic_image_mode": args.synthetic_image_mode,
                "synthetic_path": str(synthetic_path),
                "uncertainty_path": str(uncertainty_path),
                "confidence_path": str(confidence_path),
                "confidence_mean": stats.get("confidence_mean", ""),
                "uncertainty_mean": stats.get("uncertainty_mean", ""),
            }
        )
        if args.log_every > 0 and index % args.log_every == 0:
            print(
                json.dumps(
                    {
                        "processed": index,
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
        "verdict": "VIRTUAL T1C CACHE: PASS",
        "subjects": len(manifest_rows),
        "output_root": str(output_root),
        "output_manifest": str(out_manifest),
        "runtime_sec": time.perf_counter() - started,
    }
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
