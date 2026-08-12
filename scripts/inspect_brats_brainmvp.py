from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
MODALITIES = ("t1n", "t1c", "t2w", "t2f")

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect BraTS model-ready arrays and test BrainMVP input compatibility."
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
        "--summary",
        default=str(
            ROOT
            / "data"
            / "manifests"
            / "BraTS2023_HF"
            / "brats_model_ready_summary.json"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional official BrainMVP checkpoint for weight coverage validation.",
    )
    parser.add_argument("--sample-per-dataset", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--forward-subject-index", type=int, default=0)
    parser.add_argument("--skip-forward", action="store_true")
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def file_size_gb(paths: list[Path]) -> float:
    total = sum(path.stat().st_size for path in paths if path.exists())
    return total / (1024**3)


def load_shape_dtype(path: Path) -> tuple[list[int], str]:
    array = np.load(path, mmap_mode="r")
    return list(array.shape), str(array.dtype)


def sample_rows(rows: list[dict[str, str]], sample_per_dataset: int) -> list[dict[str, str]]:
    by_dataset: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_dataset[row["dataset"]].append(row)
    selected: list[dict[str, str]] = []
    for dataset in sorted(by_dataset):
        dataset_rows = by_dataset[dataset]
        if len(dataset_rows) <= sample_per_dataset:
            selected.extend(dataset_rows)
            continue
        if sample_per_dataset == 1:
            indices = [len(dataset_rows) // 2]
        else:
            indices = [
                round(i * (len(dataset_rows) - 1) / (sample_per_dataset - 1))
                for i in range(sample_per_dataset)
            ]
        selected.extend(dataset_rows[index] for index in indices)
    return selected


def inspect_sample(row: dict[str, str]) -> dict[str, Any]:
    multimodal_path = Path(row["multimodal_path"])
    seg_path = Path(row["seg_path"])
    image = np.load(multimodal_path)
    seg = np.load(seg_path)
    channel_stats = []
    for index, name in enumerate(MODALITIES):
        channel = image[index].astype(np.float32)
        finite = np.isfinite(channel)
        nonzero = channel != 0
        values = channel[nonzero]
        channel_stats.append(
            {
                "modality": name,
                "finite": bool(finite.all()),
                "min": float(channel.min()),
                "max": float(channel.max()),
                "mean_nonzero": float(values.mean()) if values.size else 0.0,
                "std_nonzero": float(values.std()) if values.size else 0.0,
                "nonzero_fraction": float(nonzero.mean()),
            }
        )
    labels, counts = np.unique(seg, return_counts=True)
    return {
        "dataset": row["dataset"],
        "subject_id": row["subject_id"],
        "multimodal_shape": list(image.shape),
        "multimodal_dtype": str(image.dtype),
        "seg_shape": list(seg.shape),
        "seg_dtype": str(seg.dtype),
        "seg_labels": {str(int(label)): int(count) for label, count in zip(labels, counts)},
        "channel_stats": channel_stats,
    }


def inspect_manifest(rows: list[dict[str, str]]) -> dict[str, Any]:
    dataset_counts = Counter(row["dataset"] for row in rows)
    missing_files = []
    shape_counts: Counter[str] = Counter()
    seg_shape_counts: Counter[str] = Counter()
    dtype_counts: Counter[str] = Counter()
    seg_dtype_counts: Counter[str] = Counter()
    all_paths: list[Path] = []
    for row in rows:
        multimodal_path = Path(row["multimodal_path"])
        seg_path = Path(row["seg_path"])
        all_paths.extend([multimodal_path, seg_path])
        if not multimodal_path.exists():
            missing_files.append(str(multimodal_path))
            continue
        if not seg_path.exists():
            missing_files.append(str(seg_path))
            continue
        shape, dtype = load_shape_dtype(multimodal_path)
        seg_shape, seg_dtype = load_shape_dtype(seg_path)
        shape_counts["x".join(str(item) for item in shape)] += 1
        seg_shape_counts["x".join(str(item) for item in seg_shape)] += 1
        dtype_counts[dtype] += 1
        seg_dtype_counts[seg_dtype] += 1

    return {
        "subjects": len(rows),
        "dataset_counts": dict(sorted(dataset_counts.items())),
        "missing_files": missing_files[:20],
        "missing_file_count": len(missing_files),
        "multimodal_shape_counts": dict(shape_counts),
        "seg_shape_counts": dict(seg_shape_counts),
        "multimodal_dtype_counts": dict(dtype_counts),
        "seg_dtype_counts": dict(seg_dtype_counts),
        "model_ready_size_gb": file_size_gb(all_paths),
    }


def run_brainmvp_forward(
    row: dict[str, str],
    checkpoint: str | None,
    device: str,
) -> dict[str, Any]:
    from models.brainmvp_encoder import BrainMVPEncoder
    from models.neurostate3d import NeuroState3D

    image = np.load(row["multimodal_path"]).astype(np.float32)
    x = torch.from_numpy(image).unsqueeze(0).to(device)
    encoder = BrainMVPEncoder(
        in_channels=4,
        checkpoint_path=checkpoint,
        freeze="freeze_all",
        error_on_low_coverage=True,
    ).to(device)
    encoder.eval()
    with torch.no_grad():
        feature_summary = encoder.feature_summary(x)

    load_report = None
    if encoder.last_load_report is not None:
        report = encoder.last_load_report
        load_report = {
            "matched_tensor_ratio": report.matched_tensor_ratio,
            "matched_parameter_ratio": report.matched_parameter_ratio,
            "missing_key_count": len(report.missing_keys),
            "unexpected_key_count": len(report.unexpected_keys),
            "shape_mismatch_count": len(report.shape_mismatch_keys),
            "warnings": report.warnings,
        }

    modality_inputs = {
        name: x[:, index : index + 1].contiguous()
        for index, name in enumerate(MODALITIES)
    }
    neuro_model = NeuroState3D(
        modalities=list(MODALITIES),
        encoder=None,
        input_channels=1,
        fusion_type="mean",
        feature_stage="stage4",
        adapter_enabled=False,
        checkpoint_path=checkpoint,
        freeze_backbone="freeze_all",
    ).to(device)
    neuro_model.eval()
    with torch.no_grad():
        neuro_output = neuro_model(modality_inputs)
    fused = neuro_output["fused_feature"]
    weights = neuro_output["fusion_weights"]

    return {
        "subject": {
            "dataset": row["dataset"],
            "subject_id": row["subject_id"],
            "input_shape": list(x.shape),
            "input_dtype": str(x.dtype),
            "device": str(x.device),
        },
        "brainmvp_4ch_feature_summary": feature_summary,
        "checkpoint_load_report": load_report,
        "neurostate_single_channel_slots": {
            "modalities": list(neuro_output["modalities"]),
            "modality_mask": neuro_output["modality_mask"].detach().cpu().tolist(),
            "fused_shape": list(fused.shape),
            "fused_dtype": str(fused.dtype),
            "fusion_weights": weights.detach().cpu().tolist(),
        },
    }


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest)
    summary_path = Path(args.summary)
    rows = read_csv(manifest_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else None

    report: dict[str, Any] = {
        "manifest": str(manifest_path),
        "summary": summary,
        "data_integrity": inspect_manifest(rows),
        "sample_qc": [
            inspect_sample(row)
            for row in sample_rows(rows, max(args.sample_per_dataset, 1))
        ],
        "torch": {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device_requested": args.device,
        },
    }
    if torch.cuda.is_available():
        report["torch"]["cuda_device"] = torch.cuda.get_device_name(0)

    if not args.skip_forward:
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        forward_index = min(max(args.forward_subject_index, 0), len(rows) - 1)
        report["brainmvp_forward"] = run_brainmvp_forward(
            rows[forward_index],
            checkpoint=args.checkpoint,
            device=args.device,
        )
    else:
        report["brainmvp_forward"] = "SKIPPED"

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
