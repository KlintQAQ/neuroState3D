from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.hcp_dataset import HCPDataset, hcp_collate
from models.neurostate3d import NeuroState3D
from models.modality_adapter import DEFAULT_ADAPTER_TYPES, MODALITY_ORDER


def finite_flags(tensor: torch.Tensor) -> dict[str, bool]:
    return {
        "nan": bool(torch.isnan(tensor).any().item()),
        "inf": bool(torch.isinf(tensor).any().item()),
    }


def feature_stats(features: dict[str, torch.Tensor]) -> dict[str, dict[str, Any]]:
    return {
        stage: {
            "shape": list(tensor.shape),
            "min": float(tensor.min().detach().cpu()),
            "max": float(tensor.max().detach().cpu()),
            "mean": float(tensor.mean().detach().cpu()),
            "std": float(tensor.std(unbiased=False).detach().cpu()),
            **finite_flags(tensor),
        }
        for stage, tensor in features.items()
    }


def cuda_memory(device: str) -> dict[str, float] | None:
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return None
    allocated = int(torch.cuda.max_memory_allocated())
    reserved = int(torch.cuda.max_memory_reserved())
    return {
        "peak_allocated_bytes": allocated,
        "peak_allocated_mb": allocated / (1024**2),
        "peak_reserved_bytes": reserved,
        "peak_reserved_mb": reserved / (1024**2),
    }


def load_yaml_config(path: str) -> dict[str, Any]:
    if not path:
        return {}
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def resolve_configured_path(value: str | None) -> str:
    if not value:
        return ""
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str(ROOT / path)


def pending_report(reason: str) -> dict[str, Any]:
    return {
        "status": "PENDING_USER_AUTHENTICATION",
        "reason": reason,
        "final_verdict": "LOCAL REAL HCP T1/T2 PIPELINE: PENDING USER AUTHENTICATION",
        "required_user_action": (
            "Authenticate through the official HCP/ConnectomeDB access path, "
            "download only authorized HCP Young Adult T1w/T2w NIfTI files, "
            "and populate the local manifest with relative paths."
        ),
    }


def build_model(checkpoint: str, fusion: str, device: str) -> NeuroState3D:
    model = NeuroState3D(
        modalities=list(MODALITY_ORDER),
        fusion_type=fusion,
        feature_stage="stage4",
        adapter_types=DEFAULT_ADAPTER_TYPES,
        checkpoint_path=checkpoint,
        freeze_backbone="freeze_all",
    )
    model.to(device)
    model.eval()
    return model


def present_modalities(batch: dict[str, Any], device: str) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for modality, tensor in batch["modalities"].items():
        if tensor is not None:
            result[modality] = tensor.to(device)
    return result


def run_batch(
    batch: dict[str, Any],
    checkpoint: str,
    device: str,
) -> dict[str, Any]:
    mask = batch["modality_mask"].to(device)
    modalities = present_modalities(batch, device)
    subject_ids = batch["subject_id"]
    started = time.perf_counter()
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    mean_model = build_model(checkpoint, "mean", device)
    concat_model = build_model(checkpoint, "concat", device)
    with torch.no_grad():
        mean_output = mean_model(modalities, modality_mask=mask)
        concat_output = concat_model(modalities, modality_mask=mask)

    t1_features = mean_output["modality_features"].get("t1")
    t2_features = mean_output["modality_features"].get("t2")
    if t1_features is None or t2_features is None:
        raise AssertionError("Real HCP smoke requires both T1 and T2.")
    mean_fused = mean_output["fused_feature"]
    concat_fused = concat_output["fused_feature"]
    return {
        "subject_id": subject_ids,
        "input_shapes": {
            name: list(tensor.shape) for name, tensor in modalities.items()
        },
        "mask": mask.detach().cpu().tolist(),
        "T1_T2_SAME_PHYSICAL_SPACE": [
            item["spatial_consistency"]["T1_T2_SAME_PHYSICAL_SPACE"]
            for item in batch["metadata"]
        ],
        "REAL_HCP_T1_FORWARD": "PASS",
        "REAL_HCP_T2_FORWARD": "PASS",
        "t1_stage_stats": feature_stats(t1_features),
        "t2_stage_stats": feature_stats(t2_features),
        "mean_fusion": {
            "shape": list(mean_fused.shape),
            **finite_flags(mean_fused),
        },
        "concat_fusion": {
            "shape": list(concat_fused.shape),
            **finite_flags(concat_fused),
        },
        "gpu_memory": cuda_memory(device),
        "runtime_sec": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run real HCP T1/T2 smoke forward.")
    parser.add_argument("--config", default="configs/hcp.yaml", type=str)
    parser.add_argument("--manifest", default="", type=str)
    parser.add_argument("--checkpoint", default="", type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--roi-size", default=0, type=int)
    parser.add_argument("--batch-size", default=0, type=int)
    parser.add_argument("--num-workers", default=-1, type=int)
    parser.add_argument("--output", default="", type=str)
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    manifest = args.manifest or str(config.get("manifest", ""))
    checkpoint = args.checkpoint or str(config.get("model", {}).get("checkpoint", ""))
    roi_size = args.roi_size or int(config.get("preprocessing", {}).get("roi_size", 96))
    batch_size = args.batch_size or int(config.get("loader", {}).get("batch_size", 1))
    num_workers = (
        args.num_workers
        if args.num_workers >= 0
        else int(config.get("loader", {}).get("num_workers", 0))
    )
    output_path = args.output or str(config.get("outputs", {}).get("smoke_report", "outputs/real_hcp_smoke_report.json"))
    manifest = resolve_configured_path(manifest)
    checkpoint = resolve_configured_path(checkpoint)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false.")

    if not manifest or not Path(manifest).exists():
        report = pending_report(f"Manifest not found: {manifest or '<empty>'}")
    elif not checkpoint or not Path(checkpoint).exists():
        report = pending_report(f"BrainMVP checkpoint not found: {checkpoint or '<empty>'}")
    else:
        dataset = HCPDataset(manifest, roi_size=roi_size)
        if len(dataset) == 0:
            report = pending_report("Manifest has no real HCP subjects.")
        else:
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=hcp_collate,
            )
            batches = [
                run_batch(batch, checkpoint, args.device)
                for batch in loader
            ]
            report = {
                "status": "PASS",
                "subjects": batches,
                "final_verdict": "LOCAL REAL HCP T1/T2 PIPELINE: PASS",
            }

    output = Path(output_path)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["final_verdict"].endswith("FAIL"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
