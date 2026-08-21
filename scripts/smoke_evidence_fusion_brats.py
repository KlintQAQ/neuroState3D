from __future__ import annotations

import argparse
import json
import math
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.brats_fusion_dataset import (  # noqa: E402
    BRATS_MODALITIES,
    BRATS_REGIONS,
    BraTSFusionDataset,
    modality_mask_from_names,
)
from models.evidence_fusion import EvidenceFusionConfig, EvidenceReliableFusion  # noqa: E402
from utils.brats_metrics import (  # noqa: E402
    counterfactual_logit_delta,
    dice_scores,
    mean_gate_by_region,
    reliability_error_auc,
    reliability_loss,
    segmentation_loss,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-train evidence-reliable BrainMVP fusion on local BraTS arrays."
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
        default=str(ROOT / "pretrained" / "BrainMVP_uniformer.pt"),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--spatial-size", type=int, default=32)
    parser.add_argument("--max-subjects", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--feature-stage", default="stage4")
    parser.add_argument("--feature-channels", type=int, default=512)
    parser.add_argument("--encoder-freeze", default="freeze_all")
    parser.add_argument("--degrade-prob", type=float, default=0.25)
    parser.add_argument("--foreground-crop-prob", type=float, default=0.8)
    parser.add_argument("--crop-mode", default="region_balanced")
    parser.add_argument("--seed", type=int, default=46)
    parser.add_argument(
        "--report-path",
        default=str(ROOT / "reports" / "evidence_fusion_smoke.json"),
    )
    return parser.parse_args()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def environment(device: str) -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": device,
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    return info


def tensor_to_device(batch: dict[str, Any], device: str) -> dict[str, Any]:
    result = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            result[key] = value.to(device)
        else:
            result[key] = value
    return result


def trainable_parameters(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    return [param for param in model.parameters() if param.requires_grad]


def run_step(
    model: EvidenceReliableFusion,
    batch: dict[str, Any],
    optimizer: torch.optim.Optimizer,
    reliability_weight: float = 0.2,
    aux_weight: float = 0.15,
) -> dict[str, Any]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    output = model(
        batch["image"],
        modality_mask=batch["modality_mask"],
        modality_state=batch["modality_state"],
    )
    logits = output["logits"]
    target = batch["target"]
    main = segmentation_loss(logits, target)
    rel = reliability_loss(
        output["reliability_logits"],
        logits,
        target,
        output["conflict"],
    )
    aux_logits = output["aux_logits"]
    mask = batch["modality_mask"]
    aux_terms = []
    for index in range(aux_logits.shape[1]):
        if mask[:, index].sum() <= 0:
            continue
        aux_terms.append(segmentation_loss(aux_logits[:, index], target))
    aux = torch.stack(aux_terms).mean() if aux_terms else torch.zeros_like(main)
    loss = main + reliability_weight * rel + aux_weight * aux
    if not torch.isfinite(loss).item():
        raise RuntimeError(f"Non-finite training loss: {float(loss.detach().cpu().item())}")
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters(model), 1.0)
    if not torch.isfinite(grad_norm).item():
        bad = []
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                if not torch.isfinite(param.grad).all().item():
                    bad.append(name)
        raise RuntimeError(f"Non-finite gradients in: {bad[:30]}")
    optimizer.step()
    return {
        "loss": float(loss.detach().cpu().item()),
        "seg_loss": float(main.detach().cpu().item()),
        "reliability_loss": float(rel.detach().cpu().item()),
        "aux_loss": float(aux.detach().cpu().item()),
        "grad_norm": float(grad_norm.detach().cpu().item()),
    }


@torch.no_grad()
def evaluate_batch(
    model: EvidenceReliableFusion,
    batch: dict[str, Any],
) -> dict[str, Any]:
    model.eval()
    full_mask = torch.ones_like(batch["modality_mask"])
    full_state = torch.zeros_like(batch["modality_state"])
    full = model(batch["image"], modality_mask=full_mask, modality_state=full_state)
    target = batch["target"]
    metrics: dict[str, Any] = {
        "full": {
            "dice": dice_scores(full["logits"], target, region_names=BRATS_REGIONS),
            "reliability_error_auc": reliability_error_auc(
                full["reliability"], full["logits"], target
            ),
            "gate_means": mean_gate_by_region(
                full["gates"],
                full_mask,
                BRATS_REGIONS,
                BRATS_MODALITIES,
            ),
            "conflict_mean": float(full["conflict"].mean().detach().cpu().item()),
            "reliability_mean": float(full["reliability"].mean().detach().cpu().item()),
        }
    }

    for removed in ("t1c", "t2f"):
        keep = [name for name in BRATS_MODALITIES if name != removed]
        mask = modality_mask_from_names(keep).to(batch["image"].device)
        mask = mask.view(1, -1).expand(batch["image"].shape[0], -1).clone()
        state = torch.zeros_like(batch["modality_state"])
        state[mask <= 0] = 1
        missing = model(batch["image"], modality_mask=mask, modality_state=state)
        metrics[f"remove_{removed}"] = {
            "dice": dice_scores(missing["logits"], target, region_names=BRATS_REGIONS),
            "logit_delta_vs_full": counterfactual_logit_delta(
                full["logits"],
                missing["logits"],
                target,
                region_names=BRATS_REGIONS,
            ),
            "reliability_error_auc": reliability_error_auc(
                missing["reliability"], missing["logits"], target
            ),
            "reliability_mean": float(
                missing["reliability"].mean().detach().cpu().item()
            ),
        }
    return metrics


def has_nonfinite(value: Any) -> bool:
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, dict):
        return any(has_nonfinite(item) for item in value.values())
    if isinstance(value, list):
        return any(has_nonfinite(item) for item in value)
    return False


def main() -> int:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats()

    dataset = BraTSFusionDataset(
        manifest_csv=args.manifest,
        spatial_size=args.spatial_size,
        max_subjects=args.max_subjects,
        modality_mask_mode="random_nonempty",
        degrade_prob=args.degrade_prob,
        foreground_crop_prob=args.foreground_crop_prob,
        crop_mode=args.crop_mode,
        seed=args.seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=args.device.startswith("cuda"),
    )
    config = EvidenceFusionConfig(
        checkpoint_path=args.checkpoint,
        encoder_freeze=args.encoder_freeze,
        feature_stage=args.feature_stage,
        feature_channels=args.feature_channels,
        hidden_channels=args.hidden_channels,
    )
    model = EvidenceReliableFusion(config).to(args.device)
    optimizer = torch.optim.AdamW(trainable_parameters(model), lr=args.lr)
    started = time.perf_counter()
    train_reports = []
    last_batch = None
    for step, batch in enumerate(loader):
        if step >= args.steps:
            break
        batch = tensor_to_device(batch, args.device)
        last_batch = batch
        train_reports.append(run_step(model, batch, optimizer))
    if last_batch is None:
        raise RuntimeError("No batch was produced by the BraTS smoke dataset.")
    eval_metrics = evaluate_batch(model, last_batch)

    load_report = model.encoder.last_load_report
    checkpoint_report = None
    if load_report is not None:
        checkpoint_report = {
            "matched_parameter_ratio": load_report.matched_parameter_ratio,
            "matched_tensor_ratio": load_report.matched_tensor_ratio,
            "missing_key_count": len(load_report.missing_keys),
            "unexpected_key_count": len(load_report.unexpected_keys),
            "shape_mismatch_key_count": len(load_report.shape_mismatch_keys),
            "warnings": load_report.warnings,
        }
    memory = None
    if args.device.startswith("cuda") and torch.cuda.is_available():
        memory = {
            "peak_allocated_mb": torch.cuda.max_memory_allocated() / (1024**2),
            "peak_reserved_mb": torch.cuda.max_memory_reserved() / (1024**2),
        }
    nonfinite = has_nonfinite(train_reports) or has_nonfinite(eval_metrics)
    report = {
        "verdict": (
            "EVIDENCE FUSION SMOKE: FAIL_NONFINITE"
            if nonfinite
            else "EVIDENCE FUSION SMOKE: PASS"
        ),
        "git_commit": git_commit(),
        "environment": environment(args.device),
        "config": {
            "manifest": str(Path(args.manifest)),
            "checkpoint": str(Path(args.checkpoint)),
            "spatial_size": args.spatial_size,
            "max_subjects": args.max_subjects,
            "batch_size": args.batch_size,
            "steps": args.steps,
            "feature_stage": args.feature_stage,
            "feature_channels": args.feature_channels,
            "hidden_channels": args.hidden_channels,
            "encoder_freeze": args.encoder_freeze,
            "degrade_prob": args.degrade_prob,
            "foreground_crop_prob": args.foreground_crop_prob,
            "crop_mode": args.crop_mode,
        },
        "subjects_in_dataset": len(dataset),
        "modalities": list(BRATS_MODALITIES),
        "regions": list(BRATS_REGIONS),
        "checkpoint_load": checkpoint_report,
        "train_steps": train_reports,
        "eval": eval_metrics,
        "memory": memory,
        "runtime_sec": time.perf_counter() - started,
    }
    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if nonfinite:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
