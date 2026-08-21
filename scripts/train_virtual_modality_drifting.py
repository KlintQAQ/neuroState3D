from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.brats_fusion_dataset import (  # noqa: E402
    BRATS_MODALITIES,
    BRATS_REGIONS,
    BraTSFusionDataset,
)
from models.virtual_modality_generator import (  # noqa: E402
    VirtualModalityGenerator,
    VirtualModalityGeneratorConfig,
    observed_mask_without_target,
)
from models.brainmvp_encoder import BrainMVPEncoder  # noqa: E402
from scripts.smoke_evidence_fusion_brats import environment, git_commit  # noqa: E402
from utils.torch_drift_loss import drift_loss  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a Drifting-style 3D generator for missing BraTS modalities. "
            "Default target is T1c: observed T1n/T2w/T2f -> synthetic T1c."
        )
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
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--target-modality", default="t1c", choices=BRATS_MODALITIES)
    parser.add_argument("--spatial-size", type=int, default=32)
    parser.add_argument("--max-subjects", type=int, default=64)
    parser.add_argument("--val-subjects", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-train-steps", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--hidden-channels", type=int, default=12)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--recon-weight", type=float, default=1.0)
    parser.add_argument("--nll-weight", type=float, default=0.05)
    parser.add_argument("--gradient-weight", type=float, default=0.1)
    parser.add_argument("--calibration-weight", type=float, default=0.05)
    parser.add_argument("--drift-weight", type=float, default=0.15)
    parser.add_argument("--drift-patch-size", type=int, default=4)
    parser.add_argument("--drift-radii", nargs="+", type=float, default=[0.2, 0.05, 0.02])
    parser.add_argument(
        "--drift-source-negative-weight",
        type=float,
        default=0.0,
        help=(
            "Weakly repel generated target tokens from observed source modalities. "
            "Keep this near 0 for MRI because source modalities share anatomy with the target."
        ),
    )
    parser.add_argument("--semantic-drift-weight", type=float, default=0.0)
    parser.add_argument("--semantic-drift-stage", default="stage2")
    parser.add_argument(
        "--brainmvp-checkpoint",
        default=str(ROOT / "pretrained" / "BrainMVP_uniformer.pt"),
    )
    parser.add_argument("--foreground-crop-prob", type=float, default=1.0)
    parser.add_argument("--crop-mode", default="region_balanced")
    parser.add_argument("--focus-base", type=float, default=0.15)
    parser.add_argument("--focus-et", type=float, default=2.5)
    parser.add_argument("--focus-tc", type=float, default=1.2)
    parser.add_argument("--focus-wt", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=46)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "outputs" / "virtual_modality_drifting"),
    )
    parser.add_argument(
        "--report-path",
        default=str(ROOT / "reports" / "virtual_modality_drifting.json"),
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_device(batch: dict[str, Any], device: str) -> dict[str, Any]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def region_focus(
    target_regions: torch.Tensor,
    base: float = 0.15,
    et_weight: float = 2.5,
    tc_weight: float = 1.2,
    wt_weight: float = 0.35,
) -> torch.Tensor:
    """Tumor-aware voxel weights, emphasizing ET/TC for missing-T1c learning."""

    et = target_regions[:, 0:1]
    tc = target_regions[:, 1:2]
    wt = target_regions[:, 2:3]
    return (
        float(base)
        + float(et_weight) * et
        + float(tc_weight) * tc
        + float(wt_weight) * wt
    ).clamp_min(max(float(base), 1e-4))


def weighted_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return (value * weight).sum() / weight.sum().clamp_min(1e-8)


def gradient_difference_loss(
    synthetic: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    losses = []
    for axis in (2, 3, 4):
        pred_grad = synthetic.diff(dim=axis)
        target_grad = target.diff(dim=axis)
        grad_weight = weight.narrow(axis, 0, weight.shape[axis] - 1)
        losses.append(weighted_mean((pred_grad - target_grad).abs(), grad_weight))
    return torch.stack(losses).mean()


def uncertainty_calibration_loss(
    uncertainty: torch.Tensor,
    abs_error: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Align predicted uncertainty with detached reconstruction error."""

    target_uncertainty = abs_error.detach().clamp_min(1e-4)
    log_uncertainty = torch.log(uncertainty.clamp_min(1e-4))
    log_target = torch.log(target_uncertainty)
    return weighted_mean((log_uncertainty - log_target).abs(), weight)


def patch_tokens_3d(
    volume: torch.Tensor,
    patch_size: int,
) -> torch.Tensor:
    """Patch mean/std/intensity-range tokens for [B, C, D, H, W] volumes."""

    if volume.ndim != 5:
        raise ValueError(f"Expected volume [B,C,D,H,W], got {tuple(volume.shape)}")
    spatial = tuple(int(item) for item in volume.shape[-3:])
    kernel = min(int(patch_size), *spatial)
    if kernel <= 1:
        mean = volume
        std = torch.zeros_like(volume)
        vmin = volume
        vmax = volume
    else:
        mean = F.avg_pool3d(volume, kernel_size=kernel, stride=kernel)
        second = F.avg_pool3d(volume.square(), kernel_size=kernel, stride=kernel)
        std = (second - mean.square()).clamp_min(0.0).sqrt()
        neg_min = F.max_pool3d(-volume, kernel_size=kernel, stride=kernel)
        vmax = F.max_pool3d(volume, kernel_size=kernel, stride=kernel)
        vmin = -neg_min
    token_feature = torch.cat([mean, std, vmax - vmin], dim=1)
    return token_feature.flatten(2).transpose(1, 2).contiguous()


def patch_weights_3d(
    weight: torch.Tensor,
    patch_size: int,
) -> torch.Tensor:
    """Patch-average weights for [B, 1, D, H, W] tumor-focus maps."""

    if weight.ndim != 5:
        raise ValueError(f"Expected weight [B,1,D,H,W], got {tuple(weight.shape)}")
    spatial = tuple(int(item) for item in weight.shape[-3:])
    kernel = min(int(patch_size), *spatial)
    if kernel <= 1:
        pooled = weight
    else:
        pooled = F.avg_pool3d(weight, kernel_size=kernel, stride=kernel)
    tokens = pooled.flatten(2).squeeze(1)
    mean = tokens.mean(dim=1, keepdim=True).clamp_min(1e-8)
    return (tokens / mean).clamp(0.25, 4.0)


def modality_drift_loss(
    synthetic: torch.Tensor,
    target: torch.Tensor,
    images: torch.Tensor,
    observed_mask: torch.Tensor,
    target_index: int,
    token_focus: torch.Tensor,
    patch_size: int,
    radii: Sequence[float],
    source_negative_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    gen_tokens = patch_tokens_3d(synthetic, patch_size)
    pos_tokens = patch_tokens_3d(target, patch_size)
    token_weights = patch_weights_3d(token_focus, patch_size).to(gen_tokens.dtype)
    b, m, d, h, w = images.shape
    neg_tokens = patch_tokens_3d(images.reshape(b * m, 1, d, h, w), patch_size)
    tokens_per_modality = neg_tokens.shape[1]
    neg_tokens = neg_tokens.view(b, m * tokens_per_modality, -1)
    neg_weight = observed_mask.clone()
    neg_weight[:, target_index] = 0.0
    neg_weight = neg_weight.repeat_interleave(tokens_per_modality, dim=1)
    neg_weight = neg_weight * max(float(source_negative_weight), 0.0)
    loss, info = drift_loss(
        generated=gen_tokens,
        positive=pos_tokens,
        negative=neg_tokens,
        weight_generated=token_weights,
        weight_positive=token_weights,
        weight_negative=neg_weight,
        radii=radii,
    )
    return loss, {key: float(value.detach().cpu().item()) for key, value in info.items()}


def semantic_feature_drift_loss(
    encoder: BrainMVPEncoder | None,
    synthetic: torch.Tensor,
    target: torch.Tensor,
    token_focus: torch.Tensor,
    stage: str,
    radii: Sequence[float],
) -> tuple[torch.Tensor, dict[str, float]]:
    if encoder is None:
        zero = synthetic.new_zeros(())
        return zero, {}
    if synthetic.shape[1] != 1 or target.shape[1] != 1:
        raise ValueError("semantic_feature_drift_loss expects single-channel volumes.")
    generated_feature = encoder(synthetic)[stage]
    with torch.no_grad():
        positive_feature = encoder(target.detach())[stage].detach()
    generated_tokens = F.normalize(generated_feature.flatten(2).transpose(1, 2), dim=2)
    positive_tokens = F.normalize(positive_feature.flatten(2).transpose(1, 2), dim=2)
    focus_low = F.interpolate(
        token_focus,
        size=generated_feature.shape[-3:],
        mode="trilinear",
        align_corners=False,
    )
    token_weights = focus_low.flatten(2).squeeze(1)
    token_weights = token_weights / token_weights.mean(dim=1, keepdim=True).clamp_min(1e-8)
    token_weights = token_weights.clamp(0.25, 4.0)
    loss, info = drift_loss(
        generated=generated_tokens,
        positive=positive_tokens,
        weight_generated=token_weights,
        weight_positive=token_weights,
        radii=radii,
    )
    return loss, {key: float(value.detach().cpu().item()) for key, value in info.items()}


def loss_for_batch(
    model: VirtualModalityGenerator,
    semantic_encoder: BrainMVPEncoder | None,
    batch: dict[str, Any],
    target_index: int,
    drift_patch_size: int,
    drift_radii: Sequence[float],
    drift_source_negative_weight: float,
    semantic_drift_weight: float,
    semantic_drift_stage: str,
    focus_base: float,
    focus_et: float,
    focus_tc: float,
    focus_wt: float,
    recon_weight: float,
    nll_weight: float,
    gradient_weight: float,
    calibration_weight: float,
    drift_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    images = batch["image"]
    target = images[:, target_index : target_index + 1]
    observed_mask = observed_mask_without_target(
        images.shape[0],
        images.shape[1],
        target_index,
        images.device,
    ).to(images.dtype)
    generated = model(images, observed_mask, target_index)
    synthetic = generated["synthetic"]
    uncertainty = generated["uncertainty"]
    focus = region_focus(
        batch["target"],
        base=focus_base,
        et_weight=focus_et,
        tc_weight=focus_tc,
        wt_weight=focus_wt,
    )

    abs_error = (synthetic - target).abs()
    recon = weighted_mean(abs_error, focus)
    nll = weighted_mean(abs_error / uncertainty + uncertainty.log(), focus)
    grad = gradient_difference_loss(synthetic, target, focus)
    calibration = uncertainty_calibration_loss(uncertainty, abs_error, focus)
    drift, drift_info = modality_drift_loss(
        synthetic,
        target,
        images,
        observed_mask,
        target_index,
        focus,
        drift_patch_size,
        drift_radii,
        drift_source_negative_weight,
    )
    semantic_drift, semantic_info = semantic_feature_drift_loss(
        semantic_encoder,
        synthetic,
        target,
        focus,
        semantic_drift_stage,
        drift_radii,
    )
    total = (
        float(recon_weight) * recon
        + float(nll_weight) * nll
        + float(gradient_weight) * grad
        + float(calibration_weight) * calibration
        + float(drift_weight) * drift
        + float(semantic_drift_weight) * semantic_drift
    )
    parts = {
        "loss": float(total.detach().cpu().item()),
        "recon_l1": float(recon.detach().cpu().item()),
        "uncertainty_nll": float(nll.detach().cpu().item()),
        "gradient_l1": float(grad.detach().cpu().item()),
        "calibration_loss": float(calibration.detach().cpu().item()),
        "drift_loss": float(drift.detach().cpu().item()),
        "semantic_drift_loss": float(semantic_drift.detach().cpu().item()),
        "uncertainty_mean": float(uncertainty.detach().mean().cpu().item()),
        "confidence_mean": float(generated["confidence"].detach().mean().cpu().item()),
    }
    parts.update({f"drift_{key}": value for key, value in drift_info.items()})
    parts.update({f"semantic_drift_{key}": value for key, value in semantic_info.items()})
    return total, parts


def pearson_correlation(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.detach().flatten().float()
    y = y.detach().flatten().float()
    x = x - x.mean()
    y = y - y.mean()
    denom = x.square().mean().sqrt() * y.square().mean().sqrt()
    if denom.item() <= 1e-8:
        return float("nan")
    return float(((x * y).mean() / denom).cpu().item())


@torch.no_grad()
def evaluate(
    model: VirtualModalityGenerator,
    semantic_encoder: BrainMVPEncoder | None,
    loader: DataLoader,
    device: str,
    target_index: int,
    drift_patch_size: int,
    drift_radii: Sequence[float],
    drift_source_negative_weight: float,
    semantic_drift_stage: str,
    focus_base: float,
    focus_et: float,
    focus_tc: float,
    focus_wt: float,
) -> dict[str, Any]:
    model.eval()
    rows = []
    for batch in loader:
        batch = to_device(batch, device)
        images = batch["image"]
        target = images[:, target_index : target_index + 1]
        observed_mask = observed_mask_without_target(
            images.shape[0],
            images.shape[1],
            target_index,
            images.device,
        ).to(images.dtype)
        generated = model(images, observed_mask, target_index)
        synthetic = generated["synthetic"]
        uncertainty = generated["uncertainty"]
        error = (synthetic - target).abs()
        focus = region_focus(
            batch["target"],
            base=focus_base,
            et_weight=focus_et,
            tc_weight=focus_tc,
            wt_weight=focus_wt,
        )
        mse = (synthetic - target).square().mean().clamp_min(1e-8)
        psnr = 20.0 * math.log10(2.0) - 10.0 * math.log10(float(mse.cpu().item()))
        drift, _ = modality_drift_loss(
            synthetic,
            target,
            images,
            observed_mask,
            target_index,
            focus,
            drift_patch_size,
            drift_radii,
            drift_source_negative_weight,
        )
        semantic_drift, _ = semantic_feature_drift_loss(
            semantic_encoder,
            synthetic,
            target,
            focus,
            semantic_drift_stage,
            drift_radii,
        )
        region_mae: dict[str, float] = {}
        for index, region in enumerate(BRATS_REGIONS):
            mask = batch["target"][:, index : index + 1] > 0.5
            if mask.any():
                region_mae[f"mae_{region}"] = float(error[mask].mean().cpu().item())
            else:
                region_mae[f"mae_{region}"] = float("nan")
        rows.append(
            {
                "mae": float(error.mean().cpu().item()),
                "weighted_mae": float(weighted_mean(error, focus).cpu().item()),
                "mse": float(mse.cpu().item()),
                "psnr": psnr,
                "uncertainty_mean": float(uncertainty.mean().cpu().item()),
                "confidence_mean": float(generated["confidence"].mean().cpu().item()),
                "uncertainty_error_corr": pearson_correlation(uncertainty, error),
                "drift_loss": float(drift.cpu().item()),
                "semantic_drift_loss": float(semantic_drift.cpu().item()),
                **region_mae,
            }
        )

    def mean_key(key: str) -> float:
        values = [row[key] for row in rows if math.isfinite(row[key])]
        return float(np.mean(values)) if values else float("nan")

    keys = list(rows[0]) if rows else []
    return {key: mean_key(key) for key in keys}


def train_one_epoch(
    model: VirtualModalityGenerator,
    semantic_encoder: BrainMVPEncoder | None,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    target_index: int,
    args: argparse.Namespace,
    epoch: int,
) -> dict[str, Any]:
    model.train()
    losses = []
    started = time.perf_counter()
    last_log = started
    for step, batch in enumerate(loader):
        if args.max_train_steps > 0 and step >= args.max_train_steps:
            break
        batch = to_device(batch, device)
        loss, parts = loss_for_batch(
            model,
            semantic_encoder,
            batch,
            target_index,
            args.drift_patch_size,
            args.drift_radii,
            args.drift_source_negative_weight,
            args.semantic_drift_weight,
            args.semantic_drift_stage,
            args.focus_base,
            args.focus_et,
            args.focus_tc,
            args.focus_wt,
            args.recon_weight,
            args.nll_weight,
            args.gradient_weight,
            args.calibration_weight,
            args.drift_weight,
        )
        if not torch.isfinite(loss).item():
            raise RuntimeError(f"Non-finite generator loss at step {step}: {parts}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(grad_norm).item():
            raise RuntimeError(f"Non-finite grad norm at step {step}")
        optimizer.step()
        parts["grad_norm"] = float(grad_norm.detach().cpu().item())
        losses.append(parts)
        if args.log_every > 0 and (step + 1) % args.log_every == 0:
            now = time.perf_counter()
            print(
                json.dumps(
                    {
                        "epoch": epoch,
                        "step": step + 1,
                        **parts,
                        "elapsed_sec": now - started,
                        "sec_since_last_log": now - last_log,
                    }
                ),
                flush=True,
            )
            last_log = now
    if not losses:
        raise RuntimeError("No training steps were executed.")
    return {
        "epoch": epoch,
        "steps": len(losses),
        "runtime_sec": time.perf_counter() - started,
        **{
            f"mean_{key}": float(np.mean([row[key] for row in losses]))
            for key in losses[0]
        },
        "first_loss": losses[0],
        "last_loss": losses[-1],
    }


def main() -> int:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false.")
    set_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    target_index = BRATS_MODALITIES.index(args.target_modality)
    max_subjects = None if args.max_subjects <= 0 else args.max_subjects
    dataset = BraTSFusionDataset(
        manifest_csv=args.manifest,
        spatial_size=args.spatial_size,
        max_subjects=max_subjects,
        modality_mask_mode="all",
        foreground_crop_prob=args.foreground_crop_prob,
        crop_mode=args.crop_mode,
        seed=args.seed,
    )
    val_count = min(args.val_subjects, max(1, len(dataset) // 4))
    train_count = len(dataset) - val_count
    if train_count <= 0:
        raise ValueError("Need at least one training subject after val split.")
    indices = list(range(len(dataset)))
    train_set = Subset(dataset, indices[:train_count])
    val_set = Subset(dataset, indices[train_count:])
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )
    model = VirtualModalityGenerator(
        VirtualModalityGeneratorConfig(hidden_channels=args.hidden_channels)
    ).to(args.device)
    semantic_encoder = None
    if args.semantic_drift_weight > 0:
        semantic_encoder = BrainMVPEncoder(
            checkpoint_path=args.brainmvp_checkpoint,
            freeze="freeze_all",
        ).to(args.device)
        semantic_encoder.eval()
        for param in semantic_encoder.parameters():
            param.requires_grad = False
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    started = time.perf_counter()
    epoch_reports = []
    eval_reports = []
    for epoch in range(args.epochs):
        dataset.set_epoch(epoch)
        train_report = train_one_epoch(
            model,
            semantic_encoder,
            train_loader,
            optimizer,
            args.device,
            target_index,
            args,
            epoch,
        )
        eval_report = evaluate(
            model,
            semantic_encoder,
            val_loader,
            args.device,
            target_index,
            args.drift_patch_size,
            args.drift_radii,
            args.drift_source_negative_weight,
            args.semantic_drift_stage,
            args.focus_base,
            args.focus_et,
            args.focus_tc,
            args.focus_wt,
        )
        epoch_reports.append(train_report)
        eval_reports.append({"epoch": epoch, "metrics": eval_report})
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "train_loss": train_report["mean_loss"],
                    "val_mae": eval_report["mae"],
                    "val_weighted_mae": eval_report["weighted_mae"],
                    "val_psnr": eval_report["psnr"],
                    "val_uncertainty_error_corr": eval_report[
                        "uncertainty_error_corr"
                    ],
                    "val_drift_loss": eval_report["drift_loss"],
                },
                indent=2,
            ),
            flush=True,
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "virtual_modality_generator_last.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "config": vars(args),
            "modalities": BRATS_MODALITIES,
            "target_modality": args.target_modality,
        },
        checkpoint_path,
    )
    memory = None
    if args.device.startswith("cuda") and torch.cuda.is_available():
        memory = {
            "peak_allocated_mb": torch.cuda.max_memory_allocated() / (1024**2),
            "peak_reserved_mb": torch.cuda.max_memory_reserved() / (1024**2),
        }
    report = {
        "verdict": "VIRTUAL MODALITY DRIFTING: PASS",
        "git_commit": git_commit(),
        "environment": environment(args.device),
        "config": vars(args),
        "target_modality": args.target_modality,
        "observed_modalities": [
            name for name in BRATS_MODALITIES if name != args.target_modality
        ],
        "train_subjects": train_count,
        "val_subjects": val_count,
        "epochs": epoch_reports,
        "eval_by_epoch": eval_reports,
        "final_eval": eval_reports[-1]["metrics"],
        "checkpoint_path": str(checkpoint_path),
        "memory": memory,
        "runtime_sec": time.perf_counter() - started,
    }
    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
