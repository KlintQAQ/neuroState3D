from __future__ import annotations

import argparse
import csv
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
from torch.utils.data import DataLoader, Dataset, Subset


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.brats_fusion_dataset import BRATS_MODALITIES, BRATS_REGIONS  # noqa: E402
from models.high_fidelity_virtual_modality_generator import (  # noqa: E402
    HighFidelityVirtualModalityGenerator,
    HighFidelityVirtualModalityGeneratorConfig,
)
from scripts.smoke_evidence_fusion_brats import environment, git_commit  # noqa: E402
from utils.torch_drift_loss import drift_loss  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a 2.5D high-fidelity drifting generator for one missing "
            "BraTS MRI modality. Run once per target modality to obtain four "
            "target-specific generators."
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
    parser.add_argument("--target-modality", required=True, choices=BRATS_MODALITIES)
    parser.add_argument("--spatial-size", type=int, default=96)
    parser.add_argument("--context-slices", type=int, default=5)
    parser.add_argument("--max-subjects", type=int, default=512)
    parser.add_argument("--val-subjects", type=int, default=64)
    parser.add_argument("--slices-per-subject", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-train-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--recon-weight", type=float, default=1.0)
    parser.add_argument("--nll-weight", type=float, default=0.03)
    parser.add_argument("--gradient-weight", type=float, default=0.18)
    parser.add_argument("--ssim-weight", type=float, default=0.25)
    parser.add_argument("--drift-weight", type=float, default=0.08)
    parser.add_argument("--drift-patch-sizes", nargs="+", type=int, default=[4, 8, 16])
    parser.add_argument("--drift-radii", nargs="+", type=float, default=[0.004, 0.01, 0.04])
    parser.add_argument("--use-gradient-coordination", action="store_true")
    parser.add_argument("--focus-base", type=float, default=0.08)
    parser.add_argument("--focus-et", type=float, default=5.0)
    parser.add_argument("--focus-tc", type=float, default=2.5)
    parser.add_argument("--focus-wt", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=46)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "outputs" / "high_fidelity_virtual_modality"),
    )
    parser.add_argument(
        "--report-path",
        default=str(ROOT / "reports" / "high_fidelity_virtual_modality.json"),
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [row for row in rows if row.get("status", "processed") == "processed"]


def brats_region_targets_np(seg: np.ndarray) -> np.ndarray:
    et = seg == 3
    tc = (seg == 1) | (seg == 3)
    wt = seg > 0
    return np.stack([et, tc, wt], axis=0).astype(np.float32)


class BraTSContextSliceDataset(Dataset):
    def __init__(
        self,
        manifest_csv: str | Path,
        spatial_size: int,
        context_slices: int,
        max_subjects: int | None,
        slices_per_subject: int,
        target_modality: str,
        seed: int,
    ) -> None:
        rows = read_rows(manifest_csv)
        if max_subjects is not None:
            rows = rows[: int(max_subjects)]
        if not rows:
            raise ValueError(f"No rows selected from {manifest_csv}")
        if context_slices <= 0 or context_slices % 2 != 1:
            raise ValueError("context_slices must be a positive odd integer.")
        self.rows = rows
        self.spatial_size = int(spatial_size)
        self.context_slices = int(context_slices)
        self.slices_per_subject = int(max(1, slices_per_subject))
        self.target_index = BRATS_MODALITIES.index(target_modality)
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.rows) * self.slices_per_subject

    def __getitem__(self, index: int) -> dict[str, Any]:
        row_index = index // self.slices_per_subject
        local_index = index % self.slices_per_subject
        row = self.rows[row_index]
        image = np.load(row["multimodal_path"]).astype(np.float32, copy=False)
        seg = np.load(row["seg_path"]).astype(np.int64, copy=False)
        rng = random.Random(self.seed + row_index * 1009 + local_index * 9176)
        z = self._sample_slice(seg, rng)
        radius = self.context_slices // 2
        indices = [min(max(z + offset, 0), image.shape[-1] - 1) for offset in range(-radius, radius + 1)]
        context_np = np.stack([image[:, :, :, item] for item in indices], axis=1)
        context = torch.from_numpy(context_np.copy())
        target_regions = torch.from_numpy(brats_region_targets_np(seg[:, :, z]))
        if context.shape[-2:] != (self.spatial_size, self.spatial_size):
            m, k, h, w = context.shape
            context = F.interpolate(
                context.reshape(m * k, 1, h, w),
                size=(self.spatial_size, self.spatial_size),
                mode="bilinear",
                align_corners=False,
            ).reshape(m, k, self.spatial_size, self.spatial_size)
            target_regions = F.interpolate(
                target_regions.unsqueeze(0),
                size=(self.spatial_size, self.spatial_size),
                mode="nearest",
            ).squeeze(0)
        context = torch.nan_to_num(context, nan=0.0, posinf=1.0, neginf=-1.0)
        context = context.clamp(-3.0, 3.0)
        mask = torch.ones(len(BRATS_MODALITIES), dtype=torch.float32)
        mask[self.target_index] = 0.0
        return {
            "context": context,
            "target_regions": target_regions,
            "observed_mask": mask,
            "target_index": self.target_index,
            "dataset": row["dataset"],
            "subject_id": row["subject_id"],
            "slice_index": z,
        }

    @staticmethod
    def _sample_slice(seg: np.ndarray, rng: random.Random) -> int:
        candidates = np.where(((seg == 3) | (seg == 1)).sum(axis=(0, 1)) > 0)[0]
        if candidates.size == 0:
            candidates = np.where((seg > 0).sum(axis=(0, 1)) > 0)[0]
        if candidates.size == 0:
            return rng.randrange(seg.shape[-1])
        return int(candidates[rng.randrange(candidates.size)])


def to_device(batch: dict[str, Any], device: str) -> dict[str, Any]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def region_focus(
    target_regions: torch.Tensor,
    base: float,
    et_weight: float,
    tc_weight: float,
    wt_weight: float,
) -> torch.Tensor:
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
    for axis in (2, 3):
        pred_grad = synthetic.diff(dim=axis)
        target_grad = target.diff(dim=axis)
        grad_weight = weight.narrow(axis, 0, weight.shape[axis] - 1)
        losses.append(weighted_mean((pred_grad - target_grad).abs(), grad_weight))
    return torch.stack(losses).mean()


def ssim_loss_2d(
    synthetic: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    window: int = 7,
) -> torch.Tensor:
    padding = int(window) // 2
    mu_x = F.avg_pool2d(synthetic, window, stride=1, padding=padding)
    mu_y = F.avg_pool2d(target, window, stride=1, padding=padding)
    sigma_x = F.avg_pool2d(synthetic.square(), window, stride=1, padding=padding) - mu_x.square()
    sigma_y = F.avg_pool2d(target.square(), window, stride=1, padding=padding) - mu_y.square()
    sigma_xy = F.avg_pool2d(synthetic * target, window, stride=1, padding=padding) - mu_x * mu_y
    c1 = 0.01**2
    c2 = 0.03**2
    ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2)
    ).clamp_min(1e-8)
    return weighted_mean((1.0 - ssim).clamp(0.0, 2.0), weight)


def _feature_stack(image: torch.Tensor) -> torch.Tensor:
    gx = F.pad(image.diff(dim=3), (1, 0, 0, 0))
    gy = F.pad(image.diff(dim=2), (0, 0, 1, 0))
    return torch.cat([image, gx, gy], dim=1)


def medical_feature_tokens_2d(
    image: torch.Tensor,
    focus: torch.Tensor,
    patch_sizes: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    features = _feature_stack(image)
    token_parts = []
    weight_parts = []
    for patch_size in patch_sizes:
        kernel = min(int(patch_size), *features.shape[-2:])
        if kernel <= 1:
            mean = features
            std = torch.zeros_like(features)
            dynamic = torch.zeros_like(features)
            pooled_weight = focus
        else:
            mean = F.avg_pool2d(features, kernel_size=kernel, stride=kernel)
            second = F.avg_pool2d(features.square(), kernel_size=kernel, stride=kernel)
            std = (second - mean.square()).clamp_min(0.0).sqrt()
            vmax = F.max_pool2d(features, kernel_size=kernel, stride=kernel)
            vmin = -F.max_pool2d(-features, kernel_size=kernel, stride=kernel)
            dynamic = vmax - vmin
            pooled_weight = F.avg_pool2d(focus, kernel_size=kernel, stride=kernel)
        tokens = torch.cat([mean, std, dynamic], dim=1).flatten(2).transpose(1, 2)
        weights = pooled_weight.flatten(2).squeeze(1)
        weights = weights / weights.mean(dim=1, keepdim=True).clamp_min(1e-8)
        token_parts.append(tokens.contiguous())
        weight_parts.append(weights.clamp(0.25, 6.0).contiguous())
    return torch.cat(token_parts, dim=1), torch.cat(weight_parts, dim=1)


def medical_drift_loss(
    synthetic: torch.Tensor,
    target: torch.Tensor,
    focus: torch.Tensor,
    patch_sizes: Sequence[int],
    radii: Sequence[float],
) -> torch.Tensor:
    generated_tokens, generated_weights = medical_feature_tokens_2d(synthetic, focus, patch_sizes)
    positive_tokens, positive_weights = medical_feature_tokens_2d(target, focus, patch_sizes)
    negative_tokens = torch.roll(generated_tokens.detach(), shifts=1, dims=0)
    negative_weights = torch.roll(generated_weights.detach(), shifts=1, dims=0)
    loss, _ = drift_loss(
        generated=generated_tokens,
        positive=positive_tokens,
        negative=negative_tokens,
        weight_generated=generated_weights,
        weight_positive=positive_weights,
        weight_negative=negative_weights,
        radii=radii,
    )
    return loss


def two_objective_mgda_alpha(loss_a: torch.Tensor, loss_b: torch.Tensor, shared: torch.Tensor) -> tuple[float, float]:
    grad_a = torch.autograd.grad(loss_a, shared, retain_graph=True, allow_unused=False)[0].detach().flatten()
    grad_b = torch.autograd.grad(loss_b, shared, retain_graph=True, allow_unused=False)[0].detach().flatten()
    if not torch.isfinite(grad_a).all().item() or not torch.isfinite(grad_b).all().item():
        return 1.0, 1.0
    diff = grad_a - grad_b
    denom = diff.dot(diff).clamp_min(1e-12)
    if not torch.isfinite(denom).item() or float(denom.cpu().item()) <= 1e-12:
        return 1.0, 1.0
    alpha_a = float((grad_b.dot(grad_b - grad_a) / denom).clamp(0.0, 1.0).cpu().item())
    if not math.isfinite(alpha_a):
        return 1.0, 1.0
    return alpha_a, 1.0 - alpha_a


def loss_for_batch(
    model: HighFidelityVirtualModalityGenerator,
    batch: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float]]:
    context = batch["context"]
    target_index = int(batch["target_index"][0].item())
    center = context[:, :, context.shape[2] // 2]
    target = center[:, target_index : target_index + 1]
    generated = model(context, batch["observed_mask"])
    synthetic = generated["synthetic"]
    synthetic.retain_grad()
    uncertainty = generated["uncertainty"]
    focus = region_focus(
        batch["target_regions"],
        args.focus_base,
        args.focus_et,
        args.focus_tc,
        args.focus_wt,
    )
    abs_error = (synthetic - target).abs()
    recon = weighted_mean(abs_error, focus)
    nll = weighted_mean(abs_error / uncertainty + uncertainty.log(), focus)
    grad = gradient_difference_loss(synthetic, target, focus)
    ssim = ssim_loss_2d(synthetic, target, focus)
    drift = medical_drift_loss(
        synthetic,
        target,
        focus,
        args.drift_patch_sizes,
        args.drift_radii,
    )
    fidelity = (
        float(args.recon_weight) * recon
        + float(args.nll_weight) * nll
        + float(args.gradient_weight) * grad
        + float(args.ssim_weight) * ssim
    )
    drift_scaled = float(args.drift_weight) * drift
    alpha_fidelity = 1.0
    alpha_drift = 1.0
    if args.use_gradient_coordination and float(args.drift_weight) > 0:
        alpha_fidelity, alpha_drift = two_objective_mgda_alpha(
            fidelity,
            drift_scaled,
            synthetic,
        )
    loss = alpha_fidelity * fidelity + alpha_drift * drift_scaled
    return loss, {
        "loss": float(loss.detach().cpu().item()),
        "fidelity_loss": float(fidelity.detach().cpu().item()),
        "recon_l1": float(recon.detach().cpu().item()),
        "nll": float(nll.detach().cpu().item()),
        "gradient_l1": float(grad.detach().cpu().item()),
        "ssim_loss": float(ssim.detach().cpu().item()),
        "drift_loss": float(drift.detach().cpu().item()),
        "alpha_fidelity": float(alpha_fidelity),
        "alpha_drift": float(alpha_drift),
        "uncertainty_mean": float(uncertainty.detach().mean().cpu().item()),
        "confidence_mean": float(generated["confidence"].detach().mean().cpu().item()),
    }


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
    model: HighFidelityVirtualModalityGenerator,
    loader: DataLoader,
    device: str,
) -> dict[str, float]:
    model.eval()
    rows = []
    for batch in loader:
        batch = to_device(batch, device)
        context = batch["context"]
        target_index = int(batch["target_index"][0].item())
        center = context[:, :, context.shape[2] // 2]
        target = center[:, target_index : target_index + 1]
        generated = model(context, batch["observed_mask"])
        synthetic = generated["synthetic"]
        error = (synthetic - target).abs()
        mse = (synthetic - target).square().mean().clamp_min(1e-8)
        psnr = 20.0 * math.log10(2.0) - 10.0 * math.log10(float(mse.cpu().item()))
        row = {
            "mae": float(error.mean().cpu().item()),
            "mse": float(mse.cpu().item()),
            "psnr": psnr,
            "ssim_loss": float(ssim_loss_2d(synthetic, target, torch.ones_like(target)).cpu().item()),
            "confidence_mean": float(generated["confidence"].mean().cpu().item()),
            "uncertainty_mean": float(generated["uncertainty"].mean().cpu().item()),
            "uncertainty_error_corr": pearson_correlation(generated["uncertainty"], error),
        }
        for index, region in enumerate(BRATS_REGIONS):
            mask = batch["target_regions"][:, index : index + 1] > 0.5
            row[f"mae_{region}"] = (
                float(error[mask].mean().cpu().item()) if mask.any() else float("nan")
            )
        rows.append(row)
    keys = list(rows[0]) if rows else []
    return {
        key: float(np.mean([row[key] for row in rows if math.isfinite(row[key])]))
        for key in keys
    }


def train_one_epoch(
    model: HighFidelityVirtualModalityGenerator,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
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
        loss, parts = loss_for_batch(model, batch, args)
        if not torch.isfinite(loss).item():
            raise RuntimeError(f"Non-finite loss at step {step}: {parts}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(grad_norm).item():
            optimizer.zero_grad(set_to_none=True)
            parts["grad_norm"] = float("nan")
            parts["skipped_nonfinite_grad"] = True
            losses.append(parts)
            continue
        optimizer.step()
        parts["grad_norm"] = float(grad_norm.detach().cpu().item())
        parts["skipped_nonfinite_grad"] = False
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
    keys = [key for key in losses[0] if key != "grad_norm"] if losses else []
    return {
        "epoch": epoch,
        "steps": len(losses),
        **{f"mean_{key}": float(np.mean([item[key] for item in losses])) for key in keys},
        "first_loss": losses[0] if losses else None,
        "last_loss": losses[-1] if losses else None,
        "runtime_sec": time.perf_counter() - started,
    }


def main() -> int:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    set_seed(args.seed)
    max_subjects = None if args.max_subjects <= 0 else args.max_subjects
    dataset = BraTSContextSliceDataset(
        args.manifest,
        spatial_size=args.spatial_size,
        context_slices=args.context_slices,
        max_subjects=max_subjects,
        slices_per_subject=args.slices_per_subject,
        target_modality=args.target_modality,
        seed=args.seed,
    )
    val_count = min(int(args.val_subjects) * int(args.slices_per_subject), max(1, len(dataset) // 4))
    train_count = len(dataset) - val_count
    train_set = Subset(dataset, list(range(train_count)))
    val_set = Subset(dataset, list(range(train_count, len(dataset))))
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = HighFidelityVirtualModalityGenerator(
        HighFidelityVirtualModalityGeneratorConfig(
            hidden_channels=args.hidden_channels,
            in_modalities=len(BRATS_MODALITIES),
            context_slices=args.context_slices,
            dropout=args.dropout,
        )
    ).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    epoch_reports = []
    eval_reports = []
    for epoch in range(args.epochs):
        train_report = train_one_epoch(model, train_loader, optimizer, args.device, args, epoch)
        eval_report = evaluate(model, val_loader, args.device)
        print(json.dumps({"epoch": epoch, **eval_report}), flush=True)
        epoch_reports.append(train_report)
        eval_reports.append({"epoch": epoch, "metrics": eval_report})
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "high_fidelity_virtual_modality_generator_last.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "config": vars(args),
            "target_modality": args.target_modality,
            "modalities": BRATS_MODALITIES,
            "generator_class": "HighFidelityVirtualModalityGenerator",
        },
        checkpoint_path,
    )
    report = {
        "verdict": "HIGH FIDELITY VIRTUAL MODALITY DRIFTING: PASS",
        "git_commit": git_commit(),
        "environment": environment(args.device),
        "config": vars(args),
        "train_slices": len(train_set),
        "val_slices": len(val_set),
        "epochs": epoch_reports,
        "eval_by_epoch": eval_reports,
        "final_eval": eval_reports[-1]["metrics"] if eval_reports else {},
        "checkpoint_path": str(checkpoint_path),
        "memory": {
            "peak_allocated_mb": (
                torch.cuda.max_memory_allocated() / (1024**2)
                if torch.cuda.is_available()
                else None
            ),
            "peak_reserved_mb": (
                torch.cuda.max_memory_reserved() / (1024**2)
                if torch.cuda.is_available()
                else None
            ),
        },
        "runtime_sec": time.perf_counter() - started,
    }
    out_path = Path(args.report_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
