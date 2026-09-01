from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
from models.prompted_slice_virtual_modality_generator import (  # noqa: E402
    PromptedSliceVirtualModalityGenerator,
    PromptedSliceVirtualModalityGeneratorConfig,
)
from models.slice_drift_transport_generator import (  # noqa: E402
    SliceDriftTransportGenerator,
    SliceDriftTransportGeneratorConfig,
)

BRATS_DATASETS = ("GLI", "MEN", "PED")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize a 2D slice generator for missing BraTS MRI modalities."
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
    parser.add_argument("--target-modality", default="", choices=("", *BRATS_MODALITIES))
    parser.add_argument("--spatial-size", type=int, default=128)
    parser.add_argument("--num-cases", type=int, default=6)
    parser.add_argument(
        "--case-selection",
        default="representative",
        choices=("representative", "best", "worst"),
    )
    parser.add_argument("--selection-pool", type=int, default=120)
    parser.add_argument("--batch-slices", type=int, default=16)
    parser.add_argument("--slice-crop-size", type=int, default=0)
    parser.add_argument("--apply-brain-mask", action="store_true")
    parser.add_argument("--support-threshold", type=float, default=1e-5)
    parser.add_argument("--support-dilation", type=int, default=2)
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "reports" / "visuals" / "slice_virtual_modality"),
    )
    return parser.parse_args()


def read_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return [row for row in csv.DictReader(handle) if row.get("status", "processed") == "processed"]


def load_model(
    checkpoint_path: str | Path,
    device: str,
    override_target: str,
) -> tuple[
    SliceVirtualModalityGenerator | PromptedSliceVirtualModalityGenerator | SliceDriftTransportGenerator,
    str,
]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = payload.get("config", {})
    target = override_target or str(
        payload.get("target_modality", config.get("target_modality", "t1c"))
    )
    model_kind = str(config.get("model_kind", "slice"))
    context_depth = 2 * int(config.get("slice_context_radius", 0)) + 1
    in_modalities = int(config.get("in_modalities", len(BRATS_MODALITIES) * context_depth))
    if model_kind == "prompted":
        model = PromptedSliceVirtualModalityGenerator(
            PromptedSliceVirtualModalityGeneratorConfig(
                in_modalities=in_modalities,
                hidden_channels=int(config.get("hidden_channels", 32)),
                prompt_channels=3,
                residual_scale=float(config.get("residual_scale", 0.35)),
                lesion_residual_scale=float(config.get("lesion_residual_scale", 0.45)),
                detail_residual_scale=float(config.get("detail_residual_scale", 0.20)),
                enhancement_residual_scale=float(config.get("enhancement_residual_scale", 0.0)),
                class_channels=len(BRATS_DATASETS),
                class_conditioned=bool(config.get("class_conditioned", False)),
                output_activation=str(config.get("output_activation", "tanh")),
                positive_lesion_residual=bool(config.get("positive_lesion_residual", False)),
            )
        )
    elif model_kind == "transport":
        model = SliceDriftTransportGenerator(
            SliceDriftTransportGeneratorConfig(
                in_modalities=in_modalities,
                hidden_channels=int(config.get("hidden_channels", 32)),
                transport_steps=int(config.get("transport_steps", 4)),
                transport_step_scale=float(config.get("transport_step_scale", 1.0)),
                velocity_scale=float(config.get("transport_velocity_scale", 1.0)),
                init_blur_kernel=int(config.get("transport_init_blur_kernel", 5)),
                class_channels=len(BRATS_DATASETS),
                class_conditioned=bool(config.get("class_conditioned", False)),
                output_activation=str(config.get("output_activation", "hardtanh")),
                gated_refinement=bool(config.get("gated_refinement", False)),
                refinement_residual_scale=float(config.get("refinement_residual_scale", 0.25)),
                gate_bias_init=float(config.get("gate_bias_init", -3.0)),
                refinement_acceptance_gate=bool(config.get("refinement_acceptance_gate", False)),
                accept_bias_init=float(config.get("accept_bias_init", -2.0)),
                refinement_channels_multiplier=int(
                    config.get("refinement_channels_multiplier", 1)
                ),
                refinement_blocks=int(config.get("refinement_blocks", 1)),
                refinement_detail_features=bool(
                    config.get("refinement_detail_features", False)
                ),
            )
        )
    else:
        model = SliceVirtualModalityGenerator(
            SliceVirtualModalityGeneratorConfig(
                in_modalities=in_modalities,
                hidden_channels=int(config.get("hidden_channels", 32)),
            )
        )
    model.load_state_dict(payload["model"], strict=False)
    model.to(device).eval()
    return model, target


def context_slice_tensor(
    image_np: np.ndarray,
    z: int,
    radius: int,
) -> torch.Tensor:
    if radius <= 0:
        return torch.from_numpy(image_np[:, :, :, z].copy())
    depth = image_np.shape[-1]
    slices = []
    for modality_index in range(image_np.shape[0]):
        for offset in range(-radius, radius + 1):
            zz = max(0, min(depth - 1, int(z) + offset))
            slices.append(image_np[modality_index, :, :, zz])
    return torch.from_numpy(np.stack(slices, axis=0).copy())


def context_slice_batch(
    image: torch.Tensor,
    start: int,
    end: int,
    radius: int,
) -> torch.Tensor:
    if radius <= 0:
        return image[:, :, :, start:end].permute(3, 0, 1, 2).contiguous()
    depth = image.shape[-1]
    batch_slices = []
    for z in range(int(start), int(end)):
        channels = []
        for modality_index in range(image.shape[0]):
            for offset in range(-int(radius), int(radius) + 1):
                zz = max(0, min(depth - 1, z + offset))
                channels.append(image[modality_index, :, :, zz])
        batch_slices.append(torch.stack(channels, dim=0))
    return torch.stack(batch_slices, dim=0)


def resize_image(image: torch.Tensor, spatial_size: int) -> torch.Tensor:
    if spatial_size <= 0 or tuple(image.shape[-3:]) == (spatial_size,) * 3:
        return image
    return F.interpolate(
        image.unsqueeze(0),
        size=(spatial_size, spatial_size, spatial_size),
        mode="trilinear",
        align_corners=False,
    ).squeeze(0)


def resize_seg(seg: torch.Tensor, spatial_size: int) -> torch.Tensor:
    if spatial_size <= 0 or tuple(seg.shape[-3:]) == (spatial_size,) * 3:
        return seg
    return F.interpolate(
        seg.unsqueeze(0).unsqueeze(0).float(),
        size=(spatial_size, spatial_size, spatial_size),
        mode="nearest",
    ).squeeze(0).squeeze(0).long()


def observed_brain_support_np(
    image: np.ndarray,
    target_index: int,
    threshold: float,
    dilation: int,
) -> np.ndarray:
    observed_indices = [index for index in range(image.shape[0]) if index != target_index]
    observed = np.max(np.abs(image[observed_indices]), axis=0)
    support = torch.from_numpy((observed > float(threshold)).astype(np.float32))
    if dilation > 0:
        support = F.max_pool3d(
            support.unsqueeze(0).unsqueeze(0),
            kernel_size=int(dilation) * 2 + 1,
            stride=1,
            padding=int(dilation),
        ).squeeze(0).squeeze(0)
    return support.numpy() > 0.5


@torch.no_grad()
def generate_volume(
    model: SliceVirtualModalityGenerator | PromptedSliceVirtualModalityGenerator | SliceDriftTransportGenerator,
    image: torch.Tensor,
    target_index: int,
    batch_slices: int,
    dataset_name: str = "",
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray,
]:
    device = next(model.parameters()).device
    image = torch.nan_to_num(image.to(device), nan=0.0, posinf=1.0, neginf=-1.0).clamp(-3.0, 3.0)
    in_modalities = int(model.config.in_modalities)
    context_depth = max(1, in_modalities // len(BRATS_MODALITIES))
    context_radius = context_depth // 2
    mask = torch.ones(1, in_modalities, device=device)
    target_start = int(target_index) * context_depth
    mask[:, target_start : target_start + context_depth] = 0.0
    class_condition = class_condition_from_name(dataset_name, device)
    _, _, _, width = image.shape
    synthetic_slices = []
    confidence_slices = []
    uncertainty_slices = []
    prompt_slices = []
    gate_slices = []
    acceptance_slices = []
    stage1_slices = []
    for start in range(0, width, int(batch_slices)):
        end = min(start + int(batch_slices), width)
        slices = context_slice_batch(image, start, end, context_radius)
        batch_mask = mask.expand(slices.shape[0], -1)
        batch_class = (
            class_condition.expand(slices.shape[0], -1)
            if class_condition is not None
            else None
        )
        output = model(slices, batch_mask, batch_class)
        synthetic_slices.append(output["synthetic"].squeeze(1).cpu())
        confidence_slices.append(output["confidence"].squeeze(1).cpu())
        uncertainty_slices.append(output["uncertainty"].squeeze(1).cpu())
        stage1_slices.append(output.get("stage1_synthetic", output["synthetic"]).squeeze(1).cpu())
        if "prompt_probs" in output:
            prompt_slices.append(output["prompt_probs"].cpu())
        if "refinement_gate" in output:
            gate_slices.append(output["refinement_gate"].squeeze(1).cpu())
        if "refinement_acceptance" in output:
            acceptance_slices.append(output["refinement_acceptance"].squeeze(1).cpu())
    synthetic = torch.cat(synthetic_slices, dim=0).numpy().transpose(1, 2, 0)
    confidence = torch.cat(confidence_slices, dim=0).numpy().transpose(1, 2, 0)
    uncertainty = torch.cat(uncertainty_slices, dim=0).numpy().transpose(1, 2, 0)
    prompt = None
    if prompt_slices:
        prompt = torch.cat(prompt_slices, dim=0).numpy().transpose(1, 2, 3, 0)
    gate = None
    if gate_slices:
        gate = torch.cat(gate_slices, dim=0).numpy().transpose(1, 2, 0)
    acceptance = None
    if acceptance_slices:
        acceptance = torch.cat(acceptance_slices, dim=0).numpy().transpose(1, 2, 0)
    stage1 = torch.cat(stage1_slices, dim=0).numpy().transpose(1, 2, 0)
    return synthetic, confidence, uncertainty, prompt, gate, acceptance, stage1


def robust_limits(array: np.ndarray) -> tuple[float, float]:
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(finite, [1, 99])
    if hi <= lo:
        hi = lo + 1e-6
    return float(lo), float(hi)


def selected_slice(seg: np.ndarray, target: np.ndarray) -> int:
    et = (seg == 3).sum(axis=(0, 1))
    if et.max() > 0:
        return int(et.argmax())
    foreground = (seg > 0).sum(axis=(0, 1))
    if foreground.max() > 0:
        return int(foreground.argmax())
    signal = np.abs(target).sum(axis=(0, 1))
    return int(signal.argmax()) if signal.max() > 0 else int(seg.shape[-1] // 2)


def brats_region_targets_np(seg: np.ndarray) -> np.ndarray:
    et = seg == 3
    tc = (seg == 1) | (seg == 3)
    wt = seg > 0
    return np.stack([et, tc, wt], axis=0).astype(np.float32)


def crop_slice_around_region_np(
    image_slice: np.ndarray,
    seg_slice: np.ndarray,
    crop_size: int,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int]]:
    _, height, width = image_slice.shape
    crop = int(max(1, min(crop_size, height, width)))
    if crop >= height and crop >= width:
        return image_slice, seg_slice, (0, 0, crop)

    coords = None
    for mask in (seg_slice == 3, (seg_slice == 1) | (seg_slice == 3), seg_slice > 0):
        found = np.argwhere(mask)
        if found.size > 0:
            coords = found
            break
    if coords is None:
        found = np.argwhere(np.max(np.abs(image_slice), axis=0) > 1e-5)
        coords = found if found.size > 0 else np.array([[height // 2, width // 2]])

    center = coords[len(coords) // 2]
    cy, cx = int(center[0]), int(center[1])
    top = max(0, min(height - crop, cy - crop // 2))
    left = max(0, min(width - crop, cx - crop // 2))
    return (
        image_slice[:, top : top + crop, left : left + crop],
        seg_slice[top : top + crop, left : left + crop],
        (top, left, crop),
    )


def class_condition_from_name(dataset_name: str, device: torch.device) -> torch.Tensor | None:
    if dataset_name not in BRATS_DATASETS:
        return None
    condition = torch.zeros(1, len(BRATS_DATASETS), device=device)
    condition[0, BRATS_DATASETS.index(dataset_name)] = 1.0
    return condition


@torch.no_grad()
def generate_slice(
    model: SliceVirtualModalityGenerator | PromptedSliceVirtualModalityGenerator | SliceDriftTransportGenerator,
    image_slice: torch.Tensor,
    base_target_index: int,
    dataset_name: str = "",
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray,
]:
    device = next(model.parameters()).device
    image_slice = torch.nan_to_num(
        image_slice.to(device),
        nan=0.0,
        posinf=1.0,
        neginf=-1.0,
    ).clamp(-3.0, 3.0)
    in_modalities = int(model.config.in_modalities)
    context_depth = max(1, in_modalities // len(BRATS_MODALITIES))
    mask = torch.ones(1, in_modalities, device=device)
    start = int(base_target_index) * context_depth
    mask[:, start : start + context_depth] = 0.0
    output = model(
        image_slice.unsqueeze(0),
        mask,
        class_condition_from_name(dataset_name, device),
    )
    synthetic = output["synthetic"].squeeze(0).squeeze(0).cpu().numpy()
    confidence = output["confidence"].squeeze(0).squeeze(0).cpu().numpy()
    uncertainty = output["uncertainty"].squeeze(0).squeeze(0).cpu().numpy()
    prompt = output.get("prompt_probs")
    prompt_np = prompt.squeeze(0).cpu().numpy() if prompt is not None else None
    gate = output.get("refinement_gate")
    gate_np = gate.squeeze(0).squeeze(0).cpu().numpy() if gate is not None else None
    acceptance = output.get("refinement_acceptance")
    acceptance_np = (
        acceptance.squeeze(0).squeeze(0).cpu().numpy() if acceptance is not None else None
    )
    stage1 = output.get("stage1_synthetic", output["synthetic"])
    stage1_np = stage1.squeeze(0).squeeze(0).cpu().numpy()
    return synthetic, confidence, uncertainty, prompt_np, gate_np, acceptance_np, stage1_np


def show_slice(
    axis: plt.Axes,
    volume: np.ndarray,
    z: int,
    title: str,
    cmap: str = "gray",
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    if vmin is None or vmax is None:
        vmin, vmax = robust_limits(volume)
    axis.imshow(volume[:, :, z].T, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    axis.set_title(title, fontsize=9)
    axis.axis("off")


def psnr(mse: float) -> float:
    return 20.0 * math.log10(2.0) - 10.0 * math.log10(max(float(mse), 1e-8))


def choose_cases(rows: list[dict[str, str]], num_cases: int) -> list[dict[str, str]]:
    limits = defaultdict(lambda: max(1, num_cases // 3))
    for name in ("GLI", "MEN", "PED"):
        limits[name] = max(1, num_cases // 3)
    counts: dict[str, int] = defaultdict(int)
    chosen = []
    probe = np.linspace(0, len(rows) - 1, num=min(len(rows), max(240, num_cases * 90)), dtype=int)
    for index in probe:
        row = rows[int(index)]
        dataset = row["dataset"]
        if counts[dataset] >= limits[dataset]:
            continue
        seg = np.load(row["seg_path"])
        if int((seg > 0).sum()) < 500:
            continue
        chosen.append(row)
        counts[dataset] += 1
        if len(chosen) >= num_cases:
            break
    for row in rows:
        if len(chosen) >= num_cases:
            break
        if row not in chosen:
            chosen.append(row)
    return chosen[:num_cases]


@torch.no_grad()
def choose_cases_by_generation_error(
    rows: list[dict[str, str]],
    model: SliceVirtualModalityGenerator | PromptedSliceVirtualModalityGenerator | SliceDriftTransportGenerator,
    target_index: int,
    spatial_size: int,
    num_cases: int,
    selection_pool: int,
    mode: str,
) -> list[dict[str, str]]:
    if mode == "representative":
        return choose_cases(rows, num_cases)
    if not rows:
        return []
    pool_size = min(len(rows), max(int(num_cases), int(selection_pool)))
    probe = np.linspace(0, len(rows) - 1, num=pool_size, dtype=int)
    in_modalities = int(model.config.in_modalities)
    context_depth = max(1, in_modalities // len(BRATS_MODALITIES))
    context_radius = context_depth // 2
    scored: list[tuple[float, dict[str, str]]] = []
    for index in probe:
        row = rows[int(index)]
        try:
            image = torch.from_numpy(np.load(row["multimodal_path"]).astype(np.float32, copy=False))
            seg = torch.from_numpy(np.load(row["seg_path"]).astype(np.int64, copy=False))
            image = resize_image(image, int(spatial_size))
            seg = resize_seg(seg, int(spatial_size))
            image_np = image.cpu().numpy()
            seg_np = seg.cpu().numpy()
            real = image_np[target_index]
            z = selected_slice(seg_np, real)
            context_tensor = context_slice_tensor(image_np, z, context_radius)
            synthetic, _, _, _, _, _, _ = generate_slice(
                model,
                context_tensor,
                target_index,
                row["dataset"],
            )
            lesion = seg_np[:, :, z] > 0
            if lesion.any():
                score = float(np.abs(synthetic - real[:, :, z])[lesion].mean())
            else:
                score = float(np.abs(synthetic - real[:, :, z]).mean())
            scored.append((score, row))
        except Exception as exc:
            print(
                json.dumps(
                    {
                        "case_selection_skip": row.get("subject_id", ""),
                        "error": type(exc).__name__ + ": " + str(exc),
                    }
                ),
                flush=True,
            )
    reverse = mode == "worst"
    return [row for _, row in sorted(scored, key=lambda item: item[0], reverse=reverse)[:num_cases]]


@torch.no_grad()
def main() -> int:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    model, target_modality = load_model(args.checkpoint, args.device, args.target_modality)
    target_index = BRATS_MODALITIES.index(target_modality)
    context_depth = max(1, int(model.config.in_modalities) // len(BRATS_MODALITIES))
    context_radius = context_depth // 2
    all_rows = read_rows(args.manifest)
    rows = choose_cases_by_generation_error(
        all_rows,
        model,
        target_index,
        int(args.spatial_size),
        int(args.num_cases),
        int(args.selection_pool),
        args.case_selection,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    case_records = []
    summary_items = []
    for case_index, row in enumerate(rows):
        image = torch.from_numpy(np.load(row["multimodal_path"]).astype(np.float32, copy=False))
        seg = torch.from_numpy(np.load(row["seg_path"]).astype(np.int64, copy=False))
        image = resize_image(image, int(args.spatial_size))
        seg = resize_seg(seg, int(args.spatial_size))
        image_np = image.cpu().numpy()
        seg_np = seg.cpu().numpy()
        real = image_np[target_index]
        z = selected_slice(seg_np, real)
        crop_record = None
        if int(args.slice_crop_size) > 0:
            context_tensor = context_slice_tensor(image_np, z, context_radius)
            crop_image, crop_seg, crop_record = crop_slice_around_region_np(
                image_np[:, :, :, z],
                seg_np[:, :, z],
                int(args.slice_crop_size),
            )
            top, left, crop = crop_record
            crop_context = context_tensor[:, top : top + crop, left : left + crop]
            crop_tensor = crop_context.clone()
            display_tensor = torch.from_numpy(crop_image.copy())
            if tuple(crop_tensor.shape[-2:]) != (int(args.spatial_size), int(args.spatial_size)):
                crop_tensor = F.interpolate(
                    crop_tensor.unsqueeze(0),
                    size=(int(args.spatial_size), int(args.spatial_size)),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
                display_tensor = F.interpolate(
                    display_tensor.unsqueeze(0),
                    size=(int(args.spatial_size), int(args.spatial_size)),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
                crop_seg_tensor = F.interpolate(
                    torch.from_numpy(crop_seg).unsqueeze(0).unsqueeze(0).float(),
                    size=(int(args.spatial_size), int(args.spatial_size)),
                    mode="nearest",
                ).squeeze(0).squeeze(0).long()
                crop_seg = crop_seg_tensor.numpy()
            else:
                crop_seg = crop_seg.astype(np.int64, copy=False)
            (
                synthetic_2d,
                confidence_2d,
                uncertainty_2d,
                prompt_2d,
                gate_2d,
                acceptance_2d,
                stage1_2d,
            ) = generate_slice(
                model,
                crop_tensor,
                target_index,
                row["dataset"],
            )
            image_np = display_tensor.cpu().numpy()[:, :, :, None]
            seg_np = crop_seg[:, :, None]
            real = image_np[target_index]
            synthetic = synthetic_2d[:, :, None]
            confidence = confidence_2d[:, :, None]
            uncertainty = uncertainty_2d[:, :, None]
            prompt = prompt_2d[:, :, :, None] if prompt_2d is not None else None
            gate = gate_2d[:, :, None] if gate_2d is not None else None
            acceptance = acceptance_2d[:, :, None] if acceptance_2d is not None else None
            stage1 = stage1_2d[:, :, None]
            z = 0
        else:
            synthetic, confidence, uncertainty, prompt, gate, acceptance, stage1 = generate_volume(
                model,
                image,
                target_index,
                int(args.batch_slices),
                row["dataset"],
            )
            if args.apply_brain_mask:
                support = observed_brain_support_np(
                    image_np,
                    target_index,
                    args.support_threshold,
                    args.support_dilation,
                )
                synthetic = synthetic * support.astype(np.float32)
                confidence = confidence * support.astype(np.float32)
                stage1 = stage1 * support.astype(np.float32)
                if prompt is not None:
                    prompt = prompt * support[None].astype(np.float32)
                if gate is not None:
                    gate = gate * support.astype(np.float32)
                if acceptance is not None:
                    acceptance = acceptance * support.astype(np.float32)
            if prompt is not None:
                prompt = prompt
        error = np.abs(synthetic - real)
        stage1_error = np.abs(stage1 - real)
        refinement_delta = np.abs(synthetic - stage1)
        mse = float(np.mean((synthetic - real) ** 2))
        record = {
            "dataset": row["dataset"],
            "subject_id": row["subject_id"],
            "target_modality": target_modality,
            "slice_index": z,
            "mae": float(error.mean()),
            "mse": mse,
            "psnr": psnr(mse),
            "confidence_mean": float(confidence.mean()),
            "uncertainty_mean": float(uncertainty.mean()),
            "stage1_mae": float(stage1_error.mean()),
            "refinement_delta_mean": float(refinement_delta.mean()),
        }
        if gate is not None:
            record["gate_mean"] = float(gate.mean())
        if acceptance is not None:
            record["acceptance_mean"] = float(acceptance.mean())
        if crop_record is not None:
            record["crop_top"] = int(crop_record[0])
            record["crop_left"] = int(crop_record[1])
            record["crop_size"] = int(crop_record[2])
        for label, region_mask in {
            "ET": seg_np == 3,
            "TC": (seg_np == 1) | (seg_np == 3),
            "WT": seg_np > 0,
        }.items():
            record[f"mae_{label}"] = (
                float(error[region_mask].mean()) if region_mask.any() else None
            )
            record[f"stage1_mae_{label}"] = (
                float(stage1_error[region_mask].mean()) if region_mask.any() else None
            )
            record[f"refinement_delta_{label}"] = (
                float(refinement_delta[region_mask].mean()) if region_mask.any() else None
            )

        lo, hi = robust_limits(real)
        error_hi = float(np.percentile(error, 99))
        delta_hi = float(np.percentile(refinement_delta, 99))
        input_modalities = [
            (BRATS_MODALITIES[i].upper(), image_np[i])
            for i in range(len(BRATS_MODALITIES))
            if i != target_index
        ]
        panels: list[tuple[str, np.ndarray, str, float | None, float | None]] = [
            *[(f"{name} input", volume, "gray", None, None) for name, volume in input_modalities[:3]],
            (f"Real {target_modality.upper()}", real, "gray", lo, hi),
            (f"Stage1 {target_modality.upper()}", stage1, "gray", lo, hi),
            (f"Generated {target_modality.upper()}", synthetic, "gray", lo, hi),
            ("Abs error", error, "magma", 0.0, error_hi),
            ("Refinement delta", refinement_delta, "magma", 0.0, delta_hi),
            ("Confidence", confidence, "viridis", 0.0, 1.0),
            ("Seg label", seg_np, "tab10", 0.0, float(max(3, int(seg_np.max())))),
        ]
        if gate is not None:
            panels.append(("Refinement gate", gate, "viridis", 0.0, 1.0))
        if acceptance is not None:
            panels.append(("Acceptance gate", acceptance, "viridis", 0.0, 1.0))
        if prompt is not None:
            panels.extend(
                [
                    ("Prompt ET", prompt[0], "viridis", 0.0, 1.0),
                    ("Prompt TC", prompt[1], "viridis", 0.0, 1.0),
                    ("Prompt WT", prompt[2], "viridis", 0.0, 1.0),
                    ("Uncertainty", uncertainty, "viridis", None, None),
                ]
            )
        panel_rows = int(math.ceil(len(panels) / 4))
        fig, axes = plt.subplots(panel_rows, 4, figsize=(14, 3.4 * panel_rows), dpi=160)
        axes_array = np.asarray(axes).reshape(-1)
        for axis, (title, volume, cmap, vmin, vmax) in zip(axes_array, panels):
            show_slice(axis, volume, z, title, cmap, vmin, vmax)
        for axis in axes_array[len(panels) :]:
            axis.axis("off")
        fig.suptitle(
            f"{row['dataset']} {row['subject_id']} | missing {target_modality.upper()} "
            f"| slice {z} | MAE={record['mae']:.4f} | PSNR={record['psnr']:.2f}",
            fontsize=11,
        )
        fig.tight_layout()
        panel_path = output_dir / (
            f"case_{case_index:02d}_{row['dataset']}_{row['subject_id']}_"
            f"{target_modality}_compare.png"
        )
        fig.savefig(panel_path, bbox_inches="tight")
        plt.close(fig)
        record["panel_path"] = str(panel_path.resolve())
        case_records.append(record)
        summary_items.append((row, z, real, stage1, synthetic, error, gate, lo, hi, error_hi))

    has_gate = any(item[6] is not None for item in summary_items)
    summary_cols = 5 if has_gate else 4
    fig, axes = plt.subplots(
        len(summary_items),
        summary_cols,
        figsize=(3.0 * summary_cols, 2.6 * len(summary_items)),
        dpi=160,
    )
    if len(summary_items) == 1:
        axes = np.expand_dims(axes, axis=0)
    for row_index, (row, z, real, stage1, synthetic, error, gate, lo, hi, error_hi) in enumerate(summary_items):
        show_slice(axes[row_index, 0], real, z, f"{row['dataset']} {row['subject_id']} real", "gray", lo, hi)
        show_slice(axes[row_index, 1], stage1, z, "stage1", "gray", lo, hi)
        show_slice(axes[row_index, 2], synthetic, z, "generated", "gray", lo, hi)
        show_slice(axes[row_index, 3], error, z, "abs error", "magma", 0.0, error_hi)
        if has_gate:
            if gate is None:
                axes[row_index, 4].axis("off")
            else:
                show_slice(axes[row_index, 4], gate, z, "refinement gate", "viridis", 0.0, 1.0)
    fig.suptitle(
        f"{args.spatial_size}x{args.spatial_size} comparison: real vs generated {target_modality.upper()}",
        fontsize=12,
    )
    fig.tight_layout()
    summary_png = output_dir / f"summary_real_vs_generated_{target_modality}_{args.spatial_size}.png"
    fig.savefig(summary_png, bbox_inches="tight")
    plt.close(fig)

    report = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "target_modality": target_modality,
        "spatial_size": int(args.spatial_size),
        "summary_png": str(summary_png.resolve()),
        "cases": case_records,
        "mean_mae": float(np.mean([item["mae"] for item in case_records])),
        "mean_psnr": float(np.mean([item["psnr"] for item in case_records])),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
