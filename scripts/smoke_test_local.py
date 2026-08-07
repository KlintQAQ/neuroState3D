from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.brainmvp_encoder import BrainMVPEncoder
from models.modality_adapter import DEFAULT_ADAPTER_TYPES, MODALITY_ORDER
from models.multimodal_fusion import stack_modalities
from models.neurostate3d import NeuroState3D


CASES = {
    "t1_only": ["t1"],
    "t1_t2": ["t1", "t2"],
    "t1_fa": ["t1", "fa"],
    "t1_t2_fa": ["t1", "t2", "fa"],
    "t1_t2_fa_alff": ["t1", "t2", "fa", "alff"],
    "t1_t2_fa_md_alff": ["t1", "t2", "fa", "md", "alff"],
}


class TinyMultiScaleEncoder(nn.Module):
    def __init__(self, in_channels: int = 1) -> None:
        super().__init__()
        self.stage1 = nn.Conv3d(in_channels, 4, kernel_size=3, padding=1)
        self.stage2 = nn.Conv3d(4, 8, kernel_size=3, stride=2, padding=1)
        self.stage3 = nn.Conv3d(8, 16, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        s0 = x
        s1 = torch.relu(self.stage1(s0))
        s2 = torch.relu(self.stage2(s1))
        s3 = torch.relu(self.stage3(s2))
        return {"stage0": s0, "stage1": s1, "stage2": s2, "stage3": s3}


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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment() -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def make_inputs(
    batch_size: int,
    size: int,
    device: str,
    modalities: list[str] | None = None,
) -> dict[str, torch.Tensor]:
    selected = modalities or list(MODALITY_ORDER)
    return {
        modality: torch.randn(
            batch_size,
            1,
            size,
            size,
            size,
            device=device,
            dtype=torch.float32,
        )
        for modality in selected
    }


def tensor_shape(tensor: torch.Tensor) -> list[int]:
    return [int(dim) for dim in tensor.shape]


def finite_flags(tensor: torch.Tensor) -> dict[str, bool]:
    return {
        "has_nan": bool(torch.isnan(tensor).any().item()),
        "has_inf": bool(torch.isinf(tensor).any().item()),
    }


def assert_finite(tensor: torch.Tensor, label: str) -> None:
    flags = finite_flags(tensor)
    if flags["has_nan"] or flags["has_inf"]:
        raise AssertionError(f"{label} contains NaN/Inf: {flags}")


def max_abs_diff(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.detach() - right.detach()).abs().max().item())


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


def build_model(
    checkpoint: Path | None,
    fusion: str,
    feature_stage: str,
    device: str,
    tiny_encoder: bool = False,
) -> NeuroState3D:
    encoder: nn.Module
    if tiny_encoder:
        encoder = TinyMultiScaleEncoder()
    else:
        if checkpoint is None:
            raise ValueError("A checkpoint is required for pretrained BrainMVP smoke.")
        encoder = BrainMVPEncoder(
            checkpoint_path=str(checkpoint),
            freeze="freeze_all",
            min_parameter_coverage=0.95,
        )
    model = NeuroState3D(
        modalities=list(MODALITY_ORDER),
        encoder=encoder,
        fusion_type=fusion,
        feature_stage=feature_stage if not tiny_encoder else "stage3",
        adapter_types=DEFAULT_ADAPTER_TYPES,
    )
    model.to(device)
    return model


def run_forward_case(
    model: NeuroState3D,
    case_name: str,
    inputs: dict[str, torch.Tensor],
    fusion: str,
    device: str,
    modality_mask: torch.Tensor | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    model.eval()
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        adapter_shapes = {}
        adapter_finite = {}
        for modality, tensor in inputs.items():
            adapted = model.adapters(tensor, modality)
            if adapted.shape != tensor.shape:
                raise AssertionError(
                    f"Adapter changed shape for {modality}: "
                    f"{tuple(tensor.shape)} -> {tuple(adapted.shape)}"
                )
            assert_finite(adapted, f"{case_name}:{modality}:adapter")
            adapter_shapes[modality] = tensor_shape(adapted)
            adapter_finite[modality] = finite_flags(adapted)

        output = model(inputs, modality_mask=modality_mask)
        fused = output["fused_feature"]
        mask = output["modality_mask"]
        assert mask.shape == (fused.shape[0], len(MODALITY_ORDER))
        if torch.any(mask.sum(dim=1) <= 0):
            raise AssertionError("Every sample must have at least one modality.")
        assert_finite(fused, f"{case_name}:fused")

        selected_features = output["selected_features"]
        selected_shapes = {
            modality: tensor_shape(feature)
            for modality, feature in selected_features.items()
        }
        reference_shape = next(iter(selected_features.values())).shape
        for modality, feature in selected_features.items():
            if feature.shape != reference_shape:
                raise AssertionError(
                    f"Selected feature shape mismatch for {modality}: "
                    f"{tuple(feature.shape)} vs {tuple(reference_shape)}"
                )
            assert_finite(feature, f"{case_name}:{modality}:selected")

        feature_shapes = {
            modality: {
                stage: tensor_shape(feature)
                for stage, feature in features.items()
            }
            for modality, features in output["modality_features"].items()
        }

    return {
        "case": case_name,
        "fusion": fusion,
        "synthetic_input": "SYNTHETIC SMOKE INPUT",
        "available_modalities": list(inputs),
        "input_shapes": {name: tensor_shape(tensor) for name, tensor in inputs.items()},
        "modality_mask": mask.detach().cpu().tolist(),
        "adapter_output_shapes": adapter_shapes,
        "adapter_finite": adapter_finite,
        "brainmvp_feature_shapes": feature_shapes,
        "selected_fusion_stage": model.feature_stage,
        "selected_feature_shapes": selected_shapes,
        "fused_feature_shape": tensor_shape(fused),
        "fused_finite": finite_flags(fused),
        "dtype": str(fused.dtype),
        "runtime_sec": time.perf_counter() - started,
        "gpu_memory": cuda_memory(device),
        "pass": True,
    }


def checkpoint_report(checkpoint: Path) -> dict[str, Any]:
    model = BrainMVPEncoder(checkpoint_path=str(checkpoint), freeze="freeze_all")
    report = model.last_load_report
    if report is None:
        raise AssertionError("Checkpoint did not produce a load report.")
    return {
        "path": str(checkpoint),
        "exists": checkpoint.exists(),
        "sha256": sha256(checkpoint),
        "checkpoint_loaded": True,
        "matched_tensor_count": report.matched_tensor_count,
        "total_encoder_tensor_count": report.total_encoder_tensor_count,
        "matched_parameter_count": report.matched_parameter_count,
        "total_encoder_parameter_count": report.total_encoder_parameter_count,
        "parameter_coverage": report.matched_parameter_ratio,
        "missing_keys": len(report.missing_keys),
        "shape_mismatch_keys": len(report.shape_mismatch_keys),
        "pass": report.matched_parameter_ratio >= 0.95
        and not report.missing_keys
        and not report.shape_mismatch_keys,
    }


def run_tiny_cases(
    checkpoint: Path,
    device: str,
    size: int,
    feature_stage: str,
    fusions: list[str],
) -> list[dict[str, Any]]:
    results = []
    for fusion in fusions:
        model = build_model(checkpoint, fusion, feature_stage, device)
        base = make_inputs(1, size, device)
        for case_name, keep in CASES.items():
            case_inputs = {name: base[name] for name in keep}
            results.append(run_forward_case(model, case_name, case_inputs, fusion, device))

        mixed_inputs = make_inputs(2, size, device)
        mixed_mask = torch.tensor(
            [[1, 1, 0, 0, 0], [1, 1, 1, 0, 1]],
            dtype=torch.float32,
            device=device,
        )
        results.append(
            run_forward_case(
                model,
                "mixed_mask_batch",
                mixed_inputs,
                fusion,
                device,
                modality_mask=mixed_mask,
            )
        )
    return results


def run_cpu_tiny_structural(size: int) -> dict[str, Any]:
    device = "cpu"
    model = build_model(None, "mean", "stage3", device, tiny_encoder=True)
    inputs = make_inputs(1, size, device, modalities=["t1"])
    result = run_forward_case(model, "cpu_tiny_structural_t1", inputs, "mean", device)
    return {
        "encoder": "TinyMultiScaleEncoder",
        "device": device,
        "result": result,
        "pass": bool(result["pass"]),
    }


def run_mean_numerical_validation(
    checkpoint: Path,
    device: str,
    size: int,
    feature_stage: str,
) -> dict[str, Any]:
    model = build_model(checkpoint, "mean", feature_stage, device)
    model.eval()
    tolerance = 1e-5
    with torch.no_grad():
        single = make_inputs(1, size, device, modalities=["t1"])
        single_out = model(single)
        single_diff = max_abs_diff(
            single_out["fused_feature"],
            single_out["selected_features"]["t1"],
        )

        pair = make_inputs(1, size, device, modalities=["t1", "t2"])
        pair_out = model(pair)
        pair_expected = (
            pair_out["selected_features"]["t1"] + pair_out["selected_features"]["t2"]
        ) / 2.0
        pair_diff = max_abs_diff(pair_out["fused_feature"], pair_expected)

        missing_fa = make_inputs(1, size, device, modalities=["t1", "t2", "fa"])
        missing_fa_mask = torch.tensor(
            [[1, 1, 0, 0, 0]], dtype=torch.float32, device=device
        )
        missing_out = model(missing_fa, modality_mask=missing_fa_mask)
        missing_expected = (
            missing_out["selected_features"]["t1"]
            + missing_out["selected_features"]["t2"]
        ) / 2.0
        wrong_divide3 = (
            missing_out["selected_features"]["t1"]
            + missing_out["selected_features"]["t2"]
            + missing_out["selected_features"]["fa"]
        ) / 3.0
        missing_diff = max_abs_diff(missing_out["fused_feature"], missing_expected)
        wrong_divide3_diff = max_abs_diff(missing_out["fused_feature"], wrong_divide3)

        mixed = make_inputs(2, size, device)
        mixed_mask = torch.tensor(
            [[1, 1, 0, 0, 0], [1, 1, 1, 0, 1]],
            dtype=torch.float32,
            device=device,
        )
        mixed_out = model(mixed, modality_mask=mixed_mask)
        selected = mixed_out["selected_features"]
        mixed_expected_0 = (selected["t1"][0] + selected["t2"][0]) / 2.0
        mixed_expected_1 = (
            selected["t1"][1]
            + selected["t2"][1]
            + selected["fa"][1]
            + selected["alff"][1]
        ) / 4.0
        mixed_diff_0 = max_abs_diff(mixed_out["fused_feature"][0], mixed_expected_0)
        mixed_diff_1 = max_abs_diff(mixed_out["fused_feature"][1], mixed_expected_1)

    return {
        "tolerance": tolerance,
        "t1_only_max_abs_diff": single_diff,
        "t1_t2_max_abs_diff": pair_diff,
        "t1_t2_missing_fa_divide2_max_abs_diff": missing_diff,
        "t1_t2_missing_fa_wrong_divide3_max_abs_diff": wrong_divide3_diff,
        "mixed_mask_sample0_max_abs_diff": mixed_diff_0,
        "mixed_mask_sample1_max_abs_diff": mixed_diff_1,
        "pass": all(
            value <= tolerance
            for value in [
                single_diff,
                pair_diff,
                missing_diff,
                mixed_diff_0,
                mixed_diff_1,
            ]
        ),
    }


def run_concat_validation(
    checkpoint: Path,
    device: str,
    size: int,
    feature_stage: str,
) -> dict[str, Any]:
    model = build_model(checkpoint, "concat", feature_stage, device)
    model.eval()
    output_shapes = {}
    stack_shapes = {}
    missing_slot_zero_checks = {}
    projection_input_channels = None
    projection_output_channels = None
    with torch.no_grad():
        for case_name, keep in CASES.items():
            inputs = make_inputs(1, size, device, modalities=keep)
            output = model(inputs)
            fused = output["fused_feature"]
            mask = output["modality_mask"]
            stacked = stack_modalities(
                output["selected_features"],
                mask.clone(),
                list(MODALITY_ORDER),
            )
            output_shapes[case_name] = tensor_shape(fused)
            stack_shapes[case_name] = tensor_shape(stacked)
            missing_slot_zero_checks[case_name] = {}
            for index, modality in enumerate(MODALITY_ORDER):
                if modality not in keep:
                    missing_slot_zero_checks[case_name][modality] = bool(
                        torch.count_nonzero(stacked[:, index]).item() == 0
                    )
            assert_finite(fused, f"concat:{case_name}:fused")

        if model.fusion is None:
            raise AssertionError("ConcatFusion was not initialized.")
        projection = model.fusion.projection[0]
        projection_input_channels = int(projection.in_channels)
        projection_output_channels = int(model.fusion.projection[-1].out_channels)

    unique_output_shapes = {tuple(shape) for shape in output_shapes.values()}
    unique_stack_modalities = {shape[1] for shape in stack_shapes.values()}
    first_shape = next(iter(output_shapes.values()))
    channels = first_shape[1]
    expected_projection_in = len(MODALITY_ORDER) * channels + len(MODALITY_ORDER)
    all_missing_slots_zero = all(
        all(checks.values()) for checks in missing_slot_zero_checks.values()
    )
    return {
        "modality_order": list(MODALITY_ORDER),
        "output_shapes": output_shapes,
        "stack_shapes": stack_shapes,
        "missing_slot_zero_checks": missing_slot_zero_checks,
        "projection_input_channels": projection_input_channels,
        "expected_projection_input_channels": expected_projection_in,
        "projection_output_channels": projection_output_channels,
        "fixed_output_shape": len(unique_output_shapes) == 1,
        "fixed_modality_slots": unique_stack_modalities == {len(MODALITY_ORDER)},
        "all_missing_slots_zero": all_missing_slots_zero,
        "pass": len(unique_output_shapes) == 1
        and unique_stack_modalities == {len(MODALITY_ORDER)}
        and all_missing_slots_zero
        and projection_input_channels == expected_projection_in,
    }


def clone_named_parameters(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: param.detach().cpu().clone()
        for name, param in module.named_parameters()
    }


def max_parameter_change(
    before: dict[str, torch.Tensor],
    module: nn.Module,
) -> float:
    max_change = 0.0
    for name, param in module.named_parameters():
        if name not in before:
            continue
        change = (param.detach().cpu() - before[name]).abs().max().item()
        max_change = max(max_change, float(change))
    return max_change


def finite_grads(named_params: list[tuple[str, torch.nn.Parameter]]) -> dict[str, Any]:
    missing = []
    nonfinite = []
    present = []
    for name, param in named_params:
        if param.grad is None:
            missing.append(name)
            continue
        present.append(name)
        if not torch.isfinite(param.grad).all().item():
            nonfinite.append(name)
    return {
        "present_count": len(present),
        "missing": missing,
        "nonfinite": nonfinite,
        "pass": bool(present) and not missing and not nonfinite,
    }


def run_gradient_validation(
    checkpoint: Path,
    device: str,
    size: int,
    feature_stage: str,
) -> dict[str, Any]:
    model = build_model(checkpoint, "concat", feature_stage, device)
    model.train()
    inputs = make_inputs(1, size, device)

    output = model(inputs)
    loss = output["fused_feature"].square().mean()
    if model.fusion is None:
        raise AssertionError("ConcatFusion was not initialized before optimizer step.")

    backbone_named = list(model.encoder.encoder.named_parameters())
    adapter_named = [
        (name, param)
        for name, param in model.adapters.named_parameters()
        if param.requires_grad
    ]
    fusion_named = [
        (name, param)
        for name, param in model.fusion.named_parameters()
        if param.requires_grad
    ]
    trainable_params = [param for _, param in adapter_named + fusion_named]
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-3)

    backbone_before = clone_named_parameters(model.encoder.encoder)
    adapter_before = clone_named_parameters(model.adapters)
    fusion_before = clone_named_parameters(model.fusion)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    backbone_grad_none = all(param.grad is None for _, param in backbone_named)
    adapter_grad = finite_grads(adapter_named)
    fusion_grad = finite_grads(fusion_named)
    optimizer.step()

    backbone_change = max_parameter_change(backbone_before, model.encoder.encoder)
    adapter_change = max_parameter_change(adapter_before, model.adapters)
    fusion_change = max_parameter_change(fusion_before, model.fusion)

    return {
        "loss": float(loss.detach().cpu().item()),
        "model_train_called": True,
        "brainmvp_encoder_training_flag": bool(model.encoder.encoder.training),
        "backbone_requires_grad_false": all(
            not param.requires_grad for _, param in backbone_named
        ),
        "backbone_grad_none": backbone_grad_none,
        "adapter_grad": adapter_grad,
        "fusion_grad": fusion_grad,
        "backbone_changed": backbone_change != 0.0,
        "adapter_changed": adapter_change != 0.0,
        "fusion_changed": fusion_change != 0.0,
        "backbone_max_abs_change": backbone_change,
        "adapter_max_abs_change": adapter_change,
        "fusion_max_abs_change": fusion_change,
        "pass": all(
            [
                all(not param.requires_grad for _, param in backbone_named),
                backbone_grad_none,
                adapter_grad["pass"],
                fusion_grad["pass"],
                backbone_change == 0.0,
                adapter_change > 0.0,
                fusion_change > 0.0,
            ]
        ),
    }


def run_all_zero_mask_check(
    checkpoint: Path,
    device: str,
    size: int,
    feature_stage: str,
) -> dict[str, Any]:
    model = build_model(checkpoint, "mean", feature_stage, device)
    inputs = make_inputs(1, size, device, modalities=["t1"])
    bad_mask = torch.zeros(1, len(MODALITY_ORDER), dtype=torch.float32, device=device)
    try:
        model(inputs, modality_mask=bad_mask)
    except ValueError as exc:
        return {
            "raised": True,
            "exception": str(exc),
            "pass": True,
        }
    return {
        "raised": False,
        "exception": None,
        "pass": False,
    }


def run_train_eval_forward_check(
    checkpoint: Path,
    device: str,
    size: int,
    feature_stage: str,
) -> dict[str, Any]:
    model = build_model(checkpoint, "mean", feature_stage, device)
    inputs = make_inputs(1, size, device, modalities=["t1", "t2"])
    model.eval()
    with torch.no_grad():
        eval_output = model(inputs)
    model.train()
    train_output = model(inputs)
    assert_finite(eval_output["fused_feature"], "eval_forward")
    assert_finite(train_output["fused_feature"], "train_forward")
    return {
        "eval_forward": True,
        "train_forward": True,
        "brainmvp_encoder_training_after_model_train": bool(model.encoder.encoder.training),
        "pass": True,
    }


def run_real_size(
    checkpoint: Path,
    device: str,
    real_size: int,
    feature_stage: str,
) -> dict[str, Any]:
    if device.startswith("cuda") and not torch.cuda.is_available():
        return {
            "status": "SKIPPED_NO_CUDA",
            "pass": False,
        }
    model = build_model(checkpoint, "mean", feature_stage, device)
    results: dict[str, Any] = {}

    def run_real_case(case_name: str, modalities: list[str]) -> dict[str, Any]:
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        inputs = make_inputs(1, real_size, device, modalities=modalities)
        return run_forward_case(model, case_name, inputs, "mean", device)

    results["t1_only"] = run_real_case("real_96_t1_only", ["t1"])
    try:
        results["t1_t2"] = run_real_case("real_96_t1_t2", ["t1", "t2"])
        multimodal_status = "PASS"
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
        results["t1_t2"] = {
            "pass": False,
            "memory_limited": True,
            "error": str(exc),
        }
        multimodal_status = "LOCAL REAL-SIZE MULTIMODAL: MEMORY LIMITED"

    return {
        "real_size": real_size,
        "device": device,
        "cases": results,
        "multimodal_status": multimodal_status,
        "pass": bool(results["t1_only"]["pass"])
        and (
            bool(results["t1_t2"].get("pass"))
            or bool(results["t1_t2"].get("memory_limited"))
        ),
    }


def final_tests(report: dict[str, Any]) -> dict[str, bool]:
    tiny_results = report["tiny_pretrained_cases"]
    return {
        "backbone_checkpoint": bool(report["checkpoint"]["pass"]),
        "tiny_mean": all(
            item["pass"] for item in tiny_results if item["fusion"] == "mean"
        ),
        "tiny_concat": all(
            item["pass"] for item in tiny_results if item["fusion"] == "concat"
        ),
        "mixed_mask_batch": all(
            item["pass"]
            for item in tiny_results
            if item["case"] == "mixed_mask_batch"
        ),
        "mean_fusion_math": bool(report["mean_fusion_validation"]["pass"]),
        "concat_fixed_slot": bool(report["concat_fusion_validation"]["pass"]),
        "nan_inf": not any(
            item["fused_finite"]["has_nan"] or item["fused_finite"]["has_inf"]
            for item in tiny_results
        ),
        "backward": bool(report["gradient_validation"]["pass"]),
        "backbone_frozen": bool(
            report["gradient_validation"]["backbone_requires_grad_false"]
            and report["gradient_validation"]["backbone_grad_none"]
            and not report["gradient_validation"]["backbone_changed"]
        ),
        "adapter_grad": bool(report["gradient_validation"]["adapter_grad"]["pass"]),
        "fusion_grad": bool(report["gradient_validation"]["fusion_grad"]["pass"]),
        "optimizer_step": bool(
            report["gradient_validation"]["adapter_changed"]
            and report["gradient_validation"]["fusion_changed"]
            and not report["gradient_validation"]["backbone_changed"]
        ),
        "real_96_t1": bool(
            report["real_size_synthetic"]["cases"]["t1_only"]["pass"]
        ),
        "real_96_t1_t2": bool(
            report["real_size_synthetic"]["cases"]["t1_t2"].get("pass", False)
        ),
        "all_zero_mask_raises": bool(report["all_zero_mask_check"]["pass"]),
        "train_eval_forward": bool(report["train_eval_forward"]["pass"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local NeuroState-3D smoke tests.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--tiny-size", default=16, type=int)
    parser.add_argument("--real-size", default=96, type=int)
    parser.add_argument("--feature-stage", default="stage4", type=str)
    parser.add_argument(
        "--fusion",
        action="append",
        choices=["mean", "concat"],
        default=None,
        help="Fusion(s) to test. Repeat for multiple. Defaults to both.",
    )
    parser.add_argument("--skip-real-size", action="store_true")
    parser.add_argument(
        "--report-path",
        default=Path("outputs/local_smoke_report.json"),
        type=Path,
    )
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false.")
    checkpoint = args.checkpoint if args.checkpoint.is_absolute() else ROOT / args.checkpoint
    if not checkpoint.exists():
        raise SystemExit(f"Checkpoint does not exist: {checkpoint}")

    fusions = args.fusion or ["mean", "concat"]
    torch.manual_seed(46)
    started = time.perf_counter()

    report: dict[str, Any] = {
        "git_commit": git_commit(),
        "environment": environment(),
        "synthetic_input_label": "SYNTHETIC SMOKE INPUT",
        "modality_order": list(MODALITY_ORDER),
        "adapter_types": DEFAULT_ADAPTER_TYPES,
        "fusion_stage": args.feature_stage,
        "fp32": True,
        "checkpoint": checkpoint_report(checkpoint),
        "cpu_tiny_structural": run_cpu_tiny_structural(args.tiny_size),
        "tiny_pretrained_cases": run_tiny_cases(
            checkpoint,
            args.device,
            args.tiny_size,
            args.feature_stage,
            fusions,
        ),
        "mean_fusion_validation": run_mean_numerical_validation(
            checkpoint, args.device, args.tiny_size, args.feature_stage
        ),
        "concat_fusion_validation": run_concat_validation(
            checkpoint, args.device, args.tiny_size, args.feature_stage
        ),
        "gradient_validation": run_gradient_validation(
            checkpoint, args.device, args.tiny_size, args.feature_stage
        ),
        "all_zero_mask_check": run_all_zero_mask_check(
            checkpoint, args.device, args.tiny_size, args.feature_stage
        ),
        "train_eval_forward": run_train_eval_forward_check(
            checkpoint, args.device, args.tiny_size, args.feature_stage
        ),
    }

    if args.skip_real_size:
        report["real_size_synthetic"] = {
            "skipped": True,
            "pass": False,
        }
    else:
        report["real_size_synthetic"] = run_real_size(
            checkpoint, args.device, args.real_size, args.feature_stage
        )

    tests = final_tests(report)
    real_multimodal = report["real_size_synthetic"]["cases"]["t1_t2"]
    required = [
        tests["backbone_checkpoint"],
        tests["tiny_mean"],
        tests["tiny_concat"],
        tests["mixed_mask_batch"],
        tests["mean_fusion_math"],
        tests["concat_fixed_slot"],
        tests["nan_inf"],
        tests["backward"],
        tests["backbone_frozen"],
        tests["adapter_grad"],
        tests["fusion_grad"],
        tests["optimizer_step"],
        tests["real_96_t1"],
        tests["all_zero_mask_raises"],
        tests["train_eval_forward"],
    ]
    if real_multimodal.get("memory_limited"):
        tests["real_96_t1_t2_memory_limited_allowed"] = True
    else:
        required.append(tests["real_96_t1_t2"])
    report["tests"] = tests
    report["runtime_sec"] = time.perf_counter() - started
    report["final_verdict"] = (
        "LOCAL NEUROSTATE3D SMOKE: PASS"
        if all(required)
        else "LOCAL NEUROSTATE3D SMOKE: FAIL"
    )

    report_path = args.report_path
    if not report_path.is_absolute():
        report_path = ROOT / report_path
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    for key, value in tests.items():
        print(f"{key}: {'PASS' if value else 'FAIL'}")
    print(report["final_verdict"])
    print(f"report: {report_path}")

    if report["final_verdict"].endswith("FAIL"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
