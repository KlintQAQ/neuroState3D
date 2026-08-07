from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.brainmvp_encoder import BrainMVPEncoder, CheckpointLoadReport


CORE_PREFIXES = (
    "uniformer.patch_embed1.",
    "uniformer.patch_embed2.",
    "uniformer.patch_embed3.",
    "uniformer.patch_embed4.",
    "uniformer.blocks1.",
    "uniformer.blocks2.",
    "uniformer.blocks3.",
    "uniformer.blocks4.",
    "uniformer.norm.",
)

REPRESENTATIVE_GROUPS = {
    "patch_embed1": ("uniformer.patch_embed1.proj.weight",),
    "blocks1": (
        "uniformer.blocks1.0.conv1.weight",
        "uniformer.blocks1.0.attn.weight",
    ),
    "blocks2": (
        "uniformer.blocks2.0.conv1.weight",
        "uniformer.blocks2.0.attn.weight",
    ),
    "blocks3": (
        "uniformer.blocks3.0.attn.qkv.weight",
        "uniformer.blocks3.0.mlp.fc1.weight",
    ),
    "blocks4": (
        "uniformer.blocks4.0.attn.qkv.weight",
        "uniformer.blocks4.0.mlp.fc1.weight",
    ),
    "norm": ("uniformer.norm.weight",),
}


def tensor_stats(tensor: torch.Tensor) -> dict[str, Any]:
    tensor = tensor.detach().float().cpu()
    return {
        "shape": list(tensor.shape),
        "mean": float(tensor.mean()),
        "std": float(tensor.std(unbiased=False)),
        "min": float(tensor.min()),
        "max": float(tensor.max()),
    }


def diff_stats(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
    diff = (left.detach().float().cpu() - right.detach().float().cpu()).abs()
    return {
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
    }


def is_core_encoder_key(key: str) -> bool:
    return key.startswith(CORE_PREFIXES)


def classify_unexpected_key(key: str) -> str:
    if ".decoder." in key or key.startswith("decoder."):
        return "decoder"
    if key.endswith("rep_template") or ".rep_template" in key:
        return "rep_template"
    if ".projector." in key or ".projection." in key or ".head." in key:
        return "pretraining_head"
    return "other"


def summarize_load_report(
    report: CheckpointLoadReport | None,
    full_report: bool = False,
) -> dict[str, object] | None:
    if report is None:
        return None

    unexpected_groups = Counter(classify_unexpected_key(key) for key in report.unexpected_keys)
    shape_mismatch_keys = report.shape_mismatch_keys
    core_missing = [key for key in report.missing_keys if is_core_encoder_key(key)]
    core_shape_mismatch = [
        item
        for item in shape_mismatch_keys
        if is_core_encoder_key(str(item.get("model_key", "")))
    ]
    sample_count = None if full_report else 50

    summarized: dict[str, object] = {
        "path": report.path,
        "checkpoint_state_source": report.checkpoint_state_source,
        "checkpoint_tensor_count": report.checkpoint_tensor_count,
        "checkpoint_parameter_count": report.checkpoint_parameter_count,
        "total_encoder_tensor_count": report.total_encoder_tensor_count,
        "total_encoder_parameter_count": report.total_encoder_parameter_count,
        "matched_tensor_count": report.matched_tensor_count,
        "matched_parameter_count": report.matched_parameter_count,
        "matched_tensor_ratio": report.matched_tensor_ratio,
        "matched_parameter_ratio": report.matched_parameter_ratio,
        "missing_key_count": len(report.missing_keys),
        "missing_keys": report.missing_keys[:sample_count],
        "unexpected_key_count": len(report.unexpected_keys),
        "unexpected_key_groups": dict(sorted(unexpected_groups.items())),
        "unexpected_keys": report.unexpected_keys[:sample_count],
        "shape_mismatch_key_count": len(shape_mismatch_keys),
        "shape_mismatch_keys": shape_mismatch_keys[:sample_count],
        "core_missing_key_count": len(core_missing),
        "core_missing_keys": core_missing[:sample_count],
        "core_shape_mismatch_key_count": len(core_shape_mismatch),
        "core_shape_mismatch_keys": core_shape_mismatch[:sample_count],
        "matched_keys_sample": report.matched_keys[:50],
        "matched_model_to_checkpoint_sample": [
            {"model_key": model_key, "checkpoint_key": checkpoint_key}
            for model_key, checkpoint_key in list(
                report.matched_model_to_checkpoint.items()
            )[:50]
        ],
        "warnings": report.warnings,
    }
    if full_report:
        summarized["matched_keys"] = report.matched_keys
        summarized["matched_model_to_checkpoint"] = report.matched_model_to_checkpoint
    return summarized


def select_representative_keys(
    report: CheckpointLoadReport,
    parameter_keys: set[str],
) -> dict[str, str]:
    selected: dict[str, str] = {}
    matched_keys = [
        key for key in report.matched_model_to_checkpoint if key in parameter_keys
    ]
    for group, candidates in REPRESENTATIVE_GROUPS.items():
        key = next(
            (
                candidate
                for candidate in candidates
                if candidate in report.matched_model_to_checkpoint
                and candidate in parameter_keys
            ),
            None,
        )
        if key is None:
            prefix = f"uniformer.{group}."
            key = next((item for item in matched_keys if item.startswith(prefix)), None)
        if key is not None:
            selected[group] = key
    return selected


def compare_loaded_weights(
    checkpoint_path: str,
    loaded_model: BrainMVPEncoder,
    random_model: BrainMVPEncoder,
) -> dict[str, object]:
    if loaded_model.last_load_report is None:
        return {"status": "no_checkpoint_report"}

    raw = torch.load(checkpoint_path, map_location="cpu")
    _, checkpoint_state = BrainMVPEncoder.extract_checkpoint_state(raw)
    checkpoint_tensors = BrainMVPEncoder.tensor_state_dict(checkpoint_state)
    random_state = random_model.encoder.state_dict()
    loaded_state = loaded_model.encoder.state_dict()
    parameter_keys = set(dict(loaded_model.encoder.named_parameters()))

    changed_count = 0
    equals_checkpoint_count = 0
    checked_count = 0
    for model_key, checkpoint_key in loaded_model.last_load_report.matched_model_to_checkpoint.items():
        if model_key not in parameter_keys:
            continue
        checked_count += 1
        random_tensor = random_state[model_key].detach().cpu()
        loaded_tensor = loaded_state[model_key].detach().cpu()
        checkpoint_tensor = checkpoint_tensors[checkpoint_key].detach().cpu()
        if not torch.allclose(random_tensor, loaded_tensor):
            changed_count += 1
        if torch.allclose(loaded_tensor, checkpoint_tensor):
            equals_checkpoint_count += 1

    comparisons = []
    selected = select_representative_keys(loaded_model.last_load_report, parameter_keys)
    for group, model_key in selected.items():
        checkpoint_key = loaded_model.last_load_report.matched_model_to_checkpoint[model_key]
        random_tensor = random_state[model_key].detach().cpu()
        loaded_tensor = loaded_state[model_key].detach().cpu()
        checkpoint_tensor = checkpoint_tensors[checkpoint_key].detach().cpu()
        comparisons.append(
            {
                "group": group,
                "model_key": model_key,
                "checkpoint_key": checkpoint_key,
                "differs_from_random_init": not torch.allclose(
                    random_tensor, loaded_tensor
                ),
                "loaded_equals_checkpoint_tensor": torch.allclose(
                    loaded_tensor, checkpoint_tensor
                ),
                "random_vs_loaded": diff_stats(random_tensor, loaded_tensor),
                "loaded_vs_checkpoint": diff_stats(loaded_tensor, checkpoint_tensor),
                "loaded_stats": tensor_stats(loaded_tensor),
                "checkpoint_stats": tensor_stats(checkpoint_tensor),
            }
        )

    selected_all_changed = bool(comparisons) and all(
        bool(item["differs_from_random_init"]) for item in comparisons
    )
    selected_all_equal_checkpoint = bool(comparisons) and all(
        bool(item["loaded_equals_checkpoint_tensor"]) for item in comparisons
    )

    return {
        "checked_parameter_tensors": checked_count,
        "changed_from_random_count": changed_count,
        "equals_checkpoint_count": equals_checkpoint_count,
        "all_checked_parameters_equal_checkpoint": checked_count
        == equals_checkpoint_count,
        "any_checked_parameter_changed": changed_count > 0,
        "representative_group_count": len(comparisons),
        "representative_all_changed_from_random": selected_all_changed,
        "representative_all_equal_checkpoint": selected_all_equal_checkpoint,
        "representative_comparisons": comparisons,
    }


def verify_official_encoder_equivalence(
    model: BrainMVPEncoder,
    x: torch.Tensor,
) -> dict[str, object]:
    from models.Uniformer import SSLEncoder

    official = SSLEncoder(num_phase=model.in_channels).to(x.device)
    official.load_state_dict(model.encoder.state_dict(), strict=True)
    official.eval()
    official_x = x
    if model.input_layout == "bcdhw":
        official_x = x.permute(0, 1, 3, 4, 2).contiguous()

    with torch.no_grad():
        wrapper_features = model(x)
        official_features = official(official_x)

    stage_reports = []
    max_abs_diff = 0.0
    for index, official_tensor in enumerate(official_features):
        wrapper_tensor = wrapper_features[f"stage{index}"]
        stage_diff = diff_stats(wrapper_tensor, official_tensor)
        max_abs_diff = max(max_abs_diff, stage_diff["max_abs_diff"])
        stage_reports.append(
            {
                "stage": f"stage{index}",
                "shape": list(wrapper_tensor.shape),
                "max_abs_diff": stage_diff["max_abs_diff"],
                "mean_abs_diff": stage_diff["mean_abs_diff"],
                "allclose": torch.allclose(wrapper_tensor, official_tensor),
            }
        )

    return {
        "stage_reports": stage_reports,
        "max_abs_diff": max_abs_diff,
        "all_stages_allclose": all(bool(item["allclose"]) for item in stage_reports),
    }


def validation_summary(
    report: CheckpointLoadReport | None,
    weight_verification: dict[str, object],
    wrapper_equivalence: dict[str, object] | None,
    features: dict[str, object] | None,
    min_parameter_coverage: float,
    skipped_forward: bool,
) -> dict[str, object]:
    if report is None:
        return {
            "status": "PENDING",
            "reasons": ["no checkpoint provided"],
        }

    reasons = []
    if report.matched_parameter_ratio < min_parameter_coverage:
        reasons.append(
            "parameter coverage below threshold: "
            f"{report.matched_parameter_ratio:.6f} < {min_parameter_coverage:.6f}"
        )
    if report.missing_keys:
        reasons.append(f"missing encoder keys: {len(report.missing_keys)}")
    if report.shape_mismatch_keys:
        reasons.append(f"shape mismatch keys: {len(report.shape_mismatch_keys)}")
    if not bool(weight_verification.get("representative_all_changed_from_random")):
        reasons.append("representative loaded tensors did not all differ from random init")
    if not bool(weight_verification.get("representative_all_equal_checkpoint")):
        reasons.append("representative loaded tensors did not all equal checkpoint tensors")
    if not skipped_forward and features is None:
        reasons.append("forward pass was requested but no feature summary was produced")
    if wrapper_equivalence is not None and not bool(
        wrapper_equivalence.get("all_stages_allclose")
    ):
        reasons.append("wrapper outputs differ from official SSLEncoder outputs")

    return {
        "status": "PASS" if not reasons else "FAIL",
        "reasons": reasons,
        "min_parameter_coverage": min_parameter_coverage,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect BrainMVP encoder features.")
    parser.add_argument("--checkpoint", default="", type=str)
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument("--size", default=96, type=int)
    parser.add_argument("--batch-size", default=1, type=int)
    parser.add_argument("--in-channels", default=1, type=int)
    parser.add_argument("--seed", default=46, type=int)
    parser.add_argument("--skip-forward", action="store_true")
    parser.add_argument("--min-parameter-coverage", default=0.95, type=float)
    parser.add_argument("--allow-low-coverage", action="store_true")
    parser.add_argument("--full-report", action="store_true")
    parser.add_argument(
        "--freeze",
        default="freeze_all",
        choices=["freeze_all", "unfreeze_last_stage", "full_finetune"],
    )
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(
            "CUDA was requested, but this PyTorch build cannot use CUDA. "
            "Install a CUDA-enabled torch wheel or run with --device cpu."
        )

    checkpoint = args.checkpoint if args.checkpoint else None
    checkpoint_structure = (
        BrainMVPEncoder.checkpoint_structure(checkpoint) if checkpoint else None
    )
    if checkpoint is None:
        print("CHECKPOINT VALIDATION PENDING: no --checkpoint was provided.")

    torch.manual_seed(args.seed)
    random_model = BrainMVPEncoder(in_channels=args.in_channels, freeze=args.freeze)
    torch.manual_seed(args.seed)
    model = BrainMVPEncoder(
        in_channels=args.in_channels,
        checkpoint_path=checkpoint,
        freeze=args.freeze,
        min_parameter_coverage=args.min_parameter_coverage,
        error_on_low_coverage=not args.allow_low_coverage,
    )
    model.to(args.device)
    model.eval()

    weight_verification = (
        compare_loaded_weights(checkpoint, model, random_model)
        if checkpoint
        else {"status": "checkpoint_not_provided"}
    )

    features = None
    wrapper_equivalence = None
    gpu_memory = None
    if checkpoint and not args.skip_forward:
        x = torch.randn(
            args.batch_size,
            args.in_channels,
            args.size,
            args.size,
            args.size,
            device=args.device,
        )
        if args.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            features = model.feature_summary(x)
            wrapper_equivalence = verify_official_encoder_equivalence(model, x)
        if args.device.startswith("cuda") and torch.cuda.is_available():
            peak_allocated = int(torch.cuda.max_memory_allocated())
            peak_reserved = int(torch.cuda.max_memory_reserved())
            gpu_memory = {
                "peak_allocated_bytes": peak_allocated,
                "peak_allocated_mb": peak_allocated / (1024**2),
                "peak_reserved_bytes": peak_reserved,
                "peak_reserved_mb": peak_reserved / (1024**2),
            }

    validation = validation_summary(
        model.last_load_report,
        weight_verification,
        wrapper_equivalence,
        features,
        args.min_parameter_coverage,
        args.skip_forward,
    )

    output = {
        "input_shape": [
            args.batch_size,
            args.in_channels,
            args.size,
            args.size,
            args.size,
        ],
        "checkpoint_structure": checkpoint_structure,
        "checkpoint_mapping": summarize_load_report(
            model.last_load_report,
            full_report=args.full_report,
        ),
        "weight_verification": weight_verification,
        "features": features,
        "wrapper_equivalence": wrapper_equivalence,
        "gpu_memory": gpu_memory,
        "validation": validation,
    }
    print(json.dumps(output, indent=2))

    if validation["status"] == "FAIL":
        raise SystemExit(
            "BrainMVP checkpoint validation failed; inspect JSON report above."
        )


if __name__ == "__main__":
    main()
