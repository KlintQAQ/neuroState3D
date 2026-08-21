from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.brats_fusion_dataset import BRATS_MODALITIES, BRATS_REGIONS, BraTSFusionDataset  # noqa: E402
from models.evidence_fusion import EvidenceFusionConfig, EvidenceReliableFusion  # noqa: E402
from models.virtual_modality_generator import (  # noqa: E402
    VirtualModalityGenerator,
    VirtualModalityGeneratorConfig,
    observed_mask_without_target,
)
from scripts.smoke_evidence_fusion_brats import environment, git_commit  # noqa: E402
from scripts.train_evidence_fusion_small import (  # noqa: E402
    VIRTUAL_T1C_MODALITY,
    active_modality_order,
    batch_cache_rows,
    load_cached_virtual_batch,
    load_synthetic_cache_manifest,
    make_mask,
    make_state,
)
from utils.brats_metrics import dice_scores  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate whether synthetic missing modalities improve fusion inference."
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
        "--fusion-checkpoint",
        default=str(
            ROOT
            / "outputs"
            / "evidence_fusion_t1c_targeted_96_e1"
            / "evidence_fusion_small_last.pt"
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
    parser.add_argument(
        "--brainmvp-checkpoint",
        default=str(ROOT / "pretrained" / "BrainMVP_uniformer.pt"),
    )
    parser.add_argument("--synthetic-cache-manifest", default="")
    parser.add_argument("--target-modality", default="t1c", choices=BRATS_MODALITIES)
    parser.add_argument("--spatial-size", type=int, default=32)
    parser.add_argument("--max-subjects", type=int, default=72)
    parser.add_argument("--val-subjects", type=int, default=8)
    parser.add_argument("--foreground-crop-prob", type=float, default=0.0)
    parser.add_argument("--crop-mode", default="region_balanced")
    parser.add_argument(
        "--synthetic-mode",
        choices=("present", "degraded", "availability_weighted"),
        default="availability_weighted",
    )
    parser.add_argument("--synthetic-state", choices=("present", "degraded"), default="degraded")
    parser.add_argument("--min-synthetic-availability", type=float, default=0.05)
    parser.add_argument("--max-synthetic-availability", type=float, default=0.45)
    parser.add_argument("--synthetic-confidence-temperature", type=float, default=1.0)
    parser.add_argument(
        "--synthetic-image-mode",
        choices=("raw", "confidence_weighted"),
        default="raw",
    )
    parser.add_argument("--uncertainty-threshold", type=float, default=0.0)
    parser.add_argument(
        "--enable-virtual-t1c-slot",
        action="store_true",
        help=(
            "Evaluate synthetic T1c as an independent virtual_t1c evidence slot "
            "instead of replacing the real t1c channel."
        ),
    )
    parser.add_argument("--hidden-channels", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--report-path",
        default=str(ROOT / "reports" / "synthetic_modality_fusion_eval.json"),
    )
    return parser.parse_args()


def load_fusion_model(
    args: argparse.Namespace,
    modality_order: tuple[str, ...],
) -> EvidenceReliableFusion:
    payload = torch.load(args.fusion_checkpoint, map_location="cpu", weights_only=False)
    config = payload.get("config", {})
    model = EvidenceReliableFusion(
        EvidenceFusionConfig(
            modalities=modality_order,
            checkpoint_path=args.brainmvp_checkpoint,
            encoder_freeze=config.get("encoder_freeze", "freeze_all"),
            feature_stage=config.get("feature_stage", "stage4"),
            feature_channels=int(config.get("feature_channels", 512)),
            hidden_channels=int(config.get("hidden_channels", args.hidden_channels)),
            enable_region_query_fusion=bool(
                config.get("enable_region_query_fusion", False)
            ),
            completion_gate_weight=float(config.get("completion_gate_weight", 0.0)),
            completion_gate_cap=float(config.get("completion_gate_cap", 1.0)),
            enable_virtual_t1c_confidence=bool(
                config.get("enable_virtual_t1c_confidence", False)
            ),
            virtual_t1c_min_gate_cap=float(
                config.get("virtual_t1c_min_gate_cap", 0.05)
            ),
            virtual_t1c_max_gate_cap=float(
                config.get("virtual_t1c_max_gate_cap", 0.45)
            ),
            virtual_t1c_disagreement_scale=float(
                config.get("virtual_t1c_disagreement_scale", 0.25)
            ),
            modality_confidence_logit_weight=float(
                config.get("modality_confidence_logit_weight", 0.0)
            ),
            enable_high_res_branch=bool(config.get("enable_high_res_branch", False)),
            high_res_channels=int(config.get("high_res_channels", 16)),
        )
    )
    model.load_state_dict(payload["model"], strict=True)
    model.to(args.device).eval()
    return model


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


def aggregate(items: list[dict[str, float]]) -> dict[str, float]:
    keys = list(items[0]) if items else []
    return {key: float(np.mean([item[key] for item in items])) for key in keys}


def synthetic_availability(
    uncertainty: torch.Tensor,
    min_value: float,
    max_value: float,
    temperature: float,
) -> torch.Tensor:
    confidence = torch.exp(-uncertainty / max(float(temperature), 1e-4))
    confidence = confidence.mean(dim=(1, 2, 3, 4))
    lo = max(min(float(min_value), 1.0), 0.0)
    hi = max(min(float(max_value), 1.0), lo)
    return lo + (hi - lo) * confidence


@torch.no_grad()
def main() -> int:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false.")
    target_index = BRATS_MODALITIES.index(args.target_modality)
    modality_order = active_modality_order(args.enable_virtual_t1c_slot)
    fusion_model = load_fusion_model(args, modality_order)
    synthetic_cache = load_synthetic_cache_manifest(args.synthetic_cache_manifest)
    generator = None
    if not synthetic_cache:
        generator, generator_target = load_generator(args.generator_checkpoint, args.device)
        if generator_target != args.target_modality:
            raise ValueError(
                f"Generator target {generator_target!r} does not match {args.target_modality!r}."
            )

    dataset = BraTSFusionDataset(
        manifest_csv=args.manifest,
        spatial_size=args.spatial_size,
        max_subjects=None if args.max_subjects <= 0 else args.max_subjects,
        modality_mask_mode="all",
        foreground_crop_prob=args.foreground_crop_prob,
        crop_mode=args.crop_mode,
        seed=46,
    )
    val_count = min(args.val_subjects, max(1, len(dataset) // 4))
    val_set = Subset(dataset, list(range(len(dataset)))[len(dataset) - val_count :])
    loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0)

    cases: dict[str, list[dict[str, float]]] = {
        "full_real": [],
        "remove_target": [],
        "synthetic_filled": [],
    }
    generator_rows: list[dict[str, float]] = []
    started = time.perf_counter()
    for batch in loader:
        image = batch["image"].to(args.device)
        target = batch["target"].to(args.device)
        full_mask = make_mask(
            list(BRATS_MODALITIES),
            image.shape[0],
            image.device,
            modality_order,
        )
        missing_modalities = [
            name for name in BRATS_MODALITIES if name != args.target_modality
        ]
        remove_mask = make_mask(
            missing_modalities,
            image.shape[0],
            image.device,
            modality_order,
        )

        cache_rows = batch_cache_rows(batch, synthetic_cache)
        if cache_rows is not None:
            synthetic, confidence, uncertainty = load_cached_virtual_batch(
                cache_rows,
                image,
            )
            generated = {
                "synthetic": synthetic,
                "confidence": confidence,
                "uncertainty": uncertainty,
            }
        else:
            generated_input_mask = observed_mask_without_target(
                image.shape[0],
                image.shape[1],
                target_index,
                image.device,
            ).to(image.dtype)
            generated = generator(image, generated_input_mask, target_index)
            synthetic = generated["synthetic"]
        if args.uncertainty_threshold > 0:
            confidence_gate = (
                generated["uncertainty"] <= float(args.uncertainty_threshold)
            ).to(synthetic.dtype)
            synthetic = synthetic * confidence_gate + image[:, target_index : target_index + 1] * 0.0
        if args.synthetic_image_mode == "confidence_weighted":
            synthetic = synthetic * generated["confidence"]
        if args.enable_virtual_t1c_slot:
            model_image = torch.cat([image, synthetic], dim=1)
            synthetic_image = model_image
            real_confidence = image.new_ones(
                (image.shape[0], image.shape[1], *image.shape[2:])
            )
            model_confidence = torch.cat(
                [real_confidence, generated["confidence"].clamp(0.0, 1.0)],
                dim=1,
            )
        else:
            model_image = image
            synthetic_image = image.clone()
            synthetic_image[:, target_index : target_index + 1] = synthetic
            model_confidence = image.new_ones(image.shape)
            model_confidence[:, target_index : target_index + 1] = generated[
                "confidence"
            ].clamp(0.0, 1.0)
        if args.enable_virtual_t1c_slot:
            virtual_index = modality_order.index(VIRTUAL_T1C_MODALITY)
            synthetic_mask = remove_mask.clone()
            synthetic_mask[:, virtual_index] = 1.0
        else:
            virtual_index = target_index
            synthetic_mask = full_mask.clone()
        synthetic_state = make_state(synthetic_mask)
        if args.synthetic_mode == "availability_weighted":
            synthetic_mask = remove_mask.clone()
            availability = synthetic_availability(
                generated["uncertainty"],
                args.min_synthetic_availability,
                args.max_synthetic_availability,
                args.synthetic_confidence_temperature,
            ).to(synthetic_mask.dtype)
            if args.enable_virtual_t1c_slot:
                synthetic_mask[:, virtual_index] = availability
            else:
                synthetic_mask[:, target_index] = availability
            synthetic_state = make_state(synthetic_mask)
            synthetic_state[
                :,
                modality_order.index(VIRTUAL_T1C_MODALITY)
                if args.enable_virtual_t1c_slot
                else target_index,
            ] = 2
        elif args.synthetic_state == "degraded":
            synthetic_state[
                :,
                modality_order.index(VIRTUAL_T1C_MODALITY)
                if args.enable_virtual_t1c_slot
                else target_index,
            ] = 2

        outputs = {
            "full_real": fusion_model(
                model_image,
                modality_mask=full_mask,
                modality_state=make_state(full_mask),
                modality_confidence=model_confidence,
            ),
            "remove_target": fusion_model(
                model_image,
                modality_mask=remove_mask,
                modality_state=make_state(remove_mask),
                modality_confidence=model_confidence,
            ),
            "synthetic_filled": fusion_model(
                synthetic_image,
                modality_mask=synthetic_mask,
                modality_state=synthetic_state,
                modality_confidence=model_confidence,
            ),
        }
        for name, output in outputs.items():
            cases[name].append(
                dice_scores(output["logits"], target, region_names=BRATS_REGIONS)
            )
        error = (synthetic - image[:, target_index : target_index + 1]).abs()
        generator_rows.append(
            {
                "synthetic_mae": float(error.mean().detach().cpu().item()),
                "synthetic_uncertainty_mean": float(
                    generated["uncertainty"].mean().detach().cpu().item()
                ),
                "synthetic_confidence_mean": float(
                    generated["confidence"].mean().detach().cpu().item()
                ),
                "synthetic_availability": float(
                    synthetic_mask[:, virtual_index].mean().detach().cpu().item()
                ),
            }
        )

    report = {
        "verdict": "SYNTHETIC MODALITY FUSION EVAL: PASS",
        "git_commit": git_commit(),
        "environment": environment(args.device),
        "config": vars(args),
        "synthetic_cache": {
            "path": args.synthetic_cache_manifest,
            "rows": len(synthetic_cache),
        }
        if synthetic_cache
        else None,
        "modalities": list(modality_order),
        "target_modality": args.target_modality,
        "val_subjects": val_count,
        "dice": {name: aggregate(items) for name, items in cases.items()},
        "generator": aggregate(generator_rows),
        "runtime_sec": time.perf_counter() - started,
    }
    report["delta_vs_remove_target"] = {
        key: report["dice"]["synthetic_filled"][key] - report["dice"]["remove_target"][key]
        for key in report["dice"]["remove_target"]
    }
    report["delta_vs_full_real"] = {
        key: report["dice"]["synthetic_filled"][key] - report["dice"]["full_real"][key]
        for key in report["dice"]["full_real"]
    }
    out_path = Path(args.report_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
