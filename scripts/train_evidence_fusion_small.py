from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

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
    modality_mask_from_names,
)
from models.evidence_fusion import EvidenceFusionConfig, EvidenceReliableFusion  # noqa: E402
from models.virtual_modality_generator import (  # noqa: E402
    VirtualModalityGenerator,
    VirtualModalityGeneratorConfig,
    observed_mask_without_target,
)
from scripts.smoke_evidence_fusion_brats import (  # noqa: E402
    environment,
    git_commit,
    trainable_parameters,
)
from utils.brats_metrics import (  # noqa: E402
    counterfactual_logit_delta,
    dice_scores,
    mean_gate_by_region,
    reliability_error_auc,
    reliability_loss,
    segmentation_loss,
)

VIRTUAL_T1C_MODALITY = "virtual_t1c"


def active_modality_order(enable_virtual_t1c_slot: bool) -> tuple[str, ...]:
    if not enable_virtual_t1c_slot:
        return tuple(BRATS_MODALITIES)
    return tuple(BRATS_MODALITIES) + (VIRTUAL_T1C_MODALITY,)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a small local NeuroState evidence-fusion trend experiment."
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
    parser.add_argument("--spatial-size", type=int, default=48)
    parser.add_argument("--max-subjects", type=int, default=40)
    parser.add_argument("--val-subjects", type=int, default=8)
    parser.add_argument("--split-seed", type=int, default=4601)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-train-steps", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--hidden-channels", type=int, default=16)
    parser.add_argument("--enable-region-query-fusion", action="store_true")
    parser.add_argument("--completion-gate-weight", type=float, default=0.0)
    parser.add_argument("--completion-gate-cap", type=float, default=1.0)
    parser.add_argument("--enable-virtual-t1c-confidence", action="store_true")
    parser.add_argument("--virtual-t1c-min-gate-cap", type=float, default=0.05)
    parser.add_argument("--virtual-t1c-max-gate-cap", type=float, default=0.45)
    parser.add_argument("--virtual-t1c-disagreement-scale", type=float, default=0.25)
    parser.add_argument("--modality-confidence-logit-weight", type=float, default=0.0)
    parser.add_argument("--enable-high-res-branch", action="store_true")
    parser.add_argument("--high-res-channels", type=int, default=16)
    parser.add_argument("--semantic-aux-weight", type=float, default=0.1)
    parser.add_argument("--feature-stage", default="stage4")
    parser.add_argument("--feature-channels", type=int, default=512)
    parser.add_argument("--encoder-freeze", default="freeze_all")
    parser.add_argument("--degrade-prob", type=float, default=0.15)
    parser.add_argument("--foreground-crop-prob", type=float, default=0.8)
    parser.add_argument("--crop-mode", default="region_balanced")
    parser.add_argument("--distill-weight", type=float, default=0.2)
    parser.add_argument("--prototype-distill-weight", type=float, default=0.1)
    parser.add_argument("--completion-distill-weight", type=float, default=0.0)
    parser.add_argument("--gate-weight", type=float, default=0.05)
    parser.add_argument("--completion-gate-penalty-weight", type=float, default=0.0)
    parser.add_argument("--virtual-t1c-risk-weight", type=float, default=0.0)
    parser.add_argument("--synthetic-modality-checkpoint", default="")
    parser.add_argument("--synthetic-cache-manifest", default="")
    parser.add_argument("--synthetic-modality-prob", type=float, default=0.0)
    parser.add_argument("--synthetic-target-modality", default="t1c", choices=BRATS_MODALITIES)
    parser.add_argument("--synthetic-min-availability", type=float, default=0.05)
    parser.add_argument("--synthetic-max-availability", type=float, default=0.45)
    parser.add_argument("--synthetic-confidence-temperature", type=float, default=1.0)
    parser.add_argument(
        "--synthetic-image-mode",
        choices=("raw", "confidence_weighted"),
        default="raw",
        help="How to feed generated content into the virtual modality slot.",
    )
    parser.add_argument(
        "--enable-virtual-t1c-slot",
        action="store_true",
        help=(
            "Append an independent virtual_t1c evidence slot instead of writing "
            "synthetic T1c back into the real t1c channel."
        ),
    )
    parser.add_argument("--full-supervision-weight", type=float, default=0.25)
    parser.add_argument("--virtual-evidence-distill-weight", type=float, default=0.0)
    parser.add_argument("--distill-prob", type=float, default=0.7)
    parser.add_argument(
        "--train-mask-policy",
        choices=("dataset", "targeted_t1c"),
        default="dataset",
        help=(
            "dataset uses the dataset's sampled mask; targeted_t1c oversamples "
            "full inputs and the clinically hard remove_t1c case."
        ),
    )
    parser.add_argument("--targeted-missing-t1c-prob", type=float, default=0.0)
    parser.add_argument("--targeted-full-prob", type=float, default=0.0)
    parser.add_argument("--critical-distill-multiplier", type=float, default=1.0)
    parser.add_argument("--resume-checkpoint", default="")
    parser.add_argument("--partial-resume", action="store_true")
    parser.add_argument(
        "--train-scope",
        choices=("all", "completion_query", "fusion_head"),
        default="all",
    )
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=46)
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "outputs" / "evidence_fusion_small"),
    )
    parser.add_argument(
        "--report-path",
        default=str(ROOT / "reports" / "evidence_fusion_small.json"),
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_device(batch: dict[str, Any], device: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in batch.items():
        result[key] = value.to(device) if torch.is_tensor(value) else value
    return result


def nonfinite(value: Any) -> bool:
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, dict):
        return any(nonfinite(item) for item in value.values())
    if isinstance(value, list):
        return any(nonfinite(item) for item in value)
    return False


def make_mask(
    names: list[str],
    batch_size: int,
    device: torch.device,
    modality_order: tuple[str, ...] = BRATS_MODALITIES,
) -> torch.Tensor:
    mask = modality_mask_from_names(names, modality_order=modality_order).to(device)
    return mask.view(1, -1).expand(batch_size, -1).clone()


def make_state(mask: torch.Tensor) -> torch.Tensor:
    state = torch.zeros(mask.shape, dtype=torch.long, device=mask.device)
    state[mask <= 0] = 1
    return state


def full_real_mask(
    batch_size: int,
    device: torch.device,
    modality_order: tuple[str, ...],
) -> torch.Tensor:
    return make_mask(list(BRATS_MODALITIES), batch_size, device, modality_order)


def random_nonempty_mask(batch_size: int, device: torch.device) -> torch.Tensor:
    mask = torch.bernoulli(
        torch.full(
            (batch_size, len(BRATS_MODALITIES)),
            0.5,
            dtype=torch.float32,
            device=device,
        )
    )
    empty = mask.sum(dim=1) <= 0
    if empty.any():
        choices = torch.randint(
            0,
            len(BRATS_MODALITIES),
            (int(empty.sum().item()),),
            device=device,
        )
        mask[empty] = 0.0
        mask[empty, choices] = 1.0
    return mask


def sample_training_mask(
    batch_size: int,
    device: torch.device,
    policy: str,
    missing_t1c_prob: float,
    full_prob: float,
) -> torch.Tensor | None:
    if policy == "dataset":
        return None
    if policy != "targeted_t1c":
        raise ValueError(f"Unknown train mask policy: {policy}")
    missing_t1c_prob = min(max(float(missing_t1c_prob), 0.0), 1.0)
    full_prob = min(max(float(full_prob), 0.0), 1.0 - missing_t1c_prob)
    masks = []
    for _ in range(batch_size):
        draw = random.random()
        if draw < missing_t1c_prob:
            masks.append([1.0, 0.0, 1.0, 1.0])
        elif draw < missing_t1c_prob + full_prob:
            masks.append([1.0, 1.0, 1.0, 1.0])
        else:
            masks.append(random_nonempty_mask(1, device).squeeze(0).tolist())
    return torch.tensor(masks, dtype=torch.float32, device=device)


def apply_training_mask(
    batch: dict[str, Any],
    sampled_mask: torch.Tensor | None,
) -> None:
    if sampled_mask is None:
        return
    old_state = batch["modality_state"]
    state = make_state(sampled_mask)
    state[(sampled_mask > 0) & (old_state == 2)] = 2
    batch["modality_mask"] = sampled_mask
    batch["modality_state"] = state


def load_virtual_modality_generator(
    checkpoint_path: str,
    device: str,
) -> tuple[VirtualModalityGenerator, str]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = payload.get("config", {})
    hidden = int(config.get("hidden_channels", 12))
    target = str(payload.get("target_modality", config.get("target_modality", "t1c")))
    model = VirtualModalityGenerator(
        VirtualModalityGeneratorConfig(hidden_channels=hidden)
    )
    model.load_state_dict(payload["model"], strict=True)
    model.to(device).eval()
    for param in model.parameters():
        param.requires_grad = False
    return model, target


def load_synthetic_cache_manifest(path: str) -> dict[tuple[str, str], dict[str, str]]:
    if not path:
        return {}
    cache_path = Path(path)
    if not cache_path.exists():
        raise FileNotFoundError(f"Synthetic cache manifest not found: {cache_path}")
    with cache_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    cache: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        cache[(row["dataset"], row["subject_id"])] = row
    return cache


def batch_cache_rows(
    batch: dict[str, Any],
    cache: dict[tuple[str, str], dict[str, str]],
) -> list[dict[str, str]] | None:
    if not cache:
        return None
    datasets = batch["dataset"]
    subject_ids = batch["subject_id"]
    rows = []
    for dataset, subject_id in zip(datasets, subject_ids):
        key = (str(dataset), str(subject_id))
        if key not in cache:
            raise KeyError(f"Missing synthetic cache row for {key}")
        rows.append(cache[key])
    return rows


def load_cached_virtual_batch(
    rows: list[dict[str, str]],
    image: torch.Tensor,
    load_confidence: bool = True,
    load_uncertainty: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    synthetic_items = []
    confidence_items = []
    uncertainty_items = []
    for row in rows:
        synthetic_items.append(
            torch.from_numpy(np.load(row["synthetic_path"]).astype(np.float32, copy=False))
        )
        if load_confidence:
            confidence_items.append(
                torch.from_numpy(np.load(row["confidence_path"]).astype(np.float32, copy=False))
            )
        if load_uncertainty:
            uncertainty_items.append(
                torch.from_numpy(np.load(row["uncertainty_path"]).astype(np.float32, copy=False))
            )
    synthetic = torch.stack(synthetic_items, dim=0).unsqueeze(1).to(
        device=image.device,
        dtype=image.dtype,
    )
    if confidence_items:
        confidence = torch.stack(confidence_items, dim=0).unsqueeze(1).to(
            device=image.device,
            dtype=image.dtype,
        )
    else:
        scalar_confidence = []
        for row in rows:
            try:
                value = float(row.get("confidence_mean", 1.0))
            except (TypeError, ValueError):
                value = 1.0
            scalar_confidence.append(max(0.0, min(value, 1.0)))
        confidence = image.new_tensor(scalar_confidence).view(
            len(rows),
            1,
            1,
            1,
            1,
        )
        confidence = confidence.expand(len(rows), 1, *synthetic.shape[-3:])
    if uncertainty_items:
        uncertainty = torch.stack(uncertainty_items, dim=0).unsqueeze(1).to(
            device=image.device,
            dtype=image.dtype,
        )
    else:
        uncertainty = -confidence.clamp_min(1e-6).log()
    if tuple(synthetic.shape[-3:]) != tuple(image.shape[-3:]):
        synthetic = F.interpolate(
            synthetic,
            size=image.shape[-3:],
            mode="trilinear",
            align_corners=False,
        )
        confidence = F.interpolate(
            confidence,
            size=image.shape[-3:],
            mode="trilinear",
            align_corners=False,
        )
        uncertainty = F.interpolate(
            uncertainty,
            size=image.shape[-3:],
            mode="trilinear",
            align_corners=False,
        )
    return synthetic, confidence, uncertainty


@torch.no_grad()
def maybe_apply_synthetic_modality(
    batch: dict[str, Any],
    generator: VirtualModalityGenerator | None,
    cache_rows: list[dict[str, str]] | None,
    target_modality: str,
    probability: float,
    min_availability: float,
    max_availability: float,
    confidence_temperature: float,
    enable_virtual_t1c_slot: bool = False,
    synthetic_image_mode: str = "raw",
) -> dict[str, float]:
    if target_modality not in BRATS_MODALITIES:
        raise ValueError(f"Unknown synthetic target modality: {target_modality}")
    target_index = BRATS_MODALITIES.index(target_modality)
    image = batch["image"]
    if enable_virtual_t1c_slot:
        batch_size = image.shape[0]
        virtual_image = image.new_zeros((batch_size, 1, *image.shape[2:]))
        virtual_mask = image.new_zeros((batch_size, 1))
        virtual_confidence = image.new_zeros((batch_size, 1, *image.shape[2:]))
        virtual_state = torch.ones(
            (batch_size, 1),
            dtype=batch["modality_state"].dtype,
            device=image.device,
        )
        used = image.new_zeros((batch_size,), dtype=torch.bool)
        availability = image.new_zeros((batch_size,))
        if (generator is not None or cache_rows is not None) and probability > 0:
            if cache_rows is not None:
                generated_synthetic, cached_confidence, cached_uncertainty = (
                    load_cached_virtual_batch(
                        cache_rows,
                        image,
                        load_confidence=False,
                        load_uncertainty=False,
                    )
                )
                uncertainty = cached_uncertainty
                confidence = cached_confidence.pow(
                    1.0 / max(float(confidence_temperature), 1e-4)
                )
                virtual_image = generated_synthetic
            else:
                synthetic_input_mask = observed_mask_without_target(
                    batch_size,
                    image.shape[1],
                    target_index,
                    image.device,
                ).to(image.dtype)
                generated = generator(image, synthetic_input_mask, target_index)
                uncertainty = generated["uncertainty"]
                confidence = torch.exp(
                    -uncertainty / max(float(confidence_temperature), 1e-4)
                )
                virtual_image = generated["synthetic"]
                if synthetic_image_mode == "confidence_weighted":
                    virtual_image = virtual_image * confidence
                elif synthetic_image_mode != "raw":
                    raise ValueError(f"Unknown synthetic image mode: {synthetic_image_mode}")
            availability = confidence.mean(dim=(1, 2, 3, 4))
            lo = max(min(float(min_availability), 1.0), 0.0)
            hi = max(min(float(max_availability), 1.0), lo)
            availability = lo + (hi - lo) * availability
            missing_target = batch["modality_mask"][:, target_index] <= 0
            sampled = torch.rand(batch_size, device=image.device) < float(probability)
            used = missing_target & sampled
            virtual_mask[used, 0] = availability[used].to(virtual_mask.dtype)
            virtual_confidence[used] = confidence[used].to(virtual_confidence.dtype)
            virtual_state[used, 0] = 2

        batch["image"] = torch.cat([image, virtual_image], dim=1)
        batch["modality_mask"] = torch.cat(
            [batch["modality_mask"], virtual_mask], dim=1
        )
        batch["modality_state"] = torch.cat(
            [batch["modality_state"], virtual_state], dim=1
        )
        batch["modality_confidence"] = None
        active_availability = availability[used]
        return {
            "used_synthetic_modality": float(used.any().detach().cpu().item()),
            "synthetic_availability": float(
                active_availability.mean().detach().cpu().item()
                if active_availability.numel()
                else 0.0
            ),
        }
    if (generator is None and cache_rows is None) or probability <= 0:
        return {"used_synthetic_modality": 0.0, "synthetic_availability": 0.0}
    if cache_rows is not None:
        synthetic, cached_confidence, cached_uncertainty = load_cached_virtual_batch(
            cache_rows,
            image,
            load_confidence=False,
            load_uncertainty=False,
        )
        uncertainty = cached_uncertainty
        confidence = cached_confidence.pow(
            1.0 / max(float(confidence_temperature), 1e-4)
        )
    else:
        synthetic_input_mask = observed_mask_without_target(
            image.shape[0],
            image.shape[1],
            target_index,
            image.device,
        ).to(image.dtype)
        generated = generator(image, synthetic_input_mask, target_index)
        synthetic = generated["synthetic"]
        uncertainty = generated["uncertainty"]
        confidence = torch.exp(-uncertainty / max(float(confidence_temperature), 1e-4))
    if synthetic_image_mode == "confidence_weighted":
        synthetic = synthetic * confidence
    elif synthetic_image_mode != "raw":
        raise ValueError(f"Unknown synthetic image mode: {synthetic_image_mode}")
    availability = confidence.mean(dim=(1, 2, 3, 4))
    lo = max(min(float(min_availability), 1.0), 0.0)
    hi = max(min(float(max_availability), 1.0), lo)
    availability = lo + (hi - lo) * availability
    mask = batch["modality_mask"].clone()
    state = batch["modality_state"].clone()
    missing_target = mask[:, target_index] <= 0
    sampled = torch.rand(image.shape[0], device=image.device) < float(probability)
    used = missing_target & sampled

    batch["image"] = image.clone()
    if used.any():
        batch["image"][used, target_index : target_index + 1] = synthetic[used]
        mask[used, target_index] = availability[used].to(mask.dtype)
        state[used, target_index] = 2
    batch["modality_mask"] = mask
    batch["modality_state"] = state
    confidence_map = image.new_ones(image.shape)
    if used.any():
        confidence_map[used, target_index : target_index + 1] = confidence[used]
    batch["modality_confidence"] = confidence_map
    active_availability = availability[used]
    return {
        "used_synthetic_modality": float(used.any().detach().cpu().item()),
        "synthetic_availability": float(
            active_availability.mean().detach().cpu().item()
            if active_availability.numel()
            else 0.0
        ),
    }


def is_missing_t1c(mask: torch.Tensor) -> bool:
    t1c_index = BRATS_MODALITIES.index("t1c")
    return bool((mask[:, t1c_index] <= 0).any().item())


def mask_case_name(mask: torch.Tensor) -> str:
    row = mask[0].detach().cpu()
    values = tuple(int(item > 0.5) for item in row[: len(BRATS_MODALITIES)].tolist())
    named = {
        (1, 1, 1, 1): "full",
        (1, 0, 1, 1): "remove_t1c",
        (1, 1, 1, 0): "remove_t2f",
        (1, 0, 0, 0): "t1n_only",
        (0, 1, 0, 0): "t1c_only",
        (0, 0, 1, 0): "t2w_only",
        (0, 0, 0, 1): "t2f_only",
    }
    name = named.get(values, "".join(str(item) for item in values))
    if row.numel() > len(BRATS_MODALITIES) and row[len(BRATS_MODALITIES)] > 0:
        name = f"{name}+{VIRTUAL_T1C_MODALITY}"
    return name


def load_checkpoint_into_model(
    model: EvidenceReliableFusion,
    checkpoint_path: str,
    partial: bool,
) -> dict[str, Any]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = payload["model"]
    if not partial:
        model.load_state_dict(state, strict=True)
        return {
            "path": checkpoint_path,
            "partial": False,
            "loaded_tensors": len(state),
            "skipped_tensors": 0,
        }
    current = model.state_dict()
    compatible = {
        key: value
        for key, value in state.items()
        if key in current and tuple(current[key].shape) == tuple(value.shape)
    }
    skipped = sorted(set(state) - set(compatible))
    merged = dict(current)
    merged.update(compatible)
    model.load_state_dict(merged, strict=True)
    return {
        "path": checkpoint_path,
        "partial": True,
        "loaded_tensors": len(compatible),
        "skipped_tensors": len(skipped),
        "first_skipped": skipped[:20],
    }


def apply_train_scope(model: EvidenceReliableFusion, scope: str) -> None:
    if scope == "all":
        return
    if scope not in {"completion_query", "fusion_head"}:
        raise ValueError(f"Unknown train scope: {scope}")
    for param in model.parameters():
        param.requires_grad = False
    if scope == "completion_query":
        trainable_prefixes = (
            "feature_completion.",
            "region_query_fusers.",
            "seg_decoder.",
            "reliability_decoder.",
        )
    else:
        trainable_prefixes = (
            "modality_embedding.",
            "state_embedding.",
            "stage_adapters.",
            "aux_decoders.",
            "gate_nets.",
            "feature_completion.",
            "region_query_fusers.",
            "seg_decoder.",
            "reliability_decoder.",
            "high_res_branch.",
            "detail_fusion_head.",
        )
    for name, param in model.named_parameters():
        if name.startswith(trainable_prefixes):
            param.requires_grad = True
    if not trainable_parameters(model):
        raise RuntimeError(f"train-scope {scope} left no trainable parameters.")


def loss_for_output(
    output: dict[str, torch.Tensor],
    target: torch.Tensor,
    modality_mask: torch.Tensor,
    reliability_weight: float = 0.2,
    aux_weight: float = 0.15,
    semantic_aux_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    main = segmentation_loss(output["logits"], target)
    semantic_aux = torch.zeros_like(main)
    if semantic_aux_weight > 0 and "semantic_logits" in output:
        semantic_aux = segmentation_loss(output["semantic_logits"], target)
    rel = reliability_loss(
        output["reliability_logits"],
        output["logits"],
        target,
        output["conflict"],
    )
    aux_terms = []
    aux_logits = output["aux_logits"]
    for index in range(aux_logits.shape[1]):
        if modality_mask[:, index].sum() > 0:
            aux_terms.append(segmentation_loss(aux_logits[:, index], target))
    aux = torch.stack(aux_terms).mean() if aux_terms else torch.zeros_like(main)
    total = main + reliability_weight * rel + aux_weight * aux + semantic_aux_weight * semantic_aux
    return total, {
        "loss": float(total.detach().cpu().item()),
        "seg_loss": float(main.detach().cpu().item()),
        "semantic_aux_loss": float(semantic_aux.detach().cpu().item()),
        "reliability_loss": float(rel.detach().cpu().item()),
        "aux_loss": float(aux.detach().cpu().item()),
    }


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    target: torch.Tensor,
    temperature: float = 2.0,
    region_weights: tuple[float, float, float] = (2.0, 1.5, 1.0),
) -> torch.Tensor:
    weights = torch.tensor(
        region_weights,
        dtype=student_logits.dtype,
        device=student_logits.device,
    ).view(1, -1, 1, 1, 1)
    teacher_probs = torch.sigmoid(teacher_logits.detach() / temperature)
    student_probs = torch.sigmoid(student_logits / temperature)
    foreground = target.clamp(0.0, 1.0)
    context = safe_avg_pool3d(foreground, 5).clamp(0.0, 1.0)
    focus = (0.25 + 0.75 * context) * weights
    return ((student_probs - teacher_probs).pow(2) * focus).sum() / focus.sum().clamp_min(1e-8)


def safe_avg_pool3d(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    spatial = tuple(int(item) for item in x.shape[-3:])
    kernel = min(int(kernel_size), *spatial)
    if kernel <= 1:
        return x
    padding = kernel // 2
    pooled = F.avg_pool3d(x, kernel_size=kernel, stride=1, padding=padding)
    if tuple(pooled.shape[-3:]) != spatial:
        pooled = F.interpolate(
            pooled,
            size=spatial,
            mode="trilinear",
            align_corners=False,
        )
    return pooled


def prototype_distillation_loss(
    student_output: dict[str, torch.Tensor],
    teacher_output: dict[str, torch.Tensor],
    target: torch.Tensor,
    region_weights: tuple[float, float, float] = (2.0, 1.5, 1.0),
) -> torch.Tensor:
    student = student_output["fused_feature"]
    teacher = teacher_output["fused_feature"].detach()
    b, channels, d, h, w = student.shape
    regions = target.shape[1]
    if channels % regions != 0:
        return F.mse_loss(
            F.normalize(student, dim=1),
            F.normalize(teacher, dim=1),
        )
    hidden = channels // regions
    student = F.normalize(student.view(b, regions, hidden, d, h, w), dim=2)
    teacher = F.normalize(teacher.view(b, regions, hidden, d, h, w), dim=2)
    target_low = F.interpolate(
        target,
        size=(d, h, w),
        mode="trilinear",
        align_corners=False,
    ).clamp(0.0, 1.0)
    context = safe_avg_pool3d(target_low, 3).clamp(0.0, 1.0)
    weights = torch.tensor(
        region_weights,
        dtype=student.dtype,
        device=student.device,
    ).view(1, regions, 1, 1, 1)
    focus = (0.2 + 0.8 * context) * weights
    voxel_loss = (student - teacher).pow(2).mean(dim=2)
    return (voxel_loss * focus).sum() / focus.sum().clamp_min(1e-8)


def completion_distillation_loss(
    student_output: dict[str, Any],
    teacher_output: dict[str, Any],
    modality_mask: torch.Tensor,
    target: torch.Tensor,
    stage_names: tuple[str, ...] = ("stage1", "stage2", "stage3", "stage4"),
    region_weights: tuple[float, float, float] = (2.0, 1.5, 1.0),
) -> torch.Tensor:
    student_features = student_output.get("features_by_stage", {})
    teacher_features = teacher_output.get("raw_features_by_stage", {})
    if not student_features or not teacher_features:
        reference = student_output["logits"]
        return torch.zeros((), dtype=reference.dtype, device=reference.device)
    missing = (modality_mask <= 0).to(student_output["logits"].dtype)
    if missing.sum() <= 0:
        reference = student_output["logits"]
        return torch.zeros((), dtype=reference.dtype, device=reference.device)

    losses = []
    for stage in stage_names:
        if stage not in student_features or stage not in teacher_features:
            continue
        student = student_features[stage]
        teacher = teacher_features[stage].detach()
        b, m, _, d, h, w = student.shape
        target_low = F.interpolate(
            target,
            size=(d, h, w),
            mode="trilinear",
            align_corners=False,
        ).clamp(0.0, 1.0)
        weights = torch.tensor(
            region_weights,
            dtype=student.dtype,
            device=student.device,
        ).view(1, -1, 1, 1, 1)
        tumor_focus = ((target_low * weights).sum(dim=1, keepdim=True) / weights.sum()).clamp(
            0.0,
            1.0,
        )
        focus = 0.2 + 0.8 * tumor_focus
        missing_view = missing.view(b, m, 1, 1, 1, 1)
        diff = (
            F.normalize(student, dim=2) - F.normalize(teacher, dim=2)
        ).pow(2).mean(dim=2, keepdim=True)
        denom = (focus.unsqueeze(1) * missing_view).sum().clamp_min(1e-8)
        losses.append((diff * focus.unsqueeze(1) * missing_view).sum() / denom)
    if not losses:
        reference = student_output["logits"]
        return torch.zeros((), dtype=reference.dtype, device=reference.device)
    return torch.stack(losses).mean()


def gate_quality_loss(
    output: dict[str, torch.Tensor],
    target: torch.Tensor,
    modality_mask: torch.Tensor,
    temperature: float = 0.25,
    region_weights: tuple[float, float, float] = (2.0, 1.5, 1.0),
) -> torch.Tensor:
    aux_logits = output["aux_logits"]
    gates = output["gates"]
    quality_mask = output.get("fusion_availability_mask", modality_mask)
    b, modalities, regions, d, h, w = aux_logits.shape
    target_for_aux = target.unsqueeze(1)
    context = safe_avg_pool3d(target, 5).clamp(0.0, 1.0)
    weights = torch.tensor(
        region_weights,
        dtype=aux_logits.dtype,
        device=aux_logits.device,
    ).view(1, 1, regions, 1, 1, 1)
    focus = (0.2 + 0.8 * context.unsqueeze(1)) * weights
    aux_error = (torch.sigmoid(aux_logits) - target_for_aux).abs()
    error = (aux_error * focus).sum(dim=(3, 4, 5)) / focus.sum(dim=(3, 4, 5)).clamp_min(1e-8)
    quality = torch.exp(-error / temperature).permute(0, 2, 1)
    quality = quality * quality_mask.view(b, 1, modalities)
    quality = quality / quality.sum(dim=2, keepdim=True).clamp_min(1e-8)

    _, _, _, gd, gh, gw = gates.shape
    target_low = F.interpolate(
        target,
        size=(gd, gh, gw),
        mode="trilinear",
        align_corners=False,
    ).clamp(0.0, 1.0)
    gate_focus = (0.2 + 0.8 * target_low) * weights.view(1, regions, 1, 1, 1)
    gate_mean = (gates * gate_focus.unsqueeze(2)).sum(dim=(3, 4, 5))
    gate_mean = gate_mean / gate_focus.sum(dim=(2, 3, 4)).unsqueeze(2).clamp_min(1e-8)
    return F.mse_loss(gate_mean, quality.detach())


def completion_gate_penalty_loss(
    output: dict[str, torch.Tensor],
    modality_mask: torch.Tensor,
    gate_target: float,
) -> torch.Tensor:
    gates = output["gates"]
    missing = (modality_mask <= 0).to(gates.dtype).view(
        modality_mask.shape[0], 1, -1, 1, 1, 1
    )
    if missing.sum() <= 0:
        return torch.zeros((), dtype=gates.dtype, device=gates.device)
    excess = (gates - float(gate_target)).clamp_min(0.0) * missing
    return excess.pow(2).mean()


def virtual_t1c_risk_loss(
    output: dict[str, torch.Tensor],
    target: torch.Tensor,
    modality_mask: torch.Tensor,
    region_weights: tuple[float, float, float] = (2.5, 2.0, 0.75),
) -> torch.Tensor:
    confidence = output.get("virtual_t1c_confidence")
    aux_logits = output.get("fusion_aux_logits_low")
    if confidence is None or aux_logits is None or "t1c" not in BRATS_MODALITIES:
        reference = output["logits"]
        return torch.zeros((), dtype=reference.dtype, device=reference.device)
    t1c_index = BRATS_MODALITIES.index("t1c")
    missing_t1c = (modality_mask[:, t1c_index] <= 0).to(aux_logits.dtype)
    if missing_t1c.sum() <= 0:
        return torch.zeros((), dtype=aux_logits.dtype, device=aux_logits.device)
    virtual_logits = aux_logits[:, t1c_index]
    _, regions, d, h, w = virtual_logits.shape
    target_low = F.interpolate(
        target,
        size=(d, h, w),
        mode="trilinear",
        align_corners=False,
    ).clamp(0.0, 1.0)
    context = safe_avg_pool3d(target_low, 5).clamp(0.0, 1.0)
    weights = torch.tensor(
        region_weights,
        dtype=virtual_logits.dtype,
        device=virtual_logits.device,
    ).view(1, regions, 1, 1, 1)
    focus = (0.2 + 0.8 * context) * weights
    mask = missing_t1c.view(-1, 1, 1, 1, 1)
    error = (torch.sigmoid(virtual_logits) - target_low).abs()
    risk = error * confidence.detach() * focus * mask
    denom = (focus * mask).sum().clamp_min(1e-8)
    return risk.sum() / denom


def virtual_evidence_distillation_loss(
    student_output: dict[str, torch.Tensor],
    teacher_output: dict[str, torch.Tensor],
    modality_mask: torch.Tensor,
    virtual_modality_name: str = VIRTUAL_T1C_MODALITY,
) -> torch.Tensor:
    aux_logits = student_output.get("aux_logits")
    if aux_logits is None or virtual_modality_name not in student_output.get(
        "modality_names", ()
    ):
        reference = student_output["logits"]
        return torch.zeros((), dtype=reference.dtype, device=reference.device)
    modality_names = tuple(student_output["modality_names"])
    virtual_index = modality_names.index(virtual_modality_name)
    active = (modality_mask[:, virtual_index] > 0).to(aux_logits.dtype)
    if active.sum() <= 0:
        return torch.zeros((), dtype=aux_logits.dtype, device=aux_logits.device)
    teacher_probs = torch.sigmoid(teacher_output["logits"].detach())
    virtual_probs = torch.sigmoid(aux_logits[:, virtual_index])
    active = active.view(-1, 1, 1, 1, 1)
    denom = active.sum().clamp_min(1e-8) * virtual_probs[0].numel()
    return ((virtual_probs - teacher_probs).pow(2) * active).sum() / denom


def train_one_epoch(
    model: EvidenceReliableFusion,
    synthetic_generator: VirtualModalityGenerator | None,
    synthetic_cache: dict[tuple[str, str], dict[str, str]],
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    max_steps: int,
    epoch: int,
    distill_weight: float,
    prototype_distill_weight: float,
    completion_distill_weight: float,
    gate_weight: float,
    completion_gate_penalty_weight: float,
    virtual_t1c_risk_weight: float,
    virtual_evidence_distill_weight: float,
    full_supervision_weight: float,
    distill_prob: float,
    train_mask_policy: str,
    targeted_missing_t1c_prob: float,
    targeted_full_prob: float,
    critical_distill_multiplier: float,
    semantic_aux_weight: float,
    synthetic_target_modality: str,
    synthetic_modality_prob: float,
    synthetic_min_availability: float,
    synthetic_max_availability: float,
    synthetic_confidence_temperature: float,
    synthetic_image_mode: str,
    enable_virtual_t1c_slot: bool,
    modality_order: tuple[str, ...],
    log_every: int,
) -> dict[str, Any]:
    model.train()
    losses = []
    started = time.perf_counter()
    last_log = started
    for step, batch in enumerate(loader):
        if max_steps > 0 and step >= max_steps:
            break
        batch = to_device(batch, device)
        sampled_mask = sample_training_mask(
            batch["image"].shape[0],
            batch["image"].device,
            train_mask_policy,
            targeted_missing_t1c_prob,
            targeted_full_prob,
        )
        apply_training_mask(batch, sampled_mask)
        synthetic_parts = maybe_apply_synthetic_modality(
            batch,
            synthetic_generator,
            batch_cache_rows(batch, synthetic_cache),
            synthetic_target_modality,
            synthetic_modality_prob,
            synthetic_min_availability,
            synthetic_max_availability,
            synthetic_confidence_temperature,
            enable_virtual_t1c_slot,
            synthetic_image_mode,
        )
        output = model(
            batch["image"],
            modality_mask=batch["modality_mask"],
            modality_state=batch["modality_state"],
            modality_confidence=batch.get("modality_confidence"),
        )
        loss, parts = loss_for_output(
            output,
            batch["target"],
            batch["modality_mask"],
            semantic_aux_weight=semantic_aux_weight,
        )
        base_loss = loss
        distill = torch.zeros_like(loss)
        proto = torch.zeros_like(loss)
        completion = torch.zeros_like(loss)
        virtual_evidence_distill = torch.zeros_like(loss)
        gate = gate_quality_loss(output, batch["target"], batch["modality_mask"])
        completion_gate_penalty = completion_gate_penalty_loss(
            output,
            batch["modality_mask"],
            gate_target=min(float(getattr(model.config, "completion_gate_cap", 1.0)), 1.0),
        )
        virtual_t1c_risk = virtual_t1c_risk_loss(
            output,
            batch["target"],
            batch["modality_mask"],
        )
        full_sup = torch.zeros_like(loss)
        critical_missing_t1c = is_missing_t1c(batch["modality_mask"])
        needs_distill = (
            (
                distill_weight > 0
                or prototype_distill_weight > 0
                or completion_distill_weight > 0
                or full_supervision_weight > 0
            )
            and random.random() < distill_prob
            and bool(
                (
                    batch["modality_mask"][:, : len(BRATS_MODALITIES)].sum(dim=1)
                    < len(BRATS_MODALITIES)
                ).any()
            )
        )
        if needs_distill:
            full_mask = full_real_mask(
                batch["image"].shape[0],
                batch["image"].device,
                modality_order,
            )
            teacher = model(
                batch["image"],
                modality_mask=full_mask,
                modality_state=make_state(full_mask),
                modality_confidence=batch.get("modality_confidence"),
            )
            full_sup, full_parts = loss_for_output(
                teacher,
                batch["target"],
                full_mask,
                semantic_aux_weight=semantic_aux_weight,
            )
            distill = distillation_loss(
                output["logits"],
                teacher["logits"],
                batch["target"],
            )
            proto = prototype_distillation_loss(output, teacher, batch["target"])
            completion = completion_distillation_loss(
                output,
                teacher,
                batch["modality_mask"],
                batch["target"],
            )
            if virtual_evidence_distill_weight > 0:
                virtual_evidence_distill = virtual_evidence_distillation_loss(
                    output,
                    teacher,
                    batch["modality_mask"],
                )
            gate = 0.5 * (
                gate + gate_quality_loss(teacher, batch["target"], full_mask)
            )
            distill_multiplier = (
                critical_distill_multiplier if critical_missing_t1c else 1.0
            )
            loss = (
                loss
                + full_supervision_weight * full_sup
                + distill_multiplier * distill_weight * distill
                + distill_multiplier * prototype_distill_weight * proto
                + distill_multiplier * completion_distill_weight * completion
                + distill_multiplier
                * virtual_evidence_distill_weight
                * virtual_evidence_distill
            )
            parts["full_seg_loss"] = full_parts["seg_loss"]
        loss = (
            loss
            + gate_weight * gate
            + completion_gate_penalty_weight * completion_gate_penalty
            + virtual_t1c_risk_weight * virtual_t1c_risk
        )
        parts["base_loss"] = float(base_loss.detach().cpu().item())
        parts["loss"] = float(loss.detach().cpu().item())
        parts["distill_loss"] = float(distill.detach().cpu().item())
        parts["prototype_distill_loss"] = float(proto.detach().cpu().item())
        parts["completion_distill_loss"] = float(completion.detach().cpu().item())
        parts["virtual_evidence_distill_loss"] = float(
            virtual_evidence_distill.detach().cpu().item()
        )
        parts["gate_loss"] = float(gate.detach().cpu().item())
        parts["completion_gate_penalty_loss"] = float(
            completion_gate_penalty.detach().cpu().item()
        )
        parts["virtual_t1c_risk_loss"] = float(
            virtual_t1c_risk.detach().cpu().item()
        )
        parts["full_supervision_loss"] = float(full_sup.detach().cpu().item())
        parts["used_distill"] = bool(needs_distill)
        parts["critical_missing_t1c"] = bool(critical_missing_t1c)
        parts["mask_case"] = mask_case_name(batch["modality_mask"])
        parts.update(synthetic_parts)
        if not torch.isfinite(loss).item():
            raise RuntimeError(f"Non-finite loss at epoch {epoch}, step {step}: {parts}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters(model), 1.0)
        if not torch.isfinite(grad_norm).item():
            raise RuntimeError(f"Non-finite grad norm at epoch {epoch}, step {step}")
        optimizer.step()
        parts["grad_norm"] = float(grad_norm.detach().cpu().item())
        losses.append(parts)
        if log_every > 0 and (step + 1) % log_every == 0:
            now = time.perf_counter()
            print(
                json.dumps(
                    {
                        "epoch": epoch,
                        "step": step + 1,
                        "loss": parts["loss"],
                        "seg_loss": parts["seg_loss"],
                        "distill_loss": parts["distill_loss"],
                        "prototype_distill_loss": parts[
                            "prototype_distill_loss"
                        ],
                        "completion_distill_loss": parts[
                            "completion_distill_loss"
                        ],
                        "virtual_evidence_distill_loss": parts[
                            "virtual_evidence_distill_loss"
                        ],
                        "gate_loss": parts["gate_loss"],
                        "completion_gate_penalty_loss": parts[
                            "completion_gate_penalty_loss"
                        ],
                        "virtual_t1c_risk_loss": parts[
                            "virtual_t1c_risk_loss"
                        ],
                        "used_distill": parts["used_distill"],
                        "mask_case": parts["mask_case"],
                        "used_synthetic_modality": parts[
                            "used_synthetic_modality"
                        ],
                        "synthetic_availability": parts[
                            "synthetic_availability"
                        ],
                        "elapsed_sec": now - started,
                        "sec_since_last_log": now - last_log,
                    }
                )
            )
            last_log = now
    if not losses:
        raise RuntimeError("No training steps were executed.")
    return {
        "epoch": epoch,
        "steps": len(losses),
        "mean_loss": float(np.mean([item["loss"] for item in losses])),
        "mean_seg_loss": float(np.mean([item["seg_loss"] for item in losses])),
        "mean_reliability_loss": float(
            np.mean([item["reliability_loss"] for item in losses])
        ),
        "mean_semantic_aux_loss": float(
            np.mean([item["semantic_aux_loss"] for item in losses])
        ),
        "mean_aux_loss": float(np.mean([item["aux_loss"] for item in losses])),
        "mean_distill_loss": float(
            np.mean([item["distill_loss"] for item in losses])
        ),
        "mean_prototype_distill_loss": float(
            np.mean([item["prototype_distill_loss"] for item in losses])
        ),
        "mean_completion_distill_loss": float(
            np.mean([item["completion_distill_loss"] for item in losses])
        ),
        "mean_virtual_evidence_distill_loss": float(
            np.mean([item["virtual_evidence_distill_loss"] for item in losses])
        ),
        "mean_gate_loss": float(np.mean([item["gate_loss"] for item in losses])),
        "mean_completion_gate_penalty_loss": float(
            np.mean([item["completion_gate_penalty_loss"] for item in losses])
        ),
        "mean_virtual_t1c_risk_loss": float(
            np.mean([item["virtual_t1c_risk_loss"] for item in losses])
        ),
        "mean_full_supervision_loss": float(
            np.mean([item["full_supervision_loss"] for item in losses])
        ),
        "distill_step_fraction": float(
            np.mean([1.0 if item["used_distill"] else 0.0 for item in losses])
        ),
        "critical_missing_t1c_fraction": float(
            np.mean(
                [1.0 if item["critical_missing_t1c"] else 0.0 for item in losses]
            )
        ),
        "synthetic_modality_fraction": float(
            np.mean([item["used_synthetic_modality"] for item in losses])
        ),
        "mean_synthetic_availability": float(
            np.mean([item["synthetic_availability"] for item in losses])
        ),
        "mask_case_counts": {
            name: sum(1 for item in losses if item["mask_case"] == name)
            for name in sorted({item["mask_case"] for item in losses})
        },
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "runtime_sec": time.perf_counter() - started,
    }


@torch.no_grad()
def evaluate_case(
    model: EvidenceReliableFusion,
    image: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    modality_order: tuple[str, ...],
    modality_confidence: torch.Tensor | None = None,
    return_logits: bool = False,
) -> dict[str, Any]:
    output = model(
        image,
        modality_mask=mask,
        modality_state=make_state(mask),
        modality_confidence=modality_confidence,
    )
    result = {
        "dice": dice_scores(output["logits"], target, region_names=BRATS_REGIONS),
        "reliability_error_auc": reliability_error_auc(
            output["reliability"], output["logits"], target
        ),
        "reliability_mean": float(output["reliability"].mean().detach().cpu().item()),
        "conflict_mean": float(output["conflict"].mean().detach().cpu().item()),
        "gate_means": mean_gate_by_region(
            output["gates"],
            output.get("fusion_availability_mask", mask),
            BRATS_REGIONS,
            modality_order,
        ),
    }
    if output.get("virtual_t1c_confidence") is not None:
        result["virtual_t1c_confidence_mean"] = float(
            output["virtual_t1c_confidence"].mean().detach().cpu().item()
        )
        result["virtual_t1c_disagreement_mean"] = float(
            output["virtual_t1c_disagreement"].mean().detach().cpu().item()
        )
        result["virtual_t1c_gate_cap_mean"] = float(
            output["virtual_t1c_gate_cap"].mean().detach().cpu().item()
        )
    if return_logits:
        result["logits"] = output["logits"].detach()
    return result


@torch.no_grad()
def evaluate(
    model: EvidenceReliableFusion,
    loader: DataLoader,
    device: str,
    modality_order: tuple[str, ...],
    log_every: int = 0,
) -> dict[str, Any]:
    model.eval()
    cases = {
        "full": list(BRATS_MODALITIES),
        "t1n_only": ["t1n"],
        "t1c_only": ["t1c"],
        "t2w_only": ["t2w"],
        "t2f_only": ["t2f"],
        "remove_t1c": ["t1n", "t2w", "t2f"],
        "remove_t2f": ["t1n", "t1c", "t2w"],
    }
    bucket: dict[str, list[dict[str, Any]]] = {name: [] for name in cases}
    deltas: dict[str, list[dict[str, float]]] = {"remove_t1c": [], "remove_t2f": []}
    started = time.perf_counter()
    for step, batch in enumerate(loader):
        batch = to_device(batch, device)
        image = batch["image"]
        modality_confidence = None
        if len(modality_order) > len(BRATS_MODALITIES):
            virtual = image.new_zeros((image.shape[0], 1, *image.shape[2:]))
            image = torch.cat([image, virtual], dim=1)
            real_confidence = image.new_ones(
                (image.shape[0], len(BRATS_MODALITIES), *image.shape[2:])
            )
            virtual_confidence = image.new_zeros((image.shape[0], 1, *image.shape[2:]))
            modality_confidence = torch.cat(
                [real_confidence, virtual_confidence],
                dim=1,
            )
        target = batch["target"]
        full_mask = full_real_mask(image.shape[0], image.device, modality_order)
        full_output = evaluate_case(
            model,
            image,
            target,
            full_mask,
            modality_order,
            modality_confidence=modality_confidence,
            return_logits=True,
        )
        bucket["full"].append(full_output)
        full_logits = full_output.pop("logits")
        for name, modalities in cases.items():
            if name == "full":
                continue
            mask = make_mask(modalities, image.shape[0], image.device, modality_order)
            need_delta = name in deltas
            item = evaluate_case(
                model,
                image,
                target,
                mask,
                modality_order,
                modality_confidence=modality_confidence,
                return_logits=need_delta,
            )
            bucket[name].append(item)
            if need_delta:
                item_logits = item.pop("logits")
                deltas[name].append(
                    counterfactual_logit_delta(
                        full_logits,
                        item_logits,
                        target,
                        region_names=BRATS_REGIONS,
                    )
                )
                del item_logits
        del full_logits
        if log_every > 0 and (step + 1) % log_every == 0:
            print(
                json.dumps(
                    {
                        "eval_step": step + 1,
                        "elapsed_sec": time.perf_counter() - started,
                    }
                )
            )

    def aggregate(items: list[dict[str, Any]]) -> dict[str, Any]:
        dice_keys = list(items[0]["dice"])
        gate_keys = list(items[0]["gate_means"])
        item_report = {
            "dice": {
                key: float(np.mean([item["dice"][key] for item in items]))
                for key in dice_keys
            },
            "reliability_error_auc": float(
                np.nanmean([item["reliability_error_auc"] for item in items])
            ),
            "reliability_mean": float(
                np.mean([item["reliability_mean"] for item in items])
            ),
            "conflict_mean": float(np.mean([item["conflict_mean"] for item in items])),
            "gate_means": {
                key: float(np.mean([item["gate_means"][key] for item in items]))
                for key in gate_keys
            },
        }
        for key in (
            "virtual_t1c_confidence_mean",
            "virtual_t1c_disagreement_mean",
            "virtual_t1c_gate_cap_mean",
        ):
            if key in items[0]:
                item_report[key] = float(np.mean([item[key] for item in items]))
        return item_report

    report = {name: aggregate(items) for name, items in bucket.items()}
    for name, items in deltas.items():
        if not items:
            continue
        keys = list(items[0])
        report[name]["logit_delta_vs_full"] = {
            key: float(np.mean([item[key] for item in items])) for key in keys
        }
    report["runtime_sec"] = time.perf_counter() - started
    return report


def main() -> int:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false.")
    set_seed(args.seed)
    modality_order = active_modality_order(args.enable_virtual_t1c_slot)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    max_subjects = None if args.max_subjects <= 0 else args.max_subjects
    dataset = BraTSFusionDataset(
        manifest_csv=args.manifest,
        spatial_size=args.spatial_size,
        max_subjects=max_subjects,
        modality_mask_mode="random_nonempty",
        degrade_prob=args.degrade_prob,
        foreground_crop_prob=args.foreground_crop_prob,
        crop_mode=args.crop_mode,
        seed=args.seed,
    )
    val_count = min(args.val_subjects, max(1, len(dataset) // 4))
    train_count = len(dataset) - val_count
    if train_count <= 0:
        raise ValueError("Need at least one training subject after val split.")
    indices = list(range(len(dataset)))
    random.Random(int(args.split_seed)).shuffle(indices)
    val_indices = sorted(indices[:val_count])
    train_indices = sorted(indices[val_count:])
    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)
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

    model = EvidenceReliableFusion(
        EvidenceFusionConfig(
            modalities=modality_order,
            checkpoint_path=args.checkpoint,
            encoder_freeze=args.encoder_freeze,
            feature_stage=args.feature_stage,
            feature_channels=args.feature_channels,
            hidden_channels=args.hidden_channels,
            enable_region_query_fusion=args.enable_region_query_fusion,
            completion_gate_weight=args.completion_gate_weight,
            completion_gate_cap=args.completion_gate_cap,
            enable_virtual_t1c_confidence=args.enable_virtual_t1c_confidence,
            virtual_t1c_min_gate_cap=args.virtual_t1c_min_gate_cap,
            virtual_t1c_max_gate_cap=args.virtual_t1c_max_gate_cap,
            virtual_t1c_disagreement_scale=args.virtual_t1c_disagreement_scale,
            modality_confidence_logit_weight=args.modality_confidence_logit_weight,
            enable_high_res_branch=args.enable_high_res_branch,
            high_res_channels=args.high_res_channels,
        )
    ).to(args.device)
    resume_report = None
    if args.resume_checkpoint:
        resume_report = load_checkpoint_into_model(
            model,
            args.resume_checkpoint,
            partial=args.partial_resume,
        )
    apply_train_scope(model, args.train_scope)
    synthetic_generator = None
    synthetic_generator_report = None
    synthetic_cache = load_synthetic_cache_manifest(args.synthetic_cache_manifest)
    synthetic_cache_report = None
    if synthetic_cache:
        synthetic_cache_report = {
            "path": args.synthetic_cache_manifest,
            "rows": len(synthetic_cache),
        }
    if args.synthetic_modality_checkpoint:
        synthetic_generator, synthetic_target = load_virtual_modality_generator(
            args.synthetic_modality_checkpoint,
            args.device,
        )
        if synthetic_target != args.synthetic_target_modality:
            raise ValueError(
                "Synthetic generator target mismatch: "
                f"checkpoint={synthetic_target}, requested={args.synthetic_target_modality}"
            )
        synthetic_generator_report = {
            "path": args.synthetic_modality_checkpoint,
            "target_modality": synthetic_target,
        }
    optimizer = torch.optim.AdamW(trainable_parameters(model), lr=args.lr)
    started = time.perf_counter()
    epoch_reports = []
    eval_reports = []
    for epoch in range(args.epochs):
        dataset.set_epoch(epoch)
        train_report = train_one_epoch(
            model,
            synthetic_generator,
            synthetic_cache,
            train_loader,
            optimizer,
            args.device,
            args.max_train_steps,
            epoch,
            args.distill_weight,
            args.prototype_distill_weight,
            args.completion_distill_weight,
            args.gate_weight,
            args.completion_gate_penalty_weight,
            args.virtual_t1c_risk_weight,
            args.virtual_evidence_distill_weight,
            args.full_supervision_weight,
            args.distill_prob,
            args.train_mask_policy,
            args.targeted_missing_t1c_prob,
            args.targeted_full_prob,
            args.critical_distill_multiplier,
            args.semantic_aux_weight,
            args.synthetic_target_modality,
            args.synthetic_modality_prob,
            args.synthetic_min_availability,
            args.synthetic_max_availability,
            args.synthetic_confidence_temperature,
            args.synthetic_image_mode,
            args.enable_virtual_t1c_slot,
            modality_order,
            args.log_every,
        )
        eval_report = evaluate(
            model,
            val_loader,
            args.device,
            modality_order,
            args.log_every,
        )
        epoch_reports.append(train_report)
        eval_reports.append({"epoch": epoch, "metrics": eval_report})
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "mean_loss": train_report["mean_loss"],
                    "full_dice_mean": eval_report["full"]["dice"]["dice_mean"],
                    "remove_t1c_dice_mean": eval_report["remove_t1c"]["dice"][
                        "dice_mean"
                    ],
                    "remove_t2f_dice_mean": eval_report["remove_t2f"]["dice"][
                        "dice_mean"
                    ],
                },
                indent=2,
            )
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "evidence_fusion_small_last.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "config": vars(args),
            "modalities": modality_order,
            "regions": BRATS_REGIONS,
        },
        checkpoint_path,
    )
    memory = None
    if args.device.startswith("cuda") and torch.cuda.is_available():
        memory = {
            "peak_allocated_mb": torch.cuda.max_memory_allocated() / (1024**2),
            "peak_reserved_mb": torch.cuda.max_memory_reserved() / (1024**2),
        }
    if eval_reports:
        final_eval = eval_reports[-1]["metrics"]
    else:
        final_eval = evaluate(
            model,
            val_loader,
            args.device,
            modality_order,
            log_every=max(args.log_every, 0),
        )
    first_loss = epoch_reports[0]["first_loss"]["loss"]
    last_loss = epoch_reports[-1]["last_loss"]["loss"]
    report = {
        "verdict": "EVIDENCE FUSION SMALL: PASS",
        "git_commit": git_commit(),
        "environment": environment(args.device),
        "config": vars(args),
        "resume": resume_report,
        "synthetic_generator": synthetic_generator_report,
        "synthetic_cache": synthetic_cache_report,
        "train_subjects": train_count,
        "val_subjects": val_count,
        "modalities": list(modality_order),
        "regions": list(BRATS_REGIONS),
        "loss_trend": {
            "first_step_loss": first_loss,
            "last_step_loss": last_loss,
            "relative_change": (last_loss - first_loss) / max(abs(first_loss), 1e-8),
        },
        "epochs": epoch_reports,
        "eval_by_epoch": eval_reports,
        "final_eval": final_eval,
        "checkpoint_path": str(checkpoint_path),
        "memory": memory,
        "runtime_sec": time.perf_counter() - started,
    }
    if nonfinite(report):
        report["verdict"] = "EVIDENCE FUSION SMALL: FAIL_NONFINITE"
    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 1 if report["verdict"].endswith("FAIL_NONFINITE") else 0


if __name__ == "__main__":
    raise SystemExit(main())
