from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F


def multilabel_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1.0,
) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    dims = tuple(range(2, probs.ndim))
    intersection = (probs * target).sum(dim=dims)
    denom = probs.pow(2).sum(dim=dims) + target.pow(2).sum(dim=dims)
    dice = (2 * intersection + smooth) / (denom + smooth)
    return 1.0 - dice.mean()


def segmentation_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return multilabel_dice_loss(logits, target) + F.binary_cross_entropy_with_logits(
        logits, target
    )


def dice_scores(
    logits: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    region_names: Sequence[str] = ("ET", "TC", "WT"),
) -> dict[str, float]:
    probs = torch.sigmoid(logits)
    pred = probs > threshold
    truth = target > 0.5
    scores: dict[str, float] = {}
    for index, name in enumerate(region_names):
        pred_k = pred[:, index]
        truth_k = truth[:, index]
        intersection = (pred_k & truth_k).sum().to(torch.float32)
        denom = pred_k.sum().to(torch.float32) + truth_k.sum().to(torch.float32)
        if denom.item() == 0:
            dice = torch.tensor(1.0, device=logits.device)
        else:
            dice = 2 * intersection / denom
        scores[f"dice_{name}"] = float(dice.detach().cpu().item())
    scores["dice_mean"] = sum(scores.values()) / max(len(scores), 1)
    return scores


def reliability_targets(
    logits: torch.Tensor,
    target: torch.Tensor,
    conflict: torch.Tensor | None = None,
) -> torch.Tensor:
    with torch.no_grad():
        probs = torch.sigmoid(logits)
        absolute_error = (probs - target).abs()
        reliable = 1.0 - absolute_error.clamp(0.0, 1.0)
        if conflict is not None:
            reliable = reliable * (1.0 - conflict.detach().clamp(0.0, 1.0))
        return reliable.clamp(0.0, 1.0)


def reliability_loss(
    reliability_logits: torch.Tensor,
    logits: torch.Tensor,
    target: torch.Tensor,
    conflict: torch.Tensor | None = None,
) -> torch.Tensor:
    reliable = reliability_targets(logits.detach(), target, conflict)
    return F.binary_cross_entropy_with_logits(reliability_logits, reliable)


def reliability_error_auc(
    reliability: torch.Tensor,
    logits: torch.Tensor,
    target: torch.Tensor,
) -> float:
    """Return AUC for using low reliability to predict voxel errors.

    This is intentionally lightweight and dependency-free for smoke tests.
    """

    with torch.no_grad():
        pred = (torch.sigmoid(logits) > 0.5).to(torch.float32)
        error = (pred != (target > 0.5)).to(torch.float32).flatten()
        risk = (1.0 - reliability).flatten()
        positives = error > 0.5
        negatives = ~positives
        n_pos = int(positives.sum().item())
        n_neg = int(negatives.sum().item())
        if n_pos == 0 or n_neg == 0:
            return float("nan")
        order = torch.argsort(risk)
        ranks = torch.empty_like(order, dtype=torch.float32)
        ranks[order] = torch.arange(1, risk.numel() + 1, device=risk.device).to(
            torch.float32
        )
        pos_rank_sum = ranks[positives].sum()
        auc = (pos_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
        return float(auc.detach().cpu().item())


def mean_gate_by_region(
    gates: torch.Tensor,
    modality_mask: torch.Tensor,
    region_names: Sequence[str],
    modality_names: Sequence[str],
) -> dict[str, float]:
    """Average class-conditioned gate values over the spatial grid."""

    with torch.no_grad():
        valid = modality_mask.view(modality_mask.shape[0], 1, -1, 1, 1, 1)
        gates = gates * valid
        values = gates.mean(dim=(0, 3, 4, 5))
        result: dict[str, float] = {}
        for region_index, region in enumerate(region_names):
            for modality_index, modality in enumerate(modality_names):
                result[f"gate_{region}_{modality}"] = float(
                    values[region_index, modality_index].detach().cpu().item()
                )
        return result


def counterfactual_logit_delta(
    full_logits: torch.Tensor,
    missing_logits: torch.Tensor,
    target: torch.Tensor,
    region_names: Sequence[str] = ("ET", "TC", "WT"),
) -> dict[str, float]:
    """Measure how much a prediction changes when a modality is removed."""

    with torch.no_grad():
        full_probs = torch.sigmoid(full_logits)
        missing_probs = torch.sigmoid(missing_logits)
        delta = (full_probs - missing_probs).abs()
        result: dict[str, float] = {}
        for index, region in enumerate(region_names):
            region_target = target[:, index] > 0.5
            if region_target.any():
                value = delta[:, index][region_target].mean()
            else:
                value = delta[:, index].mean()
            result[f"delta_{region}"] = float(value.detach().cpu().item())
        result["delta_mean"] = sum(result.values()) / max(len(result), 1)
        return result
