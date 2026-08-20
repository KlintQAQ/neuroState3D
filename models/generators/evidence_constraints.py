from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


def _with_sample_axis(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 5:
        return tensor[:, None]
    if tensor.ndim == 6:
        return tensor
    raise ValueError("Expected [B,C,D,H,W] or [B,N,C,D,H,W]")


def evidence_contradiction_score(
    predicted_observed: torch.Tensor,
    observed_target: torch.Tensor,
    reliability: Optional[torch.Tensor] = None,
    uncertainty: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Score confident disagreement with actually observed evidence.

    Returns ``[B,N]`` scores for one or more candidate latents/images. A high
    score means the candidate contradicts a reliable observation while not
    admitting high uncertainty. This is a training diagnostic, not a clinical
    confidence measure.
    """

    predicted = _with_sample_axis(predicted_observed).float()
    if observed_target.ndim != 5:
        raise ValueError("observed_target must be [B,C,D,H,W]")
    if predicted.shape[0] != observed_target.shape[0]:
        raise ValueError("predicted_observed and observed_target batch sizes differ")
    if predicted.shape[2:] != observed_target.shape[1:]:
        raise ValueError("predicted_observed and observed_target shapes differ")

    weight = torch.ones_like(predicted)
    if reliability is not None:
        if reliability.ndim != 5:
            raise ValueError("reliability must be [B,C|1,D,H,W]")
        weight = weight * reliability[:, None].float().clamp(0, 1)
    if uncertainty is not None:
        uncertainty_samples = _with_sample_axis(uncertainty).float()
        weight = weight * (1 - uncertainty_samples.clamp(0, 1))

    error = F.smooth_l1_loss(
        predicted,
        observed_target[:, None].expand_as(predicted),
        reduction="none",
    )
    weighted_error = error * weight
    denominator = weight.sum(dim=(2, 3, 4, 5)).clamp_min(eps)
    return weighted_error.sum(dim=(2, 3, 4, 5)) / denominator


def contradiction_negative_weights(
    scores: torch.Tensor,
    strength: float = 2.0,
    max_weight: float = 6.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Convert contradiction scores into bounded repulsive weights."""

    if scores.ndim == 1:
        scores = scores[:, None]
    if scores.ndim != 2:
        raise ValueError("scores must be [B] or [B,N]")
    if strength < 0:
        raise ValueError("strength must be non-negative")
    normalized = scores.float() / scores.float().mean(dim=1, keepdim=True).clamp_min(eps)
    return (1.0 + strength * normalized).clamp(max=max_weight)


@dataclass(frozen=True)
class ContradictionBatch:
    latents: torch.Tensor
    scores: torch.Tensor
    target_ids: torch.Tensor


class EvidenceContradictionBank:
    """Small CPU hard-negative bank keyed by target modality.

    Candidates with the highest observed-evidence contradiction are retained.
    The bank deliberately stores detached tensors and never labels them as
    observed evidence.
    """

    def __init__(self, capacity: int = 128) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self._latents: Optional[torch.Tensor] = None
        self._scores: Optional[torch.Tensor] = None
        self._target_ids: Optional[torch.Tensor] = None

    def __len__(self) -> int:
        return 0 if self._scores is None else int(self._scores.numel())

    def add(
        self,
        latents: torch.Tensor,
        scores: torch.Tensor,
        target_ids: Optional[torch.Tensor] = None,
    ) -> None:
        candidates = _with_sample_axis(latents).detach().cpu()
        batch, count = candidates.shape[:2]
        if scores.ndim == 1:
            scores = scores[:, None]
        if scores.shape != (batch, count):
            raise ValueError(
                f"scores must have shape {(batch, count)}, got {tuple(scores.shape)}"
            )
        if target_ids is None:
            target_ids = torch.zeros(batch, dtype=torch.long)
        if target_ids.shape != (batch,):
            raise ValueError("target_ids must be [B]")
        expanded_targets = target_ids[:, None].expand(batch, count).reshape(-1).cpu()
        flat_latents = candidates.reshape(-1, *candidates.shape[2:])
        flat_scores = scores.detach().float().reshape(-1).cpu()

        if self._latents is None:
            all_latents = flat_latents
            all_scores = flat_scores
            all_targets = expanded_targets
        else:
            if flat_latents.shape[1:] != self._latents.shape[1:]:
                raise ValueError("All bank latents must share the same shape")
            all_latents = torch.cat([self._latents, flat_latents], dim=0)
            all_scores = torch.cat([self._scores, flat_scores], dim=0)
            all_targets = torch.cat([self._target_ids, expanded_targets], dim=0)

        keep = torch.topk(all_scores, k=min(self.capacity, all_scores.numel())).indices
        self._latents = all_latents[keep]
        self._scores = all_scores[keep]
        self._target_ids = all_targets[keep]

    def sample(
        self,
        target_ids: torch.Tensor,
        n_samples: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> ContradictionBatch:
        if self._latents is None or self._scores is None or self._target_ids is None:
            raise RuntimeError("EvidenceContradictionBank is empty")
        if target_ids.ndim != 1:
            raise ValueError("target_ids must be [B]")
        if n_samples < 1:
            raise ValueError("n_samples must be positive")

        sampled_latents = []
        sampled_scores = []
        for target_id in target_ids.detach().cpu().long():
            matching = torch.nonzero(self._target_ids == target_id, as_tuple=False).flatten()
            if matching.numel() == 0:
                matching = torch.arange(len(self))
            ordered = matching[torch.argsort(self._scores[matching], descending=True)]
            if ordered.numel() < n_samples:
                repetitions = (n_samples + ordered.numel() - 1) // ordered.numel()
                ordered = ordered.repeat(repetitions)
            chosen = ordered[:n_samples]
            sampled_latents.append(self._latents[chosen])
            sampled_scores.append(self._scores[chosen])
        latents = torch.stack(sampled_latents).to(device=device, dtype=dtype)
        scores = torch.stack(sampled_scores).to(device=device, dtype=torch.float32)
        return ContradictionBatch(
            latents=latents,
            scores=scores,
            target_ids=target_ids.to(device=device, dtype=torch.long),
        )


def validate_nested_masks(
    coarse_mask: torch.Tensor,
    richer_mask: torch.Tensor,
) -> None:
    if coarse_mask.shape != richer_mask.shape or coarse_mask.ndim != 2:
        raise ValueError("coarse_mask and richer_mask must share [B,M] shape")
    coarse = coarse_mask > 0
    richer = richer_mask > 0
    if torch.any(coarse & ~richer):
        raise ValueError("coarse_mask must be a subset of richer_mask")
    if torch.any(richer.sum(dim=1) <= coarse.sum(dim=1)):
        raise ValueError("Every richer_mask row must add at least one modality")


def nested_subset_consistency_loss(
    coarse_samples: torch.Tensor,
    richer_samples: torch.Tensor,
    coarse_mask: torch.Tensor,
    richer_mask: torch.Tensor,
    pool_size: int = 2,
    contraction_margin: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Preserve identity and encourage uncertainty contraction with evidence.

    For identity, ``richer_samples`` are detached as a better-evidenced teacher
    and only low-frequency 3D structure is matched. For contraction, coarse
    variance is the detached upper bound and gradients reduce richer variance
    when modalities are added.
    """

    if coarse_samples.shape != richer_samples.shape or coarse_samples.ndim != 6:
        raise ValueError("samples must share [B,N,C,D,H,W] shape")
    validate_nested_masks(coarse_mask, richer_mask)
    if pool_size < 1:
        raise ValueError("pool_size must be positive")

    batch, count, channels, depth, height, width = coarse_samples.shape
    coarse_flat = coarse_samples.reshape(batch * count, channels, depth, height, width)
    richer_flat = richer_samples.detach().reshape(
        batch * count, channels, depth, height, width
    )
    if pool_size > 1:
        kernel = tuple(min(pool_size, size) for size in (depth, height, width))
        coarse_identity = F.avg_pool3d(coarse_flat, kernel_size=kernel, stride=kernel)
        richer_identity = F.avg_pool3d(richer_flat, kernel_size=kernel, stride=kernel)
    else:
        coarse_identity = coarse_flat
        richer_identity = richer_flat
    identity_loss = F.smooth_l1_loss(coarse_identity, richer_identity)

    coarse_variance = coarse_samples.detach().float().var(dim=1, unbiased=False)
    richer_variance = richer_samples.float().var(dim=1, unbiased=False)
    contraction_loss = F.relu(
        richer_variance - coarse_variance + float(contraction_margin)
    ).mean()
    diagnostics = {
        "coarse_variance": coarse_variance.mean().detach(),
        "richer_variance": richer_variance.mean().detach(),
        "added_modalities": (
            (richer_mask > 0).sum(dim=1) - (coarse_mask > 0).sum(dim=1)
        ).float().mean().detach(),
    }
    return identity_loss, contraction_loss, diagnostics
