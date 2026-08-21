from __future__ import annotations

from collections.abc import Sequence

import torch


def _as_weight(
    weight: torch.Tensor | None,
    reference: torch.Tensor,
) -> torch.Tensor:
    if weight is not None:
        return weight.to(device=reference.device, dtype=reference.dtype)
    return torch.ones(
        reference.shape[:2],
        dtype=reference.dtype,
        device=reference.device,
    )


def pairwise_l2(
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Return pairwise L2 distances for [B, N, C] and [B, M, C] tensors."""

    if x.ndim != 3 or y.ndim != 3:
        raise ValueError("pairwise_l2 expects x and y with shape [B, tokens, dim].")
    if x.shape[0] != y.shape[0] or x.shape[2] != y.shape[2]:
        raise ValueError(
            "pairwise_l2 batch and feature dimensions must match: "
            f"x={tuple(x.shape)}, y={tuple(y.shape)}"
        )
    sq = (
        x.square().sum(dim=2, keepdim=True)
        + y.square().sum(dim=2).unsqueeze(1)
        - 2.0 * torch.bmm(x, y.transpose(1, 2))
    )
    return sq.clamp_min(float(eps)).sqrt()


def drift_loss(
    generated: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor | None = None,
    weight_generated: torch.Tensor | None = None,
    weight_positive: torch.Tensor | None = None,
    weight_negative: torch.Tensor | None = None,
    radii: Sequence[float] = (0.02, 0.05, 0.2),
    eps: float = 1e-8,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """PyTorch version of the Drifting loss used by the official JAX release.

    The generated tokens receive gradients. Positive and negative tokens are
    treated as fixed anchors, matching the stop-gradient design in the paper.
    Positive anchors pull generated samples toward the target distribution;
    negative anchors repel shortcuts such as copying source modalities.
    """

    if generated.ndim != 3 or positive.ndim != 3:
        raise ValueError("generated and positive must have shape [B, tokens, dim].")
    if generated.shape[0] != positive.shape[0] or generated.shape[2] != positive.shape[2]:
        raise ValueError(
            "generated and positive batch/feature dimensions must match: "
            f"generated={tuple(generated.shape)}, positive={tuple(positive.shape)}"
        )
    if negative is None:
        negative = generated[:, :0].detach()
    if negative.ndim != 3:
        raise ValueError("negative must have shape [B, tokens, dim].")
    if negative.shape[0] != generated.shape[0] or negative.shape[2] != generated.shape[2]:
        raise ValueError(
            "negative batch/feature dimensions must match generated: "
            f"generated={tuple(generated.shape)}, negative={tuple(negative.shape)}"
        )
    if not radii:
        raise ValueError("radii must contain at least one kernel radius.")

    generated = generated.float()
    old_generated = generated.detach()
    positive = positive.detach().float()
    negative = negative.detach().float()

    weight_generated = _as_weight(weight_generated, generated).clamp_min(0.0)
    weight_positive = _as_weight(weight_positive, positive).clamp_min(0.0)
    weight_negative = _as_weight(weight_negative, negative).clamp_min(0.0)

    bsz, gen_tokens, dim = generated.shape
    neg_tokens = negative.shape[1]
    targets = torch.cat([old_generated, negative, positive], dim=1)
    target_weights = torch.cat(
        [weight_generated, weight_negative, weight_positive],
        dim=1,
    )

    with torch.no_grad():
        dist = pairwise_l2(old_generated, targets, eps=eps)
        weighted_dist = dist * target_weights.unsqueeze(1)
        scale = weighted_dist.mean(dim=(1, 2)) / target_weights.mean(dim=1).clamp_min(eps)
        scale = scale.clamp_min(1e-3)
        coord_scale = (scale / (float(dim) ** 0.5)).clamp_min(1e-3)
        old_scaled = old_generated / coord_scale.view(bsz, 1, 1)
        targets_scaled = targets / coord_scale.view(bsz, 1, 1)
        dist_normed = dist / scale.view(bsz, 1, 1)

        diag = torch.eye(
            gen_tokens,
            dtype=generated.dtype,
            device=generated.device,
        )
        diag = torch.nn.functional.pad(
            diag,
            (0, targets.shape[1] - gen_tokens),
        )
        dist_normed = dist_normed + diag.unsqueeze(0) * 100.0

        force = torch.zeros_like(old_scaled)
        info: dict[str, torch.Tensor] = {"scale": scale.mean()}
        split_index = gen_tokens + neg_tokens
        for radius in radii:
            radius = float(radius)
            if radius <= 0:
                raise ValueError(f"Drift radius must be positive, got {radius}.")
            logits = -dist_normed / radius
            row_affinity = torch.softmax(logits, dim=-1)
            col_affinity = torch.softmax(logits, dim=-2)
            affinity = (row_affinity * col_affinity).clamp_min(1e-6).sqrt()
            affinity = affinity * target_weights.unsqueeze(1)

            affinity_neg = affinity[:, :, :split_index]
            affinity_pos = affinity[:, :, split_index:]
            sum_pos = affinity_pos.sum(dim=-1, keepdim=True)
            sum_neg = affinity_neg.sum(dim=-1, keepdim=True)
            coeff_neg = -affinity_neg * sum_pos
            coeff_pos = affinity_pos * sum_neg
            coeff = torch.cat([coeff_neg, coeff_pos], dim=2)

            force_r = torch.einsum("bnt,btd->bnd", coeff, targets_scaled)
            force_r = force_r - coeff.sum(dim=-1, keepdim=True) * old_scaled
            norm = force_r.square().mean().clamp_min(eps).sqrt()
            force = force + force_r / norm
            info[f"loss_{radius:g}"] = force_r.square().mean()

        goal_scaled = (old_scaled + force).detach()

    generated_scaled = generated / coord_scale.view(bsz, 1, 1)
    loss_per_sample = (generated_scaled - goal_scaled).square().mean(dim=(1, 2))
    info["loss"] = loss_per_sample.mean().detach()
    return loss_per_sample.mean(), info
