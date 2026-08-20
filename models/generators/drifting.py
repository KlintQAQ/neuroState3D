from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.generators.common import (
    ConditionalLatentBackbone3D,
    PosteriorBackboneConfig,
    repeat_batch,
)
from models.generators.evidence_constraints import (
    contradiction_negative_weights,
    nested_subset_consistency_loss,
)


@dataclass(frozen=True)
class DriftingConfig:
    samples_per_condition: int = 4
    positive_copies: int = 1
    radii: Sequence[float] = (0.02, 0.05, 0.2)
    feature_grid: int = 4
    consistency_weight: float = 0.0
    contradiction_repulsion_strength: float = 0.0
    contradiction_max_weight: float = 6.0
    nested_identity_weight: float = 0.0
    nested_contraction_weight: float = 0.0
    nested_pool_size: int = 2
    nested_contraction_margin: float = 0.0


def _pairwise_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.cdist(x.float(), y.float()).clamp_min(1e-8)


def drifting_field_loss(
    generated: torch.Tensor,
    positives: torch.Tensor,
    negatives: Optional[torch.Tensor] = None,
    generated_weights: Optional[torch.Tensor] = None,
    positive_weights: Optional[torch.Tensor] = None,
    negative_weights: Optional[torch.Tensor] = None,
    radii: Sequence[float] = (0.02, 0.05, 0.2),
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """PyTorch port of the official Drifting field objective.

    Inputs have shape ``[B, set_size, feature_dim]``. The field target is
    stopped exactly as in the official release; gradients flow through the
    current generated features only.
    """

    if generated.ndim != 3 or positives.ndim != 3:
        raise ValueError("generated and positives must be [B,N,S]")
    if generated.shape[0] != positives.shape[0] or generated.shape[2] != positives.shape[2]:
        raise ValueError("generated/positive batch and feature dimensions must match")
    batch, generated_count, feature_dim = generated.shape
    if negatives is None:
        negatives = generated.new_empty((batch, 0, feature_dim))
    if negatives.shape[0] != batch or negatives.shape[2] != feature_dim:
        raise ValueError("negative batch and feature dimensions must match")
    positive_count = positives.shape[1]
    negative_count = negatives.shape[1]
    if generated_weights is None:
        generated_weights = generated.new_ones((batch, generated_count))
    if positive_weights is None:
        positive_weights = generated.new_ones((batch, positive_count))
    if negative_weights is None:
        negative_weights = generated.new_ones((batch, negative_count))

    old_generated = generated.detach().float()
    with torch.no_grad():
        targets = torch.cat(
            [old_generated, negatives.detach().float(), positives.detach().float()], dim=1
        )
        weights = torch.cat(
            [generated_weights.float(), negative_weights.float(), positive_weights.float()],
            dim=1,
        )
        distances = _pairwise_distance(old_generated, targets)
        scale = (distances * weights[:, None]).mean() / weights.mean().clamp_min(1e-8)
        scale_input = (scale / feature_dim**0.5).clamp_min(1e-3)
        old_scaled = old_generated / scale_input
        targets_scaled = targets / scale_input
        normalized_distance = distances / scale.clamp_min(1e-3)
        diagonal = torch.eye(
            generated_count, device=generated.device, dtype=normalized_distance.dtype
        )
        normalized_distance[:, :, :generated_count] += diagonal[None] * 100.0
        total_force = torch.zeros_like(old_scaled)
        diagnostics: dict[str, torch.Tensor] = {"scale": scale}
        split = generated_count + negative_count
        for radius in radii:
            logits = -normalized_distance / float(radius)
            row_affinity = logits.softmax(dim=-1)
            column_affinity = logits.softmax(dim=-2)
            affinity = (row_affinity * column_affinity).clamp_min(1e-6).sqrt()
            affinity = affinity * weights[:, None]
            negative_affinity = affinity[:, :, :split]
            positive_affinity = affinity[:, :, split:]
            sum_positive = positive_affinity.sum(dim=-1, keepdim=True)
            sum_negative = negative_affinity.sum(dim=-1, keepdim=True)
            coefficients = torch.cat(
                [-negative_affinity * sum_positive, positive_affinity * sum_negative],
                dim=2,
            )
            force = torch.einsum("biy,byx->bix", coefficients, targets_scaled)
            force = force - coefficients.sum(dim=-1, keepdim=True) * old_scaled
            force_norm = force.square().mean()
            diagnostics[f"force_{radius}"] = force_norm
            total_force = total_force + force / force_norm.clamp_min(1e-8).sqrt()
        goal = old_scaled + total_force
    loss = (generated.float() / scale_input - goal).square().mean()
    return loss, {key: value.detach() for key, value in diagnostics.items()}


class ConditionalLatentDrifting(nn.Module):
    """One-step conditional latent Drifting candidate.

    The original paper is class-conditional ImageNet generation. Here its
    field objective is adapted to continuous 3D evidence conditions: generated
    samples for each condition are attracted to that subject's full-evidence
    latent and repelled by optional other-subject negatives.
    """

    method = "drifting"

    def __init__(
        self,
        backbone_config: PosteriorBackboneConfig,
        config: DriftingConfig = DriftingConfig(),
    ) -> None:
        super().__init__()
        self.backbone_config = backbone_config
        self.config = config
        self.model = ConditionalLatentBackbone3D(backbone_config)

    def _features(self, latents: torch.Tensor) -> torch.Tensor:
        leading_shape = latents.shape[:-4]
        channels = latents.shape[-4]
        flattened = latents.reshape(-1, channels, *latents.shape[-3:])
        grid = min(self.config.feature_grid, *latents.shape[-3:])
        pooled = F.adaptive_avg_pool3d(flattened.float(), output_size=grid)
        features = pooled.flatten(1)
        features = F.layer_norm(features, (features.shape[-1],))
        return features.reshape(*leading_shape, -1)

    def training_loss(
        self,
        target_latent: torch.Tensor,
        condition: torch.Tensor,
        modality_mask: Optional[torch.Tensor] = None,
        target_ids: Optional[torch.Tensor] = None,
        quality: Optional[torch.Tensor] = None,
        negative_latents: Optional[torch.Tensor] = None,
        negative_scores: Optional[torch.Tensor] = None,
        consistency_target: Optional[torch.Tensor] = None,
        consistency_mask: Optional[torch.Tensor] = None,
        richer_condition: Optional[torch.Tensor] = None,
        richer_modality_mask: Optional[torch.Tensor] = None,
        richer_quality: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> dict[str, torch.Tensor]:
        if target_latent.ndim != 5:
            raise ValueError("target_latent must be [B,C,D,H,W]")
        batch_size = target_latent.shape[0]
        generated_count = self.config.samples_per_condition
        noise = torch.randn(
            (batch_size * generated_count, *target_latent.shape[1:]),
            device=target_latent.device,
            dtype=target_latent.dtype,
            generator=generator,
        )
        generated = self.model(
            noise,
            repeat_batch(condition, generated_count),
            time=noise.new_zeros((noise.shape[0],)),
            modality_mask=repeat_batch(modality_mask, generated_count),
            target_ids=repeat_batch(target_ids, generated_count),
            quality=repeat_batch(quality, generated_count),
        ).reshape(batch_size, generated_count, *target_latent.shape[1:])
        positives = target_latent[:, None].expand(
            batch_size, self.config.positive_copies, *target_latent.shape[1:]
        )
        generated_features = self._features(generated)
        positive_features = self._features(positives)
        negative_features = None
        negative_weights = None
        if negative_latents is not None:
            if negative_latents.ndim == 5:
                negative_latents = negative_latents[:, None]
            negative_features = self._features(negative_latents)
            if negative_scores is not None:
                if self.config.contradiction_repulsion_strength <= 0:
                    raise ValueError(
                        "negative_scores require contradiction_repulsion_strength > 0"
                    )
                if negative_scores.ndim == 1:
                    negative_scores = negative_scores[:, None]
                if negative_scores.shape != negative_features.shape[:2]:
                    raise ValueError(
                        "negative_scores must match negative_latents [B,N]"
                    )
                negative_weights = contradiction_negative_weights(
                    negative_scores,
                    strength=self.config.contradiction_repulsion_strength,
                    max_weight=self.config.contradiction_max_weight,
                )
        elif negative_scores is not None:
            raise ValueError("negative_scores require negative_latents")
        native_loss, diagnostics = drifting_field_loss(
            generated_features,
            positive_features,
            negatives=negative_features,
            negative_weights=negative_weights,
            radii=self.config.radii,
        )
        consistency_loss = target_latent.new_zeros(())
        if consistency_target is not None:
            difference = (generated - consistency_target[:, None]).pow(2)
            if consistency_mask is not None:
                mask = consistency_mask[:, None].to(difference.dtype).expand_as(difference)
                consistency_loss = (difference * mask).sum() / mask.sum().clamp_min(1)
            else:
                consistency_loss = difference.mean()

        nested_identity_loss = target_latent.new_zeros(())
        nested_contraction_loss = target_latent.new_zeros(())
        nested_diagnostics: dict[str, torch.Tensor] = {}
        nested_enabled = (
            self.config.nested_identity_weight > 0
            or self.config.nested_contraction_weight > 0
        )
        if nested_enabled:
            if richer_condition is None or richer_modality_mask is None:
                raise ValueError(
                    "Nested-subset losses require richer_condition and "
                    "richer_modality_mask"
                )
            if modality_mask is None:
                raise ValueError("Nested-subset losses require modality_mask")
            richer_generated = self.model(
                noise,
                repeat_batch(richer_condition, generated_count),
                time=noise.new_zeros((noise.shape[0],)),
                modality_mask=repeat_batch(richer_modality_mask, generated_count),
                target_ids=repeat_batch(target_ids, generated_count),
                quality=repeat_batch(richer_quality, generated_count),
            ).reshape(batch_size, generated_count, *target_latent.shape[1:])
            (
                nested_identity_loss,
                nested_contraction_loss,
                nested_diagnostics,
            ) = nested_subset_consistency_loss(
                generated,
                richer_generated,
                modality_mask,
                richer_modality_mask,
                pool_size=self.config.nested_pool_size,
                contraction_margin=self.config.nested_contraction_margin,
            )
        elif richer_condition is not None or richer_modality_mask is not None:
            raise ValueError(
                "richer_condition/mask were provided but nested losses are disabled"
            )

        loss = (
            native_loss
            + self.config.consistency_weight * consistency_loss
            + self.config.nested_identity_weight * nested_identity_loss
            + self.config.nested_contraction_weight * nested_contraction_loss
        )
        return {
            "loss": loss,
            "native_loss": native_loss.detach(),
            "consistency_loss": consistency_loss.detach(),
            "nested_identity_loss": nested_identity_loss.detach(),
            "nested_contraction_loss": nested_contraction_loss.detach(),
            "contradiction_weight_mean": (
                negative_weights.mean().detach()
                if negative_weights is not None
                else target_latent.new_ones(())
            ),
            "generated": generated,
            **{
                f"nested_{key}": value
                for key, value in nested_diagnostics.items()
            },
            **{f"drift_{key}": value for key, value in diagnostics.items()},
        }

    def forward(self, *args, **kwargs) -> dict[str, torch.Tensor]:
        return self.training_loss(*args, **kwargs)

    @torch.no_grad()
    def sample(
        self,
        condition: torch.Tensor,
        n_samples: int = 1,
        seed: int = 0,
        modality_mask: Optional[torch.Tensor] = None,
        target_ids: Optional[torch.Tensor] = None,
        quality: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if n_samples < 1:
            raise ValueError("n_samples must be positive")
        batch_size, _, depth, height, width = condition.shape
        rng = torch.Generator(device=condition.device)
        rng.manual_seed(seed)
        noise = torch.randn(
            (
                batch_size * n_samples,
                self.backbone_config.latent_channels,
                depth,
                height,
                width,
            ),
            device=condition.device,
            dtype=condition.dtype,
            generator=rng,
        )
        output = self.model(
            noise,
            repeat_batch(condition, n_samples),
            time=noise.new_zeros((noise.shape[0],)),
            modality_mask=repeat_batch(modality_mask, n_samples),
            target_ids=repeat_batch(target_ids, n_samples),
            quality=repeat_batch(quality, n_samples),
        )
        return output.reshape(
            batch_size,
            n_samples,
            self.backbone_config.latent_channels,
            depth,
            height,
            width,
        )

    @property
    def nfe(self) -> int:
        return 1
