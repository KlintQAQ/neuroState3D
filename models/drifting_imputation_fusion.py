from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.evidence_fusion import EvidenceReliableFusion
from models.generators.drifting import ConditionalLatentDrifting


@dataclass(frozen=True)
class DriftingImputationFusionConfig:
    """End-to-end settings for latent Drifting -> MRI imputation -> fusion."""

    samples_per_target: int = 4
    drift_loss_weight: float = 1.0
    reconstruction_weight: float = 1.0
    uncertainty_nll_weight: float = 0.05
    uncertainty_calibration_weight: float = 0.05
    decoder_hidden_channels: int = 64
    decoder_highres_channels: int = 16
    decoder_residual_scale: float = 1.0
    uncertainty_scale: float = 1.0
    minimum_generated_confidence: float = 0.05
    maximum_generated_confidence: float = 0.95
    detach_target_latent: bool = True


class LatentMRIImageDecoder(nn.Module):
    """Decode a latent using masked high-resolution observed MRI context."""

    def __init__(
        self,
        latent_channels: int,
        hidden_channels: int,
        highres_channels: int,
        num_modalities: int,
        residual_scale: float,
    ) -> None:
        super().__init__()
        groups = min(8, hidden_channels)
        while hidden_channels % groups:
            groups -= 1
        highres_groups = min(8, highres_channels)
        while highres_channels % highres_groups:
            highres_groups -= 1
        self.residual_scale = float(residual_scale)
        self.latent_net = nn.Sequential(
            nn.Conv3d(latent_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, hidden_channels),
            nn.GELU(),
            nn.Conv3d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, hidden_channels),
            nn.GELU(),
            nn.Conv3d(hidden_channels, highres_channels, kernel_size=1),
        )
        self.context_net = nn.Sequential(
            nn.Conv3d(
                num_modalities * 2,
                highres_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.GroupNorm(highres_groups, highres_channels),
            nn.GELU(),
        )
        self.refine = nn.Sequential(
            nn.Conv3d(
                highres_channels * 2,
                highres_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.GroupNorm(highres_groups, highres_channels),
            nn.GELU(),
            nn.Conv3d(highres_channels, highres_channels, kernel_size=3, padding=1),
            nn.GroupNorm(highres_groups, highres_channels),
            nn.GELU(),
        )
        self.image_head = nn.Conv3d(highres_channels, 1, kernel_size=1)
        self.uncertainty_head = nn.Conv3d(highres_channels, 1, kernel_size=1)
        nn.init.zeros_(self.uncertainty_head.weight)
        nn.init.zeros_(self.uncertainty_head.bias)

    def forward(
        self,
        latent: torch.Tensor,
        source_images: torch.Tensor,
        source_mask: torch.Tensor,
        target_spatial: Sequence[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if source_images.ndim != 5:
            raise ValueError("source_images must be [B,M,D,H,W]")
        b, m, d, h, w = source_images.shape
        if source_mask.shape != (b, m):
            raise ValueError(
                f"Expected source_mask {(b, m)}, got {tuple(source_mask.shape)}"
            )
        mask = source_mask.to(source_images.dtype).view(b, m, 1, 1, 1)
        masked_images = source_images * mask
        mask_channels = mask.expand(b, m, d, h, w)
        context = self.context_net(torch.cat([masked_images, mask_channels], dim=1))
        latent_feature = self.latent_net(latent)
        latent_feature = F.interpolate(
            latent_feature,
            size=tuple(int(value) for value in target_spatial),
            mode="trilinear",
            align_corners=False,
        )
        feature = self.refine(torch.cat([latent_feature, context], dim=1))
        image = self.residual_scale * self.image_head(feature)
        uncertainty = F.softplus(self.uncertainty_head(feature)) + 1e-4
        return image, uncertainty


class DriftingImputationFusion(nn.Module):
    """Complete missing MRI slots with Drifting before evidence fusion.

    The input image tensor keeps all modality slots in a fixed order. During
    training, masked slots still contain their ground truth MRI and are used
    only as generator targets. During inference they may contain zeros because
    every masked slot is overwritten by a generated image before final fusion.
    """

    def __init__(
        self,
        fusion: EvidenceReliableFusion,
        drifting: ConditionalLatentDrifting,
        config: DriftingImputationFusionConfig = DriftingImputationFusionConfig(),
    ) -> None:
        super().__init__()
        if config.samples_per_target < 1:
            raise ValueError("samples_per_target must be positive")
        if not 0 <= config.minimum_generated_confidence <= 1:
            raise ValueError("minimum_generated_confidence must be in [0, 1]")
        if not 0 <= config.maximum_generated_confidence <= 1:
            raise ValueError("maximum_generated_confidence must be in [0, 1]")
        if (
            config.minimum_generated_confidence
            > config.maximum_generated_confidence
        ):
            raise ValueError("minimum confidence cannot exceed maximum confidence")

        latent_channels = drifting.backbone_config.latent_channels
        condition_channels = drifting.backbone_config.condition_channels
        if latent_channels != fusion.config.feature_channels:
            raise ValueError(
                "Drifting latent_channels must equal fusion feature_channels: "
                f"{latent_channels} != {fusion.config.feature_channels}"
            )
        if condition_channels != fusion.config.feature_channels:
            raise ValueError(
                "Drifting condition_channels must equal fusion feature_channels: "
                f"{condition_channels} != {fusion.config.feature_channels}"
            )
        if drifting.backbone_config.num_modalities != fusion.num_modalities:
            raise ValueError("Drifting and fusion modality counts must match")
        if drifting.backbone_config.num_targets < fusion.num_modalities:
            raise ValueError("Drifting num_targets must cover every fusion modality")

        self.fusion = fusion
        self.drifting = drifting
        self.config = config
        self.image_decoders = nn.ModuleList(
            [
                LatentMRIImageDecoder(
                    latent_channels,
                    config.decoder_hidden_channels,
                    config.decoder_highres_channels,
                    fusion.num_modalities,
                    config.decoder_residual_scale,
                )
                for _ in range(fusion.num_modalities)
            ]
        )

    def forward(
        self,
        images: torch.Tensor,
        modality_mask: torch.Tensor,
        modality_state: torch.Tensor | None = None,
        *,
        compute_generator_loss: bool = False,
        n_samples: int | None = None,
        seed: int = 0,
    ) -> dict[str, object]:
        if images.ndim != 5:
            raise ValueError(f"Expected images [B,M,D,H,W], got {tuple(images.shape)}")
        batch, modalities = images.shape[:2]
        if modalities != self.fusion.num_modalities:
            raise ValueError(
                f"Expected {self.fusion.num_modalities} modalities, got {modalities}"
            )
        if modality_mask.shape != (batch, modalities):
            raise ValueError(
                f"Expected modality_mask {(batch, modalities)}, got "
                f"{tuple(modality_mask.shape)}"
            )
        if torch.any(modality_mask.sum(dim=1) <= 0):
            raise ValueError("Every sample must retain at least one observed modality")

        device = images.device
        modality_mask = modality_mask.to(device=device, dtype=images.dtype)
        if modality_state is None:
            modality_state = torch.zeros(
                batch, modalities, dtype=torch.long, device=device
            )
            modality_state[modality_mask <= 0] = 1
        else:
            modality_state = modality_state.to(device=device, dtype=torch.long)

        observed_output = self.fusion(
            images,
            modality_mask=modality_mask,
            modality_state=modality_state,
        )
        condition = observed_output["fused_feature"]
        pair_batch, pair_target = torch.nonzero(
            modality_mask <= 0,
            as_tuple=True,
        )
        zero = condition.new_zeros(())
        if pair_batch.numel() == 0:
            return {
                **observed_output,
                "imputed_images": images,
                "imputation_confidence": modality_mask.new_ones(modality_mask.shape),
                "imputation_confidence_map": images.new_ones(images.shape),
                "imputation_uncertainty": modality_mask.new_zeros(modality_mask.shape),
                "effective_modality_mask": modality_mask,
                "drifting_loss": zero,
                "reconstruction_loss": zero,
                "uncertainty_nll_loss": zero,
                "uncertainty_calibration_loss": zero,
                "generator_loss": zero,
                "missing_pair_count": 0,
            }

        pair_condition = condition.index_select(0, pair_batch)
        pair_mask = modality_mask.index_select(0, pair_batch)
        pair_target_ids = pair_target.to(dtype=torch.long)
        sample_count = n_samples or self.config.samples_per_target

        if compute_generator_loss:
            target_images = images[pair_batch, pair_target].unsqueeze(1)
            if self.config.detach_target_latent:
                with torch.no_grad():
                    target_latent = self.fusion.encoder(target_images)[
                        self.fusion.feature_stage
                    ]
            else:
                target_latent = self.fusion.encoder(target_images)[
                    self.fusion.feature_stage
                ]
            self._validate_target_latent(target_latent, pair_condition)
            generator = torch.Generator(device=device)
            generator.manual_seed(seed)
            drifting_output = self.drifting.training_loss(
                target_latent=target_latent,
                condition=pair_condition,
                modality_mask=pair_mask,
                target_ids=pair_target_ids,
                generator=generator,
            )
            latent_samples = drifting_output["generated"]
            drifting_loss = drifting_output["loss"]
        else:
            target_images = None
            latent_samples = self.drifting.sample(
                condition=pair_condition,
                n_samples=sample_count,
                seed=seed,
                modality_mask=pair_mask,
                target_ids=pair_target_ids,
            )
            drifting_loss = zero

        decoded_samples, predicted_uncertainty_samples = self._decode_samples_by_target(
            latent_samples,
            pair_target_ids,
            images.index_select(0, pair_batch),
            pair_mask,
            images.shape[-3:],
        )
        decoded = decoded_samples.mean(dim=1)
        epistemic_uncertainty = decoded_samples.float().var(
            dim=1,
            unbiased=False,
        ).clamp_min(1e-8).sqrt().to(images.dtype)
        aleatoric_uncertainty = predicted_uncertainty_samples.mean(dim=1)
        total_uncertainty = (
            epistemic_uncertainty + aleatoric_uncertainty
        ).clamp_min(1e-4)
        reconstruction_loss = (
            F.smooth_l1_loss(decoded, target_images)
            if target_images is not None
            else zero
        )
        if target_images is not None:
            absolute_error = (decoded - target_images).abs()
            uncertainty_nll_loss = (
                absolute_error / total_uncertainty + total_uncertainty.log()
            ).mean()
            uncertainty_calibration_loss = (
                total_uncertainty.log()
                - absolute_error.detach().clamp_min(1e-4).log()
            ).abs().mean()
        else:
            uncertainty_nll_loss = zero
            uncertainty_calibration_loss = zero
        generator_loss = (
            self.config.drift_loss_weight * drifting_loss
            + self.config.reconstruction_weight * reconstruction_loss
            + self.config.uncertainty_nll_weight * uncertainty_nll_loss
            + self.config.uncertainty_calibration_weight
            * uncertainty_calibration_loss
        )

        confidence_map = torch.exp(
            -self.config.uncertainty_scale * total_uncertainty
        ).clamp(
            min=self.config.minimum_generated_confidence,
            max=self.config.maximum_generated_confidence,
        )
        confidence = confidence_map.flatten(1).mean(dim=1).to(images.dtype)
        uncertainty = total_uncertainty.flatten(1).mean(dim=1).to(images.dtype)

        completed_images = images.clone()
        completed_images[pair_batch, pair_target] = decoded[:, 0]
        effective_mask = modality_mask.clone()
        effective_mask[pair_batch, pair_target] = 1
        effective_state = modality_state.clone()
        effective_state[pair_batch, pair_target] = 3
        modality_confidence = modality_mask.clone()
        # Confidence is supervised by reconstruction uncertainty objectives;
        # detaching prevents the task loss from gaming the fusion gate.
        modality_confidence[pair_batch, pair_target] = confidence.detach()
        modality_confidence_map = images.new_ones(images.shape)
        modality_confidence_map[pair_batch, pair_target] = confidence_map[:, 0].detach()
        uncertainty_map = modality_mask.new_zeros(modality_mask.shape)
        uncertainty_map[pair_batch, pair_target] = uncertainty

        fused_output = self.fusion(
            completed_images,
            modality_mask=effective_mask,
            modality_state=effective_state,
            modality_confidence=modality_confidence_map,
        )
        return {
            **fused_output,
            "imputed_images": completed_images,
            "imputation_confidence": modality_confidence,
            "imputation_confidence_map": modality_confidence_map,
            "imputation_uncertainty": uncertainty_map,
            "effective_modality_mask": effective_mask,
            "original_modality_mask": modality_mask,
            "drifting_loss": drifting_loss,
            "reconstruction_loss": reconstruction_loss,
            "uncertainty_nll_loss": uncertainty_nll_loss,
            "uncertainty_calibration_loss": uncertainty_calibration_loss,
            "generator_loss": generator_loss,
            "missing_pair_count": int(pair_batch.numel()),
            "observed_logits": observed_output["logits"],
        }

    def _decode_samples_by_target(
        self,
        latent_samples: torch.Tensor,
        target_ids: torch.Tensor,
        source_images: torch.Tensor,
        source_mask: torch.Tensor,
        target_spatial: Sequence[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if latent_samples.ndim != 6:
            raise ValueError("latent_samples must be [B,N,C,D,H,W]")
        batch, samples = latent_samples.shape[:2]
        output_shape = (
            batch,
            samples,
            1,
            *(int(value) for value in target_spatial),
        )
        image_output = latent_samples.new_empty(output_shape)
        uncertainty_output = latent_samples.new_empty(output_shape)
        for target_index, decoder in enumerate(self.image_decoders):
            selected = torch.nonzero(target_ids == target_index, as_tuple=False).flatten()
            if selected.numel() == 0:
                continue
            selected_latents = latent_samples.index_select(0, selected)
            selected_batch = selected_latents.shape[0]
            flat_latents = selected_latents.flatten(0, 1)
            selected_images = source_images.index_select(0, selected)
            selected_masks = source_mask.index_select(0, selected)
            repeated_images = selected_images.repeat_interleave(samples, dim=0)
            repeated_masks = selected_masks.repeat_interleave(samples, dim=0)
            decoded, predicted_uncertainty = decoder(
                flat_latents,
                repeated_images,
                repeated_masks,
                target_spatial,
            )
            image_output[selected] = decoded.reshape(
                selected_batch, samples, 1, *decoded.shape[-3:]
            )
            uncertainty_output[selected] = predicted_uncertainty.reshape(
                selected_batch,
                samples,
                1,
                *predicted_uncertainty.shape[-3:],
            )
        return image_output, uncertainty_output

    def _validate_target_latent(
        self,
        target_latent: torch.Tensor,
        condition: torch.Tensor,
    ) -> None:
        expected = (
            target_latent.shape[0],
            self.drifting.backbone_config.latent_channels,
            *condition.shape[-3:],
        )
        if target_latent.shape != expected:
            raise ValueError(
                "Target encoder feature must match the Drifting latent contract: "
                f"expected {expected}, got {tuple(target_latent.shape)}"
            )
