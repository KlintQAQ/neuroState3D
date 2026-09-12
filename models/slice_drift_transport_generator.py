from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.slice_virtual_modality_generator import ConvNormAct2d, ResidualConvBlock2d


@dataclass(frozen=True)
class SliceDriftTransportGeneratorConfig:
    in_modalities: int = 4
    base_modalities: int = 4
    target_base_index: int = 1
    hidden_channels: int = 32
    transport_steps: int = 4
    transport_step_scale: float = 1.0
    velocity_scale: float = 1.0
    uncertainty_min: float = 1e-4
    output_activation: str = "hardtanh"
    class_channels: int = 0
    class_conditioned: bool = False
    init_blur_kernel: int = 5
    gated_refinement: bool = False
    refinement_residual_scale: float = 0.25
    gate_bias_init: float = -3.0
    refinement_acceptance_gate: bool = False
    accept_bias_init: float = -2.0
    refinement_channels_multiplier: int = 1
    refinement_blocks: int = 1
    refinement_detail_features: bool = False
    medical_role_conditioning: bool = False
    learned_initial_state: bool = False
    medical_prompt_conditioning: bool = False
    role_hidden_channels: int = 0
    initial_residual_scale: float = 0.35
    prompt_channels: int = 5


class SliceDriftTransportGenerator(nn.Module):
    """Iterative drift-transport generator for missing MRI slices.

    This model does not directly regress the missing modality in one decoder
    pass. It builds a source-conditioned initial state, then repeatedly
    predicts a velocity field v_theta(x_t, observed, mask, t) and updates
    x_{t+1} = x_t + step_scale * v_theta. The returned ``synthetic`` image is
    the final state of that trajectory.
    """

    def __init__(
        self,
        config: SliceDriftTransportGeneratorConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or SliceDriftTransportGeneratorConfig()
        hidden = int(self.config.hidden_channels)
        class_channels = int(self.config.class_channels) if self.config.class_conditioned else 0
        self.base_modalities = max(1, int(self.config.base_modalities))
        self.context_depth = max(1, int(self.config.in_modalities) // self.base_modalities)
        role_enabled = bool(self.config.medical_role_conditioning)
        role_hidden = int(self.config.role_hidden_channels) or max(8, hidden // 2)
        self.role_condition_channels = hidden if role_enabled else 0
        self.prompt_condition_channels = (
            max(3, int(self.config.prompt_channels))
            if bool(self.config.medical_prompt_conditioning)
            else 0
        )
        if role_enabled:
            per_role_channels = self.context_depth * 2
            self.role_stems = nn.ModuleList(
                [
                    nn.Sequential(
                        ConvNormAct2d(per_role_channels, role_hidden),
                        ResidualConvBlock2d(role_hidden),
                    )
                    for _ in range(self.base_modalities)
                ]
            )
            fusion_in_channels = role_hidden * self.base_modalities + class_channels
            self.role_fusion = nn.Sequential(
                ConvNormAct2d(fusion_in_channels, hidden),
                ResidualConvBlock2d(hidden),
            )
            self.initial_residual_head = nn.Conv2d(hidden, 1, kernel_size=1)
            nn.init.zeros_(self.initial_residual_head.weight)
            nn.init.zeros_(self.initial_residual_head.bias)
            if self.prompt_condition_channels > 0:
                self.medical_prompt_head = nn.Sequential(
                    ConvNormAct2d(hidden, hidden),
                    ResidualConvBlock2d(hidden),
                    nn.Conv2d(hidden, self.prompt_condition_channels, kernel_size=1),
                )
        in_channels = (
            int(self.config.in_modalities) * 2
            + 3
            + class_channels
            + self.role_condition_channels
            + self.prompt_condition_channels
        )
        self.stem = nn.Sequential(
            ConvNormAct2d(in_channels, hidden),
            ResidualConvBlock2d(hidden),
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(hidden, hidden * 2, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(_group_count(hidden * 2), hidden * 2),
            nn.GELU(),
            ResidualConvBlock2d(hidden * 2),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(hidden * 2, hidden * 4, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(_group_count(hidden * 4), hidden * 4),
            nn.GELU(),
            ResidualConvBlock2d(hidden * 4),
        )
        self.bottleneck = nn.Sequential(
            ResidualConvBlock2d(hidden * 4),
            ResidualConvBlock2d(hidden * 4),
        )
        self.up1 = nn.Sequential(
            ConvNormAct2d(hidden * 4 + hidden * 2, hidden * 2),
            ResidualConvBlock2d(hidden * 2),
        )
        self.up2 = nn.Sequential(
            ConvNormAct2d(hidden * 2 + hidden, hidden),
            ResidualConvBlock2d(hidden),
        )
        self.velocity_head = nn.Conv2d(hidden, 1, kernel_size=1)
        self.uncertainty_head = nn.Conv2d(hidden, 1, kernel_size=1)
        nn.init.zeros_(self.velocity_head.weight)
        nn.init.zeros_(self.velocity_head.bias)
        if self.config.gated_refinement:
            detail_channels = 8 if bool(self.config.refinement_detail_features) else 0
            refinement_in_channels = (
                int(self.config.in_modalities) * 2
                + 3
                + class_channels
                + detail_channels
                + self.role_condition_channels
                + self.prompt_condition_channels
            )
            refinement_hidden = hidden * max(1, int(self.config.refinement_channels_multiplier))
            refinement_blocks = max(1, int(self.config.refinement_blocks))
            refinement_layers: list[nn.Module] = [ConvNormAct2d(refinement_in_channels, refinement_hidden)]
            refinement_layers.extend(ResidualConvBlock2d(refinement_hidden) for _ in range(refinement_blocks))
            refinement_layers.append(ConvNormAct2d(refinement_hidden, refinement_hidden))
            self.refinement_context = nn.Sequential(
                *refinement_layers,
            )
            self.refinement_feature_project = (
                nn.Identity()
                if refinement_hidden == hidden
                else nn.Sequential(
                    nn.Conv2d(hidden, refinement_hidden, kernel_size=1),
                    nn.GroupNorm(_group_count(refinement_hidden), refinement_hidden),
                    nn.GELU(),
                )
            )
            self.refinement_gate_head = nn.Conv2d(refinement_hidden, 1, kernel_size=1)
            self.refinement_residual_head = nn.Conv2d(refinement_hidden, 1, kernel_size=1)
            nn.init.constant_(self.refinement_gate_head.bias, float(self.config.gate_bias_init))
            nn.init.zeros_(self.refinement_residual_head.bias)
            if self.config.refinement_acceptance_gate:
                self.refinement_accept_head = nn.Conv2d(refinement_hidden, 1, kernel_size=1)
                nn.init.constant_(self.refinement_accept_head.bias, float(self.config.accept_bias_init))

    def forward(
        self,
        slices: torch.Tensor,
        modality_mask: torch.Tensor,
        class_condition: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if slices.ndim != 4:
            raise ValueError(f"Expected slices [B,M,H,W], got {tuple(slices.shape)}")
        b, m, h, w = slices.shape
        if m != int(self.config.in_modalities):
            raise ValueError(f"Expected {self.config.in_modalities} modalities, got {m}")
        modality_mask = modality_mask.to(device=slices.device, dtype=slices.dtype)
        if tuple(modality_mask.shape) != (b, m):
            raise ValueError(
                f"Expected modality_mask {(b, m)}, got {tuple(modality_mask.shape)}"
            )
        class_map = self._class_map(class_condition, b, h, w, slices)
        masked = slices * modality_mask.view(b, m, 1, 1)
        mask_channels = modality_mask.view(b, m, 1, 1).expand(b, m, h, w)
        role_context, prompt_logits, prompt_probs = self._medical_conditioning(
            masked,
            mask_channels,
            class_map,
        )
        current = self._initial_state(masked, modality_mask, role_context)
        states = [current]
        velocities = []
        uncertainty_raw = None
        last_features = None
        steps = max(1, int(self.config.transport_steps))
        for step in range(steps):
            progress = float(step) / float(max(steps - 1, 1))
            remaining = float(steps - step) / float(steps)
            velocity, uncertainty_raw, last_features = self._velocity_step(
                current,
                masked,
                mask_channels,
                progress,
                remaining,
                class_map,
                role_context,
                prompt_probs,
            )
            current = self._activate(
                current + float(self.config.transport_step_scale) * velocity
            )
            velocities.append(velocity)
            states.append(current)
        if uncertainty_raw is None:
            uncertainty_raw = torch.zeros_like(current)
        stage1 = current
        refinement_gate = None
        refinement_gate_logits = None
        refinement_residual = None
        refinement_region_gate = None
        refinement_acceptance = None
        refinement_acceptance_logits = None
        if self.config.gated_refinement:
            if last_features is None:
                raise RuntimeError("gated_refinement requires at least one transport step")
            uncertainty_preview = F.softplus(uncertainty_raw)
            source_energy = masked.abs().amax(dim=1, keepdim=True)
            refinement_inputs = [
                masked,
                mask_channels,
                stage1,
                uncertainty_preview,
                source_energy,
            ]
            if class_map is not None:
                refinement_inputs.append(class_map)
            if bool(self.config.refinement_detail_features):
                refinement_inputs.append(
                    self._refinement_detail_features(masked, mask_channels, stage1, source_energy)
                )
            if role_context is not None:
                refinement_inputs.append(role_context)
            if prompt_probs is not None:
                refinement_inputs.append(prompt_probs)
            refinement_features = self.refinement_context(torch.cat(refinement_inputs, dim=1))
            refinement_features = refinement_features + self.refinement_feature_project(last_features)
            refinement_gate_logits = self.refinement_gate_head(refinement_features)
            refinement_region_gate = torch.sigmoid(refinement_gate_logits)
            refinement_residual = float(self.config.refinement_residual_scale) * torch.tanh(
                self.refinement_residual_head(refinement_features)
            )
            refinement_gate = refinement_region_gate
            if self.config.refinement_acceptance_gate:
                refinement_acceptance_logits = self.refinement_accept_head(refinement_features)
                refinement_acceptance = torch.sigmoid(refinement_acceptance_logits)
                refinement_gate = refinement_region_gate * refinement_acceptance
            current = self._activate(stage1 + refinement_gate * refinement_residual)
        uncertainty = F.softplus(uncertainty_raw) + float(self.config.uncertainty_min)
        confidence = torch.exp(-uncertainty)
        output = {
            "synthetic": current,
            "uncertainty": uncertainty,
            "confidence": confidence,
            "drift_initial": states[0],
            "stage1_synthetic": stage1,
            "drift_states": torch.stack(states, dim=1),
            "drift_velocities": torch.stack(velocities, dim=1),
            "transport_step_scale": torch.tensor(
                float(self.config.transport_step_scale),
                device=slices.device,
                dtype=slices.dtype,
            ),
        }
        if refinement_gate is not None:
            output.update(
                {
                    "refinement_gate": refinement_gate,
                    "refinement_gate_logits": refinement_gate_logits,
                    "refinement_region_gate": refinement_region_gate,
                    "refinement_residual": refinement_residual,
                }
            )
        if refinement_acceptance is not None:
            output.update(
                {
                    "refinement_acceptance": refinement_acceptance,
                    "refinement_acceptance_logits": refinement_acceptance_logits,
                }
            )
        if prompt_logits is not None:
            output["medical_prompt_logits"] = prompt_logits
            output["medical_prompt_probs"] = prompt_probs
            output["prompt_logits"] = prompt_logits[:, :3]
        return output

    def _class_map(
        self,
        class_condition: torch.Tensor | None,
        batch: int,
        height: int,
        width: int,
        reference: torch.Tensor,
    ) -> torch.Tensor | None:
        class_channels = int(self.config.class_channels) if self.config.class_conditioned else 0
        if class_channels <= 0:
            return None
        if class_condition is None:
            class_condition = torch.zeros(
                batch,
                class_channels,
                device=reference.device,
                dtype=reference.dtype,
            )
        class_condition = class_condition.to(device=reference.device, dtype=reference.dtype)
        if tuple(class_condition.shape) != (batch, class_channels):
            raise ValueError(
                f"Expected class_condition {(batch, class_channels)}, got {tuple(class_condition.shape)}"
            )
        return class_condition.view(batch, class_channels, 1, 1).expand(
            batch,
            class_channels,
            height,
            width,
        )

    def _initial_state(
        self,
        masked: torch.Tensor,
        modality_mask: torch.Tensor,
        role_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        current = self._role_weighted_prior(masked, modality_mask)
        kernel = int(self.config.init_blur_kernel)
        if kernel > 1:
            if kernel % 2 == 0:
                kernel += 1
            current = F.avg_pool2d(current, kernel_size=kernel, stride=1, padding=kernel // 2)
        if (
            bool(self.config.learned_initial_state)
            and role_context is not None
            and hasattr(self, "initial_residual_head")
        ):
            current = current + float(self.config.initial_residual_scale) * torch.tanh(
                self.initial_residual_head(role_context)
            )
        return self._activate(current)

    def _velocity_step(
        self,
        current: torch.Tensor,
        masked: torch.Tensor,
        mask_channels: torch.Tensor,
        progress: float,
        remaining: float,
        class_map: torch.Tensor | None,
        role_context: torch.Tensor | None,
        prompt_probs: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, _, h, w = current.shape
        progress_map = current.new_full((b, 1, h, w), float(progress))
        remaining_map = current.new_full((b, 1, h, w), float(remaining))
        inputs = [current, masked, mask_channels, progress_map, remaining_map]
        if class_map is not None:
            inputs.append(class_map)
        if role_context is not None:
            inputs.append(role_context)
        if prompt_probs is not None:
            inputs.append(prompt_probs)
        x0 = self.stem(torch.cat(inputs, dim=1))
        x1 = self.down1(x0)
        x2 = self.bottleneck(self.down2(x1))
        x = F.interpolate(x2, size=x1.shape[2:], mode="bilinear", align_corners=False)
        x = self.up1(torch.cat([x, x1], dim=1))
        x = F.interpolate(x, size=x0.shape[2:], mode="bilinear", align_corners=False)
        x = self.up2(torch.cat([x, x0], dim=1))
        velocity = float(self.config.velocity_scale) * torch.tanh(self.velocity_head(x))
        uncertainty_raw = self.uncertainty_head(x)
        return velocity, uncertainty_raw, x

    def _medical_conditioning(
        self,
        masked: torch.Tensor,
        mask_channels: torch.Tensor,
        class_map: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        if not bool(self.config.medical_role_conditioning):
            return None, None, None
        if masked.shape[1] != self.base_modalities * self.context_depth:
            return None, None, None
        role_features = []
        for modality_index, stem in enumerate(self.role_stems):
            start = modality_index * self.context_depth
            end = start + self.context_depth
            role_input = torch.cat(
                [masked[:, start:end], mask_channels[:, start:end]],
                dim=1,
            )
            role_features.append(stem(role_input))
        fusion_inputs = [torch.cat(role_features, dim=1)]
        if class_map is not None:
            fusion_inputs.append(class_map)
        role_context = self.role_fusion(torch.cat(fusion_inputs, dim=1))
        prompt_logits = None
        prompt_probs = None
        if self.prompt_condition_channels > 0 and hasattr(self, "medical_prompt_head"):
            prompt_logits = self.medical_prompt_head(role_context)
            prompt_probs = torch.sigmoid(prompt_logits)
        return role_context, prompt_logits, prompt_probs

    def _role_weighted_prior(
        self,
        masked: torch.Tensor,
        modality_mask: torch.Tensor,
    ) -> torch.Tensor:
        b, m, h, w = masked.shape
        weights = modality_mask.to(device=masked.device, dtype=masked.dtype).view(b, m, 1, 1)
        if m != self.base_modalities * self.context_depth:
            count = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
            return masked.sum(dim=1, keepdim=True) / count

        target = int(self.config.target_base_index)
        base_weights = self._target_role_weights(masked.device, masked.dtype)
        if 0 <= target < self.base_modalities:
            base_weights[target] = 0.0

        center = self.context_depth // 2
        context_weights = masked.new_tensor(
            [1.0 / (1.0 + abs(index - center)) for index in range(self.context_depth)]
        )
        role_weights = torch.cat(
            [
                base_weights[modality_index].repeat(self.context_depth) * context_weights
                for modality_index in range(self.base_modalities)
            ],
            dim=0,
        ).view(1, m, 1, 1)
        weighted_mask = weights * role_weights
        denom = weighted_mask.sum(dim=1, keepdim=True)
        fallback_denom = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        fallback = masked.sum(dim=1, keepdim=True) / fallback_denom
        prior = (masked * weighted_mask).sum(dim=1, keepdim=True) / denom.clamp_min(1e-6)
        return torch.where(denom > 1e-6, prior, fallback)

    def _target_role_weights(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Return source-modality weights for the target BraTS contrast.

        BraTS channel order is T1n, T1c, T2w, T2-FLAIR. The weights only form
        the drift starting point; the velocity network still receives every
        observed modality and mask channel.
        """

        weights = torch.ones(self.base_modalities, device=device, dtype=dtype)
        target = int(self.config.target_base_index)
        if self.base_modalities >= 4:
            if target == 0:
                # T1n is mostly anatomy. T1c has the closest tissue contrast
                # but may contain enhancement that the learned residual should
                # remove; FLAIR/T2 provide lesion context.
                weights = torch.tensor([0.0, 0.56, 0.14, 0.30], device=device, dtype=dtype)
            elif target == 1:
                # T1c keeps T1n anatomy, then adds enhancement guided by FLAIR
                # edema extent and a smaller T2w water-content cue.
                weights = torch.tensor([0.72, 0.0, 0.06, 0.22], device=device, dtype=dtype)
            elif target == 2:
                # T2w is water/edema sensitive. FLAIR is the strongest lesion
                # partner, while T1/T1c anchor ventricles and anatomy.
                weights = torch.tensor([0.16, 0.20, 0.0, 0.64], device=device, dtype=dtype)
            elif target == 3:
                # FLAIR behaves like T2 with CSF suppression, so T2w anchors
                # lesion hyperintensity and T1/T1c help separate anatomy/core.
                weights = torch.tensor([0.20, 0.18, 0.62, 0.0], device=device, dtype=dtype)
        if 0 <= target < self.base_modalities:
            weights[target] = 0.0
        return weights

    def _refinement_detail_features(
        self,
        masked: torch.Tensor,
        mask_channels: torch.Tensor,
        stage1: torch.Tensor,
        source_energy: torch.Tensor,
    ) -> torch.Tensor:
        observed_count = mask_channels.sum(dim=1, keepdim=True).clamp_min(1.0)
        source_mean = masked.sum(dim=1, keepdim=True) / observed_count
        source_second = masked.square().sum(dim=1, keepdim=True) / observed_count
        source_std = (source_second - source_mean.square()).clamp_min(1e-6).sqrt()
        source_local_mean = F.avg_pool2d(source_mean, kernel_size=5, stride=1, padding=2)
        source_local_second = F.avg_pool2d(source_mean.square(), kernel_size=5, stride=1, padding=2)
        source_local_std = (source_local_second - source_local_mean.square()).clamp_min(1e-6).sqrt()
        source_edge, source_laplace = self._edge_laplace(source_mean)
        stage1_edge, stage1_laplace = self._edge_laplace(stage1)
        return torch.cat(
            [
                source_mean,
                source_std,
                source_energy,
                source_local_std,
                source_edge,
                source_laplace.abs(),
                stage1_edge,
                stage1_laplace.abs(),
            ],
            dim=1,
        )

    @staticmethod
    def _edge_laplace(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dtype = image.dtype
        device = image.device
        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            dtype=dtype,
            device=device,
        ).view(1, 1, 3, 3) / 8.0
        sobel_y = sobel_x.transpose(2, 3)
        laplace = torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
            dtype=dtype,
            device=device,
        ).view(1, 1, 3, 3) / 4.0
        grad_x = F.conv2d(image, sobel_x, padding=1)
        grad_y = F.conv2d(image, sobel_y, padding=1)
        edge = (grad_x.square() + grad_y.square() + 1e-8).sqrt()
        lap = F.conv2d(image, laplace, padding=1)
        return edge, lap

    def _activate(self, image: torch.Tensor) -> torch.Tensor:
        if self.config.output_activation == "tanh":
            return torch.tanh(image)
        if self.config.output_activation == "hardtanh":
            return F.hardtanh(image, min_val=-1.0, max_val=1.0)
        if self.config.output_activation == "none":
            return image
        raise ValueError(f"Unknown output_activation: {self.config.output_activation}")


def _group_count(channels: int) -> int:
    groups = min(8, channels)
    while groups > 1 and (channels % groups != 0 or channels // groups < 2):
        groups -= 1
    return groups
