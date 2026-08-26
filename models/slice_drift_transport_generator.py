from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.slice_virtual_modality_generator import ConvNormAct2d, ResidualConvBlock2d


@dataclass(frozen=True)
class SliceDriftTransportGeneratorConfig:
    in_modalities: int = 4
    hidden_channels: int = 32
    transport_steps: int = 4
    transport_step_scale: float = 1.0
    velocity_scale: float = 1.0
    uncertainty_min: float = 1e-4
    output_activation: str = "hardtanh"
    class_channels: int = 0
    class_conditioned: bool = False
    init_blur_kernel: int = 5


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
        in_channels = int(self.config.in_modalities) * 2 + 3 + class_channels
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
        current = self._initial_state(masked, modality_mask)
        states = [current]
        velocities = []
        uncertainty_raw = None
        steps = max(1, int(self.config.transport_steps))
        for step in range(steps):
            progress = float(step) / float(max(steps - 1, 1))
            remaining = float(steps - step) / float(steps)
            velocity, uncertainty_raw = self._velocity_step(
                current,
                masked,
                mask_channels,
                progress,
                remaining,
                class_map,
            )
            current = self._activate(
                current + float(self.config.transport_step_scale) * velocity
            )
            velocities.append(velocity)
            states.append(current)
        if uncertainty_raw is None:
            uncertainty_raw = torch.zeros_like(current)
        uncertainty = F.softplus(uncertainty_raw) + float(self.config.uncertainty_min)
        confidence = torch.exp(-uncertainty)
        return {
            "synthetic": current,
            "uncertainty": uncertainty,
            "confidence": confidence,
            "drift_initial": states[0],
            "drift_states": torch.stack(states, dim=1),
            "drift_velocities": torch.stack(velocities, dim=1),
            "transport_step_scale": torch.tensor(
                float(self.config.transport_step_scale),
                device=slices.device,
                dtype=slices.dtype,
            ),
        }

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
    ) -> torch.Tensor:
        weights = modality_mask.view(modality_mask.shape[0], modality_mask.shape[1], 1, 1)
        count = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        current = masked.sum(dim=1, keepdim=True) / count
        kernel = int(self.config.init_blur_kernel)
        if kernel > 1:
            if kernel % 2 == 0:
                kernel += 1
            current = F.avg_pool2d(current, kernel_size=kernel, stride=1, padding=kernel // 2)
        return self._activate(current)

    def _velocity_step(
        self,
        current: torch.Tensor,
        masked: torch.Tensor,
        mask_channels: torch.Tensor,
        progress: float,
        remaining: float,
        class_map: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, _, h, w = current.shape
        progress_map = current.new_full((b, 1, h, w), float(progress))
        remaining_map = current.new_full((b, 1, h, w), float(remaining))
        inputs = [current, masked, mask_channels, progress_map, remaining_map]
        if class_map is not None:
            inputs.append(class_map)
        x0 = self.stem(torch.cat(inputs, dim=1))
        x1 = self.down1(x0)
        x2 = self.bottleneck(self.down2(x1))
        x = F.interpolate(x2, size=x1.shape[2:], mode="bilinear", align_corners=False)
        x = self.up1(torch.cat([x, x1], dim=1))
        x = F.interpolate(x, size=x0.shape[2:], mode="bilinear", align_corners=False)
        x = self.up2(torch.cat([x, x0], dim=1))
        velocity = float(self.config.velocity_scale) * torch.tanh(self.velocity_head(x))
        uncertainty_raw = self.uncertainty_head(x)
        return velocity, uncertainty_raw

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
