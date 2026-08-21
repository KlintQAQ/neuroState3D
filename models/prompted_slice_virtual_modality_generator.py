from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.slice_virtual_modality_generator import ConvNormAct2d, ResidualConvBlock2d


@dataclass(frozen=True)
class PromptedSliceVirtualModalityGeneratorConfig:
    in_modalities: int = 4
    hidden_channels: int = 32
    prompt_channels: int = 3
    residual_scale: float = 0.35
    lesion_residual_scale: float = 0.45
    detail_residual_scale: float = 0.20
    enhancement_residual_scale: float = 0.0
    class_channels: int = 0
    class_conditioned: bool = False
    uncertainty_min: float = 1e-4
    output_activation: str = "tanh"
    positive_lesion_residual: bool = False


class PromptedSliceVirtualModalityGenerator(nn.Module):
    """Slice generator with an auxiliary medical prompt head.

    The prompt head predicts ET/TC/WT soft regions from the observed modalities.
    The synthesis head then receives both decoder features and prompt
    probabilities, making small lesion evidence harder to average away.
    """

    def __init__(
        self,
        config: PromptedSliceVirtualModalityGeneratorConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or PromptedSliceVirtualModalityGeneratorConfig()
        hidden = int(self.config.hidden_channels)
        class_channels = int(self.config.class_channels) if self.config.class_conditioned else 0
        in_channels = int(self.config.in_modalities) * 2 + class_channels
        self.stem = nn.Sequential(
            ConvNormAct2d(in_channels, hidden),
            ResidualConvBlock2d(hidden),
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(hidden, hidden * 2, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, hidden * 2),
            nn.GELU(),
            ResidualConvBlock2d(hidden * 2),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(hidden * 2, hidden * 4, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, hidden * 4),
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
        self.prompt_head = nn.Conv2d(hidden, int(self.config.prompt_channels), kernel_size=1)
        self.refine = nn.Sequential(
            ConvNormAct2d(hidden + int(self.config.prompt_channels), hidden),
            ResidualConvBlock2d(hidden),
        )
        lesion_hidden = max(8, hidden // 2)
        self.lesion_residual = nn.Sequential(
            ConvNormAct2d(int(self.config.prompt_channels), lesion_hidden),
            ResidualConvBlock2d(lesion_hidden),
            nn.Conv2d(lesion_hidden, 1, kernel_size=1),
        )
        detail_in = (
            hidden
            + int(self.config.prompt_channels)
            + int(self.config.in_modalities)
            + class_channels
        )
        self.detail_residual = nn.Sequential(
            ConvNormAct2d(detail_in, hidden),
            ResidualConvBlock2d(hidden),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )
        self.enhancement_residual = nn.Sequential(
            ConvNormAct2d(detail_in, hidden),
            ResidualConvBlock2d(hidden),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )
        self.base_out = nn.Conv2d(hidden, 2, kernel_size=1)
        self.synthetic_head = nn.Conv2d(hidden, 1, kernel_size=1)
        self.uncertainty_head = nn.Conv2d(hidden, 1, kernel_size=1)
        nn.init.zeros_(self.synthetic_head.weight)
        nn.init.zeros_(self.synthetic_head.bias)
        nn.init.zeros_(self.uncertainty_head.weight)
        nn.init.zeros_(self.uncertainty_head.bias)
        nn.init.zeros_(self.lesion_residual[-1].weight)
        nn.init.zeros_(self.lesion_residual[-1].bias)
        nn.init.zeros_(self.detail_residual[-1].weight)
        nn.init.zeros_(self.detail_residual[-1].bias)
        nn.init.zeros_(self.enhancement_residual[-1].weight)
        nn.init.zeros_(self.enhancement_residual[-1].bias)

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
        masked = slices * modality_mask.view(b, m, 1, 1)
        mask_channels = modality_mask.view(b, m, 1, 1).expand(b, m, h, w)
        class_channels = int(self.config.class_channels) if self.config.class_conditioned else 0
        if class_channels > 0:
            if class_condition is None:
                class_condition = torch.zeros(
                    b,
                    class_channels,
                    device=slices.device,
                    dtype=slices.dtype,
                )
            class_condition = class_condition.to(device=slices.device, dtype=slices.dtype)
            if tuple(class_condition.shape) != (b, class_channels):
                raise ValueError(
                    f"Expected class_condition {(b, class_channels)}, got {tuple(class_condition.shape)}"
                )
            class_map = class_condition.view(b, class_channels, 1, 1).expand(
                b,
                class_channels,
                h,
                w,
            )
        else:
            class_map = None
        stem_inputs = [masked, mask_channels]
        if class_map is not None:
            stem_inputs.append(class_map)
        x0 = self.stem(torch.cat(stem_inputs, dim=1))
        x1 = self.down1(x0)
        x2 = self.bottleneck(self.down2(x1))
        x = F.interpolate(x2, size=x1.shape[2:], mode="bilinear", align_corners=False)
        x = self.up1(torch.cat([x, x1], dim=1))
        x = F.interpolate(x, size=x0.shape[2:], mode="bilinear", align_corners=False)
        x = self.up2(torch.cat([x, x0], dim=1))
        prompt_logits = self.prompt_head(x)
        prompt_probs = torch.sigmoid(prompt_logits)
        refined = self.refine(torch.cat([x, prompt_probs], dim=1))
        base = self.base_out(x)
        residual = float(self.config.residual_scale) * self.synthetic_head(refined)
        lesion_residual_raw = self.lesion_residual(prompt_probs)
        if self.config.positive_lesion_residual:
            lesion_gate = prompt_probs[:, 0:1]
            lesion_residual = (
                float(self.config.lesion_residual_scale)
                * lesion_gate
                * torch.sigmoid(lesion_residual_raw)
            )
        else:
            lesion_residual = float(self.config.lesion_residual_scale) * lesion_residual_raw
        detail_inputs = [x, prompt_probs, masked]
        if class_map is not None:
            detail_inputs.append(class_map)
        detail_input = torch.cat(detail_inputs, dim=1)
        detail_residual = float(self.config.detail_residual_scale) * self.detail_residual(detail_input)
        prompt_enhancement_gate = (
            0.70 * prompt_probs[:, 0:1]
            + 0.25 * prompt_probs[:, 1:2]
            + 0.05 * prompt_probs[:, 2:3]
        ).clamp(0.0, 1.0)
        local_mean = F.avg_pool2d(masked, kernel_size=9, stride=1, padding=4)
        source_contrast = (masked - local_mean).abs().mean(dim=1, keepdim=True)
        contrast_flat = source_contrast.flatten(2)
        contrast_mean = contrast_flat.mean(dim=2, keepdim=True).view(b, 1, 1, 1)
        contrast_std = contrast_flat.std(dim=2, unbiased=False, keepdim=True).view(b, 1, 1, 1)
        source_contrast_gate = torch.sigmoid(
            (source_contrast - contrast_mean - 0.25 * contrast_std)
            / contrast_std.clamp_min(1e-4)
        )
        enhancement_gate = (
            0.10
            + 0.60 * prompt_enhancement_gate
            + 0.30 * source_contrast_gate
        ).clamp(0.0, 1.0)
        enhancement_raw = self.enhancement_residual(detail_input)
        centered_enhancement = F.softplus(enhancement_raw) - 0.6931471805599453
        enhancement_residual = (
            float(self.config.enhancement_residual_scale)
            * enhancement_gate
            * centered_enhancement
        )
        base_logits = base[:, :1]
        pre_enhancement_logits = base_logits + residual + lesion_residual + detail_residual
        synthetic_logits = pre_enhancement_logits + enhancement_residual
        if self.config.output_activation == "tanh":
            synthetic = torch.tanh(synthetic_logits)
            pre_enhancement_synthetic = torch.tanh(pre_enhancement_logits)
        elif self.config.output_activation == "hardtanh":
            synthetic = F.hardtanh(synthetic_logits, min_val=-1.0, max_val=1.0)
            pre_enhancement_synthetic = F.hardtanh(
                pre_enhancement_logits,
                min_val=-1.0,
                max_val=1.0,
            )
        elif self.config.output_activation == "none":
            synthetic = synthetic_logits
            pre_enhancement_synthetic = pre_enhancement_logits
        else:
            raise ValueError(f"Unknown output_activation: {self.config.output_activation}")
        uncertainty = F.softplus(
            base[:, 1:2] + float(self.config.residual_scale) * self.uncertainty_head(refined)
        ) + float(
            self.config.uncertainty_min
        )
        confidence = torch.exp(-uncertainty)
        return {
            "synthetic": synthetic,
            "uncertainty": uncertainty,
            "confidence": confidence,
            "prompt_logits": prompt_logits,
            "prompt_probs": prompt_probs,
            "residual": residual,
            "lesion_residual": lesion_residual,
            "lesion_residual_raw": lesion_residual_raw,
            "detail_residual": detail_residual,
            "enhancement_residual": enhancement_residual,
            "enhancement_residual_raw": enhancement_raw,
            "enhancement_gate": enhancement_gate,
            "base_logits": base_logits,
            "pre_enhancement_logits": pre_enhancement_logits,
            "pre_enhancement_synthetic": pre_enhancement_synthetic,
            "synthetic_logits": synthetic_logits,
        }
