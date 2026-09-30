from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from models.slice_virtual_modality_generator import ConvNormAct2d, ResidualConvBlock2d


@dataclass(frozen=True)
class ClosedLoopFeedbackConfig:
    in_modalities: int
    hidden_channels: int
    feedback_channels: int = 32
    prompt_channels: int = 0
    class_channels: int = 0
    detach_feedback: bool = False


class ClosedLoopFeedbackController(nn.Module):
    """Predict task-like feedback used to modulate the drift path.

    The controller is deliberately image-conditioned rather than label-fed:
    training can supervise its lesion logits with BraTS regions, but inference
    only uses predictions from the generated slice, observed modalities,
    prompt probabilities, and uncertainty.
    """

    def __init__(self, config: ClosedLoopFeedbackConfig) -> None:
        super().__init__()
        self.config = config
        hidden = int(config.hidden_channels)
        in_channels = (
            1
            + int(config.in_modalities) * 2
            + hidden
            + int(config.prompt_channels)
            + int(config.class_channels)
            + 2
        )
        self.context = nn.Sequential(
            ConvNormAct2d(in_channels, hidden),
            ResidualConvBlock2d(hidden),
            ResidualConvBlock2d(hidden),
        )
        self.feedback_head = nn.Conv2d(hidden, int(config.feedback_channels), kernel_size=1)
        self.lesion_head = nn.Conv2d(hidden, 3, kernel_size=1)
        self.failure_head = nn.Conv2d(hidden, 1, kernel_size=1)

    def forward(
        self,
        current: torch.Tensor,
        masked: torch.Tensor,
        mask_channels: torch.Tensor,
        base_features: torch.Tensor,
        uncertainty_raw: torch.Tensor,
        prompt_probs: torch.Tensor | None = None,
        class_map: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        b, _, h, w = current.shape
        uncertainty = torch.nn.functional.softplus(uncertainty_raw)
        source_energy = masked.abs().amax(dim=1, keepdim=True)
        inputs = [
            current,
            masked,
            mask_channels,
            base_features,
            uncertainty,
            source_energy,
        ]
        if int(self.config.prompt_channels) > 0:
            if prompt_probs is None:
                prompt_probs = current.new_zeros(
                    b,
                    int(self.config.prompt_channels),
                    h,
                    w,
                )
            inputs.append(prompt_probs)
        if int(self.config.class_channels) > 0:
            if class_map is None:
                class_map = current.new_zeros(b, int(self.config.class_channels), h, w)
            inputs.append(class_map)
        features = self.context(torch.cat(inputs, dim=1))
        feedback = torch.tanh(self.feedback_head(features))
        if bool(self.config.detach_feedback):
            feedback = feedback.detach()
        return {
            "feedback_map": feedback,
            "feedback_features": features,
            "feedback_lesion_logits": self.lesion_head(features),
            "feedback_failure_logits": self.failure_head(features),
        }
