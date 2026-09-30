from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.slice_virtual_modality_generator import ConvNormAct2d, ResidualConvBlock2d


@dataclass(frozen=True)
class TokenDriftAdapterConfig:
    in_modalities: int
    hidden_channels: int
    token_channels: int = 64
    token_heads: int = 4
    token_stride: int = 4
    feedback_channels: int = 32
    prompt_channels: int = 0
    adapter_scale: float = 0.12
    gate_bias_init: float = -3.0


class TokenDriftAdapter(nn.Module):
    """Cross-modal token adapter that predicts a residual drift velocity.

    The adapter keeps the generator in drifting form: it never directly
    regresses a final image. It turns the current target estimate into query
    tokens and the observed modalities plus feedback into key/value tokens,
    then emits a small gated delta velocity.
    """

    def __init__(self, config: TokenDriftAdapterConfig) -> None:
        super().__init__()
        self.config = config
        token_channels = int(config.token_channels)
        heads = _valid_heads(token_channels, int(config.token_heads))
        stride = max(1, int(config.token_stride))
        self.token_stride = stride
        self.current_token = nn.Conv2d(
            1,
            token_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
        )
        source_channels = (
            int(config.in_modalities) * 2
            + int(config.feedback_channels)
            + int(config.prompt_channels)
        )
        self.source_token = nn.Conv2d(
            source_channels,
            token_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
        )
        self.query_norm = nn.LayerNorm(token_channels)
        self.source_norm = nn.LayerNorm(token_channels)
        self.cross_attention = nn.MultiheadAttention(
            token_channels,
            heads,
            batch_first=True,
        )
        local_channels = source_channels + 1
        self.local_context = nn.Sequential(
            ConvNormAct2d(local_channels, int(config.hidden_channels)),
            ResidualConvBlock2d(int(config.hidden_channels)),
        )
        self.attended_project = nn.Sequential(
            nn.Conv2d(token_channels, int(config.hidden_channels), kernel_size=1),
            nn.GroupNorm(_group_count(int(config.hidden_channels)), int(config.hidden_channels)),
            nn.GELU(),
        )
        self.fuse = nn.Sequential(
            ConvNormAct2d(int(config.hidden_channels) * 2, int(config.hidden_channels)),
            ResidualConvBlock2d(int(config.hidden_channels)),
        )
        self.gate_head = nn.Conv2d(int(config.hidden_channels), 1, kernel_size=1)
        self.delta_head = nn.Conv2d(int(config.hidden_channels), 1, kernel_size=1)
        nn.init.constant_(self.gate_head.bias, float(config.gate_bias_init))
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)

    def forward(
        self,
        current: torch.Tensor,
        masked: torch.Tensor,
        mask_channels: torch.Tensor,
        feedback_map: torch.Tensor,
        prompt_probs: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        b, _, h, w = current.shape
        if int(self.config.prompt_channels) > 0:
            if prompt_probs is None:
                prompt_probs = current.new_zeros(
                    b,
                    int(self.config.prompt_channels),
                    h,
                    w,
                )
            source = torch.cat([masked, mask_channels, feedback_map, prompt_probs], dim=1)
        else:
            source = torch.cat([masked, mask_channels, feedback_map], dim=1)
        query_map = self.current_token(current)
        source_map = self.source_token(source)
        query = self.query_norm(_flatten_tokens(query_map))
        source_tokens = self.source_norm(_flatten_tokens(source_map))
        attended, _ = self.cross_attention(
            query,
            source_tokens,
            source_tokens,
            need_weights=False,
        )
        attended_map = _unflatten_tokens(attended, query_map)
        attended_map = F.interpolate(
            attended_map,
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )
        local = self.local_context(torch.cat([current, source], dim=1))
        attended_features = self.attended_project(attended_map)
        features = self.fuse(torch.cat([local, attended_features], dim=1))
        gate = torch.sigmoid(self.gate_head(features))
        delta = float(self.config.adapter_scale) * gate * torch.tanh(self.delta_head(features))
        return {
            "delta_velocity": delta,
            "adapter_gate": gate,
            "adapter_features": features,
        }


def _flatten_tokens(value: torch.Tensor) -> torch.Tensor:
    return value.flatten(2).transpose(1, 2)


def _unflatten_tokens(tokens: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    b, c, h, w = reference.shape
    return tokens.transpose(1, 2).reshape(b, c, h, w)


def _valid_heads(channels: int, requested: int) -> int:
    heads = max(1, min(channels, requested))
    while heads > 1 and channels % heads != 0:
        heads -= 1
    return heads


def _group_count(channels: int) -> int:
    groups = min(8, channels)
    while groups > 1 and (channels % groups != 0 or channels // groups < 2):
        groups -= 1
    return groups
