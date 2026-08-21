from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets.brats_fusion_dataset import BRATS_MODALITIES, BRATS_REGIONS
from models.brainmvp_encoder import BrainMVPEncoder


@dataclass(frozen=True)
class EvidenceFusionConfig:
    modalities: Sequence[str] = BRATS_MODALITIES
    regions: Sequence[str] = BRATS_REGIONS
    checkpoint_path: str | None = None
    encoder_freeze: str = "freeze_all"
    fusion_stages: Sequence[str] = ("stage1", "stage2", "stage3", "stage4")
    fusion_stage_channels: Sequence[int] = (64, 128, 320, 512)
    feature_stage: str = "stage4"
    feature_channels: int = 512
    hidden_channels: int = 64
    state_embedding_dim: int = 8
    share_encoder: bool = True
    min_parameter_coverage: float = 0.95
    enable_region_query_fusion: bool = False
    completion_gate_weight: float = 0.0
    completion_gate_cap: float = 1.0
    enable_virtual_t1c_confidence: bool = False
    virtual_t1c_min_gate_cap: float = 0.05
    virtual_t1c_max_gate_cap: float = 0.45
    virtual_t1c_disagreement_scale: float = 0.25
    modality_confidence_logit_weight: float = 0.5
    enable_high_res_branch: bool = False
    high_res_channels: int = 16


def _group_count(channels: int) -> int:
    groups = min(8, channels)
    while groups > 1 and (channels % groups != 0 or channels // groups < 2):
        groups -= 1
    return groups


class ConvNormAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimpleSegDecoder(nn.Module):
    """Compact auxiliary decoder used for per-modality evidence heads."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        self.block1 = ConvNormAct(in_channels, hidden_channels)
        self.block2 = ConvNormAct(hidden_channels, hidden_channels)
        self.out = nn.Conv3d(hidden_channels, out_channels, kernel_size=1)

    def forward(
        self,
        feature: torch.Tensor,
        target_spatial: Sequence[int],
    ) -> torch.Tensor:
        x = self.block1(feature)
        x = self.block2(x)
        x = self.out(x)
        return F.interpolate(
            x,
            size=tuple(int(item) for item in target_spatial),
            mode="trilinear",
            align_corners=False,
        )


class DecoderFuseBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            ConvNormAct(in_channels, out_channels),
            ConvNormAct(out_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiScaleSegDecoder(nn.Module):
    """Progressive 3D decoder over BrainMVP stage1-stage4 feature maps."""

    def __init__(
        self,
        stage_names: Sequence[str],
        in_channels_by_stage: Mapping[str, int],
        hidden_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        if len(stage_names) < 2:
            raise ValueError("MultiScaleSegDecoder expects at least two stages.")
        self.stage_names = tuple(stage_names)
        self.projections = nn.ModuleDict(
            {
                stage: ConvNormAct(int(in_channels_by_stage[stage]), hidden_channels)
                for stage in self.stage_names
            }
        )
        low_to_high = tuple(reversed(self.stage_names))
        self.fuse_blocks = nn.ModuleDict(
            {
                stage: DecoderFuseBlock(hidden_channels * 2, hidden_channels)
                for stage in low_to_high[1:]
            }
        )
        self.out = nn.Sequential(
            ConvNormAct(hidden_channels, hidden_channels),
            nn.Conv3d(hidden_channels, out_channels, kernel_size=1),
        )

    def forward(
        self,
        features_by_stage: Mapping[str, torch.Tensor],
        target_spatial: Sequence[int],
    ) -> torch.Tensor:
        low_to_high = tuple(reversed(self.stage_names))
        x = self.projections[low_to_high[0]](features_by_stage[low_to_high[0]])
        for stage in low_to_high[1:]:
            skip = self.projections[stage](features_by_stage[stage])
            x = F.interpolate(
                x,
                size=skip.shape[2:],
                mode="trilinear",
                align_corners=False,
            )
            x = self.fuse_blocks[stage](torch.cat([x, skip], dim=1))
        x = F.interpolate(
            x,
            size=tuple(int(item) for item in target_spatial),
            mode="trilinear",
            align_corners=False,
        )
        return self.out(x)


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvNormAct(channels, channels),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(_group_count(channels), channels),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class HighResTumorBranch(nn.Module):
    """Lightweight high-resolution detail branch for tumor boundaries."""

    def __init__(
        self,
        num_modalities: int,
        state_embedding_dim: int,
        hidden_channels: int,
    ) -> None:
        super().__init__()
        in_channels = num_modalities + num_modalities + num_modalities * state_embedding_dim
        self.stem = nn.Sequential(
            ConvNormAct(in_channels, hidden_channels),
            ResidualConvBlock(hidden_channels),
        )
        self.down1 = nn.Sequential(
            nn.Conv3d(hidden_channels, hidden_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(_group_count(hidden_channels * 2), hidden_channels * 2),
            nn.GELU(),
            ResidualConvBlock(hidden_channels * 2),
        )
        self.down2 = nn.Sequential(
            nn.Conv3d(hidden_channels * 2, hidden_channels * 4, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(_group_count(hidden_channels * 4), hidden_channels * 4),
            nn.GELU(),
            ResidualConvBlock(hidden_channels * 4),
        )
        self.up1 = nn.Sequential(
            ConvNormAct(hidden_channels * 4 + hidden_channels * 2, hidden_channels * 2),
            ResidualConvBlock(hidden_channels * 2),
        )
        self.up2 = nn.Sequential(
            ConvNormAct(hidden_channels * 2 + hidden_channels, hidden_channels),
            ResidualConvBlock(hidden_channels),
        )

    def forward(
        self,
        images: torch.Tensor,
        modality_mask: torch.Tensor,
        state_embedding: torch.Tensor,
    ) -> torch.Tensor:
        b, m, d, h, w = images.shape
        masked = images * modality_mask.view(b, m, 1, 1, 1)
        mask_channels = modality_mask.view(b, m, 1, 1, 1).expand(b, m, d, h, w)
        state_channels = state_embedding.reshape(b, m * state_embedding.shape[-1], 1, 1, 1)
        state_channels = state_channels.expand(b, -1, d, h, w)
        x0 = self.stem(torch.cat([masked, mask_channels, state_channels], dim=1))
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x = F.interpolate(x2, size=x1.shape[2:], mode="trilinear", align_corners=False)
        x = self.up1(torch.cat([x, x1], dim=1))
        x = F.interpolate(x, size=x0.shape[2:], mode="trilinear", align_corners=False)
        return self.up2(torch.cat([x, x0], dim=1))


class TumorDetailFusionHead(nn.Module):
    """Fuse BrainMVP semantic logits with high-resolution local tumor evidence."""

    def __init__(
        self,
        detail_channels: int,
        hidden_channels: int,
        num_regions: int,
    ) -> None:
        super().__init__()
        fusion_in = detail_channels + num_regions + num_regions + num_regions + 1
        self.net = nn.Sequential(
            ConvNormAct(fusion_in, hidden_channels),
            ResidualConvBlock(hidden_channels),
            ConvNormAct(hidden_channels, hidden_channels),
            nn.Conv3d(hidden_channels, num_regions, kernel_size=1),
        )

    def forward(
        self,
        detail: torch.Tensor,
        semantic_logits: torch.Tensor,
        reliability: torch.Tensor,
        conflict: torch.Tensor,
    ) -> torch.Tensor:
        residual = self.net(
            torch.cat(
                [
                    detail,
                    semantic_logits,
                    reliability,
                    conflict,
                    conflict.mean(dim=1, keepdim=True),
                ],
                dim=1,
            )
        )
        return semantic_logits + residual


class StageFeatureCompletion(nn.Module):
    """Complete missing modality features from available cross-modal context."""

    def __init__(
        self,
        num_modalities: int,
        state_embedding_dim: int,
        hidden_channels: int,
    ) -> None:
        super().__init__()
        self.num_modalities = int(num_modalities)
        self.hidden_channels = int(hidden_channels)
        self.modality_embedding = nn.Embedding(num_modalities, hidden_channels)
        self.state_projection = nn.Linear(state_embedding_dim, hidden_channels)
        in_channels = (
            hidden_channels * 2
            + hidden_channels
            + hidden_channels
            + num_modalities
        )
        self.predictors = nn.ModuleList(
            [
                nn.Sequential(
                    ConvNormAct(in_channels, hidden_channels),
                    ResidualConvBlock(hidden_channels),
                    nn.Conv3d(hidden_channels, hidden_channels, kernel_size=1),
                )
                for _ in range(num_modalities)
            ]
        )

    def forward(
        self,
        features: torch.Tensor,
        modality_mask: torch.Tensor,
        state_embedding: torch.Tensor,
        keep_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, m, c, d, h, w = features.shape
        if m != self.num_modalities or c != self.hidden_channels:
            raise ValueError(
                "StageFeatureCompletion received incompatible feature shape: "
                f"{tuple(features.shape)}"
            )
        mask = modality_mask.to(features.dtype)
        keep_mask = mask if keep_mask is None else keep_mask.to(features.dtype)
        observed = mask.view(b, m, 1, 1, 1, 1)
        denom = observed.sum(dim=1).clamp_min(1.0)
        context = (features * observed).sum(dim=1) / denom
        context = context.unsqueeze(1).expand(-1, m, -1, -1, -1, -1)
        state_context = self.state_projection(state_embedding).view(b, m, c, 1, 1, 1)
        state_context = state_context.expand(-1, -1, -1, d, h, w)
        modality_ids = torch.arange(m, device=features.device)
        modality_context = self.modality_embedding(modality_ids).view(1, m, c, 1, 1, 1)
        modality_context = modality_context.expand(b, -1, -1, d, h, w)
        mask_channels = mask.view(b, m, 1, 1, 1, 1).expand(b, m, m, d, h, w)

        completed = []
        for index in range(m):
            x = torch.cat(
                [
                    features[:, index],
                    context[:, index],
                    state_context[:, index],
                    modality_context[:, index],
                    mask_channels[:, index],
                ],
                dim=1,
            )
            prediction = self.predictors[index](x)
            keep = keep_mask[:, index].view(b, 1, 1, 1, 1)
            completed.append(features[:, index] * keep + prediction * (1.0 - keep))
        completed_features = torch.stack(completed, dim=1)
        completion_residual = (completed_features - features).abs().mean(dim=2)
        return completed_features, completion_residual, context[:, 0]


class RegionQueryCrossModalFusion(nn.Module):
    """Region-conditioned cross-modal attention for modality evidence fusion."""

    def __init__(
        self,
        num_modalities: int,
        num_regions: int,
        state_embedding_dim: int,
        hidden_channels: int,
        query_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.num_modalities = int(num_modalities)
        self.num_regions = int(num_regions)
        self.hidden_channels = int(hidden_channels)
        self.query_dim = int(query_dim or hidden_channels)
        self.region_queries = nn.Parameter(
            torch.randn(num_regions, self.query_dim) * 0.02
        )
        self.feature_to_key = nn.Conv3d(hidden_channels, self.query_dim, kernel_size=1)
        self.state_to_key = nn.Linear(state_embedding_dim, self.query_dim)
        self.modality_embedding = nn.Embedding(num_modalities, self.query_dim)
        self.value_projection = nn.Conv3d(hidden_channels, hidden_channels, kernel_size=1)
        self.out_projection = nn.Sequential(
            ConvNormAct(num_regions * hidden_channels, num_regions * hidden_channels),
            nn.Conv3d(
                num_regions * hidden_channels,
                num_regions * hidden_channels,
                kernel_size=1,
            ),
        )

    def forward(
        self,
        features: torch.Tensor,
        aux_logits_low: torch.Tensor,
        conflict_low: torch.Tensor,
        modality_mask: torch.Tensor,
        state_embedding: torch.Tensor,
        completed_modalities: bool = False,
        completion_gate_weight: float = 0.0,
        completion_gate_cap: float = 1.0,
        completion_gate_cap_map: torch.Tensor | None = None,
        completion_allowed_mask: torch.Tensor | None = None,
        gate_cap_apply_mask: torch.Tensor | None = None,
        modality_confidence_low: torch.Tensor | None = None,
        modality_confidence_logit_weight: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, m, c, d, h, w = features.shape
        if m != self.num_modalities or c != self.hidden_channels:
            raise ValueError(
                "RegionQueryCrossModalFusion received incompatible feature shape: "
                f"{tuple(features.shape)}"
            )
        qdim = self.query_dim
        key = self.feature_to_key(features.reshape(b * m, c, d, h, w))
        key = key.view(b, m, qdim, d, h, w)
        state_key = self.state_to_key(state_embedding).view(b, m, qdim, 1, 1, 1)
        modality_ids = torch.arange(m, device=features.device)
        modality_key = self.modality_embedding(modality_ids).view(1, m, qdim, 1, 1, 1)
        key = key + state_key + modality_key
        key = key.permute(0, 3, 4, 5, 1, 2)

        queries = self.region_queries.view(1, 1, 1, 1, self.num_regions, 1, qdim)
        score = (queries * key.unsqueeze(4)).sum(dim=-1) / math.sqrt(qdim)
        aux_score = aux_logits_low.permute(0, 3, 4, 5, 2, 1)
        conflict_penalty = conflict_low.permute(0, 2, 3, 4, 1).unsqueeze(-1)
        score = score + aux_score - conflict_penalty
        if completed_modalities and completion_gate_weight > 0:
            allowed = (
                torch.ones_like(modality_mask)
                if completion_allowed_mask is None
                else completion_allowed_mask.to(modality_mask.dtype)
            )
            availability = modality_mask + (1.0 - modality_mask) * float(
                completion_gate_weight
            ) * allowed
            score = score.masked_fill(
                availability.view(b, 1, 1, 1, 1, m) <= 0,
                -1e4,
            )
            score = score + availability.clamp_min(1e-4).log().view(
                b, 1, 1, 1, 1, m
            )
        else:
            missing = modality_mask.view(b, 1, 1, 1, 1, m) <= 0
            score = score.masked_fill(missing, -1e4)
        if modality_confidence_low is not None:
            confidence = modality_confidence_low.to(
                device=score.device,
                dtype=score.dtype,
            )
            confidence = confidence.clamp(1e-4, 1.0)
            confidence = confidence.permute(0, 2, 3, 4, 1).unsqueeze(4)
            score = score + float(modality_confidence_logit_weight) * confidence.log()
        gates_raw = torch.softmax(score, dim=-1)
        if (
            completed_modalities
            and completion_gate_weight > 0
            and completion_gate_cap_map is not None
        ):
            completed = (modality_mask <= 0).to(gates_raw.dtype).view(
                b, 1, 1, 1, 1, m
            )
            if gate_cap_apply_mask is not None:
                capped_observed = gate_cap_apply_mask.to(gates_raw.dtype).view(
                    b, 1, 1, 1, 1, m
                )
                completed = torch.maximum(completed, capped_observed)
            observed = 1.0 - completed
            cap = completion_gate_cap_map.to(
                device=gates_raw.device,
                dtype=gates_raw.dtype,
            )
            cap = cap.permute(0, 3, 4, 5, 1, 2).clamp(0.0, 1.0)
            completed_gates = torch.minimum(gates_raw, cap) * completed
            observed_gates = gates_raw * observed
            completed_mass = completed_gates.sum(dim=-1, keepdim=True)
            if completion_gate_cap < 1.0:
                completed_target_mass = completed_mass.clamp_max(
                    float(completion_gate_cap)
                )
                completed_scale = completed_target_mass / completed_mass.clamp_min(
                    1e-8
                )
                completed_gates = completed_gates * completed_scale
                completed_mass = completed_target_mass
            observed_mass = observed_gates.sum(dim=-1, keepdim=True)
            observed_target_mass = 1.0 - completed_mass
            observed_scale = observed_target_mass / observed_mass.clamp_min(1e-8)
            gates_raw = completed_gates + observed_gates * observed_scale
        elif completed_modalities and completion_gate_weight > 0 and completion_gate_cap < 1.0:
            completed = (modality_mask <= 0).to(gates_raw.dtype).view(
                b, 1, 1, 1, 1, m
            )
            observed = 1.0 - completed
            completed_mass = (gates_raw * completed).sum(dim=-1, keepdim=True)
            observed_mass = (gates_raw * observed).sum(dim=-1, keepdim=True)
            capped_completed_mass = completed_mass.clamp_max(float(completion_gate_cap))
            observed_target_mass = 1.0 - capped_completed_mass
            completed_scale = capped_completed_mass / completed_mass.clamp_min(1e-8)
            observed_scale = observed_target_mass / observed_mass.clamp_min(1e-8)
            gates_raw = gates_raw * completed * completed_scale + gates_raw * observed * observed_scale
        gates = gates_raw.permute(0, 4, 5, 1, 2, 3)

        value = self.value_projection(features.reshape(b * m, c, d, h, w))
        value = value.view(b, m, c, d, h, w)
        fused = (gates.unsqueeze(3) * value.unsqueeze(1)).sum(dim=2)
        fused = fused.flatten(1, 2)
        fused = fused + self.out_projection(fused)
        return gates, fused


class EvidenceReliableFusion(nn.Module):
    """BrainMVP-slot fusion with multi-scale evidence-reliability outputs.

    Input tensors:
        images: [B, M, D, H, W]
        modality_mask: [B, M] where 1 means observed and 0 means missing
        modality_state: [B, M] with 0 present, 1 missing, 2 degraded

    Outputs:
        logits: [B, K, D, H, W] for ET/TC/WT
        reliability: [B, K, D, H, W], sigmoid-space evidence reliability
        gates: [B, K, M, d, h, w], class-conditioned modality weights
    """

    def __init__(self, config: EvidenceFusionConfig) -> None:
        super().__init__()
        self.config = config
        self.modalities = [name.lower() for name in config.modalities]
        self.regions = list(config.regions)
        self.num_modalities = len(self.modalities)
        self.num_regions = len(self.regions)
        self.stage_names = tuple(config.fusion_stages)
        stage_channels = tuple(int(item) for item in config.fusion_stage_channels)
        if len(self.stage_names) != len(stage_channels):
            raise ValueError("fusion_stages and fusion_stage_channels must align.")
        if len(self.stage_names) < 2:
            raise ValueError("At least two fusion stages are required.")
        self.stage_channels = dict(zip(self.stage_names, stage_channels))
        self.feature_stage = (
            config.feature_stage
            if config.feature_stage in self.stage_channels
            else self.stage_names[-1]
        )

        self.encoder = BrainMVPEncoder(
            in_channels=1,
            checkpoint_path=config.checkpoint_path,
            freeze=config.encoder_freeze,
            min_parameter_coverage=config.min_parameter_coverage,
        )
        self.modality_embedding = nn.Embedding(
            self.num_modalities, config.state_embedding_dim
        )
        self.state_embedding = nn.Embedding(3, config.state_embedding_dim)
        self.stage_adapters = nn.ModuleDict()
        for stage in self.stage_names:
            adapter_in = self.stage_channels[stage] + 2 * config.state_embedding_dim
            self.stage_adapters[stage] = nn.ModuleList(
                [
                    nn.Sequential(
                        ConvNormAct(adapter_in, config.hidden_channels),
                        ConvNormAct(config.hidden_channels, config.hidden_channels),
                    )
                    for _ in self.modalities
                ]
            )
        self.aux_decoders = nn.ModuleList(
            [
                SimpleSegDecoder(
                    config.hidden_channels,
                    config.hidden_channels,
                    self.num_regions,
                )
                for _ in self.modalities
            ]
        )
        gate_in = (
            self.num_modalities * config.hidden_channels
            + self.num_modalities * self.num_regions
            + self.num_regions
            + self.num_modalities
        )
        self.gate_nets = nn.ModuleDict(
            {
                stage: nn.Sequential(
                    nn.Conv3d(gate_in, config.hidden_channels, kernel_size=1),
                    nn.GELU(),
                    nn.Conv3d(
                        config.hidden_channels,
                        self.num_regions * self.num_modalities,
                        kernel_size=1,
                    ),
                )
                for stage in self.stage_names
            }
        )
        self.feature_completion = None
        self.region_query_fusers = None
        if config.enable_region_query_fusion:
            self.feature_completion = nn.ModuleDict(
                {
                    stage: StageFeatureCompletion(
                        self.num_modalities,
                        config.state_embedding_dim,
                        config.hidden_channels,
                    )
                    for stage in self.stage_names
                }
            )
            self.region_query_fusers = nn.ModuleDict(
                {
                    stage: RegionQueryCrossModalFusion(
                        self.num_modalities,
                        self.num_regions,
                        config.state_embedding_dim,
                        config.hidden_channels,
                    )
                    for stage in self.stage_names
                }
            )
        fused_channels = {
            stage: self.num_regions * config.hidden_channels for stage in self.stage_names
        }
        reliability_channels = {
            stage: self.num_regions * config.hidden_channels + self.num_regions + 1
            for stage in self.stage_names
        }
        self.seg_decoder = MultiScaleSegDecoder(
            self.stage_names,
            fused_channels,
            config.hidden_channels,
            self.num_regions,
        )
        self.reliability_decoder = MultiScaleSegDecoder(
            self.stage_names,
            reliability_channels,
            config.hidden_channels,
            self.num_regions,
        )
        self.high_res_branch = None
        self.detail_fusion_head = None
        if config.enable_high_res_branch:
            high_res_channels = int(config.high_res_channels)
            self.high_res_branch = HighResTumorBranch(
                self.num_modalities,
                config.state_embedding_dim,
                high_res_channels,
            )
            self.detail_fusion_head = TumorDetailFusionHead(
                high_res_channels,
                max(config.hidden_channels, high_res_channels),
                self.num_regions,
            )

    def forward(
        self,
        images: torch.Tensor,
        modality_mask: torch.Tensor | None = None,
        modality_state: torch.Tensor | None = None,
        modality_confidence: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | Mapping[str, torch.Tensor]]:
        if images.ndim != 5:
            raise ValueError(f"Expected images [B,M,D,H,W], got {tuple(images.shape)}")
        b, m, d, h, w = images.shape
        if m != self.num_modalities:
            raise ValueError(
                f"Expected {self.num_modalities} modalities, got {m}"
            )
        device = images.device
        dtype = images.dtype
        if modality_mask is None:
            modality_mask = torch.ones(b, m, dtype=dtype, device=device)
        else:
            modality_mask = modality_mask.to(device=device, dtype=dtype)
        if modality_state is None:
            modality_state = torch.zeros(b, m, dtype=torch.long, device=device)
            modality_state[modality_mask <= 0] = 1
        else:
            modality_state = modality_state.to(device=device, dtype=torch.long)
        if torch.any(modality_mask.sum(dim=1) <= 0):
            raise ValueError("Every sample must include at least one observed modality.")
        use_modality_confidence = (
            modality_confidence is not None
            and float(self.config.modality_confidence_logit_weight) > 0
        )
        if use_modality_confidence:
            modality_confidence = modality_confidence.to(device=device, dtype=dtype)
            if modality_confidence.ndim == 6 and modality_confidence.shape[2] == 1:
                modality_confidence = modality_confidence.squeeze(2)
            if tuple(modality_confidence.shape) != (b, m, d, h, w):
                raise ValueError(
                    "Expected modality_confidence shape "
                    f"{(b, m, d, h, w)}, got {tuple(modality_confidence.shape)}"
                )
            modality_confidence = modality_confidence.clamp(0.0, 1.0)
        else:
            modality_confidence = None

        target_spatial = (d, h, w)
        features_by_stage: dict[str, list[torch.Tensor | None]] = {
            stage: [] for stage in self.stage_names
        }
        reference_by_stage: dict[str, torch.Tensor] = {}
        for index in range(self.num_modalities):
            if torch.all(modality_mask[:, index] <= 0):
                for stage in self.stage_names:
                    features_by_stage[stage].append(None)
                continue
            encoded = self.encoder(images[:, index : index + 1].contiguous())
            observed = modality_mask[:, index].view(b, 1, 1, 1, 1)
            for stage in self.stage_names:
                feature = encoded[stage]
                reference_by_stage[stage] = feature
                feature = self._apply_stage_adapter(
                    stage,
                    feature,
                    modality_state[:, index],
                    index,
                )
                features_by_stage[stage].append(feature * observed)

        if not reference_by_stage:
            raise ValueError("Every batch must include at least one observed modality.")

        stacked_by_stage: dict[str, torch.Tensor] = {}
        confidence_by_stage: dict[str, torch.Tensor | None] = {}
        for stage in self.stage_names:
            reference = reference_by_stage[stage]
            filled_features: list[torch.Tensor] = []
            for feature in features_by_stage[stage]:
                if feature is None:
                    feature = reference.new_zeros(
                        (
                            b,
                            self.config.hidden_channels,
                            reference.shape[2],
                            reference.shape[3],
                            reference.shape[4],
                        )
                    )
                filled_features.append(feature)
            stacked_by_stage[stage] = torch.stack(filled_features, dim=1)
            confidence_by_stage[stage] = (
                self._resize_modality_confidence(
                    modality_confidence,
                    reference.shape[2:],
                )
                if modality_confidence is not None
                else None
            )

        aux_source = stacked_by_stage[self.stage_names[0]]
        aux_logits = []
        aux_logits_low = []
        for index in range(self.num_modalities):
            feature = aux_source[:, index]
            logits = self.aux_decoders[index](feature, target_spatial)
            aux_logits.append(logits)
            aux_logits_low.append(
                F.interpolate(
                    logits,
                    size=feature.shape[2:],
                    mode="trilinear",
                    align_corners=False,
                )
            )
        stacked_aux = torch.stack(aux_logits, dim=1)
        stacked_aux_low = torch.stack(aux_logits_low, dim=1)

        raw_stacked_by_stage = dict(stacked_by_stage)
        raw_aux_logits_low = stacked_aux_low
        state_emb_all = self.state_embedding(modality_state.clamp(0, 2))
        completion_residual_by_stage: dict[str, torch.Tensor] = {}
        completion_context_by_stage: dict[str, torch.Tensor] = {}
        completed_modalities = False
        fusion_availability_mask = modality_mask
        completion_gate_cap_map = None
        virtual_t1c_confidence = None
        virtual_t1c_disagreement = None
        virtual_t1c_gate_cap = None
        completion_allowed_mask = None
        gate_cap_apply_mask = None
        if self.feature_completion is not None:
            completion_keep_mask = modality_mask
            completion_effective_mask = modality_mask
            if "virtual_t1c" in self.modalities:
                virtual_index = self.modalities.index("virtual_t1c")
                completion_keep_mask = modality_mask.clone()
                completion_effective_mask = modality_mask.clone()
                completion_allowed_mask = torch.ones_like(modality_mask)
                completion_keep_mask[:, virtual_index] = 1.0
                completion_effective_mask[:, virtual_index] = 0.0
                completion_allowed_mask[:, virtual_index] = 0.0
            completed_by_stage: dict[str, torch.Tensor] = {}
            for stage in self.stage_names:
                completed, residual, context = self.feature_completion[stage](
                    stacked_by_stage[stage],
                    completion_effective_mask,
                    state_emb_all,
                    keep_mask=completion_keep_mask,
                )
                completed_by_stage[stage] = completed
                completion_residual_by_stage[stage] = residual
                completion_context_by_stage[stage] = context
            stacked_by_stage = completed_by_stage
            completed_modalities = True
            completion_weight = float(self.config.completion_gate_weight)
            if completion_weight > 0:
                fusion_availability_mask = (
                    modality_mask + (1.0 - modality_mask) * completion_weight
                )
                if "virtual_t1c" in self.modalities:
                    fusion_availability_mask = fusion_availability_mask.clone()
                    fusion_availability_mask[:, virtual_index] = modality_mask[
                        :, virtual_index
                    ]
            completed_aux_source = stacked_by_stage[self.stage_names[0]]
            completed_aux_logits_low = []
            for index in range(self.num_modalities):
                feature = completed_aux_source[:, index]
                logits = self.aux_decoders[index](feature, target_spatial)
                completed_aux_logits_low.append(
                    F.interpolate(
                        logits,
                        size=feature.shape[2:],
                        mode="trilinear",
                        align_corners=False,
                    )
                )
            stacked_aux_low = torch.stack(completed_aux_logits_low, dim=1)
            if self.config.enable_virtual_t1c_confidence:
                (
                    completion_gate_cap_map,
                    virtual_t1c_confidence,
                    virtual_t1c_disagreement,
                    virtual_t1c_gate_cap,
                ) = self._virtual_t1c_gate_cap(
                    raw_aux_logits_low,
                    stacked_aux_low,
                    modality_mask,
                )
            elif "virtual_t1c" in self.modalities:
                (
                    completion_gate_cap_map,
                    virtual_t1c_confidence,
                    virtual_t1c_disagreement,
                    virtual_t1c_gate_cap,
                    gate_cap_apply_mask,
                ) = self._virtual_modality_gate_cap(
                    stacked_aux_low,
                    modality_mask,
                    "virtual_t1c",
                )

        gates_by_stage: dict[str, torch.Tensor] = {}
        fused_by_stage: dict[str, torch.Tensor] = {}
        conflict_by_stage: dict[str, torch.Tensor] = {}
        for stage in self.stage_names:
            stacked_features = stacked_by_stage[stage]
            aux_stage = self._resize_aux(stacked_aux_low, stacked_features.shape[3:])
            confidence_stage = confidence_by_stage[stage]
            conflict = self._conflict_map(aux_stage, modality_mask)
            conflict_by_stage[stage] = conflict
            if self.region_query_fusers is not None:
                gates, fused = self.region_query_fusers[stage](
                    stacked_features,
                    aux_stage,
                    conflict,
                    modality_mask,
                    state_emb_all,
                    completed_modalities=completed_modalities,
                    completion_gate_weight=float(self.config.completion_gate_weight),
                    completion_gate_cap=float(self.config.completion_gate_cap),
                    completion_gate_cap_map=self._resize_gate_cap_map(
                        completion_gate_cap_map,
                        stacked_features.shape[3:],
                    ),
                    completion_allowed_mask=completion_allowed_mask,
                    gate_cap_apply_mask=gate_cap_apply_mask,
                    modality_confidence_low=confidence_stage,
                    modality_confidence_logit_weight=float(
                        self.config.modality_confidence_logit_weight
                    ),
                )
                gates_by_stage[stage] = gates
                fused_by_stage[stage] = fused
            else:
                gates = self._evidence_gates(
                    stage,
                    stacked_features,
                    aux_stage,
                    conflict,
                    modality_mask,
                )
                gates_by_stage[stage] = gates
                fused_by_region = (
                    gates.unsqueeze(3) * stacked_features.unsqueeze(1)
                ).sum(dim=2)
                fused_by_stage[stage] = fused_by_region.flatten(1, 2)

        logits = self.seg_decoder(fused_by_stage, target_spatial)
        conflict_low = conflict_by_stage[self.stage_names[0]]
        conflict_high = F.interpolate(
            conflict_low,
            size=target_spatial,
            mode="trilinear",
            align_corners=False,
        )
        reliability_features = {
            stage: torch.cat(
                [
                    fused_by_stage[stage],
                    conflict_by_stage[stage],
                    conflict_by_stage[stage].mean(dim=1, keepdim=True),
                ],
                dim=1,
            )
            for stage in self.stage_names
        }
        reliability_logits = self.reliability_decoder(
            reliability_features,
            target_spatial,
        )
        reliability = torch.sigmoid(reliability_logits)
        semantic_logits = logits
        detail_feature = None
        if self.high_res_branch is not None and self.detail_fusion_head is not None:
            detail_feature = self.high_res_branch(images, modality_mask, state_emb_all)
            logits = self.detail_fusion_head(
                detail_feature,
                semantic_logits,
                reliability,
                conflict_high,
            )

        return {
            "logits": logits,
            "semantic_logits": semantic_logits,
            "reliability": reliability,
            "reliability_logits": reliability_logits,
            "gates": gates_by_stage[self.stage_names[0]],
            "gates_by_stage": gates_by_stage,
            "fused_feature": fused_by_stage[self.stage_names[-1]],
            "features": stacked_by_stage[self.stage_names[-1]],
            "features_by_stage": stacked_by_stage,
            "raw_features_by_stage": raw_stacked_by_stage,
            "completion_residual_by_stage": completion_residual_by_stage,
            "completion_context_by_stage": completion_context_by_stage,
            "detail_feature": detail_feature,
            "aux_logits": stacked_aux,
            "fusion_aux_logits_low": stacked_aux_low,
            "conflict": conflict_high,
            "conflict_low": conflict_low,
            "conflict_by_stage": conflict_by_stage,
            "modality_mask": modality_mask,
            "modality_names": tuple(self.modalities),
            "fusion_availability_mask": fusion_availability_mask,
            "modality_confidence": modality_confidence,
            "virtual_t1c_confidence": virtual_t1c_confidence,
            "virtual_t1c_disagreement": virtual_t1c_disagreement,
            "virtual_t1c_gate_cap": virtual_t1c_gate_cap,
            "completion_gate_cap": torch.tensor(
                float(self.config.completion_gate_cap),
                dtype=images.dtype,
                device=images.device,
            ),
            "modality_state": modality_state,
        }

    def _apply_stage_adapter(
        self,
        stage: str,
        feature: torch.Tensor,
        state: torch.Tensor,
        modality_index: int,
    ) -> torch.Tensor:
        b, _, d, h, w = feature.shape
        modality_ids = torch.full(
            (b,),
            modality_index,
            dtype=torch.long,
            device=feature.device,
        )
        modality_emb = self.modality_embedding(modality_ids)
        state_emb = self.state_embedding(state.clamp(0, 2))
        condition = torch.cat([modality_emb, state_emb], dim=1)
        condition = condition.view(b, -1, 1, 1, 1).expand(b, -1, d, h, w)
        return self.stage_adapters[stage][modality_index](
            torch.cat([feature, condition], dim=1)
        )

    def _evidence_gates(
        self,
        stage: str,
        features: torch.Tensor,
        aux_logits_low: torch.Tensor,
        conflict_low: torch.Tensor,
        modality_mask: torch.Tensor,
    ) -> torch.Tensor:
        b, m, c, d, h, w = features.shape
        flat_features = features.reshape(b, m * c, d, h, w)
        flat_aux = aux_logits_low.reshape(b, m * self.num_regions, d, h, w)
        mask_channels = modality_mask.view(b, m, 1, 1, 1).expand(b, m, d, h, w)
        gate_input = torch.cat(
            [flat_features, flat_aux, conflict_low, mask_channels],
            dim=1,
        )
        gate_logits = self.gate_nets[stage](gate_input).view(
            b, self.num_regions, self.num_modalities, d, h, w
        )
        missing = modality_mask.view(b, 1, m, 1, 1, 1) <= 0
        gate_logits = gate_logits.masked_fill(missing, -1e4)
        return torch.softmax(gate_logits, dim=2)

    def _conflict_map(
        self,
        aux_logits_low: torch.Tensor,
        modality_mask: torch.Tensor,
    ) -> torch.Tensor:
        probs = torch.sigmoid(aux_logits_low)
        b, m, k, d, h, w = probs.shape
        mask = modality_mask.view(b, m, 1, 1, 1, 1).to(probs.dtype)
        denom = mask.sum(dim=1).clamp_min(1.0)
        mean = (probs * mask).sum(dim=1) / denom
        variance = ((probs - mean.unsqueeze(1)).pow(2) * mask).sum(dim=1) / denom
        return (variance + 1e-6).sqrt()

    def _virtual_t1c_gate_cap(
        self,
        observed_aux_logits_low: torch.Tensor,
        completed_aux_logits_low: torch.Tensor,
        modality_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if "t1c" not in self.modalities:
            b, m, k, d, h, w = completed_aux_logits_low.shape
            cap = completed_aux_logits_low.new_full(
                (b, k, m, d, h, w),
                float(self.config.completion_gate_cap),
            )
            empty = completed_aux_logits_low.new_zeros((b, k, d, h, w))
            return cap, empty, empty, empty
        t1c_index = self.modalities.index("t1c")
        b, m, k, d, h, w = completed_aux_logits_low.shape
        mask = modality_mask.to(completed_aux_logits_low.dtype)
        observed = mask.view(b, m, 1, 1, 1, 1)
        denom = observed.sum(dim=1).clamp_min(1.0)
        observed_prob = (
            torch.sigmoid(observed_aux_logits_low) * observed
        ).sum(dim=1) / denom
        virtual_prob = torch.sigmoid(completed_aux_logits_low[:, t1c_index])
        disagreement = (virtual_prob - observed_prob).abs()
        scale = max(float(self.config.virtual_t1c_disagreement_scale), 1e-4)
        confidence = torch.exp(-disagreement / scale)
        min_cap = float(self.config.virtual_t1c_min_gate_cap)
        max_cap = float(self.config.virtual_t1c_max_gate_cap)
        max_cap = min(max_cap, float(self.config.completion_gate_cap))
        max_cap = max(max_cap, min_cap)
        dynamic_cap = min_cap + (max_cap - min_cap) * confidence

        cap_map = completed_aux_logits_low.new_full(
            (b, k, m, d, h, w),
            float(self.config.completion_gate_cap),
        )
        missing_t1c = (mask[:, t1c_index] <= 0).view(b, 1, 1, 1, 1)
        base_t1c_cap = cap_map[:, :, t1c_index]
        cap_map[:, :, t1c_index] = torch.where(
            missing_t1c,
            dynamic_cap,
            base_t1c_cap,
        )
        return cap_map, confidence, disagreement, dynamic_cap

    def _virtual_modality_gate_cap(
        self,
        aux_logits_low: torch.Tensor,
        modality_mask: torch.Tensor,
        virtual_name: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if virtual_name not in self.modalities:
            b, m, k, d, h, w = aux_logits_low.shape
            cap = aux_logits_low.new_full(
                (b, k, m, d, h, w),
                float(self.config.completion_gate_cap),
            )
            empty = aux_logits_low.new_zeros((b, k, d, h, w))
            apply_mask = aux_logits_low.new_zeros((b, m))
            return cap, empty, empty, empty, apply_mask

        virtual_index = self.modalities.index(virtual_name)
        b, m, k, d, h, w = aux_logits_low.shape
        mask = modality_mask.to(aux_logits_low.dtype)
        observed_mask = mask.clone()
        observed_mask[:, virtual_index] = 0.0
        observed = observed_mask.view(b, m, 1, 1, 1, 1)
        denom = observed.sum(dim=1).clamp_min(1.0)
        observed_prob = (torch.sigmoid(aux_logits_low) * observed).sum(dim=1) / denom
        virtual_prob = torch.sigmoid(aux_logits_low[:, virtual_index])
        disagreement = (virtual_prob - observed_prob).abs()
        scale = max(float(self.config.virtual_t1c_disagreement_scale), 1e-4)
        confidence = torch.exp(-disagreement / scale)
        min_cap = float(self.config.virtual_t1c_min_gate_cap)
        max_cap = float(self.config.virtual_t1c_max_gate_cap)
        max_cap = min(max_cap, float(self.config.completion_gate_cap))
        max_cap = max(max_cap, min_cap)
        dynamic_cap = min_cap + (max_cap - min_cap) * confidence

        cap_map = aux_logits_low.new_full(
            (b, k, m, d, h, w),
            float(self.config.completion_gate_cap),
        )
        virtual_active = (mask[:, virtual_index] > 0).view(b, 1, 1, 1, 1)
        base_cap = cap_map[:, :, virtual_index]
        cap_map[:, :, virtual_index] = torch.where(
            virtual_active,
            dynamic_cap,
            base_cap,
        )
        apply_mask = aux_logits_low.new_zeros((b, m))
        apply_mask[:, virtual_index] = (mask[:, virtual_index] > 0).to(
            apply_mask.dtype
        )
        return cap_map, confidence, disagreement, dynamic_cap, apply_mask

    def _resize_gate_cap_map(
        self,
        cap_map: torch.Tensor | None,
        spatial: Sequence[int],
    ) -> torch.Tensor | None:
        if cap_map is None:
            return None
        b, k, m, d, h, w = cap_map.shape
        target = tuple(int(item) for item in spatial)
        if (d, h, w) == target:
            return cap_map
        resized = F.interpolate(
            cap_map.reshape(b, k * m, d, h, w),
            size=target,
            mode="trilinear",
            align_corners=False,
        )
        return resized.view(b, k, m, *target).clamp(0.0, 1.0)

    def _resize_aux(
        self,
        aux_logits_low: torch.Tensor,
        spatial: Sequence[int],
    ) -> torch.Tensor:
        b, m, k, d, h, w = aux_logits_low.shape
        resized = F.interpolate(
            aux_logits_low.reshape(b, m * k, d, h, w),
            size=tuple(int(item) for item in spatial),
            mode="trilinear",
            align_corners=False,
        )
        return resized.view(b, m, k, *tuple(int(item) for item in spatial))

    def _resize_modality_confidence(
        self,
        modality_confidence: torch.Tensor,
        spatial: Sequence[int],
    ) -> torch.Tensor:
        b, m, d, h, w = modality_confidence.shape
        target = tuple(int(item) for item in spatial)
        if (d, h, w) == target:
            return modality_confidence
        resized = F.interpolate(
            modality_confidence.reshape(b * m, 1, d, h, w),
            size=target,
            mode="trilinear",
            align_corners=False,
        )
        return resized.view(b, m, *target).clamp(0.0, 1.0)
