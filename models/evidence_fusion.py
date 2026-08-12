from __future__ import annotations

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
    feature_stage: str = "stage4"
    feature_channels: int = 512
    hidden_channels: int = 64
    state_embedding_dim: int = 8
    share_encoder: bool = True
    min_parameter_coverage: float = 0.95


class ConvNormAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        groups = min(8, out_channels)
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimpleSegDecoder(nn.Module):
    """Small decoder for smoke and first-pass fusion experiments.

    It upsamples the selected BrainMVP feature directly to the target spatial
    size. The architecture is intentionally compact so local validation can run
    on a laptop GPU before larger H20 experiments.
    """

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


class EvidenceReliableFusion(nn.Module):
    """BrainMVP-slot fusion with conflict and evidence-reliability outputs.

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
        self.feature_stage = config.feature_stage

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
        adapter_in = config.feature_channels + 2 * config.state_embedding_dim
        self.adapters = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv3d(adapter_in, config.hidden_channels, kernel_size=1),
                    nn.InstanceNorm3d(config.hidden_channels, affine=True),
                    nn.GELU(),
                    nn.Conv3d(
                        config.hidden_channels,
                        config.feature_channels,
                        kernel_size=1,
                    ),
                )
                for _ in self.modalities
            ]
        )
        self.aux_decoders = nn.ModuleList(
            [
                SimpleSegDecoder(
                    config.feature_channels,
                    config.hidden_channels,
                    self.num_regions,
                )
                for _ in self.modalities
            ]
        )
        gate_in = (
            self.num_modalities * config.feature_channels
            + self.num_modalities * self.num_regions
            + self.num_regions
            + self.num_modalities
        )
        self.gate_net = nn.Sequential(
            nn.Conv3d(gate_in, config.hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(
                config.hidden_channels,
                self.num_regions * self.num_modalities,
                kernel_size=1,
            ),
        )
        self.seg_decoder = SimpleSegDecoder(
            config.feature_channels,
            config.hidden_channels,
            self.num_regions,
        )
        self.reliability_decoder = SimpleSegDecoder(
            config.feature_channels + self.num_regions + 1,
            config.hidden_channels,
            self.num_regions,
        )

    def forward(
        self,
        images: torch.Tensor,
        modality_mask: torch.Tensor | None = None,
        modality_state: torch.Tensor | None = None,
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

        target_spatial = (d, h, w)
        features = []
        aux_logits = []
        aux_logits_low = []
        feature_spatial = None
        for index in range(self.num_modalities):
            x = images[:, index : index + 1].contiguous()
            feature = self.encoder(x)[self.feature_stage]
            if feature_spatial is None:
                feature_spatial = feature.shape[2:]
            feature = self._apply_adapter(feature, modality_state[:, index], index)
            observed = modality_mask[:, index].view(b, 1, 1, 1, 1)
            feature = feature * observed
            features.append(feature)

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

        stacked_features = torch.stack(features, dim=1)
        stacked_aux = torch.stack(aux_logits, dim=1)
        stacked_aux_low = torch.stack(aux_logits_low, dim=1)
        conflict_low = self._conflict_map(stacked_aux_low, modality_mask)
        gates = self._evidence_gates(
            stacked_features,
            stacked_aux_low,
            conflict_low,
            modality_mask,
        )
        fused_by_region = (
            gates.unsqueeze(3) * stacked_features.unsqueeze(1)
        ).sum(dim=2)
        fused_feature = fused_by_region.mean(dim=1)
        logits = self.seg_decoder(fused_feature, target_spatial)

        conflict_high = F.interpolate(
            conflict_low,
            size=target_spatial,
            mode="trilinear",
            align_corners=False,
        )
        reliability_feature = torch.cat(
            [fused_feature, conflict_low, conflict_low.mean(dim=1, keepdim=True)],
            dim=1,
        )
        reliability_logits = self.reliability_decoder(
            reliability_feature,
            target_spatial,
        )
        reliability = torch.sigmoid(reliability_logits)

        return {
            "logits": logits,
            "reliability": reliability,
            "reliability_logits": reliability_logits,
            "gates": gates,
            "fused_feature": fused_feature,
            "features": stacked_features,
            "aux_logits": stacked_aux,
            "conflict": conflict_high,
            "conflict_low": conflict_low,
            "modality_mask": modality_mask,
            "modality_state": modality_state,
        }

    def _apply_adapter(
        self,
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
        return feature + self.adapters[modality_index](
            torch.cat([feature, condition], dim=1)
        )

    def _evidence_gates(
        self,
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
        gate_logits = self.gate_net(gate_input).view(
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
