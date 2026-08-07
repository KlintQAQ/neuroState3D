from __future__ import annotations

from typing import Dict, Mapping, Optional

import torch
import torch.nn as nn

from models.brainmvp_encoder import BrainMVPEncoder, FeatureDict
from models.modality_adapter import ModalityAdapterBank
from models.multimodal_fusion import MeanFusion, build_fusion, stack_modalities


class NeuroState3D(nn.Module):
    """First-round NeuroState-3D model.

    This version intentionally stops at deterministic mask-aware feature
    fusion. It does not implement teacher learning, diffusion, uncertainty, or
    clinical downstream heads.

    Input:
        ``modalities``: mapping from modality name to ``[B, 1, D, H, W]`` tensor
        ``modality_mask``: optional ``[B, M]`` tensor in ``self.modalities`` order

    Output:
        dictionary containing per-modality multi-scale features, selected
        features, fused representation, modality mask, and ``brain_state`` set
        to the fused representation for this first round.
    """

    def __init__(
        self,
        modalities: list[str],
        encoder: Optional[nn.Module] = None,
        fusion_type: str = "mean",
        feature_stage: str = "stage4",
        adapter_types: Optional[Mapping[str, str]] = None,
        adapter_enabled: bool = True,
        adapter_hidden_channels: int = 8,
        input_channels: int = 1,
        checkpoint_path: Optional[str] = None,
        freeze_backbone: str = "freeze_all",
        fusion_channels: Optional[int] = None,
        fusion_hidden_channels: Optional[int] = None,
    ) -> None:
        super().__init__()
        if not modalities:
            raise ValueError("At least one modality must be configured.")
        self.modalities = [m.lower() for m in modalities]
        self.feature_stage = feature_stage
        self.input_channels = input_channels

        self.adapters = (
            ModalityAdapterBank(
                modalities=self.modalities,
                adapter_types=adapter_types,
                channels=input_channels,
                hidden_channels=adapter_hidden_channels,
            )
            if adapter_enabled
            else ModalityAdapterBank(
                modalities=self.modalities,
                adapter_types={m: "identity" for m in self.modalities},
                channels=input_channels,
            )
        )

        self.encoder = encoder or BrainMVPEncoder(
            in_channels=input_channels,
            checkpoint_path=checkpoint_path,
            freeze=freeze_backbone,
        )
        self.fusion_type = fusion_type
        self._fusion_channels = fusion_channels
        self._fusion_hidden_channels = fusion_hidden_channels
        self.fusion: Optional[nn.Module] = None
        if fusion_type == "mean":
            self.fusion = MeanFusion()

    def encode_modality(self, x: torch.Tensor, modality_name: str) -> FeatureDict:
        key = modality_name.lower()
        if key not in self.modalities:
            raise KeyError(f"Unknown modality '{modality_name}'. Known: {self.modalities}")
        adapted = self.adapters(x, key)
        return self.encoder(adapted)

    def forward(
        self,
        modalities: Mapping[str, Optional[torch.Tensor]],
        modality_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, object]:
        present_modalities = {
            name.lower(): tensor
            for name, tensor in modalities.items()
            if tensor is not None
        }
        if not present_modalities:
            raise ValueError("At least one observed modality tensor is required.")

        mask = self._build_or_validate_mask(present_modalities, modality_mask)
        modality_features: Dict[str, FeatureDict] = {}
        selected_features: Dict[str, torch.Tensor] = {}
        for modality, tensor in present_modalities.items():
            if modality not in self.modalities:
                raise KeyError(f"Unknown modality '{modality}'. Known: {self.modalities}")
            features = self.encode_modality(tensor, modality)
            if self.feature_stage not in features:
                raise KeyError(
                    f"feature_stage '{self.feature_stage}' not found. "
                    f"Available stages: {list(features)}"
                )
            modality_features[modality] = features
            selected_features[modality] = features[self.feature_stage]

        stacked = stack_modalities(selected_features, mask.clone(), self.modalities)
        fusion = self._get_fusion(stacked)
        fused_feature, fusion_weights = fusion(stacked, mask)

        return {
            "modality_features": modality_features,
            "selected_features": selected_features,
            "fused_feature": fused_feature,
            "brain_state": fused_feature,
            "fusion_weights": fusion_weights,
            "modality_mask": mask,
            "modalities": self.modalities,
        }

    def _get_fusion(self, stacked_features: torch.Tensor) -> nn.Module:
        if self.fusion is not None:
            return self.fusion
        _, num_modalities, channels, _, _, _ = stacked_features.shape
        self.fusion = build_fusion(
            self.fusion_type,
            num_modalities=num_modalities,
            in_channels=channels,
            out_channels=self._fusion_channels or channels,
            hidden_channels=self._fusion_hidden_channels,
        )
        self.fusion.to(device=stacked_features.device)
        return self.fusion

    def _build_or_validate_mask(
        self,
        present_modalities: Mapping[str, torch.Tensor],
        modality_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        reference = next(iter(present_modalities.values()))
        batch_size = reference.shape[0]
        device = reference.device
        if modality_mask is None:
            mask = torch.zeros(batch_size, len(self.modalities), device=device)
            for index, modality in enumerate(self.modalities):
                if modality in present_modalities:
                    mask[:, index] = 1
            return mask

        if modality_mask.shape != (batch_size, len(self.modalities)):
            raise ValueError(
                f"Expected modality_mask shape {(batch_size, len(self.modalities))}, "
                f"got {tuple(modality_mask.shape)}"
            )
        mask = modality_mask.to(device=device, dtype=reference.dtype).clone()
        for index, modality in enumerate(self.modalities):
            if modality not in present_modalities:
                mask[:, index] = 0
        if torch.any(mask.sum(dim=1) <= 0):
            raise ValueError("Every sample must have at least one observed modality.")
        return mask
