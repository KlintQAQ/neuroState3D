from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional
import warnings

import torch
import torch.nn as nn


FeatureDict = Dict[str, torch.Tensor]


@dataclass
class CheckpointLoadReport:
    """Summary of a BrainMVP checkpoint load attempt."""

    path: str
    checkpoint_state_source: str
    checkpoint_tensor_count: int
    checkpoint_parameter_count: int
    total_encoder_tensor_count: int
    total_encoder_parameter_count: int
    matched_keys: list[str]
    matched_model_to_checkpoint: dict[str, str]
    matched_tensor_count: int
    matched_parameter_count: int
    matched_tensor_ratio: float
    matched_parameter_ratio: float
    missing_keys: list[str]
    unexpected_keys: list[str]
    shape_mismatch_keys: list[dict[str, object]]
    warnings: list[str]

    @property
    def has_large_mismatch(self) -> bool:
        return (
            len(self.missing_keys) > 0
            or len(self.shape_mismatch_keys) > 0
            or self.matched_parameter_ratio < 0.95
        )


class BrainMVPEncoder(nn.Module):
    """Wrapper around the official BrainMVP SSLEncoder.

    External NeuroState-3D tensors use ``[B, C, D, H, W]``. The official
    BrainMVP UniFormer path was trained with MONAI-style ``[B, C, H, W, D]``
    inputs and internally permutes to ``[B, C, D, H, W]``. This wrapper keeps
    the official source unchanged and isolates that convention at the boundary.

    Input shape:
        ``x``: ``[B, 1, D, H, W]``

    Output shape:
        ``{"stage0": [B, C0, D, H, W], "stage1": [B, C1, D/2, H/2, W/2], ...}``
        with channels and spatial sizes determined by the actual BrainMVP
        forward pass.
    """

    def __init__(
        self,
        in_channels: int = 1,
        checkpoint_path: Optional[str] = None,
        freeze: str = "freeze_all",
        input_layout: str = "bcdhw",
        strict_checkpoint: bool = False,
        min_parameter_coverage: float = 0.95,
        error_on_low_coverage: bool = True,
    ) -> None:
        super().__init__()
        if input_layout not in {"bcdhw", "official"}:
            raise ValueError("input_layout must be 'bcdhw' or 'official'.")
        self.in_channels = in_channels
        self.input_layout = input_layout
        self.last_load_report: Optional[CheckpointLoadReport] = None

        try:
            from models.Uniformer import SSLEncoder
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "BrainMVPEncoder requires the official BrainMVP dependencies "
                "(notably MONAI and timm). Install requirements.txt before "
                "using the real BrainMVP backbone."
            ) from exc

        self.encoder = SSLEncoder(num_phase=in_channels)

        if checkpoint_path:
            self.last_load_report = self.load_brainmvp_checkpoint(
                checkpoint_path,
                strict=strict_checkpoint,
                min_parameter_coverage=min_parameter_coverage,
                error_on_low_coverage=error_on_low_coverage,
            )
        self.set_freeze_mode(freeze)

    def set_freeze_mode(self, mode: str) -> None:
        """Set trainability for the wrapped BrainMVP encoder."""

        if mode == "freeze_all":
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()
        elif mode == "full_finetune":
            for param in self.encoder.parameters():
                param.requires_grad = True
        elif mode == "unfreeze_last_stage":
            for param in self.encoder.parameters():
                param.requires_grad = False
            for name, param in self.encoder.named_parameters():
                if ".blocks4." in name or ".norm." in name or ".patch_embed4." in name:
                    param.requires_grad = True
        else:
            raise ValueError(
                "freeze must be one of: freeze_all, unfreeze_last_stage, full_finetune"
            )

    def forward(self, x: torch.Tensor) -> FeatureDict:
        if x.ndim != 5:
            raise ValueError(f"Expected [B,C,D,H,W] tensor, got shape {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, got {x.shape[1]}"
            )

        official_x = x
        if self.input_layout == "bcdhw":
            official_x = x.permute(0, 1, 3, 4, 2).contiguous()

        features = self.encoder(official_x)
        return {f"stage{i}": feat for i, feat in enumerate(features)}

    def feature_summary(self, x: torch.Tensor) -> dict[str, dict[str, object]]:
        """Run a forward pass and return shape, dtype, memory, and channel info."""

        with torch.no_grad():
            features = self.forward(x)
        summary: dict[str, dict[str, object]] = {}
        for name, feat in features.items():
            summary[name] = {
                "shape": list(feat.shape),
                "dtype": str(feat.dtype),
                "memory_bytes": feat.numel() * feat.element_size(),
                "memory_mb": (feat.numel() * feat.element_size()) / (1024**2),
                "spatial_resolution": list(feat.shape[2:]),
                "channels": int(feat.shape[1]),
            }
        return summary

    def load_brainmvp_checkpoint(
        self,
        checkpoint_path: str,
        strict: bool = False,
        min_parameter_coverage: float = 0.95,
        error_on_low_coverage: bool = True,
    ) -> CheckpointLoadReport:
        """Load official BrainMVP-style weights into the wrapped SSLEncoder.

        The mapper first inspects checkpoint keys and encoder keys, then aligns
        them by the longest unique suffix. This avoids treating ``strict=False``
        as evidence of success. Shape-mismatched tensors are skipped and
        reported.
        """

        path = str(Path(checkpoint_path))
        raw = torch.load(path, map_location="cpu")
        state_source, raw_state = self.extract_checkpoint_state(raw)
        checkpoint_tensors = self.tensor_state_dict(raw_state)
        model_state = self.encoder.state_dict()
        parameter_keys = set(dict(self.encoder.named_parameters()))

        normalized, mapping, unexpected, shape_mismatch = self._map_checkpoint_tensors(
            checkpoint_tensors, model_state
        )
        incompatible = self.encoder.load_state_dict(normalized, strict=False)
        missing = sorted(set(incompatible.missing_keys) | (set(model_state) - set(mapping)))
        matched_parameter_count = sum(
            model_state[key].numel() for key in mapping if key in parameter_keys
        )
        total_encoder_parameter_count = sum(
            param.numel() for param in self.encoder.parameters()
        )
        total_encoder_tensor_count = len(model_state)
        checkpoint_parameter_count = sum(tensor.numel() for tensor in checkpoint_tensors.values())
        matched_tensor_count = len(mapping)
        matched_tensor_ratio = (
            matched_tensor_count / total_encoder_tensor_count
            if total_encoder_tensor_count
            else 0.0
        )
        matched_parameter_ratio = (
            matched_parameter_count / total_encoder_parameter_count
            if total_encoder_parameter_count
            else 0.0
        )

        report_warnings: list[str] = []
        if matched_parameter_ratio < min_parameter_coverage:
            report_warnings.append(
                "Low BrainMVP checkpoint parameter coverage: "
                f"{matched_parameter_ratio:.4f} < {min_parameter_coverage:.4f}"
            )

        report = CheckpointLoadReport(
            path=path,
            checkpoint_state_source=state_source,
            checkpoint_tensor_count=len(checkpoint_tensors),
            checkpoint_parameter_count=checkpoint_parameter_count,
            total_encoder_tensor_count=total_encoder_tensor_count,
            total_encoder_parameter_count=total_encoder_parameter_count,
            matched_keys=sorted(mapping),
            matched_model_to_checkpoint=dict(sorted(mapping.items())),
            matched_tensor_count=matched_tensor_count,
            matched_parameter_count=matched_parameter_count,
            matched_tensor_ratio=matched_tensor_ratio,
            matched_parameter_ratio=matched_parameter_ratio,
            missing_keys=missing,
            unexpected_keys=sorted(set(unexpected) | set(incompatible.unexpected_keys)),
            shape_mismatch_keys=shape_mismatch,
            warnings=report_warnings,
        )
        if strict and (
            report.missing_keys or report.unexpected_keys or report.shape_mismatch_keys
        ):
            raise RuntimeError(f"Checkpoint did not load strictly: {report}")
        self.last_load_report = report
        for message in report_warnings:
            warnings.warn(message, RuntimeWarning)
        if report_warnings and error_on_low_coverage:
            raise RuntimeError("; ".join(report_warnings))
        return report

    @staticmethod
    def tensor_state_dict(state: Mapping[object, object]) -> OrderedDict[str, torch.Tensor]:
        return OrderedDict(
            (str(key), value) for key, value in state.items() if torch.is_tensor(value)
        )

    @classmethod
    def extract_checkpoint_state(
        cls, checkpoint: object
    ) -> tuple[str, Mapping[object, object]]:
        if not isinstance(checkpoint, Mapping):
            raise ValueError(f"Unsupported checkpoint object type: {type(checkpoint)!r}")
        preferred = ("state_dict", "model", "model_state_dict", "network", "net")
        for key in preferred:
            value = checkpoint.get(key)
            if isinstance(value, Mapping) and cls.tensor_state_dict(value):
                return key, value
        root_tensors = cls.tensor_state_dict(checkpoint)
        if root_tensors:
            return "root", checkpoint
        nested_candidates: list[tuple[str, Mapping[object, object], int]] = []
        for key, value in checkpoint.items():
            if isinstance(value, Mapping):
                tensor_count = len(cls.tensor_state_dict(value))
                if tensor_count:
                    nested_candidates.append((str(key), value, tensor_count))
        if nested_candidates:
            nested_candidates.sort(key=lambda item: item[2], reverse=True)
            name, value, _ = nested_candidates[0]
            return name, value
        raise ValueError("No tensor state dictionary found in checkpoint.")

    @staticmethod
    def checkpoint_structure(checkpoint_path: str, first_n: int = 50) -> dict[str, object]:
        raw = torch.load(checkpoint_path, map_location="cpu")
        top_level_keys = list(raw.keys()) if isinstance(raw, Mapping) else []
        state_source, state = BrainMVPEncoder.extract_checkpoint_state(raw)
        tensors = BrainMVPEncoder.tensor_state_dict(state)
        first_items = list(tensors.items())[:first_n]
        return {
            "checkpoint_type": type(raw).__name__,
            "top_level_keys": [str(key) for key in top_level_keys],
            "state_source": state_source,
            "number_of_tensors": len(tensors),
            "first_parameter_names": [name for name, _ in first_items],
            "first_tensor_shapes": {
                name: list(tensor.shape) for name, tensor in first_items
            },
            "checkpoint_total_parameter_count": int(
                sum(tensor.numel() for tensor in tensors.values())
            ),
        }

    @classmethod
    def _map_checkpoint_tensors(
        cls,
        checkpoint_tensors: Mapping[str, torch.Tensor],
        model_state: Mapping[str, torch.Tensor],
    ) -> tuple[
        OrderedDict[str, torch.Tensor],
        dict[str, str],
        list[str],
        list[dict[str, object]],
    ]:
        suffix_to_model: dict[str, list[str]] = {}
        for model_key in model_state:
            parts = model_key.split(".")
            for start in range(len(parts) - 1):
                suffix = ".".join(parts[start:])
                suffix_to_model.setdefault(suffix, []).append(model_key)

        normalized: OrderedDict[str, torch.Tensor] = OrderedDict()
        mapping: dict[str, str] = {}
        unexpected: list[str] = []
        shape_mismatch: list[dict[str, object]] = []
        for checkpoint_key, tensor in checkpoint_tensors.items():
            model_key = cls._find_unique_suffix_match(checkpoint_key, suffix_to_model)
            if model_key is None:
                unexpected.append(checkpoint_key)
                continue
            if model_key in mapping:
                unexpected.append(f"{checkpoint_key} (duplicate for {model_key})")
                continue
            model_tensor = model_state[model_key]
            if tuple(model_tensor.shape) != tuple(tensor.shape):
                shape_mismatch.append(
                    {
                        "checkpoint_key": checkpoint_key,
                        "model_key": model_key,
                        "checkpoint_shape": list(tensor.shape),
                        "model_shape": list(model_tensor.shape),
                    }
                )
                continue
            normalized[model_key] = tensor
            mapping[model_key] = checkpoint_key
        return normalized, mapping, unexpected, shape_mismatch

    @staticmethod
    def _find_unique_suffix_match(
        checkpoint_key: str, suffix_to_model: Mapping[str, list[str]]
    ) -> Optional[str]:
        parts = checkpoint_key.split(".")
        for start in range(len(parts) - 1):
            suffix = ".".join(parts[start:])
            matches = suffix_to_model.get(suffix, [])
            if len(matches) == 1:
                return matches[0]
        return None
