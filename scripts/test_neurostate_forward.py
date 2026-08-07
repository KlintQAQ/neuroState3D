from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.neurostate3d import NeuroState3D
from models.brainmvp_encoder import BrainMVPEncoder
from models.modality_adapter import DEFAULT_ADAPTER_TYPES, MODALITY_ORDER


class TinyMultiScaleEncoder(nn.Module):
    """Small test encoder with the same dictionary contract as BrainMVPEncoder."""

    def __init__(self, in_channels: int = 1) -> None:
        super().__init__()
        self.stage1 = nn.Conv3d(in_channels, 4, kernel_size=3, padding=1)
        self.stage2 = nn.Conv3d(4, 8, kernel_size=3, stride=2, padding=1)
        self.stage3 = nn.Conv3d(8, 16, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        s0 = x
        s1 = torch.relu(self.stage1(s0))
        s2 = torch.relu(self.stage2(s1))
        s3 = torch.relu(self.stage3(s2))
        return {"stage0": s0, "stage1": s1, "stage2": s2, "stage3": s3}


def make_inputs(batch_size: int, size: int, device: str) -> dict[str, torch.Tensor]:
    return {
        "t1": torch.randn(batch_size, 1, size, size, size, device=device),
        "t2": torch.randn(batch_size, 1, size, size, size, device=device),
        "fa": torch.randn(batch_size, 1, size, size, size, device=device),
        "md": torch.randn(batch_size, 1, size, size, size, device=device),
        "alff": torch.randn(batch_size, 1, size, size, size, device=device),
    }


def assert_forward(model: NeuroState3D, inputs: dict[str, torch.Tensor]) -> None:
    output = model(inputs)
    fused = output["fused_feature"]
    mask = output["modality_mask"]
    assert isinstance(fused, torch.Tensor)
    assert isinstance(mask, torch.Tensor)
    assert fused.ndim == 5
    assert torch.isfinite(fused).all()
    if output["fusion_weights"] is not None:
        weights = output["fusion_weights"]
        assert torch.isfinite(weights).all()
        observed = mask.sum(dim=1).clamp_min(1.0)
        assert torch.allclose(
            weights.sum(dim=1).flatten(), torch.ones_like(observed).flatten()
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test NeuroState3D forward.")
    parser.add_argument("--fusion", default="mean", choices=["mean", "concat"])
    parser.add_argument("--size", default=16, type=int)
    parser.add_argument("--batch-size", default=2, type=int)
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument("--encoder", default="tiny", choices=["tiny", "brainmvp"])
    parser.add_argument("--checkpoint", default="", type=str)
    parser.add_argument("--feature-stage", default="", type=str)
    args = parser.parse_args()

    modalities = list(MODALITY_ORDER)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but this PyTorch build cannot use CUDA.")
    base = make_inputs(args.batch_size, args.size, args.device)
    if args.encoder == "brainmvp":
        if not args.checkpoint:
            raise ValueError("--checkpoint is required when --encoder brainmvp")
        encoder: nn.Module = BrainMVPEncoder(
            checkpoint_path=args.checkpoint,
            freeze="freeze_all",
        )
        feature_stage = args.feature_stage or "stage4"
    else:
        encoder = TinyMultiScaleEncoder()
        feature_stage = args.feature_stage or "stage3"

    model = NeuroState3D(
        modalities=modalities,
        encoder=encoder,
        fusion_type=args.fusion,
        feature_stage=feature_stage,
        adapter_types=DEFAULT_ADAPTER_TYPES,
    )
    model.to(args.device)
    model.eval()

    cases = {
        "t1_only": ["t1"],
        "t1_t2": ["t1", "t2"],
        "t1_fa": ["t1", "fa"],
        "t1_t2_fa_alff": ["t1", "t2", "fa", "alff"],
        "all": modalities,
    }
    with torch.no_grad():
        for case_name, keep in cases.items():
            case_inputs = {name: base[name] for name in keep}
            assert_forward(model, case_inputs)
            print(f"ok: {case_name}")

        mixed_inputs = {name: base[name] for name in modalities}
        mixed_mask = torch.zeros(args.batch_size, len(modalities), device=args.device)
        for row in range(args.batch_size):
            if row % 2 == 0:
                mixed_mask[row, 0] = 1
            else:
                mixed_mask[row, [0, 1, 2, 4]] = 1
        mixed = model(mixed_inputs, modality_mask=mixed_mask)
        assert torch.isfinite(mixed["fused_feature"]).all()
        assert not math.isnan(float(mixed["fused_feature"].mean()))
        print("ok: mixed_batch_mask")


if __name__ == "__main__":
    main()
