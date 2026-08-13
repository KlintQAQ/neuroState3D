from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from models.generators import (
    DiffusionConfig,
    DriftingConfig,
    PosteriorBackboneConfig,
    build_posterior_generator,
)


def run_method(method: str, device: torch.device, size: int) -> dict[str, object]:
    backbone = PosteriorBackboneConfig(
        latent_channels=4,
        condition_channels=4,
        hidden_channels=16,
        depth=2,
        embedding_dim=32,
    )
    if method == "diffusion":
        model = build_posterior_generator(
            method,
            backbone,
            diffusion=DiffusionConfig(timesteps=16, sampling_steps=4),
        )
    else:
        model = build_posterior_generator(
            method,
            backbone,
            drifting=DriftingConfig(samples_per_condition=3, feature_grid=2),
        )
    model = model.to(device)
    target = torch.randn(2, 4, size, size, size, device=device)
    condition = torch.randn_like(target)
    mask = torch.tensor(
        [[1, 1, 1, 0, 0], [1, 0, 1, 0, 1]], dtype=target.dtype, device=device
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    output = model.training_loss(target, condition, modality_mask=mask)
    optimizer.zero_grad(set_to_none=True)
    output["loss"].backward()
    optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    samples = model.sample(condition, n_samples=2, seed=17, modality_mask=mask)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    latency_ms = (time.perf_counter() - started) * 1000
    return {
        "method": method,
        "loss": float(output["loss"].detach().cpu()),
        "finite": bool(torch.isfinite(samples).all().item()),
        "shape": list(samples.shape),
        "nfe": model.nfe,
        "latency_ms": latency_ms,
        "peak_vram_mb": (
            torch.cuda.max_memory_allocated(device) / 1024**2
            if device.type == "cuda"
            else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--size", type=int, default=4)
    parser.add_argument("--output", default="outputs/posterior_generator_smoke.json")
    args = parser.parse_args()
    device = torch.device(args.device)
    report = {
        "scope": "engineering_smoke_not_scientific_comparison",
        "device": str(device),
        "results": [
            run_method("diffusion", device, args.size),
            run_method("drifting", device, args.size),
        ],
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
