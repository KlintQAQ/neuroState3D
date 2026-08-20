from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.generators import (  # noqa: E402
    DriftingConfig,
    EvidenceContradictionBank,
    PosteriorBackboneConfig,
    build_posterior_generator,
    evidence_contradiction_score,
)


def build_config(method: str) -> DriftingConfig:
    contradiction_enabled = method in {"contradiction", "both"}
    nested_enabled = method in {"nested", "both"}
    return DriftingConfig(
        samples_per_condition=4,
        feature_grid=2,
        contradiction_repulsion_strength=2.0 if contradiction_enabled else 0.0,
        contradiction_max_weight=6.0,
        nested_identity_weight=0.5 if nested_enabled else 0.0,
        nested_contraction_weight=0.5 if nested_enabled else 0.0,
        nested_pool_size=2,
    )


def run(method: str, device: torch.device, size: int) -> dict[str, object]:
    torch.manual_seed(31)
    backbone = PosteriorBackboneConfig(
        latent_channels=3,
        condition_channels=4,
        hidden_channels=8,
        depth=2,
        num_modalities=5,
        num_targets=5,
        quality_channels=1,
        embedding_dim=16,
    )
    model = build_posterior_generator(
        "drifting", backbone, drifting=build_config(method)
    ).to(device)
    # The production backbone starts with a zero output projection for stable
    # training. A tiny non-zero smoke initialization makes the optional nested
    # diagnostics observable before a real checkpoint has been trained.
    torch.nn.init.normal_(model.model.output_projection.weight, std=0.02)
    batch_size = 2
    target = torch.randn(batch_size, 3, size, size, size, device=device)
    coarse_condition = torch.randn(
        batch_size, 4, size, size, size, device=device
    )
    richer_condition = coarse_condition + 0.1 * torch.randn_like(coarse_condition)
    coarse_mask = torch.tensor(
        [[1, 0, 1, 0, 0], [1, 0, 1, 0, 0]],
        device=device,
        dtype=target.dtype,
    )
    richer_mask = torch.tensor(
        [[1, 1, 1, 0, 0], [1, 0, 1, 1, 0]],
        device=device,
        dtype=target.dtype,
    )
    target_ids = torch.ones(batch_size, device=device, dtype=torch.long)
    quality = torch.ones(batch_size, 1, size, size, size, device=device)

    kwargs: dict[str, torch.Tensor] = {}
    bank_size = 0
    if method in {"contradiction", "both"}:
        candidate_negatives = target[:, None] + torch.stack(
            [
                0.2 * torch.randn_like(target),
                0.8 * torch.randn_like(target),
                1.6 * torch.randn_like(target),
            ],
            dim=1,
        )
        scores = evidence_contradiction_score(
            candidate_negatives,
            target,
            reliability=quality,
            uncertainty=torch.zeros_like(candidate_negatives),
        )
        bank = EvidenceContradictionBank(capacity=16)
        bank.add(candidate_negatives, scores, target_ids=target_ids)
        sampled = bank.sample(
            target_ids,
            n_samples=2,
            device=device,
            dtype=target.dtype,
        )
        kwargs["negative_latents"] = sampled.latents
        kwargs["negative_scores"] = sampled.scores
        bank_size = len(bank)

    if method in {"nested", "both"}:
        kwargs["richer_condition"] = richer_condition
        kwargs["richer_modality_mask"] = richer_mask
        kwargs["richer_quality"] = quality

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    output = model.training_loss(
        target,
        coarse_condition,
        modality_mask=coarse_mask,
        target_ids=target_ids,
        quality=quality,
        **kwargs,
    )
    optimizer.zero_grad(set_to_none=True)
    output["loss"].backward()
    gradient_norm = torch.sqrt(
        sum(
            parameter.grad.detach().float().square().sum()
            for parameter in model.parameters()
            if parameter.grad is not None
        )
    )
    optimizer.step()

    return {
        "method": method,
        "loss": float(output["loss"].detach().cpu()),
        "native_loss": float(output["native_loss"].cpu()),
        "contradiction_weight_mean": float(
            output["contradiction_weight_mean"].cpu()
        ),
        "nested_identity_loss": float(output["nested_identity_loss"].cpu()),
        "nested_contraction_loss": float(
            output["nested_contraction_loss"].cpu()
        ),
        "bank_size": bank_size,
        "gradient_norm": float(gradient_norm.cpu()),
        "finite": bool(torch.isfinite(output["loss"]).item()),
        "generated_shape": list(output["generated"].shape),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run contradiction, nested-subset, or combined Drifting smoke tests."
    )
    parser.add_argument(
        "--method",
        choices=("contradiction", "nested", "both"),
        required=True,
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--size", type=int, default=4)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    report = {
        "scope": "synthetic engineering smoke; not a scientific efficacy result",
        "device": args.device,
        "result": run(args.method, torch.device(args.device), args.size),
    }
    output = Path(args.output) if args.output else Path(
        f"outputs/drifting_{args.method}_smoke.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
