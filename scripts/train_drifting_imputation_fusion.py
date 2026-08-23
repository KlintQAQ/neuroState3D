from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.brats_fusion_dataset import BraTSFusionDataset  # noqa: E402
from models.drifting_imputation_fusion import (  # noqa: E402
    DriftingImputationFusion,
    DriftingImputationFusionConfig,
)
from models.evidence_fusion import EvidenceFusionConfig, EvidenceReliableFusion  # noqa: E402
from models.generators import (  # noqa: E402
    DriftingConfig,
    PosteriorBackboneConfig,
    build_posterior_generator,
)
from utils.brats_metrics import dice_scores, segmentation_loss  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Drifting MRI imputation followed by evidence fusion."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", default="pretrained/BrainMVP_uniformer.pt")
    parser.add_argument("--output", default="outputs/drifting_imputation_fusion.pt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--spatial-size", type=int, default=96)
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--fusion-warmup-epochs", type=int, default=1)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--feature-stage", default="stage4")
    parser.add_argument("--feature-channels", type=int, default=512)
    parser.add_argument("--fusion-hidden-channels", type=int, default=64)
    parser.add_argument("--drifting-hidden-channels", type=int, default=96)
    parser.add_argument("--drifting-depth", type=int, default=4)
    parser.add_argument("--decoder-hidden-channels", type=int, default=64)
    parser.add_argument("--decoder-highres-channels", type=int, default=16)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--generator-weight", type=float, default=1.0)
    parser.add_argument("--segmentation-weight", type=float, default=1.0)
    parser.add_argument("--uncertainty-nll-weight", type=float, default=0.05)
    parser.add_argument("--uncertainty-calibration-weight", type=float, default=0.05)
    parser.add_argument(
        "--freeze-fusion-during-generator",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Freeze the warmed-up fusion model for a fair generator comparison.",
    )
    parser.add_argument(
        "--mask-mode",
        choices=("fixed", "single_random", "random_nonempty"),
        default="random_nonempty",
    )
    parser.add_argument(
        "--fixed-modalities",
        nargs="+",
        default=("t1n", "t2w", "t2f"),
        help="Observed modalities for --mask-mode fixed; default holds out t1c.",
    )
    parser.add_argument("--seed", type=int, default=46)
    return parser.parse_args()


def build_model(args: argparse.Namespace) -> DriftingImputationFusion:
    fusion_config = EvidenceFusionConfig(
        checkpoint_path=args.checkpoint or None,
        encoder_freeze="freeze_all",
        feature_stage=args.feature_stage,
        feature_channels=args.feature_channels,
        hidden_channels=args.fusion_hidden_channels,
    )
    fusion = EvidenceReliableFusion(fusion_config)
    backbone = PosteriorBackboneConfig(
        latent_channels=args.feature_channels,
        condition_channels=args.feature_channels,
        hidden_channels=args.drifting_hidden_channels,
        depth=args.drifting_depth,
        num_modalities=fusion.num_modalities,
        num_targets=fusion.num_modalities,
    )
    drifting = build_posterior_generator(
        "drifting",
        backbone,
        drifting=DriftingConfig(
            samples_per_condition=args.samples,
            feature_grid=4,
        ),
    )
    return DriftingImputationFusion(
        fusion,
        drifting,
        DriftingImputationFusionConfig(
            samples_per_target=args.samples,
            decoder_hidden_channels=args.decoder_hidden_channels,
            decoder_highres_channels=getattr(args, "decoder_highres_channels", 16),
            uncertainty_nll_weight=getattr(args, "uncertainty_nll_weight", 0.05),
            uncertainty_calibration_weight=getattr(
                args, "uncertainty_calibration_weight", 0.05
            ),
        ),
    )


def to_device(batch: dict[str, object], device: str) -> dict[str, object]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def set_trainable(module: torch.nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = enabled


def trainable_parameters(module: torch.nn.Module) -> list[torch.nn.Parameter]:
    return [parameter for parameter in module.parameters() if parameter.requires_grad]


def mean_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    return {
        key: sum(row[key] for row in rows) / len(rows)
        for key in rows[0]
    }


def warmup_fusion_epoch(
    model: DriftingImputationFusion,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> dict[str, float]:
    model.train()
    rows = []
    for batch in loader:
        batch = to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        full_mask = torch.ones_like(batch["modality_mask"])
        full_state = torch.zeros_like(batch["modality_state"])
        full = model.fusion(batch["image"], full_mask, full_state)
        missing = model.fusion(
            batch["image"],
            batch["modality_mask"],
            batch["modality_state"],
        )
        full_loss = segmentation_loss(full["logits"], batch["target"])
        missing_loss = segmentation_loss(missing["logits"], batch["target"])
        loss = 0.5 * (full_loss + missing_loss)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite fusion warmup loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_parameters(model.fusion), 1.0)
        optimizer.step()
        rows.append(
            {
                "loss": float(loss.detach().cpu()),
                "full_loss": float(full_loss.detach().cpu()),
                "missing_loss": float(missing_loss.detach().cpu()),
            }
        )
    return mean_rows(rows)


def train_generator_epoch(
    model: DriftingImputationFusion,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    epoch: int,
) -> dict[str, float]:
    model.train()
    if args.freeze_fusion_during_generator:
        model.fusion.eval()
    parameters = trainable_parameters(model)
    rows = []
    for batch in loader:
        batch = to_device(batch, args.device)
        if (
            args.freeze_fusion_during_generator
            and torch.all(batch["modality_mask"] > 0)
        ):
            # With frozen fusion and no missing slot, the generator is not in
            # the graph and there is no meaningful generator update.
            continue
        optimizer.zero_grad(set_to_none=True)
        output = model(
            batch["image"],
            batch["modality_mask"],
            batch["modality_state"],
            compute_generator_loss=True,
            seed=args.seed + epoch,
        )
        segmentation = segmentation_loss(output["logits"], batch["target"])
        loss = (
            args.segmentation_weight * segmentation
            + args.generator_weight * output["generator_loss"]
        )
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite generator training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        rows.append(
            {
                "loss": float(loss.detach().cpu()),
                "segmentation_loss": float(segmentation.detach().cpu()),
                "drifting_loss": float(output["drifting_loss"].detach().cpu()),
                "reconstruction_loss": float(
                    output["reconstruction_loss"].detach().cpu()
                ),
                "uncertainty_nll_loss": float(
                    output["uncertainty_nll_loss"].detach().cpu()
                ),
                "uncertainty_calibration_loss": float(
                    output["uncertainty_calibration_loss"].detach().cpu()
                ),
            }
        )
    if not rows:
        raise RuntimeError(
            "Generator epoch contained no missing-modality batches; use "
            "--mask-mode fixed or increase the dataset size."
        )
    return mean_rows(rows)


@torch.no_grad()
def evaluate(
    model: DriftingImputationFusion,
    loader: DataLoader,
    args: argparse.Namespace,
) -> dict[str, object]:
    model.eval()
    cases: dict[str, list[dict[str, float]]] = {
        "full_real": [],
        "missing": [],
        "drifting_filled": [],
    }
    image_rows = []
    confidence_values = []
    error_values = []
    generated_pairs = 0
    for batch in loader:
        batch = to_device(batch, args.device)
        images = batch["image"]
        mask = batch["modality_mask"]
        state = batch["modality_state"]
        full_mask = torch.ones_like(mask)
        full_state = torch.zeros_like(state)
        full = model.fusion(images, full_mask, full_state)
        missing_output = model.fusion(images, mask, state)
        inference_images = images.clone()
        inference_images[mask <= 0] = 0
        filled = model(
            inference_images,
            mask,
            state,
            n_samples=args.samples,
            seed=args.seed,
        )
        for name, output in (
            ("full_real", full),
            ("missing", missing_output),
            ("drifting_filled", filled),
        ):
            cases[name].append(dice_scores(output["logits"], batch["target"]))
        missing_slots = mask <= 0
        if missing_slots.any():
            absolute_error = (
                filled["imputed_images"][missing_slots] - images[missing_slots]
            ).abs().flatten(1).mean(dim=1)
            confidence = filled["imputation_confidence"][missing_slots]
            image_rows.append(
                {
                    "mae": float(absolute_error.mean().cpu()),
                    "confidence": float(confidence.mean().cpu()),
                }
            )
            confidence_values.append(confidence.cpu())
            error_values.append(absolute_error.cpu())
        generated_pairs += int(filled["missing_pair_count"])
    aggregate = {name: mean_rows(rows) for name, rows in cases.items()}
    aggregate["delta_filled_vs_missing"] = {
        key: aggregate["drifting_filled"][key] - aggregate["missing"][key]
        for key in aggregate["missing"]
    }
    correlation = None
    if confidence_values:
        confidence = torch.cat(confidence_values).float()
        error = torch.cat(error_values).float()
        if confidence.numel() > 1 and confidence.std() > 0 and error.std() > 0:
            correlation = float(torch.corrcoef(torch.stack([confidence, error]))[0, 1])
    aggregate["imputation"] = {
        **mean_rows(image_rows),
        "confidence_error_correlation": correlation,
        "generated_pairs": generated_pairs,
    }
    return aggregate


def main() -> int:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    if args.epochs < 1:
        raise ValueError("epochs must be at least 1")
    if args.fusion_warmup_epochs < 1 and args.freeze_fusion_during_generator:
        raise ValueError(
            "Frozen-fusion training requires at least one fusion warmup epoch"
        )
    torch.manual_seed(args.seed)
    if not 0 < args.val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")
    dataset = BraTSFusionDataset(
        args.manifest,
        spatial_size=args.spatial_size,
        max_subjects=args.max_subjects,
        modality_mask_mode=args.mask_mode,
        fixed_modalities=args.fixed_modalities if args.mask_mode == "fixed" else None,
        seed=args.seed,
    )
    if len(dataset) < 2:
        raise ValueError("At least two subjects are required for train/validation split")
    split_generator = torch.Generator().manual_seed(args.seed)
    shuffled = torch.randperm(len(dataset), generator=split_generator).tolist()
    val_count = max(1, int(round(len(dataset) * args.val_fraction)))
    val_count = min(val_count, len(dataset) - 1)
    val_indices = shuffled[:val_count]
    train_indices = shuffled[val_count:]
    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )
    model = build_model(args).to(args.device)
    warmup_history = []
    fusion_optimizer = torch.optim.AdamW(
        trainable_parameters(model.fusion), lr=args.lr
    )
    for epoch in range(args.fusion_warmup_epochs):
        dataset.set_epoch(epoch)
        warmup_history.append(
            warmup_fusion_epoch(
                model, train_loader, fusion_optimizer, args.device
            )
        )

    if args.freeze_fusion_during_generator:
        set_trainable(model.fusion, False)
    generator_optimizer = torch.optim.AdamW(trainable_parameters(model), lr=args.lr)
    generator_history = []
    for epoch in range(args.epochs):
        dataset.set_epoch(args.fusion_warmup_epochs + epoch)
        generator_history.append(
            train_generator_epoch(
                model,
                train_loader,
                generator_optimizer,
                args,
                epoch,
            )
        )
    dataset.set_epoch(100_000)
    evaluation = evaluate(model, val_loader, args)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "args": vars(args),
            "fusion_warmup_history": warmup_history,
            "generator_history": generator_history,
            "evaluation": evaluation,
            "split": {"train_indices": train_indices, "val_indices": val_indices},
        },
        output_path,
    )
    report = {
        "checkpoint": str(output_path),
        "train_subjects": len(train_indices),
        "val_subjects": len(val_indices),
        "fusion_warmup": warmup_history,
        "generator_training": generator_history,
        "evaluation": evaluation,
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
