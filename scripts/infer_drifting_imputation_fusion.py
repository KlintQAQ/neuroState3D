from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.brats_fusion_dataset import BraTSFusionDataset  # noqa: E402
from scripts.train_drifting_imputation_fusion import build_model, to_device  # noqa: E402
from utils.brats_metrics import dice_scores  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Infer missing MRI modalities with Drifting, then run fusion."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--spatial-size", type=int, default=96)
    parser.add_argument("--max-subjects", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument(
        "--observed-modalities",
        nargs="+",
        default=("t1n", "t2w", "t2f"),
        help="Available MRI modalities; default evaluates missing t1c.",
    )
    parser.add_argument("--predictions", default=None)
    parser.add_argument("--seed", type=int, default=46)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint = torch.load(args.model, map_location="cpu")
    saved_args = dict(checkpoint["args"])
    # The complete encoder is already stored in state_dict; avoid reloading the
    # original external BrainMVP checkpoint during inference reconstruction.
    saved_args["checkpoint"] = ""
    model = build_model(argparse.Namespace(**saved_args))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(args.device).eval()

    dataset = BraTSFusionDataset(
        args.manifest,
        spatial_size=args.spatial_size,
        max_subjects=args.max_subjects,
        modality_mask_mode="fixed",
        fixed_modalities=args.observed_modalities,
        seed=args.seed,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0)
    reports = []
    saved_predictions = []
    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch, args.device)
            inference_images = batch["image"].clone()
            inference_images[batch["modality_mask"] <= 0] = 0
            output = model(
                inference_images,
                batch["modality_mask"],
                batch["modality_state"],
                n_samples=args.samples or saved_args["samples"],
                seed=args.seed,
            )
            missing_confidence = output["imputation_confidence"][
                batch["modality_mask"] <= 0
            ]
            reports.append(
                {
                    "subject_id": list(batch["subject_id"]),
                    "dice": dice_scores(output["logits"], batch["target"]),
                    "generated_pairs": output["missing_pair_count"],
                    "mean_generated_confidence": (
                        float(missing_confidence.mean().cpu())
                        if missing_confidence.numel()
                        else None
                    ),
                }
            )
            if args.predictions:
                saved_predictions.append(
                    {
                        "subject_id": list(batch["subject_id"]),
                        "logits": output["logits"].cpu(),
                        "imputed_images": output["imputed_images"].cpu(),
                        "confidence": output["imputation_confidence"].cpu(),
                        "confidence_map": output[
                            "imputation_confidence_map"
                        ].cpu(),
                    }
                )
    if args.predictions:
        prediction_path = Path(args.predictions)
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(saved_predictions, prediction_path)
    print(json.dumps({"subjects": reports}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
