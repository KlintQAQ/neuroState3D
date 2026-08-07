from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preprocessing.normalization import (
    BRAINMVP_PREPROCESSING_AUDIT,
    preprocess_nifti_for_brainmvp,
)
from preprocessing.spatial_utils import compare_nifti_space, inspect_nifti


def load_manifest_subject(
    manifest_path: Path,
    subject_id: str | None,
) -> tuple[dict[str, Any], Path]:
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    subjects = manifest.get("subjects", [])
    if not subjects:
        raise ValueError("Manifest has no subjects. HCP USER AUTHENTICATION REQUIRED.")
    if subject_id is None:
        subject = subjects[0]
    else:
        subject = next(
            (item for item in subjects if str(item.get("subject_id")) == subject_id),
            None,
        )
        if subject is None:
            raise KeyError(f"Subject {subject_id} not found in manifest.")
    return subject, Path(manifest["data_root"])


def subject_paths(args: argparse.Namespace) -> tuple[str, Path, Path]:
    if args.t1 and args.t2:
        return args.subject_id or "manual_subject", Path(args.t1), Path(args.t2)
    if not args.manifest:
        raise ValueError("Provide --manifest or both --t1 and --t2.")
    subject, data_root = load_manifest_subject(Path(args.manifest), args.subject_id or None)
    modalities = subject.get("modalities", {})
    return (
        str(subject["subject_id"]),
        data_root / modalities["t1"],
        data_root / modalities["t2"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect one HCP T1/T2 subject.")
    parser.add_argument("--manifest", default="", type=str)
    parser.add_argument("--subject-id", default="", type=str)
    parser.add_argument("--t1", default="", type=str)
    parser.add_argument("--t2", default="", type=str)
    parser.add_argument("--roi-size", default=96, type=int)
    parser.add_argument("--preprocess", action="store_true")
    parser.add_argument("--output", default="", type=str)
    args = parser.parse_args()

    if args.manifest and not Path(args.manifest).exists():
        report = {
            "status": "PENDING_USER_AUTHENTICATION",
            "reason": f"Manifest not found: {args.manifest}",
            "final_verdict": "LOCAL REAL HCP T1/T2 PIPELINE: PENDING USER AUTHENTICATION",
            "BRAINMVP_PREPROCESSING_AUDIT": BRAINMVP_PREPROCESSING_AUDIT,
        }
        print(json.dumps(report, indent=2))
        if args.output:
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return

    try:
        subject_id, t1_path, t2_path = subject_paths(args)
    except ValueError as exc:
        report = {
            "status": "PENDING_USER_AUTHENTICATION",
            "reason": str(exc),
            "BRAINMVP_PREPROCESSING_AUDIT": BRAINMVP_PREPROCESSING_AUDIT,
        }
        print(json.dumps(report, indent=2))
        if args.output:
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return

    t1_audit = inspect_nifti(t1_path)
    t2_audit = inspect_nifti(t2_path)
    spatial = compare_nifti_space(t1_audit, t2_audit)
    report: dict[str, Any] = {
        "subject_id": subject_id,
        "t1": t1_audit.__dict__,
        "t2": t2_audit.__dict__,
        "spatial_consistency": spatial,
        "BRAINMVP_PREPROCESSING_AUDIT": BRAINMVP_PREPROCESSING_AUDIT,
    }

    if args.preprocess:
        t1_tensor, t1_preprocess = preprocess_nifti_for_brainmvp(t1_path, args.roi_size)
        t2_tensor, t2_preprocess = preprocess_nifti_for_brainmvp(t2_path, args.roi_size)
        report["preprocessing"] = {
            "roi_size": args.roi_size,
            "t1": t1_preprocess,
            "t2": t2_preprocess,
            "t1_tensor_shape": list(t1_tensor.shape),
            "t2_tensor_shape": list(t2_tensor.shape),
        }

    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
