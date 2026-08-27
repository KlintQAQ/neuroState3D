from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom


ROOT = Path(__file__).resolve().parents[1]
MODALITIES = ("t1n", "t1c", "t2w", "t2f")
REQUIRED_SUFFIXES = (*MODALITIES, "seg")


@dataclass(frozen=True)
class SubjectCase:
    dataset: str
    subject_id: str
    subject_dir: Path
    files: dict[str, Path]


def find_subjects(raw_root: Path) -> list[SubjectCase]:
    subjects: list[SubjectCase] = []
    if not raw_root.exists():
        return subjects
    for dataset_dir in sorted(item for item in raw_root.iterdir() if item.is_dir()):
        for subject_dir in sorted(dataset_dir.rglob("BraTS-*")):
            if not subject_dir.is_dir():
                continue
            files: dict[str, Path] = {}
            for suffix in REQUIRED_SUFFIXES:
                matches = sorted(subject_dir.glob(f"*-{suffix}.nii.gz"))
                if matches:
                    files[suffix] = matches[0]
            if all(suffix in files for suffix in REQUIRED_SUFFIXES):
                subjects.append(
                    SubjectCase(
                        dataset=dataset_dir.name,
                        subject_id=subject_dir.name,
                        subject_dir=subject_dir,
                        files=files,
                    )
                )
    return subjects


def foreground_bbox(arrays: list[np.ndarray], margin: int) -> tuple[np.ndarray, np.ndarray]:
    mask = np.zeros(arrays[0].shape, dtype=bool)
    for array in arrays:
        mask |= np.isfinite(array) & (array != 0)
    coords = np.argwhere(mask)
    if coords.size == 0:
        start = np.zeros(3, dtype=int)
        end = np.asarray(arrays[0].shape, dtype=int)
    else:
        start = np.maximum(coords.min(axis=0) - margin, 0)
        end = np.minimum(coords.max(axis=0) + margin + 1, np.asarray(arrays[0].shape))
    return start, end


def crop_and_pad_to_cube(array: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    cropped = array[start[0] : end[0], start[1] : end[1], start[2] : end[2]]
    side = int(max(cropped.shape))
    output = np.zeros((side, side, side), dtype=cropped.dtype)
    offsets = [(side - size) // 2 for size in cropped.shape]
    output[
        offsets[0] : offsets[0] + cropped.shape[0],
        offsets[1] : offsets[1] + cropped.shape[1],
        offsets[2] : offsets[2] + cropped.shape[2],
    ] = cropped
    return output


def resize_array(array: np.ndarray, target_shape: tuple[int, int, int], order: int) -> np.ndarray:
    factors = [target / source for target, source in zip(target_shape, array.shape)]
    return zoom(array, zoom=factors, order=order)


def normalize_image(array: np.ndarray) -> np.ndarray:
    foreground = array[array != 0]
    if foreground.size == 0:
        return array.astype(np.float32)
    mean = float(foreground.mean())
    std = float(foreground.std())
    if std < 1e-6:
        std = 1.0
    normalized = (array.astype(np.float32) - mean) / std
    normalized[array == 0] = 0
    normalized = np.clip(normalized, -5.0, 5.0) / 5.0
    return normalized.astype(np.float32)


def load_nifti(path: Path) -> np.ndarray:
    image = nib.load(str(path))
    return np.asarray(image.get_fdata(dtype=np.float32))


def process_subject(
    case: SubjectCase,
    output_root: Path,
    target_shape: tuple[int, int, int],
    crop_margin: int,
    force: bool,
) -> dict[str, Any]:
    output_dir = output_root / case.dataset / case.subject_id
    output_dir.mkdir(parents=True, exist_ok=True)
    multimodal_path = output_dir / "multimodal_4ch.npy"
    seg_path = output_dir / "seg.npy"
    metadata_path = output_dir / "metadata.json"

    if not force and multimodal_path.exists() and seg_path.exists() and metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            metadata = {}
        return {
            "dataset": case.dataset,
            "subject_id": case.subject_id,
            "status": "processed",
            "original_shape": "x".join(str(item) for item in metadata.get("original_shape", [])),
            "target_shape": "x".join(str(item) for item in metadata.get("target_shape", target_shape)),
            "multimodal_path": str(multimodal_path),
            "seg_path": str(seg_path),
        }

    image_arrays = [load_nifti(case.files[modality]) for modality in MODALITIES]
    seg_array = load_nifti(case.files["seg"])
    original_shape = list(image_arrays[0].shape)
    if any(list(array.shape) != original_shape for array in image_arrays):
        raise ValueError(f"Modality shape mismatch for {case.subject_id}")
    if list(seg_array.shape) != original_shape:
        raise ValueError(f"Segmentation shape mismatch for {case.subject_id}")

    start, end = foreground_bbox(image_arrays, crop_margin)
    processed_modalities: list[np.ndarray] = []
    for array in image_arrays:
        cropped = crop_and_pad_to_cube(array, start, end)
        resized = resize_array(cropped, target_shape, order=1)
        processed_modalities.append(normalize_image(resized))
    cropped_seg = crop_and_pad_to_cube(seg_array.astype(np.int16), start, end)
    resized_seg = resize_array(cropped_seg, target_shape, order=0).astype(np.uint8)

    stacked = np.stack(processed_modalities, axis=0).astype(np.float16)
    np.save(multimodal_path, stacked)
    np.save(seg_path, resized_seg)
    metadata = {
        "dataset": case.dataset,
        "subject_id": case.subject_id,
        "modalities": list(MODALITIES),
        "input_files": {key: str(value) for key, value in case.files.items()},
        "original_shape": original_shape,
        "target_shape": list(target_shape),
        "crop_start": start.tolist(),
        "crop_end": end.tolist(),
        "output_multimodal": str(multimodal_path),
        "output_seg": str(seg_path),
        "dtype": "float16",
        "seg_dtype": "uint8",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {
        "dataset": case.dataset,
        "subject_id": case.subject_id,
        "status": "processed",
        "original_shape": "x".join(str(item) for item in original_shape),
        "target_shape": "x".join(str(item) for item in target_shape),
        "multimodal_path": str(multimodal_path),
        "seg_path": str(seg_path),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare BraTS NIfTI data as model-ready 4-channel arrays.")
    parser.add_argument("--raw-root", default=str(ROOT / "data" / "raw" / "BraTS2023_HF"))
    parser.add_argument("--output-root", default=str(ROOT / "data" / "model_ready" / "BraTS2023_HF_128"))
    parser.add_argument("--manifest-dir", default=str(ROOT / "data" / "manifests" / "BraTS2023_HF"))
    parser.add_argument("--target-shape", nargs=3, type=int, default=[128, 128, 128])
    parser.add_argument("--crop-margin", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess subjects even when model-ready outputs already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_root = Path(args.raw_root)
    output_root = Path(args.output_root)
    manifest_dir = Path(args.manifest_dir)
    target_shape = tuple(int(item) for item in args.target_shape)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    subjects = find_subjects(raw_root)
    audit_rows = [
        {
            "dataset": case.dataset,
            "subject_id": case.subject_id,
            **{f"{suffix}_path": str(case.files[suffix]) for suffix in REQUIRED_SUFFIXES},
        }
        for case in subjects
    ]
    write_csv(manifest_dir / "brats_model_ready_audit.csv", audit_rows)
    print(
        json.dumps(
            {
                "status": "AUDIT_DONE",
                "raw_root": str(raw_root),
                "complete_subjects": len(subjects),
                "output_root": str(output_root),
                "target_shape": list(target_shape),
            },
            indent=2,
        ),
        flush=True,
    )

    processed: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max(args.workers, 1)) as executor:
        futures = {
            executor.submit(
                process_subject,
                case,
                output_root,
                target_shape,
                args.crop_margin,
                args.force,
            ): case
            for case in subjects
        }
        for index, future in enumerate(as_completed(futures), start=1):
            case = futures[future]
            try:
                row = future.result()
                processed.append(row)
            except Exception as exc:  # noqa: BLE001 - record per subject.
                failed.append(
                    {
                        "dataset": case.dataset,
                        "subject_id": case.subject_id,
                        "status": "failed",
                        "error": type(exc).__name__ + ": " + str(exc),
                    }
                )
            if index == 1 or index % 25 == 0 or index == len(subjects):
                print(f"prepare_progress [{index}/{len(subjects)}]", flush=True)

    write_csv(manifest_dir / "brats_model_ready_processed.csv", processed)
    write_csv(manifest_dir / "brats_model_ready_failed.csv", failed)
    summary = {
        "status": "PREPARE_DONE",
        "complete_subjects_found": len(subjects),
        "processed": len(processed),
        "failed": len(failed),
        "output_root": str(output_root),
        "processed_csv": str(manifest_dir / "brats_model_ready_processed.csv"),
        "failed_csv": str(manifest_dir / "brats_model_ready_failed.csv"),
    }
    (manifest_dir / "brats_model_ready_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    sys.exit(main())
