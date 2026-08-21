from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.hcp_s1200_audit import data_root, ensure_safe_data_root, manifests_dir
from scripts.hcp_s1200_download import DOWNLOAD_FIELDS, load_config


QC_FIELDS = [
    "subject_id",
    "local_complete",
    "total_files",
    "present_files",
    "size_matched_files",
    "t1_shape",
    "t2_shape",
    "dwi_shape",
    "dwi_volumes",
    "bval_count",
    "bvec_count",
    "dwi_gradients_match",
    "rest_runs_present",
    "rest_shapes",
    "rest_4d",
    "optional_files_present",
    "source_qc_pass",
    "failure_reasons",
]

REST_RUNS = ("REST1_LR", "REST1_RL", "REST2_LR", "REST2_RL")


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [field for field in DOWNLOAD_FIELDS if field not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Download manifest is missing required fields: {missing}")
        return list(reader)


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=QC_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def shape_text(path: Path) -> tuple[str, tuple[int, ...]]:
    image = nib.load(str(path))
    shape = tuple(int(item) for item in image.shape)
    return "x".join(str(item) for item in shape), shape


def bval_count(path: Path) -> int:
    values = np.loadtxt(path, dtype=float)
    return int(np.ravel(values).shape[0])


def bvec_count(path: Path) -> int:
    values = np.loadtxt(path, dtype=float)
    if values.ndim == 1:
        return int(values.shape[0])
    if values.shape[0] == 3:
        return int(values.shape[1])
    if values.shape[1] == 3:
        return int(values.shape[0])
    return int(max(values.shape))


def row_size_match(row: dict[str, str]) -> bool:
    path = Path(row["local_path"])
    expected = int(row["file_size"])
    return path.exists() and path.is_file() and path.stat().st_size == expected


def find_rest_run(row: dict[str, str]) -> str:
    text = row["remote_path"] + " " + row["local_path"]
    for run in REST_RUNS:
        if run in text:
            return run
    return "UNKNOWN"


def qc_subject(subject_id: str, rows: list[dict[str, str]]) -> dict[str, str]:
    present = [row for row in rows if Path(row["local_path"]).exists()]
    size_matched = [row for row in rows if row_size_match(row)]
    failures: list[str] = []
    local_complete = len(size_matched) == len(rows)
    if not local_complete:
        failures.append("incomplete_download")

    by_modality: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_modality[row["source_modality"]].append(row)

    output = {
        "subject_id": subject_id,
        "local_complete": str(local_complete).lower(),
        "total_files": str(len(rows)),
        "present_files": str(len(present)),
        "size_matched_files": str(len(size_matched)),
        "t1_shape": "",
        "t2_shape": "",
        "dwi_shape": "",
        "dwi_volumes": "0",
        "bval_count": "0",
        "bvec_count": "0",
        "dwi_gradients_match": "false",
        "rest_runs_present": "0",
        "rest_shapes": "",
        "rest_4d": "false",
        "optional_files_present": str(len([row for row in by_modality["optional_source"] if Path(row["local_path"]).exists()])),
        "source_qc_pass": "false",
        "failure_reasons": "",
    }

    if not local_complete:
        output["failure_reasons"] = ";".join(failures)
        return output

    try:
        t1_path = Path(by_modality["t1"][0]["local_path"])
        t2_path = Path(by_modality["t2"][0]["local_path"])
        dwi_path = Path(by_modality["dwi"][0]["local_path"])
        bval_path = Path(by_modality["bval"][0]["local_path"])
        bvec_path = Path(by_modality["bvec"][0]["local_path"])
    except IndexError:
        failures.append("missing_required_manifest_entry")
        output["failure_reasons"] = ";".join(failures)
        return output

    try:
        output["t1_shape"], t1_shape = shape_text(t1_path)
        output["t2_shape"], t2_shape = shape_text(t2_path)
        output["dwi_shape"], dwi_shape = shape_text(dwi_path)
        if len(t1_shape) != 3:
            failures.append("t1_not_3d")
        if len(t2_shape) != 3:
            failures.append("t2_not_3d")
        if len(dwi_shape) != 4:
            failures.append("dwi_not_4d")
            dwi_volumes = 0
        else:
            dwi_volumes = int(dwi_shape[3])
        output["dwi_volumes"] = str(dwi_volumes)

        bvals = bval_count(bval_path)
        bvecs = bvec_count(bvec_path)
        output["bval_count"] = str(bvals)
        output["bvec_count"] = str(bvecs)
        gradients_match = dwi_volumes > 0 and bvals == dwi_volumes and bvecs == dwi_volumes
        output["dwi_gradients_match"] = str(gradients_match).lower()
        if not gradients_match:
            failures.append("dwi_bval_bvec_count_mismatch")

        rest_rows = by_modality["rfmri"]
        rest_by_run = {find_rest_run(row): row for row in rest_rows}
        rest_shapes: list[str] = []
        rest_4d = True
        for run in REST_RUNS:
            row = rest_by_run.get(run)
            if row is None:
                rest_4d = False
                failures.append(f"missing_{run}")
                continue
            shape_str, shape = shape_text(Path(row["local_path"]))
            rest_shapes.append(f"{run}:{shape_str}")
            if len(shape) != 4:
                rest_4d = False
                failures.append(f"{run}_not_4d")
        output["rest_runs_present"] = str(len([run for run in REST_RUNS if run in rest_by_run]))
        output["rest_shapes"] = "|".join(rest_shapes)
        output["rest_4d"] = str(rest_4d).lower()
    except Exception as exc:  # noqa: BLE001 - record the failing source file check.
        failures.append(type(exc).__name__ + ":" + str(exc))

    output["source_qc_pass"] = str(not failures).lower()
    output["failure_reasons"] = ";".join(failures)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QC downloaded HCP S1200 source files.")
    parser.add_argument("--config", default="configs/hcp_s1200.yaml")
    parser.add_argument("--manifest", default="")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--subjects", nargs="*", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    root = data_root(config)
    ensure_safe_data_root(root, config)
    mdir = manifests_dir(config)
    if args.manifest:
        manifest_path = Path(args.manifest)
    elif args.run_name:
        manifest_path = (
            mdir
            / str(config["downloads"].get("runs_dir", "downloads"))
            / args.run_name
            / config["downloads"]["manifest_csv"]
        )
    else:
        manifest_path = mdir / config["downloads"]["manifest_csv"]

    rows = read_manifest(manifest_path)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    requested = set(args.subjects)
    for row in rows:
        if requested and row["subject_id"] not in requested:
            continue
        grouped[row["subject_id"]].append(row)

    qc_rows = [qc_subject(subject_id, grouped[subject_id]) for subject_id in sorted(grouped)]
    qc_dir = root / config["layout"]["qc"] / "source_qc"
    run_label = args.run_name or manifest_path.parent.name or "default"
    csv_path = qc_dir / f"{run_label}_source_qc.csv"
    json_path = qc_dir / f"{run_label}_source_qc_summary.json"
    write_csv(csv_path, qc_rows)
    summary = {
        "status": "SOURCE_QC_DONE",
        "run_name": run_label,
        "subjects": len(qc_rows),
        "source_qc_pass": sum(row["source_qc_pass"] == "true" for row in qc_rows),
        "local_complete": sum(row["local_complete"] == "true" for row in qc_rows),
        "manifest": str(manifest_path),
        "qc_csv": str(csv_path),
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
