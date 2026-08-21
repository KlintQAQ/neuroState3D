from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.hcp_s1200_audit import data_root, ensure_safe_data_root, manifests_dir


BATCH_PLAN_FIELDS = [
    "batch_id",
    "subject_count",
    "estimated_download_gb",
    "required_free_space_gb",
    "first_subject",
    "last_subject",
    "subject_file",
    "download_run_dir",
    "status",
]

BATCH_STATUS_FIELDS = [
    "batch_id",
    "download_status",
    "source_qc_status",
    "derivation_status",
    "registration_status",
    "model_ready_status",
    "raw_cleanup_status",
    "final_status",
]


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def read_subjects(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]


def read_sizes(path: Path) -> dict[str, float]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = csv.DictReader(handle)
        return {
            row["subject_id"]: float(row["estimated_download_size_gb"])
            for row in rows
            if row.get("strict_eligible") == "true"
        }


def write_lines(path: Path, values: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(values) + ("\n" if values else ""), encoding="utf-8")


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def disk_free_gb(path: Path) -> float:
    usage = shutil.disk_usage(path.anchor or str(path))
    return usage.free / (1024**3)


def build_batches(
    subjects: list[str],
    sizes_gb: dict[str, float],
    batch_size: int,
    max_batch_download_gb: float,
) -> list[list[str]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    batches: list[list[str]] = []
    current: list[str] = []
    current_gb = 0.0
    for subject_id in subjects:
        subject_gb = sizes_gb.get(subject_id)
        if subject_gb is None:
            raise KeyError(f"Subject {subject_id} is missing from audit CSV size table.")
        size_full = len(current) >= batch_size
        size_over = max_batch_download_gb > 0 and current and current_gb + subject_gb > max_batch_download_gb
        if size_full or size_over:
            batches.append(current)
            current = []
            current_gb = 0.0
        current.append(subject_id)
        current_gb += subject_gb
    if current:
        batches.append(current)
    return batches


def data_layout_readme_text(config: dict[str, Any]) -> str:
    root = data_root(config)
    return f"""# NeuroState-3D Data Layout

Root:

```text
{root}
```

Directory contract:

```text
raw/HCP_S1200/<SUBJECT_ID>/          temporary HCP source files for the active batch
derived/HCP_S1200/<SUBJECT_ID>/      FA/MD/ALFF native or intermediate derivatives
processed/HCP_S1200/<SUBJECT_ID>/    registered five-volume NIfTI outputs
model_ready/HCP_S1200/<SUBJECT_ID>/  final tensors used by Fusion
qc/<SUBJECT_ID>/                     automatic and visual QC artifacts
manifests/                           audit, batch, download, status, and split files
```

Raw cleanup rule:

```text
Delete raw/HCP_S1200/<SUBJECT_ID> only after:
source QC PASS
FA/MD PASS
ALFF PASS
registration PASS
model_ready PASS
```

Fusion must read only:

```text
model_ready/HCP_S1200/<SUBJECT_ID>/
```

Expected model-ready files:

```text
T1.npy
T2.npy
FA.npy
MD.npy
ALFF.npy
multimodal_5ch.npy
metadata.json
```

The raw directory is a cache, not the scientific dataset. The model-ready
directory is the stable dataset for future Fusion experiments.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create clear HCP S1200 batch manifests for raw-cache processing."
    )
    parser.add_argument("--config", default="configs/hcp_s1200.yaml")
    parser.add_argument("--audit-csv", default="")
    parser.add_argument("--cohort-file", default="")
    parser.add_argument("--batch-size-subjects", type=int, default=0)
    parser.add_argument(
        "--max-batch-download-gb",
        type=float,
        default=0.0,
        help="Optional cap on estimated raw download size per batch.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    root = data_root(config)
    ensure_safe_data_root(root, config)
    mdir = manifests_dir(config)
    batch_cfg = config.get("batches", {})
    batch_dir = mdir / str(batch_cfg.get("output_dir", "batches"))
    batch_dir.mkdir(parents=True, exist_ok=True)

    audit_csv = Path(args.audit_csv) if args.audit_csv else mdir / config["audit"]["output_all_subjects"]
    cohort_file = (
        Path(args.cohort_file)
        if args.cohort_file
        else mdir / config["audit"]["output_strict_subjects"]
    )
    batch_size = args.batch_size_subjects or int(batch_cfg.get("batch_size_subjects", 30))
    sizes_gb = read_sizes(audit_csv)
    subjects = read_subjects(cohort_file)
    batches = build_batches(subjects, sizes_gb, batch_size, args.max_batch_download_gb)

    multiplier = float(config["safety"].get("required_free_space_multiplier", 1.5))
    runs_dir = mdir / str(config["downloads"].get("runs_dir", "downloads"))
    plan_rows: list[dict[str, Any]] = []
    status_rows: list[dict[str, Any]] = []
    for index, batch_subjects in enumerate(batches, 1):
        batch_id = f"batch_{index:04d}"
        subject_file = batch_dir / f"{batch_id}_subjects.txt"
        estimated_gb = sum(sizes_gb[subject_id] for subject_id in batch_subjects)
        required_gb = estimated_gb * multiplier
        write_lines(subject_file, batch_subjects)
        metadata = {
            "batch_id": batch_id,
            "subject_count": len(batch_subjects),
            "estimated_download_gb": round(estimated_gb, 6),
            "required_free_space_gb": round(required_gb, 6),
            "subjects": batch_subjects,
            "raw_cleanup_policy": "raw is deleted only after model_ready_pass",
        }
        (batch_dir / f"{batch_id}_metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        plan_rows.append(
            {
                "batch_id": batch_id,
                "subject_count": len(batch_subjects),
                "estimated_download_gb": f"{estimated_gb:.6f}",
                "required_free_space_gb": f"{required_gb:.6f}",
                "first_subject": batch_subjects[0],
                "last_subject": batch_subjects[-1],
                "subject_file": str(subject_file),
                "download_run_dir": str(runs_dir / batch_id),
                "status": "pending",
            }
        )
        status_rows.append(
            {
                "batch_id": batch_id,
                "download_status": "pending",
                "source_qc_status": "pending",
                "derivation_status": "pending",
                "registration_status": "pending",
                "model_ready_status": "pending",
                "raw_cleanup_status": "blocked_until_model_ready_pass",
                "final_status": "pending",
            }
        )

    plan_csv = batch_dir / str(batch_cfg.get("plan_csv", "hcp_s1200_batch_plan.csv"))
    status_csv = batch_dir / str(batch_cfg.get("status_csv", "hcp_s1200_batch_status.csv"))
    write_csv(plan_csv, BATCH_PLAN_FIELDS, plan_rows)
    write_csv(status_csv, BATCH_STATUS_FIELDS, status_rows)
    readme_name = str(batch_cfg.get("data_layout_readme", "README_NEUROSTATE3D_DATA_LAYOUT.md"))
    (root / readme_name).write_text(data_layout_readme_text(config), encoding="utf-8")
    (batch_dir / "README.md").write_text(
        data_layout_readme_text(config)
        + "\nBatch download example:\n\n"
        + "```powershell\n"
        + "python scripts/hcp_s1200_download.py --config configs/hcp_s1200.yaml "
        + "--profile hcp --cohort-file E:/NeuroState3D_Data/manifests/batches/batch_0001_subjects.txt "
        + "--run-name batch_0001\n"
        + "```\n",
        encoding="utf-8",
    )

    first_required = float(plan_rows[0]["required_free_space_gb"]) if plan_rows else 0.0
    report = {
        "status": "BATCH_PLAN_READY",
        "subjects": len(subjects),
        "num_batches": len(batches),
        "batch_size_subjects": batch_size,
        "first_batch_required_free_space_gb": round(first_required, 6),
        "current_free_space_gb": round(disk_free_gb(root), 6),
        "batch_plan": str(plan_csv),
        "batch_status": str(status_csv),
        "data_layout_readme": str(root / readme_name),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
