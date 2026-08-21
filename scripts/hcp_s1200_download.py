from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.hcp_s1200_audit import (
    CSV_FIELDS,
    S3AuditError,
    S3Probe,
    data_root,
    disk_summary,
    ensure_safe_data_root,
    manifests_dir,
)


DOWNLOAD_FIELDS = [
    "subject_id",
    "source_modality",
    "remote_path",
    "local_path",
    "file_size",
    "download_status",
    "retry_count",
    "verification_status",
    "purpose",
]

SUBJECT_STATUS_FIELDS = [
    "subject_id",
    "structural_downloaded",
    "diffusion_downloaded",
    "rfmri_downloaded",
    "source_qc_pass",
    "FA_generated",
    "MD_generated",
    "ALFF_generated",
    "registration_pass",
    "model_ready_pass",
    "final_status",
]

REQUIRED_KEY_COLUMNS = {
    "t1": ("structural", "selected_t1_key", "T1w anatomical source"),
    "t2": ("structural", "selected_t2_key", "T2w anatomical source"),
    "dwi": ("diffusion", "selected_dwi_key", "preprocessed diffusion MRI"),
    "bval": ("diffusion", "selected_bval_key", "DWI b-values"),
    "bvec": ("diffusion", "selected_bvec_key", "DWI b-vectors"),
    "REST1_LR": ("rfmri", "selected_REST1_LR_key", "REST1 LR volumetric BOLD"),
    "REST1_RL": ("rfmri", "selected_REST1_RL_key", "REST1 RL volumetric BOLD"),
    "REST2_LR": ("rfmri", "selected_REST2_LR_key", "REST2 LR volumetric BOLD"),
    "REST2_RL": ("rfmri", "selected_REST2_RL_key", "REST2 RL volumetric BOLD"),
}


@dataclass(frozen=True)
class DownloadItem:
    subject_id: str
    source_modality: str
    remote_path: str
    local_path: Path
    file_size: int
    purpose: str


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def read_audit_csv(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [field for field in CSV_FIELDS if field not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Audit CSV is missing required fields: {missing}")
        return {row["subject_id"]: row for row in reader}


def read_subjects(path: Path, limit: int = 0) -> list[str]:
    subjects = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    return subjects[:limit] if limit else subjects


def relative_key_after_subject(key: str, release_prefix: str, subject_id: str) -> Path:
    prefix = f"{release_prefix.strip('/')}/{subject_id}/"
    if not key.startswith(prefix):
        raise ValueError(f"Remote key does not match subject prefix: {key}")
    return Path(key[len(prefix) :])


def build_items(
    subjects: list[str],
    audit_rows: dict[str, dict[str, str]],
    config: dict[str, Any],
) -> list[DownloadItem]:
    root = data_root(config)
    raw_root = root / config["layout"]["raw"]
    release_prefix = str(config["release_prefix"]).strip("/")
    items: list[DownloadItem] = []
    for subject_id in subjects:
        row = audit_rows.get(subject_id)
        if row is None:
            raise KeyError(f"Subject {subject_id} is not present in audit CSV.")
        for name, (group, column, purpose) in REQUIRED_KEY_COLUMNS.items():
            key = row.get(column, "")
            if not key:
                raise ValueError(f"Subject {subject_id} missing required audited key {column}.")
            local_path = raw_root / subject_id / relative_key_after_subject(
                key, release_prefix, subject_id
            )
            items.append(
                DownloadItem(
                    subject_id=subject_id,
                    source_modality=group if name.startswith("REST") else name,
                    remote_path=key,
                    local_path=local_path,
                    file_size=0,
                    purpose=purpose,
                )
            )
        optional_keys = [value for value in row.get("selected_optional_keys", "").split("|") if value]
        for key in optional_keys:
            local_path = raw_root / subject_id / relative_key_after_subject(
                key, release_prefix, subject_id
            )
            items.append(
                DownloadItem(
                    subject_id=subject_id,
                    source_modality="optional_source",
                    remote_path=key,
                    local_path=local_path,
                    file_size=0,
                    purpose="optional source/QC support file",
                )
            )
    return items


def sized_item(item: DownloadItem, probe: S3Probe, retry_limit: int) -> DownloadItem:
    last_error = ""
    for attempt in range(retry_limit + 1):
        try:
            result = probe.head(item.remote_path)
            if result.available:
                return DownloadItem(
                    subject_id=item.subject_id,
                    source_modality=item.source_modality,
                    remote_path=item.remote_path,
                    local_path=item.local_path,
                    file_size=result.size,
                    purpose=item.purpose,
                )
            last_error = result.error or "not_available"
        except Exception as exc:  # noqa: BLE001 - transient S3/network errors are retried.
            last_error = type(exc).__name__ + ": " + str(exc)
        if attempt < retry_limit:
            time.sleep(min(2**attempt, 10))
    raise S3AuditError(
        f"Remote object is inaccessible after {retry_limit + 1} size probes: "
        f"{item.remote_path}; last_error={last_error}"
    )


def fill_remote_sizes(
    items: list[DownloadItem],
    probe: S3Probe,
    workers: int = 16,
    retry_limit: int = 5,
) -> list[DownloadItem]:
    if workers <= 1:
        sized: list[DownloadItem] = []
        for index, item in enumerate(items, start=1):
            sized.append(sized_item(item, probe, retry_limit))
            if index == 1 or index % 25 == 0 or index == len(items):
                print(f"size_probe [{index}/{len(items)}]", flush=True)
        return sized

    sized: list[DownloadItem | None] = [None] * len(items)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(sized_item, item, probe, retry_limit): index
            for index, item in enumerate(items)
        }
        for done, future in enumerate(as_completed(futures), start=1):
            index = futures[future]
            sized[index] = future.result()
            if done == 1 or done % 25 == 0 or done == len(items):
                print(f"size_probe [{done}/{len(items)}]", flush=True)
    return [item for item in sized if item is not None]


def bytes_completed(item: DownloadItem) -> int:
    if item.local_path.exists() and item.local_path.stat().st_size == item.file_size:
        return item.file_size
    part = part_path(item.local_path)
    if part.exists():
        return min(part.stat().st_size, item.file_size)
    return 0


def part_path(path: Path) -> Path:
    return path.with_name(path.name + ".part")


def remaining_bytes(items: list[DownloadItem]) -> int:
    return sum(max(item.file_size - bytes_completed(item), 0) for item in items)


def ensure_space(items: list[DownloadItem], config: dict[str, Any]) -> dict[str, Any]:
    root = data_root(config)
    disk = disk_summary(root)
    multiplier = float(config["safety"].get("required_free_space_multiplier", 1.5))
    needed = remaining_bytes(items)
    required = int(needed * multiplier)
    ok = int(disk["free_space_bytes"]) >= required
    return {
        "remaining_download_bytes": needed,
        "remaining_download_gb": round(needed / (1024**3), 6),
        "required_free_space_bytes": required,
        "required_free_space_gb": round(required / (1024**3), 6),
        "disk": disk,
        "space_ok": ok,
    }


def stream_download(
    client: Any,
    bucket: str,
    item: DownloadItem,
    chunk_size: int,
) -> tuple[str, str]:
    if item.local_path.exists() and item.local_path.stat().st_size == item.file_size:
        return "skipped_existing", "size_match"
    item.local_path.parent.mkdir(parents=True, exist_ok=True)
    part = part_path(item.local_path)
    offset = part.stat().st_size if part.exists() else 0
    if offset > item.file_size:
        part.unlink()
        offset = 0
    request: dict[str, Any] = {"Bucket": bucket, "Key": item.remote_path}
    if offset > 0:
        request["Range"] = f"bytes={offset}-"
    response = client.get_object(**request)
    mode = "ab" if offset else "wb"
    with part.open(mode) as handle:
        body = response["Body"]
        for chunk in body.iter_chunks(chunk_size=chunk_size):
            if chunk:
                handle.write(chunk)
    if part.stat().st_size != item.file_size:
        return "partial", f"partial_size={part.stat().st_size}"
    os.replace(part, item.local_path)
    if item.local_path.stat().st_size != item.file_size:
        return "failed", "final_size_mismatch"
    return "downloaded", "size_match"


def download_with_retries(
    probe: S3Probe,
    item: DownloadItem,
    config: dict[str, Any],
) -> dict[str, str]:
    retry_limit = int(config["downloads"].get("retry_count", 3))
    chunk_size = int(config["downloads"].get("chunk_size_mb", 16)) * 1024 * 1024
    status = "failed"
    verification = "not_checked"
    attempt = 0
    for attempt in range(retry_limit + 1):
        try:
            status, verification = stream_download(
                probe.client, str(config["bucket"]), item, chunk_size
            )
            if status in {"downloaded", "skipped_existing"} and verification == "size_match":
                break
        except Exception as exc:  # noqa: BLE001 - recorded per item.
            status = "failed"
            verification = type(exc).__name__ + ": " + str(exc)
        time.sleep(min(2**attempt, 10))
    return {
        "subject_id": item.subject_id,
        "source_modality": item.source_modality,
        "remote_path": item.remote_path,
        "local_path": str(item.local_path),
        "file_size": str(item.file_size),
        "download_status": status,
        "retry_count": str(attempt),
        "verification_status": verification,
        "purpose": item.purpose,
    }


def download_subject(
    probe: S3Probe,
    subject_id: str,
    subject_items: list[DownloadItem],
    config: dict[str, Any],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    total = len(subject_items)
    for index, item in enumerate(subject_items, start=1):
        size_gb = item.file_size / (1024**3)
        completed_gb = bytes_completed(item) / (1024**3)
        resume_note = (
            f", resume_from={completed_gb:.3f}GB" if completed_gb > 0 else ""
        )
        print(
            f"{subject_id} [{index}/{total}] start {item.source_modality}: "
            f"{item.local_path.name} ({size_gb:.3f}GB{resume_note})",
            flush=True,
        )
        row = download_with_retries(probe, item, config)
        print(
            f"{subject_id} [{index}/{total}] {row['download_status']} "
            f"{item.source_modality}: {item.local_path.name}; "
            f"{row['verification_status']}; retries={row['retry_count']}",
            flush=True,
        )
        rows.append(row)
    return rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def group_by_subject(items: list[DownloadItem]) -> dict[str, list[DownloadItem]]:
    grouped: dict[str, list[DownloadItem]] = {}
    for item in items:
        grouped.setdefault(item.subject_id, []).append(item)
    return grouped


def item_to_manifest_row(item: DownloadItem, status: str = "planned") -> dict[str, str]:
    return {
        "subject_id": item.subject_id,
        "source_modality": item.source_modality,
        "remote_path": item.remote_path,
        "local_path": str(item.local_path),
        "file_size": str(item.file_size),
        "download_status": status,
        "retry_count": "0",
        "verification_status": "not_checked",
        "purpose": item.purpose,
    }


def subject_status(subject_id: str, rows: list[dict[str, str]]) -> dict[str, str]:
    by_group = {
        "structural": [row for row in rows if row["source_modality"] in {"t1", "t2"}],
        "diffusion": [row for row in rows if row["source_modality"] in {"dwi", "bval", "bvec"}],
        "rfmri": [row for row in rows if row["source_modality"] == "rfmri"],
    }

    def passed(group: str) -> str:
        group_rows = by_group[group]
        if not group_rows:
            return "false"
        return str(
            all(
                row["download_status"] in {"downloaded", "skipped_existing"}
                and row["verification_status"] == "size_match"
                for row in group_rows
            )
        ).lower()

    all_pass = all(passed(group) == "true" for group in ("structural", "diffusion", "rfmri"))
    return {
        "subject_id": subject_id,
        "structural_downloaded": passed("structural"),
        "diffusion_downloaded": passed("diffusion"),
        "rfmri_downloaded": passed("rfmri"),
        "source_qc_pass": "pending",
        "FA_generated": "pending",
        "MD_generated": "pending",
        "ALFF_generated": "pending",
        "registration_pass": "pending",
        "model_ready_pass": "pending",
        "final_status": "downloaded" if all_pass else "download_failed",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build and optionally execute a resumable HCP S1200 download plan. "
            "Defaults to dry-run; pass --execute to download."
        )
    )
    parser.add_argument("--config", default="configs/hcp_s1200.yaml")
    parser.add_argument("--audit-csv", default="")
    parser.add_argument("--cohort-file", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-parallel-downloads", type=int, default=0)
    parser.add_argument("--profile", default="", help="AWS profile name for HCP S3 access.")
    parser.add_argument("--region", default="", help="AWS region, defaults to config value.")
    parser.add_argument(
        "--run-name",
        default="",
        help=(
            "Optional manifest subdirectory name, e.g. batch_0001. "
            "Keeps batch download manifests separate."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.profile:
        config["aws_profile"] = args.profile
    if args.region:
        config["region"] = args.region
    root = data_root(config)
    ensure_safe_data_root(root, config)
    mdir = manifests_dir(config)
    mdir.mkdir(parents=True, exist_ok=True)
    audit_csv = Path(args.audit_csv) if args.audit_csv else mdir / config["audit"]["output_all_subjects"]
    cohort_file = (
        Path(args.cohort_file)
        if args.cohort_file
        else mdir / config["audit"]["output_strict_subjects"]
    )
    audit_rows = read_audit_csv(audit_csv)
    subjects = read_subjects(cohort_file, limit=args.limit)
    if not subjects:
        raise SystemExit(f"No subjects found in cohort file: {cohort_file}")

    try:
        probe = S3Probe(
            str(config["bucket"]),
            unsigned=bool(config.get("unsigned", False)),
            profile=str(config.get("aws_profile", "")),
            region=str(config.get("region", "us-east-1")),
        )
        size_workers = int(config["downloads"].get("size_probe_workers", 16))
        size_retries = int(config["downloads"].get("size_probe_retry_count", 5))
        items = fill_remote_sizes(
            build_items(subjects, audit_rows, config),
            probe,
            size_workers,
            size_retries,
        )
    except S3AuditError as exc:
        report = {
            "status": "PENDING_HCP_AWS_AUTHENTICATION",
            "reason": str(exc),
            "bucket": config["bucket"],
            "aws_profile": str(config.get("aws_profile", "")),
            "region": str(config.get("region", "us-east-1")),
            "download_started": False,
        }
        print(json.dumps(report, indent=2))
        raise SystemExit(2) from exc
    space = ensure_space(items, config)
    if not space["space_ok"]:
        raise SystemExit(
            "Insufficient E: drive space for planned download. "
            f"Need {space['required_free_space_gb']} GB free, have "
            f"{space['disk']['free_space_gb']} GB. Refusing to use C:."
        )

    run_name = args.run_name.strip()
    output_dir = mdir
    if run_name:
        output_dir = mdir / str(config["downloads"].get("runs_dir", "downloads")) / run_name
        output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / config["downloads"]["manifest_csv"]
    status_path = output_dir / config["downloads"]["subject_status_csv"]
    completed_path = output_dir / config["downloads"]["completed_subjects"]
    failed_path = output_dir / config["downloads"]["failed_subjects"]

    if not args.execute:
        planned_rows = [item_to_manifest_row(item) for item in items]
        write_csv(manifest_path, DOWNLOAD_FIELDS, planned_rows)
        print(
            json.dumps(
                {
                    "status": "DRY_RUN",
                    "subjects": len(subjects),
                    "files": len(items),
                    "run_name": run_name,
                    "manifest": str(manifest_path),
                    "space": space,
                    "next_step": "Re-run with --execute after verifying the plan.",
                },
                indent=2,
            )
        )
        return

    grouped = group_by_subject(items)
    max_workers = args.max_parallel_downloads or int(config["downloads"]["max_parallel_downloads"])
    rows_by_subject: dict[str, list[dict[str, str]]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(download_subject, probe, subject_id, subject_items, config): subject_id
            for subject_id, subject_items in grouped.items()
        }
        for future in as_completed(futures):
            subject_id = futures[future]
            try:
                rows_by_subject[subject_id] = future.result()
            except Exception as exc:  # noqa: BLE001 - one subject must not stop all.
                rows_by_subject[subject_id] = [
                    {
                        "subject_id": subject_id,
                        "source_modality": "subject",
                        "remote_path": "",
                        "local_path": "",
                        "file_size": "0",
                        "download_status": "failed",
                        "retry_count": "0",
                        "verification_status": type(exc).__name__ + ": " + str(exc),
                        "purpose": "subject download",
                    }
                ]
            print(
                f"{subject_id}: subject_done "
                f"{rows_by_subject[subject_id][-1]['download_status']}",
                flush=True,
            )

    all_rows = [row for subject in subjects for row in rows_by_subject.get(subject, [])]
    status_rows = [subject_status(subject, rows_by_subject.get(subject, [])) for subject in subjects]
    write_csv(manifest_path, DOWNLOAD_FIELDS, all_rows)
    write_csv(status_path, SUBJECT_STATUS_FIELDS, status_rows)
    completed = [row["subject_id"] for row in status_rows if row["final_status"] == "downloaded"]
    failed = [row["subject_id"] for row in status_rows if row["final_status"] != "downloaded"]
    completed_path.write_text("\n".join(completed) + ("\n" if completed else ""), encoding="utf-8")
    failed_path.write_text("\n".join(failed) + ("\n" if failed else ""), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "DONE",
                "completed_subjects": len(completed),
                "failed_subjects": len(failed),
                "run_name": run_name,
                "download_manifest": str(manifest_path),
                "subject_status": str(status_path),
                "next_stage": "Run source QC, FA/MD derivation, ALFF derivation, registration, and model-ready conversion. Do not start Fusion.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
