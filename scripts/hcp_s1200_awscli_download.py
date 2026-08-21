from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.hcp_s1200_audit import data_root, ensure_safe_data_root, manifests_dir
from scripts.hcp_s1200_download import (
    DOWNLOAD_FIELDS,
    SUBJECT_STATUS_FIELDS,
    group_by_subject,
    load_config,
    subject_status,
    write_csv,
)


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [field for field in DOWNLOAD_FIELDS if field not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Download manifest is missing required fields: {missing}")
        return list(reader)


def size_match(row: dict[str, str]) -> bool:
    path = Path(row["local_path"])
    expected = int(row["file_size"])
    return path.exists() and path.is_file() and path.stat().st_size == expected


def download_row(
    row: dict[str, str],
    bucket: str,
    profile: str,
    retry_count: int,
    timeout_seconds: int,
    raw_root: Path,
    aws_command: list[str],
) -> dict[str, str]:
    local_path = Path(row["local_path"])
    if size_match(row):
        return {
            **row,
            "download_status": "skipped_existing",
            "retry_count": "0",
            "verification_status": "size_match",
        }

    local_path.parent.mkdir(parents=True, exist_ok=True)
    source = f"s3://{bucket}/{row['remote_path']}"
    env = os.environ.copy()
    env["AWS_RETRY_MODE"] = "standard"
    env["AWS_MAX_ATTEMPTS"] = "12"

    last_error = ""
    for attempt in range(retry_count + 1):
        print(
            f"{row['subject_id']} start awscli {row['source_modality']}: "
            f"{local_path.name} attempt={attempt + 1}/{retry_count + 1}",
            flush=True,
        )
        cmd = [
            *aws_command,
            "s3",
            "cp",
            source,
            str(local_path),
            "--profile",
            profile,
            "--only-show-errors",
            "--cli-connect-timeout",
            "30",
            "--cli-read-timeout",
            str(timeout_seconds),
        ]
        try:
            result = subprocess.run(
                cmd,
                cwd=str(ROOT),
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
        except FileNotFoundError as exc:
            last_error = type(exc).__name__ + ": " + str(exc)
            print(
                f"{row['subject_id']} failed awscli {row['source_modality']}: "
                f"{local_path.name}; {last_error}",
                flush=True,
            )
            time.sleep(min(2**attempt, 30))
            continue
        if result.returncode == 0 and size_match(row):
            remove_stale_part(local_path, raw_root)
            print(
                f"{row['subject_id']} done awscli {row['source_modality']}: "
                f"{local_path.name} size_match retries={attempt}",
                flush=True,
            )
            return {
                **row,
                "download_status": "downloaded",
                "retry_count": str(attempt),
                "verification_status": "size_match",
            }
        last_error = (result.stderr or result.stdout or f"returncode={result.returncode}").strip()
        print(
            f"{row['subject_id']} failed awscli {row['source_modality']}: "
            f"{local_path.name}; {last_error[:500]}",
            flush=True,
        )
        time.sleep(min(2**attempt, 30))

    verification = "size_mismatch"
    if not local_path.exists():
        verification = "missing_local_file"
    elif local_path.stat().st_size != int(row["file_size"]):
        verification = f"local_size={local_path.stat().st_size}"
    return {
        **row,
        "download_status": "failed",
        "retry_count": str(retry_count),
        "verification_status": f"{verification}; {last_error[:500]}",
    }


def remove_stale_part(local_path: Path, raw_root: Path) -> None:
    part = local_path.with_name(local_path.name + ".part")
    if not part.exists():
        return
    resolved_part = part.resolve()
    resolved_raw = raw_root.resolve()
    if resolved_part.is_relative_to(resolved_raw):
        part.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download an audited HCP S1200 manifest using AWS CLI transfer manager."
    )
    parser.add_argument("--config", default="configs/hcp_s1200.yaml")
    parser.add_argument("--manifest", default="")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--profile", default="hcp")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--retry-count", type=int, default=5)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    return parser.parse_args()


def resolve_aws_command() -> list[str]:
    aws_cmd = shutil.which("aws.cmd")
    if aws_cmd:
        return [aws_cmd]
    aws_exe = shutil.which("aws.exe")
    if aws_exe:
        return [aws_exe]
    aws_plain = shutil.which("aws")
    if aws_plain:
        return [aws_plain]
    if os.name == "nt":
        return ["cmd.exe", "/d", "/c", "aws"]
    return ["aws"]


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    root = data_root(config)
    ensure_safe_data_root(root, config)
    mdir = manifests_dir(config)
    run_name = args.run_name.strip()
    if args.manifest:
        manifest_path = Path(args.manifest)
        output_dir = manifest_path.parent
    elif run_name:
        output_dir = mdir / str(config["downloads"].get("runs_dir", "downloads")) / run_name
        manifest_path = output_dir / config["downloads"]["manifest_csv"]
    else:
        output_dir = mdir
        manifest_path = output_dir / config["downloads"]["manifest_csv"]

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_manifest(manifest_path)
    raw_root = root / config["layout"]["raw"]
    bucket = str(config["bucket"])
    max_workers = max(args.max_workers, 1)
    aws_command = resolve_aws_command()

    print(
        json.dumps(
            {
                "status": "AWSCLI_DOWNLOAD_START",
                "files": len(rows),
                "already_complete": sum(size_match(row) for row in rows),
                "remaining": sum(not size_match(row) for row in rows),
                "manifest": str(manifest_path),
                "profile": args.profile,
                "max_workers": max_workers,
                "aws_command": aws_command,
            },
            indent=2,
        ),
        flush=True,
    )

    completed_rows: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                download_row,
                row,
                bucket,
                args.profile,
                args.retry_count,
                args.timeout_seconds,
                raw_root,
                aws_command,
            )
            for row in rows
        ]
        for future in as_completed(futures):
            completed_rows.append(future.result())
            if len(completed_rows) % 25 == 0 or len(completed_rows) == len(rows):
                print(f"awscli_progress [{len(completed_rows)}/{len(rows)}]", flush=True)

    by_subject = group_by_subject_like_rows(completed_rows)
    subjects = sorted(by_subject)
    ordered_rows = [
        row
        for subject_id in subjects
        for row in sorted(by_subject[subject_id], key=lambda item: item["remote_path"])
    ]
    status_rows = [subject_status(subject, by_subject[subject]) for subject in subjects]

    status_path = output_dir / config["downloads"]["subject_status_csv"]
    completed_path = output_dir / config["downloads"]["completed_subjects"]
    failed_path = output_dir / config["downloads"]["failed_subjects"]
    write_csv(manifest_path, DOWNLOAD_FIELDS, ordered_rows)
    write_csv(status_path, SUBJECT_STATUS_FIELDS, status_rows)
    completed = [row["subject_id"] for row in status_rows if row["final_status"] == "downloaded"]
    failed = [row["subject_id"] for row in status_rows if row["final_status"] != "downloaded"]
    completed_path.write_text("\n".join(completed) + ("\n" if completed else ""), encoding="utf-8")
    failed_path.write_text("\n".join(failed) + ("\n" if failed else ""), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "AWSCLI_DOWNLOAD_DONE",
                "completed_subjects": len(completed),
                "failed_subjects": len(failed),
                "download_manifest": str(manifest_path),
                "subject_status": str(status_path),
            },
            indent=2,
        ),
        flush=True,
    )


def group_by_subject_like_rows(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["subject_id"], []).append(row)
    return grouped


if __name__ == "__main__":
    main()
