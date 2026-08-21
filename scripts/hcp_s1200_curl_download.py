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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.hcp_s1200_audit import data_root, ensure_safe_data_root, manifests_dir
from scripts.hcp_s1200_download import (
    DOWNLOAD_FIELDS,
    SUBJECT_STATUS_FIELDS,
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


def safe_is_relative(path: Path, root: Path) -> bool:
    return path.resolve().is_relative_to(root.resolve())


def adopt_python_part(local_path: Path, raw_root: Path, expected_size: int) -> None:
    if local_path.exists():
        return
    part = local_path.with_name(local_path.name + ".part")
    if not part.exists():
        return
    if not safe_is_relative(part, raw_root) or not safe_is_relative(local_path.parent, raw_root):
        return
    if part.stat().st_size <= expected_size:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        part.replace(local_path)


def remove_stale_part(local_path: Path, raw_root: Path) -> None:
    part = local_path.with_name(local_path.name + ".part")
    if part.exists() and safe_is_relative(part, raw_root):
        part.unlink()


def remove_safe_tree(path: Path, raw_root: Path) -> None:
    if path.exists() and path.is_dir() and safe_is_relative(path, raw_root):
        shutil.rmtree(path)


def resolve_curl() -> str:
    curl = shutil.which("curl.exe") or shutil.which("curl")
    if not curl:
        raise RuntimeError("curl was not found on PATH.")
    return curl


def presign_url(row: dict[str, str], bucket: str, profile: str, expires: int) -> str:
    s3_url = f"s3://{bucket}/{row['remote_path']}"
    cmd = [
        sys.executable,
        "-m",
        "awscli",
        "s3",
        "presign",
        s3_url,
        "--profile",
        profile,
        "--expires-in",
        str(expires),
    ]
    result = subprocess.run(
        cmd,
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout).strip())
    return result.stdout.strip()


def curl_download_row(
    row: dict[str, str],
    bucket: str,
    profile: str,
    curl: str,
    raw_root: Path,
    retry_count: int,
    expires: int,
    segment_size_mb: int,
    segment_workers: int,
    curl_max_time: int,
) -> dict[str, str]:
    local_path = Path(row["local_path"])
    expected_size = int(row["file_size"])
    if size_match(row):
        return {
            **row,
            "download_status": "skipped_existing",
            "retry_count": "0",
            "verification_status": "size_match",
        }

    adopt_python_part(local_path, raw_root, expected_size)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    if expected_size >= segment_size_mb * 1024 * 1024:
        return segmented_curl_download_row(
            row,
            bucket,
            profile,
            curl,
            raw_root,
            retry_count,
            expires,
            segment_size_mb,
            segment_workers,
            curl_max_time,
        )

    last_error = ""
    for attempt in range(retry_count + 1):
        try:
            url = presign_url(row, bucket, profile, expires)
            print(
                f"{row['subject_id']} start curl {row['source_modality']}: "
                f"{local_path.name} attempt={attempt + 1}/{retry_count + 1}",
                flush=True,
            )
            cmd = [
                curl,
                "--location",
                "--fail",
                "--silent",
                "--show-error",
                "--http1.1",
                "--ssl-no-revoke",
                "--retry",
                "20",
                "--retry-delay",
                "2",
                "--retry-max-time",
                "0",
                "--retry-all-errors",
                "--connect-timeout",
                "30",
                "--max-time",
                str(curl_max_time),
                "--speed-time",
                "120",
                "--speed-limit",
                "1024",
                "-C",
                "-",
                "-o",
                str(local_path),
                url,
            ]
            result = subprocess.run(
                cmd,
                cwd=str(ROOT),
                text=True,
                capture_output=True,
                check=False,
            )
            if result.returncode == 0 and size_match(row):
                remove_stale_part(local_path, raw_root)
                print(
                    f"{row['subject_id']} done curl {row['source_modality']}: "
                    f"{local_path.name} retries={attempt}",
                    flush=True,
                )
                return {
                    **row,
                    "download_status": "downloaded",
                    "retry_count": str(attempt),
                    "verification_status": "size_match",
                }
            last_error = (result.stderr or result.stdout or f"returncode={result.returncode}").strip()
        except Exception as exc:  # noqa: BLE001 - record and retry.
            last_error = type(exc).__name__ + ": " + str(exc)
        print(
            f"{row['subject_id']} failed curl {row['source_modality']}: "
            f"{local_path.name}; {last_error[:500]}",
            flush=True,
        )
        time.sleep(min(2**attempt, 30))

    verification = "missing_local_file"
    if local_path.exists():
        verification = f"local_size={local_path.stat().st_size}"
    return {
        **row,
        "download_status": "failed",
        "retry_count": str(retry_count),
        "verification_status": f"{verification}; {last_error[:500]}",
    }


def segment_specs(expected_size: int, segment_size_mb: int) -> list[tuple[int, int, int]]:
    segment_size = segment_size_mb * 1024 * 1024
    specs: list[tuple[int, int, int]] = []
    start = 0
    index = 0
    while start < expected_size:
        end = min(start + segment_size - 1, expected_size - 1)
        specs.append((index, start, end))
        start = end + 1
        index += 1
    return specs


def segment_path(segments_dir: Path, index: int) -> Path:
    return segments_dir / f"segment_{index:05d}.part"


def curl_segment(
    curl: str,
    url: str,
    path: Path,
    start: int,
    end: int,
    raw_root: Path,
    curl_max_time: int,
) -> tuple[bool, str]:
    expected = end - start + 1
    if path.exists() and path.stat().st_size == expected:
        return True, "segment_exists"
    if path.exists() and safe_is_relative(path, raw_root):
        path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        curl,
        "--location",
        "--fail",
        "--silent",
        "--show-error",
        "--http1.1",
        "--ssl-no-revoke",
        "--retry",
        "10",
        "--retry-delay",
        "2",
        "--retry-all-errors",
        "--connect-timeout",
        "30",
        "--max-time",
        str(curl_max_time),
        "--speed-time",
        "120",
        "--speed-limit",
        "1024",
        "--range",
        f"{start}-{end}",
        "-o",
        str(path),
        url,
    ]
    result = subprocess.run(
        cmd,
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode == 0 and path.exists() and path.stat().st_size == expected:
        return True, "segment_size_match"
    detail = (result.stderr or result.stdout or f"returncode={result.returncode}").strip()
    if path.exists():
        detail = f"local_size={path.stat().st_size}; {detail}"
    return False, detail


def segmented_curl_download_row(
    row: dict[str, str],
    bucket: str,
    profile: str,
    curl: str,
    raw_root: Path,
    retry_count: int,
    expires: int,
    segment_size_mb: int,
    segment_workers: int,
    curl_max_time: int,
) -> dict[str, str]:
    local_path = Path(row["local_path"])
    expected_size = int(row["file_size"])
    segments_dir = local_path.with_name(local_path.name + ".segments")
    if not safe_is_relative(segments_dir, raw_root):
        raise RuntimeError(f"Unsafe segment directory: {segments_dir}")
    specs = segment_specs(expected_size, segment_size_mb)
    last_error = ""

    if local_path.exists() and not size_match(row) and safe_is_relative(local_path, raw_root):
        local_path.unlink()

    for attempt in range(retry_count + 1):
        url = presign_url(row, bucket, profile, expires)
        print(
            f"{row['subject_id']} start segmented curl {row['source_modality']}: "
            f"{local_path.name} segments={len(specs)} attempt={attempt + 1}/{retry_count + 1}",
            flush=True,
        )
        pending = [
            (index, start, end)
            for index, start, end in specs
            if not (
                segment_path(segments_dir, index).exists()
                and segment_path(segments_dir, index).stat().st_size == end - start + 1
            )
        ]
        ok = True
        with ThreadPoolExecutor(max_workers=max(segment_workers, 1)) as executor:
            futures = [
                executor.submit(
                    curl_segment,
                    curl,
                    url,
                    segment_path(segments_dir, index),
                    start,
                    end,
                    raw_root,
                    curl_max_time,
                )
                for index, start, end in pending
            ]
            for future in as_completed(futures):
                segment_ok, detail = future.result()
                if not segment_ok:
                    ok = False
                    last_error = detail
        if ok:
            try:
                with local_path.open("wb") as output:
                    for index, start, end in specs:
                        path = segment_path(segments_dir, index)
                        expected = end - start + 1
                        if not path.exists() or path.stat().st_size != expected:
                            raise RuntimeError(f"segment missing or mismatched: {path}")
                        with path.open("rb") as segment:
                            shutil.copyfileobj(segment, output, length=1024 * 1024)
                if size_match(row):
                    remove_safe_tree(segments_dir, raw_root)
                    remove_stale_part(local_path, raw_root)
                    print(
                        f"{row['subject_id']} done segmented curl {row['source_modality']}: "
                        f"{local_path.name} retries={attempt}",
                        flush=True,
                    )
                    return {
                        **row,
                        "download_status": "downloaded",
                        "retry_count": str(attempt),
                        "verification_status": "size_match",
                    }
                last_error = f"merged_size={local_path.stat().st_size if local_path.exists() else 0}"
            except Exception as exc:  # noqa: BLE001 - record and retry.
                last_error = type(exc).__name__ + ": " + str(exc)
        print(
            f"{row['subject_id']} failed segmented curl {row['source_modality']}: "
            f"{local_path.name}; {last_error[:500]}",
            flush=True,
        )
        time.sleep(min(2**attempt, 30))

    verification = "missing_local_file"
    if local_path.exists():
        verification = f"local_size={local_path.stat().st_size}"
    return {
        **row,
        "download_status": "failed",
        "retry_count": str(retry_count),
        "verification_status": f"{verification}; {last_error[:500]}",
    }


def group_by_subject(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["subject_id"], []).append(row)
    return grouped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download an audited HCP S1200 manifest with presigned S3 URLs and curl resume."
    )
    parser.add_argument("--config", default="configs/hcp_s1200.yaml")
    parser.add_argument("--manifest", default="")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--profile", default="hcp")
    parser.add_argument("--max-workers", type=int, default=5)
    parser.add_argument("--retry-count", type=int, default=4)
    parser.add_argument("--expires-in", type=int, default=604800)
    parser.add_argument("--segment-size-mb", type=int, default=64)
    parser.add_argument("--segment-workers", type=int, default=8)
    parser.add_argument("--curl-max-time", type=int, default=300)
    return parser.parse_args()


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

    rows = read_manifest(manifest_path)
    raw_root = root / config["layout"]["raw"]
    curl = resolve_curl()
    bucket = str(config["bucket"])
    max_workers = max(args.max_workers, 1)
    print(
        json.dumps(
            {
                "status": "CURL_DOWNLOAD_START",
                "files": len(rows),
                "already_complete": sum(size_match(row) for row in rows),
                "remaining": sum(not size_match(row) for row in rows),
                "manifest": str(manifest_path),
                "curl": curl,
                "max_workers": max_workers,
            },
            indent=2,
        ),
        flush=True,
    )

    completed_rows: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                curl_download_row,
                row,
                bucket,
                args.profile,
                curl,
                raw_root,
                args.retry_count,
                args.expires_in,
                args.segment_size_mb,
                args.segment_workers,
                args.curl_max_time,
            )
            for row in rows
        ]
        for future in as_completed(futures):
            completed_rows.append(future.result())
            if len(completed_rows) % 25 == 0 or len(completed_rows) == len(rows):
                print(f"curl_progress [{len(completed_rows)}/{len(rows)}]", flush=True)

    by_subject = group_by_subject(completed_rows)
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
                "status": "CURL_DOWNLOAD_DONE",
                "completed_subjects": len(completed),
                "failed_subjects": len(failed),
                "download_manifest": str(manifest_path),
                "subject_status": str(status_path),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
