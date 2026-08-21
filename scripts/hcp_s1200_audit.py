from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


try:
    from botocore import UNSIGNED
    from botocore.config import Config
    from botocore.exceptions import (
        ClientError,
        NoCredentialsError,
        PartialCredentialsError,
        ProfileNotFound,
    )
    from botocore.session import get_session
except Exception:  # pragma: no cover - import failure is reported at runtime.
    UNSIGNED = None
    Config = None
    ClientError = Exception
    NoCredentialsError = Exception
    PartialCredentialsError = Exception
    ProfileNotFound = Exception
    get_session = None


REST_RUNS = ("REST1_LR", "REST1_RL", "REST2_LR", "REST2_RL")

CSV_FIELDS = [
    "subject_id",
    "T1_available",
    "T2_available",
    "DWI_available",
    "bval_available",
    "bvec_available",
    "REST1_LR_available",
    "REST1_RL_available",
    "REST2_LR_available",
    "REST2_RL_available",
    "num_valid_rest_runs",
    "FA_possible",
    "MD_possible",
    "ALFF_possible",
    "eligible",
    "strict_eligible",
    "relaxed_eligible",
    "estimated_download_size_gb",
    "exclusion_reason",
    "strict_exclusion_reason",
    "selected_t1_key",
    "selected_t2_key",
    "selected_dwi_key",
    "selected_bval_key",
    "selected_bvec_key",
    "selected_REST1_LR_key",
    "selected_REST1_RL_key",
    "selected_REST2_LR_key",
    "selected_REST2_RL_key",
    "selected_optional_keys",
    "s3_audit_notes",
]


@dataclass(frozen=True)
class ObjectProbe:
    available: bool
    key: str = ""
    size: int = 0
    error: str = ""


class S3AuditError(RuntimeError):
    pass


class S3Probe:
    def __init__(
        self,
        bucket: str,
        unsigned: bool = False,
        profile: str = "",
        region: str = "us-east-1",
    ) -> None:
        if get_session is None:
            raise S3AuditError(
                "botocore is required for S3 audit. Install botocore or run in the "
                "project Python environment."
            )
        if Config is None:
            raise S3AuditError("botocore Config is unavailable.")
        config_kwargs = {
            "connect_timeout": 10,
            "read_timeout": 120,
            "max_pool_connections": 64,
            "retries": {"max_attempts": 10, "mode": "standard"},
        }
        if unsigned:
            if UNSIGNED is None:
                raise S3AuditError("Unsigned S3 access requested but botocore config is unavailable.")
            config = Config(signature_version=UNSIGNED, **config_kwargs)
        else:
            config = Config(**config_kwargs)
        session = get_session()
        if profile:
            session.set_config_variable("profile", profile)
        try:
            self.client = session.create_client("s3", region_name=region, config=config)
        except ProfileNotFound as exc:
            raise S3AuditError(
                f"AWS profile '{profile}' was not found. Configure it with authorized HCP credentials "
                "or pass --profile for an existing profile."
            ) from exc
        self.bucket = bucket

    def list_subject_ids(self, release_prefix: str) -> list[str]:
        prefix = release_prefix.strip("/") + "/"
        try:
            paginator = self.client.get_paginator("list_objects_v2")
            subject_ids: list[str] = []
            for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix, Delimiter="/"):
                for item in page.get("CommonPrefixes", []):
                    child = item.get("Prefix", "").strip("/").split("/")[-1]
                    if child.isdigit():
                        subject_ids.append(child)
            return sorted(set(subject_ids))
        except (NoCredentialsError, PartialCredentialsError) as exc:
            raise S3AuditError(
                "AWS credentials are required for HCP S1200 S3 listing. "
                "Run aws configure with authorized HCP/ConnectomeDB credentials."
            ) from exc
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "UNKNOWN")
            raise S3AuditError(f"S3 subject listing failed with {code}: {exc}") from exc

    def head(self, key: str) -> ObjectProbe:
        try:
            response = self.client.head_object(Bucket=self.bucket, Key=key)
        except (NoCredentialsError, PartialCredentialsError) as exc:
            raise S3AuditError(
                "AWS credentials are required for HCP S1200 S3 head_object checks."
            ) from exc
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "UNKNOWN")
            if code in {"404", "NoSuchKey", "NotFound"}:
                return ObjectProbe(False, key=key, error=code)
            if code in {"403", "AccessDenied"}:
                return ObjectProbe(False, key=key, error=code)
            return ObjectProbe(False, key=key, error=f"{code}: {exc}")
        return ObjectProbe(True, key=key, size=int(response.get("ContentLength", 0)))

    def first_available(self, candidates: Iterable[str]) -> ObjectProbe:
        errors: list[str] = []
        for key in candidates:
            result = self.head(key)
            if result.available:
                return result
            if result.error:
                errors.append(f"{key}:{result.error}")
        return ObjectProbe(False, error=";".join(errors))


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def data_root(config: dict[str, Any]) -> Path:
    return Path(str(config["data_root"])).expanduser()


def manifests_dir(config: dict[str, Any]) -> Path:
    return data_root(config) / str(config["layout"]["manifests"])


def ensure_safe_data_root(path: Path, config: dict[str, Any]) -> None:
    safety = config.get("safety", {})
    forbidden = {str(item).upper().rstrip(":") for item in safety.get("forbidden_drives", [])}
    drive = path.drive.upper().rstrip(":")
    if drive in forbidden:
        raise ValueError(f"Refusing to use forbidden data drive {path.drive}: {path}")
    resolved_root = path.resolve()
    repo_root = ROOT.resolve()
    if safety.get("forbid_repo_data", True) and resolved_root.is_relative_to(repo_root):
        raise ValueError(f"Refusing to place HCP data inside the git repository: {path}")


def render_candidates(
    spec: dict[str, Any],
    release_prefix: str,
    subject_id: str,
) -> list[str]:
    candidates = spec.get("candidates", [])
    return [
        str(item).format(release_prefix=release_prefix, subject_id=subject_id)
        for item in candidates
    ]


def check_spec(
    probe: S3Probe,
    spec: dict[str, Any],
    release_prefix: str,
    subject_id: str,
) -> ObjectProbe:
    candidates = render_candidates(spec, release_prefix, subject_id)
    return probe.first_available(candidates)


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def reason_join(reasons: list[str]) -> str:
    return ";".join(reasons) if reasons else "PASS"


def audit_subject(subject_id: str, config: dict[str, Any], probe: S3Probe) -> dict[str, str]:
    prefix = str(config["release_prefix"]).strip("/")
    remote = config["remote_files"]
    notes: list[str] = []
    optional_keys: list[str] = []

    t1 = check_spec(probe, remote["structural"]["t1"], prefix, subject_id)
    t2 = check_spec(probe, remote["structural"]["t2"], prefix, subject_id)
    dwi = check_spec(probe, remote["diffusion"]["dwi"], prefix, subject_id)
    bval = check_spec(probe, remote["diffusion"]["bval"], prefix, subject_id)
    bvec = check_spec(probe, remote["diffusion"]["bvec"], prefix, subject_id)

    rest: dict[str, ObjectProbe] = {}
    for run in REST_RUNS:
        rest[run] = check_spec(probe, remote["resting_fmri"][run], prefix, subject_id)

    for group_name in ("structural", "diffusion"):
        for name, spec in remote[group_name].items():
            if not spec.get("optional"):
                continue
            result = check_spec(probe, spec, prefix, subject_id)
            if result.available:
                optional_keys.append(result.key)
            else:
                notes.append(f"optional_missing:{group_name}.{name}")

    valid_rest_runs = sum(1 for run in REST_RUNS if rest[run].available)
    fa_possible = dwi.available and bval.available and bvec.available
    md_possible = fa_possible
    alff_possible = valid_rest_runs >= int(config["audit"]["min_valid_rest_runs_relaxed"])
    relaxed = t1.available and t2.available and fa_possible and md_possible and alff_possible
    strict = relaxed and valid_rest_runs == len(REST_RUNS)

    relaxed_reasons: list[str] = []
    if not t1.available:
        relaxed_reasons.append("missing_T1")
    if not t2.available:
        relaxed_reasons.append("missing_T2")
    if not dwi.available:
        relaxed_reasons.append("missing_DWI")
    if not bval.available:
        relaxed_reasons.append("missing_bval")
    if not bvec.available:
        relaxed_reasons.append("missing_bvec")
    if valid_rest_runs < int(config["audit"]["min_valid_rest_runs_relaxed"]):
        relaxed_reasons.append("missing_all_resting_fMRI")

    strict_reasons = list(relaxed_reasons)
    missing_rest = [run for run in REST_RUNS if not rest[run].available]
    if not missing_rest:
        pass
    elif relaxed:
        strict_reasons.append("strict_missing_rest_runs:" + ",".join(missing_rest))

    selected_required = [t1, t2, dwi, bval, bvec, *[rest[run] for run in REST_RUNS]]
    selected_optional_size = sum(probe_result.size for probe_result in [probe.head(key) for key in optional_keys])
    estimated_bytes = sum(item.size for item in selected_required if item.available) + selected_optional_size

    row: dict[str, str] = {
        "subject_id": subject_id,
        "T1_available": bool_text(t1.available),
        "T2_available": bool_text(t2.available),
        "DWI_available": bool_text(dwi.available),
        "bval_available": bool_text(bval.available),
        "bvec_available": bool_text(bvec.available),
        "REST1_LR_available": bool_text(rest["REST1_LR"].available),
        "REST1_RL_available": bool_text(rest["REST1_RL"].available),
        "REST2_LR_available": bool_text(rest["REST2_LR"].available),
        "REST2_RL_available": bool_text(rest["REST2_RL"].available),
        "num_valid_rest_runs": str(valid_rest_runs),
        "FA_possible": bool_text(fa_possible),
        "MD_possible": bool_text(md_possible),
        "ALFF_possible": bool_text(alff_possible),
        "eligible": bool_text(relaxed),
        "strict_eligible": bool_text(strict),
        "relaxed_eligible": bool_text(relaxed),
        "estimated_download_size_gb": f"{estimated_bytes / (1024**3):.6f}",
        "exclusion_reason": reason_join(relaxed_reasons),
        "strict_exclusion_reason": reason_join(strict_reasons),
        "selected_t1_key": t1.key if t1.available else "",
        "selected_t2_key": t2.key if t2.available else "",
        "selected_dwi_key": dwi.key if dwi.available else "",
        "selected_bval_key": bval.key if bval.available else "",
        "selected_bvec_key": bvec.key if bvec.available else "",
        "selected_REST1_LR_key": rest["REST1_LR"].key if rest["REST1_LR"].available else "",
        "selected_REST1_RL_key": rest["REST1_RL"].key if rest["REST1_RL"].available else "",
        "selected_REST2_LR_key": rest["REST2_LR"].key if rest["REST2_LR"].available else "",
        "selected_REST2_RL_key": rest["REST2_RL"].key if rest["REST2_RL"].available else "",
        "selected_optional_keys": "|".join(optional_keys),
        "s3_audit_notes": ";".join(notes),
    }
    return row


def failed_audit_row(subject_id: str, reason: str) -> dict[str, str]:
    row = {field: "" for field in CSV_FIELDS}
    row.update(
        {
            "subject_id": subject_id,
            "T1_available": "false",
            "T2_available": "false",
            "DWI_available": "false",
            "bval_available": "false",
            "bvec_available": "false",
            "REST1_LR_available": "false",
            "REST1_RL_available": "false",
            "REST2_LR_available": "false",
            "REST2_RL_available": "false",
            "num_valid_rest_runs": "0",
            "FA_possible": "false",
            "MD_possible": "false",
            "ALFF_possible": "false",
            "eligible": "false",
            "strict_eligible": "false",
            "relaxed_eligible": "false",
            "estimated_download_size_gb": "0.000000",
            "exclusion_reason": "s3_audit_error",
            "strict_exclusion_reason": "s3_audit_error",
            "s3_audit_notes": reason,
        }
    )
    return row


def read_subject_ids(path: Path) -> list[str]:
    subjects: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if value and not value.startswith("#"):
            subjects.append(value)
    return subjects


def write_lines(path: Path, values: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(values) + ("\n" if values else ""), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def disk_summary(root: Path) -> dict[str, int | str]:
    usage = shutil.disk_usage(root.anchor or str(root))
    return {
        "root": str(root),
        "total_space_bytes": int(usage.total),
        "free_space_bytes": int(usage.free),
        "total_space_gb": f"{usage.total / (1024**3):.3f}",
        "free_space_gb": f"{usage.free / (1024**3):.3f}",
    }


def build_summary(rows: list[dict[str, str]], config: dict[str, Any]) -> dict[str, Any]:
    strict_rows = [row for row in rows if row["strict_eligible"] == "true"]
    relaxed_rows = [row for row in rows if row["relaxed_eligible"] == "true"]
    excluded_rows = [row for row in rows if row["relaxed_eligible"] != "true"]
    estimated_strict_gb = sum(float(row["estimated_download_size_gb"]) for row in strict_rows)
    multiplier = float(config["safety"].get("required_free_space_multiplier", 1.5))
    required_gb = estimated_strict_gb * multiplier
    root = data_root(config)
    disk = disk_summary(root)
    reasons = Counter(row["exclusion_reason"] for row in excluded_rows)
    return {
        "release": config.get("release", "HCP-S1200"),
        "bucket": config["bucket"],
        "release_prefix": config["release_prefix"],
        "data_root": str(root),
        "N_total": len(rows),
        "N_strict": len(strict_rows),
        "N_relaxed": len(relaxed_rows),
        "N_excluded": len(excluded_rows),
        "estimated_total_download_gb_strict": round(estimated_strict_gb, 6),
        "estimated_total_download_tb": round(estimated_strict_gb / 1024, 6),
        "required_free_space_gb": round(required_gb, 6),
        "required_free_space_multiplier": multiplier,
        "disk": disk,
        "space_ok": float(disk["free_space_bytes"]) >= required_gb * (1024**3),
        "top_exclusion_reasons": reasons.most_common(20),
        "post_download_validation_pending": [
            "DWI volume count vs bval/bvec count",
            "FA/MD tensor fitting QC",
            "ALFF run-level stability QC",
            "registration and five-volume physical-space QC",
        ],
    }


def output_paths(config: dict[str, Any]) -> dict[str, Path]:
    mdir = manifests_dir(config)
    audit_cfg = config["audit"]
    return {
        "all": mdir / audit_cfg["output_all_subjects"],
        "eligible": mdir / audit_cfg["output_eligible_subjects"],
        "strict": mdir / audit_cfg["output_strict_subjects"],
        "relaxed": mdir / audit_cfg["output_relaxed_subjects"],
        "summary": mdir / audit_cfg["output_summary_json"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit HCP S1200 S3 assets for NeuroState-3D five-map eligibility. "
            "This command lists/heads S3 objects only and does not download data."
        )
    )
    parser.add_argument("--config", default="configs/hcp_s1200.yaml")
    parser.add_argument("--subject-list", default="", help="Optional local subject ID list.")
    parser.add_argument("--limit", type=int, default=0, help="Audit only the first N subjects.")
    parser.add_argument("--unsigned", action="store_true", help="Try unsigned S3 access.")
    parser.add_argument("--profile", default="", help="AWS profile name for HCP S3 access.")
    parser.add_argument("--region", default="", help="AWS region, defaults to config value.")
    parser.add_argument("--max-workers", default=0, type=int, help="Concurrent S3 head checks.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.unsigned:
        config["unsigned"] = True
    if args.profile:
        config["aws_profile"] = args.profile
    if args.region:
        config["region"] = args.region
    root = data_root(config)
    ensure_safe_data_root(root, config)
    manifests_dir(config).mkdir(parents=True, exist_ok=True)

    try:
        probe = S3Probe(
            str(config["bucket"]),
            unsigned=bool(config.get("unsigned", False)),
            profile=str(config.get("aws_profile", "")),
            region=str(config.get("region", "us-east-1")),
        )
        if args.subject_list:
            subjects = read_subject_ids(Path(args.subject_list))
        else:
            subjects = probe.list_subject_ids(str(config["release_prefix"]))
        if args.limit:
            subjects = subjects[: args.limit]
        if not subjects:
            raise S3AuditError("No HCP S1200 subject IDs were found.")
        print(f"Found {len(subjects)} HCP S1200 subjects to audit.", flush=True)

        max_workers = args.max_workers or int(config["audit"].get("max_workers", 1))
        rows = []
        if max_workers <= 1:
            for index, subject_id in enumerate(subjects, 1):
                row = audit_subject(subject_id, config, probe)
                rows.append(row)
                print(
                    f"[{index}/{len(subjects)}] {subject_id} "
                    f"strict={row['strict_eligible']} relaxed={row['relaxed_eligible']} "
                    f"rest_runs={row['num_valid_rest_runs']}",
                    flush=True,
                )
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(audit_subject, subject_id, config, probe): subject_id
                    for subject_id in subjects
                }
                for index, future in enumerate(as_completed(futures), 1):
                    subject_id = futures[future]
                    try:
                        row = future.result()
                    except Exception as exc:  # noqa: BLE001 - record per subject and continue.
                        row = failed_audit_row(subject_id, f"{type(exc).__name__}: {exc}")
                    rows.append(row)
                    print(
                        f"[{index}/{len(subjects)}] {subject_id} "
                        f"strict={row['strict_eligible']} relaxed={row['relaxed_eligible']} "
                        f"rest_runs={row['num_valid_rest_runs']}",
                        flush=True,
                    )
        rows = sorted(rows, key=lambda row: row["subject_id"])
    except S3AuditError as exc:
        report = {
            "status": "PENDING_HCP_AWS_AUTHENTICATION",
            "reason": str(exc),
            "bucket": config["bucket"],
            "release_prefix": config["release_prefix"],
            "aws_profile": str(config.get("aws_profile", "")),
            "region": str(config.get("region", "us-east-1")),
            "data_root": str(root),
            "download_started": False,
        }
        paths = output_paths(config)
        paths["summary"].write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        raise SystemExit(2) from exc

    paths = output_paths(config)
    write_csv(paths["all"], rows)
    relaxed_ids = [row["subject_id"] for row in rows if row["relaxed_eligible"] == "true"]
    strict_ids = [row["subject_id"] for row in rows if row["strict_eligible"] == "true"]
    selected_eligible = strict_ids if config["audit"].get("write_eligible_subjects_as") == "strict" else relaxed_ids
    write_lines(paths["eligible"], selected_eligible)
    write_lines(paths["strict"], strict_ids)
    write_lines(paths["relaxed"], relaxed_ids)
    summary = build_summary(rows, config)
    paths["summary"].write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
