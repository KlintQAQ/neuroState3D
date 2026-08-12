from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from huggingface_hub import snapshot_download


ROOT = Path(__file__).resolve().parents[1]

DATASETS = {
    "gli": {
        "repo_id": "MedOtter/brats2023-gli-dataset",
        "local_subdir": "GLI",
        "expected_modalities": ("t1n", "t1c", "t2w", "t2f", "seg"),
    },
    "men": {
        "repo_id": "MedOtter/brats2023-men-dataset",
        "local_subdir": "MEN",
        "expected_modalities": ("t1n", "t1c", "t2w", "t2f", "seg"),
    },
    "ped": {
        "repo_id": "MedOtter/brats2023-ped-dataset",
        "local_subdir": "PED",
        "expected_modalities": ("t1n", "t1c", "t2w", "t2f", "seg"),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download BraTS 2023 NIfTI mirrors from Hugging Face.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["gli", "men", "ped"],
        choices=sorted(DATASETS),
    )
    parser.add_argument("--data-root", default=str(ROOT / "data"))
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--retries", type=int, default=20)
    parser.add_argument("--retry-sleep", type=int, default=120)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    raw_root = data_root / "raw" / "BraTS2023_HF"
    cache_dir = data_root / "cache" / "huggingface"
    manifest_dir = data_root / "manifests" / "BraTS2023_HF"
    raw_root.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("HF_XET_CACHE", str(cache_dir / "xet"))
    results = []
    for name in args.datasets:
        spec = DATASETS[name]
        local_dir = raw_root / spec["local_subdir"]
        local_dir.mkdir(parents=True, exist_ok=True)
        path = None
        for attempt in range(1, args.retries + 1):
            print(
                json.dumps(
                    {
                        "status": "START",
                        "dataset": name,
                        "repo_id": spec["repo_id"],
                        "local_dir": str(local_dir),
                        "attempt": attempt,
                        "retries": args.retries,
                    },
                    indent=2,
                ),
                flush=True,
            )
            try:
                path = snapshot_download(
                    repo_id=spec["repo_id"],
                    repo_type="dataset",
                    local_dir=str(local_dir),
                    cache_dir=str(cache_dir),
                    max_workers=args.max_workers,
                    resume_download=True,
                )
                break
            except Exception as exc:
                print(
                    json.dumps(
                        {
                            "status": "RETRY",
                            "dataset": name,
                            "attempt": attempt,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "sleep_seconds": args.retry_sleep,
                        },
                        indent=2,
                    ),
                    flush=True,
                )
                if attempt >= args.retries:
                    raise
                time.sleep(args.retry_sleep)
        if path is None:
            raise RuntimeError(f"Download did not return a path for {name}")
        nii_files = sorted(local_dir.rglob("*.nii.gz"))
        result = {
            "dataset": name,
            "repo_id": spec["repo_id"],
            "local_dir": str(local_dir),
            "snapshot_path": path,
            "nii_gz_files": len(nii_files),
            "subjects_estimated": len({item.parent.name for item in nii_files}),
            "expected_modalities": list(spec["expected_modalities"]),
        }
        (manifest_dir / f"{name}_download_summary.json").write_text(
            json.dumps(result, indent=2),
            encoding="utf-8",
        )
        print(json.dumps({"status": "DONE", **result}, indent=2), flush=True)
        results.append(result)

    (manifest_dir / "download_summary.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "ALL_DONE",
                "datasets": [item["dataset"] for item in results],
                "manifest_dir": str(manifest_dir),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    sys.exit(main())
