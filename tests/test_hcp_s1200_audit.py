from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.hcp_s1200_audit import ObjectProbe, audit_subject, ensure_safe_data_root


class FakeProbe:
    def __init__(self, available: dict[str, int]) -> None:
        self.available = available

    def first_available(self, candidates: list[str]) -> ObjectProbe:
        for key in candidates:
            if key in self.available:
                return ObjectProbe(True, key=key, size=self.available[key])
        return ObjectProbe(False, error="missing")

    def head(self, key: str) -> ObjectProbe:
        if key in self.available:
            return ObjectProbe(True, key=key, size=self.available[key])
        return ObjectProbe(False, key=key, error="missing")


def config_for(tmp_root: Path) -> dict:
    return {
        "data_root": str(tmp_root),
        "release_prefix": "HCP_1200",
        "layout": {"manifests": "manifests"},
        "safety": {"forbidden_drives": ["C"], "forbid_repo_data": True},
        "audit": {"min_valid_rest_runs_relaxed": 1},
        "remote_files": {
            "structural": {
                "t1": {"candidates": ["HCP_1200/{subject_id}/T1w/T1.nii.gz"]},
                "t2": {"candidates": ["HCP_1200/{subject_id}/T1w/T2.nii.gz"]},
            },
            "diffusion": {
                "dwi": {"candidates": ["HCP_1200/{subject_id}/DWI/data.nii.gz"]},
                "bval": {"candidates": ["HCP_1200/{subject_id}/DWI/bvals"]},
                "bvec": {"candidates": ["HCP_1200/{subject_id}/DWI/bvecs"]},
            },
            "resting_fmri": {
                run: {"candidates": [f"HCP_1200/{{subject_id}}/{run}.nii.gz"]}
                for run in ("REST1_LR", "REST1_RL", "REST2_LR", "REST2_RL")
            },
        },
    }


class HCPS1200AuditTest(unittest.TestCase):
    def test_strict_and_relaxed_eligibility_are_distinct(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = config_for(Path(tmp))
            available = {
                "HCP_1200/100206/T1w/T1.nii.gz": 10,
                "HCP_1200/100206/T1w/T2.nii.gz": 10,
                "HCP_1200/100206/DWI/data.nii.gz": 30,
                "HCP_1200/100206/DWI/bvals": 1,
                "HCP_1200/100206/DWI/bvecs": 1,
                "HCP_1200/100206/REST1_LR.nii.gz": 100,
            }
            row = audit_subject("100206", cfg, FakeProbe(available))  # type: ignore[arg-type]
            self.assertEqual(row["relaxed_eligible"], "true")
            self.assertEqual(row["strict_eligible"], "false")
            self.assertEqual(row["num_valid_rest_runs"], "1")
            self.assertEqual(row["strict_exclusion_reason"], "strict_missing_rest_runs:REST1_RL,REST2_LR,REST2_RL")

    def test_missing_dwi_excludes_subject(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = config_for(Path(tmp))
            available = {
                "HCP_1200/100307/T1w/T1.nii.gz": 10,
                "HCP_1200/100307/T1w/T2.nii.gz": 10,
                "HCP_1200/100307/REST1_LR.nii.gz": 100,
                "HCP_1200/100307/REST1_RL.nii.gz": 100,
                "HCP_1200/100307/REST2_LR.nii.gz": 100,
                "HCP_1200/100307/REST2_RL.nii.gz": 100,
            }
            row = audit_subject("100307", cfg, FakeProbe(available))  # type: ignore[arg-type]
            self.assertEqual(row["eligible"], "false")
            self.assertIn("missing_DWI", row["exclusion_reason"])

    def test_refuses_repo_data_root(self) -> None:
        cfg = {
            "safety": {"forbidden_drives": ["C"], "forbid_repo_data": True},
        }
        with self.assertRaisesRegex(ValueError, "inside the git repository"):
            ensure_safe_data_root(ROOT / "data", cfg)


if __name__ == "__main__":
    unittest.main()
