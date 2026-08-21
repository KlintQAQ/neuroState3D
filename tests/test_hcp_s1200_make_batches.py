from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.hcp_s1200_make_batches import build_batches


class HCPS1200BatchPlanTest(unittest.TestCase):
    def test_fixed_subject_count_batches(self) -> None:
        subjects = ["s1", "s2", "s3", "s4", "s5"]
        sizes = {subject: 1.0 for subject in subjects}
        self.assertEqual(
            build_batches(subjects, sizes, batch_size=2, max_batch_download_gb=0),
            [["s1", "s2"], ["s3", "s4"], ["s5"]],
        )

    def test_download_size_cap_splits_early(self) -> None:
        subjects = ["s1", "s2", "s3"]
        sizes = {"s1": 3.0, "s2": 3.0, "s3": 1.0}
        self.assertEqual(
            build_batches(subjects, sizes, batch_size=10, max_batch_download_gb=5.0),
            [["s1"], ["s2", "s3"]],
        )


if __name__ == "__main__":
    unittest.main()
