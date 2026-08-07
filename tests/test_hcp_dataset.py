from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.hcp_dataset import HCPDataset, hcp_collate
from preprocessing.spatial_utils import compare_nifti_space, inspect_nifti


class HCPDatasetSyntheticNiftiTest(unittest.TestCase):
    """SYNTHETIC NIFTI TEST ONLY: not a real HCP validation."""

    def test_manifest_dataset_and_dataloader(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw" / "sub-0001"
            raw.mkdir(parents=True)
            affine = np.diag([1.0, 1.0, 1.0, 1.0])
            t1 = raw / "T1w.nii.gz"
            t2 = raw / "T2w.nii.gz"
            rng = np.random.default_rng(46)
            nib.save(
                nib.Nifti1Image(rng.normal(size=(32, 34, 36)).astype(np.float32), affine),
                t1,
            )
            nib.save(
                nib.Nifti1Image(rng.normal(size=(32, 34, 36)).astype(np.float32), affine),
                t2,
            )
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "release": "SYNTHETIC NIFTI TEST ONLY",
                        "data_root": str(root),
                        "subjects": [
                            {
                                "subject_id": "sub-0001",
                                "modalities": {
                                    "t1": "raw/sub-0001/T1w.nii.gz",
                                    "t2": "raw/sub-0001/T2w.nii.gz",
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            t1_audit = inspect_nifti(t1)
            t2_audit = inspect_nifti(t2)
            self.assertEqual(
                compare_nifti_space(t1_audit, t2_audit)["T1_T2_SAME_PHYSICAL_SPACE"],
                "TRUE",
            )

            dataset = HCPDataset(manifest, roi_size=16)
            sample = dataset[0]
            self.assertEqual(sample["subject_id"], "sub-0001")
            self.assertEqual(sample["modality_mask"].tolist(), [1.0, 1.0, 0.0, 0.0, 0.0])
            self.assertEqual(list(sample["modalities"]["t1"].shape), [1, 16, 16, 16])
            self.assertEqual(list(sample["modalities"]["t2"].shape), [1, 16, 16, 16])
            self.assertIsNone(sample["modalities"]["fa"])
            self.assertTrue(torch.isfinite(sample["modalities"]["t1"]).all())
            self.assertTrue(torch.isfinite(sample["modalities"]["t2"]).all())

            loader = DataLoader(dataset, batch_size=1, collate_fn=hcp_collate)
            batch = next(iter(loader))
            self.assertEqual(list(batch["modalities"]["t1"].shape), [1, 1, 16, 16, 16])
            self.assertEqual(list(batch["modalities"]["t2"].shape), [1, 1, 16, 16, 16])
            self.assertEqual(list(batch["modality_mask"].shape), [1, 5])

    def test_space_mismatch_is_detected_before_fusion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw" / "sub-0002"
            raw.mkdir(parents=True)
            t1 = raw / "T1w.nii.gz"
            t2 = raw / "T2w.nii.gz"
            nib.save(
                nib.Nifti1Image(np.ones((16, 16, 16), dtype=np.float32), np.eye(4)),
                t1,
            )
            affine = np.diag([2.0, 2.0, 2.0, 1.0])
            nib.save(
                nib.Nifti1Image(np.ones((16, 16, 16), dtype=np.float32), affine),
                t2,
            )
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "release": "SYNTHETIC NIFTI TEST ONLY",
                        "data_root": str(root),
                        "subjects": [
                            {
                                "subject_id": "sub-0002",
                                "modalities": {
                                    "t1": "raw/sub-0002/T1w.nii.gz",
                                    "t2": "raw/sub-0002/T2w.nii.gz",
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            t1_audit = inspect_nifti(t1)
            t2_audit = inspect_nifti(t2)
            self.assertEqual(
                compare_nifti_space(t1_audit, t2_audit)["T1_T2_SAME_PHYSICAL_SPACE"],
                "FALSE",
            )
            with self.assertRaisesRegex(ValueError, "not in the same physical space"):
                _ = HCPDataset(manifest, roi_size=16)[0]


if __name__ == "__main__":
    unittest.main()
