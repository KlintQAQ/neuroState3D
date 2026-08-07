from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from models.modality_adapter import MODALITY_ORDER
from preprocessing.normalization import preprocess_nifti_for_brainmvp
from preprocessing.spatial_utils import compare_nifti_space, inspect_nifti


class HCPDataset(Dataset):
    """Manifest-driven HCP T1/T2 dataset for engineering smoke tests.

    The manifest uses a data root plus relative paths. It must not contain
    passwords, tokens, local user account paths, or absolute subject paths.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        roi_size: int = 96,
        require_same_space: bool = True,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            self.manifest = json.load(handle)
        self.data_root = Path(self.manifest["data_root"])
        self.subjects = list(self.manifest.get("subjects", []))
        self.roi_size = roi_size
        self.require_same_space = require_same_space

    def __len__(self) -> int:
        return len(self.subjects)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.subjects[index]
        subject_id = str(item["subject_id"])
        modalities = item.get("modalities", {})
        t1_path = self._resolve_required(modalities, "t1", subject_id)
        t2_path = self._resolve_required(modalities, "t2", subject_id)

        t1_audit = inspect_nifti(t1_path)
        t2_audit = inspect_nifti(t2_path)
        spatial = compare_nifti_space(t1_audit, t2_audit)
        if self.require_same_space and spatial["T1_T2_SAME_PHYSICAL_SPACE"] == "FALSE":
            raise ValueError(
                f"T1/T2 are not in the same physical space for subject {subject_id}: {spatial}"
            )

        t1_tensor, t1_preprocess = preprocess_nifti_for_brainmvp(t1_path, self.roi_size)
        t2_tensor, t2_preprocess = preprocess_nifti_for_brainmvp(t2_path, self.roi_size)
        if t1_tensor.shape != t2_tensor.shape:
            raise ValueError(
                f"Preprocessed T1/T2 shape mismatch for {subject_id}: "
                f"{tuple(t1_tensor.shape)} vs {tuple(t2_tensor.shape)}"
            )

        modality_tensors: dict[str, torch.Tensor | None] = {
            "t1": t1_tensor,
            "t2": t2_tensor,
            "fa": None,
            "md": None,
            "alff": None,
        }
        mask = torch.tensor(
            [1.0 if modality_tensors[name] is not None else 0.0 for name in MODALITY_ORDER],
            dtype=torch.float32,
        )
        return {
            "subject_id": subject_id,
            "modalities": modality_tensors,
            "modality_mask": mask,
            "metadata": {
                "t1": t1_audit.__dict__,
                "t2": t2_audit.__dict__,
                "spatial_consistency": spatial,
                "preprocessing": {
                    "t1": t1_preprocess,
                    "t2": t2_preprocess,
                },
            },
        }

    def _resolve_required(
        self,
        modalities: dict[str, str],
        modality: str,
        subject_id: str,
    ) -> Path:
        relative = modalities.get(modality)
        if not relative:
            raise KeyError(f"Subject {subject_id} is missing {modality} in manifest.")
        path = self.data_root / relative
        if not path.exists():
            raise FileNotFoundError(f"Missing {modality} file for subject {subject_id}: {path}")
        return path


def hcp_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if not batch:
        raise ValueError("Cannot collate an empty HCP batch.")
    masks = torch.stack([item["modality_mask"] for item in batch], dim=0)
    if torch.any(masks.sum(dim=1) <= 0):
        raise ValueError("Every HCP subject must have at least one observed modality.")

    collated_modalities: dict[str, torch.Tensor | None] = {}
    for modality in MODALITY_ORDER:
        tensors = [item["modalities"][modality] for item in batch]
        if all(tensor is None for tensor in tensors):
            collated_modalities[modality] = None
        elif any(tensor is None for tensor in tensors):
            raise ValueError(
                f"Mixed present/missing {modality} within a batch is not supported yet."
            )
        else:
            collated_modalities[modality] = torch.stack(
                [tensor for tensor in tensors if tensor is not None],
                dim=0,
            )

    return {
        "subject_id": [item["subject_id"] for item in batch],
        "modalities": collated_modalities,
        "modality_mask": masks,
        "metadata": [item["metadata"] for item in batch],
    }

