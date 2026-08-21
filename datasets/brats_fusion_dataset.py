from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


BRATS_MODALITIES = ("t1n", "t1c", "t2w", "t2f")
BRATS_REGIONS = ("ET", "TC", "WT")


def brats_region_targets(seg: torch.Tensor) -> torch.Tensor:
    """Convert BraTS integer labels to ET/TC/WT multilabel targets.

    The local preprocessed masks use labels:
    0 background, 1 non-enhancing/necrotic core, 2 edema, 3 enhancing tumor.
    """

    et = seg == 3
    tc = (seg == 1) | (seg == 3)
    wt = seg > 0
    return torch.stack([et, tc, wt], dim=0).to(torch.float32)


def modality_mask_from_names(
    names: Iterable[str],
    modality_order: Sequence[str] = BRATS_MODALITIES,
) -> torch.Tensor:
    selected = {name.lower() for name in names}
    return torch.tensor(
        [1.0 if name in selected else 0.0 for name in modality_order],
        dtype=torch.float32,
    )


class BraTSFusionDataset(Dataset):
    """Model-ready BraTS2023 dataset for missing-modality fusion experiments.

    Each item contains four MRI modalities in fixed order, multilabel BraTS
    region targets, and a sampled modality-availability mask. Missing modalities
    are not removed from the tensor; the mask tells the model which slots are
    observable so the same batch shape can represent all 15 non-empty
    combinations.
    """

    def __init__(
        self,
        manifest_csv: str | Path,
        spatial_size: int | Sequence[int] | None = None,
        max_subjects: int | None = None,
        datasets: Sequence[str] | None = None,
        modality_mask_mode: str = "all",
        fixed_modalities: Sequence[str] | None = None,
        degrade_prob: float = 0.0,
        foreground_crop_prob: float = 0.0,
        crop_mode: str = "foreground",
        region_crop_weights: Mapping[str, float] | None = None,
        seed: int = 46,
    ) -> None:
        self.manifest_csv = Path(manifest_csv)
        self.spatial_size = self._normalize_spatial_size(spatial_size)
        self.modality_mask_mode = modality_mask_mode
        self.fixed_modalities = (
            tuple(name.lower() for name in fixed_modalities)
            if fixed_modalities is not None
            else None
        )
        self.degrade_prob = float(degrade_prob)
        self.foreground_crop_prob = float(foreground_crop_prob)
        self.crop_mode = crop_mode
        self.region_crop_weights = dict(
            region_crop_weights
            or {
                "ET": 0.45,
                "TC": 0.30,
                "WT": 0.20,
                "foreground": 0.05,
            }
        )
        self.seed = int(seed)
        self.epoch = 0

        rows = self._read_rows(self.manifest_csv)
        if datasets is not None:
            keep = {name.upper() for name in datasets}
            rows = [row for row in rows if row["dataset"].upper() in keep]
        if max_subjects is not None:
            rows = rows[: int(max_subjects)]
        if not rows:
            raise ValueError(f"No BraTS rows selected from {self.manifest_csv}")
        self.rows = rows

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.rows[index]
        image = torch.from_numpy(
            np.load(row["multimodal_path"]).astype(np.float32, copy=True)
        )
        seg = torch.from_numpy(np.load(row["seg_path"]).astype(np.int64, copy=True))
        rng = random.Random(self.seed + self.epoch * 100_003 + index * 9176)

        if image.shape[0] != len(BRATS_MODALITIES):
            raise ValueError(
                f"Expected {len(BRATS_MODALITIES)} modalities, got {tuple(image.shape)}"
            )
        if (
            self.spatial_size is not None
            and self.foreground_crop_prob > 0
            and rng.random() < self.foreground_crop_prob
        ):
            image, seg = self._region_crop(
                image,
                seg,
                self.spatial_size,
                rng,
                self.crop_mode,
                self.region_crop_weights,
            )
        if self.spatial_size is not None:
            image = F.interpolate(
                image.unsqueeze(0),
                size=self.spatial_size,
                mode="trilinear",
                align_corners=False,
            ).squeeze(0)
            seg = F.interpolate(
                seg.unsqueeze(0).unsqueeze(0).to(torch.float32),
                size=self.spatial_size,
                mode="nearest",
            ).squeeze(0).squeeze(0).to(torch.int64)

        mask = self._sample_mask(index)
        modality_state = torch.zeros(len(BRATS_MODALITIES), dtype=torch.long)
        modality_state[mask <= 0] = 1

        if self.degrade_prob > 0:
            for modality_index in range(len(BRATS_MODALITIES)):
                if mask[modality_index] <= 0:
                    continue
                if rng.random() < self.degrade_prob:
                    image[modality_index] = self._degrade_channel(
                        image[modality_index], rng
                    )
                    modality_state[modality_index] = 2

        target = brats_region_targets(seg)
        return {
            "image": image,
            "target": target,
            "seg": seg,
            "modality_mask": mask,
            "modality_state": modality_state,
            "dataset": row["dataset"],
            "subject_id": row["subject_id"],
            "modalities": BRATS_MODALITIES,
            "regions": BRATS_REGIONS,
        }

    def _sample_mask(self, index: int) -> torch.Tensor:
        mode = self.modality_mask_mode
        if mode == "all":
            return torch.ones(len(BRATS_MODALITIES), dtype=torch.float32)
        if mode == "fixed":
            if not self.fixed_modalities:
                raise ValueError("fixed modality_mask_mode requires fixed_modalities")
            return modality_mask_from_names(self.fixed_modalities)
        if mode == "single_random":
            rng = random.Random(self.seed + self.epoch * 100_003 + index * 3571)
            selected = rng.randrange(len(BRATS_MODALITIES))
            mask = torch.zeros(len(BRATS_MODALITIES), dtype=torch.float32)
            mask[selected] = 1.0
            return mask
        if mode == "random_nonempty":
            rng = random.Random(self.seed + self.epoch * 100_003 + index * 7919)
            mask = torch.tensor(
                [1.0 if rng.random() < 0.5 else 0.0 for _ in BRATS_MODALITIES],
                dtype=torch.float32,
            )
            if mask.sum() <= 0:
                mask[rng.randrange(len(BRATS_MODALITIES))] = 1.0
            return mask
        raise ValueError(
            "modality_mask_mode must be one of: all, fixed, single_random, random_nonempty"
        )

    @staticmethod
    def _degrade_channel(channel: torch.Tensor, rng: random.Random) -> torch.Tensor:
        degraded = channel.clone()
        choice = rng.choice(("noise", "scale_shift", "blur"))
        if choice == "noise":
            std = float(degraded.std().clamp_min(1e-3).item())
            degraded = degraded + torch.randn_like(degraded) * (0.08 * std)
        elif choice == "scale_shift":
            scale = rng.uniform(0.65, 1.35)
            shift = rng.uniform(-0.15, 0.15)
            degraded = degraded * scale + shift
        else:
            degraded = F.avg_pool3d(
                degraded.unsqueeze(0).unsqueeze(0),
                kernel_size=3,
                stride=1,
                padding=1,
            ).squeeze(0).squeeze(0)
        return degraded

    @staticmethod
    def _region_crop(
        image: torch.Tensor,
        seg: torch.Tensor,
        crop_size: tuple[int, int, int],
        rng: random.Random,
        crop_mode: str = "foreground",
        region_crop_weights: Mapping[str, float] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        spatial = tuple(int(item) for item in seg.shape)
        if any(crop_size[axis] >= spatial[axis] for axis in range(3)):
            return image, seg
        foreground = BraTSFusionDataset._crop_candidates(
            seg,
            crop_mode,
            region_crop_weights,
            rng,
        )
        if foreground.numel() == 0:
            return image, seg
        center_index = rng.randrange(foreground.shape[0])
        center = foreground[center_index].tolist()
        slices = []
        for axis, size in enumerate(crop_size):
            half = size // 2
            low = int(center[axis]) - half
            high = low + size
            if low < 0:
                low = 0
                high = size
            if high > spatial[axis]:
                high = spatial[axis]
                low = high - size
            slices.append(slice(low, high))
        cropped_image = image[:, slices[0], slices[1], slices[2]]
        cropped_seg = seg[slices[0], slices[1], slices[2]]
        return cropped_image, cropped_seg

    @staticmethod
    def _crop_candidates(
        seg: torch.Tensor,
        crop_mode: str,
        region_crop_weights: Mapping[str, float] | None,
        rng: random.Random,
    ) -> torch.Tensor:
        if crop_mode == "foreground":
            return torch.nonzero(seg > 0, as_tuple=False)
        if crop_mode != "region_balanced":
            raise ValueError("crop_mode must be one of: foreground, region_balanced")
        weights = region_crop_weights or {}
        choices = [
            ("ET", max(float(weights.get("ET", 0.0)), 0.0)),
            ("TC", max(float(weights.get("TC", 0.0)), 0.0)),
            ("WT", max(float(weights.get("WT", 0.0)), 0.0)),
            ("foreground", max(float(weights.get("foreground", 0.0)), 0.0)),
        ]
        total = sum(weight for _, weight in choices)
        if total <= 0:
            choices = [("ET", 0.45), ("TC", 0.30), ("WT", 0.20), ("foreground", 0.05)]
            total = 1.0
        draw = rng.random() * total
        cumulative = 0.0
        selected = "foreground"
        for name, weight in choices:
            cumulative += weight
            if draw <= cumulative:
                selected = name
                break
        region_masks = {
            "ET": seg == 3,
            "TC": (seg == 1) | (seg == 3),
            "WT": seg > 0,
            "foreground": seg > 0,
        }
        for name in [selected, "ET", "TC", "WT", "foreground"]:
            candidates = torch.nonzero(region_masks[name], as_tuple=False)
            if candidates.numel() > 0:
                return candidates
        return torch.empty((0, 3), dtype=torch.long)

    @staticmethod
    def _read_rows(path: Path) -> list[dict[str, str]]:
        with path.open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        required = {"dataset", "subject_id", "multimodal_path", "seg_path"}
        missing = required - set(rows[0]) if rows else required
        if missing:
            raise ValueError(f"Manifest {path} is missing columns: {sorted(missing)}")
        return rows

    @staticmethod
    def _normalize_spatial_size(
        spatial_size: int | Sequence[int] | None,
    ) -> tuple[int, int, int] | None:
        if spatial_size is None:
            return None
        if isinstance(spatial_size, int):
            return (spatial_size, spatial_size, spatial_size)
        values = tuple(int(item) for item in spatial_size)
        if len(values) != 3:
            raise ValueError("spatial_size must be an int or a length-3 sequence")
        return values
