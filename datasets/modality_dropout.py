from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, Mapping, Optional

import torch


@dataclass(frozen=True)
class ModalityMask:
    """Observed modality mask in a fixed modality order."""

    modalities: list[str]
    mask: torch.Tensor

    def observed_for_sample(self, sample_index: int) -> list[str]:
        row = self.mask[sample_index]
        return [m for m, value in zip(self.modalities, row) if bool(value.item())]


def build_modality_mask(
    modalities: Mapping[str, Optional[torch.Tensor]],
    modality_order: list[str],
) -> torch.Tensor:
    """Create a ``[B, M]`` mask from a modality dictionary.

    Missing modalities are represented by ``None`` or absent dictionary keys.
    They are not converted into all-zero image volumes.
    """

    reference = next((v for v in modalities.values() if v is not None), None)
    if reference is None:
        raise ValueError("At least one modality tensor is required to infer batch size.")
    batch_size = reference.shape[0]
    mask = torch.zeros(batch_size, len(modality_order), device=reference.device)
    for index, modality in enumerate(modality_order):
        tensor = modalities.get(modality)
        if tensor is not None:
            if tensor.shape[0] != batch_size:
                raise ValueError(f"Batch size mismatch for modality '{modality}'")
            mask[:, index] = 1
    return mask


class ModalityDropout:
    """Reproducible modality dropout for subject-level multimodal batches."""

    def __init__(
        self,
        modality_order: list[str],
        drop_probability: float = 0.0,
        number_to_drop: Optional[int] = None,
        fixed_keep: Optional[Iterable[str]] = None,
        seed: Optional[int] = None,
    ) -> None:
        if not 0.0 <= drop_probability <= 1.0:
            raise ValueError("drop_probability must be in [0, 1].")
        self.modality_order = [m.lower() for m in modality_order]
        self.drop_probability = drop_probability
        self.number_to_drop = number_to_drop
        self.fixed_keep = {m.lower() for m in fixed_keep} if fixed_keep else None
        self.rng = random.Random(seed)

    def __call__(
        self, modalities: Mapping[str, Optional[torch.Tensor]]
    ) -> tuple[dict[str, Optional[torch.Tensor]], torch.Tensor]:
        output = {m: modalities.get(m) for m in self.modality_order}
        observed = [m for m, tensor in output.items() if tensor is not None]
        if not observed:
            raise ValueError("Cannot apply modality dropout to an empty modality set.")

        if self.fixed_keep is not None:
            keep = set(observed).intersection(self.fixed_keep)
        else:
            keep = set(observed)
            droppable = observed.copy()
            if self.number_to_drop is not None:
                drop_count = min(self.number_to_drop, max(len(droppable) - 1, 0))
                drop = set(self.rng.sample(droppable, drop_count))
            else:
                drop = {
                    m for m in droppable if self.rng.random() < self.drop_probability
                }
                if len(drop) == len(observed):
                    drop.remove(self.rng.choice(observed))
            keep -= drop

        for modality in observed:
            if modality not in keep:
                output[modality] = None
        return output, build_modality_mask(output, self.modality_order)
