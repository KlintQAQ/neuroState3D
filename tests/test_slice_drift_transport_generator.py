from __future__ import annotations

import csv

import numpy as np
import torch

from models.slice_drift_transport_generator import (
    SliceDriftTransportGenerator,
    SliceDriftTransportGeneratorConfig,
)
from scripts.train_slice_virtual_modality_drifting import (
    BraTSSliceDataset,
    MedicalDriftBank2D,
    selection_score,
    transport_path_loss,
    transport_velocity_loss,
)
from scripts.visualize_slice_virtual_modality_generation import context_slice_batch


def test_transport_generator_outputs_iterative_state_shapes() -> None:
    model = SliceDriftTransportGenerator(
        SliceDriftTransportGeneratorConfig(
            in_modalities=12,
            hidden_channels=8,
            transport_steps=3,
            class_conditioned=True,
            class_channels=3,
        )
    )
    image = torch.randn(2, 12, 24, 24)
    mask = torch.ones(2, 12)
    mask[:, 3:6] = 0.0
    class_condition = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    output = model(image, mask, class_condition)

    assert output["synthetic"].shape == (2, 1, 24, 24)
    assert output["uncertainty"].shape == (2, 1, 24, 24)
    assert output["drift_states"].shape == (2, 4, 1, 24, 24)
    assert output["drift_velocities"].shape == (2, 3, 1, 24, 24)


def test_gated_refinement_preserves_stage1_shapes() -> None:
    model = SliceDriftTransportGenerator(
        SliceDriftTransportGeneratorConfig(
            in_modalities=4,
            hidden_channels=8,
            transport_steps=2,
            gated_refinement=True,
            refinement_residual_scale=0.3,
            refinement_acceptance_gate=True,
            refinement_channels_multiplier=2,
            refinement_blocks=2,
        )
    )
    image = torch.randn(2, 4, 20, 20)
    mask = torch.ones(2, 4)
    mask[:, 1] = 0.0

    output = model(image, mask)

    assert output["stage1_synthetic"].shape == (2, 1, 20, 20)
    assert output["refinement_gate"].shape == (2, 1, 20, 20)
    assert output["refinement_gate_logits"].shape == (2, 1, 20, 20)
    assert output["refinement_region_gate"].shape == (2, 1, 20, 20)
    assert output["refinement_acceptance"].shape == (2, 1, 20, 20)
    assert output["refinement_acceptance_logits"].shape == (2, 1, 20, 20)
    assert output["refinement_residual"].shape == (2, 1, 20, 20)
    assert output["synthetic"].shape == (2, 1, 20, 20)
    assert output["refinement_gate"].min() >= 0.0
    assert output["refinement_gate"].max() <= 1.0
    assert output["refinement_acceptance"].min() >= 0.0
    assert output["refinement_acceptance"].max() <= 1.0


def test_detail_feature_refinement_accepts_context_channels() -> None:
    model = SliceDriftTransportGenerator(
        SliceDriftTransportGeneratorConfig(
            in_modalities=12,
            hidden_channels=8,
            transport_steps=2,
            gated_refinement=True,
            refinement_acceptance_gate=True,
            refinement_detail_features=True,
            refinement_channels_multiplier=2,
            refinement_blocks=2,
        )
    )
    image = torch.randn(2, 12, 20, 20)
    mask = torch.ones(2, 12)
    mask[:, 3:6] = 0.0

    output = model(image, mask)

    assert output["synthetic"].shape == (2, 1, 20, 20)
    assert output["stage1_synthetic"].shape == (2, 1, 20, 20)
    assert output["refinement_gate"].shape == (2, 1, 20, 20)
    assert output["refinement_acceptance"].shape == (2, 1, 20, 20)
    assert output["refinement_residual"].shape == (2, 1, 20, 20)


def test_transport_losses_backpropagate_to_velocity_head() -> None:
    model = SliceDriftTransportGenerator(
        SliceDriftTransportGeneratorConfig(
            in_modalities=4,
            hidden_channels=8,
            transport_steps=2,
        )
    )
    image = torch.randn(1, 4, 16, 16)
    mask = torch.tensor([[1.0, 0.0, 1.0, 1.0]])
    target = image[:, 1:2]
    focus = torch.ones_like(target)

    output = model(image, mask)
    loss = (
        output["synthetic"].sub(target).abs().mean()
        + transport_velocity_loss(output, target, focus, step_scale=1.0)
        + transport_path_loss(output, target, focus)
    )
    loss.backward()

    grad = model.velocity_head.weight.grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert grad.abs().sum() > 0


def test_stochastic_initial_state_is_controllable_and_eval_is_deterministic() -> None:
    model = SliceDriftTransportGenerator(
        SliceDriftTransportGeneratorConfig(
            in_modalities=4,
            hidden_channels=8,
            transport_steps=2,
            medical_role_conditioning=True,
            learned_initial_state=True,
            stochastic_initial_state=True,
            stochastic_noise_scale=0.1,
        )
    )
    image = torch.randn(1, 4, 16, 16)
    mask = torch.tensor([[1.0, 0.0, 1.0, 1.0]])

    model.eval()
    deterministic_a = model(image, mask)
    deterministic_b = model(image, mask)
    assert torch.equal(deterministic_a["synthetic"], deterministic_b["synthetic"])
    assert deterministic_a["initial_sigma"].min() > 0

    noise_a = torch.zeros(1, 1, 16, 16)
    noise_b = torch.ones(1, 1, 16, 16)
    stochastic_a = model(image, mask, stochastic=True, initial_noise=noise_a)
    stochastic_b = model(image, mask, stochastic=True, initial_noise=noise_b)
    assert not torch.equal(stochastic_a["drift_initial"], stochastic_b["drift_initial"])


def test_training_slice_sampling_changes_by_epoch_but_is_reproducible(tmp_path) -> None:
    image_path = tmp_path / "image.npy"
    seg_path = tmp_path / "seg.npy"
    manifest_path = tmp_path / "manifest.csv"
    np.save(image_path, np.zeros((4, 8, 8, 8), dtype=np.float32))
    seg = np.zeros((8, 8, 8), dtype=np.int64)
    seg[3, 3, :] = 1
    np.save(seg_path, seg)
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("dataset", "subject_id", "multimodal_path", "seg_path"),
        )
        writer.writeheader()
        writer.writerow(
            {
                "dataset": "GLI",
                "subject_id": "subject-1",
                "multimodal_path": str(image_path),
                "seg_path": str(seg_path),
            }
        )
    dataset = BraTSSliceDataset(
        manifest_path,
        spatial_size=8,
        max_subjects=None,
        slices_per_subject=16,
        target_modality="t1c",
        seed=46,
    )

    epoch_zero = [dataset[index]["slice_index"] for index in range(len(dataset))]
    dataset.set_epoch(1)
    epoch_one = [dataset[index]["slice_index"] for index in range(len(dataset))]
    dataset.set_epoch(0)
    epoch_zero_repeat = [dataset[index]["slice_index"] for index in range(len(dataset))]

    assert epoch_zero != epoch_one
    assert epoch_zero == epoch_zero_repeat


def test_medical_bank_batches_hierarchical_updates_without_changing_fifo_contents() -> None:
    bank = MedicalDriftBank2D(max_tokens=100, max_add_tokens=100, memory_tokens=3)
    tokens = torch.arange(2 * 5 * 3, dtype=torch.float32).view(2, 5, 3)
    batch = {
        "dataset": ["GLI", "MEN"],
        "dataset_id": torch.tensor([0, 1]),
        "region_bin": torch.tensor([3, 2]),
        "lesion_area_bin": torch.tensor([1, 2]),
        "enhancement_bin": torch.tensor([2, 1]),
        "z_bin": torch.tensor([1, 4]),
    }

    suffixes = bank.batch_condition_suffixes(batch, batch_size=2)
    descriptors = {"energy": tokens}
    bank.update(
        "t1c",
        descriptors,
        {"energy": tokens + 100.0},
        batch,
        suffixes,
    )

    expected_positive = tokens.flatten(0, 1)
    assert torch.equal(bank.positive._storage["t1c:energy"], expected_positive)
    assert torch.equal(bank.positive._storage["t1c:energy:global"], expected_positive)
    assert bank.positive.count("t1c:energy:ds=GLI:r=ET:z=1:e=high:a=small") == 5
    assert bank.positive.count("t1c:energy:ds=MEN:r=TC:z=4:e=mid:a=medium") == 5

    sampled = bank.sample_positive(
        "t1c",
        "energy",
        batch_size=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
        batch=batch,
        condition_suffixes=suffixes,
    )
    assert sampled is not None
    assert sampled.shape == (2, 3, 3)


def test_context_slice_batch_matches_expected_channel_order() -> None:
    image = torch.arange(4 * 5 * 6 * 7, dtype=torch.float32).view(4, 5, 6, 7)
    batch = context_slice_batch(image, start=2, end=4, radius=1)

    expected = []
    for z in (2, 3):
        channels = []
        for modality_index in range(4):
            for offset in (-1, 0, 1):
                channels.append(image[modality_index, :, :, z + offset])
        expected.append(torch.stack(channels, dim=0))
    expected_tensor = torch.stack(expected, dim=0)

    assert batch.shape == (2, 12, 5, 6)
    assert torch.equal(batch, expected_tensor)


def test_noharm_selection_penalizes_stage1_regression() -> None:
    shared = {
        "mae": 0.06,
        "ssim": 0.7,
        "high_tumor_mae": 0.20,
        "high_tumor_under": 0.14,
        "mae_ET": 0.20,
        "mae_TC": 0.21,
        "mae_WT": 0.18,
        "edge_mae": 0.02,
        "laplacian_mae": 0.01,
        "ssim_ET": 0.4,
        "ssim_TC": 0.4,
        "ssim_WT": 0.4,
    }
    safe = {
        **shared,
        "stage1_mae": 0.061,
        "mae_ET_delta_from_stage1": -0.01,
        "mae_TC_delta_from_stage1": -0.01,
        "mae_WT_delta_from_stage1": -0.01,
        "high_tumor_mae_delta_from_stage1": -0.01,
        "background_harm_rate": 0.02,
        "background_delta_from_stage1": 0.001,
    }
    harmful = {
        **shared,
        "stage1_mae": 0.052,
        "mae_ET_delta_from_stage1": 0.04,
        "mae_TC_delta_from_stage1": 0.03,
        "mae_WT_delta_from_stage1": 0.02,
        "high_tumor_mae_delta_from_stage1": 0.05,
        "background_harm_rate": 0.20,
        "background_delta_from_stage1": 0.020,
    }

    assert selection_score(harmful, "lesion_noharm_composite") > selection_score(
        safe,
        "lesion_noharm_composite",
    )
