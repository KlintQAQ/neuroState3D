from __future__ import annotations

import torch

from models.slice_drift_transport_generator import (
    SliceDriftTransportGenerator,
    SliceDriftTransportGeneratorConfig,
)
from scripts.train_slice_virtual_modality_drifting import (
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
