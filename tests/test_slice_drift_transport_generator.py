from __future__ import annotations

import torch

from models.slice_drift_transport_generator import (
    SliceDriftTransportGenerator,
    SliceDriftTransportGeneratorConfig,
)
from scripts.train_slice_virtual_modality_drifting import (
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
