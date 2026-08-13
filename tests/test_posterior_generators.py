from __future__ import annotations

import unittest

import torch

from models.generators import (
    DiffusionConfig,
    DriftingConfig,
    PosteriorBackboneConfig,
    build_posterior_generator,
    drifting_field_loss,
)


class PosteriorGeneratorTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.backbone = PosteriorBackboneConfig(
            latent_channels=3,
            condition_channels=4,
            hidden_channels=8,
            depth=2,
            num_modalities=5,
            num_targets=5,
            quality_channels=2,
            embedding_dim=16,
        )
        self.target = torch.randn(2, 3, 4, 4, 4)
        self.condition = torch.randn(2, 4, 4, 4, 4)
        self.mask = torch.tensor([[1, 1, 0, 0, 0], [1, 0, 1, 1, 0.0]])
        self.target_ids = torch.tensor([3, 4])
        self.quality = torch.rand(2, 2, 4, 4, 4)

    def test_official_style_drifting_loss_has_gradient(self) -> None:
        generated = torch.randn(2, 4, 32, requires_grad=True)
        positives = torch.randn(2, 2, 32)
        negatives = torch.randn(2, 3, 32)
        loss, info = drifting_field_loss(generated, positives, negatives)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(generated.grad.abs().sum().item(), 0)
        self.assertIn("scale", info)

    def test_diffusion_training_and_sampling_contract(self) -> None:
        model = build_posterior_generator(
            "diffusion",
            self.backbone,
            diffusion=DiffusionConfig(
                timesteps=8, sampling_steps=3, consistency_weight=0.1
            ),
        )
        output = model.training_loss(
            self.target,
            self.condition,
            modality_mask=self.mask,
            target_ids=self.target_ids,
            quality=self.quality,
            consistency_target=self.target,
            consistency_mask=torch.ones(2, 1, 4, 4, 4),
        )
        output["loss"].backward()
        self.assertTrue(torch.isfinite(output["loss"]))
        samples_a = model.sample(
            self.condition,
            n_samples=2,
            seed=11,
            modality_mask=self.mask,
            target_ids=self.target_ids,
            quality=self.quality,
        )
        samples_b = model.sample(
            self.condition,
            n_samples=2,
            seed=11,
            modality_mask=self.mask,
            target_ids=self.target_ids,
            quality=self.quality,
        )
        self.assertEqual(samples_a.shape, (2, 2, 3, 4, 4, 4))
        self.assertTrue(torch.equal(samples_a, samples_b))
        self.assertEqual(model.nfe, 3)

    def test_drifting_training_and_sampling_contract(self) -> None:
        model = build_posterior_generator(
            "drifting",
            self.backbone,
            drifting=DriftingConfig(
                samples_per_condition=3,
                feature_grid=2,
                consistency_weight=0.1,
            ),
        )
        negatives = self.target.flip(0)
        output = model.training_loss(
            self.target,
            self.condition,
            modality_mask=self.mask,
            target_ids=self.target_ids,
            quality=self.quality,
            negative_latents=negatives,
            consistency_target=self.target,
            consistency_mask=torch.ones(2, 1, 4, 4, 4),
        )
        output["loss"].backward()
        self.assertTrue(torch.isfinite(output["loss"]))
        gradient_sum = sum(
            parameter.grad.abs().sum().item()
            for parameter in model.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(gradient_sum, 0)
        samples = model.sample(
            self.condition,
            n_samples=2,
            seed=13,
            modality_mask=self.mask,
            target_ids=self.target_ids,
            quality=self.quality,
        )
        self.assertEqual(samples.shape, (2, 2, 3, 4, 4, 4))
        self.assertEqual(model.nfe, 1)

    def test_missing_modalities_are_metadata_not_observed_tokens(self) -> None:
        model = build_posterior_generator(
            "diffusion",
            self.backbone,
            diffusion=DiffusionConfig(timesteps=4, sampling_steps=1),
        )
        with self.assertRaisesRegex(ValueError, "modality_mask"):
            model.sample(self.condition, modality_mask=torch.ones(2, 4))


if __name__ == "__main__":
    unittest.main()
