from __future__ import annotations

import unittest
from argparse import Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.drifting_imputation_fusion import (
    DriftingImputationFusion,
    DriftingImputationFusionConfig,
)
from models.evidence_fusion import EvidenceFusionConfig, EvidenceReliableFusion
from models.generators import (
    DriftingConfig,
    PosteriorBackboneConfig,
    build_posterior_generator,
)
from scripts.train_drifting_imputation_fusion import (
    evaluate,
    set_trainable,
    train_generator_epoch,
    trainable_parameters,
    warmup_fusion_epoch,
)


class TinyEncoder(nn.Module):
    def __init__(self, channels: int = 4) -> None:
        super().__init__()
        self.projection = nn.Conv3d(1, channels, kernel_size=3, padding=1)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"stage4": F.avg_pool3d(self.projection(image), kernel_size=2)}


class DriftingImputationFusionTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(17)
        fusion_config = EvidenceFusionConfig(
            feature_stage="stage4",
            feature_channels=4,
            hidden_channels=4,
            state_embedding_dim=2,
        )
        fusion = EvidenceReliableFusion(fusion_config, encoder=TinyEncoder())
        backbone = PosteriorBackboneConfig(
            latent_channels=4,
            condition_channels=4,
            hidden_channels=8,
            depth=1,
            num_modalities=4,
            num_targets=4,
            embedding_dim=16,
        )
        drifting = build_posterior_generator(
            "drifting",
            backbone,
            drifting=DriftingConfig(samples_per_condition=2, feature_grid=2),
        )
        self.model = DriftingImputationFusion(
            fusion,
            drifting,
            DriftingImputationFusionConfig(
                samples_per_target=2,
                decoder_hidden_channels=4,
            ),
        )
        self.images = torch.randn(2, 4, 8, 8, 8)
        self.mask = torch.tensor(
            [[1.0, 0.0, 1.0, 1.0], [1.0, 1.0, 0.0, 1.0]]
        )
        self.state = torch.zeros(2, 4, dtype=torch.long)
        self.state[self.mask <= 0] = 1

    def test_training_completes_missing_slots_before_fusion(self) -> None:
        output = self.model(
            self.images,
            self.mask,
            self.state,
            compute_generator_loss=True,
            seed=9,
        )
        self.assertEqual(output["logits"].shape, (2, 3, 8, 8, 8))
        self.assertEqual(output["missing_pair_count"], 2)
        self.assertTrue(torch.equal(output["effective_modality_mask"], torch.ones_like(self.mask)))
        self.assertTrue(torch.all(output["modality_state"][self.mask <= 0] == 3))
        self.assertTrue(torch.all(output["imputation_confidence"][self.mask <= 0] < 1))
        # The Drifting output projection starts collapsed at zero. Confidence
        # must still stay moderate because the learned aleatoric scale prevents
        # zero sample variance from masquerading as certainty.
        self.assertTrue(torch.all(output["imputation_confidence"][self.mask <= 0] < 0.8))
        self.assertFalse(output["imputation_confidence"].requires_grad)
        self.assertEqual(output["imputation_confidence_map"].shape, self.images.shape)
        self.assertFalse(output["imputation_confidence_map"].requires_grad)
        self.assertTrue(torch.isfinite(output["uncertainty_nll_loss"]))
        self.assertTrue(torch.isfinite(output["uncertainty_calibration_loss"]))
        observed = self.mask.bool().view(2, 4, 1, 1, 1).expand_as(self.images)
        self.assertTrue(
            torch.equal(output["imputed_images"][observed], self.images[observed])
        )
        loss = output["generator_loss"] + output["logits"].square().mean()
        loss.backward()
        gradient = sum(
            parameter.grad.abs().sum().item()
            for parameter in self.model.drifting.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(gradient, 0)
        uncertainty_gradient = sum(
            parameter.grad.abs().sum().item()
            for decoder in self.model.image_decoders
            for parameter in decoder.uncertainty_head.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(uncertainty_gradient, 0)

    def test_inference_overwrites_zero_placeholders(self) -> None:
        inference_images = self.images.clone()
        inference_images[self.mask <= 0] = 0
        self.model.eval()
        with torch.no_grad():
            output = self.model(
                inference_images,
                self.mask,
                self.state,
                n_samples=3,
                seed=4,
            )
        generated = output["imputed_images"][self.mask <= 0]
        self.assertGreater(generated.abs().sum().item(), 0)
        self.assertEqual(output["imputation_uncertainty"].shape, self.mask.shape)
        self.assertTrue(torch.all(output["effective_modality_mask"] == 1))

    def test_hidden_ground_truth_cannot_leak_into_inference(self) -> None:
        first = self.images.clone()
        second = self.images.clone()
        first[self.mask <= 0] = 0
        second[self.mask <= 0] = torch.randn_like(second[self.mask <= 0]) * 100
        self.model.eval()
        with torch.no_grad():
            output_a = self.model(first, self.mask, self.state, n_samples=2, seed=8)
            output_b = self.model(second, self.mask, self.state, n_samples=2, seed=8)
        self.assertTrue(torch.equal(output_a["logits"], output_b["logits"]))
        self.assertTrue(
            torch.equal(output_a["imputed_images"], output_b["imputed_images"])
        )

    def test_no_missing_modalities_preserves_fusion_contract(self) -> None:
        mask = torch.ones_like(self.mask)
        state = torch.zeros_like(self.state)
        output = self.model(self.images, mask, state)
        self.assertEqual(output["missing_pair_count"], 0)
        self.assertEqual(float(output["generator_loss"]), 0.0)
        self.assertTrue(torch.equal(output["imputed_images"], self.images))

    def test_staged_training_and_three_way_validation_contract(self) -> None:
        batch = {
            "image": self.images,
            "target": (torch.rand(2, 3, 8, 8, 8) > 0.8).float(),
            "modality_mask": self.mask,
            "modality_state": self.state,
        }
        fusion_optimizer = torch.optim.AdamW(
            trainable_parameters(self.model.fusion), lr=1e-4
        )
        warmup = warmup_fusion_epoch(
            self.model, [batch], fusion_optimizer, "cpu"
        )
        self.assertTrue(torch.isfinite(torch.tensor(warmup["loss"])))

        set_trainable(self.model.fusion, False)
        args = Namespace(
            device="cpu",
            seed=12,
            samples=2,
            freeze_fusion_during_generator=True,
            segmentation_weight=1.0,
            generator_weight=1.0,
        )
        generator_optimizer = torch.optim.AdamW(
            trainable_parameters(self.model), lr=1e-4
        )
        training = train_generator_epoch(
            self.model, [batch], generator_optimizer, args, epoch=0
        )
        self.assertTrue(torch.isfinite(torch.tensor(training["loss"])))
        report = evaluate(self.model, [batch], args)
        self.assertIn("full_real", report)
        self.assertIn("missing", report)
        self.assertIn("drifting_filled", report)
        self.assertIn("delta_filled_vs_missing", report)


if __name__ == "__main__":
    unittest.main()
