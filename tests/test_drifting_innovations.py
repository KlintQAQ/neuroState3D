from __future__ import annotations

import unittest

import torch

from models.generators import (
    DriftingConfig,
    EvidenceContradictionBank,
    PosteriorBackboneConfig,
    build_posterior_generator,
    contradiction_negative_weights,
    evidence_contradiction_score,
    nested_subset_consistency_loss,
)


class DriftingInnovationTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(23)
        self.backbone = PosteriorBackboneConfig(
            latent_channels=2,
            condition_channels=3,
            hidden_channels=8,
            depth=2,
            num_modalities=4,
            num_targets=4,
            embedding_dim=16,
        )
        self.target = torch.randn(2, 2, 4, 4, 4)
        self.condition = torch.randn(2, 3, 4, 4, 4)
        self.coarse_mask = torch.tensor([[1, 0, 1, 0], [1, 0, 1, 0.0]])
        self.richer_mask = torch.tensor([[1, 1, 1, 0], [1, 0, 1, 1.0]])
        self.target_ids = torch.tensor([1, 1])

    def test_contradiction_score_and_weight_rank_bad_candidate(self) -> None:
        candidates = torch.stack(
            [self.target + 0.01, self.target + 2.0], dim=1
        )
        scores = evidence_contradiction_score(candidates, self.target)
        weights = contradiction_negative_weights(scores, strength=2.0)
        self.assertTrue(torch.all(scores[:, 1] > scores[:, 0]))
        self.assertTrue(torch.all(weights[:, 1] > weights[:, 0]))

    def test_contradiction_bank_is_target_keyed(self) -> None:
        bank = EvidenceContradictionBank(capacity=4)
        latents = torch.stack([self.target, self.target + 3.0], dim=1)
        scores = torch.tensor([[0.2, 2.0], [0.3, 3.0]])
        bank.add(latents, scores, target_ids=torch.tensor([1, 2]))
        sampled = bank.sample(
            torch.tensor([1, 2]), n_samples=1, device="cpu", dtype=torch.float32
        )
        self.assertEqual(sampled.latents.shape, (2, 1, 2, 4, 4, 4))
        self.assertEqual(sampled.target_ids.tolist(), [1, 2])
        self.assertAlmostEqual(float(sampled.scores[0, 0]), 2.0)
        self.assertAlmostEqual(float(sampled.scores[1, 0]), 3.0)

    def test_nested_subset_loss_checks_masks_and_contraction(self) -> None:
        coarse = torch.randn(2, 4, 2, 4, 4, 4)
        richer = 2.0 * torch.randn_like(coarse)
        identity, contraction, info = nested_subset_consistency_loss(
            coarse,
            richer,
            self.coarse_mask,
            self.richer_mask,
            pool_size=2,
        )
        self.assertTrue(torch.isfinite(identity))
        self.assertGreater(float(contraction), 0)
        self.assertGreater(float(info["richer_variance"]), float(info["coarse_variance"]))
        with self.assertRaisesRegex(ValueError, "subset"):
            nested_subset_consistency_loss(
                coarse, richer, self.richer_mask, self.coarse_mask
            )

    def _run_mode(self, mode: str) -> dict[str, torch.Tensor]:
        contradiction = mode in {"contradiction", "both"}
        nested = mode in {"nested", "both"}
        model = build_posterior_generator(
            "drifting",
            self.backbone,
            drifting=DriftingConfig(
                samples_per_condition=3,
                feature_grid=2,
                contradiction_repulsion_strength=2.0 if contradiction else 0.0,
                nested_identity_weight=0.5 if nested else 0.0,
                nested_contraction_weight=0.5 if nested else 0.0,
            ),
        )
        kwargs: dict[str, torch.Tensor] = {}
        if contradiction:
            kwargs["negative_latents"] = self.target[:, None] + torch.randn(
                2, 2, 2, 4, 4, 4
            )
            kwargs["negative_scores"] = torch.tensor([[0.2, 2.0], [0.3, 3.0]])
        if nested:
            kwargs["richer_condition"] = self.condition + 0.1
            kwargs["richer_modality_mask"] = self.richer_mask
        output = model.training_loss(
            self.target,
            self.condition,
            modality_mask=self.coarse_mask,
            target_ids=self.target_ids,
            **kwargs,
        )
        output["loss"].backward()
        self.assertTrue(torch.isfinite(output["loss"]))
        return output

    def test_contradiction_only(self) -> None:
        output = self._run_mode("contradiction")
        self.assertGreater(float(output["contradiction_weight_mean"]), 1.0)
        self.assertEqual(float(output["nested_identity_loss"]), 0.0)

    def test_nested_only(self) -> None:
        output = self._run_mode("nested")
        self.assertEqual(float(output["contradiction_weight_mean"]), 1.0)
        self.assertIn("nested_coarse_variance", output)

    def test_combined_mode(self) -> None:
        output = self._run_mode("both")
        self.assertGreater(float(output["contradiction_weight_mean"]), 1.0)
        self.assertIn("nested_richer_variance", output)


if __name__ == "__main__":
    unittest.main()
