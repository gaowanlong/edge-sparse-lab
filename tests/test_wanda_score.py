"""Tests for Wanda score computation."""

import torch

from src.edgesparse.pruning.wanda import compute_wanda_scores
from src.edgesparse.pruning.mask_utils import (
    create_unstructured_mask,
    apply_mask_to_weight,
    compute_sparsity,
)


class TestWandaScore:
    def test_score_shape_and_values(self):
        """Score = |W| * act_norm, broadcast over out_features dim."""
        weight = torch.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]])  # (2, 3)
        norms = torch.tensor([1.0, 2.0, 3.0])  # matches in_features=3
        scores = compute_wanda_scores(weight, norms)
        assert scores.shape == weight.shape, f"Expected {weight.shape}, got {scores.shape}"
        # |W| * norms broadcasts: (2,3) * (1,3) → (2,3)
        expected = weight.abs() * norms.view(1, -1)
        assert torch.allclose(scores, expected), f"Scores mismatch"

    def test_zero_norm_gives_zero_score(self):
        weight = torch.randn(4, 8)      # (out=4, in=8)
        norms = torch.zeros(8)           # matches in_features=8
        scores = compute_wanda_scores(weight, norms)
        assert torch.all(scores == 0.0), "Zero norm should give zero score"

    def test_negative_weights(self):
        weight = torch.tensor([[-3.0, 2.0], [1.0, -4.0]])  # (2, 2)
        norms = torch.tensor([2.0, 3.0])                    # in=2
        scores = compute_wanda_scores(weight, norms)
        # |W| = [[3,2],[1,4]], norms=[2,3] → [[6,6],[2,12]]
        expected = torch.tensor([[6.0, 6.0], [2.0, 12.0]])
        assert torch.allclose(scores, expected)

    def test_wanda_50_percent_pruning(self):
        torch.manual_seed(42)
        weight = torch.randn(8, 16)       # (out=8, in=16)
        norms = torch.randn(16).abs()      # matches in_features=16

        scores = compute_wanda_scores(weight, norms)
        mask = create_unstructured_mask(scores, sparsity=0.5)
        apply_mask_to_weight(weight, mask)
        sp = compute_sparsity(weight)
        assert 0.45 <= sp <= 0.55, f"Expected ~0.5 sparsity, got {sp}"

    def test_all_channels_same_norm(self):
        weight = torch.randn(4, 8)       # (out=4, in=8)
        norms = torch.ones(8) * 5.0       # matches in_features=8
        scores = compute_wanda_scores(weight, norms)
        assert torch.allclose(scores, weight.abs() * 5.0)
