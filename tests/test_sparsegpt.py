"""Tests for SparseGPT pruning."""

import torch

from src.edgesparse.pruning.sparsegpt import (
    compute_hessian,
    sparsegpt_prune_layer,
    sparsegpt_prune,
)
from src.edgesparse.pruning.mask_utils import create_unstructured_mask, compute_sparsity


class TestSparseGPT:
    def test_compute_hessian_shape(self):
        inputs = torch.randn(10, 32)
        H = compute_hessian(inputs, damp=0.01)
        assert H.shape == (32, 32), f"Expected (32, 32), got {H.shape}"
        assert torch.allclose(H, H.T, atol=1e-5), "Hessian must be symmetric"

    def test_hessian_positive_definite(self):
        inputs = torch.randn(16, 8)
        H = compute_hessian(inputs, damp=0.1)
        eigvals = torch.linalg.eigvalsh(H)
        assert torch.all(eigvals > 0), "Hessian must be positive definite"

    def test_prune_layer_sparsity(self):
        torch.manual_seed(42)
        weight = torch.randn(16, 32)
        inputs = torch.randn(64, 32)
        H = compute_hessian(inputs, damp=0.01)
        w_copy = weight.clone()

        pruned, mask = sparsegpt_prune_layer(w_copy, H, sparsity=0.5)
        sp = compute_sparsity(pruned)
        assert 0.45 <= sp <= 0.55, f"Expected ~0.5 sparsity, got {sp}"
        assert mask.shape == weight.shape

    def test_prune_layer_weight_update(self):
        """After pruning, the remaining weights should be updated to compensate."""
        torch.manual_seed(1)
        weight = torch.randn(4, 16)
        inputs = torch.randn(32, 16)
        H = compute_hessian(inputs, damp=0.01)

        w_before = weight.clone()
        pruned, _ = sparsegpt_prune_layer(weight.clone(), H, sparsity=0.5)

        # Some remaining weights should be different (compensation was applied)
        mask = pruned != 0
        diff = (pruned[mask] - w_before[mask]).abs()
        assert diff.sum() > 0, "Compensation should modify remaining weights"

    def test_precomputed_mask(self):
        torch.manual_seed(42)
        weight = torch.randn(8, 16)
        inputs = torch.randn(32, 16)
        H = compute_hessian(inputs, damp=0.01)

        # Precompute a 50% mask
        mask = create_unstructured_mask(weight, 0.5)
        pruned, final_mask = sparsegpt_prune_layer(
            weight.clone(), H, sparsity=0.5, mask=mask
        )
        sp = compute_sparsity(pruned)
        assert 0.45 <= sp <= 0.55

    def test_zero_sparsity(self):
        weight = torch.randn(4, 8)
        inputs = torch.randn(16, 8)
        H = compute_hessian(inputs, damp=0.01)
        pruned, mask = sparsegpt_prune_layer(weight.clone(), H, sparsity=0.0)
        assert torch.allclose(pruned, weight), "Zero sparsity should not change weights"

    def test_full_sparsity(self):
        weight = torch.randn(4, 8)
        inputs = torch.randn(16, 8)
        H = compute_hessian(inputs, damp=0.01)
        pruned, mask = sparsegpt_prune_layer(weight.clone(), H, sparsity=1.0)
        assert torch.all(pruned == 0), "Full sparsity should zero all weights"
