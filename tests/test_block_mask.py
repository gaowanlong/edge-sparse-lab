"""Tests for block-sparse mask."""

import torch

from src.edgesparse.pruning.mask_utils import (
    create_block_mask,
    apply_mask_to_weight,
    compute_sparsity,
)


class TestBlockMask:
    def test_4x8_block_50_percent(self):
        weight = torch.randn(16, 32)
        mask = create_block_mask(weight, block_rows=4, block_cols=8, sparsity=0.5)
        assert mask.shape == weight.shape
        # Blocks that are kept vs zeroed
        sp = compute_sparsity(mask.float())
        # Should be close to 0.5
        assert 0.35 <= sp <= 0.65, f"Block mask sparsity {sp} too far from 0.5"

    def test_8x8_block_25_percent(self):
        weight = torch.randn(32, 32)
        mask = create_block_mask(weight, block_rows=8, block_cols=8, sparsity=0.25)
        sp = compute_sparsity(mask.float())
        assert 0.15 <= sp <= 0.35, f"Block mask sparsity {sp} too far from 0.25"

    def test_16x16_block_50_percent(self):
        weight = torch.randn(64, 64)
        mask = create_block_mask(weight, block_rows=16, block_cols=16, sparsity=0.5)
        sp = compute_sparsity(mask.float())
        assert 0.35 <= sp <= 0.65, f"Block mask sparsity {sp} too far from 0.5"

    def test_non_power_of_two_dim(self):
        weight = torch.randn(15, 21)
        mask = create_block_mask(weight, block_rows=4, block_cols=8, sparsity=0.5)
        assert mask.shape == (15, 21), f"Shape mismatch: {mask.shape}"

    def test_zero_sparsity(self):
        weight = torch.randn(16, 16)
        mask = create_block_mask(weight, block_rows=4, block_cols=4, sparsity=0.0)
        assert mask.all(), "Zero sparsity should keep all weights"

    def test_full_sparsity_high(self):
        weight = torch.randn(16, 16)
        mask = create_block_mask(weight, block_rows=4, block_cols=4, sparsity=0.99)
        # With 16 blocks, 0.99 ~ prune all
        sp = compute_sparsity(mask.float())
        assert sp >= 0.9, f"Should be nearly fully sparse, got {sp}"

    def test_block_alignment_with_padding(self):
        """Test that blocks align correctly when padding is needed."""
        weight = torch.randn(10, 14)
        mask = create_block_mask(weight, block_rows=4, block_cols=8, sparsity=0.5)
        assert mask.shape == (10, 14)
        # Check block-aligned regions
        # The top-left 8x8 region should have same mask pattern per block
        block_tl = mask[:4, :8]
        assert block_tl.shape == (4, 8)

    def test_apply_mask(self):
        weight = torch.randn(8, 16)
        orig = weight.clone()
        mask = create_block_mask(weight, block_rows=4, block_cols=8, sparsity=0.5)
        apply_mask_to_weight(weight, mask)
        # Zeroed positions
        assert torch.all(weight[~mask] == 0.0), "Pruned weights should be zero"
        # Kept positions unchanged
        assert torch.allclose(weight[mask], orig[mask]), "Kept weights unchanged"

    def test_different_block_sizes(self):
        """Test various block sizes produce valid masks."""
        shapes = [(16, 32), (32, 64), (8, 24)]
        block_configs = [(4, 8), (8, 8), (4, 4)]
        for shape in shapes:
            for b_rows, b_cols in block_configs:
                weight = torch.randn(*shape)
                try:
                    mask = create_block_mask(
                        weight, block_rows=b_rows, block_cols=b_cols, sparsity=0.5
                    )
                    assert mask.shape == weight.shape, (
                        f"Shape mismatch for {shape} with {b_rows}x{b_cols}: "
                        f"{mask.shape}"
                    )
                except Exception as e:
                    assert False, (
                        f"Failed for {shape} with {b_rows}x{b_cols}: {e}"
                    )
