"""Tests for N:M structured mask."""

import torch

from src.edgesparse.pruning.mask_utils import (
    create_nm_mask,
    apply_mask_to_weight,
    compute_sparsity,
)


class TestNMMask:
    def test_2_4_basic(self):
        weight = torch.tensor([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]])
        mask = create_nm_mask(weight, n=2, m=4)
        assert mask.shape == weight.shape
        # Each row: 4 elements, keep 2 → exactly 2 survivors per row
        survivors_per_row = mask.sum(dim=1)
        assert torch.all(survivors_per_row == 2), f"Got {survivors_per_row}"

    def test_2_4_sparsity(self):
        torch.manual_seed(42)
        weight = torch.randn(16, 64)
        mask = create_nm_mask(weight, n=2, m=4)
        sp = compute_sparsity(mask.float())
        # 2/4 = 50% sparsity
        assert abs(sp - 0.5) < 1e-6, f"Expected 0.5 sparsity, got {sp}"

    def test_1_4_pattern(self):
        weight = torch.randn(8, 32)
        mask = create_nm_mask(weight, n=1, m=4)
        sp = compute_sparsity(mask.float())
        assert abs(sp - 0.75) < 1e-6, f"Expected 0.75 sparsity, got {sp}"

    def test_4_8_pattern(self):
        weight = torch.randn(4, 64)
        mask = create_nm_mask(weight, n=4, m=8)
        sp = compute_sparsity(mask.float())
        assert abs(sp - 0.5) < 1e-6, f"Expected 0.5 sparsity, got {sp}"

    def test_invalid_n_equal_m(self):
        weight = torch.randn(4, 8)
        try:
            create_nm_mask(weight, n=4, m=4)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_last_dim_not_divisible(self):
        weight = torch.randn(4, 10)
        try:
            create_nm_mask(weight, n=2, m=4)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_small_weight(self):
        weight = torch.randn(2, 8)
        mask = create_nm_mask(weight, n=1, m=4)
        assert mask.shape == (2, 8)
        g1 = mask[0, :4].sum()
        g2 = mask[0, 4:].sum()
        assert g1 == 1, f"Expected 1 survivor in group 0, got {g1}"
        assert g2 == 1, f"Expected 1 survivor in group 1, got {g2}"

    def test_tie_handling(self):
        """When multiple weights have the same magnitude, mask should still
        yield exactly n survivors per group."""
        weight = torch.ones(2, 8) * 0.5
        mask = create_nm_mask(weight, n=2, m=4)
        groups = mask.view(-1, 4)
        assert torch.all(groups.sum(dim=1) == 2), (
            f"Tie case should yield 2 per group, got {groups.sum(dim=1)}"
        )

    def test_exact_top_n_preserved(self):
        weight = torch.tensor(
            [[0.1, 0.5, 0.9, 0.3]], dtype=torch.float32  # rows=1, cols=4
        )
        mask = create_nm_mask(weight, n=2, m=4)
        # The two largest in the group are 0.5 and 0.9 (indices 1, 2)
        assert mask[0, 1] == True, "0.5 should be kept"
        assert mask[0, 2] == True, "0.9 should be kept"
        assert mask[0, 0] == False, "0.1 should be pruned"
        assert mask[0, 3] == False, "0.3 should be pruned"
