"""Mask primitives: creation, application, and sparsity statistics."""

from __future__ import annotations

import torch
import math


def create_unstructured_mask(
    weight: torch.Tensor, sparsity: float
) -> torch.Tensor:
    """Create a binary unstructured mask (1 = keep, 0 = prune).

    The smallest *sparsity* fraction of weights by absolute value
    are zeroed (mask = 0).
    """
    if not 0.0 <= sparsity < 1.0:
        raise ValueError(f"sparsity must be in [0, 1), got {sparsity}")
    numel = weight.numel()
    k = int(round(numel * sparsity))
    if k == 0:
        return torch.ones_like(weight, dtype=torch.bool)

    flat = weight.abs().view(-1)
    threshold = flat.kthvalue(k).values
    mask = weight.abs() > threshold
    return mask


def create_nm_mask(weight: torch.Tensor, n: int, m: int) -> torch.Tensor:
    """Create an N:M structured mask (1 = keep, 0 = prune).

    Every contiguous group of *m* elements along the *last* dimension
    retains the *n* elements with the largest absolute value.
    """
    if n <= 0 or m <= 0:
        raise ValueError(f"n and m must be positive, got n={n}, m={m}")
    if n >= m:
        raise ValueError(f"n must be < m, got n={n}, m={m}")
    if weight.size(-1) % m != 0:
        raise ValueError(
            f"Last dim ({weight.size(-1)}) must be divisible by m ({m})"
        )

    # reshape: (..., M) where M = m
    orig_shape = weight.shape
    flat_last = weight.view(-1, weight.size(-1))
    groups = flat_last.view(-1, m)

    threshold = groups.abs().kthvalue(m - n, dim=1, keepdim=True).values
    mask_2d = (groups.abs() >= threshold).to(groups.dtype)

    # If there are ties that cause >n survivors, clamp back to n
    survivors = mask_2d.sum(dim=1, keepdim=True)
    excess = (survivors - n).clamp(min=0)
    if excess.any():
        # zero out the smallest among the survivors in over-full groups
        masked_abs = groups.abs() * mask_2d + (1 - mask_2d) * (-1e9)
        order = masked_abs.argsort(dim=1, descending=True)
        rank = order.argsort(dim=1)
        mask_2d = (rank < n).to(groups.dtype)

    mask_2d = mask_2d.view(flat_last.shape)
    return mask_2d.view(orig_shape).bool()


def create_block_mask(
    weight: torch.Tensor,
    block_rows: int,
    block_cols: int,
    sparsity: float,
) -> torch.Tensor:
    """Create a block-sparse mask (1 = keep, 0 = prune).

    The weight matrix is tiled into (block_rows × block_cols) blocks.
    A block is kept if its L1 norm ranks above the *sparsity* percentile.
    """
    if not 0.0 <= sparsity < 1.0:
        raise ValueError(f"sparsity must be in [0, 1), got {sparsity}")
    if block_rows <= 0 or block_cols <= 0:
        raise ValueError(
            f"block dimensions must be positive, got {block_rows}x{block_cols}"
        )

    rows, cols = weight.shape
    r_blocks = math.ceil(rows / block_rows)
    c_blocks = math.ceil(cols / block_cols)

    # Pad to block boundary
    padded = weight
    pad_r = r_blocks * block_rows - rows
    pad_c = c_blocks * block_cols - cols
    if pad_r > 0 or pad_c > 0:
        padded = torch.nn.functional.pad(weight, (0, pad_c, 0, pad_r))

    # Compute block norms
    blocks = padded.view(r_blocks, block_rows, c_blocks, block_cols)
    block_norm = blocks.abs().sum(dim=(1, 3))  # (r_blocks, c_blocks)

    num_blocks = block_norm.numel()
    k = int(round(num_blocks * sparsity))
    if k == 0:
        # Keep all blocks
        keep = torch.ones_like(block_norm, dtype=torch.bool)
    elif k >= num_blocks:
        keep = torch.zeros_like(block_norm, dtype=torch.bool)
    else:
        flat = block_norm.view(-1)
        threshold = flat.kthvalue(k).values
        keep = block_norm > threshold

    # Expand keep to full padded shape
    keep_full = (
        keep.unsqueeze(1).unsqueeze(3).expand(r_blocks, block_rows, c_blocks, block_cols)
    )
    keep_full = keep_full.reshape(padded.shape).bool()

    if pad_r > 0 or pad_c > 0:
        keep_full = keep_full[:rows, :cols]

    return keep_full


def apply_mask_to_weight(weight: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Apply a binary mask to a weight tensor in-place and return it."""
    weight.data.mul_(mask.to(weight.dtype))
    return weight


def compute_sparsity(weight: torch.Tensor) -> float:
    """Return the fraction of zero elements (as a float in [0, 1])."""
    total = weight.numel()
    if total == 0:
        return 0.0
    zeros = total - weight.count_nonzero().item()
    return zeros / total


def count_nonzero(weight: torch.Tensor) -> int:
    """Return the number of nonzero elements."""
    return weight.count_nonzero().item()
