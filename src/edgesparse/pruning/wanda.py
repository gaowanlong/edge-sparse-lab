"""Wanda pruning: weight * input-activation-norm scoring."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from .mask_utils import (
    apply_mask_to_weight,
    compute_sparsity,
    create_unstructured_mask,
    create_nm_mask,
    create_block_mask,
)

if TYPE_CHECKING:
    from transformers import AutoModelForCausalLM

logger = logging.getLogger(__name__)


def compute_wanda_scores(
    weight: torch.Tensor, activation_norm: torch.Tensor
) -> torch.Tensor:
    """Compute Wanda score = |weight| * activation_norm (per output channel)."""
    if activation_norm.ndim == 1:
        activation_norm = activation_norm.view(1, -1)
    return weight.abs() * activation_norm


def _apply_wanda_mask(
    weight: torch.Tensor,
    scores: torch.Tensor,
    sparsity: float,
    pattern: str,
    n: int | None = None,
    m: int | None = None,
    block_rows: int | None = None,
    block_cols: int | None = None,
) -> torch.Tensor:
    """Create and apply a mask based on Wanda scores in-place."""
    if pattern == "unstructured":
        mask = create_unstructured_mask(scores, sparsity)
    elif pattern == "nm":
        if n is None or m is None:
            raise ValueError("n and m are required for N:M pattern")
        mask = create_nm_mask(scores, n, m)
    elif pattern == "block":
        if block_rows is None or block_cols is None:
            raise ValueError("block_rows and block_cols required for block pattern")
        mask = create_block_mask(scores, block_rows, block_cols, sparsity)
    else:
        raise ValueError(f"Unknown pattern: {pattern}")

    apply_mask_to_weight(weight, mask)
    return mask


def wanda_prune(
    model: "AutoModelForCausalLM",
    activation_norms: dict[str, torch.Tensor],
    sparsity: float | None = None,
    pattern: str = "unstructured",
    n: int | None = None,
    m: int | None = None,
    block_rows: int | None = None,
    block_cols: int | None = None,
    skip_names: set[str] | None = None,
    verbose: bool = True,
) -> dict[str, float]:
    """Prune *nn.Linear* weights using Wanda scores.

    Args:
        model: The model to prune in-place.
        activation_norms: Dict from Linear module name -> activation norm
            per output channel (from activation_capture).
        sparsity: Fraction to prune (required for unstructured & block).
        pattern: "unstructured", "nm", or "block".
        n, m: For N:M pattern.
        block_rows, block_cols: For block pattern.
        skip_names: Module name fragments to skip.
        verbose: Log per-layer sparsity.

    Returns:
        Dict mapping layer name -> sparsity achieved.
    """
    if skip_names is None:
        skip_names = {"lm_head", "embed"}

    if pattern in ("unstructured", "block") and sparsity is None:
        raise ValueError("sparsity is required for unstructured and block patterns")

    sparsity_report: dict[str, float] = {}

    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if any(skip in name for skip in skip_names):
            if verbose:
                logger.info("  Skip %s", name)
            continue

        weight = module.weight.data
        if name not in activation_norms:
            if verbose:
                logger.warning("  No activation norm for %s, skipping", name)
            continue

        norm = activation_norms[name].to(weight.device, weight.dtype)
        scores = compute_wanda_scores(weight, norm)

        _apply_wanda_mask(
            weight,
            scores,
            sparsity=sparsity if pattern in ("unstructured", "block") else 0.0,
            pattern=pattern,
            n=n,
            m=m,
            block_rows=block_rows,
            block_cols=block_cols,
        )

        achieved = compute_sparsity(weight)
        sparsity_report[name] = achieved
        if verbose:
            logger.info("  %-60s  sparsity=%.4f", name, achieved)

    overall = _overall_sparsity(model)
    logger.info("Overall sparsity: %.4f", overall)
    return sparsity_report


def _overall_sparsity(model: "AutoModelForCausalLM") -> float:
    total = 0
    nonzero = 0
    for p in model.parameters():
        total += p.numel()
        nonzero += p.count_nonzero().item()
    if total == 0:
        return 0.0
    return 1.0 - nonzero / total
