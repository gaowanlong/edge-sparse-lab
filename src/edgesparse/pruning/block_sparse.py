"""Block-sparse pruning wrapper."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .mask_utils import (
    apply_mask_to_weight,
    compute_sparsity,
    create_block_mask,
)

if TYPE_CHECKING:
    from transformers import AutoModelForCausalLM

import torch

logger = logging.getLogger(__name__)


def block_prune(
    model: "AutoModelForCausalLM",
    block_rows: int,
    block_cols: int,
    sparsity: float,
    scoring: str = "magnitude",
    activation_norms: dict[str, torch.Tensor] | None = None,
    skip_names: set[str] | None = None,
    verbose: bool = True,
) -> dict[str, float]:
    """Prune *nn.Linear* layers with a block-sparse mask.

    Args:
        model: The model to prune in-place.
        block_rows, block_cols: Block tile dimensions.
        sparsity: Fraction of blocks to zero.
        scoring: "magnitude" or "wanda".
        activation_norms: Required if scoring="wanda".
        skip_names: Module name fragments to skip.
        verbose: Log per-layer sparsity.

    Returns:
        Dict mapping layer name -> sparsity achieved.
    """
    if skip_names is None:
        skip_names = {"lm_head", "embed"}

    if scoring == "wanda" and activation_norms is None:
        raise ValueError("activation_norms required for wanda scoring")

    sparsity_report: dict[str, float] = {}

    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if any(skip in name for skip in skip_names):
            if verbose:
                logger.info("  Skip %s", name)
            continue

        weight = module.weight.data

        if scoring == "wanda":
            norm = activation_norms[name].to(weight.device, weight.dtype)
            scores = weight.abs() * norm.view(-1, 1)
            mask = create_block_mask(scores, block_rows, block_cols, sparsity)
        else:
            mask = create_block_mask(weight, block_rows, block_cols, sparsity)

        apply_mask_to_weight(weight, mask)
        achieved = compute_sparsity(weight)
        sparsity_report[name] = achieved
        if verbose:
            logger.info("  %-60s  sparsity=%.4f", name, achieved)

    overall = _overall_sparsity(model)
    logger.info("Overall block sparsity: %.4f", overall)
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
