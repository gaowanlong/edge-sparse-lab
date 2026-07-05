"""Magnitude-based unstructured pruning."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from .mask_utils import (
    apply_mask_to_weight,
    compute_sparsity,
    create_unstructured_mask,
)

if TYPE_CHECKING:
    from transformers import AutoModelForCausalLM

logger = logging.getLogger(__name__)


def magnitude_prune(
    model: "AutoModelForCausalLM",
    sparsity: float,
    skip_names: set[str] | None = None,
    verbose: bool = True,
) -> dict[str, float]:
    """Prune *nn.Linear* weight matrices by magnitude (unstructured).

    Args:
        model: The model to prune in-place.
        sparsity: Fraction of smallest-magnitude weights to zero.
        skip_names: Module name fragments to skip (e.g., {"lm_head", "embed"}).
        verbose: Log per-layer sparsity.

    Returns:
        Dict mapping layer name -> sparsity achieved.
    """
    if skip_names is None:
        skip_names = {"lm_head", "embed"}

    sparsity_report: dict[str, float] = {}

    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if any(skip in name for skip in skip_names):
            if verbose:
                logger.info("  Skip %s", name)
            continue

        mask = create_unstructured_mask(module.weight.data, sparsity)
        apply_mask_to_weight(module.weight, mask)

        achieved = compute_sparsity(module.weight)
        sparsity_report[name] = achieved
        if verbose:
            logger.info("  %-60s  sparsity=%.4f", name, achieved)

    overall = _overall_sparsity(model)
    logger.info("Overall sparsity: %.4f", overall)
    return sparsity_report


def magnitude_prune_(
    model: "AutoModelForCausalLM",
    sparsity: float,
    skip_names: set[str] | None = None,
) -> dict[str, float]:
    """Alias: in-place magnitude prune with less verbose logging."""
    return magnitude_prune(model, sparsity, skip_names, verbose=False)


def _overall_sparsity(model: "AutoModelForCausalLM") -> float:
    total = 0
    nonzero = 0
    for p in model.parameters():
        total += p.numel()
        nonzero += p.count_nonzero().item()
    if total == 0:
        return 0.0
    return 1.0 - nonzero / total
