"""N:M structured pruning wrapper."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from .mask_utils import (
    apply_mask_to_weight,
    compute_sparsity,
    create_nm_mask,
)

if TYPE_CHECKING:
    from transformers import AutoModelForCausalLM

logger = logging.getLogger(__name__)


def nm_prune(
    model: "AutoModelForCausalLM",
    n: int,
    m: int,
    scoring: str = "magnitude",
    activation_norms: dict[str, torch.Tensor] | None = None,
    skip_names: set[str] | None = None,
    verbose: bool = True,
) -> dict[str, float]:
    """Prune *nn.Linear* layers with an N:M structured mask.

    Args:
        model: The model to prune in-place.
        n: Keep top-n per group.
        m: Group size.
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
            mask = create_nm_mask(scores, n, m)
        else:
            mask = create_nm_mask(weight, n, m)

        apply_mask_to_weight(weight, mask)
        achieved = compute_sparsity(weight)
        sparsity_report[name] = achieved
        if verbose:
            logger.info("  %-60s  sparsity=%.4f", name, achieved)

    overall = _overall_sparsity(model)
    logger.info("Overall N:M sparsity: %.4f (target=%.4f)", overall, 1.0 - n / m)
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
