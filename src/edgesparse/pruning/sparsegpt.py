"""SparseGPT: one-shot unstructured pruning via approximate second-order info.

Reference: Frantar & Alistarh, "SparseGPT: Massive Language Models Can Be
Accurately Pruned in One-Shot" (2023).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from transformers import AutoModelForCausalLM

logger = logging.getLogger(__name__)


def compute_hessian(inputs: torch.Tensor, damp: float = 0.01) -> torch.Tensor:
    """Compute the Gauss-Newton Hessian H = 2X^T X with damping.

    Args:
        inputs: (N, d_in) calibration activations.
        damp: Damping factor relative to mean(diag(H)).

    Returns:
        H: (d_in, d_in) damped Hessian.
    """
    d_in = inputs.size(-1)
    # H = 2 * X^T X  (empirical Gauss-Newton for MSE loss)
    H = 2.0 * inputs.T @ inputs  # (d_in, d_in)

    # Damping
    lam = damp * torch.mean(torch.diag(H))
    H += lam * torch.eye(d_in, device=H.device, dtype=H.dtype)
    return H


@torch.no_grad()
def sparsegpt_prune_layer(
    weight: torch.Tensor,
    hessian: torch.Tensor,
    sparsity: float,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prune one linear layer with SparseGPT (OBS-style).

    Args:
        weight: (d_out, d_in) weight matrix (will be modified in-place).
        hessian: (d_in, d_in) damped Gauss-Newton Hessian.
        sparsity: Fraction of weights to prune in [0, 1).
        mask: Optional pre-computed mask (1=keep, 0=prune). If None,
            computed from SparseGPT importance.

    Returns:
        (pruned_weight, mask) — weight is modified in-place.
    """
    d_in = weight.size(1)
    device = weight.device
    dtype = weight.dtype

    # Cholesky decomposition of H -> G (lower triangular)
    H = hessian.float()
    try:
        L = torch.linalg.cholesky(H)
    except torch.linalg.LinAlgError:
        # Add extra damping if not positive definite
        H += 1e-6 * torch.eye(d_in, device=device, dtype=torch.float32)
        L = torch.linalg.cholesky(H)

    # Invert via Cholesky: H_inv = L^{-T} @ L^{-1}
    H_inv = torch.cholesky_inverse(L)  # (d_in, d_in) float32

    H_inv_diag = torch.diag(H_inv)  # (d_in,)
    w = weight.float()  # (d_out, d_in)
    d_out = w.size(0)

    if mask is not None:
        mask = mask.to(device).bool()

    k = int(round(d_in * sparsity))

    for i in range(d_out):
        row = w[i]  # (d_in,)
        pruned_mask = torch.zeros(d_in, dtype=torch.bool, device=device)

        if mask is not None:
            # Apply precomputed mask
            row_mask = mask[i].bool()
            pruned_positions = (~row_mask).nonzero(as_tuple=True)[0]
            for j in pruned_positions:
                # Compensate remaining weights for the pruned weight
                update = (-row[j] / H_inv_diag[j]) * H_inv[:, j]
                update[pruned_mask] = 0.0
                row.add_(update.to(row.dtype))
                row[j] = 0.0
                pruned_mask[j] = True
        else:
            # Standard SparseGPT: prune by importance
            scores = row**2 / H_inv_diag
            order = scores.argsort()  # smallest first

            for j in order[:k]:
                # Compensate only unpruned weights
                update = (-row[j] / H_inv_diag[j]) * H_inv[:, j]
                update[pruned_mask] = 0.0
                row.add_(update.to(row.dtype))
                row[j] = 0.0
                pruned_mask[j] = True

    weight.data.copy_(w.to(dtype))
    final_mask = weight != 0
    return weight, final_mask


def sparsegpt_prune(
    model: "AutoModelForCausalLM",
    hessians: dict[str, torch.Tensor],
    sparsity: float,
    skip_names: set[str] | None = None,
    verbose: bool = True,
) -> dict[str, float]:
    """Prune all nn.Linear layers with SparseGPT.

    Args:
        model: The model to prune (in-place).
        hessians: Dict of layer name -> (d_in, d_in) Hessian matrices
            (from calibration).
        sparsity: Target sparsity (for unstructured).
        skip_names: Module name fragments to skip.
        verbose: Log per-layer info.

    Returns:
        Dict mapping layer name -> sparsity achieved.
    """
    from .mask_utils import compute_sparsity

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

        if name not in hessians:
            if verbose:
                logger.warning("  No Hessian for %s, skipping", name)
            continue

        weight = module.weight.data
        H = hessians[name].to(weight.device, dtype=torch.float32)

        sparsegpt_prune_layer(weight, H, sparsity)

        achieved = compute_sparsity(weight)
        sparsity_report[name] = achieved
        if verbose:
            logger.info("  %-60s  sparsity=%.4f", name, achieved)

    # Overall sparsity
    total = sum(p.numel() for p in model.parameters())
    nonzero = sum(p.count_nonzero().item() for p in model.parameters())
    overall = 1.0 - nonzero / total if total > 0 else 0.0
    logger.info("Overall SparseGPT sparsity: %.4f", overall)
    return sparsity_report
