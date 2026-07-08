"""Inference-time activation capture for Wanda scoring."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from transformers import AutoModelForCausalLM

logger = logging.getLogger(__name__)


class ActivationCache:
    """A registry that collects hidden-state statistics per linear layer.

    Stores the per-channel L2 norm of input activations seen during the
    calibration forward pass.
    """

    def __init__(self) -> None:
        self.activation_norms: dict[str, torch.Tensor] = {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def _hook_fn(self, name: str) -> callable:
        def hook(_module, input_, _output):
            # input_[0] shape: (batch, seq, hidden_dim) or (batch, hidden_dim)
            inp = input_[0].detach()
            # Compute in float32 to avoid overflow in later layers
            inp = inp.float()
            # Per-channel L2 norm across all non-feature dims
            norm = inp.norm(p=2, dim=tuple(range(inp.ndim - 1))).pow(2)
            if name in self.activation_norms:
                self.activation_norms[name] += norm
            else:
                self.activation_norms[name] = norm

        return hook

    def register_linear_layers(
        self, model: "AutoModelForCausalLM"
    ) -> None:
        """Register forward hooks on all nn.Linear layers in the model."""
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                handle = module.register_forward_hook(self._hook_fn(name))
                self._handles.append(handle)
        logger.info(
            "Registered hooks on %d Linear layers", len(self._handles)
        )

    def remove_hooks(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def get_norms(self) -> dict[str, torch.Tensor]:
        return dict(self.activation_norms)

    def finalize(self, num_samples: int) -> dict[str, torch.Tensor]:
        """Divide accumulators by the number of samples and return."""
        norms = {}
        for name, acc in self.activation_norms.items():
            norms[name] = (acc / num_samples).sqrt().half()  # back to float16
        self.activation_norms = norms
        return norms


def capture_activations(
    model: "AutoModelForCausalLM",
    input_ids: torch.Tensor,
    batch_size: int = 1,
) -> dict[str, torch.Tensor]:
    """Run forward pass(es) and return per-layer activation norms.

    Processes samples one at a time (or in small batches) to avoid OOM on MPS.

    Returns a dict mapping layer name -> per-output-channel norm vector.
    """
    cache = ActivationCache()
    cache.register_linear_layers(model)

    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    num_samples = input_ids.size(0)

    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            batch = input_ids[i : i + batch_size]
            model(batch)

    norms = cache.finalize(num_samples)
    cache.remove_hooks()
    logger.info(
        "Captured activations for %d layers (%d samples, batch=%d)",
        len(norms),
        num_samples,
        batch_size,
    )
    return norms


class HessianCache:
    """Accumulates Hessian approximations H = 2*X^T X for each Linear layer."""

    def __init__(self) -> None:
        self.hessians: dict[str, torch.Tensor] = {}
        self._counter: dict[str, int] = {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def _hook_fn(self, name: str) -> callable:
        def hook(_module, input_, _output):
            inp = input_[0].detach().float()  # (batch, seq, d_in) or (batch, d_in)
            # Flatten batch & seq dims
            inp_2d = inp.view(-1, inp.size(-1))  # (N, d_in)
            # Accumulate X^T X
            xtx = inp_2d.T @ inp_2d  # (d_in, d_in)
            if name in self.hessians:
                self.hessians[name] += xtx
            else:
                self.hessians[name] = xtx
        return hook

    def register_linear_layers(self, model: "AutoModelForCausalLM") -> None:
        from typing import TYPE_CHECKING
        if TYPE_CHECKING:
            from transformers import AutoModelForCausalLM
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                handle = module.register_forward_hook(self._hook_fn(name))
                self._handles.append(handle)
        logger.info("Registered Hessian hooks on %d Linear layers", len(self._handles))

    def remove_hooks(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def finalize(self, num_total_tokens: int) -> dict[str, torch.Tensor]:
        """Scale by 2/total_tokens and return."""
        for name, H in self.hessians.items():
            self.hessians[name] = H * (2.0 / max(num_total_tokens, 1))
        return dict(self.hessians)


def capture_hessians(
    model: "AutoModelForCausalLM",
    input_ids: torch.Tensor,
    batch_size: int = 1,
) -> dict[str, torch.Tensor]:
    """Run forward pass and return per-layer Gauss-Newton Hessian matrices.

    H = (2/N) * X^T X  where X is the input activation matrix for that layer.

    Returns dict mapping layer name -> (d_in, d_in) Hessian (float32).
    """
    cache = HessianCache()
    cache.register_linear_layers(model)

    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    num_samples = input_ids.size(0)

    total_tokens = 0
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            batch = input_ids[i : i + batch_size]
            model(batch)
            total_tokens += batch.numel()

    hessians = cache.finalize(total_tokens)
    cache.remove_hooks()
    logger.info("Captured Hessians for %d layers (%d tokens)", len(hessians), total_tokens)
    return hessians
