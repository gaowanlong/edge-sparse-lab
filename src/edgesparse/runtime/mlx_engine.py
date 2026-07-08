"""MLX runtime for inference of pruned models on Apple Silicon."""

from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING

import mlx.core as mx
import mlx.nn as mlx_nn
import numpy as np

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


def convert_torch_to_mlx(
    model: "torch.nn.Module",
) -> dict[str, mx.array]:
    """Convert PyTorch model weights to MLX arrays.

    Args:
        model: PyTorch model (can be pruned; zeros are preserved).

    Returns:
        Dict of MLX arrays keyed by parameter name.
    """
    import torch
    weights = {}
    for name, param in model.named_parameters():
        weights[name] = mx.array(param.detach().cpu().numpy())
    logger.info("Converted %d parameters to MLX", len(weights))
    return weights


def save_mlx_weights(
    weights: dict[str, mx.array],
    output_dir: str,
    metadata: dict | None = None,
) -> str:
    """Save MLX weights as .safetensors + metadata.

    Args:
        weights: Dict of MLX arrays.
        output_dir: Output directory.
        metadata: Optional metadata dict.

    Returns:
        Path to the saved weights directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    from mlx.utils import tree_flatten, tree_unflatten

    flat = tree_flatten(weights)
    safetensors_path = os.path.join(output_dir, "mlx_model.safetensors")
    # Use dict with string keys
    save_dict = {"/".join(k) if isinstance(k, tuple) else k: v for k, v in flat}
    mx.save_safetensors(safetensors_path, save_dict)
    logger.info("MLX weights saved to %s", safetensors_path)

    if metadata:
        meta_path = os.path.join(output_dir, "mlx_metadata.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

    return safetensors_path


class MLXRuntime:
    """Minimal MLX inference runtime for pruned LM heads."""

    def __init__(self, weights: dict[str, mx.array], config: dict) -> None:
        self.weights = weights
        self.config = config
        self.dtype = mx.float16

    @classmethod
    def from_pytorch(
        cls,
        model: "torch.nn.Module",
        config: dict,
    ) -> "MLXRuntime":
        """Create MLXRuntime from a PyTorch model."""
        weights = convert_torch_to_mlx(model)
        return cls(weights, config)

    def generate(
        self,
        tokenizer: "PreTrainedTokenizerBase | None" = None,
        prompt_ids: list[int] | None = None,
        max_tokens: int = 32,
        temperature: float = 0.0,
    ) -> list[int]:
        """Simple greedy/argmax generation.

        This is a minimal implementation that runs one MLX forward pass.
        For a full implementation, consider using mlx_lm.generate().

        Args:
            tokenizer: Not used directly, kept for API compatibility.
            prompt_ids: Input token IDs.
            max_tokens: Number of tokens to generate.
            temperature: Sampling temperature (0 = greedy).

        Returns:
            Generated token IDs (including prompt).
        """
        if prompt_ids is None:
            raise ValueError("prompt_ids is required")

        # This is a simplified forward pass
        # A full implementation would:
        # 1. Build the model graph from weights
        # 2. Do KV-cached autoregressive generation
        # 3. Return generated tokens

        # For now, return the input as-is to validate the pipeline
        logger.info(
            "MLXRuntime.generate() called with %d prompt tokens",
            len(prompt_ids),
        )
        return prompt_ids

    def get_sparsity(self) -> float:
        """Compute sparsity of MLX weights (fraction of zeros)."""
        total = 0
        zeros = 0
        for name, w in self.weights.items():
            total += w.size
            zeros += mx.sum(w == 0).item()
        return zeros / max(total, 1)
