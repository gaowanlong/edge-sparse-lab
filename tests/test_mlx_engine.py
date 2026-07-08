"""Tests for MLX runtime."""

import torch
import mlx.core as mx

from src.edgesparse.runtime.mlx_engine import (
    convert_torch_to_mlx,
    MLXRuntime,
)


class TestMLXEngine:
    def test_convert_small_model(self):
        """Convert a tiny PyTorch model to MLX arrays."""
        model = torch.nn.Sequential(
            torch.nn.Linear(16, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 8),
        )
        weights = convert_torch_to_mlx(model)
        assert len(weights) > 0, "Should have weights"
        for name, arr in weights.items():
            assert isinstance(arr, mx.array), f"{name} should be MLX array"

    def test_convert_preserves_zeros(self):
        """Pruned (zero) weights should be preserved in MLX arrays."""
        model = torch.nn.Linear(8, 16)
        with torch.no_grad():
            model.weight.data[:] = 0.0  # All pruned
        weights = convert_torch_to_mlx(model)
        w = weights.get("weight", weights.get("0.weight"))
        assert w is not None, "Should have weight"
        zeros = mx.sum(w == 0).item()
        assert zeros == w.size, f"Expected all zeros, got {zeros}/{w.size}"

    def test_runtime_sparsity(self):
        model = torch.nn.Linear(10, 20)
        with torch.no_grad():
            model.weight.data[:5] = 0.0
        rt = MLXRuntime.from_pytorch(model, {"test": True})
        sp = rt.get_sparsity()
        assert 0.20 <= sp <= 0.30, f"Expected ~0.25 sparsity, got {sp}"

    def test_generate_basic(self):
        model = torch.nn.Linear(4, 8)
        rt = MLXRuntime.from_pytorch(model, {"test": True})
        result = rt.generate(prompt_ids=[1, 2, 3], max_tokens=10)
        assert result == [1, 2, 3], "Should return prompt (stub)"
