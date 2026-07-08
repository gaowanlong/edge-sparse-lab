"""Tests for system analysis."""

import json
import os
import tempfile

from src.edgesparse.system.analyze import (
    load_experiment,
    compare_experiments,
)


class TestAnalyze:
    def test_load_experiment_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = load_experiment(tmp)
            assert "dir" in result

    def test_compare_experiments_empty(self):
        result = compare_experiments([])
        assert "Method × Pattern" in result

    def test_compare_with_real_data(self):
        exp_dir = "outputs/runs/dense_baseline"
        if os.path.exists(exp_dir):
            result = compare_experiments([exp_dir])
            assert "19.99" in result or "Perplexity" in result

    def test_get_per_layer(self):
        from src.edgesparse.system.analyze import get_per_layer_degradation
        # Test with missing dirs - should not crash
        with tempfile.TemporaryDirectory() as tmp:
            result = get_per_layer_degradation(tmp, tmp)
            assert isinstance(result, dict)
