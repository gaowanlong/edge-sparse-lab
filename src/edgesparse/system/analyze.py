"""Per-layer analysis and experiment comparison tools."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


def load_experiment(exp_dir: str) -> dict[str, Any]:
    """Load all JSON reports from an experiment directory."""
    result: dict[str, Any] = {"dir": exp_dir}
    for fname in ("config.json", "sparsity_report.json", "quality_report.json"):
        path = os.path.join(exp_dir, fname)
        if os.path.exists(path):
            with open(path) as f:
                result[fname.replace(".json", "")] = json.load(f)
    return result


def compare_experiments(
    exp_dirs: list[str],
    output_dir: str | None = None,
) -> str:
    """Compare multiple experiments and generate a Markdown table.

    Args:
        exp_dirs: List of experiment directories.
        output_dir: Optional output directory for the report.

    Returns:
        Markdown table as a string.
    """
    exps = [load_experiment(d) for d in exp_dirs]

    lines: list[str] = []
    lines.append("# Method × Pattern Comparison")
    lines.append("")
    lines.append(f"| Experiment | Method | Pattern | Sparsity Target | Overall Sparsity | PPL |")
    lines.append(f"|---|---|---|---|---|---|")

    for exp in exps:
        cfg = exp.get("config", {})
        sp = exp.get("sparsity_report", {})
        ql = exp.get("quality_report", {})

        name = os.path.basename(exp["dir"])
        method = cfg.get("method", "?")
        pattern = cfg.get("pattern", "?")
        sp_target = cfg.get("sparsity", cfg.get("n", "?"))
        if "n" in cfg and "m" in cfg:
            sp_target = f'{cfg.get("n")}:{cfg.get("m")}'

        overall = sp.get("overall", "?")
        if isinstance(overall, float):
            overall = f"{overall:.2%}"

        ppl = ql.get("perplexity_after", "?")
        if isinstance(ppl, float):
            ppl = f"{ppl:.2f}"

        lines.append(f"| {name} | {method} | {pattern} | {sp_target} | {overall} | {ppl} |")

    lines.append("")

    # Per-layer analysis
    lines.append("## Per-Layer Sparsity Details")
    lines.append("")
    for exp in exps:
        sp = exp.get("sparsity_report", {})
        per_layer = sp.get("per_layer", {})
        if per_layer:
            name = os.path.basename(exp["dir"])
            lines.append(f"### {name}")
            lines.append("")
            lines.append("| Layer | Sparsity |")
            lines.append("|-------|----------|")
            for layer, s in sorted(per_layer.items()):
                lines.append(f"| {layer} | {s:.2%} |")
            lines.append("")

    text = "\n".join(lines)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "comparison.md")
        with open(path, "w") as f:
            f.write(text)
        logger.info("Comparison report saved to %s", path)

    return text


def get_per_layer_degradation(
    dense_exp: str, pruned_exp: str
) -> dict[str, float]:
    """Analyze per-layer PPL contribution (placeholder).

    In a full implementation, this would compute the delta in
    output MSE per layer. For now, returns the sparsity delta.
    """
    dense = load_experiment(dense_exp)
    pruned = load_experiment(pruned_exp)

    dense_sp = dense.get("sparsity_report", {}).get("per_layer", {})
    pruned_sp = pruned.get("sparsity_report", {}).get("per_layer", {})

    delta: dict[str, float] = {}
    all_layers = set(dense_sp) | set(pruned_sp)
    for layer in sorted(all_layers):
        ds = dense_sp.get(layer, 0.0)
        ps = pruned_sp.get(layer, 0.0)
        delta[layer] = ps - ds

    return delta
