"""Experiment report generation (JSON + Markdown)."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SparsityReport:
    """Per-layer and overall sparsity statistics."""

    overall: float
    per_layer: dict[str, float] = field(default_factory=dict)
    n_params_total: int = 0
    n_params_remaining: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SparsityReport":
        return cls(**d)

    @staticmethod
    def build(
        per_layer: dict[str, float], model=None
    ) -> "SparsityReport":
        total = 0
        nonzero = 0
        if model is not None:
            for p in model.parameters():
                total += p.numel()
                nonzero += p.count_nonzero().item()

        overall = 0.0
        if total > 0:
            overall = 1.0 - nonzero / total

        return cls(
            overall=overall,
            per_layer=per_layer,
            n_params_total=total,
            n_params_remaining=nonzero,
        )


@dataclass
class QualityReport:
    """Quality evaluation results."""

    perplexity_before: float | None = None
    perplexity_after: float | None = None
    dataset: str = ""
    num_samples: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "QualityReport":
        return cls(**d)


@dataclass
class ExperimentReport:
    """Top-level experiment report."""

    experiment_name: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    config: dict[str, Any] = field(default_factory=dict)
    sparsity: SparsityReport | None = None
    quality: QualityReport | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "experiment_name": self.experiment_name,
            "timestamp": self.timestamp,
            "config": self.config,
        }
        if self.sparsity is not None:
            d["sparsity"] = self.sparsity.to_dict()
        if self.quality is not None:
            d["quality"] = self.quality.to_dict()
        return d


def save_config(config: dict[str, Any], output_dir: str) -> str:
    """Save experiment configuration as JSON."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "config.json")
    with open(path, "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    logger.info("Config saved to %s", path)
    return path


def save_report_json(report: ExperimentReport, output_dir: str) -> str:
    """Save experiment report as JSON."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "report.json")
    with open(path, "w") as f:
        json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
    logger.info("Report JSON saved to %s", path)
    return path


def save_report_markdown(
    report: ExperimentReport,
    output_dir: str,
) -> str:
    """Generate and save a human-readable Markdown report."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "report.md")

    lines: list[str] = []
    lines.append(f"# Experiment: {report.experiment_name}")
    lines.append("")
    lines.append(f"- **Timestamp:** {report.timestamp}")
    lines.append("")

    # Config
    lines.append("## Configuration")
    lines.append("")
    lines.append("| Key | Value |")
    lines.append("|-----|-------|")
    for k, v in report.config.items():
        lines.append(f"| {k} | {v} |")
    lines.append("")

    # Sparsity
    if report.sparsity is not None:
        s = report.sparsity
        lines.append("## Sparsity Report")
        lines.append("")
        lines.append(f"- **Overall sparsity:** {s.overall:.4%}")
        lines.append(
            f"- **Parameters remaining:** "
            f"{s.n_params_remaining:,} / {s.n_params_total:,}"
        )
        lines.append("")
        if s.per_layer:
            lines.append("### Per-Layer Sparsity")
            lines.append("")
            lines.append("| Layer | Sparsity |")
            lines.append("|-------|----------|")
            for layer, sp in sorted(s.per_layer.items()):
                lines.append(f"| {layer} | {sp:.4%} |")
            lines.append("")

    # Quality
    if report.quality is not None:
        q = report.quality
        lines.append("## Quality Report")
        lines.append("")
        lines.append(f"- **Dataset:** {q.dataset}")
        lines.append(f"- **Samples:** {q.num_samples}")
        if q.perplexity_before is not None:
            lines.append(f"- **Perplexity (dense):** {q.perplexity_before:.4f}")
        if q.perplexity_after is not None:
            lines.append(f"- **Perplexity (pruned):** {q.perplexity_after:.4f}")
        if q.perplexity_before and q.perplexity_after:
            ratio = q.perplexity_after / max(q.perplexity_before, 1e-10)
            lines.append(f"- **PPL ratio (after/before):** {ratio:.4f}")
        lines.append("")

    text = "\n".join(lines)
    with open(path, "w") as f:
        f.write(text)
    logger.info("Report markdown saved to %s", path)
    return path


def save_sparsity_json(
    per_layer: dict[str, float], overall: float, output_dir: str
) -> str:
    """Save sparsity report as JSON."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "sparsity_report.json")
    data = {"overall": overall, "per_layer": per_layer}
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info("Sparsity report saved to %s", path)
    return path


def save_quality_json(
    ppl_before: float | None,
    ppl_after: float | None,
    dataset: str,
    num_samples: int,
    output_dir: str,
) -> str:
    """Save quality report as JSON."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "quality_report.json")
    data = {
        "perplexity_before": ppl_before,
        "perplexity_after": ppl_after,
        "dataset": dataset,
        "num_samples": num_samples,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info("Quality report saved to %s", path)
    return path
