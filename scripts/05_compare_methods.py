#!/usr/bin/env python3
"""Batch run multiple pruning methods × patterns and generate a comparison report."""

import argparse
import json
import logging
import os
import subprocess
import sys

sys.path.insert(0, ".")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


COMPARISON_CONFIGS = [
    # (method, pattern, sparsity_arg_name, sparsity_value)
    # "unstructured" patterns
    ("magnitude", "unstructured", "--sparsity", 0.25),
    ("magnitude", "unstructured", "--sparsity", 0.50),
    ("wanda", "unstructured", "--sparsity", 0.25),
    ("wanda", "unstructured", "--sparsity", 0.50),
    # N:M patterns
    ("wanda", "nm", "--n", 2, "--m", 4),
    ("wanda", "nm", "--n", 4, "--m", 8),
    # Block patterns
    ("wanda", "block", "--sparsity", 0.50, "--block-rows", 4, "--block-cols", 8),
    ("wanda", "block", "--sparsity", 0.50, "--block-rows", 8, "--block-cols", 8),
]


def run_experiment(
    model: str,
    method: str,
    pattern: str,
    sparsity_arg: str | None,
    sparsity_val: float | None,
    extra_args: list[str] | None,
    calib_samples: int,
    output_base: str,
) -> str | None:
    """Run 02_prune.py for one configuration and return the output dir."""
    import random
    import string

    tag = f"{model.split('/')[-1].lower()}_{method}_{pattern}"
    if sparsity_arg == "--sparsity" and sparsity_val is not None:
        tag += f"_{int(sparsity_val*100)}"
    elif sparsity_arg == "--n":
        tag += f"_{sparsity_val}"
        # Get the m value from extra_args
        m_idx = extra_args.index("--m") + 1 if extra_args and "--m" in extra_args else None
        if m_idx and m_idx < len(extra_args):
            tag += f"_{extra_args[m_idx]}"
    elif sparsity_arg in ("--block-rows",):
        br_idx = extra_args.index("--block-rows") + 1 if extra_args and "--block-rows" in extra_args else None
        bc_idx = extra_args.index("--block-cols") + 1 if extra_args and "--block-cols" in extra_args else None
        if br_idx and br_idx < len(extra_args):
            tag += f"_{extra_args[br_idx]}x{extra_args[bc_idx]}"

    output_dir = os.path.join(output_base, tag)

    # Skip if already done
    sp_report = os.path.join(output_dir, "sparsity_report.json")
    if os.path.exists(sp_report):
        logger.info("  [skip] %s (already exists)", output_dir)
        return output_dir

    cmd = [
        sys.executable,
        "scripts/02_prune.py",
        "--model", model,
        "--method", method,
        "--pattern", pattern,
        "--calib-samples", str(calib_samples),
        "--output", output_dir,
    ]

    if sparsity_arg and sparsity_val is not None:
        cmd.extend([sparsity_arg, str(sparsity_val)])
    if extra_args:
        cmd.extend(extra_args)

    logger.info("  Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error("  FAILED: %s\n%s", output_dir, result.stderr[-500:])
        return None

    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch compare pruning methods and patterns"
    )
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen3-0.6B", help="Model name"
    )
    parser.add_argument(
        "--calib-samples", type=int, default=64, help="Calibration samples"
    )
    parser.add_argument(
        "--output", type=str, default="outputs/comparison", help="Output base dir"
    )
    parser.add_argument(
        "--eval-samples", type=int, default=128, help="Eval samples for quality"
    )
    parser.add_argument(
        "--dense-eval", action="store_true", help="Run dense eval as baseline"
    )
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    results: list[str] = []

    # Optional: dense baseline eval
    if args.dense_eval:
        dense_dir = os.path.join(args.output, "dense_baseline")
        qr = os.path.join(dense_dir, "quality_report.json")
        if not os.path.exists(qr):
            logger.info("Running dense eval...")
            subprocess.run([
                sys.executable, "scripts/03_eval_quality.py",
                "--model", args.model,
                "--dataset", "Salesforce/wikitext",
                "--max-samples", str(args.eval_samples),
                "--seq-len", "2048",
                "--stride", "512",
                "--output", dense_dir,
            ])
        results.append(dense_dir)

    # Run each config
    for cfg in COMPARISON_CONFIGS:
        method, pattern = cfg[0], cfg[1]
        extra_idx = 2
        sparsity_arg = cfg[extra_idx]
        sparsity_val = cfg[extra_idx + 1] if len(cfg) > extra_idx + 1 and isinstance(cfg[extra_idx + 1], (int, float)) else None
        extra_args = list(cfg[extra_idx + 2:]) if len(cfg) > extra_idx + 2 else None

        out = run_experiment(
            args.model, method, pattern, sparsity_arg, sparsity_val,
            extra_args, args.calib_samples, args.output,
        )
        if out:
            results.append(out)

    # Run eval on all results
    for exp_dir in results:
        qr = os.path.join(exp_dir, "quality_report.json")
        if not os.path.exists(qr):
            logger.info("Evaluating %s...", exp_dir)
            subprocess.run([
                sys.executable, "scripts/03_eval_quality.py",
                "--model", exp_dir,
                "--dataset", "Salesforce/wikitext",
                "--max-samples", str(args.eval_samples),
                "--seq-len", "2048",
                "--stride", "512",
                "--output", exp_dir,
            ])

    # Generate comparison report
    from src.edgesparse.system.analyze import compare_experiments
    report = compare_experiments(results, args.output)
    print("\n" + report)


if __name__ == "__main__":
    main()
