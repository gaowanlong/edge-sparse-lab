#!/usr/bin/env python3
"""Apply pruning to a model and save the pruned weights + reports."""

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, ".")

import torch

from src.edgesparse.models.loader import load_model_and_tokenizer, save_pruned_model
from src.edgesparse.calibration.dataset import get_calibration_dataset
from src.edgesparse.calibration.activation_capture import capture_activations
from src.edgesparse.pruning.magnitude import magnitude_prune
from src.edgesparse.pruning.wanda import wanda_prune
from src.edgesparse.pruning.structured_nm import nm_prune
from src.edgesparse.pruning.block_sparse import block_prune
from src.edgesparse.eval.report import (
    save_config,
    save_sparsity_json,
    SparsityReport,
)
from src.edgesparse.pruning.mask_utils import compute_sparsity

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prune a causal LM and save the result"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Model name or path",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["magnitude", "wanda"],
        default="wanda",
        help="Pruning method",
    )
    parser.add_argument(
        "--sparsity",
        type=float,
        default=None,
        help="Target sparsity (for unstructured / block)",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        choices=["unstructured", "nm", "block"],
        default="unstructured",
        help="Pruning pattern",
    )
    parser.add_argument("--n", type=int, default=None, help="N for N:M pruning")
    parser.add_argument("--m", type=int, default=None, help="M for N:M pruning")
    parser.add_argument(
        "--block-rows", type=int, default=None, help="Block rows for block sparse"
    )
    parser.add_argument(
        "--block-cols", type=int, default=None, help="Block cols for block sparse"
    )
    parser.add_argument(
        "--calib-dataset",
        type=str,
        default="Salesforce/wikitext",
        help="Calibration dataset",
    )
    parser.add_argument(
        "--calib-subset",
        type=str,
        default="wikitext-2-raw-v1",
    )
    parser.add_argument(
        "--calib-samples",
        type=int,
        default=128,
        help="Number of calibration samples",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=2048,
        help="Sequence length",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for pruned model and reports",
    )
    parser.add_argument(
        "--precomputed-calib",
        type=str,
        default=None,
        help="Path to precomputed calibration dir (activations + norms)",
    )
    args = parser.parse_args()

    # Validate args
    if args.pattern == "nm":
        if args.n is None or args.m is None:
            parser.error("--n and --m are required for NM pattern")
        if args.sparsity is None:
            args.sparsity = 1.0 - args.n / args.m
    elif args.pattern == "block":
        if args.block_rows is None or args.block_cols is None:
            parser.error("--block-rows and --block-cols required for block pattern")
    if args.pattern in ("unstructured", "block") and args.sparsity is None:
        parser.error("--sparsity is required for unstructured and block patterns")

    os.makedirs(args.output, exist_ok=True)

    # Save config
    config = vars(args)
    save_config(config, args.output)

    # 1. Load model
    model, tokenizer = load_model_and_tokenizer(args.model)

    # 2. Calibration (if needed)
    activation_norms = None
    if args.method == "wanda":
        if args.precomputed_calib:
            norms_path = os.path.join(args.precomputed_calib, "activation_norms.pt")
            if os.path.exists(norms_path):
                activation_norms = torch.load(norms_path, map_location="cpu")
                logger.info("Loaded precomputed norms from %s", norms_path)
            else:
                logger.warning(
                    "Precomputed norms not found at %s, recomputing...", norms_path
                )

        if activation_norms is None:
            logger.info("Capturing activations for Wanda...")
            calib_ids = get_calibration_dataset(
                name=args.calib_dataset,
                subset=args.calib_subset,
                split="train",
                max_samples=args.calib_samples,
                seq_len=args.seq_len,
                tokenizer=tokenizer,
            )
            activation_norms = capture_activations(model, calib_ids)

    # 3. Prune
    logger.info(
        "Pruning with method=%s pattern=%s sparsity=%s",
        args.method,
        args.pattern,
        args.sparsity,
    )

    if args.method == "magnitude":
        per_layer = magnitude_prune(model, args.sparsity)
    elif args.method == "wanda":
        if args.pattern == "unstructured":
            per_layer = wanda_prune(
                model, activation_norms, sparsity=args.sparsity, pattern="unstructured"
            )
        elif args.pattern == "nm":
            per_layer = wanda_prune(
                model,
                activation_norms,
                pattern="nm",
                n=args.n,
                m=args.m,
            )
        elif args.pattern == "block":
            per_layer = wanda_prune(
                model,
                activation_norms,
                sparsity=args.sparsity,
                pattern="block",
                block_rows=args.block_rows,
                block_cols=args.block_cols,
            )

    # 4. Compute overall sparsity
    total = sum(p.numel() for p in model.parameters())
    nonzero = sum(p.count_nonzero().item() for p in model.parameters())
    overall = 1.0 - nonzero / total if total > 0 else 0.0

    logger.info("Overall sparsity: %.4f (%d / %d params remain)", overall, nonzero, total)

    # 5. Save sparsity report
    save_sparsity_json(per_layer, overall, args.output)

    # 6. Save model
    save_pruned_model(model, tokenizer, args.output)


if __name__ == "__main__":
    main()
