#!/usr/bin/env python3
"""Evaluate perplexity of a (dense or pruned) model on a dataset."""

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, ".")

from src.edgesparse.models.loader import load_model_and_tokenizer
from src.edgesparse.calibration.dataset import get_eval_dataset
from src.edgesparse.eval.perplexity import evaluate_perplexity
from src.edgesparse.eval.report import save_quality_json
from src.edgesparse.pruning.mask_utils import compute_sparsity

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate perplexity of a sparse model"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name or path to pruned model directory",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="Salesforce/wikitext",
        help="Evaluation dataset",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="wikitext-2-raw-v1",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=128,
        help="Maximum evaluation samples",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=2048,
        help="Sequence length",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=512,
        help="Sliding window stride for perplexity",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for reports (default: same as model dir)",
    )
    args = parser.parse_args()

    # Determine output dir
    output_dir = args.output or args.model

    # 1. Load model
    logger.info("Loading model from %s", args.model)
    model, tokenizer = load_model_and_tokenizer(args.model)

    # Print sparsity info
    total = sum(p.numel() for p in model.parameters())
    nonzero_ = sum(p.count_nonzero().item() for p in model.parameters())
    sparsity = 1.0 - nonzero_ / total if total > 0 else 0.0
    logger.info(
        "Model sparsity: %.4f (%d / %d params remain)",
        sparsity,
        nonzero_,
        total,
    )

    # 2. Load eval data
    input_ids = get_eval_dataset(
        name=args.dataset,
        subset=args.subset,
        split=args.split,
        max_samples=args.max_samples,
        seq_len=args.seq_len,
        tokenizer=tokenizer,
    )

    # 3. Evaluate perplexity
    ppl = evaluate_perplexity(
        model,
        tokenizer,
        input_ids,
        stride=args.stride,
    )

    logger.info("Perplexity on %s: %.4f", args.dataset, ppl)

    # 4. Save report
    save_quality_json(
        ppl_before=None,
        ppl_after=ppl,
        dataset=args.dataset,
        num_samples=args.max_samples,
        output_dir=output_dir,
    )

    # 5. Also print summary
    print()
    print("=" * 60)
    print(f"Model:     {args.model}")
    print(f"Sparsity:  {sparsity:.4%}")
    print(f"Dataset:   {args.dataset} ({args.max_samples} samples)")
    print(f"Perplexity: {ppl:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
