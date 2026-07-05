#!/usr/bin/env python3
"""Download a Hugging Face model for local experimentation."""

import argparse
import logging
import sys

sys.path.insert(0, ".")

from src.edgesparse.models.loader import load_model_and_tokenizer, save_pruned_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and cache a Hugging Face causal LM"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Hugging Face model name (default: Qwen/Qwen3-0.6B)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Local save path (default: None = cache only)",
    )
    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    logger.info("Downloading model: %s", args.model)

    model, tokenizer = load_model_and_tokenizer(args.model)

    if args.output:
        save_pruned_model(model, tokenizer, args.output)
        logger.info("Model saved to %s", args.output)
    else:
        logger.info("Model cached in Hugging Face cache directory.")


if __name__ == "__main__":
    main()
