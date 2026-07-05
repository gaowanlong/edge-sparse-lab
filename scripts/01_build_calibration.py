#!/usr/bin/env python3
"""Build and save a calibration dataset & activation norms."""

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, ".")

from src.edgesparse.models.loader import load_model_and_tokenizer
from src.edgesparse.calibration.dataset import get_calibration_dataset
from src.edgesparse.calibration.activation_capture import capture_activations

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build calibration data and capture activation norms"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Model name or path",
    )
    parser.add_argument(
        "--calib-dataset",
        type=str,
        default="Salesforce/wikitext",
        help="Calibration dataset name",
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
        help="Sequence length for calibration",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/calibration",
        help="Output directory for calibration data",
    )
    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    os.makedirs(args.output, exist_ok=True)

    # 1. Load model
    model, tokenizer = load_model_and_tokenizer(args.model)

    # 2. Get calibration data
    calib_ids = get_calibration_dataset(
        name=args.calib_dataset,
        subset=args.calib_subset,
        split="train",
        max_samples=args.calib_samples,
        seq_len=args.seq_len,
        tokenizer=tokenizer,
    )

    # Save calibration input IDs
    calib_path = os.path.join(args.output, "calib_input_ids.pt")
    import torch
    torch.save(calib_ids, calib_path)
    logger.info("Calibration data saved to %s", calib_path)

    # 3. Capture activation norms
    norms = capture_activations(model, calib_ids)

    # Save norms
    norms_path = os.path.join(args.output, "activation_norms.pt")
    torch.save(norms, norms_path)
    logger.info("Activation norms saved to %s", norms_path)

    # Also save metadata
    meta = {
        "model": args.model,
        "dataset": args.calib_dataset,
        "num_samples": args.calib_samples,
        "seq_len": args.seq_len,
        "num_layers": len(norms),
    }
    meta_path = os.path.join(args.output, "calibration_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("Calibration metadata saved to %s", meta_path)

    logger.info("Done. Calibration data ready at %s", args.output)


if __name__ == "__main__":
    main()
