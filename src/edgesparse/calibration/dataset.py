"""Calibration and evaluation dataset utilities."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


def get_calibration_dataset(
    name: str = "Salesforce/wikitext",
    subset: str = "wikitext-2-raw-v1",
    split: str = "train",
    max_samples: int = 128,
    seq_len: int = 2048,
    tokenizer: "PreTrainedTokenizerBase | None" = None,
) -> torch.Tensor:
    """Load and tokenize a calibration dataset.

    Returns a tensor of shape (N, seq_len) with padding to seq_len.
    """
    from datasets import load_dataset

    logger.info("Loading calibration dataset: %s/%s (%s)", name, subset, split)
    dataset = load_dataset(name, subset, split=split, trust_remote_code=False)
    dataset = dataset.filter(lambda x: x["text"] and x["text"].strip())
    if max_samples and len(dataset) > max_samples:
        dataset = dataset.select(range(max_samples))

    if tokenizer is None:
        raise ValueError("tokenizer is required")

    texts = list(dataset["text"])
    encoded = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=seq_len,
        return_tensors="pt",
    )
    logger.info("Calibration data shape: %s", encoded.input_ids.shape)
    return encoded.input_ids


def get_eval_dataset(
    name: str = "Salesforce/wikitext",
    subset: str = "wikitext-2-raw-v1",
    split: str = "test",
    max_samples: int | None = None,
    seq_len: int = 2048,
    tokenizer: "PreTrainedTokenizerBase | None" = None,
) -> torch.Tensor:
    """Load and tokenize an evaluation dataset.

    Returns a 1D tensor of *concatenated* token IDs (no padding).
    This is the correct format for sliding-window perplexity evaluation.
    """
    from datasets import load_dataset

    logger.info("Loading eval dataset: %s/%s (%s)", name, subset, split)
    dataset = load_dataset(name, subset, split=split, trust_remote_code=False)
    dataset = dataset.filter(lambda x: x["text"] and x["text"].strip())
    if max_samples and len(dataset) > max_samples:
        dataset = dataset.select(range(max_samples))

    if tokenizer is None:
        raise ValueError("tokenizer is required")

    texts = list(dataset["text"])

    # Tokenize each text individually (no padding), then concatenate
    all_ids: list[int] = []
    for text in texts:
        ids = tokenizer.encode(text, truncation=True, max_length=seq_len)
        all_ids.extend(ids)

    result = torch.tensor(all_ids, dtype=torch.long)
    logger.info(
        "Eval data: %d samples → %d tokens", len(texts), result.size(0)
    )
    return result
