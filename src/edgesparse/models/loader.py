"""Hugging Face causal LM loading with MPS / CPU support."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    """Select the best available device: MPS > CPU."""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        logger.info("Falling back to CPU")
    return device


def load_model_and_tokenizer(
    model_name_or_path: str,
    dtype: torch.dtype = torch.float16,
    use_cache: bool = True,
    trust_remote_code: bool = False,
    device_map: str | None = None,
) -> tuple["AutoModelForCausalLM", "AutoTokenizer"]:
    """Load a Hugging Face causal LM and its tokenizer.

    Args:
        model_name_or_path: HF hub name or local path.
        dtype: Model weight dtype (default float16 for memory efficiency).
        use_cache: Enable KV cache.
        trust_remote_code: Allow remote code execution.
        device_map: Device map string (e.g. "auto", "mps", "cpu").
            If None, uses the best available device.

    Returns:
        (model, tokenizer)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = get_device() if device_map is None else torch.device(device_map)

    logger.info("Loading model %s …", model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        use_cache=use_cache,
        trust_remote_code=trust_remote_code,
        device_map=None,  # manual placement
        low_cpu_mem_usage=True,
    )
    model = model.to(device=device, dtype=dtype)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    logger.info(
        "Model loaded: %s  |  params %.0fM  |  device %s  |  dtype %s",
        model_name_or_path,
        sum(p.numel() for p in model.parameters()) / 1e6,
        device,
        dtype,
    )
    return model, tokenizer


def save_pruned_model(
    model: "AutoModelForCausalLM",
    tokenizer: "AutoTokenizer",
    save_dir: str,
) -> None:
    """Save model weights and tokenizer to disk."""
    import os

    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir, safe_serialization=True)
    tokenizer.save_pretrained(save_dir)
    logger.info("Pruned model saved to %s", save_dir)
