"""Perplexity evaluation for causal LMs using sliding-window."""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from transformers import AutoModelForCausalLM, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


@torch.no_grad()
def evaluate_perplexity(
    model: "AutoModelForCausalLM",
    tokenizer: "PreTrainedTokenizerBase",
    input_ids: torch.Tensor,
    stride: int = 512,
    max_length: int | None = None,
) -> float:
    """Compute perplexity on a tokenized dataset using sliding-window evaluation.

    Accepts either:
    - 1D tensor: concatenated token IDs (from get_eval_dataset with no padding)
    - 2D tensor (N, seq_len): batched, padded token IDs

    Args:
        model: The model to evaluate.
        tokenizer: Tokenizer (used for context length).
        input_ids: 1D (concatenated) or 2D (batched) token IDs.
        stride: Sliding window stride.
        max_length: Model maximum context length. Auto-detected if None.

    Returns:
        Perplexity score.
    """
    model.eval()
    device = next(model.parameters()).device

    if max_length is None:
        max_length = model.config.max_position_embeddings or 2048
    max_length = min(max_length, 2048)

    # Handle both 1D (concatenated) and 2D (batched) input
    if input_ids.ndim == 2:
        # Flatten all sequences into one long sequence
        all_tokens = input_ids.view(-1)
    else:
        all_tokens = input_ids  # already 1D

    nll = 0.0
    nll_count = 0
    total_tokens = all_tokens.size(0)

    for i in range(0, total_tokens - 1, stride):
        end_loc = min(i + max_length, total_tokens - 1)
        trg_len = max_length if end_loc - i == max_length else end_loc - i
        begin_loc = end_loc - trg_len

        input_block = all_tokens[begin_loc:end_loc].unsqueeze(0)  # (1, T)
        label_block = input_block.clone()
        # Only the last trg_len tokens have valid targets
        label_block[:, :-trg_len] = -100
        label_block = label_block.to(device)

        outputs = model(input_block.to(device))
        logits = outputs.logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = label_block[..., 1:].contiguous()

        loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="sum",
        )
        nll += loss.item()
        nll_count += (shift_labels != -100).sum().item()

        if (i // stride) % 10 == 0:
            logger.debug(
                "  ppl so far: %.2f  (pos %d / %d)",
                math.exp(nll / max(nll_count, 1)),
                end_loc,
                total_tokens,
            )

    if nll_count == 0:
        return float("inf")

    ppl = math.exp(nll / nll_count)
    logger.info("Perplexity: %.4f", ppl)
    return ppl
