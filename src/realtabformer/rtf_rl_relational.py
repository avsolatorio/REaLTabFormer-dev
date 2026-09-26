"""RL fine-tuning building blocks for a REaLTabFormer RELATIONAL (parent/child) model. The
parent-side mechanism (`rtf_rl.py`'s `log_prob_of_sequences`) does NOT transfer unmodified, for two
independently-verified reasons.

1. Masking mechanism. `TabularSampler` overrides `_constraint_logits_processor` with the fast
   vectorised `ColumnMaskLogitsProcessor` (a precomputed `[steps, vocab]` tensor); `RelationalSampler`
   does not (only one override in rtf_sampler.py, inside `TabularSampler`'s class body). Real
   relational sampling constrains tokens via the slower per-row
   `_get_relational_col_idx_ids(len(input_ids))` callback instead, whose step->column mapping CYCLES
   with period `col_size + 2` (one cycle per relational row group), not the tabular case's one-shot,
   monotonically-advancing mapping.

2. Indexing convention. `ColumnMaskLogitsProcessor.__call__` uses `step = len(input_ids) - 1` to
   predict the token about to be generated. `_get_relational_col_idx_ids` is called as
   `_get_relational_col_idx_ids(len(input_ids))` -- no `-1`. So for a teacher-forced forward pass
   with `decoder_input_ids` of length L, `logits[:, i, :]` (predicting position `i+1`) must be
   compared against `_get_relational_col_idx_ids(i + 1)`, not `_get_relational_col_idx_ids(i)`. A
   naive port of the tabular off-by-one convention would silently mask the wrong position.

This module precomputes its mask by calling the sampler's OWN `_get_relational_col_idx_ids`, never
reimplementing the cyclic arithmetic -- the same reason `rtf_rl.log_prob_of_sequences` mirrors
`ColumnMaskLogitsProcessor` instead of hand-deriving the tabular mask.

Promoted from the GMD (Vietnam VHLSS household-survey synthesis) investigation's own
`gmd_rl_relational.py` (see the independent `synthetic-gmd` repo's lab notebook, G26-G60). That
project's own `household_reward` (which bridges to its `household_table`'s survey-specific column
assumptions) is NOT included here -- it stays in that project, as the one GMD-specific piece on top
of this otherwise general mechanism; build an equivalent reward function for any other reward table
using `rtf_rl.marginal_joint_reward` directly.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import torch


def build_relational_mask(sampler, max_steps: int, vocab_size: int, device: torch.device) -> torch.Tensor:
    """`[max_steps + 1, vocab_size]` boolean mask, position `p` = valid tokens for decoder sequence
    position `p` (predicted from `logits[:, p-1, :]` in a teacher-forced forward pass; position 0,
    the decoder_start token, is never predicted and its row is left all-False by construction).
    Built by calling the sampler's real `_get_relational_col_idx_ids(p)` for every `p`, exactly
    reproducing what real generation constrains position `p` to, not a reimplementation of the
    modulo arithmetic.
    """
    mask = torch.zeros(max_steps + 1, vocab_size, dtype=torch.bool)
    for p in range(1, max_steps + 1):
        valid = sampler._get_relational_col_idx_ids(p)
        mask[p, valid] = True
    return mask.to(device)


def relational_log_prob_of_sequences(
    model, enc_input_ids: torch.Tensor, dec_sequences: torch.Tensor, mask: torch.Tensor,
    eos_token_id: int, requires_grad: bool = False,
) -> torch.Tensor:
    """Sum of log p(token_p | encoder row, decoder tokens_{<p}) under the model's OWN constrained
    decoding distribution, for p = 1..seq_len-1 (position 0 is the decoder_start token, never
    predicted). Mirrors `rtf_rl.log_prob_of_sequences` in every respect that DOES carry over from
    the tabular case: dropout forced off regardless of `requires_grad`, tokens at or after a
    sequence's own EOS excluded from the sum, train/eval mode restored on exit.

    `enc_input_ids`: (batch, enc_len) encoder (parent-row) token ids, already expanded to match
    `dec_sequences`' batch size (one encoder row per decoder rollout, even when several rollouts
    share the same parent row -- see `build_relational_mask`'s docstring on GRPO groups).
    `dec_sequences`: (batch, dec_len) decoder token ids, decoder_start-first, as produced by
    `RelationalSampler._generate`.
    `mask`: from `build_relational_mask`, built with the SAME `max_steps` >= `dec_sequences.shape[1]`.
    """
    was_training = model.training
    model.eval()
    try:
        with torch.enable_grad() if requires_grad else torch.no_grad():
            out = model(input_ids=enc_input_ids, decoder_input_ids=dec_sequences)
            logits = out.logits  # (batch, dec_len, vocab)
    finally:
        model.train(was_training)

    dec_len = dec_sequences.shape[1]
    device = logits.device
    total = torch.zeros(dec_sequences.shape[0], device=device, dtype=logits.dtype)
    active = torch.ones(dec_sequences.shape[0], dtype=torch.bool, device=device)
    for i in range(dec_len - 1):
        position = i + 1  # predicting position i+1 from logits[:, i, :] -- see module docstring
        step_logits = logits[:, i, :].masked_fill(~mask[position], float("-inf"))
        logp = torch.log_softmax(step_logits, dim=-1)
        tok = dec_sequences[:, position]
        token_logp = logp.gather(1, tok.unsqueeze(1)).squeeze(1)
        total = total + torch.where(active, token_logp, torch.zeros_like(token_logp))
        active = active & (tok != eos_token_id)
    return total


def encode_parent_batch(parent_df: pd.DataFrame, in_col_transform_data: dict, vocab_encoder: dict, device: torch.device) -> torch.Tensor:
    """Encoder (parent-row) token ids for a batch of already-real (not yet re-encoded) parent rows.
    `in_col_transform_data`/`vocab_encoder` come from the fitted relational model
    (`cm.in_col_transform_data`, `cm.vocab["encoder"]`)."""
    from .data_utils import process_data
    from .data_utils.dataset import make_dataset

    processed, _, _ = process_data(parent_df, col_transform_data=in_col_transform_data)
    ds = make_dataset(processed, vocab_encoder, seed=0)
    return torch.tensor(ds["input_ids"], device=device)


def sample_child_rollouts(sampler, model, device, enc_ids: torch.Tensor, eos_token_id: int, bos_token_id: int, pad_token_id: int, max_length: int, seed: int) -> np.ndarray:
    """One decoder rollout per row of `enc_ids` (batch of encoder token ids, one per parent row).
    Returns the raw token id matrix (batch, dec_len), decoder_start-first, as produced by
    `RelationalSampler._generate` -- NOT yet decoded to a dataframe (see `decode_child_batch`)."""
    torch.manual_seed(seed)
    raw = sampler._generate(
        device=device, as_numpy=True, constrain_tokens_gen=True, inputs=enc_ids, do_sample=True,
        suppress_tokens=None, max_length=max_length,
        bos_token_id=bos_token_id, pad_token_id=pad_token_id, eos_token_id=eos_token_id,
    )
    return raw


def decode_child_batch(sampler, raw_sequences: np.ndarray, vocab_decoder: dict, relate_ids: Sequence, key: str = "id") -> pd.DataFrame:
    """Decode raw rollout token sequences into a child dataframe, one relational-key-linked row per
    generated child row -- reuses `REaLSampler._processes_sample`, the same decoder the library's
    own `sample_relational` uses, rather than reimplementing token->value decoding.

    `_processes_sample` returns `relate_ids` as the dataframe's INDEX, not a column -- callers that
    need it as an explicit column should do the same `.rename_axis(key).reset_index()` this
    function does (matching how `gmd_pipeline.py` already handles it).
    """
    out = sampler._processes_sample(sample_outputs=raw_sequences, vocab=vocab_decoder, relate_ids=list(relate_ids))
    return out.rename_axis(key).reset_index()
