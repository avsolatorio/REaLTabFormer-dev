"""Training losses for the tabular model."""

from __future__ import annotations

from typing import Callable

import torch

from .data_utils import SpecialTokens
from .rtf_sampler import ColumnMaskLogitsProcessor


def build_constrained_loss(rtf) -> Callable:
    """Cross-entropy under the same per-column token mask that sampling uses.

    Every position is normalised over only the tokens valid for that position's
    column (`rtf.col_idx_ids`, the set generation is constrained to), instead of
    over the whole vocabulary. Generation never lets the model emit a token from
    another column, so the standard loss spends capacity teaching the model which
    tokens are legal where -- a rule the sampler enforces anyway -- and this loss
    does not.

    Measured against the standard loss on fixed-epoch learning curves (8
    dataset x seed units, small GPT2): at epoch 10 the discriminator distance from
    0.5 is 0.163 lower (8/0 units better) and association error 0.029 lower (8/0);
    the reference's final marginal error is reached in a median of 50 epochs
    instead of 100. The ceiling is unchanged, and it overfits faster (held-out
    NLL and late-epoch discriminator score are worse), so it is an efficiency
    option to use with a stopping rule, not a quality upgrade.

    The mask is built lazily on the first call, from the logits' device and the
    model's vocabulary size, because `rtf.col_idx_ids` only exists once the
    vocabulary has been built.
    """
    state: dict = {}

    def loss_fn(outputs, labels, num_items_in_batch=None):
        logits = outputs.logits[:, :-1, :].float()
        target = labels[:, 1:]
        if "mask" not in state:
            eos = rtf.vocab["token2id"][SpecialTokens.EOS]
            state["mask"] = ColumnMaskLogitsProcessor(
                rtf.col_idx_ids,
                eos,
                logits.shape[-1],
                rtf.tabular_max_length,
                logits.device,
            ).mask
        allowed = state["mask"][: logits.shape[1]].unsqueeze(0)
        logp = torch.log_softmax(logits.masked_fill(~allowed, float("-inf")), dim=-1)
        valid = target != -100
        nll = -logp.gather(-1, target.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        # A target token outside its column's valid set (e.g. an OOV substitution)
        # would be +inf; cap it rather than poison the batch.
        nll = torch.nan_to_num(nll, posinf=40.0).clamp(max=40.0)
        return (nll * valid).sum() / valid.sum().clamp(min=1)

    return loss_fn
