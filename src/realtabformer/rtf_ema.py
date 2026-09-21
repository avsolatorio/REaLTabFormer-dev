"""Exponential moving average (EMA) of the model weights (tabular and relational models)."""

from __future__ import annotations

import math
from contextlib import contextmanager

import torch
from transformers import Seq2SeqTrainer, TrainerCallback


class WeightEMA:
    """A running exponential average of a model's parameters.

    The average is kept apart from the model: `update` folds the current weights
    in after each optimizer step, `averaged(model)` temporarily loads the average
    into the model (for sampling or saving) and restores the raw training weights
    on exit, and `copy_to(model)` overwrites the model with the average for good.

    On fixed-epoch learning curves (9 dataset x seed units, small GPT2) sampling
    from the average instead of the raw weights, at the same step, lowered
    marginal error by ~0.018 at epochs 30 and 50 (8/9 and 9/9 units better) and
    reached the raw weights' final marginal error at a median of epoch 30 instead
    of 100, at no extra training cost. A horizon of ~1 epoch was used; ~4 epochs
    lags badly early in training.
    """

    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError(f"decay must be in (0, 1), got {decay}")
        self.decay = decay
        self.shadow = [p.detach().clone() for p in model.parameters()]
        self._backup = None

    @staticmethod
    def decay_for_horizon(horizon_epochs: float, steps_per_epoch: int) -> float:
        """Per-step decay whose averaging time constant is `horizon_epochs`."""
        return math.exp(-1.0 / max(horizon_epochs * max(steps_per_epoch, 1), 1.0))

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        params = [p.detach() for p in model.parameters()]
        torch._foreach_mul_(self.shadow, self.decay)
        torch._foreach_add_(self.shadow, params, alpha=1.0 - self.decay)

    @torch.no_grad()
    def swap_in(self, model: torch.nn.Module) -> None:
        if self._backup is not None:  # already swapped in
            return
        self._backup = [p.detach().clone() for p in model.parameters()]
        torch._foreach_copy_([p.data for p in model.parameters()], self.shadow)

    @torch.no_grad()
    def swap_out(self, model: torch.nn.Module) -> None:
        if self._backup is None:  # nothing to restore
            return
        torch._foreach_copy_([p.data for p in model.parameters()], self._backup)
        self._backup = None

    @contextmanager
    def averaged(self, model: torch.nn.Module):
        self.swap_in(model)
        try:
            yield model
        finally:
            self.swap_out(model)

    @torch.no_grad()
    def copy_to(self, model: torch.nn.Module) -> None:
        torch._foreach_copy_([p.data for p in model.parameters()], self.shadow)


class EMACallback(TrainerCallback):
    """Feeds a `WeightEMA` after every optimizer step.

    The EMA lives in `holder["ema"]`, owned by the REaLTabFormer instance and shared
    across trainers, because the critic loop rebuilds the Trainer every `n_critic`
    epochs -- state kept in the callback would be lost each time. It is created at
    the start of the first training run, when the number of optimizer steps per
    epoch is known.
    """

    def __init__(self, holder: dict, horizon_epochs: float) -> None:
        self.holder = holder
        self.horizon_epochs = horizon_epochs

    def on_train_begin(self, args, state, control, model=None, train_dataloader=None, **kwargs):
        if "ema" not in self.holder and model is not None:
            steps_per_epoch = state.max_steps / max(state.num_train_epochs, 1)
            if args.max_steps > 0 and train_dataloader is not None:
                # `max_steps` overrides the epoch count, so `state.max_steps / num_train_epochs`
                # no longer means "steps per epoch"; count them from the data.
                steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
            self.holder["ema"] = WeightEMA(
                model, WeightEMA.decay_for_horizon(self.horizon_epochs, int(steps_per_epoch))
            )

    def on_step_end(self, args, state, control, model=None, **kwargs):
        ema = self.holder.get("ema")
        if ema is not None and model is not None:
            ema.update(model)


class EMASeq2SeqTrainer(Seq2SeqTrainer):
    """A `Seq2SeqTrainer` (the relational model's trainer) that evaluates and checkpoints the AVERAGED weights.

    The average is fed after every optimizer step by an `EMACallback`. Evaluation and every checkpoint save run with
    the average swapped in (and the raw training weights restored afterwards), so eval-loss early stopping compares
    averaged models and `load_best_model_at_end` returns averaged weights. Without `load_best_model_at_end` the
    trained model is still the raw one when `train()` returns; the caller copies the average in (see
    `REaLTabFormer.fit`).
    """

    def __init__(self, *args, ema_holder: dict, ema_horizon: float, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.ema_holder = ema_holder
        self.add_callback(EMACallback(ema_holder, ema_horizon))

    def evaluate(self, *args, **kwargs):
        ema = self.ema_holder.get("ema")
        if ema is None:  # before the first training step (or EMA not started yet)
            return super().evaluate(*args, **kwargs)
        with ema.averaged(self.model):
            return super().evaluate(*args, **kwargs)

    def save_model(self, *args, **kwargs):
        ema = self.ema_holder.get("ema")
        if ema is None:
            return super().save_model(*args, **kwargs)
        with ema.averaged(self.model):
            return super().save_model(*args, **kwargs)
