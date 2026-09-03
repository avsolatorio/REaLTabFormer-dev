"""CUSUM-based, generation-free overfitting/memorization detection.

Replaces (as an opt-in alternative to) the bootstrap-DCR sensitivity
mechanism in ``_train_with_sensitivity`` with a training-time monitor
that requires no ``.generate()`` calls and no additional held-out data.

Design summary (full derivation, literature grounding, and the
experiments that led to each choice are in the research log this
feature was built from):

- Per-token statistic: a "gated" (Winsorized) log-likelihood --
  ``log P(true_token)`` when the model's argmax already matches the
  true token, and a fixed floor (``log(1/vocab_size)``, "no better than
  a uniform guess") when it doesn't. This is continuous (unlike a
  binary exact-match indicator, which collapses to an unobservably rare
  event over a whole row) and bounded (unlike raw negative
  log-likelihood, whose unbounded penalty for wrong tokens can inject
  large outliers from ordinarily-hard rows).
- Per-row score: the mean gated statistic over a row's own valid target
  positions -- computed for free from the same forward pass already
  needed for the training loss (``reduction="none"`` instead of the
  batch mean).
- Reference: each row's own score at its one-time first exposure
  (before its first gradient update -- a genuinely prequential,
  zero-cost snapshot), compared against its own score later in
  training. This survives exhaustion of "never-yet-trained" rows in
  multi-epoch training on a fixed-size dataset (a real bug found and
  fixed during development: comparing two *populations* -- fresh vs.
  seen -- breaks once the fresh population runs out, which happens at
  the end of epoch 1 regardless of dataset size).
- Cooldown: a row only enters the comparison pool once at least
  ``cooldown_steps`` training steps have passed since it was last
  trained on. This excludes the ordinary post-gradient-step "recency
  bump" (a row just updated on fits it directly; that fit's extra
  contribution beyond the general training trend fades within roughly
  10-20 steps, confirmed by direct measurement) from being mistaken for
  memorization. Deliberately a small, fixed step count, not scaled to
  epoch length -- an epoch-scaled cooldown would make the whole
  mechanism unusable on a large dataset trained for only one or two
  epochs, since it could never accumulate enough steps to even start.
- Calibration: "normal," non-memorization-driven improvement is
  estimated from a window of checks once the cooldown pool becomes
  available (assumed safe -- real memorization is a many-exposure
  phenomenon, not something that appears in the first few checks). Two
  refinements found necessary from replaying a real training run's
  logged trajectory (a 100-epoch full-Adult-dataset run): (1) the pool
  right as it first becomes eligible is dominated by rows whose
  first-exposure baseline was captured at/near model initialization --
  the population-level swing from "near-random init" to "a few hundred
  steps in" is large and itself noisy check-to-check, which has nothing
  to do with memorization but inflates the calibrated noise scale by
  orders of magnitude if measured there. ``warmup_settle_checks``
  discards that many checks before calibration starts collecting, so
  calibration lands once the pool's mix of row ages has stabilized.
  (2) even past that point, the metric keeps drifting slowly for a
  while (ordinary learning-curve improvement, not memorization) --
  ``sigma0`` is estimated via first-differencing the calibration window
  rather than its raw standard deviation, since raw std conflates that
  drift with actual check-to-check noise and further inflates the
  scale (`_robust_noise_std`). (3) settle+warmup together are a *fixed
  step count* (``check_every``-spaced), which is fine on a large
  dataset but -- confirmed on a real 768-row run -- can silently
  consume 100+ epochs on a small one, by which point the calibrated
  "normal" baseline has already absorbed real memorization as normal
  and the detector gives essentially no protection.
  ``max_calibration_epochs`` bounds this by checking *more often*
  during calibration only (`adjust_calibration_pace_for_steps_per_epoch`)
  -- the full ``warmup_checks`` sample count is preserved, just
  gathered in fewer real steps, so ``sigma0`` doesn't get noisier the
  way shrinking the sample count would.
- Stopping rule: the classical Gaussian mean-shift CUSUM (Page, 1954)
  on the calibration-normalized z-score of each check's paired
  improvement. Lorden's theorem (1971) is the reason this is a
  principled choice, not just a heuristic: CUSUM minimizes the
  worst-case expected detection delay among all stopping rules for a
  given false-alarm-rate constraint. ``delta`` (the effect size CUSUM
  is tuned to detect) may be a single float (one tracker, the original
  behavior) or a sequence of floats (an ensemble: several trackers run
  in parallel on the same per-check z, alarm fires the moment any one
  crosses its own threshold) -- no single delta is well-matched to
  both a slow, gradual drift and a sharp, sudden one, and there's no
  way to know in advance which shape a given run's signal will take.
  Each tracker's threshold is Bonferroni-corrected (calibrated at a
  stricter quantile, the false-alarm budget split across trackers) so
  the ensemble's combined false-alarm rate stays at or below the
  single-tracker target (see ``_calibrate_thresholds``).

Known limitation, stated plainly: like any reactive/sequential
detector, this needs at least a little data past the true change point
before it can fire -- concretely, at least ``warmup_settle_checks +
warmup_checks`` post-cooldown checks before it can make any decision at
all. It cannot tell you in advance whether a single, non-repeatable
epoch of training already overfit; that requires a second, later
measurement to compare against, which by construction doesn't exist
yet after only one epoch.
"""

from __future__ import annotations

import math
import warnings
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

from .rtf_trainer import ResumableTrainer


def compute_gated_row_scores(
    logits: torch.Tensor,
    labels: torch.Tensor,
    floor_log_prob: float,
) -> torch.Tensor:
    """Per-row mean gated log-likelihood.

    ``logits``: (batch, seq_len, vocab) raw model output.
    ``labels``: (batch, seq_len) target ids, with ``-100`` at positions
      to ignore (matches HF's causal-LM convention).

    Uses the standard causal-LM shift (predictions at position i predict
    the token at position i+1), matches HF's own internal loss
    computation exactly (verified bit-for-bit against
    ``ForCausalLMLoss`` during development).
    """
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]

    valid = shift_labels != -100
    safe_labels = shift_labels.clamp(min=0)

    log_probs = F.log_softmax(shift_logits, dim=-1)
    true_log_prob = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)

    argmax_pred = shift_logits.argmax(-1)
    correct = argmax_pred == shift_labels

    gated = torch.where(
        correct, true_log_prob, torch.full_like(true_log_prob, floor_log_prob)
    )
    gated = torch.where(valid, gated, torch.zeros_like(gated))

    row_sum = gated.sum(dim=1)
    row_count = valid.sum(dim=1).clamp(min=1)
    return row_sum / row_count


def calibrate_gaussian_cusum_threshold(
    delta: float,
    n_checks_horizon: int,
    target_quantile: float = 0.99,
    n_sims: int = 3000,
    seed: Optional[int] = None,
) -> float:
    """Monte-Carlo calibration of the CUSUM decision threshold ``h``.

    Simulates the null hypothesis (a stream of iid standard-normal
    z-scores -- i.e. no real shift) ``n_sims`` times over a horizon of
    ``n_checks_horizon`` checks, and returns the ``target_quantile``
    quantile of each simulated run's maximum CUSUM path. This bounds
    the false-alarm rate at the actual scale of the real run (its
    number of checks), rather than reusing a threshold calibrated for a
    different horizon.
    """
    rng = np.random.default_rng(seed)
    llr_delta = delta
    llr_offset = delta**2 / 2
    max_S = np.empty(n_sims)
    for s in range(n_sims):
        z = rng.normal(0.0, 1.0, size=n_checks_horizon)
        llr = llr_delta * z - llr_offset
        running_max = 0.0
        acc = 0.0
        for x in llr:
            acc = max(0.0, acc + x)
            if acc > running_max:
                running_max = acc
        max_S[s] = running_max
    return float(np.quantile(max_S, target_quantile))


def _robust_noise_std(values: Sequence[float]) -> float:
    """Successive-difference ("Rice estimator") noise scale.

    Robust to a slow trend within ``values`` -- the raw std of a
    trending window conflates genuine drift with actual check-to-check
    noise, inflating the estimate. First-differencing cancels a
    slow/linear trend almost entirely while preserving the noise term:
    each first difference has variance ``2 * sigma**2`` under i.i.d.
    noise (hence the ``sqrt(2)`` correction), so this is consistent for
    the true noise scale whether or not a trend is present.
    """
    arr = np.asarray(values, dtype=float)
    if arr.size < 2:
        return 1e-6
    diffs = np.diff(arr)
    if diffs.size == 1:
        return max(float(abs(diffs[0])) / math.sqrt(2), 1e-6)
    return max(float(np.std(diffs, ddof=1)) / math.sqrt(2), 1e-6)


class CUSUMOverfittingMonitor:
    """Owns all state for the CUSUM overfitting detector.

    One instance is shared between the ``CUSUMTrainer`` (which feeds it
    per-row scores for free during each training step's forward pass)
    and the ``CUSUMEarlyStoppingCallback`` (which periodically asks it
    to run the actual detection check, using a resampled reference
    pool).
    """

    def __init__(
        self,
        vocab_size: int,
        check_every: int = 20,
        cooldown_steps: int = 40,
        warmup_checks: int = 10,
        warmup_settle_checks: Optional[int] = None,
        max_calibration_epochs: float = 10.0,
        seen_pool_size: int = 256,
        min_seen_pool: int = 32,
        delta: Union[float, Sequence[float]] = 0.5,
        target_quantile: float = 0.99,
        total_checks_horizon: Optional[int] = None,
        random_state: Optional[int] = None,
    ) -> None:
        if check_every <= 0:
            raise ValueError("check_every must be a positive integer.")
        if cooldown_steps < 0:
            raise ValueError("cooldown_steps must be non-negative.")
        if warmup_checks < 2:
            raise ValueError("warmup_checks must be at least 2 to estimate a variance.")
        if warmup_settle_checks is not None and warmup_settle_checks < 0:
            raise ValueError("warmup_settle_checks must be non-negative.")
        if max_calibration_epochs <= 0:
            raise ValueError("max_calibration_epochs must be positive.")

        # `delta` may be a single float (the original, still-default
        # behavior -- exactly one tracker) or a sequence of floats (an
        # ensemble: several CUSUM accumulators run in parallel, one per
        # assumed effect size, fed the SAME z-score each check -- see
        # the module docstring's "Stopping rule" bullet for why: no
        # single delta is well-matched to both a slow, gradual drift
        # and a sharp, sudden one, and there's no way to know in
        # advance which shape a given run's memorization signal will
        # take). Alarm fires the moment ANY tracker crosses ITS OWN
        # threshold.
        self.deltas: List[float] = (
            [float(delta)]
            if isinstance(delta, (int, float))
            else [float(d) for d in delta]
        )
        if not self.deltas:
            raise ValueError(
                "delta must be a positive float or a non-empty sequence of them."
            )
        if any(d <= 0 for d in self.deltas):
            raise ValueError("All delta values must be positive.")
        if len(set(self.deltas)) != len(self.deltas):
            raise ValueError("delta values in an ensemble must be distinct.")

        self.floor_log_prob = float(math.log(1.0 / vocab_size))
        self.check_every = check_every
        self.cooldown_steps = cooldown_steps
        self.warmup_checks = warmup_checks
        # Defaults to warmup_checks itself (a "settle" window as long as
        # the calibration window that follows it) -- see the module
        # docstring's "Calibration" bullet for why this exists: the pool
        # right as it first becomes eligible is dominated by rows whose
        # baseline was captured at/near model init, which makes the
        # earliest checks a bad place to calibrate "normal" noise from.
        self.warmup_settle_checks = (
            warmup_checks if warmup_settle_checks is None else warmup_settle_checks
        )
        self._settle_checks_remaining = self.warmup_settle_checks
        # `check_every` paces monitoring throughout the whole run, but a
        # fixed step count for settle+warmup (below) means calibration
        # itself can silently consume far more epochs on a small dataset
        # than on a large one -- found the hard way: on a real 768-row
        # dataset, calibration alone took 100 epochs before CUSUM ever
        # started watching, by which point the model had already
        # memorized enough that the "normal" baseline it calibrated
        # treated that memorization as normal, and the detector never
        # meaningfully protected against it. `max_calibration_epochs`
        # bounds that -- see `adjust_calibration_pace_for_steps_per_epoch`
        # -- by checking *more often* during calibration only, not by
        # collecting fewer calibration samples (which would just make
        # sigma0 noisier instead).
        self.max_calibration_epochs = max_calibration_epochs
        self._calibration_check_every = check_every
        self.seen_pool_size = seen_pool_size
        self.min_seen_pool = min_seen_pool
        self.target_quantile = target_quantile
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)

        # Calibrated lazily: the false-alarm-rate guarantee is only
        # meaningful once we know how many checks the run will actually
        # have. If the caller doesn't know the horizon up front (e.g.
        # training length isn't fixed ahead of time), a generous default
        # horizon is used and can be widened later via `extend_horizon`.
        self._horizon = total_checks_horizon or 1000
        self.cusum_h_by_delta: Dict[float, float] = self._calibrate_thresholds()

        self.baseline_score: Dict[int, float] = {}
        self.last_seen_step: Dict[int, int] = {}
        self.seen_indices: Set[int] = set()

        self.mu0: Optional[float] = None
        self.sigma0: Optional[float] = None
        self._warmup_deltas: List[float] = []

        self.cusum_S_by_delta: Dict[float, float] = {d: 0.0 for d in self.deltas}
        self.alarm_step: Optional[int] = None
        self.alarm_delta: Optional[float] = None
        self.alarm_checkpoint_dir: Optional[str] = None
        # (step, Delta, Z, S_by_delta) -- S_by_delta is a dict even in
        # the single-tracker (default) case, for one uniform shape.
        self.history: List[Tuple[int, float, float, Dict[float, float]]] = []
        # (step, delta) pairs for alarms an external confirmation check
        # (see CUSUMEarlyStoppingCallback's confirm_fn) investigated and
        # found to be false positives -- see reset_after_false_alarm.
        self.false_alarms: List[Tuple[int, float]] = []

    @property
    def delta(self) -> float:
        """Backward-compat alias for the primary (first) tracked delta."""
        return self.deltas[0]

    @property
    def cusum_S(self) -> float:
        """Backward-compat alias for the primary tracker's accumulator."""
        return self.cusum_S_by_delta[self.deltas[0]]

    @property
    def cusum_h(self) -> float:
        """Backward-compat alias for the primary tracker's threshold."""
        return self.cusum_h_by_delta[self.deltas[0]]

    def _calibrate_thresholds(self) -> Dict[float, float]:
        """Monte-Carlo-calibrates a threshold for every tracked delta.

        With more than one delta (an ensemble), calibrating each
        independently at ``target_quantile`` and firing on "any tracker
        crosses" would inflate the true combined false-alarm rate above
        the target -- multiple looks at the same data. Bonferroni-
        corrects for this: each tracker is calibrated at a stricter
        quantile (the false-alarm budget split across trackers), so the
        ensemble's overall false-alarm rate stays at or below the
        nominal target (real trackers are positively correlated --
        sharing the same z each check -- so this errs conservatively,
        firing falsely less often than the budget, never more).
        """
        n = len(self.deltas)
        far = 1.0 - self.target_quantile
        adjusted_quantile = 1.0 - (far / n)
        return {
            d: calibrate_gaussian_cusum_threshold(
                delta=d,
                n_checks_horizon=self._horizon,
                target_quantile=adjusted_quantile,
                seed=self.random_state,
            )
            for d in self.deltas
        }

    def extend_horizon(self, total_checks_horizon: int) -> None:
        """Recalibrate every tracker's threshold once the true check horizon is known."""
        if total_checks_horizon <= self._horizon:
            return
        self._horizon = total_checks_horizon
        self.cusum_h_by_delta = self._calibrate_thresholds()

    def adjust_cooldown_for_steps_per_epoch(self, steps_per_epoch: int) -> None:
        """Caps ``cooldown_steps`` against the ACTUAL observed steps-per-
        epoch, found necessary the hard way: a row only stops being
        "recently touched" once ``cooldown_steps`` have passed since its
        last training step, but if the training loop cycles back through
        the whole dataset faster than that (i.e. ``cooldown_steps >=
        steps_per_epoch`` -- possible even at "reasonable" dataset sizes,
        since HF's default `gradient_accumulation_steps` multiplies how
        much data one optimizer step covers, shrinking steps_per_epoch
        well below what a naive per-row-batch-size estimate would
        suggest), EVERY row gets re-touched and its cooldown clock reset
        before it can ever qualify -- the reference pool stays
        permanently empty and the detector never activates, silently.
        Caps at half an epoch's worth of steps so every row spends at
        least part of each cycle eligible, regardless of dataset size,
        batch size, or gradient accumulation -- a no-op whenever the
        configured cooldown already comfortably fits (the common case
        on any reasonably-sized dataset).
        """
        if steps_per_epoch <= 0:
            return
        safe_cap = max(1, steps_per_epoch // 2)
        if self.cooldown_steps > safe_cap:
            self.cooldown_steps = safe_cap

    def adjust_calibration_pace_for_steps_per_epoch(self, steps_per_epoch: int) -> None:
        """Paces settle+warmup checks faster on small datasets, so
        calibration -- a fixed ``warmup_settle_checks + warmup_checks``
        checks -- can't silently consume more than
        ``max_calibration_epochs`` epochs before CUSUM starts actually
        monitoring. Confirmed on real data: on a 768-row dataset, the
        default ``check_every=20`` meant calibration alone took ~100
        epochs (400 steps at 4 steps/epoch), and by then the model had
        already memorized enough that the calibrated "normal" baseline
        (``mu0``) treated that memorization as normal -- the detector's
        eventual alarm gave essentially no protection (its
        ``frac_suspicious`` matched an unrestricted full-schedule run).

        Deliberately checks *more often* during calibration rather than
        collecting *fewer* calibration samples (the more obvious lever)
        -- that would keep calibration fast but make the ``sigma0``
        estimate noisier from having fewer data points. This keeps the
        full ``warmup_checks`` sample count, just gathers it in fewer
        real steps; ``check_every`` itself (the post-calibration
        monitoring pace) is untouched. A no-op whenever calibration
        already comfortably fits the budget (the common case on any
        reasonably large dataset, e.g. Adult).
        """
        if steps_per_epoch <= 0:
            return
        total_checks = self.warmup_settle_checks + self.warmup_checks
        if total_checks <= 0:
            return
        budget_steps = max(
            total_checks, round(self.max_calibration_epochs * steps_per_epoch)
        )
        self._calibration_check_every = max(
            1, min(self.check_every, budget_steps // total_checks)
        )

    @property
    def current_check_every(self) -> int:
        """The check interval to use right now: the (possibly tighter)
        calibration pace while ``mu0`` is still unset, ``check_every``
        itself once calibrated."""
        if self.mu0 is None:
            return self._calibration_check_every
        return self.check_every

    def record_batch(
        self,
        row_idx: Sequence[int],
        logits: torch.Tensor,
        labels: torch.Tensor,
        step: int,
    ) -> None:
        """Called once per training step, from inside the loss
        computation -- costs nothing extra beyond a per-row (rather
        than batch-mean) reduction of a quantity already computed.
        """
        with torch.no_grad():
            scores = compute_gated_row_scores(logits, labels, self.floor_log_prob)
        scores_list = scores.detach().to("cpu").tolist()
        for idx, score in zip(row_idx, scores_list):
            idx = int(idx)
            if idx not in self.baseline_score:
                self.baseline_score[idx] = float(score)
            self.seen_indices.add(idx)
            self.last_seen_step[idx] = step

    def _cooled_pool(self, step: int) -> List[int]:
        cooldown = self.cooldown_steps
        return [
            idx
            for idx in self.seen_indices
            if (step - self.last_seen_step[idx]) >= cooldown
        ]

    def maybe_check(
        self,
        step: int,
        model: torch.nn.Module,
        get_rows,
    ) -> bool:
        """Run one detection check, if enough cooled reference rows are
        available. Returns True exactly on the step the alarm fires
        (i.e. once, not on every subsequent step after firing).

        ``get_rows(indices)`` must return ``(input_ids, labels)`` torch
        tensors for the given row indices from the training dataset.
        """
        pool = self._cooled_pool(step)
        if len(pool) < self.min_seen_pool:
            return False

        if self.mu0 is None and self._settle_checks_remaining > 0:
            # Consume settle checks without paying for a forward pass --
            # the calibration window hasn't started yet, so there's
            # nothing to score against.
            self._settle_checks_remaining -= 1
            return False

        sample_size = min(self.seen_pool_size, len(pool))
        sample_idx = self._rng.choice(pool, size=sample_size, replace=False).tolist()

        was_training = model.training
        model.eval()
        with torch.no_grad():
            input_ids, labels = get_rows(sample_idx)
            outputs = model(input_ids=input_ids, labels=labels)
            current = compute_gated_row_scores(
                outputs.logits, labels, self.floor_log_prob
            )
        if was_training:
            model.train()

        current_np = current.detach().to("cpu").numpy()
        base_np = np.array([self.baseline_score[idx] for idx in sample_idx])
        paired_diff = current_np - base_np

        n = len(sample_idx)
        Delta = float(paired_diff.mean())
        var = float(paired_diff.var(ddof=1)) if n > 1 else float("nan")

        if self.mu0 is None:
            self._warmup_deltas.append(Delta)
            if len(self._warmup_deltas) >= self.warmup_checks:
                self.mu0 = float(np.mean(self._warmup_deltas))
                self.sigma0 = _robust_noise_std(self._warmup_deltas)
            return False

        if np.isnan(var) or var <= 0:
            return False

        se = float(np.sqrt(var / n + self.sigma0**2))
        if se <= 0:
            return False

        z = (Delta - self.mu0) / se
        # z doesn't depend on delta -- computed once, shared by every
        # tracker in the ensemble. Only the LLR increment (and each
        # tracker's own threshold) does.
        fired_delta = None
        for d in self.deltas:
            llr = d * z - d**2 / 2
            s = max(0.0, self.cusum_S_by_delta[d] + llr)
            self.cusum_S_by_delta[d] = s
            if fired_delta is None and s >= self.cusum_h_by_delta[d]:
                fired_delta = d
        self.history.append((step, Delta, z, dict(self.cusum_S_by_delta)))

        if fired_delta is not None and self.alarm_step is None:
            self.alarm_step = step
            self.alarm_delta = fired_delta
            return True
        return False

    def reset_after_false_alarm(self) -> None:
        """Called when an external confirmation check (e.g. a
        sensitivity-style generate-and-DCR check, see
        `CUSUMEarlyStoppingCallback`'s `confirm_fn`) investigates a
        fired alarm and determines it was a false positive -- clears
        the accumulated CUSUM evidence (`cusum_S_by_delta`) and the
        `alarm_step`/`alarm_delta` markers so monitoring genuinely
        resumes and can fire again on NEW evidence, rather than sitting
        at/above threshold forever (`maybe_check`'s `self.alarm_step is
        None` guard means it would otherwise never fire a second time).

        Standard practice for a change-point detector after an
        investigated false alarm: restart the accumulator. `mu0`/
        `sigma0` (the calibrated "normal" noise baseline) are left
        untouched -- they're independent of the alarm event itself and
        presumably still valid; only the accumulated drift EVIDENCE is
        discarded, not the calibration.
        """
        if self.alarm_step is not None:
            self.false_alarms.append((self.alarm_step, self.alarm_delta))
        self.cusum_S_by_delta = {d: 0.0 for d in self.deltas}
        self.alarm_step = None
        self.alarm_delta = None


class CUSUMTrainer(ResumableTrainer):
    """``ResumableTrainer`` that also feeds per-row gated scores to a
    ``CUSUMOverfittingMonitor``, for free, from the training forward
    pass already being computed.

    The dataset must have an ``idx`` integer column identifying each
    row (added by ``REaLTabFormer._fit_tabular(..., add_row_idx=True)``)
    -- it's popped out of ``inputs`` before being passed to the model,
    since the model itself doesn't accept it as a forward argument.
    """

    def __init__(
        self, *args, cusum_monitor: Optional[CUSUMOverfittingMonitor] = None, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.cusum_monitor = cusum_monitor

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        row_idx = inputs.pop("idx", None)
        loss, outputs = super().compute_loss(
            model, inputs, return_outputs=True, **kwargs
        )
        if row_idx is not None and self.cusum_monitor is not None and model.training:
            self.cusum_monitor.record_batch(
                row_idx.detach().to("cpu").tolist(),
                outputs.logits,
                inputs["labels"],
                self.state.global_step,
            )
        return (loss, outputs) if return_outputs else loss


class CUSUMEarlyStoppingCallback(TrainerCallback):
    """Runs the periodic CUSUM check and requests training stop when it
    fires. Requires the training dataset to be indexable by row (a plain
    ``datasets.Dataset`` with ``input_ids``/``labels`` columns works).

    Saves the exact model state at the moment the alarm fires to
    ``alarm_checkpoint_dir`` (if given), via a direct
    ``model.save_pretrained(...)`` call -- independent of HF's own
    step-based checkpoint/``should_save`` machinery and, critically, of
    ``load_best_model_at_end`` (which -- if left enabled -- reloads
    whatever checkpoint had the best held-out eval loss once
    ``trainer.train()`` returns, for ANY stop reason, silently replacing
    the alarm-point model with an unrelated one picked by a different
    criterion). The caller (``_train_with_cusum``) also disables
    ``load_best_model_at_end`` for this path so the in-memory model
    stays exactly where training stopped, but this explicit, on-disk
    save is the actual guarantee: the alarm-point weights exist on disk
    the instant the alarm fires, regardless of any other mechanism's
    later behavior. Still sets ``control.should_save = True`` too, for
    interoperability with ``resume_from_checkpoint``.

    ``confirm_fn``, if given, turns a fired alarm into a candidate
    rather than an immediate stop: called as ``confirm_fn(model)`` the
    moment CUSUM fires, it does its own (typically much more expensive
    but more trustworthy) check and returns ``True`` to confirm the
    stop, or ``False`` to treat it as a false alarm -- in which case
    ``monitor.reset_after_false_alarm()`` is called and training
    continues, un-stopped, with CUSUM's accumulated evidence cleared so
    it can genuinely fire again later on NEW evidence. Exists because a
    real investigation (see this feature's own research log) found
    CUSUM sometimes fires on a slow, sustained-but-genuine improvement
    trend it can't statistically distinguish from memorization onset --
    a cheap, trustworthy confirmation at the actual candidate stop
    point (rather than a periodic schedule from epoch 0, which is what
    the sensitivity mechanism this can reuse for confirmation already
    pays for) directly resolves that ambiguity without needing a new
    detection theory. ``None`` (default): unchanged behavior, alarm
    always stops training immediately.
    """

    def __init__(
        self,
        monitor: CUSUMOverfittingMonitor,
        train_dataset,
        alarm_checkpoint_dir: Optional[str] = None,
        confirm_fn: Optional[Callable[[torch.nn.Module], bool]] = None,
    ) -> None:
        self.monitor = monitor
        self.train_dataset = train_dataset
        self.alarm_checkpoint_dir = alarm_checkpoint_dir
        self.confirm_fn = confirm_fn

    def _get_rows(self, indices: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        rows = self.train_dataset[indices]
        device = next(self._model.parameters()).device
        # torch.as_tensor (not torch.tensor) since the dataset may already
        # be torch-formatted (`set_format(type="torch", ...)`, used when
        # this feature is threaded on top of a branch that also needs
        # field_weights/digit_entropy_weighting's own extra columns) --
        # re-wrapping an existing tensor via torch.tensor(...) triggers a
        # UserWarning and an avoidable copy; as_tensor is a no-op then.
        input_ids = torch.as_tensor(rows["input_ids"], device=device)
        labels = torch.as_tensor(rows["labels"], device=device)
        return input_ids, labels

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        effective_batch_size = max(
            1, args.per_device_train_batch_size * args.gradient_accumulation_steps
        )
        steps_per_epoch = max(1, len(self.train_dataset) // effective_batch_size)
        prior_cooldown = self.monitor.cooldown_steps
        self.monitor.adjust_cooldown_for_steps_per_epoch(steps_per_epoch)
        if self.monitor.cooldown_steps != prior_cooldown:
            warnings.warn(
                f"cusum_cooldown_steps={prior_cooldown} would never be "
                f"satisfiable at this training's actual steps-per-epoch "
                f"({steps_per_epoch}, from per_device_train_batch_size="
                f"{args.per_device_train_batch_size} x "
                f"gradient_accumulation_steps="
                f"{args.gradient_accumulation_steps}) -- every row would be "
                f"re-touched before it could cool down, and the detector "
                f"would never activate. Capped to "
                f"{self.monitor.cooldown_steps} steps instead."
            )
        self.monitor.adjust_calibration_pace_for_steps_per_epoch(steps_per_epoch)

        total_steps = (
            state.max_steps if state.max_steps and state.max_steps > 0 else None
        )
        if total_steps:
            horizon = total_steps // self.monitor.check_every + 10
            self.monitor.extend_horizon(horizon)
        return control

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        step = state.global_step
        if step == 0 or step % self.monitor.current_check_every != 0:
            return control

        model = kwargs.get("model")
        if model is None:
            return control
        self._model = model

        alarmed = self.monitor.maybe_check(step, model, self._get_rows)
        if alarmed and self.confirm_fn is not None:
            # `confirm_fn` typically calls `.sample()` to generate real
            # data for its own check -- generation is normally called
            # BETWEEN separate `trainer.train()` calls elsewhere in this
            # codebase (`_train_with_sensitivity`'s own periodic checks),
            # never from inside an ACTIVE one. Found the hard way: doing
            # it from here, mid-`trainer.train()`, can leave the model
            # split across devices (generation-related device placement
            # disagreeing with the live Trainer's own), crashing the
            # very next training step with an opaque device-mismatch
            # error. Capture the device beforehand and force the model
            # back onto it afterward, regardless of what confirm_fn did
            # internally -- cheap, and makes this safe by construction
            # rather than by trusting `.sample()` never to move it.
            was_training = model.training
            device = next(model.parameters()).device
            confirmed = self.confirm_fn(model)
            model.to(device)
            if was_training:
                model.train()
            if not confirmed:
                self.monitor.reset_after_false_alarm()
                alarmed = False
        if alarmed:
            control.should_training_stop = True
            control.should_save = True
            if self.alarm_checkpoint_dir is not None:
                model.save_pretrained(self.alarm_checkpoint_dir)
                self.monitor.alarm_checkpoint_dir = self.alarm_checkpoint_dir
        return control
