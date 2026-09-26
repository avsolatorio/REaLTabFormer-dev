"""Verification instrumentation for RL fine-tuning runs (see `rtf_rl.py`/`rtf_rl_relational.py`).
Built as a standing process after real bugs (dropout-in-train-mode, a checkpoint-eval sizing
crash, a data-prep NaN-to-string bug, a reward-discretisation bug) were found at various points
during the GMD survey-synthesis investigation this was promoted from -- some of them only after
they'd already informed a conclusion. The lesson: verification should not be a one-off check run
when something looks suspicious, but a standing part of how every RL run is produced and read, so
an anomaly shows up in the run's own log instead of requiring a later reconstruction.

Three layers:
  - `BatchDiagnostics`: cheap, per-round checks during training (unique-value counts for numeric
    reward targets, reward finiteness/scale) -- printed live, so a problem is visible while the run
    is still going, not only after the fact.
  - `audit_trajectory`: a post-hoc scan of a finished run's checkpoint trajectory for anomaly
    SHAPES that matter regardless of the specific reward (a metric jumping far more between two
    checkpoints than the run's own trajectory noise would suggest, a value outside a physically
    sane range, a metric that never moved at all).
  - `PatienceStopper`/`MultiMetricPatienceStopper`: an ONLINE stopping decision for the RL loop,
    mirroring REaLTabFormer's own `n_critic`/`n_critic_stop` mechanism for the MLE side (periodic
    checkpoint evaluation, stop after `patience` consecutive non-improving checks), but with a
    NOISE-INFORMED epsilon measured fresh per run rather than a hard threshold.

Conservative by design: `BatchDiagnostics`/`audit_trajectory` flag things to LOOK AT, not things to
silently correct or hide. A flag does not mean the run is wrong; it means check before trusting it.
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np
import pandas as pd


class BatchDiagnostics:
    """Call `.check(...)` once per RL round with that round's decoded batch and reward. Cheap
    (no model forward passes), so it runs unconditionally, not just when something seems off.
    """

    def __init__(self, numeric_columns: Sequence[str], low_cardinality_bins: int = 10) -> None:
        self.numeric_columns = numeric_columns
        self.low_cardinality_bins = low_cardinality_bins
        self.flags: List[str] = []
        self._reward_range = None

    def check(self, round_: int, batch_df: pd.DataFrame, reward: np.ndarray) -> List[str]:
        this_round_flags = []

        # 1. A numeric reward-target column collapsing to few enough unique values that a
        # discretisation scheme's bin-reuse path could misbehave, or a genuinely degenerate model
        # output that's worth flagging regardless.
        for c in self.numeric_columns:
            if c not in batch_df.columns:
                continue
            n_unique = pd.to_numeric(batch_df[c], errors="coerce").nunique()
            if n_unique <= self.low_cardinality_bins:
                msg = f"round {round_}: '{c}' collapsed to {n_unique} unique values in this batch (<= {self.low_cardinality_bins})"
                self.flags.append(msg)
                this_round_flags.append(msg)

        # 2. Reward must be finite. A NaN/Inf reward would silently propagate into the advantage
        # and the gradient with no visible symptom other than a model that mysteriously breaks.
        if not np.all(np.isfinite(reward)):
            msg = f"round {round_}: reward contains {np.sum(~np.isfinite(reward))} non-finite value(s)"
            self.flags.append(msg)
            this_round_flags.append(msg)

        # 3. Reward that never varies within a batch (std==0) makes group_relative_advantage
        # divide by its own epsilon floor -- not broken, but a dead training signal worth knowing
        # about rather than discovering only when a whole run shows no movement at all.
        if reward.std() == 0:
            msg = f"round {round_}: reward has zero variance across the batch (every sample scored identically)"
            self.flags.append(msg)
            this_round_flags.append(msg)

        return this_round_flags


class PatienceStopper:
    """Sensitivity-style ONLINE stopping for the RL loop: call `.update(value)` once per checkpoint
    evaluation (lower `value` = better; negate a "higher is better" metric before passing it in).

    `epsilon` should be measured FRESH per run from that run's own checkpoint-eval noise (e.g. the
    std of several independent evaluation draws on the frozen pre-RL model), not a guessed
    constant -- mirrors REaLTabFormer's own `sensitivity_threshold` being bootstrap-derived from
    real data at the start of each fit call, not hardcoded. A value only counts as a genuine
    improvement if it beats the running best by MORE than `epsilon`; anything within that band is
    "no improvement" for stopping purposes (though the CALLER's separate, more sensitive
    best-checkpoint tracking -- unaffected by this class -- can still keep it as the reported best,
    since there's no harm in preferring a noise-sized improvement even if it isn't stop-worthy).

    Stops after `patience` CONSECUTIVE non-improving checks -- exactly REaLTabFormer's own
    `n_critic_stop` pattern, applied to a continuous noise-aware metric instead of a hard
    threshold-breach test.
    """

    def __init__(self, epsilon: float, patience: int, initial_best: float) -> None:
        assert epsilon >= 0 and patience >= 1
        self.epsilon = epsilon
        self.patience = patience
        self.best = initial_best
        self.rounds_since_improve = 0
        self.history: List[float] = []

    def update(self, value: float) -> bool:
        """Record one checkpoint's value (lower = better). Returns True if the caller should stop
        NOW (this call already counted toward the decision)."""
        self.history.append(value)
        if self.best - value > self.epsilon:
            self.best = value
            self.rounds_since_improve = 0
        else:
            self.rounds_since_improve += 1
        return self.rounds_since_improve >= self.patience


class MultiMetricPatienceStopper:
    """Like `PatienceStopper`, but tracks SEVERAL metrics simultaneously and only counts a
    checkpoint as "no improvement" when NONE of them improved beyond its own noise-derived epsilon.

    Useful whenever a reward targets several statistics that can plateau on different timescales: a
    single-metric stopper watching only the fastest-plateauing one would stop the whole run while
    slower-moving targets are still genuinely improving. Requiring ALL tracked metrics to
    simultaneously stop improving is the direct fix: "still making progress on ANY tracked
    objective" is sufficient reason to keep going, matching the fact this is a multi-objective
    reward, not a single scalar to plateau-detect.

    `window` (default 1, a no-op): compares a TRAILING MEAN of the last `window` raw readings of
    each metric against `best`, instead of the single latest raw reading -- for when even a
    reasonably-sized single checkpoint's reading of a noisy statistic can swamp a real, gradual
    trend (the same noise-averaging principle as `EMAFrequencyTracker` in `rtf_rl.py` and this
    library's own weight-EMA feature). `epsilon` should be scaled down accordingly by the CALLER (a
    trailing mean of `window` roughly-independent draws has std smaller by about sqrt(window), so
    passing the raw single-draw epsilon unchanged would make stopping needlessly conservative) --
    this class does not do that scaling itself, since it has no way to know if the caller already
    accounted for it.
    """

    def __init__(self, epsilons: dict, patience: int, initial_best: dict, window: int = 1) -> None:
        assert set(epsilons) == set(initial_best), (epsilons, initial_best)
        assert patience >= 1 and window >= 1
        self.epsilons = dict(epsilons)
        self.patience = patience
        self.window = window
        self.best = dict(initial_best)
        self.rounds_since_any_improve = 0
        self.history: List[dict] = []
        self.raw_history: dict = {k: [] for k in initial_best}

    def update(self, values: dict) -> bool:
        """Record one checkpoint's metric values (lower = better for every key). Returns True if
        the caller should stop NOW.

        While the trailing window isn't full yet (fewer than `window` readings recorded), the
        "smoothed" value would just be an under-smoothed partial average -- comparing THAT against
        `best` risks a single early lucky-low raw reading locking in as an unbeatable `best` before
        smoothing ever gets a chance to apply. So these warm-up calls are recorded but don't count
        toward the patience decision at all.
        """
        self.history.append(dict(values))
        smoothed = {}
        ready = True
        for k, v in values.items():
            self.raw_history[k].append(v)
            if len(self.raw_history[k]) < self.window:
                ready = False
            smoothed[k] = float(np.mean(self.raw_history[k][-self.window:]))
        if not ready:
            return False
        improved = False
        for k, v in smoothed.items():
            if self.best[k] - v > self.epsilons[k]:
                self.best[k] = v
                improved = True
        if improved:
            self.rounds_since_any_improve = 0
        else:
            self.rounds_since_any_improve += 1
        return self.rounds_since_any_improve >= self.patience


def should_act_on_stop(stop_signal: bool, round_: int, min_stop_round: int) -> bool:
    """Gate a stopper's raw signal by a minimum-round floor. The stopper's own patience/window
    bookkeeping (`MultiMetricPatienceStopper.update`) still runs every checkpoint regardless of this
    floor, so it stays warmed up; only ACTING on a True result is gated here.

    Useful because a checkpoint schedule is often densest early, so a short patience window can
    trigger long before training reaches the round range where real gains actually show up --
    `min_stop_round` should be set from a run's own observed best-checkpoint rounds, not guessed.
    """
    return stop_signal and round_ >= min_stop_round


def per_column_tail_err(train: pd.DataFrame, synth: pd.DataFrame) -> dict:
    """Decompose an aggregate tail-error fidelity metric (`|synth_q99 - train_q99| / scale`, the
    same formula this library's other fidelity tooling uses) into its PER-COLUMN contributions, so
    a regression in the aggregate can be traced to the specific column responsible instead of
    guessed at.
    """
    out = {}
    for c in train.columns:
        if pd.api.types.is_numeric_dtype(train[c]) and train[c].nunique() > 10:
            a = train[c].astype(float).to_numpy()
            b = pd.to_numeric(synth[c], errors="coerce").astype(float).dropna().to_numpy()
            q1, q99 = np.quantile(a, [0.01, 0.99])
            scale = max(q99 - q1, 1e-9)
            out[c] = dict(
                train_q99=float(q99), synth_q99=float(np.quantile(b, 0.99)) if len(b) else float("nan"),
                tail_err=float(abs(np.quantile(b, 0.99) - q99) / scale) if len(b) else float("nan"),
            )
    return out


def audit_trajectory(checkpoints: list, before: dict, metric: str = "disc_auc", jump_factor: float = 3.0) -> List[str]:
    """Post-hoc scan of a finished run's checkpoint trajectory. Flags:
      - a jump between consecutive checkpoints more than `jump_factor` times the trajectory's own
        median absolute step (a sudden discontinuity, as opposed to the gradual drift a real
        reward-hacking trajectory typically shows -- a real bug looks different from real
        reward-hacking, and this is meant to tell them apart);
      - any `disc_auc`-named metric value outside [0, 1] (a physically impossible value -- would
        indicate a scoring bug, not a model-quality issue);
      - a trajectory that is EXACTLY flat (every checkpoint bit-identical), which would indicate the
        reward or the optimizer step silently did nothing rather than converged.
    Returns a list of human-readable flags; empty means nothing anomalous was found (not proof of
    correctness, only that these specific, previously-seen anomaly shapes are absent).
    """
    flags = []
    values = [before.get(metric)] + [c.get(metric) for c in checkpoints]
    values = [v for v in values if v is not None]
    if len(values) < 3:
        return ["not enough checkpoints to audit a trajectory"]

    if metric == "disc_auc" and any((v < 0 or v > 1) for v in values):
        flags.append(f"{metric} value outside [0,1] somewhere in the trajectory: {values}")

    diffs = np.abs(np.diff(values))
    if len(diffs) >= 3:
        med_step = np.median(diffs)
        if med_step > 0:
            for i, d in enumerate(diffs):
                if d > jump_factor * med_step and d > 0.05:  # also require an absolute-magnitude floor
                    flags.append(
                        f"discontinuous jump at checkpoint {i}: |delta {metric}|={d:.4f}, "
                        f"{d / med_step:.1f}x the trajectory's own median step ({med_step:.4f})"
                    )

    if len(set(np.round(values, 6))) == 1:
        flags.append(f"{metric} is EXACTLY flat across the whole trajectory ({values[0]}) -- suspicious, not just stable")

    return flags
