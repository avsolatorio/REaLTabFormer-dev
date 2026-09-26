"""RL fine-tuning building blocks for a REaLTabFormer TABULAR model: GRPO-style policy-gradient
fine-tuning against a reward built directly from marginal/joint target distributions (as opposed
to an indirect, black-box discriminator score).

`log_prob_of_sequences` is the correctness-critical piece: the log-probability of an ALREADY-
SAMPLED sequence under the model's OWN constrained-decoding distribution -- i.e., exactly the
distribution `TabularSampler` actually sampled from, column-validity mask included. This is the
single easiest place for a silent, invisible bug: if the mask used here doesn't exactly match the
mask used at generation time (wrong step-index convention, wrong vocab_size, a stale
`col_idx_ids`, ...), the computed "probabilities" don't correspond to any real distribution the
model implements, the policy-gradient direction is then mathematically wrong, and nothing about
running it would look obviously broken -- it would just quietly optimise against the wrong
objective and produce numbers that could easily be misread as a real (positive OR negative)
result. Verified against an EXACT, falsifiable property before anything is built on top of it:
enumerate every possible sequence for a tiny toy schema (small enough to enumerate completely),
and check the computed probabilities sum to EXACTLY 1 (see `tests/realtabformer/test_rtf_rl.py`).

`compute_target_stats` / `marginal_joint_reward` implement a reward built DIRECTLY from marginal
and joint distributions, rewarding a sample by how much it would move the CURRENT BATCH's own
empirical statistics toward the TRUE (training-data) statistics, for a chosen set of column
marginals and column-pair joints.

None of this is specific to survey data or any other particular use case -- it is a general RL
fine-tuning mechanism for any REaLTabFormer/REaLTabFormer2 tabular model, used identically whether
the target statistics come from a survey's population shares or any other reference distribution.
Promoted from the GMD (Vietnam VHLSS household-survey synthesis) investigation's own `gmd_rl.py`
(see the independent `synthetic-gmd` repo's lab notebook, entries G17-G60, for the empirical
history) -- generalized by removing that project's one real dependency
(`regression_influence_reward` used to import an OLS helper from that project's own metrics
module; it now carries its own small, generic version).
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch


def log_prob_of_sequences(
    model, sequences: torch.Tensor, col_idx_ids: dict, eos_token_id: int, vocab_size: int, max_steps: int,
    requires_grad: bool = False,
) -> torch.Tensor:
    """Sum of log p(token_i | tokens_{<i}) under the model's OWN constrained-decoding distribution,
    for i = 1..seq_len-1 (position 0 is BOS, never predicted). Tokens at or after a sequence's own
    EOS don't contribute (matches how generation actually stops: nothing after EOS was "chosen"
    under this distribution, so it must not enter the sum).

    `sequences`: LongTensor (batch, seq_len) on `model`'s device, BOS-first, as produced by
    `TabularSampler._generate` (i.e., the raw generate() output, not yet decoded to columns).
    The other args must be the EXACT values the sampler used -- see
    `REaLSampler._constraint_logits_processor` in rtf_sampler.py, which this mirrors precisely:
    `max_steps = max(rtf_model.tabular_max_length, len(rtf_model.col_idx_ids))`,
    `vocab_size = rtf_model.model.config.vocab_size`,
    `eos_token_id = rtf_model.vocab["token2id"][SpecialTokens.EOS]`.
    """
    from .rtf_sampler import ColumnMaskLogitsProcessor

    # Dropout must be OFF regardless of `requires_grad`: if the model is left in training mode,
    # dropout is stochastically active and independently sampled PER ROW -- rows sharing an
    # identical prefix (e.g. two sequences differing only in their last column) would then get
    # DIFFERENT logits for that shared prefix, which is impossible under proper causal attention
    # and silently breaks the "sums to 1" property this module exists to guarantee. (Found exactly
    # this way in the originating investigation: an enumerated-probability self-test summed to
    # 0.991, not 1.0, traced to three sequences sharing a prefix getting three different
    # probabilities for it.) `requires_grad=True` (for an actual RL update) still disables
    # dropout, only re-enables gradient tracking; the caller's train/eval mode is restored on the
    # way out either way.
    was_training = model.training
    model.eval()
    try:
        device = sequences.device
        proc = ColumnMaskLogitsProcessor(col_idx_ids, eos_token_id, vocab_size, max_steps, device)
        with torch.enable_grad() if requires_grad else torch.no_grad():
            logits = model(input_ids=sequences).logits  # (batch, seq_len, vocab) -- one forward pass, teacher-forced
    finally:
        model.train(was_training)

    seq_len = sequences.shape[1]
    total = torch.zeros(sequences.shape[0], device=device, dtype=logits.dtype)
    active = torch.ones(sequences.shape[0], dtype=torch.bool, device=device)
    for i in range(1, seq_len):
        step_logits = logits[:, i - 1, :]
        mask = proc.mask[min(i - 1, proc.n_steps - 1)]
        masked = step_logits.masked_fill(~mask, float("-inf"))
        logp = torch.log_softmax(masked, dim=-1)
        tok = sequences[:, i]
        token_logp = logp.gather(1, tok.unsqueeze(1)).squeeze(1)
        total = total + torch.where(active, token_logp, torch.zeros_like(token_logp))
        active = active & (tok != eos_token_id)
    return total


def group_relative_advantage(rewards: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """GRPO's core mechanism: normalise rewards by the GROUP's own mean/std, so the update signal
    is "better or worse than the alternatives sampled for this same prompt", not a raw reward scale
    that's awkward to calibrate in absolute terms (a discriminator probability, a rule-satisfaction
    flag, ... don't have an obvious "good" absolute value, but relative ranking within a group is
    meaningful). For an UNCONDITIONAL generator (a tabular model: every "prompt" is the same empty
    BOS, so there is only one group), this is mathematically the batch as a whole -- the same thing
    as the classic "REINFORCE with a batch-mean baseline" variance-reduction trick, which predates
    GRPO by decades. GRPO's per-prompt grouping becomes a genuinely different, non-degenerate
    mechanism once prompts differ (a relational/child model, conditioned per parent row) -- noted
    here rather than glossed over, since claiming this IS "real GRPO" in the unconditional case
    would overstate what's actually being used; see `grouped_relative_advantage` for that case.
    """
    return (rewards - rewards.mean()) / (rewards.std(unbiased=False) + eps)


def grouped_relative_advantage(rewards: torch.Tensor, group_size: int, eps: float = 1e-8) -> torch.Tensor:
    """Per-prompt GRPO: `rewards` is `n_groups * group_size` values, ordered so every consecutive
    block of `group_size` is K independent rollouts of the SAME prompt (e.g. the same parent row's
    encoder input, repeated). Normalises each group by its OWN mean/std, not the whole batch's --
    the baseline for rollout i of group h becomes "better or worse than the OTHER K-1 rollouts of
    THIS SAME group", not "better or worse than K*n_groups rollouts of DIFFERENT groups", which is
    what `group_relative_advantage` gives when called on a batch of different groups.

    `group_size=1` is meaningless here (a lone rollout has no within-group variance to normalise
    against, giving a degenerate all-zero advantage) and is intentionally NOT treated as reducing
    to `group_relative_advantage`'s pooled behaviour -- a caller wanting the pooled/no-K-rollout
    case should call `group_relative_advantage` directly, not this function with group_size=1. The
    reduction this function DOES guarantee (see `tests/realtabformer/test_rtf_rl.py`):
    `group_size = len(rewards)` (a single group spanning the WHOLE input) reproduces
    `group_relative_advantage` exactly.
    """
    assert rewards.numel() % group_size == 0, (
        f"rewards ({rewards.numel()}) must be an exact multiple of group_size ({group_size})"
    )
    r = rewards.view(-1, group_size)
    mean = r.mean(dim=1, keepdim=True)
    std = r.std(dim=1, unbiased=False, keepdim=True)
    return ((r - mean) / (std + eps)).view(-1)


def grpo_loss(
    new_logp: torch.Tensor, ref_logp: torch.Tensor, advantage: torch.Tensor, kl_coef: float,
    sample_weight: torch.Tensor = None,
) -> dict:
    """Policy-gradient loss with a group-relative baseline (`advantage`, already normalised by
    `group_relative_advantage`/`grouped_relative_advantage`) plus a KL penalty toward a FROZEN
    reference policy.

    The KL term is the guard against the documented failure mode of this whole technique family
    (adversarial/reward-driven sequence generation, SeqGAN-lineage): reward hacking toward a
    narrow, high-reward-but-low-diversity mode. It keeps the fine-tuned policy close to the
    MLE-trained reference, whose broad calibration is worth preserving, and only nudges it toward
    whatever the reward signal is rewarding.

    `new_logp` MUST have been computed with `requires_grad=True`; `ref_logp` must be detached (the
    reference model is never updated). `advantage` is treated as a fixed target (`.detach()`ed
    here) -- rewards are not backpropagated through, only used to weight the log-prob gradient,
    which is what makes this a valid (if not variance-optimal) policy-gradient estimator.

    `sample_weight` (optional, default None = plain `.mean()`, i.e. no behaviour change): a
    per-row weight aligned with this batch (e.g. a survey design weight). When given, both terms
    become a WEIGHTED mean (`sum(w*x)/sum(w)`) instead of a plain mean -- mathematically invariant
    to any constant rescaling of the weights, so passing raw, un-normalised weights is fine here.
    This is the same `sum(w*x)/sum(w)` convention `WeightedLabelSmoother` (rtf_trainer.py) uses
    for the MLE loss, so the same weight column is safely reusable in both places.
    """
    advantage = advantage.detach()
    if sample_weight is None:
        policy_loss = -(advantage * new_logp).mean()
        kl = (new_logp - ref_logp.detach()).mean()  # standard k1-style MC estimate of KL(policy || ref)
    else:
        w = sample_weight.detach()
        denom = w.sum().clamp_min(1e-12)
        policy_loss = -((advantage * new_logp) * w).sum() / denom
        kl = ((new_logp - ref_logp.detach()) * w).sum() / denom
    return dict(loss=policy_loss + kl_coef * kl, policy_loss=policy_loss.detach(), kl=kl.detach())


# --------------------------------------------------------------------------------------------
# A reward built directly from marginal and joint (pairwise) target distributions.
# --------------------------------------------------------------------------------------------


def detect_heavy_tailed_columns(
    df: pd.DataFrame, columns: Sequence[str], min_unique: int = 30, tail_ratio_threshold: float = 0.85,
    quantiles: Tuple[float, float, float] = (0.5, 0.95, 0.99),
) -> list:
    """Auto-detect which of `columns` are heavy-tailed enough to need tail-aware quantile bins
    (`TAIL_AWARE_QUANTILES` below) or a `tail_boost` (see `marginal_joint_reward`), instead of a
    hand-curated list.

    Two-step rule, each independently motivated, not combined by guesswork:
      1. Exclude any column with `min_unique` or fewer distinct values. A bounded count or id has
         COARSE, discrete quantile spacing by construction, which can mimic a heavy tail without
         being one.
      2. Among columns that pass step 1, compute `(q_hi - q_mid) / (q_mid - q_lo)` at `quantiles`
         (default the median, 95th and 99th percentiles). A ratio near or above 1 means the extreme
         tail (95th-99th) is comparably or more spread out than the entire bulk (50th-95th) -- a
         genuine heavy tail, not an artifact of discreteness.
    """
    out = []
    for c in columns:
        s = pd.to_numeric(df[c], errors="coerce").dropna().astype(float)
        if s.nunique() <= min_unique:
            continue
        q_lo, q_mid, q_hi = np.quantile(s, quantiles)
        ratio = (q_hi - q_mid) / max(q_mid - q_lo, 1e-9)
        if ratio >= tail_ratio_threshold:
            out.append(c)
    return out


# Default: 10 equal-FREQUENCY bins (deciles). A column in `tail_aware_columns` instead gets bin
# edges placed at these exact quantile points -- coarse in the bulk, much finer approaching the
# extreme tail. 10 equal-frequency bins put the ENTIRE top 10% of a column into ONE bin, so a
# reward built from those bins can reward "be somewhere in the top decile" without caring at all
# whether a sample lands near the 90th or the 99.9th percentile within it -- zero gradient toward
# a metric that specifically measures the 99th percentile.
TAIL_AWARE_QUANTILES = np.array([0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.97, 0.99, 0.995, 1.0])


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
    """Population-weighted quantiles, the standard "Hazen" / midpoint-of-mass convention used
    throughout survey statistics (e.g. R's `Hmisc::wtd.quantile` default): sort by value, place
    each point at the MIDPOINT of its own cumulative weight mass, interpolate.

    Under uniform weights this reproduces `np.quantile(..., method="hazen")` EXACTLY -- NOT
    `np.quantile`'s own default ("linear"/Type 7) method, which uses a different convention
    (`i/(n-1)` positions) that has no numerically well-behaved weighted generalisation for
    genuinely skewed weights. Hazen was chosen deliberately for its good behaviour under real,
    skewed survey weights, not because it happens to be numpy's default -- it isn't.
    """
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cw = np.cumsum(w) - 0.5 * w  # midpoint of each value's own weight mass, standard weighted-quantile convention
    cw = cw / w.sum()
    return np.interp(quantiles, cw, v)


def weighted_mean_std(values: np.ndarray, weights: np.ndarray = None) -> Tuple[float, float]:
    """Population-weighted (mean, std). `weights=None` reproduces plain pandas/numpy `.mean()`/
    `.std(ddof=1)` EXACTLY -- ddof=1 (sample std), not numpy's own `.std()` default (ddof=0,
    population std), matching pandas' convention deliberately so this is a strict drop-in
    replacement in the unweighted case.

    The weighted case uses the "reliability weights" (Stata `aweight`-style) bias correction --
    `var = sum(w*(x-m)^2) / (sum(w) - sum(w^2)/sum(w))` -- which reduces EXACTLY to the plain
    ddof=1 formula when every weight is equal, so this is a strict generalisation of the
    unweighted convention, not a different one that happens to coincide sometimes.
    """
    v = np.asarray(values, dtype=float)
    if weights is None:
        return float(v.mean()), float(v.std(ddof=1))
    w = np.asarray(weights, dtype=float)
    m = float(np.average(v, weights=w))
    num = float(np.sum(w * (v - m) ** 2))
    denom = float(w.sum() - (w ** 2).sum() / w.sum())
    var = num / denom if denom > 1e-12 else 0.0
    return m, float(np.sqrt(max(var, 0.0)))


def _discretise(s: pd.Series, edges: np.ndarray = None, n_bins: int = 10, tail_aware: bool = False,
                 weights: pd.Series = None):
    """Numeric -> quantile-bin labels ('q0'..'qk'); categorical -> its own string values,
    unchanged. `edges` (if given) are REUSED, not recomputed -- the target stats and every batch
    scored against them must use the IDENTICAL bin boundaries, computed once from training data,
    or "batch vs target" frequency comparisons are comparing different things under the same name.
    Returns (labels, edges) so a caller can capture edges the first time and reuse them after.

    `weights` (optional, default None = unweighted, i.e. no behaviour change): when computing NEW
    edges from `s`, places them at POPULATION-weighted quantiles instead of plain sample quantiles.
    Only matters when edges are being computed fresh (from real training data); a caller reusing
    already-computed edges on a batch never passes weights here (a generated batch has no survey
    design behind it -- see `compute_target_stats`'s docstring).

    The "should this be binned at all" check (`s.nunique() > n_bins`) is skipped entirely when
    `edges` is ALREADY GIVEN for reuse -- that decision was already made when the edges were first
    computed from the (larger) reference dataset; re-deciding it on a small batch that happens to
    have few unique values (e.g. a batch where a numeric column's generated values happen to
    collapse) would silently fall through to treating each raw float as its own string category,
    bypassing the bin edges entirely.
    """
    if edges is not None:
        labels = pd.cut(s.astype(float), edges, labels=False, include_lowest=True)
        return labels.map(lambda b: f"q{int(b)}" if pd.notna(b) else "NA"), edges
    if pd.api.types.is_numeric_dtype(s) and s.nunique() > n_bins:
        q = TAIL_AWARE_QUANTILES if tail_aware else np.linspace(0, 1, n_bins + 1)
        sv = s.astype(float)
        valid = sv.notna()
        if weights is None:
            edges = np.unique(np.quantile(sv[valid], q))
        else:
            edges = np.unique(_weighted_quantile(sv[valid].to_numpy(), weights[valid].to_numpy().astype(float), q))
        edges[0], edges[-1] = -np.inf, np.inf
        labels = pd.cut(sv, edges, labels=False, include_lowest=True)
        return labels.map(lambda b: f"q{int(b)}" if pd.notna(b) else "NA"), edges
    return s.astype(str), edges


def _weighted_value_counts(labels: pd.Series, weights: pd.Series = None) -> dict:
    """`labels.value_counts(normalize=True)`, but weighted when `weights` is given -- each row
    contributes its own weight to its label's total instead of a flat 1, so the resulting
    frequency table represents the POPULATION, not the raw sample. `weights=None` reproduces plain
    `value_counts(normalize=True)` exactly."""
    if weights is None:
        return labels.value_counts(normalize=True).to_dict()
    w = pd.Series(np.asarray(weights, dtype=float), index=labels.index)
    tot = w.groupby(labels).sum()
    return (tot / tot.sum()).to_dict()


def compute_target_stats(
    df: pd.DataFrame, columns: Sequence[str], pairs: Sequence[Tuple[str, str]] = (), n_bins: int = 10,
    tail_aware_columns: Sequence[str] = (), weights: pd.Series = None,
) -> dict:
    """Reference marginal (per column) and joint (per pair) frequency tables from REAL data --
    call with training-side data only. Returns {"col": {col: {value: freq, ...}}, "pair": {(a,b):
    {(va,vb): freq, ...}}, "edges": {col: edges}}. `tail_aware_columns`: use `TAIL_AWARE_QUANTILES`
    bin edges for these columns instead of plain deciles.

    `weights` (optional, default None = unweighted, i.e. no behaviour change): a per-row survey (or
    other design) weight, aligned with `df`'s index. When given, BOTH the quantile bin edges AND
    the frequency table within each bin are computed population-weighted -- a bin edge placed at
    the plain (unweighted) sample's 90th percentile can sit at a materially different value than
    the population's true 90th percentile if sampling weights vary a lot. Only ever applies to REAL
    data -- the reward's own per-round batch frequency (see `marginal_joint_reward`) is
    deliberately never weighted, since a generated row has no design behind it to correct for.
    """
    col_stats, edges_cache = {}, {}
    for c in columns:
        labels, edges = _discretise(df[c], tail_aware=c in tail_aware_columns, weights=weights)
        edges_cache[c] = edges
        col_stats[c] = _weighted_value_counts(labels, weights)
    pair_stats = {}
    for a, b in pairs:
        la, _ = _discretise(df[a], edges_cache.get(a))
        lb, _ = _discretise(df[b], edges_cache.get(b))
        pair_stats[(a, b)] = _weighted_value_counts(pd.Series(list(zip(la, lb)), index=df.index), weights)
    return dict(col=col_stats, pair=pair_stats, edges=edges_cache)


class EMAFrequencyTracker:
    """A running, LOW-NOISE estimate of the CURRENT POLICY's own per-column/pair output
    distribution, updated a little by every round's batch (exponential moving average), instead of
    trusting one round's raw batch frequency as "what the policy is currently doing".

    Useful whenever a target has many categories relative to the batch size (a fine-grained
    categorical can get as few as a handful of samples per category per round, making a single
    batch's own empirical frequency very noisy). The same idea as `WeightEMA` (weight averaging,
    elsewhere in this library) and batch-norm running statistics: average away per-step noise
    instead of reacting to it every step.
    """

    def __init__(self, decay: float = 0.9) -> None:
        self.decay = decay
        self.col_freq: Dict[str, Dict] = {}
        self.pair_freq: Dict[Tuple[str, str], Dict] = {}

    def _ema_update(self, table: dict, key, vc: dict) -> None:
        prev = table.get(key)
        if prev is None:
            table[key] = dict(vc)
            return
        keys = set(vc) | set(prev)
        table[key] = {k: self.decay * prev.get(k, 0.0) + (1 - self.decay) * vc.get(k, 0.0) for k in keys}

    def update(self, batch_df: pd.DataFrame, columns: Sequence[str], pairs: Sequence, target: dict) -> None:
        """Fold this round's batch into the running estimate. Call AFTER computing this round's
        reward (see `smoothed_marginal_joint_reward`), so a batch never partly scores itself."""
        for c in columns:
            labels, _ = _discretise(batch_df[c], target["edges"].get(c))
            self._ema_update(self.col_freq, c, labels.value_counts(normalize=True).to_dict())
        for a, b in pairs:
            la, _ = _discretise(batch_df[a], target["edges"].get(a))
            lb, _ = _discretise(batch_df[b], target["edges"].get(b))
            vc = pd.Series(list(zip(la, lb))).value_counts(normalize=True).to_dict()
            self._ema_update(self.pair_freq, (a, b), vc)


def smoothed_marginal_joint_reward(
    batch_df: pd.DataFrame, target: dict, tracker: "EMAFrequencyTracker", columns: Sequence[str],
    pairs: Sequence[Tuple[str, str]] = (), col_weight: float = 1.0, pair_weight: float = 1.0,
) -> np.ndarray:
    """Same reward logic as `marginal_joint_reward` (target_freq - "current" freq for this sample's
    value), except "current" is the EMA-smoothed running estimate (`tracker`), not this one batch's
    own raw, noisy empirical frequency. Falls back to a neutral (0.5) prior for a column/pair the
    tracker hasn't seen yet (first round). Does NOT update the tracker -- call `tracker.update(...)`
    separately, after scoring, so this batch is scored against the state BEFORE it.
    """
    n = len(batch_df)
    reward = np.zeros(n)
    for c in columns:
        labels, _ = _discretise(batch_df[c], target["edges"].get(c))
        ref = tracker.col_freq.get(c, {})
        tgt = target["col"][c]
        contrib = labels.map(lambda v: tgt.get(v, 0.0) - ref.get(v, tgt.get(v, 0.0)))
        reward += col_weight * contrib.to_numpy()
    for a, b in pairs:
        la, _ = _discretise(batch_df[a], target["edges"].get(a))
        lb, _ = _discretise(batch_df[b], target["edges"].get(b))
        pairs_series = pd.Series(list(zip(la, lb)))
        ref = tracker.pair_freq.get((a, b), {})
        tgt = target["pair"][(a, b)]
        contrib = pairs_series.map(lambda v: tgt.get(v, 0.0) - ref.get(v, tgt.get(v, 0.0)))
        reward += pair_weight * contrib.to_numpy()
    return reward


def _tail_bin_weights(tgt: dict, tail_boost: float, tail_freq_cutoff: float) -> dict:
    """Per-bin-label multiplier: `tail_boost` for a bin whose TARGET frequency is <=
    `tail_freq_cutoff` (a rare bin -- under tail-aware quantile edges, this picks out the extreme
    upper-tail bins, which have small target frequency BY CONSTRUCTION), 1.0 otherwise.

    Motivation: `marginal_joint_reward`'s contribution for a bin is `target_freq - batch_freq`,
    which is mechanically SMALL for a rare bin no matter how well-calibrated the model is (a bulk
    decile bin has target_freq ~0.1-0.25; the tail-aware scheme's extreme bins have target_freq as
    low as 0.005). Summed across many targeted columns/pairs where only a few are tail-aware, this
    rare-bin signal is easily swamped by bulk-bin contributions from every OTHER column, even after
    group-relative normalisation. Boosting the rare bins directly compensates for that built-in
    scale asymmetry.
    """
    return {v: (tail_boost if f <= tail_freq_cutoff else 1.0) for v, f in tgt.items()}


def _ols_fit(X: np.ndarray, y: np.ndarray):
    """OLS with an intercept, plain normal-equations solution. Returns (coef, resid, Xc) -- coef[0]
    is the intercept, Xc is the intercept-augmented design matrix (returned so callers needing it
    too, e.g. `regression_influence_reward`, don't rebuild it a second time). No external
    dependency -- a small, generic helper for `regression_influence_reward`."""
    Xc = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(Xc, y, rcond=None)
    resid = y - Xc @ beta
    return beta, resid, Xc


def regression_influence_reward(
    X: np.ndarray, y: np.ndarray, target_beta: Sequence[float], weight_intercept: bool = False,
) -> np.ndarray:
    """Per-row reward from a BATCH-level OLS coefficient-alignment target. `X`/`y` are the
    already-extracted design matrix and outcome for this round's batch; `target_beta` is that same
    regression's own coefficients fit once on real training data (same extraction, e.g. via
    `sklearn`/`statsmodels`/a caller's own OLS helper -- this function only needs `X`, `y`, and
    `target_beta` as plain arrays, not any particular extraction pipeline).

    A regression coefficient fit on `batch_df` is a property of the WHOLE batch, not any single
    row, so it cannot become a per-row reward by broadcasting the same scalar to every row:
    `group_relative_advantage`'s mean-centering removes a constant added identically to every
    reward in a group down to floating-point noise -- a naive "compute the batch's own coefficient
    gap, hand it to everyone" design would be mechanically inert, contributing zero gradient.

    Instead, this uses each row's own regression INFLUENCE: to first order, upweighting row i by a
    small amount shifts the fitted coefficient vector by `(X'X)^-1 x_i * resid_i` (the standard OLS
    empirical influence function -- the same quantity behind DFBETA diagnostics, without the
    leave-one-out denominator adjustment, since here the question is "what does MORE of row i do",
    not "what does removing it do"). A row whose influence points TOWARD `target_beta` (shrinking
    the current batch's own gap) gets a positive reward; one whose influence points AWAY gets a
    negative one -- the genuinely per-row, non-degenerate signal a batch-level regression statistic
    can support.

    `weight_intercept=False` (default) zeroes the intercept's contribution to the reward, since the
    intercept is rarely a real "relationship" worth optimising toward on its own.
    """
    beta_batch, resid, Xc = _ols_fit(X, y)
    xtx_inv = np.linalg.pinv(Xc.T @ Xc)
    infl = (Xc @ xtx_inv) * resid[:, None]  # (n, K): row i = (X'X)^-1 x_i * resid_i
    gap = beta_batch - np.asarray(target_beta, dtype=float)
    if not weight_intercept:
        gap = gap.copy()
        gap[0] = 0.0
    return -(infl * gap).sum(axis=1)


def marginal_joint_reward(
    batch_df: pd.DataFrame, target: dict, columns: Sequence[str], pairs: Sequence[Tuple[str, str]] = (),
    col_weight: float = 1.0, pair_weight: float = 1.0, tail_boost_columns: Sequence[str] = (),
    tail_boost: float = 1.0, tail_freq_cutoff: float = 0.02,
) -> np.ndarray:
    """Per-sample reward: for each targeted column (and pair), this batch's OWN empirical
    frequency for the sample's value is compared to the TRUE (target) frequency; the sample gets
    `target_freq - batch_freq` for that value -- positive when the value is under-represented in
    THIS batch relative to the true distribution (push for more of it), negative when
    over-represented (push for less). Summed across every targeted column and pair, weighted.

    `tail_boost_columns` / `tail_boost` / `tail_freq_cutoff`: for these columns only, multiply the
    contribution of any bin whose TARGET frequency is <= `tail_freq_cutoff` by `tail_boost` (see
    `_tail_bin_weights`) -- directly compensates for tail bins' mechanically small reward scale
    under tail-aware quantile edges. Default `tail_boost=1.0` is a no-op.

    Deliberately simple and directly interpretable, unlike a discriminator's black-box score: the
    sign and magnitude of every contribution can be read off directly against real numbers. Reward
    SCALE is not hand-tuned here -- `group_relative_advantage`/`grouped_relative_advantage`
    normalises by the batch's own mean/std before this ever reaches a gradient, so only the
    RELATIVE ranking within a batch matters, not the absolute units different columns' contributions
    happen to be in.
    """
    n = len(batch_df)
    reward = np.zeros(n)
    for c in columns:
        labels, _ = _discretise(batch_df[c], target["edges"].get(c))
        batch_freq = labels.value_counts(normalize=True)
        tgt = target["col"][c]
        contrib = labels.map(lambda v: tgt.get(v, 0.0) - batch_freq.get(v, 0.0))
        if c in tail_boost_columns and tail_boost != 1.0:
            w = _tail_bin_weights(tgt, tail_boost, tail_freq_cutoff)
            contrib = contrib * labels.map(lambda v: w.get(v, 1.0))
        reward += col_weight * contrib.to_numpy()
    for a, b in pairs:
        la, _ = _discretise(batch_df[a], target["edges"].get(a))
        lb, _ = _discretise(batch_df[b], target["edges"].get(b))
        pairs_series = pd.Series(list(zip(la, lb)))
        batch_freq = pairs_series.value_counts(normalize=True)
        tgt = target["pair"][(a, b)]
        contrib = pairs_series.map(lambda v: tgt.get(v, 0.0) - batch_freq.get(v, 0.0))
        reward += pair_weight * contrib.to_numpy()
    return reward


def _loo_categorical_reward(labels: pd.Series, tgt: dict) -> np.ndarray:
    """EXACT (not first-order-approximate) leave-one-out contribution of each row to how well
    `labels`' own frequency table matches `tgt`, under squared error: `L = sum_v (p_v - c_v/n)^2`
    where `p_v = tgt.get(v, 0.0)` and `c_v` is `v`'s raw count in `labels`. Each row's reward is
    `L(without that row) - L(with it)` -- positive when removing the row would make the fit WORSE
    (the row is pulling its own value's frequency toward the target, so it should be encouraged),
    negative when removing it would make the fit BETTER (the row is on the over-represented side of
    its value's frequency).

    Computed in closed form, not by literally rebuilding the frequency table `n` times: the three
    category-level sums `S1 = sum_v p_v^2`, `S2 = sum_v p_v*c_v`, `S3 = sum_v c_v^2` fully
    determine `L(B)` at ANY sample size (`L = S1 - 2*S2/m + S3/m^2` for `m` rows), computed ONCE in
    O(number of distinct values); a specific row's own leave-one-out score then only needs
    correcting the ONE category-level term its own value contributes, an O(1) lookup after that.
    `tests/realtabformer/test_rtf_rl.py` cross-checks this closed form against literally dropping
    each row and rebuilding the frequency table from scratch, on a random batch -- not just
    algebra that looks right.
    """
    n = len(labels)
    if n <= 1:
        return np.zeros(n)  # no leave-one-out is meaningful with 0 rows left behind
    counts = labels.value_counts().to_dict()
    values = set(counts) | set(tgt)
    p = {v: tgt.get(v, 0.0) for v in values}
    c = {v: counts.get(v, 0) for v in values}
    s1 = sum(p[v] ** 2 for v in values)
    s2 = sum(p[v] * c[v] for v in values)
    s3 = sum(c[v] ** 2 for v in values)

    def loss(m: float) -> float:
        return s1 - 2.0 * s2 / m + s3 / (m ** 2)

    l_full = loss(n)
    m = n - 1
    l0 = loss(m)  # every category's term computed at the LEFT-OUT sample size, before per-value correction

    def loo_reward(v) -> float:
        p_v, c_v = p[v], c[v]
        old_term = (p_v - c_v / m) ** 2
        new_term = (p_v - (c_v - 1) / m) ** 2
        l_excl = l0 - old_term + new_term  # correct only the one category this row actually belongs to
        return l_excl - l_full

    cache = {v: loo_reward(v) for v in values}
    return labels.map(cache).to_numpy()


def leave_one_out_marginal_reward(
    batch_df: pd.DataFrame, target: dict, columns: Sequence[str], pairs: Sequence[Tuple[str, str]] = (),
    col_weight: float = 1.0, pair_weight: float = 1.0,
) -> np.ndarray:
    """Per-row reward from each row's own EXACT leave-one-out contribution to marginal/joint
    fidelity (see `_loo_categorical_reward`), as an alternative to `marginal_joint_reward`'s pooled,
    LINEAR `target_freq - batch_freq`.

    Honest about what this does and does not change: for a SINGLE targeted column or pair, every
    row sharing the same discretised value still gets the IDENTICAL reward here too -- a categorical
    bin's count has no notion of "this particular row's" contribution beyond which bin it landed in,
    so there is no finer-than-the-bin signal to extract for either reward. The actual difference is
    the FUNCTIONAL FORM: this is the exact leave-one-out effect on a squared-error fit statistic
    (nonlinear in the bin's own count and the batch size `n`), not a value linearly proportional to
    `target_freq - batch_freq` -- e.g. removing one row from a bin that already has very few members
    moves that bin's frequency proportionally much further than removing one from a large bin, an
    effect the pooled linear reward does not capture at all. When SUMMED across several targeted
    columns/pairs (the realistic multi-target case), each row's TOTAL reward is still genuinely
    row-specific, exactly as it already is for the pooled reward -- summing several per-column
    scalars that depend on that row's own combination of values.

    Combines with `marginal_joint_reward`/`regression_influence_reward`/a privacy term (see
    `privacy_penalty`) by simple addition -- callers choosing between the pooled and leave-one-out
    marginal reward, or blending both, do so at the call site, not inside either function.
    """
    n = len(batch_df)
    reward = np.zeros(n)
    for c in columns:
        labels, _ = _discretise(batch_df[c], target["edges"].get(c))
        reward += col_weight * _loo_categorical_reward(labels, target["col"][c])
    for a, b in pairs:
        la, _ = _discretise(batch_df[a], target["edges"].get(a))
        lb, _ = _discretise(batch_df[b], target["edges"].get(b))
        pairs_series = pd.Series(list(zip(la, lb)))
        reward += pair_weight * _loo_categorical_reward(pairs_series, target["pair"][(a, b)])
    return reward


def privacy_penalty(nearest_real_distance: np.ndarray, threshold: float, scale: float = 1.0) -> np.ndarray:
    """Per-row reward PENALTY (always <= 0) for a synthetic row that sits too close to its nearest
    real training row -- a cheap proxy for memorization risk, meant to be ADDED to a fidelity
    reward (e.g. `marginal_joint_reward`/`leave_one_out_marginal_reward`) so an RL fine-tuning loop
    optimizing distributional fidelity cannot freely trade privacy away for it. Motivated directly
    by a real, measured failure elsewhere in this library's own use: embedding enough distributional
    signal as an explicit generation target once produced synthetic rows a large fraction of which
    were near-exact duplicates of real training rows -- fidelity metrics alone never would have
    caught that; only a distance-to-real-data check did.

    `nearest_real_distance`: each synthetic row's OWN distance to its closest real training row, in
    whatever metric/units the caller already trusts (e.g. a normalized Euclidean or Gower distance,
    or an existing project's own validated DCR-style computation) -- this function deliberately only
    turns an already-computed distance into a reward signal, it does not compute the distance
    itself, so any already-validated nearest-neighbor implementation can be reused directly rather
    than re-derived (and re-verified) here.

    `threshold`: distances at or above this are safe (penalty 0); distances below it are penalized
    LINEARLY in the shortfall (`threshold - distance`), scaled by `scale`. A hard cutoff rather than
    a smooth one is a deliberate choice for interpretability: the reward can be read directly as
    "how far into unsafe territory is this row", not squashed through an arbitrary nonlinearity.
    """
    d = np.asarray(nearest_real_distance, dtype=float)
    gap = np.clip(threshold - d, 0.0, None)
    return -scale * gap
