import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from transformers.models.gpt2 import GPT2LMHeadModel

from realtabformer.realtabformer import REaLTabFormer
from realtabformer.rtf_cusum import (
    CUSUMOverfittingMonitor,
    _robust_noise_std,
    calibrate_gaussian_cusum_threshold,
    compute_gated_row_scores,
)

RANDOM_SEED = 1029


# ---------------------------------------------------------------------
# compute_gated_row_scores
# ---------------------------------------------------------------------
def test_compute_gated_row_scores_perfect_prediction():
    # vocab of size 4; logits put all mass on the true token at every
    # position -> the gated score should equal log(1) = 0 everywhere.
    vocab_size = 4
    labels = torch.tensor([[1, 2, 3]])  # 1 row, 3 positions after BOS
    input_ids = torch.tensor([[0, 1, 2, 3]])  # BOS + 3 tokens
    logits = torch.full((1, 4, vocab_size), -1e4)
    # shift_logits = logits[:, :-1, :] -> predicts labels[:, 1:] = [2, 3]...
    # Build labels aligned with the model's own shift convention directly:
    # shift_logits has length 3 (input_ids[:-1]), shift_labels = labels[:, 1:]
    # Simpler: construct a case with explicit shift semantics below.
    del labels, input_ids, logits

    B, T, V = 1, 3, 4
    logits = torch.full((B, T + 1, V), -1e4)
    true_tokens = [1, 2, 3]
    for t, tok in enumerate(true_tokens):
        logits[0, t, tok] = 1e4  # position t's argmax matches labels[t+1]
    labels = torch.tensor([[-1] + true_tokens])  # position 0 unused after shift

    floor = math.log(1.0 / V)
    scores = compute_gated_row_scores(logits, labels, floor)
    assert scores.shape == (1,)
    assert scores.item() == pytest.approx(0.0, abs=1e-3)


def test_compute_gated_row_scores_all_wrong_hits_floor():
    B, T, V = 1, 3, 5
    logits = torch.zeros((B, T + 1, V))
    true_tokens = [1, 2, 3]
    for t, tok in enumerate(true_tokens):
        # argmax picks a DIFFERENT token than the true one.
        wrong = (tok + 1) % V
        logits[0, t, wrong] = 1e4
    labels = torch.tensor([[-1] + true_tokens])

    floor = math.log(1.0 / V)
    scores = compute_gated_row_scores(logits, labels, floor)
    assert scores.item() == pytest.approx(floor, abs=1e-3)


def test_compute_gated_row_scores_ignores_padding():
    B, T, V = 1, 3, 4
    logits = torch.full((B, T + 1, V), -1e4)
    true_tokens = [1, -100, 3]  # middle position is padding
    logits[0, 0, 1] = 1e4
    logits[0, 2, 3] = 1e4
    labels = torch.tensor([[-1] + true_tokens])

    floor = math.log(1.0 / V)
    scores = compute_gated_row_scores(logits, labels, floor)
    # Only 2 valid positions, both "correct" -> mean gated score 0.
    assert scores.item() == pytest.approx(0.0, abs=1e-3)


def test_compute_gated_row_scores_batched_and_gradient_free():
    B, T, V = 5, 4, 6
    torch.manual_seed(RANDOM_SEED)
    logits = torch.randn(B, T + 1, V, requires_grad=True)
    labels = torch.randint(0, V, (B, T + 1))
    labels[:, 0] = -1  # first shifted-out position unused
    floor = math.log(1.0 / V)
    scores = compute_gated_row_scores(logits, labels, floor)
    assert scores.shape == (B,)
    # The function itself doesn't force no-grad (callers -- record_batch
    # / maybe_check -- wrap it in torch.no_grad() themselves); just check
    # it runs cleanly end to end and produces finite values.
    assert torch.isfinite(scores).all()


# ---------------------------------------------------------------------
# calibrate_gaussian_cusum_threshold
# ---------------------------------------------------------------------
def test_calibrate_threshold_positive_and_reproducible():
    h1 = calibrate_gaussian_cusum_threshold(
        delta=0.5, n_checks_horizon=100, n_sims=500, seed=RANDOM_SEED
    )
    h2 = calibrate_gaussian_cusum_threshold(
        delta=0.5, n_checks_horizon=100, n_sims=500, seed=RANDOM_SEED
    )
    assert h1 > 0
    assert h1 == h2  # deterministic given the same seed


def test_calibrate_threshold_grows_with_horizon():
    h_short = calibrate_gaussian_cusum_threshold(
        delta=0.5, n_checks_horizon=20, n_sims=1000, seed=RANDOM_SEED
    )
    h_long = calibrate_gaussian_cusum_threshold(
        delta=0.5, n_checks_horizon=200, n_sims=1000, seed=RANDOM_SEED
    )
    # A longer horizon gives the null more chances to wander -> the 99th
    # percentile of its running max should not decrease.
    assert h_long >= h_short


# ---------------------------------------------------------------------
# CUSUMOverfittingMonitor
# ---------------------------------------------------------------------
def _make_monitor(**overrides):
    kwargs = dict(
        vocab_size=50,
        check_every=5,
        cooldown_steps=3,
        warmup_checks=3,
        # Tests below exercise cooldown/calibration/CUSUM-accumulation
        # logic independently of the settle-skip mechanism (which has
        # its own dedicated tests) -- default it off here so existing
        # check counts don't need to change.
        warmup_settle_checks=0,
        seen_pool_size=32,
        min_seen_pool=4,
        delta=0.5,
        target_quantile=0.99,
        total_checks_horizon=50,
        random_state=RANDOM_SEED,
    )
    kwargs.update(overrides)
    return CUSUMOverfittingMonitor(**kwargs)


def test_monitor_rejects_bad_config():
    with pytest.raises(ValueError):
        _make_monitor(check_every=0)
    with pytest.raises(ValueError):
        _make_monitor(cooldown_steps=-1)
    with pytest.raises(ValueError):
        _make_monitor(warmup_checks=1)
    with pytest.raises(ValueError):
        _make_monitor(warmup_settle_checks=-1)
    with pytest.raises(ValueError):
        _make_monitor(delta=[])
    with pytest.raises(ValueError):
        _make_monitor(delta=[0.5, -0.1])
    with pytest.raises(ValueError):
        _make_monitor(delta=[0.5, 0.5])  # duplicates not allowed


# ---------------------------------------------------------------------
# delta ensemble
# ---------------------------------------------------------------------
def test_monitor_single_delta_is_backward_compatible():
    # A plain float (the original API) must produce EXACTLY the same
    # cusum_h as before the ensemble feature existed -- Bonferroni
    # correction with a single tracker is a no-op (far / 1 == far).
    horizon = 50
    mon = _make_monitor(delta=0.5, total_checks_horizon=horizon)
    expected_h = calibrate_gaussian_cusum_threshold(
        delta=0.5, n_checks_horizon=horizon, target_quantile=0.99, seed=RANDOM_SEED
    )
    assert mon.deltas == [0.5]
    assert mon.delta == 0.5
    assert mon.cusum_h == pytest.approx(expected_h)
    assert mon.cusum_S == 0.0


def test_monitor_ensemble_thresholds_are_bonferroni_corrected():
    horizon = 50
    deltas = [0.25, 0.5, 1.0]
    mon = _make_monitor(delta=deltas, total_checks_horizon=horizon)

    assert mon.deltas == deltas
    assert set(mon.cusum_h_by_delta.keys()) == set(deltas)

    # far=0.01 split three ways -> each tracker individually calibrated
    # at target_quantile = 1 - 0.01/3, not the naive 0.99.
    adjusted_quantile = 1.0 - (0.01 / 3)
    for d in deltas:
        expected_h = calibrate_gaussian_cusum_threshold(
            delta=d, n_checks_horizon=horizon, target_quantile=adjusted_quantile, seed=RANDOM_SEED
        )
        assert mon.cusum_h_by_delta[d] == pytest.approx(expected_h)

    # A stricter per-tracker quantile means a HIGHER threshold than the
    # single-tracker (non-Bonferroni-corrected) calibration would give.
    single_h = calibrate_gaussian_cusum_threshold(
        delta=deltas[0], n_checks_horizon=horizon, target_quantile=0.99, seed=RANDOM_SEED
    )
    assert mon.cusum_h_by_delta[deltas[0]] > single_h

    # Backward-compat aliases point at the first (primary) tracker.
    assert mon.delta == deltas[0]
    assert mon.cusum_h == mon.cusum_h_by_delta[deltas[0]]
    assert mon.cusum_S == mon.cusum_S_by_delta[deltas[0]]


def test_monitor_ensemble_fires_on_first_tracker_to_cross():
    # A small delta accumulates evidence readily from a small, steady
    # shift -- confirm the ensemble fires via that tracker (not
    # necessarily the "primary"/first-listed one) and records which.
    torch.manual_seed(RANDOM_SEED)
    V = 30
    mon = _make_monitor(
        delta=[0.15, 0.5, 1.5],
        cooldown_steps=1,
        min_seen_pool=8,
        warmup_checks=3,
        check_every=1,
        total_checks_horizon=200,
    )

    class ConstShiftLM(torch.nn.Module):
        def __init__(self, vocab_size, bias_scale=0.0):
            super().__init__()
            self.vocab_size = vocab_size
            self.bias_scale = bias_scale
            self.training = True

        def eval(self):
            self.training = False
            return self

        def train(self, mode=True):
            self.training = mode
            return self

        def forward(self, input_ids, labels):
            B, T = labels.shape
            logits = torch.randn(B, T, self.vocab_size) * 0.1
            for b in range(B):
                for t in range(T):
                    tok = int(labels[b, t].item())
                    if tok >= 0:
                        logits[b, t, tok] += self.bias_scale

            class Out:
                pass

            out = Out()
            out.logits = logits
            return out

        def parameters(self):
            return iter([torch.nn.Parameter(torch.zeros(1))])

    model = ConstShiftLM(V, bias_scale=0.0)
    row_ids = list(range(20))
    T = 3
    row_labels = torch.randint(1, V, (len(row_ids), T + 1))
    row_labels[:, 0] = -1
    labels_by_id = {idx: row_labels[i : i + 1] for i, idx in enumerate(row_ids)}

    def get_rows(indices):
        return None, torch.cat([labels_by_id[i] for i in indices], dim=0)

    logits = model(None, row_labels).logits
    mon.record_batch(row_ids, logits, row_labels, step=1)

    step = 2
    for _ in range(5):
        assert mon.maybe_check(step, model, get_rows) is False
        step += 1
    assert mon.mu0 is not None

    # A small, persistent shift -- exactly what a small delta should
    # catch faster than a large one.
    model.bias_scale = 1.5
    fired_at = None
    for _ in range(100):
        if mon.maybe_check(step, model, get_rows):
            fired_at = step
            break
        step += 1

    assert fired_at is not None, "expected the ensemble to fire on a persistent shift"
    assert mon.alarm_step == fired_at
    assert mon.alarm_delta in mon.deltas
    # The firing tracker's own accumulator must have actually crossed
    # its own (Bonferroni-corrected) threshold.
    assert mon.cusum_S_by_delta[mon.alarm_delta] >= mon.cusum_h_by_delta[mon.alarm_delta]


def test_monitor_warmup_settle_checks_defaults_to_warmup_checks():
    # Constructed directly (not via _make_monitor, which pins
    # warmup_settle_checks=0 for the tests that don't exercise it) to
    # check the library's own real default.
    mon = CUSUMOverfittingMonitor(vocab_size=50, warmup_checks=7)
    assert mon.warmup_settle_checks == 7
    assert mon._settle_checks_remaining == 7


# ---------------------------------------------------------------------
# _robust_noise_std
# ---------------------------------------------------------------------
def test_robust_noise_std_recovers_true_noise_under_linear_trend():
    # Regression test for a real bug found by replaying a real training
    # run's logged CUSUM trajectory: the calibration window's raw std
    # conflates a slow trend (ordinary, non-memorization improvement)
    # with actual check-to-check noise, inflating sigma0 by orders of
    # magnitude and delaying detection. A linear ramp + small iid noise
    # is the textbook case first-differencing is built to handle: the
    # constant per-step increment cancels out of every difference,
    # leaving only `2 * true_noise**2` per difference (hence /sqrt(2)).
    rng = np.random.default_rng(RANDOM_SEED)
    true_noise = 0.02
    trend = np.linspace(0.1, 0.3, 10)
    values = trend + rng.normal(0.0, true_noise, size=10)

    raw_std = float(np.std(values, ddof=1))
    robust = _robust_noise_std(values)

    assert robust < raw_std / 2  # trend contamination dominated raw std
    assert abs(robust - true_noise) < abs(raw_std - true_noise)


def test_robust_noise_std_matches_raw_std_scale_without_trend():
    # With no trend, differencing shouldn't produce a wildly different
    # answer -- both estimators should land in the same ballpark.
    rng = np.random.default_rng(RANDOM_SEED)
    values = rng.normal(0.5, 0.05, size=20)
    raw_std = float(np.std(values, ddof=1))
    robust = _robust_noise_std(values)
    assert 0.3 * raw_std < robust < 3.0 * raw_std


def test_robust_noise_std_handles_degenerate_inputs():
    assert _robust_noise_std([]) == 1e-6
    assert _robust_noise_std([1.0]) == 1e-6
    # Two points -> a single difference; still a positive estimate.
    assert _robust_noise_std([1.0, 1.2]) > 0


def test_monitor_adjust_cooldown_caps_when_it_would_never_fit():
    # Found the hard way: with gradient_accumulation_steps multiplying
    # how much data one optimizer step covers, steps_per_epoch can end
    # up smaller than a "reasonable-looking" fixed cooldown -- if the
    # cooldown is never satisfiable within one epoch's cycle, every row
    # gets re-touched before it can cool, and the detector never
    # activates. steps_per_epoch=19 (matching the real Adult/grad-accum=4
    # scenario that surfaced this) with cooldown_steps=40 must be capped.
    mon = _make_monitor(cooldown_steps=40)
    mon.adjust_cooldown_for_steps_per_epoch(steps_per_epoch=19)
    assert mon.cooldown_steps == 9  # 19 // 2
    assert mon.cooldown_steps < 19  # now satisfiable within one epoch


def test_monitor_adjust_cooldown_is_noop_when_already_safe():
    mon = _make_monitor(cooldown_steps=10)
    mon.adjust_cooldown_for_steps_per_epoch(steps_per_epoch=1000)
    assert mon.cooldown_steps == 10  # unchanged -- comfortably fits already


def test_monitor_record_batch_tracks_baseline_once():
    mon = _make_monitor()
    V = 50
    B, T = 4, 3
    logits = torch.randn(B, T + 1, V)
    labels = torch.randint(0, V, (B, T + 1))
    labels[:, 0] = -1

    mon.record_batch([0, 1, 2, 3], logits, labels, step=1)
    assert set(mon.seen_indices) == {0, 1, 2, 3}
    assert set(mon.baseline_score.keys()) == {0, 1, 2, 3}
    first_baseline = dict(mon.baseline_score)

    # Re-recording the SAME rows at a later step must NOT overwrite their
    # baseline (it's a one-time, first-exposure snapshot).
    logits2 = torch.randn(B, T + 1, V)
    mon.record_batch([0, 1, 2, 3], logits2, labels, step=7)
    assert mon.baseline_score == first_baseline
    assert mon.last_seen_step == {0: 7, 1: 7, 2: 7, 3: 7}


def test_monitor_cooldown_filters_recent_rows():
    mon = _make_monitor(cooldown_steps=10)
    V = 50
    logits = torch.randn(2, 3, V)
    labels = torch.tensor([[-1, 1, 2], [-1, 3, 4]])
    mon.record_batch([0, 1], logits, labels, step=5)

    assert mon._cooled_pool(step=10) == []  # only 5 steps have passed
    assert set(mon._cooled_pool(step=15)) == {0, 1}  # exactly cooldown_steps
    assert set(mon._cooled_pool(step=20)) == {0, 1}


def test_monitor_maybe_check_noop_below_min_pool():
    mon = _make_monitor(min_seen_pool=100)
    V = 50
    logits = torch.randn(4, 3, V)
    labels = torch.tensor([[-1, 1, 2]] * 4)
    mon.record_batch([0, 1, 2, 3], logits, labels, step=1)

    calls = []

    def get_rows(indices):
        calls.append(indices)
        raise AssertionError("get_rows should not be called")

    fired = mon.maybe_check(step=100, model=torch.nn.Linear(1, 1), get_rows=get_rows)
    assert fired is False
    assert calls == []


def test_monitor_calibrates_then_accumulates_and_can_fire():
    torch.manual_seed(RANDOM_SEED)
    V = 30
    mon = _make_monitor(
        cooldown_steps=1, min_seen_pool=8, warmup_checks=3, check_every=1,
        total_checks_horizon=40,
    )

    class TinyLM(torch.nn.Module):
        """A tiny stand-in "model" whose logits improve deterministically
        with a per-row bias, so we can force a detectable shift without
        needing a real trained network."""

        def __init__(self, vocab_size, bias_scale=0.0):
            super().__init__()
            self.vocab_size = vocab_size
            self.bias_scale = bias_scale
            self.training = True

        def eval(self):
            self.training = False
            return self

        def train(self, mode=True):
            self.training = mode
            return self

        def forward(self, input_ids, labels):
            B, T = labels.shape
            logits = torch.randn(B, T, self.vocab_size) * 0.1
            for b in range(B):
                for t in range(T):
                    tok = int(labels[b, t].item())
                    if tok >= 0:
                        logits[b, t, tok] += self.bias_scale

            class Out:
                pass

            out = Out()
            out.logits = logits
            return out

        def parameters(self):
            return iter([torch.nn.Parameter(torch.zeros(1))])

    model = TinyLM(V, bias_scale=0.0)

    # Fixed per-row target labels -- the whole point of the paired
    # baseline-vs-current comparison is re-scoring the SAME row against
    # the SAME target later; using fresh random labels each call (as an
    # earlier version of this test mock did) compares apples to oranges
    # and can never show a stable shift.
    row_ids = list(range(20))
    T = 3
    row_labels = torch.randint(1, V, (len(row_ids), T + 1))
    row_labels[:, 0] = -1
    labels_by_id = {idx: row_labels[i : i + 1] for i, idx in enumerate(row_ids)}

    def get_rows(indices):
        labels = torch.cat([labels_by_id[i] for i in indices], dim=0)
        return None, labels

    # Establish baselines for the pool at low confidence.
    logits = model(None, row_labels).logits
    mon.record_batch(row_ids, logits, row_labels, step=1)

    # Warm up calibration: several checks with the SAME low-confidence
    # regime (nothing anomalous yet).
    step = 2
    for _ in range(5):
        fired = mon.maybe_check(step, model, get_rows)
        assert fired is False
        step += 1
    assert mon.mu0 is not None
    assert mon.sigma0 is not None

    # Now make the model dramatically more confident on the SAME rows
    # (simulating memorization) and confirm the CUSUM can detect it
    # within a bounded number of additional checks.
    model.bias_scale = 8.0
    fired_at = None
    for _ in range(30):
        fired = mon.maybe_check(step, model, get_rows)
        if fired:
            fired_at = step
            break
        step += 1

    assert fired_at is not None, "expected the CUSUM to fire on an obvious shift"
    assert mon.alarm_step == fired_at
    # Firing again afterwards must not double-report.
    again = mon.maybe_check(step + 1, model, get_rows)
    assert again is False


def test_monitor_settle_checks_are_discarded_before_calibration():
    # Regression test for a real bug found by replaying a real training
    # run's logged CUSUM trajectory: calibrating sigma0/mu0 from checks
    # right as the reference pool first becomes eligible lands on the
    # most volatile window of training (dominated by rows whose
    # baseline was captured at/near model init), inflating sigma0 by
    # orders of magnitude and delaying detection. Settle checks must be
    # consumed silently -- no warmup sample collected, no history, mu0
    # stays unset -- before calibration starts.
    torch.manual_seed(RANDOM_SEED)
    V = 30
    mon = _make_monitor(
        cooldown_steps=1,
        min_seen_pool=8,
        warmup_checks=3,
        warmup_settle_checks=2,
        check_every=1,
        total_checks_horizon=40,
    )

    class ConstLM(torch.nn.Module):
        def __init__(self, vocab_size):
            super().__init__()
            self.vocab_size = vocab_size
            self.training = True

        def eval(self):
            self.training = False
            return self

        def train(self, mode=True):
            self.training = mode
            return self

        def forward(self, input_ids, labels):
            B, T = labels.shape
            logits = torch.randn(B, T, self.vocab_size) * 0.1

            class Out:
                pass

            out = Out()
            out.logits = logits
            return out

        def parameters(self):
            return iter([torch.nn.Parameter(torch.zeros(1))])

    model = ConstLM(V)
    row_ids = list(range(20))
    T = 3
    row_labels = torch.randint(1, V, (len(row_ids), T + 1))
    row_labels[:, 0] = -1
    labels_by_id = {idx: row_labels[i : i + 1] for i, idx in enumerate(row_ids)}

    def get_rows(indices):
        return None, torch.cat([labels_by_id[i] for i in indices], dim=0)

    logits = model(None, row_labels).logits
    mon.record_batch(row_ids, logits, row_labels, step=1)

    for step in (2, 3):  # warmup_settle_checks=2 -> both must be no-ops
        fired = mon.maybe_check(step, model, get_rows)
        assert fired is False
        assert len(mon._warmup_deltas) == 0
        assert mon.mu0 is None
    assert mon._settle_checks_remaining == 0

    for step in (4, 5):  # first 2 of warmup_checks=3
        mon.maybe_check(step, model, get_rows)
        assert mon.mu0 is None
    mon.maybe_check(6, model, get_rows)  # 3rd -> calibration completes
    assert mon.mu0 is not None
    assert mon.sigma0 is not None
    assert len(mon._warmup_deltas) == 3
    assert mon.history == []  # calibration checks aren't post-calibration history


# ---------------------------------------------------------------------
# End-to-end integration through REaLTabFormer.fit
# ---------------------------------------------------------------------
def _tiny_df(n=80, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "age": rng.integers(18, 80, n),
            "amount": rng.normal(100, 20, n).round(2),
            "category": rng.choice(["A", "B", "C"], n),
        }
    )


def test_fit_invalid_overfitting_detection_method_raises():
    model = REaLTabFormer(model_type="tabular", epochs=1, batch_size=8, random_state=RANDOM_SEED)
    with pytest.raises(ValueError):
        model.fit(_tiny_df(20), device="cpu", overfitting_detection_method="bogus")


def test_fit_cusum_path_runs_and_can_sample():
    df = _tiny_df(n=100)
    model = REaLTabFormer(
        model_type="tabular", epochs=6, batch_size=16, random_state=RANDOM_SEED, train_size=1.0
    )
    trainer = model.fit(
        df,
        device="cpu",
        overfitting_detection_method="cusum",
        cusum_check_every=1,
        cusum_cooldown_steps=2,
        cusum_warmup_checks=3,
    )
    assert model.cusum_monitor is not None
    # "idx" must not leak into the trainer's own reported dataset columns
    # in a way that breaks anything -- and remove_unused_columns must be
    # disabled specifically (and only) for this run.
    assert trainer.args.remove_unused_columns is False

    samples = model.sample(n_samples=5, device="cpu")
    assert len(samples) <= 5
    assert list(samples.columns) == list(df.columns)


def test_fit_cusum_saves_alarm_checkpoint_matching_final_model(tmp_path):
    # Regression test for a real bug: `load_best_model_at_end` defaults
    # to True, which -- left on -- reloads whatever checkpoint had the
    # best held-out eval loss once trainer.train() returns, silently
    # replacing the exact alarm-point model with one picked by an
    # unrelated criterion. Confirms both halves of the fix: the flag is
    # off for this path, and the on-disk alarm checkpoint is bit-for-bit
    # identical to the model actually left in memory (and therefore to
    # whatever `.sample()` uses afterward).
    df = _tiny_df(n=100)
    model = REaLTabFormer(
        model_type="tabular",
        epochs=30,
        batch_size=16,
        random_state=RANDOM_SEED,
        train_size=1.0,
        checkpoints_dir=str(tmp_path / "checkpoints"),
    )
    trainer = model.fit(
        df,
        device="cpu",
        overfitting_detection_method="cusum",
        cusum_check_every=1,
        cusum_cooldown_steps=2,
        cusum_warmup_checks=3,
    )
    mon = model.cusum_monitor
    assert mon.alarm_step is not None, "expected this config to trigger an alarm"
    assert trainer.args.load_best_model_at_end is False
    assert mon.alarm_checkpoint_dir is not None

    ckpt_path = Path(mon.alarm_checkpoint_dir)
    assert (ckpt_path / "model.safetensors").exists() or (
        ckpt_path / "pytorch_model.bin"
    ).exists()

    saved = GPT2LMHeadModel.from_pretrained(str(ckpt_path))
    saved_state = saved.state_dict()
    current_state = trainer.model.state_dict()
    assert saved_state.keys() == current_state.keys()
    for key in saved_state:
        assert torch.equal(saved_state[key], current_state[key].cpu())


def test_fit_cusum_alarm_checkpoint_reloadable_via_load_from_dir(tmp_path):
    # Regression test for a real bug: the alarm callback only wrote
    # HF-format weights (model.safetensors/config.json), which
    # REaLTabFormer.load_from_dir cannot read -- it needs the
    # rtf_config.json/rtf_model.pt pair that only REaLTabFormer.save()
    # produces. Confirms the checkpoint is actually a loadable
    # REaLTabFormer model, not just a bag of HF weights.
    df = _tiny_df(n=100)
    model = REaLTabFormer(
        model_type="tabular",
        epochs=30,
        batch_size=16,
        random_state=RANDOM_SEED,
        train_size=1.0,
        checkpoints_dir=str(tmp_path / "checkpoints"),
    )
    trainer = model.fit(
        df,
        device="cpu",
        overfitting_detection_method="cusum",
        cusum_check_every=1,
        cusum_cooldown_steps=2,
        cusum_warmup_checks=3,
    )
    mon = model.cusum_monitor
    assert mon.alarm_step is not None, "expected this config to trigger an alarm"

    ckpt_path = Path(mon.alarm_checkpoint_dir)
    assert (ckpt_path / "rtf_config.json").exists()
    assert (ckpt_path / "rtf_model.pt").exists()

    reloaded = REaLTabFormer.load_from_dir(ckpt_path)
    assert reloaded.vocab.keys() == model.vocab.keys()
    assert reloaded.processed_columns == model.processed_columns

    current_state = trainer.model.state_dict()
    reloaded_state = reloaded.model.state_dict()
    assert reloaded_state.keys() == current_state.keys()
    for key in current_state:
        assert torch.equal(reloaded_state[key].cpu(), current_state[key].cpu())

    samples = reloaded.sample(10, device="cpu")
    assert len(samples) == 10


def test_fit_default_path_unaffected_by_cusum_module():
    df = _tiny_df(n=60)
    model = REaLTabFormer(
        model_type="tabular", epochs=1, batch_size=8, random_state=RANDOM_SEED, train_size=1.0
    )
    trainer = model.fit(df, device="cpu", n_critic=0)
    # No opt-in machinery should have touched the plain path.
    assert "idx" not in trainer.train_dataset.column_names
    assert trainer.args.remove_unused_columns is True
    assert not hasattr(model, "cusum_monitor") or model.cusum_monitor is None
