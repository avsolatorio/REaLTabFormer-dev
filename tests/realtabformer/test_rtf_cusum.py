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
