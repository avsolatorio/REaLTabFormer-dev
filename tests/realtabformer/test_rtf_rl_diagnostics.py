"""Tests for rtf_rl_diagnostics.py -- verification instrumentation for RL fine-tuning runs.
Ported and adapted from the GMD survey-synthesis investigation's own gmd_rl_selftest.py (see the
independent synthetic-gmd repo's lab notebook, G17-G60).
"""
import numpy as np
import pandas as pd

from realtabformer import rtf_rl_diagnostics as D


def test_batch_diagnostics_flags_collapse_nonfinite_and_zero_variance():
    diag = D.BatchDiagnostics(numeric_columns=["welfare"])
    healthy = pd.DataFrame({"welfare": np.random.default_rng(0).lognormal(10, 1, 256)})
    reward_ok = np.random.default_rng(2).normal(0, 1, 256)
    assert diag.check(0, healthy, reward_ok) == []

    collapsed = pd.DataFrame({"welfare": [50000.0] * 250 + [50001.0, 50002.0, 50003.0, 50004.0, 50005.0, 50006.0]})
    assert any("collapsed" in f for f in diag.check(1, collapsed, reward_ok))

    bad_reward = reward_ok.copy()
    bad_reward[3] = np.nan
    assert any("non-finite" in f for f in diag.check(2, healthy, bad_reward))

    assert any("zero variance" in f for f in diag.check(3, healthy, np.ones(256) * 0.5))


def test_audit_trajectory_smooth_vs_jumpy_vs_flat():
    smooth = [dict(round=r, disc_auc=v) for r, v in zip(range(10, 100, 10), np.linspace(0.50, 0.65, 9))]
    assert not any("discontinuous" in f for f in D.audit_trajectory(smooth, {"disc_auc": 0.50}, metric="disc_auc"))

    jumpy = [dict(round=r, disc_auc=v) for r, v in enumerate([0.50, 0.51, 0.52, 0.95, 0.51, 0.52])]
    assert any("discontinuous" in f for f in D.audit_trajectory(jumpy, {"disc_auc": 0.50}, metric="disc_auc"))

    flat_traj = [dict(round=r, disc_auc=0.50) for r in range(10, 40, 10)]
    assert any("EXACTLY flat" in f for f in D.audit_trajectory(flat_traj, {"disc_auc": 0.50}, metric="disc_auc"))


def test_per_column_tail_err_isolates_the_shifted_column():
    rng = np.random.default_rng(0)
    train = pd.DataFrame({"a": rng.normal(0, 1, 2000), "b": rng.normal(0, 1, 2000)})
    matched = pd.DataFrame({"a": rng.normal(0, 1, 2000), "b": rng.normal(0, 1, 2000)})
    out = D.per_column_tail_err(train, matched)
    assert out["a"]["tail_err"] < 0.15 and out["b"]["tail_err"] < 0.15

    shifted = pd.DataFrame({"a": rng.normal(0, 1, 2000), "b": rng.normal(3, 1, 2000)})
    out2 = D.per_column_tail_err(train, shifted)
    assert out2["a"]["tail_err"] < 0.15 and out2["b"]["tail_err"] > out2["a"]["tail_err"] * 3


def test_patience_stopper_three_scenarios():
    # 1. Steadily improving beyond epsilon: never stops.
    s = D.PatienceStopper(epsilon=0.01, patience=3, initial_best=1.0)
    trajectory = [1.0 - 0.05 * i for i in range(1, 21)]
    stopped = [s.update(v) for v in trajectory]
    assert not any(stopped)
    assert np.isclose(s.best, trajectory[-1])

    # 2. Flat within epsilon: stops exactly on the patience-th consecutive flat call.
    s = D.PatienceStopper(epsilon=0.01, patience=3, initial_best=0.50)
    flat = [0.502, 0.499, 0.503]
    results = [s.update(v) for v in flat]
    assert results == [False, False, True]

    # 3. A genuine mid-run improvement resets the counter.
    s = D.PatienceStopper(epsilon=0.01, patience=3, initial_best=0.50)
    seq = [0.502, 0.499, 0.30, 0.502, 0.499, 0.503]
    results = [s.update(v) for v in seq]
    assert results == [False, False, False, False, False, True]
    assert np.isclose(s.best, 0.30)


def test_multi_metric_patience_stopper_requires_all_metrics_to_plateau():
    # 1. One metric still improving a lot: never stops.
    s = D.MultiMetricPatienceStopper(epsilons={"a": 0.01, "b": 0.01}, patience=4, initial_best={"a": 0.50, "b": 1.0})
    stopped = [s.update({"a": 0.50 + 0.001 * i, "b": 1.0 - 0.1 * i}) for i in range(1, 10)]
    assert not any(stopped)

    # 2. Both flat for `patience` checks: stops exactly then.
    s = D.MultiMetricPatienceStopper(epsilons={"a": 0.01, "b": 0.01}, patience=3, initial_best={"a": 0.50, "b": 1.0})
    results = [s.update(v) for v in [{"a": 0.502, "b": 1.001}, {"a": 0.499, "b": 0.998}, {"a": 0.503, "b": 1.002}]]
    assert results == [False, False, True]

    # 3. A late improvement in just one metric resets the shared counter.
    s = D.MultiMetricPatienceStopper(epsilons={"a": 0.01, "b": 0.01}, patience=3, initial_best={"a": 0.50, "b": 1.0})
    seq = [{"a": 0.502, "b": 1.001}, {"a": 0.499, "b": 0.998}, {"a": 0.503, "b": 0.80},
           {"a": 0.502, "b": 0.801}, {"a": 0.499, "b": 0.798}, {"a": 0.503, "b": 0.802}]
    results = [s.update(v) for v in seq]
    assert results == [False, False, False, False, False, True]
    assert np.isclose(s.best["b"], 0.80)


def test_multi_metric_stopper_window_smoothing_avoids_false_stop_on_noisy_trend():
    base = [1.0 - 0.02 * i for i in range(1, 21)]
    noise = [0.12 if i % 2 == 0 else -0.12 for i in range(1, 21)]
    raw = [b + n for b, n in zip(base, noise)]

    s1 = D.PatienceStopper(epsilon=0.10, patience=4, initial_best=1.0)
    r1 = [s1.update(v) for v in raw]
    assert True in r1, "the unsmoothed stopper is EXPECTED to false-stop on this noisy-but-improving sequence"

    s3 = D.MultiMetricPatienceStopper(epsilons={"x": 0.10 / np.sqrt(3)}, patience=4, initial_best={"x": 1.0}, window=3)
    r3 = [s3.update({"x": v}) for v in raw]
    assert not any(r3)
    assert s3.best["x"] < 0.65

    # Cold-start: an early lucky-low reading must not lock in as best before the window fills.
    s_cold = D.MultiMetricPatienceStopper(epsilons={"x": 0.05}, patience=3, initial_best={"x": 1.0}, window=3)
    lucky_then_flat = [0.5, 0.9, 0.9, 0.9, 0.9, 0.9]
    [s_cold.update({"x": v}) for v in lucky_then_flat]
    assert s_cold.best["x"] > 0.6


def test_should_act_on_stop_min_round_floor():
    assert D.should_act_on_stop(True, round_=100, min_stop_round=250) is False
    assert D.should_act_on_stop(True, round_=250, min_stop_round=250) is True
    assert D.should_act_on_stop(True, round_=300, min_stop_round=250) is True
    assert D.should_act_on_stop(False, round_=300, min_stop_round=250) is False
