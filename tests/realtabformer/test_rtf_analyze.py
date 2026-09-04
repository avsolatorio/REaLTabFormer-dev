import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances
from sklearn.svm import LinearSVC

from realtabformer.rtf_analyze import SyntheticDataBench

RANDOM_SEED = 1029


def _binary_df(n=60, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    target = (x1 + x2 > 0).astype(int)
    return pd.DataFrame({"x1": x1, "x2": x2, "target": target})


def test_measure_ml_efficiency_with_predict_proba_classifier():
    # Regression test for a real bug: predict_proba on a binary
    # classifier returns an (n, 2) array, which pandas refuses to put
    # into a single DataFrame column ("Per-column arrays must each be
    # 1-dimensional") -- any estimator with predict_proba (the common
    # case, e.g. LogisticRegression) crashed this call outright.
    train = _binary_df(seed=1)
    synthetic = _binary_df(seed=2)
    test = _binary_df(seed=3)

    result = SyntheticDataBench.measure_ml_efficiency(
        model=LogisticRegression(),
        train=train,
        synthetic=synthetic,
        test=test,
        target_col="target",
        random_state=RANDOM_SEED,
    )
    assert list(result.columns) == [
        "actual",
        "original_predictions",
        "synthetic_predictions",
    ]
    assert len(result) == len(test)
    # Positive-class probabilities, not the raw (n, 2) predict_proba output.
    assert result["original_predictions"].between(0, 1).all()
    assert result["synthetic_predictions"].between(0, 1).all()


def test_measure_ml_efficiency_falls_back_to_predict_without_proba():
    # Estimators without predict_proba (e.g. LinearSVC) must still work,
    # falling back to hard-label .predict() as before.
    train = _binary_df(seed=1)
    synthetic = _binary_df(seed=2)
    test = _binary_df(seed=3)

    result = SyntheticDataBench.measure_ml_efficiency(
        model=LinearSVC(),
        train=train,
        synthetic=synthetic,
        test=test,
        target_col="target",
        random_state=RANDOM_SEED,
    )
    assert set(result["original_predictions"].unique()) <= {0, 1}
    assert set(result["synthetic_predictions"].unique()) <= {0, 1}


# ---------------------------------------------------------------------
# compute_distance_to_closest_records: NearestNeighbors-backed path,
# switched from a brute-force dense pairwise matrix (benchmarked
# directly before making the change -- see the method's own docstring).
# ---------------------------------------------------------------------
def _numeric_df(n, dim, seed):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.normal(size=(n, dim)), columns=[f"c{i}" for i in range(dim)]
    )


def test_dcr_default_metric_matches_brute_force_reference():
    # The whole point of the NearestNeighbors switch is that it computes
    # the EXACT same row-wise minimum as the original dense-matrix
    # brute-force approach, just without materializing the full matrix
    # -- confirm that directly against an independently-computed brute
    # -force reference, not just "doesn't crash".
    original = _numeric_df(500, 12, seed=1)
    synthetic = _numeric_df(200, 12, seed=2)

    reference = manhattan_distances(original.values, synthetic.values).min(axis=0)
    result = SyntheticDataBench.compute_distance_to_closest_records(
        original, synthetic, n_test=len(synthetic)
    )
    assert np.allclose(result.values, reference, atol=1e-8)
    assert list(result.index) == list(synthetic.index)


def test_dcr_known_alternate_metric_matches_brute_force_reference():
    # euclidean_distances is also mapped to NearestNeighbors' own
    # "euclidean" metric name (not just the default manhattan) -- confirm
    # that mapping is correct too, not just the default case.
    original = _numeric_df(300, 8, seed=3)
    synthetic = _numeric_df(120, 8, seed=4)

    reference = euclidean_distances(original.values, synthetic.values).min(axis=0)
    result = SyntheticDataBench.compute_distance_to_closest_records(
        original, synthetic, n_test=len(synthetic), distance=euclidean_distances
    )
    assert np.allclose(result.values, reference, atol=1e-6)


def test_dcr_unrecognized_custom_distance_falls_back_to_brute_force():
    # A distance callable this codebase doesn't recognize (i.e. not one
    # of the specific sklearn functions mapped to a NearestNeighbors
    # metric name) must still work -- via the original brute-force path,
    # called exactly once, not silently ignored or double-computed.
    original = _numeric_df(200, 6, seed=5)
    synthetic = _numeric_df(80, 6, seed=6)

    calls = {"n": 0}

    def custom_distance(a, b):
        calls["n"] += 1
        return manhattan_distances(a, b)

    reference = manhattan_distances(original.values, synthetic.values).min(axis=0)
    result = SyntheticDataBench.compute_distance_to_closest_records(
        original, synthetic, n_test=len(synthetic), distance=custom_distance
    )
    assert calls["n"] == 1
    assert np.allclose(result.values, reference, atol=1e-8)


def test_dcr_respects_n_test_smaller_than_synthetic():
    original = _numeric_df(200, 5, seed=7)
    synthetic = _numeric_df(80, 5, seed=8)

    result = SyntheticDataBench.compute_distance_to_closest_records(
        original, synthetic, n_test=30
    )
    assert len(result) == 30
    reference = manhattan_distances(original.values, synthetic.iloc[:30].values).min(
        axis=0
    )
    assert np.allclose(result.values, reference, atol=1e-8)


def test_get_dcr_end_to_end_via_bench():
    # Exercises the real caller (SyntheticDataBench.get_dcr), not just
    # the static method directly -- confirms the preprocessing/object-
    # dtype branch and register_synthetic_data wiring still produce
    # sane, non-negative DCR values of the expected length.
    rng = np.random.default_rng(RANDOM_SEED)
    n = 300
    data = pd.DataFrame(
        {
            "num1": rng.normal(size=n),
            "num2": rng.normal(size=n),
            "cat1": rng.choice(["a", "b", "c"], size=n),
            "target": rng.integers(0, 2, size=n),
        }
    )
    bench = SyntheticDataBench(
        data=data,
        target_col="target",
        categorical=True,
        target_pos_val=1,
        test_size=0.2,
        random_state=RANDOM_SEED,
    )
    synthetic = data.sample(
        n=bench.n_train + bench.n_test, replace=True, random_state=RANDOM_SEED
    ).reset_index(drop=True)
    bench.register_synthetic_data(synthetic)

    dcr_synth = bench.get_dcr(is_test=False)
    dcr_test = bench.get_dcr(is_test=True)
    assert len(dcr_synth) == bench.n_test
    assert len(dcr_test) == bench.n_test
    assert (dcr_synth >= 0).all()
    assert (dcr_test >= 0).all()


# ---------------------------------------------------------------------
# compute_sensitivity_metric's `preprocessor` param + its
# _maybe_fit_shared_preprocessor helper -- lets many calls against the
# same reference population (compute_sensitivity_threshold's bootstrap
# loop chief among them) reuse ONE fitted StandardScaler+OneHotEncoder
# instead of each one refitting its own (benchmarked ~2.2x faster on
# the preprocessing step alone at Adult-like scale; confirmed not a
# leakage risk since every caller's `original` is already a subset of
# whatever population the shared preprocessor is fit on, and
# `synthetic` is never part of the fit either way).
# ---------------------------------------------------------------------
def _mixed_df(n, seed):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "num1": rng.normal(size=n),
            "num2": rng.normal(size=n),
            "cat1": rng.choice(["a", "b", "c"], size=n),
        }
    )


def test_maybe_fit_shared_preprocessor_none_for_numeric_low_dim():
    # Matches compute_sensitivity_metric's own internal no-op condition
    # (`object_dtypes.empty and shape[1] < max_col_nums`) -- purely
    # numeric, low-dimensional data needs no preprocessing at all.
    numeric_only = pd.DataFrame(
        np.random.default_rng(0).normal(size=(100, 5)), columns=list("abcde")
    )
    result = SyntheticDataBench._maybe_fit_shared_preprocessor(
        numeric_only, max_col_nums=50
    )
    assert result is None


def test_maybe_fit_shared_preprocessor_fits_for_categorical_data():
    data = _mixed_df(100, seed=0)
    result = SyntheticDataBench._maybe_fit_shared_preprocessor(data, max_col_nums=50)
    assert result is not None
    # A real, usable fitted transformer -- not just a truthy placeholder.
    transformed = result.transform(data)
    transformed = (
        transformed.toarray() if hasattr(transformed, "toarray") else transformed
    )
    assert transformed.shape[0] == len(data)


def test_compute_sensitivity_metric_shared_preprocessor_matches_fresh_fit():
    # When the shared preprocessor is fit on EXACTLY the same data the
    # internal fresh-fit path would fit on (`original`), the two paths
    # must agree exactly -- confirms `preprocessor=...` is a genuine
    # drop-in replacement for the internal fit, not a different
    # computation that happens to look similar.
    original = _mixed_df(300, seed=1)
    synthetic = _mixed_df(120, seed=2)
    test = _mixed_df(120, seed=3)

    baseline = SyntheticDataBench.compute_sensitivity_metric(
        original=original, synthetic=synthetic, test=test, qt_interval=50
    )

    shared = SyntheticDataBench._maybe_fit_shared_preprocessor(original)
    assert shared is not None
    reused = SyntheticDataBench.compute_sensitivity_metric(
        original=original,
        synthetic=synthetic,
        test=test,
        qt_interval=50,
        preprocessor=shared,
    )
    assert baseline == pytest.approx(reused, abs=1e-10)


def test_compute_sensitivity_metric_shared_preprocessor_numeric_only_is_noop():
    # A caller can pass `preprocessor=_maybe_fit_shared_preprocessor(...)`
    # unconditionally -- when the underlying data doesn't need
    # preprocessing at all, that helper returns None, so this must
    # behave exactly like the plain default-argument call.
    rng = np.random.default_rng(4)
    original = pd.DataFrame(rng.normal(size=(200, 4)), columns=list("wxyz"))
    synthetic = pd.DataFrame(rng.normal(size=(80, 4)), columns=list("wxyz"))
    test = pd.DataFrame(rng.normal(size=(80, 4)), columns=list("wxyz"))

    baseline = SyntheticDataBench.compute_sensitivity_metric(
        original=original, synthetic=synthetic, test=test, qt_interval=50
    )
    shared = SyntheticDataBench._maybe_fit_shared_preprocessor(original)
    assert shared is None
    reused = SyntheticDataBench.compute_sensitivity_metric(
        original=original,
        synthetic=synthetic,
        test=test,
        qt_interval=50,
        preprocessor=shared,
    )
    assert baseline == pytest.approx(reused, abs=1e-10)
