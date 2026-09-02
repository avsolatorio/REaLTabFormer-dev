import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
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
    assert list(result.columns) == ["actual", "original_predictions", "synthetic_predictions"]
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
