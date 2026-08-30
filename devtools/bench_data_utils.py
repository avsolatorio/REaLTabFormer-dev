"""Benchmark harness for the data_utils pipeline.

Not a pytest test -- run manually:

    PYTHONPATH=src python devtools/bench_data_utils.py

Times process_data, build_vocab, and make_dataset independently on a
synthetic mixed-dtype dataframe at a few (rows, columns) scales, so a
performance change to one stage can be attributed correctly instead of
lumped into one end-to-end number.
"""

import time
from typing import Callable

import numpy as np
import pandas as pd

from realtabformer.data_utils import build_vocab, make_dataset, process_data


def make_synthetic_df(n_rows: int, n_numeric: int, n_categorical: int, n_datetime: int) -> pd.DataFrame:
    rng = np.random.default_rng(1029)
    data = {}
    for i in range(n_numeric):
        data[f"num_{i}"] = rng.normal(loc=100, scale=50, size=n_rows).round(4)
    for i in range(n_categorical):
        n_unique = min(n_rows, 50)
        cats = [f"cat_{i}_{j}" for j in range(n_unique)]
        data[f"cat_{i}"] = rng.choice(cats, size=n_rows)
    for i in range(n_datetime):
        start = pd.Timestamp("2015-01-01").value // 10**9
        end = pd.Timestamp("2024-01-01").value // 10**9
        secs = rng.integers(start, end, size=n_rows)
        data[f"date_{i}"] = pd.to_datetime(secs, unit="s")
    return pd.DataFrame(data)


def timed(label: str, fn: Callable) -> float:
    start = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - start
    print(f"  {label}: {elapsed:.4f}s")
    return elapsed, result


def bench(n_rows: int, n_numeric: int, n_categorical: int, n_datetime: int) -> None:
    n_cols = n_numeric + n_categorical + n_datetime
    print(f"\n=== rows={n_rows} cols={n_cols} (numeric={n_numeric}, categorical={n_categorical}, datetime={n_datetime}) ===")

    df = make_synthetic_df(n_rows, n_numeric, n_categorical, n_datetime)

    _, (pr_df, ctd, _) = timed("process_data", lambda: process_data(df))

    vocab_input = pr_df.copy()
    from realtabformer.data_utils.constants import SpecialTokens

    _, vocab = timed(
        "build_vocab",
        lambda: build_vocab(vocab_input, special_tokens=SpecialTokens.tokens(), add_columns=True),
    )

    _, dataset = timed(
        "make_dataset",
        lambda: make_dataset(pr_df, vocab, mask_rate=0.15),
    )
    print(f"  dataset size: {len(dataset)} rows")


if __name__ == "__main__":
    for n_rows in (10_000, 100_000):
        bench(n_rows, n_numeric=15, n_categorical=15, n_datetime=10)
