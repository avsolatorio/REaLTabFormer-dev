"""Aggregate bench.py results: per-arm means and paired deltas vs a reference.

    python research/summarize.py research/results/m1 [--ref base/default]
                                    [--by-dataset]

An "arm" is `config/variant`. Deltas are paired on (dataset, seed), so they
cancel the dominant between-dataset/between-split variance; the reported
"+-" is the standard error of the paired difference and "w/l" counts the
paired units the arm beat / lost to the reference on (a metric-appropriate
sign, ties ignored). Lower is better for every column except tstr.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = [  # (column, higher_is_better)
    ("marg_mean", False), ("tail_err", False), ("assoc_diff", False),
    ("tstr", True), ("abs_gap", False), ("disc_dev", False),
    ("frac_suspicious", False), ("exact_dup", False), ("dcr_ratio", None),
    ("stopped_epoch", None), ("fit_s", None),
]


def load(d: Path) -> pd.DataFrame:
    rows, failed = [], []
    for f in sorted(d.glob("*.json")):
        r = json.loads(f.read_text())
        if "error" in r:
            failed.append(f.stem)
            continue
        for v, m in r["variants"].items():
            if "error" in m:
                failed.append(f"{f.stem}/{v}")
                continue
            rows.append(dict(
                dataset=r["job"]["dataset"], config=r["job"]["config"], seed=r["job"]["seed"],
                variant=v, arm=f"{r['job']['config']}/{v}",
                stopped_epoch=r["stopped_epoch"], fit_s=r["fit_s"], **m))
    if failed:
        print(f"!! {len(failed)} failed jobs/variants: {failed[:6]}{' ...' if len(failed) > 6 else ''}")
    df = pd.DataFrame(rows)
    if len(df):
        df["abs_gap"] = df["util_gap"].abs()
        df["disc_dev"] = (df["disc_auc"] - 0.5).abs()
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--ref", default=None)
    ap.add_argument("--by-dataset", action="store_true")
    a = ap.parse_args()
    df = load(Path(a.dir))
    if df.empty:
        print("no results")
        return
    arms = list(dict.fromkeys(df["arm"]))
    ref = a.ref or arms[0]
    cols = [m for m, _ in METRICS if m in df]
    pd.set_option("display.width", 250, "display.max_columns", 30)

    print(f"\n{df.groupby('arm').size().rename('n_units').to_string()}\n")
    print("== arm means (over all dataset x seed units) ==")
    print(df.groupby("arm")[cols].mean().reindex(arms).round(4).to_string())

    key = ["dataset", "seed"]
    r = df[df["arm"] == ref].set_index(key)
    print(f"\n== paired delta vs {ref}  (arm - ref, mean +- s.e.; w/l = arm better/worse) ==")
    out = []
    for arm in arms:
        if arm == ref:
            continue
        x = df[df["arm"] == arm].set_index(key)
        idx = x.index.intersection(r.index)
        row = {"arm": arm, "n": len(idx)}
        for m, hib in METRICS:
            if m not in df or len(idx) < 2:
                continue
            d = (x.loc[idx, m] - r.loc[idx, m]).dropna()
            if d.empty:
                continue
            s = f"{d.mean():+.4f}±{d.std(ddof=1) / np.sqrt(len(d)):.4f}"
            if hib is not None:
                better = (d > 0) if hib else (d < 0)
                worse = (d < 0) if hib else (d > 0)
                s += f" {int(better.sum())}/{int(worse.sum())}"
            row[m] = s
        out.append(row)
    if out:
        print(pd.DataFrame(out).set_index("arm").to_string())

    if a.by_dataset:
        for m in ["marg_mean", "assoc_diff", "tstr", "frac_suspicious"]:
            print(f"\n== {m} by dataset ==")
            print(df.pivot_table(index="arm", columns="dataset", values=m).reindex(arms).round(4).to_string())


if __name__ == "__main__":
    main()
