"""M5: model size x checkpoint rule. Usage: python research/m5_analyze.py research/results/m5 [--datasets ...]
Tables (paired on dataset x seed; lower is better except tstr; dcr_share ~0.5 = no memorisation):
  1. arm x rule means
  2. rule effect within each arm, vs `mean_best` (the recipe's recommendation)
  3. size effect (wk - default) at each rule   <- the fair size comparison
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd

RULES = {"mean_best_disc_model": "mean_best", "best_disc_model": "best_disc", "not_best_disc_model": "not_best", "last_epoch_model": "last_epoch"}
M = ["epoch", "marg_mean", "assoc_diff", "disc_dev", "tstr", "dcr_share"]

def load(d, datasets=None):
    rows = []
    for f in sorted(Path(d).glob("*.json")):
        if f.name.startswith("."): continue
        r = json.loads(f.read_text())
        if "error" in r: print("!! failed", f.stem); continue
        if datasets and r["job"]["dataset"] not in datasets: continue
        for k, v in r["checkpoints"].items():
            if v: rows.append(dict(dataset=r["job"]["dataset"], arm=r["job"]["arm"], seed=r["job"]["seed"], rule=RULES[k], stop_epoch=r["stop_epoch"], fit_s=r["fit_s"], **v))
    df = pd.DataFrame(rows); df["disc_dev"] = (df.disc_auc - .5).abs(); return df

def paired(a, b, cols=M[1:]):
    k = ["dataset", "seed"]; a, b = a.set_index(k), b.set_index(k); idx = a.index.intersection(b.index); out = []
    for c in cols:
        d = (a.loc[idx, c] - b.loc[idx, c]).dropna()
        if len(d) < 2: continue
        s = f"{c} {d.mean():+.3f}±{d.std(ddof=1)/np.sqrt(len(d)):.3f}"
        if c != "dcr_share": s += f" ({int(((d>0) if c=='tstr' else (d<0)).sum())}/{int(((d<0) if c=='tstr' else (d>0)).sum())})"
        out.append(s)
    return len(idx), "  ".join(out)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("dir"); ap.add_argument("--datasets", nargs="*"); a = ap.parse_args()
    df = load(a.dir, a.datasets); pd.set_option("display.width", 220)
    print(f"{df.groupby(['dataset','arm','seed']).ngroups} fits; datasets: {sorted(df.dataset.unique())}\n")
    print("== 1. arm x rule means ==")
    print(df.groupby(["arm", "rule"])[M].mean().round(3).to_string())
    print("\n== 2. rule effect within an arm, vs mean_best (what the recipe loads) ==")
    for arm in sorted(df.arm.unique()):
        ref = df[(df.arm == arm) & (df.rule == "mean_best")]
        for rule in ["best_disc", "not_best", "last_epoch"]:
            n, s = paired(df[(df.arm == arm) & (df.rule == rule)], ref); print(f"  {arm:8s} {rule:10s} (n={n}) {s}")
    print("\n== 3. size effect (wk - default) at each rule (n = paired fits) ==")
    for rule in RULES.values():
        n, s = paired(df[(df.arm == "wk") & (df.rule == rule)], df[(df.arm == "default") & (df.rule == rule)]); print(f"  {rule:10s} (n={n}) {s}")
    print("\n== stop epochs: ", df.drop_duplicates(['dataset','arm','seed']).groupby('arm').stop_epoch.median().round(0).to_dict(), " fit_s (median):", df.drop_duplicates(['dataset','arm','seed']).groupby('arm').fit_s.median().round(0).to_dict())

if __name__ == "__main__": main()
