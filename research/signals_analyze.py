"""Judge the label-free signals (H15) against ground truth. Usage:
    python research/signals_analyze.py research/results/s1
Per run (dataset x arm x seed) and pooled over runs:
  * Spearman(signal, gap_true) and Spearman(signal, dcr_share) over the checkpoints
  * ep_star = epoch of minimum held-out NLL (the classical stopping point);
    where each signal's own minimum falls relative to it
  * how the true optimum for SAMPLE quality (min marg_mean, min |disc_auc-0.5|)
    relates to ep_star -- likelihood overfitting is not sample-quality overfitting
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr

SIGS = ["srlg_mean", "srlg_ks", "srlg_tail"]

def load(d):
    out = []
    for f in sorted(Path(d).glob("*.json")):
        if f.name.startswith("."): continue
        r = json.loads(f.read_text())
        if "error" in r: print("!! failed:", f.stem); continue
        df = pd.DataFrame(r["rows"]); df["dataset"], df["arm"], df["seed"] = r["job"]["dataset"], r["job"]["arm"], r["job"]["seed"]
        out.append(df)
    return pd.concat(out, ignore_index=True)

def sp(a, b):
    ok = a.notna() & b.notna()
    return spearmanr(a[ok], b[ok])[0] if ok.sum() >= 4 and a[ok].nunique() > 1 and b[ok].nunique() > 1 else np.nan

def main(d):
    df = load(d)
    df["disc_dev"] = (df["disc_auc"] - 0.5).abs()
    rows = []
    for (ds, arm, seed), g in df.groupby(["dataset", "arm", "seed"]):
        g = g.sort_values("epoch"); star = int(g.loc[g["nll_test"].idxmin(), "epoch"])
        r = dict(dataset=ds, arm=arm, seed=seed, n_ck=len(g), ep_max=int(g.epoch.max()), ep_star_nll=star,
                 ep_best_marg=int(g.loc[g["marg_mean"].idxmin(), "epoch"]),
                 ep_best_disc=int(g.loc[g["disc_dev"].idxmin(), "epoch"]))
        for s in SIGS:
            r[f"rho_gap[{s}]"] = sp(g[s], g["gap_true"]); r[f"rho_dcr[{s}]"] = sp(g[s], g["dcr_share"])
            r[f"argmin[{s}]"] = int(g.loc[g[s].idxmin(), "epoch"])
        rows.append(r)
    R = pd.DataFrame(rows); pd.set_option("display.width", 250, "display.max_columns", 40)
    print(f"\n{len(R)} runs\n\n== where the optimum falls (epochs), median [q25,q75] per arm ==")
    for arm, g in R.groupby("arm"):
        q = lambda c: f"{g[c].median():.0f} [{g[c].quantile(.25):.0f},{g[c].quantile(.75):.0f}]"
        print(f"{arm:8s} min held-out NLL: {q('ep_star_nll'):>14s} | best marg_mean: {q('ep_best_marg'):>14s} | best disc: {q('ep_best_disc'):>14s} | run length {g.ep_max.median():.0f}")
    print("\n== within-run Spearman with the TRUE generalisation gap (nll_test - nll_train), median [q25,q75] ; share of runs > 0.6 ==")
    for arm, g in R.groupby("arm"):
        for s in SIGS:
            c = g[f"rho_gap[{s}]"].dropna(); print(f"{arm:8s} {s:10s} {c.median():+.2f} [{c.quantile(.25):+.2f},{c.quantile(.75):+.2f}]  >0.6: {(c>0.6).mean():.0%}  (n={len(c)})")
    print("\n== within-run Spearman with MEMORISATION (dcr_share) ==")
    for arm, g in R.groupby("arm"):
        for s in SIGS:
            c = g[f"rho_dcr[{s}]"].dropna(); print(f"{arm:8s} {s:10s} {c.median():+.2f} [{c.quantile(.25):+.2f},{c.quantile(.75):+.2f}]  >0.6: {(c>0.6).mean():.0%}  (n={len(c)})")
    print("\n== signal argmin minus held-out-NLL argmin (epochs; 0 = the signal's turning point matches the truth) ==")
    for arm, g in R.groupby("arm"):
        for s in SIGS:
            d_ = (g[f"argmin[{s}]"] - g["ep_star_nll"]); print(f"{arm:8s} {s:10s} median {d_.median():+.0f}  mean|d| {d_.abs().mean():.1f}")
    R.to_csv(Path(d) / "signals_summary.csv", index=False)

if __name__ == "__main__":
    main(sys.argv[1])
