"""Compare learning curves against a reference arm. Usage:
    python research/curves_analyze.py research/results/c1 [--ref wk] [--weights raw]
Paired on (dataset, seed, epoch). Reports, per arm: paired deltas vs the reference at
selected epochs (mean +- s.e., wins/losses; lower is better except tstr), the
best value along the curve, and epochs-to-quality (first epoch at which the arm's
marg_mean <= the reference's value at its LAST epoch -- an efficiency measure).
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd

METRICS = [("marg_mean", False), ("assoc_diff", False), ("disc_dev", False), ("tstr", True), ("dcr_share", None), ("nll_test", False)]

def load(dirs):
    out = []
    for d in dirs:
        for f in sorted(Path(d).glob("*.json")):
            if f.name.startswith("."): continue
            r = json.loads(f.read_text())
            if "error" in r: print("!! failed:", f.stem); continue
            df = pd.DataFrame(r["rows"]); df["dataset"], df["arm"], df["seed"] = r["job"]["dataset"], r["job"]["arm"], r["job"]["seed"]
            df["fit_s"], df["n_params"] = r["fit_s"], r["n_params"]
            out.append(df)
    df = pd.concat(out, ignore_index=True); df["disc_dev"] = (df["disc_auc"] - 0.5).abs(); return df

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("dirs", nargs="+"); ap.add_argument("--ref", default="wk"); ap.add_argument("--weights", default="raw")
    ap.add_argument("--epochs", nargs="+", type=int, default=[10, 30, 50, 100]); a = ap.parse_args()
    df = load(a.dirs); df = df[df["weights"] == a.weights] if a.weights != "any" else df
    df["arm"] = np.where(df["weights"] != "raw", df["arm"] + "/" + df["weights"], df["arm"]) if a.weights == "any" else df["arm"]
    key = ["dataset", "seed", "epoch"]; ref = df[df.arm == a.ref].set_index(key)
    pd.set_option("display.width", 250, "display.max_columns", 40)
    print(f"reference arm: {a.ref}   units: {ref.reset_index()[['dataset','seed']].drop_duplicates().shape[0]}   weights: {a.weights}")
    tgt = ref.reset_index().sort_values("epoch").groupby(["dataset", "seed"])["marg_mean"].last()
    for arm in [x for x in dict.fromkeys(df.arm) if x != a.ref]:
        x = df[df.arm == arm].set_index(key); idx = x.index.intersection(ref.index)
        if len(idx) == 0: continue
        print(f"\n=== {arm} vs {a.ref} ===")
        for ep in a.epochs:
            ii = [i for i in idx if i[2] == ep]
            if len(ii) < 2: continue
            cells = []
            for m, hib in METRICS:
                d = (x.loc[ii, m] - ref.loc[ii, m]).dropna()
                if d.empty: continue
                s = f"{m} {d.mean():+.4f}±{d.std(ddof=1)/np.sqrt(len(d)):.4f}"
                if hib is not None: s += f" ({int(((d>0) if hib else (d<0)).sum())}/{int(((d<0) if hib else (d>0)).sum())})"
                cells.append(s)
            print(f"  epoch {ep:>3} (n={len(ii):>2}): " + " | ".join(cells))
        xr = x.reset_index(); e2q = []
        for (ds, sd), g in xr.groupby(["dataset", "seed"]):
            if (ds, sd) not in tgt.index: continue
            hit = g[g.marg_mean <= tgt[(ds, sd)]].epoch
            e2q.append(hit.min() if len(hit) else np.nan)
        e2q = pd.Series(e2q); print(f"  epochs to reach the reference's final marg_mean: median {e2q.median():.0f}  reached in {e2q.notna().mean():.0%} of runs  (reference final epoch = {int(ref.reset_index().epoch.max())})")
        print(f"  best marg_mean along curve: {xr.groupby(['dataset','seed']).marg_mean.min().mean():.4f}   (reference: {ref.reset_index().groupby(['dataset','seed']).marg_mean.min().mean():.4f})   dcr_share at last epoch: {xr[xr.epoch==xr.epoch.max()].dcr_share.mean():.3f}   fit_s: {xr.drop_duplicates(['dataset','seed']).fit_s.mean():.0f}")

if __name__ == "__main__":
    main()
