"""M6: constrained_loss vs the standard loss in the REAL regime (paired with M5 on dataset x seed x rule).
    python research/m6_analyze.py research/results/m5 ../REaLTabFormer-dev-cl/research/results/m6
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
import m5_analyze as M

def main(m5dir, m6dir, suffix="_cl"):
    a = M.load(m5dir); b = M.load(m6dir); df = pd.concat([a, b], ignore_index=True)
    print(f"m5 fits: {a.groupby(['dataset','arm','seed']).ngroups}  m6 fits: {b.groupby(['dataset','arm','seed']).ngroups}")
    stop = df.drop_duplicates(["dataset", "arm", "seed"]).set_index(["dataset", "arm", "seed"])
    for base in ("default", "wk"):
        cl = base + suffix
        print(f"\n=== {cl} vs {base} (paired on dataset x seed; lower better except tstr) ===")
        for rule in M.RULES.values():
            n, s = M.paired(df[(df.arm == cl) & (df.rule == rule)], df[(df.arm == base) & (df.rule == rule)]); print(f"  {rule:10s} (n={n}) {s}")
        both = [(d, s) for (d, arm, s) in stop.index if arm == cl and (d, base, s) in stop.index]
        if both:
            se = [(stop.loc[(d, cl, s), "stop_epoch"], stop.loc[(d, base, s), "stop_epoch"], stop.loc[(d, cl, s), "fit_s"], stop.loc[(d, base, s), "fit_s"]) for d, s in both]
            se = np.array(se); print(f"  stop epoch (median): {np.median(se[:,0]):.0f} vs {np.median(se[:,1]):.0f}   fit_s (median, load-confounded): {np.median(se[:,2]):.0f} vs {np.median(se[:,3]):.0f}")
        for rule in ("mean_best", "last_epoch"):
            x = df[(df.arm == cl) & (df.rule == rule)]; y = df[(df.arm == base) & (df.rule == rule)]
            print(f"  loaded epoch [{rule}]: {x.epoch.median():.0f} vs {y.epoch.median():.0f}")
if __name__ == "__main__": main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "_cl")
