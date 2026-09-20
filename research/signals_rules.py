"""Evaluate concrete label-free stopping rules built on the SRLG signals, using the
recorded curves (no retraining). A rule sees ONLY label-free columns
(srlg_ks, srlg_mean, srlg_tail) up to each checkpoint and returns the first epoch at
which it fires. Its quality is then read off the ground-truth columns at that epoch.

Rules:  rise(signal, delta)  = first epoch where signal >= running_min(signal) + delta
        ("the model has started to fit its training rows better than its samples")
Baselines: oracle_disc (best discriminator epoch), nll_min (held-out likelihood optimum,
needs held-out data), fixed_early / fixed_late (a fixed epoch), last (no stopping).

    python research/signals_rules.py research/results/s1
Judged by (lower is better): disc_dev = |disc_auc - 0.5|, marg_mean; and dcr_share
(memorisation, 0.5 = none). Rows are checkpoints (every 5 or 10 epochs), so stop epochs
are on that grid.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

def load(d):
    out = []
    for f in sorted(Path(d).glob("*.json")):
        if f.name.startswith("."): continue
        r = json.loads(f.read_text())
        if "error" in r: continue
        g = pd.DataFrame(r["rows"]).sort_values("epoch").reset_index(drop=True)
        g["dataset"], g["arm"], g["seed"] = r["job"]["dataset"], r["job"]["arm"], r["job"]["seed"]
        g["disc_dev"] = (g["disc_auc"] - 0.5).abs(); out.append(g)
    return out

def rise_stop(g, sig, delta):
    v = g[sig].to_numpy(); rm = np.minimum.accumulate(v)
    hit = np.where(v >= rm + delta)[0]
    # ignore the initial transient: a rule may only fire after the running minimum has been set
    hit = hit[hit > 0]
    return int(hit[0]) if len(hit) else len(g) - 1   # never fires -> runs to the end

def at(g, i): return g.iloc[i]

def main(d):
    runs = load(d); pd.set_option("display.width", 220)
    rules = {}
    for sig, deltas in (("srlg_ks", (0.05, 0.1, 0.2, 0.3, 0.5)), ("srlg_mean", (0.5, 1, 2, 4)), ("srlg_tail", (0.5, 1, 2))):
        for dl in deltas: rules[f"rise({sig},{dl})"] = (lambda g, s=sig, x=dl: rise_stop(g, s, x))
    rules["oracle_disc"] = lambda g: int(g["disc_dev"].idxmin())
    rules["nll_min (needs held-out)"] = lambda g: int(g["nll_test"].idxmin())
    rules["last (no stopping)"] = lambda g: len(g) - 1
    for arm in sorted({g.arm[0] for g in runs}):
        A = [g for g in runs if g.arm[0] == arm]
        ep = lambda g, e: int(np.argmin(np.abs(g.epoch.to_numpy() - e)))
        fixed = {"default": (30, 60), "wk": (100, 200)}.get(arm, (30, 100))
        rules_a = dict(rules); rules_a[f"fixed {fixed[0]} ep"] = lambda g, e=fixed[0]: ep(g, e); rules_a[f"fixed {fixed[1]} ep"] = lambda g, e=fixed[1]: ep(g, e)
        rows = []
        for name, fn in rules_a.items():
            st = [fn(g) for g in A]
            rows.append(dict(rule=name, stop_epoch=np.median([at(g, i).epoch for g, i in zip(A, st)]),
                             disc_dev=np.mean([at(g, i).disc_dev for g, i in zip(A, st)]),
                             marg_mean=np.mean([at(g, i).marg_mean for g, i in zip(A, st)]),
                             dcr_share=np.mean([at(g, i).dcr_share for g, i in zip(A, st)]),
                             frac_fired=np.mean([i < len(g) - 1 for g, i in zip(A, st)])))
        print(f"\n===== arm {arm}: {len(A)} runs ({sorted({g.dataset[0] for g in A})}) =====")
        print(pd.DataFrame(rows).set_index("rule").round(3).to_string())

if __name__ == "__main__":
    main(sys.argv[1])
