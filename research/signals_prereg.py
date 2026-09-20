"""Score the pre-registered rule R* (H15b) against its pre-registered criteria, on held-out runs.
    python research/signals_prereg.py research/results/s1h_d research/results/s1h_w [--delta 0.25]
Rule R*: stop at the first checkpoint where srlg_ks >= running_min(srlg_ks) + delta.
Criteria (fixed before the data): (i) fires in >=80% of runs; (ii) stop epoch within a factor 2 of the
discriminator-optimal epoch in >=75% of runs; (iii) mean discriminator distance at R*'s stop is lower than at the
last epoch AND lower than at the held-out-NLL minimum; (iv) mean dcr_share at R*'s stop >= 0.03 lower than at the last epoch.
Falsified if (iii) fails or R* fires in fewer than half the runs. Nothing here is tuned on these runs."""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
import signals_rules as SR

def rstar(g, delta):
    v = g["srlg_ks"].to_numpy(); hit = np.where(v >= np.minimum.accumulate(v) + delta)[0]; hit = hit[hit > 0]
    return (int(hit[0]), True) if len(hit) else (len(g) - 1, False)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("dirs", nargs="+"); ap.add_argument("--delta", type=float, default=0.25); a = ap.parse_args()
    runs = [g for d in a.dirs for g in SR.load(d)]; print(f"{len(runs)} held-out runs: {sorted({(g.dataset[0], g.arm[0]) for g in runs})}\n")
    rows = []
    for g in runs:
        i, fired = rstar(g, a.delta); orc = int(g["disc_dev"].idxmin()); nll = int(g["nll_test"].idxmin()); last = len(g) - 1
        e = lambda k: g.epoch.iloc[k]
        rows.append(dict(dataset=g.dataset[0], arm=g.arm[0], seed=g.seed[0], fired=fired, stop=e(i), oracle=e(orc),
                         ratio=e(i) / max(e(orc), 1), disc_stop=g.disc_dev.iloc[i], disc_last=g.disc_dev.iloc[last], disc_nll=g.disc_dev.iloc[nll],
                         dcr_stop=g.dcr_share.iloc[i], dcr_last=g.dcr_share.iloc[last], marg_stop=g.marg_mean.iloc[i], marg_last=g.marg_mean.iloc[last]))
    R = pd.DataFrame(rows); pd.set_option("display.width", 220)
    print(R.round(3).to_string(index=False)); print()
    within2 = ((R.ratio >= 0.5) & (R.ratio <= 2.0)).mean()
    c = {"(i) fires in >=80% of runs": (R.fired.mean(), R.fired.mean() >= 0.8),
         "(ii) stop within x2 of disc-optimal epoch in >=75%": (within2, within2 >= 0.75),
         "(iii-a) mean disc distance at stop < at last epoch": ((R.disc_stop.mean(), R.disc_last.mean()), R.disc_stop.mean() < R.disc_last.mean()),
         "(iii-b) mean disc distance at stop < at held-out-NLL min": ((R.disc_stop.mean(), R.disc_nll.mean()), R.disc_stop.mean() < R.disc_nll.mean()),
         "(iv) mean dcr_share at stop >=0.03 below last epoch": ((R.dcr_stop.mean(), R.dcr_last.mean()), (R.dcr_last.mean() - R.dcr_stop.mean()) >= 0.03)}
    for k, (v, ok) in c.items(): print(f"  [{'PASS' if ok else 'FAIL'}] {k}: {np.round(v, 3)}")
    print(f"\n  falsified? (iii) fails: {not (c['(iii-a) mean disc distance at stop < at last epoch'][1] and c['(iii-b) mean disc distance at stop < at held-out-NLL min'][1])}   fires in <half the runs: {R.fired.mean() < 0.5}")
    for arm, g in R.groupby("arm"): print(f"  {arm}: median stop {g.stop.median():.0f} (oracle {g.oracle.median():.0f}); disc at stop {g.disc_stop.mean():.3f} vs last {g.disc_last.mean():.3f} vs NLL-min {g.disc_nll.mean():.3f}; dcr_share {g.dcr_stop.mean():.3f} vs last {g.dcr_last.mean():.3f}; marg {g.marg_stop.mean():.3f} vs last {g.marg_last.mean():.3f}")
if __name__ == "__main__": main()
