"""OOV-handling experiment (hypothesis H8): what happens when a seed_input
carries a category value the model never saw in training?

Design. Pick a categorical column C and hold ONE level L out of the training
data entirely, so L is genuinely out-of-vocabulary. Train on the rest. Then
seed generation with {C: L} and look at the *other* columns the model
produces. Two references, both computed from real data:
  * the true conditional  P(other | C = L)   -- from the held-out L rows
  * the marginal          P(other)           -- from the training rows
A useful OOV policy should behave like "I don't know this value": ideally it
falls back to something at least as close to the truth as the marginal.
Reported distances are the mean over the non-seeded columns of KS (numeric)
or TVD (categorical); lower is closer.

Control. The same measurement for *known* levels shows how much conditioning
can help at all (`known_gain`), so OOV results can be read against it.

Arms are (training dropout) x (OOV policy at seed-encoding time). The
`oov_strategy` switch only changes how seeds are encoded, so one trained
model is evaluated under both policies.

    python research/oov_bench.py --seed 0 --out research/results/oov
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "cusum_validation"))
sys.path.insert(0, str(REPO / "research"))
warnings.filterwarnings("ignore")

import bench  # noqa: E402

N_PER_TRIAL = 300
N_TRIALS = 6


def col_dist(a: pd.DataFrame, b: pd.DataFrame, cols) -> float:
    """Mean per-column KS (numeric, >10 levels) / TVD (otherwise)."""
    from scipy.stats import ks_2samp

    out = []
    for c in cols:
        if bench._is_num(b[c]):
            x = pd.to_numeric(a[c], errors="coerce").dropna().astype(float)
            out.append(ks_2samp(x, b[c].astype(float)).statistic if len(x) else 1.0)
        else:
            p = a[c].astype(str).value_counts(normalize=True)
            q = b[c].astype(str).value_counts(normalize=True)
            idx = p.index.union(q.index)
            out.append(0.5 * float(np.abs(p.reindex(idx, fill_value=0) - q.reindex(idx, fill_value=0)).sum()))
    return float(np.mean(out))


def seeded(model, level, col, n_trials, device="cuda") -> pd.DataFrame:
    parts = []
    for _ in range(n_trials):
        s = model.sample(n_samples=N_PER_TRIAL, seed_input={col: level}, device=device, gen_batch=N_PER_TRIAL)
        parts.append(s)
    return pd.concat(parts, ignore_index=True)


def train_model(train, seed, unk_dropout, tmp, target):
    import torch
    from transformers import GPT2Config

    import run_experiment as rx
    from realtabformer.realtabformer import REaLTabFormer

    torch.manual_seed(seed)
    model = REaLTabFormer(
        model_type="tabular", tabular_config=GPT2Config(n_layer=6), epochs=300,
        batch_size=8, random_state=seed, checkpoints_dir=str(tmp / "ckpt"),
        unk_dropout=unk_dropout,
    )
    t0 = time.time()
    trainer = model.fit(
        train, device="cuda", n_critic=5, n_critic_stop=2,
        num_bootstrap=rx.default_sensitivity_num_bootstrap(len(train)),
        sensitivity_cache_dir=str(REPO / "research" / "cache"),
        sensitivity_bootstrap_n_jobs=8, load_from_best_mean_sensitivity=True,
        gen_kwargs={"gen_batch": 512}, target_col=None,
    )
    return model, dict(fit_s=time.time() - t0, global_step=int(trainer.state.global_step))


def run(seed: int, out: Path, dataset: str, col: str, dropouts) -> None:
    import shutil

    df, target, _, _ = bench.load_dataset(dataset)
    df = df[[col] + [c for c in df.columns if c != col]]  # v1 seeds must be a column prefix

    rng = np.random.default_rng(seed)
    freq = df[col].value_counts(normalize=True)
    cands = sorted(freq[(freq >= 0.03) & (freq <= 0.10)].index)
    level = str(cands[int(rng.integers(len(cands)))])
    held = df[df[col] == level].reset_index(drop=True)
    train = df[df[col] != level].reset_index(drop=True)
    others = [c for c in df.columns if c != col]

    res = dict(dataset=dataset, col=col, seed=seed, level=level, n_held=len(held),
               n_train=len(train), level_freq=float(freq[level]),
               held_vs_marginal=col_dist(held, train, others), arms={})
    print(f"seed={seed} level={level!r} n_held={len(held)} held_vs_marginal={res['held_vs_marginal']:.4f}", flush=True)

    known = [l for l in train[col].value_counts().index if (train[col] == l).sum() >= 40][:3]
    for d in dropouts:
        tmp = out / f"tmp_s{seed}_d{d}"
        try:
            model, info = train_model(train, seed, d, tmp, target)
            unc = model.sample(n_samples=N_PER_TRIAL * N_TRIALS, device="cuda", gen_batch=N_PER_TRIAL * 2)[list(df.columns)]
            info["unc_vs_held"] = col_dist(unc, held, others)
            info["unc_vs_marginal"] = col_dist(unc, train, others)
            # Known-level control (only meaningful for the trained model).
            gains = []
            for l in known:
                g = seeded(model, l, col, 2)
                ref = train[train[col] == l]
                gains.append(col_dist(g, train, others) - col_dist(g, ref, others))
            info["known_gain"] = float(np.mean(gains))
            for strat in ("random", "unk"):
                model.vocab["oov_strategy"] = strat
                g = seeded(model, level, col, N_TRIALS)
                per_trial = [
                    col_dist(g.iloc[i * N_PER_TRIAL:(i + 1) * N_PER_TRIAL], held, others) for i in range(N_TRIALS)
                ]
                res["arms"][f"d{d}/{strat}"] = dict(
                    vs_true_cond=col_dist(g, held, others),
                    vs_marginal=col_dist(g, train, others),
                    trial_sd=float(np.std(per_trial)), **info,
                )
                print(f"  d={d} {strat:6s} vs_true_cond={res['arms'][f'd{d}/{strat}']['vs_true_cond']:.4f} "
                      f"vs_marginal={res['arms'][f'd{d}/{strat}']['vs_marginal']:.4f} "
                      f"(unc_vs_held={info['unc_vs_held']:.4f} known_gain={info['known_gain']:+.4f})", flush=True)
        except Exception:
            res["arms"][f"d{d}/ERROR"] = traceback.format_exc()
            print(traceback.format_exc(), flush=True)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        (out / f"{dataset}_{col}_s{seed}.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset", default="adult5k")
    ap.add_argument("--col", default="occupation")
    ap.add_argument("--dropouts", nargs="+", type=float, default=[0.0, 0.03, 0.10])
    a = ap.parse_args()
    o = Path(a.out)
    o.mkdir(parents=True, exist_ok=True)
    run(a.seed, o, a.dataset, a.col, a.dropouts)
