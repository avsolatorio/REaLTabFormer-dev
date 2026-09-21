"""Multi-seed benchmark harness for REaLTabFormer design experiments.

Why this exists: every result in DECISION_LOG.md is single-seed, and
quality-only metrics reward memorisation (see its "measure privacy
alongside quality, always" finding). This harness scores every run on
fidelity, downstream utility AND privacy together, across several seeds,
so a difference between two configs can be judged against seed noise.

One *job* = (dataset, config name, seed). A job trains one model, then
scores it under every entry of the config's `sample_variants` (so
sampling-time experiments reuse one training run instead of retraining).
Results are one JSON per job under `--out`; already-finished jobs are
skipped, so an interrupted matrix resumes where it stopped.

Usage (repo root of the worktree):
    python research/bench.py run --name m1 --configs base qenc \
        --datasets diabetes insurance --seeds 0 1 2 --workers 4
    python research/summarize.py research/results/m1
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "research"
# RTF_SRC lets a job run against a different checkout of the library (see the `src` key in configs.py),
# e.g. the code as it was before a change, for a true before/after comparison.
sys.path.insert(0, os.environ.get("RTF_SRC") or str(REPO / "src"))
sys.path.insert(0, str(REPO / "cusum_validation"))
sys.path.insert(0, str(EXP))

warnings.filterwarnings("ignore")

# Datasets used for exploration ("dev") vs. held back until a finding needs
# confirming ("holdout"), so picking a winner on dev doesn't overfit it.
DEV_DATASETS = ["diabetes", "insurance", "abalone", "adult5k"]
SYNTHETIC_DATASETS = ["hicard"]  # purpose-built to test one specific mechanism
HOLDOUT_DATASETS = ["wilt", "churn2"]


def make_hicard(n: int = 4000, n_levels: int = 300, seed: int = 0) -> pd.DataFrame:
    """Synthetic table with a Zipf-distributed 300-level categorical.

    Built to exercise columns that have far more than 50 admissible tokens
    (the HF default `top_k=50` truncates those); none of the bundled real
    datasets has such a column. `city` drives `income` (via a per-city
    offset) and, through it, `segment`; `age` is independent noise.
    """
    rng = np.random.default_rng(seed)
    w = 1.0 / np.arange(1, n_levels + 1) ** 1.1
    city_id = rng.choice(n_levels, size=n, p=w / w.sum())
    city_off = rng.normal(0, 1, n_levels)
    income = np.exp(10 + 0.5 * city_off[city_id] + rng.normal(0, 0.4, n))
    seg = np.where(income > np.quantile(income, 0.7), "high", np.where(income > np.quantile(income, 0.3), "mid", "low"))
    return pd.DataFrame(dict(
        city=[f"city_{i:03d}" for i in city_id],
        age=rng.integers(18, 80, n),
        income=income.round(2),
        segment=seg,
    ))


def load_dataset(name: str):
    """-> (df, target_col, is_classification, target_pos_val)"""
    import run_experiment as rx

    if name == "hicard":
        return make_hicard(), "segment", True, "high"

    if name == "adult5k":
        df = rx.load_adult().sample(5000, random_state=0).reset_index(drop=True)
        return df, "income", True, ">50K"
    loader, target, categorical, pos = rx.DATASET_CONFIGS[name]
    return loader().reset_index(drop=True), target, categorical, pos


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def _is_num(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and s.nunique() > 10


def _cramers_v(a: pd.Series, b: pd.Series) -> float:
    from scipy.stats import chi2_contingency

    tab = pd.crosstab(a, b)
    if tab.shape[0] < 2 or tab.shape[1] < 2:
        return 0.0
    chi2 = chi2_contingency(tab, correction=False)[0]
    n = tab.to_numpy().sum()
    return float(np.sqrt(chi2 / (n * (min(tab.shape) - 1))))


def _discretise(df: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    """Numeric cols -> 10 quantile bins (edges from `ref`), others -> str."""
    out = {}
    for c in df.columns:
        if _is_num(ref[c]):
            edges = np.unique(np.quantile(ref[c].astype(float), np.linspace(0, 1, 11)))
            edges[0], edges[-1] = -np.inf, np.inf
            out[c] = pd.cut(df[c].astype(float), edges, labels=False).astype(str)
        else:
            out[c] = df[c].astype(str)
    return pd.DataFrame(out)


def fidelity_metrics(train: pd.DataFrame, synth: pd.DataFrame) -> dict:
    from scipy.stats import ks_2samp

    ks, tvd, tail = [], [], []
    for c in train.columns:
        if _is_num(train[c]):
            a = train[c].astype(float).to_numpy()
            b = pd.to_numeric(synth[c], errors="coerce").astype(float).dropna().to_numpy()
            ks.append(ks_2samp(a, b).statistic)
            lo, q1, q99 = np.quantile(a, [0.01, 0.01, 0.99])
            scale = max(q99 - q1, 1e-9)
            tail.append(abs(np.quantile(b, 0.99) - q99) / scale)
        else:
            p = train[c].astype(str).value_counts(normalize=True)
            q = synth[c].astype(str).value_counts(normalize=True)
            idx = p.index.union(q.index)
            tvd.append(0.5 * float(np.abs(p.reindex(idx, fill_value=0) - q.reindex(idx, fill_value=0)).sum()))

    dtr, dsy = _discretise(train, train), _discretise(synth, train)
    cols = list(train.columns)
    diffs = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            diffs.append(
                abs(_cramers_v(dtr[cols[i]], dtr[cols[j]]) - _cramers_v(dsy[cols[i]], dsy[cols[j]]))
            )
    return dict(
        ks_mean=float(np.mean(ks)) if ks else float("nan"),
        tvd_mean=float(np.mean(tvd)) if tvd else float("nan"),
        marg_mean=float(np.mean(ks + tvd)),
        tail_err=float(np.mean(tail)) if tail else float("nan"),
        assoc_diff=float(np.mean(diffs)),
    )


def _encode(frames: dict, target: str, ref: pd.DataFrame, include_target: bool = False):
    from sklearn.preprocessing import OrdinalEncoder

    feats = [c for c in ref.columns if include_target or c != target]
    cat = [c for c in feats if not pd.api.types.is_numeric_dtype(ref[c])]
    enc = None
    if cat:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan)
        enc.fit(ref[cat].astype(str))
    out = {}
    for k, f in frames.items():
        X = f[feats].copy()
        for c in feats:
            if c not in cat:
                X[c] = pd.to_numeric(X[c], errors="coerce").astype(float)
        if cat:
            X[cat] = enc.transform(X[cat].astype(str))
        out[k] = X
    # HistGradientBoosting rejects categorical features with >255 levels;
    # such columns stay ordinal-coded and are treated as numeric.
    mask = [c in cat and ref[c].nunique() <= 255 for c in feats]
    return out, mask


def utility_metrics(train, synth, test, target, is_cls, pos) -> dict:
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.metrics import r2_score, roc_auc_score

    X, mask = _encode(dict(tr=train, sy=synth, te=test), target, train)
    kw = dict(max_iter=150, random_state=0, categorical_features=mask if any(mask) else None)

    def score(Xa, ya):
        if is_cls:
            ya = ya.astype(str)
            if ya.nunique() < 2:
                return 0.5
            m = HistGradientBoostingClassifier(**kw).fit(Xa, ya)
            cls = list(m.classes_)
            if str(pos) not in cls:
                return 0.5
            p = m.predict_proba(X["te"])[:, cls.index(str(pos))]
            return float(roc_auc_score((test[target].astype(str) == str(pos)).astype(int), p))
        m = HistGradientBoostingRegressor(**kw).fit(Xa, pd.to_numeric(ya, errors="coerce").astype(float))
        return float(r2_score(test[target].astype(float), m.predict(X["te"])))

    trtr, tstr = score(X["tr"], train[target]), score(X["sy"], synth[target])
    return dict(trtr=trtr, tstr=tstr, util_gap=trtr - tstr)


def discriminator_auc(train, synth, target, seed) -> float:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    # A discriminator should see the whole row, target included.
    X, mask = _encode(dict(tr=train, sy=synth), target, train, include_target=True)
    Xa = pd.concat([X["tr"], X["sy"]], ignore_index=True)
    y = np.r_[np.zeros(len(X["tr"])), np.ones(len(X["sy"]))]
    clf = HistGradientBoostingClassifier(
        max_iter=100, random_state=0, categorical_features=mask if any(mask) else None
    )
    p = cross_val_predict(
        clf, Xa, y, cv=StratifiedKFold(5, shuffle=True, random_state=seed), method="predict_proba"
    )[:, 1]
    return float(roc_auc_score(y, p))


def _row_keys(df: pd.DataFrame) -> pd.Series:
    parts = []
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            parts.append(pd.to_numeric(s, errors="coerce").astype(float).round(4).map("{:.4f}".format))
        else:
            parts.append(s.astype(str))
    return pd.concat(parts, axis=1).agg("|".join, axis=1)


def dcr_share(bench, synth_all, seed) -> float:
    """Direct memorisation check: of `n_test` synthetic rows, the share whose
    nearest real neighbour is in the TRAINING set rather than in the held-out
    set, using equally sized reference sets (a random train subsample the size
    of the test set) so 0.5 is the no-memorisation value -- synthetic data no
    closer to what the model saw than to real data it never saw. Values well
    above 0.5 mean the generator sits closer to its training rows than
    unseen real rows do. (Ties count 0.5; features are standardised numerics +
    one-hot categoricals, as in `SyntheticDataBench.preprocess_data`.)"""
    from sklearn.neighbors import NearestNeighbors

    cols = list(bench.train_df.columns)
    n = len(bench.test_df)
    tr = bench.train_df.sample(n=n, random_state=seed)
    sy = synth_all[cols].sample(n=n, random_state=seed)
    te = bench.test_df[cols]
    proc = bench.preprocess_data(data=tr, other=[sy, te])
    sy_p, te_p = proc["other"]
    d_tr = NearestNeighbors(n_neighbors=1, metric="manhattan").fit(proc["data"]).kneighbors(sy_p)[0].ravel()
    d_te = NearestNeighbors(n_neighbors=1, metric="manhattan").fit(te_p).kneighbors(sy_p)[0].ravel()
    return float(np.mean((d_tr < d_te) + 0.5 * (d_tr == d_te)))


def privacy_metrics(bench, synth_all, dcr_test) -> dict:
    bench.register_synthetic_data(synth_all)
    dcr_synth = bench.get_dcr(is_test=False)
    thr = dcr_test.quantile(0.05)
    dup = _row_keys(bench.synth_train_df[bench.train_df.columns]).isin(set(_row_keys(bench.train_df)))
    return dict(
        dcr_synth_mean=float(dcr_synth.mean()),
        dcr_test_mean=float(dcr_test.mean()),
        dcr_ratio=float(dcr_synth.mean() / max(dcr_test.mean(), 1e-9)),
        frac_suspicious=float((dcr_synth < thr).mean()),
        exact_dup=float(dup.mean()),
        dcr_share=dcr_share(bench, synth_all, bench.random_state),
    )


# --------------------------------------------------------------------------
# One job
# --------------------------------------------------------------------------
def run_job(job: dict) -> dict:
    import torch
    from transformers import GPT2Config

    import configs as cfgmod
    import run_experiment as rx
    from realtabformer.realtabformer import REaLTabFormer
    from realtabformer.rtf_analyze import SyntheticDataBench

    cfg = cfgmod.CONFIGS[job["config"]]
    seed = job["seed"]
    df, target, is_cls, pos = load_dataset(job["dataset"])

    bench = SyntheticDataBench(
        data=df, target_col=target, categorical=is_cls, target_pos_val=pos,
        test_size=0.2, random_state=seed,
    )
    train = bench.train_df.reset_index(drop=True)
    test = bench.test_df.reset_index(drop=True)
    bench.train_df = train
    cols = list(train.columns)

    torch.manual_seed(seed)
    np.random.seed(seed)

    gpt2 = GPT2Config(n_layer=6)
    for k, v in cfg.get("gpt2", {}).items():
        setattr(gpt2, k, v)

    tmp = Path(job["tmp"])
    tmp.mkdir(parents=True, exist_ok=True)
    model = REaLTabFormer(
        model_type="tabular",
        tabular_config=gpt2,
        epochs=cfg.get("epochs", 300),
        batch_size=cfg.get("batch_size", 8),
        random_state=seed,
        checkpoints_dir=str(tmp / "ckpt"),
        **cfg.get("init", {}),
    )
    model.training_args_kwargs.update(cfg.get("train_args", {}))

    fit_kw = dict(
        n_critic=5, n_critic_stop=2,
        num_bootstrap=rx.default_sensitivity_num_bootstrap(len(train)),
        sensitivity_cache_dir=str(EXP / "cache"),
        sensitivity_bootstrap_n_jobs=8,
        load_from_best_mean_sensitivity=True,
        gen_kwargs={"gen_batch": 512},
        target_col=target if cfg.get("teacher_force", True) else None,
    )
    fit_kw.update(cfg.get("fit", {}))

    t0 = time.time()
    trainer = model.fit(train, device="cuda", **fit_kw)
    fit_s = time.time() - t0
    steps_per_epoch = max(
        1,
        len(train) // (trainer.args.per_device_train_batch_size * trainer.args.gradient_accumulation_steps),
    )
    n_params = sum(p.numel() for p in model.model.parameters())

    dcr_test = None
    n_need = len(train) + len(test)
    out = dict(
        job=job, n_train=len(train), fit_s=fit_s, n_params=int(n_params),
        stopped_epoch=trainer.state.global_step / steps_per_epoch,
        global_step=int(trainer.state.global_step), variants={},
    )
    # Noise floor: how far *real held-out data* is from the training data on
    # the same metrics -- the best any generator could do at this sample size.
    orc = fidelity_metrics(train, test)
    orc["disc_auc"] = discriminator_auc(train, test, target, seed)
    out["oracle"] = orc
    for vname, skw in cfg.get("sample_variants", {"default": {}}).items():
        t1 = time.time()
        synth = model.sample(n_samples=int(n_need * 1.15) + 50, device="cuda", gen_batch=1024, **skw)
        n_raw = len(synth)
        synth = synth[cols].dropna().reset_index(drop=True)
        nan_row_frac = 1 - len(synth) / max(n_raw, 1)  # before truncating to n_need
        if len(synth) < n_need:
            out["variants"][vname] = dict(error=f"only {len(synth)}/{n_need} valid rows", n_raw=n_raw)
            continue
        synth = synth.iloc[:n_need]
        if dcr_test is None:
            bench.register_synthetic_data(synth)
            dcr_test = bench.get_dcr(is_test=True)
        s_train = synth.sample(n=len(train), random_state=seed).reset_index(drop=True)
        m = dict(sample_s=time.time() - t1, nan_row_frac=nan_row_frac)
        m.update(fidelity_metrics(train, s_train))
        m.update(utility_metrics(train, s_train, test, target, is_cls, pos))
        m["disc_auc"] = discriminator_auc(train, s_train, target, seed)
        m.update(privacy_metrics(bench, synth, dcr_test))
        out["variants"][vname] = m
        if job.get("save_synth"):
            d = Path(job["out_json"]).with_suffix("")
            d.mkdir(parents=True, exist_ok=True)
            s_train.to_csv(d / f"{vname}.csv.gz", index=False)
    return out


def child_main(job_path: str) -> None:
    job = json.loads(Path(job_path).read_text())
    res = None
    try:
        res = run_job(job)
    except Exception:  # record the failure instead of losing the whole matrix
        res = dict(job=job, error=traceback.format_exc())
    finally:
        shutil.rmtree(job["tmp"], ignore_errors=True)
    Path(job["out_json"]).write_text(json.dumps(res, indent=1, default=float))


# --------------------------------------------------------------------------
# Matrix driver
# --------------------------------------------------------------------------
def run_matrix(args) -> None:
    out = EXP / "results" / args.name
    out.mkdir(parents=True, exist_ok=True)
    jobs = []
    for d in args.datasets:
        for s in args.seeds:
            for c in args.configs:
                oj = out / f"{d}__{c}__s{s}.json"
                if oj.exists() and "error" not in json.loads(oj.read_text()):
                    continue
                jobs.append(dict(dataset=d, config=c, seed=s, out_json=str(oj),
                                 tmp=str(EXP / "tmp" / f"{args.name}_{d}_{c}_{s}"),
                                 save_synth=args.save_synth))
    print(f"{len(jobs)} jobs to run, {args.workers} workers", flush=True)
    # One free-GPU token per worker slot, dealt round-robin over `--gpus`:
    # a worker takes a token for the duration of its job. (Indexing by job
    # number instead -- the earlier scheme -- put every slow job of a
    # [slow, fast, slow, fast, ...] job list on the same GPU.)
    import queue

    slots: "queue.Queue[int]" = queue.Queue()
    for w in range(args.workers):
        slots.put(args.gpus[w % len(args.gpus)])

    import configs as cfgmod

    def work(i_job):
        i, job = i_job
        gpu = slots.get()
        jp = out / f".job_{i}.json"
        jp.write_text(json.dumps(job))
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
                   OMP_NUM_THREADS="8", TOKENIZERS_PARALLELISM="false")
        src = cfgmod.CONFIGS.get(job["config"], {}).get("src")
        if src:
            env["RTF_SRC"] = src
        t0 = time.time()
        log = out / "logs"
        log.mkdir(exist_ok=True)
        with open(log / f"{Path(job['out_json']).stem}.log", "w") as lf:
            try:
                subprocess.run([sys.executable, __file__, "child", str(jp)], env=env, stdout=lf, stderr=subprocess.STDOUT)
            finally:
                slots.put(gpu)
        jp.unlink(missing_ok=True)
        r = json.loads(Path(job["out_json"]).read_text())
        print(f"[{time.time() - t0:6.0f}s] {Path(job['out_json']).stem} {'FAILED' if 'error' in r else 'ok'}", flush=True)

    with ThreadPoolExecutor(args.workers) as ex:
        list(ex.map(work, enumerate(jobs)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--name", required=True)
    r.add_argument("--configs", nargs="+", required=True)
    r.add_argument("--datasets", nargs="+", default=DEV_DATASETS)
    r.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    r.add_argument("--workers", type=int, default=4)
    r.add_argument("--gpus", nargs="+", type=int, default=[0, 1])
    r.add_argument("--save-synth", action="store_true")
    c = sub.add_parser("child")
    c.add_argument("job")
    a = ap.parse_args()
    run_matrix(a) if a.cmd == "run" else child_main(a.job)
