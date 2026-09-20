"""Signal-validation study for label-free overfitting detection (hypothesis H15).

Trains a model with NO stopping rule and, every `--every` epochs, records:

  label-free signals (use only the training rows and the model's own samples)
    nll_train, nll_samp          mean per-row NLL under the (column-constrained)
                                 model of training rows / of model samples
    srlg_mean                    nll_samp - nll_train
    srlg_ks                      KS distance between the two per-row NLL dists
    srlg_tail                    5th-percentile NLL of samples minus of train rows
  ground truth (used ONLY to judge the signals, never by them)
    nll_test, gap_true           held-out NLL; nll_test - nll_train
    dcr_share                    memorisation: share of synthetic rows nearest to
                                 a train row vs an equal-size held-out set (0.5 = none)
    marg_mean, assoc_diff, disc_auc   fidelity of the decoded samples

All NLLs are taken under the model restricted to the tokens valid for each
column (the same constraint sampling uses), so E_q[NLL_q] is the entropy of
what the model can actually generate.

    python research/signals.py run --name s1 --arms default wk \
        --datasets diabetes insurance --seeds 0 1 --workers 4
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
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "cusum_validation"))
sys.path.insert(0, str(EXP))
warnings.filterwarnings("ignore")

NLL_CLAMP = 40.0  # nats per token; caps a token the constrained model deems impossible

ARMS = {
    "default": dict(gpt2={}, train_args={}),
    "wk": dict(
        gpt2={"n_embd": 128, "n_head": 4, "n_layer": 3},
        train_args={"learning_rate": 3e-4, "warmup_steps": 0.05},
    ),
}


def row_nll(model, ids, mask, pad_id, batch=256):
    """Per-row NLL (nats, summed over tokens) of token sequences `ids` [N, L]
    (position 0 = BOS) under `model` restricted, at each step, to that step's
    valid tokens (`mask` [steps, vocab]); positions equal to `pad_id` are
    ignored (the padding after a generated sequence's EOS)."""
    import torch

    out = []
    with torch.no_grad():
        for i in range(0, len(ids), batch):
            x = ids[i : i + batch].to(model.device)
            logits = model(input_ids=x).logits[:, :-1, :].float()
            m = mask[: logits.shape[1]].unsqueeze(0)
            logp = torch.log_softmax(logits.masked_fill(~m, float("-inf")), dim=-1)
            tok = x[:, 1:]
            nll = -logp.gather(-1, tok.unsqueeze(-1)).squeeze(-1)
            nll = torch.nan_to_num(nll, posinf=NLL_CLAMP).clamp(max=NLL_CLAMP)
            nll = nll.masked_fill(tok == pad_id, 0.0)
            out.append(nll.sum(dim=1).cpu())
    return torch.cat(out).numpy()


def encode_rows(rtf, df):
    """Token ids for raw rows `df` through the model's own fitted pipeline."""
    import torch

    from realtabformer import data_utils as du

    processed, _, _ = du.process_data(
        df,
        numeric_max_len=rtf.numeric_max_len,
        numeric_precision=rtf.numeric_precision,
        numeric_nparts=rtf.numeric_nparts,
        target_col=rtf.target_col,
        col_transform_data=rtf.col_transform_data,
        numeric_categorical_threshold=rtf.numeric_categorical_threshold,
        numeric_quantile_encoding=rtf.numeric_quantile_encoding,
        numeric_quantile_bins=rtf.numeric_quantile_bins,
    )
    processed = processed[rtf.processed_columns]
    ds = du.make_dataset(processed, rtf.vocab, mask_rate=0, affix_eos=True, seed=0)
    return torch.tensor(ds["input_ids"])


def make_callback(rtf, bench_obj, train_raw, test_raw, cols, target, device, every, n_gen, seed):
    import torch
    from scipy.stats import ks_2samp
    from transformers import TrainerCallback

    import bench as B
    from realtabformer.data_utils import SpecialTokens
    from realtabformer.rtf_sampler import ColumnMaskLogitsProcessor, TabularSampler

    t2i = rtf.vocab["token2id"]
    pad, eos, bos = t2i[SpecialTokens.PAD], t2i[SpecialTokens.EOS], t2i[SpecialTokens.BOS]
    L = rtf.tabular_max_length
    dev = torch.device(device)
    mask = ColumnMaskLogitsProcessor(
        rtf.col_idx_ids, eos, len(rtf.vocab["id2token"]), L, dev
    ).mask

    tr = rtf.dataset["train_dataset"]["input_ids"]
    tr = torch.stack(list(tr)) if isinstance(tr, list) else tr
    rng = np.random.default_rng(seed)
    tr = tr[torch.as_tensor(rng.permutation(len(tr))[:2000])]
    te = encode_rows(rtf, test_raw)[:2000]
    sampler = TabularSampler.sampler_from_model(rtf, device=device)
    n_gen = max(n_gen, len(test_raw) + 64)

    class Signals(TrainerCallback):
        rows: list = []

        def measure(self, model, epoch, step):
            model.eval()
            torch.manual_seed(seed * 100003 + epoch)
            toks = sampler._generate(
                device=dev, as_numpy=False, constrain_tokens_gen=True,
                inputs=torch.tensor([[bos]], device=dev), do_sample=True, max_length=L,
                num_return_sequences=n_gen, bos_token_id=bos, pad_token_id=pad,
                eos_token_id=eos, suppress_tokens=None, forced_decoder_ids=None,
            )
            nll_tr = row_nll(model, tr, mask, pad)
            nll_te = row_nll(model, te, mask, pad)
            nll_sa = row_nll(model, toks, mask, pad)
            r = dict(
                epoch=epoch, step=int(step),
                nll_train=float(nll_tr.mean()), nll_test=float(nll_te.mean()),
                nll_samp=float(nll_sa.mean()),
                srlg_mean=float(nll_sa.mean() - nll_tr.mean()),
                srlg_ks=float(ks_2samp(nll_tr, nll_sa).statistic),
                srlg_tail=float(np.quantile(nll_sa, 0.05) - np.quantile(nll_tr, 0.05)),
                gap_true=float(nll_te.mean() - nll_tr.mean()),
            )
            try:
                synth = sampler._processes_sample(
                    sample_outputs=toks.cpu().numpy(), vocab=rtf.vocab,
                    validator=None, column_order=None,
                )[cols].dropna().reset_index(drop=True)
                r["n_valid"] = int(len(synth))
                s_tr = synth.iloc[: len(train_raw)]
                fm = B.fidelity_metrics(train_raw, s_tr)
                r.update(marg_mean=fm["marg_mean"], assoc_diff=fm["assoc_diff"])
                r["disc_auc"] = B.discriminator_auc(train_raw, s_tr, target, seed)
                r["dcr_share"] = (
                    B.dcr_share(bench_obj, synth, seed) if len(synth) >= len(test_raw) else float("nan")
                )
            except Exception as e:  # keep the likelihood signals even if decoding fails
                r["decode_error"] = f"{type(e).__name__}: {str(e)[:80]}"
            model.train()
            return r

        def on_epoch_end(self, args, state, control, model=None, **kw):
            ep = int(round(state.epoch))
            if ep == 1 or ep % every == 0:
                self.rows.append(self.measure(model, ep, state.global_step))

    cb = Signals()
    cb.rows = []
    return cb


def run_one(job):
    import torch
    from transformers import GPT2Config

    import bench as B
    from realtabformer.realtabformer import REaLTabFormer
    from realtabformer.rtf_analyze import SyntheticDataBench

    seed, device = job["seed"], "cuda"
    df, target, is_cls, pos = B.load_dataset(job["dataset"])
    bench_obj = SyntheticDataBench(
        data=df, target_col=target, categorical=is_cls, target_pos_val=pos,
        test_size=0.2, random_state=seed,
    )
    train = bench_obj.train_df.reset_index(drop=True)
    test = bench_obj.test_df.reset_index(drop=True)
    bench_obj.train_df = train
    cols = list(train.columns)

    torch.manual_seed(seed)
    np.random.seed(seed)
    arm = ARMS[job["arm"]]
    gpt2 = GPT2Config(n_layer=6)
    for k, v in arm["gpt2"].items():
        setattr(gpt2, k, v)
    tmp = Path(job["tmp"])
    tmp.mkdir(parents=True, exist_ok=True)
    rtf = REaLTabFormer(
        model_type="tabular", tabular_config=gpt2, epochs=job["epochs"], batch_size=8,
        random_state=seed, checkpoints_dir=str(tmp / "ckpt"),
        unk_dropout=0.0, oov_strategy="random",
    )
    rtf.training_args_kwargs.update(
        eval_strategy="no", save_strategy="no", load_best_model_at_end=False,
        logging_steps=100000, report_to=[], **arm["train_args"],
    )
    rtf.target_col = None
    rtf.trainer_kwargs = {}  # normally set by fit(), which this study bypasses on purpose
    t0 = time.time()
    trainer = rtf._fit_tabular(train, device=device)
    cb = make_callback(rtf, bench_obj, train, test, cols, target, device, job["every"], 1024, seed)
    trainer.add_callback(cb)
    trainer.train()
    return dict(job=job, n_train=len(train), n_test=len(test), fit_s=time.time() - t0, rows=cb.rows)


def child_main(path):
    job = json.loads(Path(path).read_text())
    try:
        res = run_one(job)
    except Exception:
        res = dict(job=job, error=traceback.format_exc())
    finally:
        shutil.rmtree(job["tmp"], ignore_errors=True)
    Path(job["out_json"]).write_text(json.dumps(res, indent=1, default=float))


def run_matrix(a):
    import queue

    out = EXP / "results" / a.name
    out.mkdir(parents=True, exist_ok=True)
    jobs = []
    for d in a.datasets:
        for s in a.seeds:
            for arm in a.arms:
                oj = out / f"{d}__{arm}__s{s}.json"
                if oj.exists() and "error" not in json.loads(oj.read_text()):
                    continue
                ep = a.epochs if a.epochs else (120 if arm == "default" else 300)
                ev = a.every if a.every else (5 if arm == "default" else 10)
                jobs.append(dict(dataset=d, arm=arm, seed=s, epochs=ep, every=ev, out_json=str(oj),
                                 tmp=str(EXP / "tmp" / f"{a.name}_{d}_{arm}_{s}")))
    print(f"{len(jobs)} jobs, {a.workers} workers", flush=True)
    slots: "queue.Queue[int]" = queue.Queue()
    for w in range(a.workers):
        slots.put(a.gpus[w % len(a.gpus)])

    def work(ij):
        i, job = ij
        gpu = slots.get()
        jp = out / f".job_{i}.json"
        jp.write_text(json.dumps(job))
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="8", TOKENIZERS_PARALLELISM="false")
        (out / "logs").mkdir(exist_ok=True)
        t0 = time.time()
        with open(out / "logs" / f"{Path(job['out_json']).stem}.log", "w") as lf:
            try:
                subprocess.run([sys.executable, __file__, "child", str(jp)], env=env, stdout=lf, stderr=subprocess.STDOUT)
            finally:
                slots.put(gpu)
        jp.unlink(missing_ok=True)
        r = json.loads(Path(job["out_json"]).read_text())
        print(f"[{time.time() - t0:6.0f}s] {Path(job['out_json']).stem} {'FAILED' if 'error' in r else 'ok'}", flush=True)

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(work, enumerate(jobs)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--name", required=True)
    r.add_argument("--arms", nargs="+", default=["default", "wk"])
    r.add_argument("--datasets", nargs="+", default=["diabetes", "insurance", "abalone", "adult5k"])
    r.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    r.add_argument("--workers", type=int, default=4)
    r.add_argument("--gpus", nargs="+", type=int, default=[0, 1])
    r.add_argument("--epochs", type=int, default=0)
    r.add_argument("--every", type=int, default=0)
    c = sub.add_parser("child")
    c.add_argument("job")
    a = ap.parse_args()
    run_matrix(a) if a.cmd == "run" else child_main(a.job)
