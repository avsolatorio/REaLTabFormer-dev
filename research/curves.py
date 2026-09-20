"""Learning-curve protocol for Program 2 (H11-H14, H16, H17).

Fixed-epoch training with NO stopping rule and dense checkpoints: every `--every`
epochs the model is sampled and scored (fidelity, discriminator, TSTR, memorisation
`dcr_share`, held-out NLL under the column-constrained model). A variant's whole
curve -- how fast it learns, its ceiling, and what memorisation does along the way --
is compared with the reference arm on identical (dataset, seed) splits.

Why not the sensitivity-stopping matrices: (1) the library's default sensitivity
path silently drops `compute_loss_func` (found while building this), so custom
losses cannot be tested there; (2) stopping-rule noise would sit on top of every
comparison; (3) the curves make efficiency ("epochs to reach quality X") directly
measurable. Arms may change: the GPT2 config, TrainingArguments, REaLTabFormer(...)
kwargs, the column order, the loss (constrained / label-smoothed), and may evaluate
EMA copies of the weights on the very same trajectory (a paired comparison).

    python research/curves.py run --name c1 --arms wk wk_cl --datasets diabetes \
        --seeds 0 1 2 --workers 4
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

TINY = {"n_embd": 128, "n_head": 4, "n_layer": 3}
LEGACY = {"unk_dropout": 0.0, "oov_strategy": "random"}
WK_ARGS = {"learning_rate": 3e-4, "warmup_steps": 0.05}


def _arm(**kw):
    a = dict(gpt2=dict(TINY), train_args=dict(WK_ARGS), init=dict(LEGACY), order="orig",
             loss=None, eps=0.0, ema=(), batch_size=8)
    a.update(kw)
    return a


ARMS = {
    "wk": _arm(ema=(1, 4)),                                   # reference; also records EMA copies
    # H17 constraint-aware loss / H12 smoothing
    "wk_cl": _arm(loss="constrained"),
    "wk_cl_ls05": _arm(loss="constrained", eps=0.05),
    "wk_ls05": _arm(train_args={**WK_ARGS, "label_smoothing_factor": 0.05}),
    # H11 column order
    "wk_rev": _arm(order="reverse"),
    "wk_hub": _arm(order="hub_first"),
    "wk_ent": _arm(order="entropy_asc"),
    "wk_rand": _arm(order="random"),
    # H14 numeric representation (qenc = quantile encoding)
    "wk_qenc": _arm(init={**LEGACY, "numeric_quantile_encoding": True}),
    "wk_qenc_p3": _arm(init={**LEGACY, "numeric_quantile_encoding": True, "numeric_precision": 3}),
    "wk_qenc_np2": _arm(init={**LEGACY, "numeric_quantile_encoding": True, "numeric_nparts": 2}),
    "wk_nct20": _arm(init={**LEGACY, "numeric_categorical_threshold": 20}),
    # H12 regularisation
    "wk_drop0": _arm(gpt2={**TINY, "resid_pdrop": 0.0, "embd_pdrop": 0.0, "attn_pdrop": 0.0}),
    "wk_drop2": _arm(gpt2={**TINY, "resid_pdrop": 0.2, "embd_pdrop": 0.2, "attn_pdrop": 0.2}),
    "wk_wd": _arm(train_args={**WK_ARGS, "weight_decay": 0.05}),
    # H13 efficiency
    "wk_bs32": _arm(batch_size=32, train_args={**WK_ARGS, "gradient_accumulation_steps": 1}),
    # the big default model, for reference against the small-model story
    "big": dict(gpt2={}, train_args={}, init=dict(LEGACY), order="orig", loss=None, eps=0.0, ema=(), batch_size=8),
}


# ---------------------------------------------------------------- helpers ---
def order_columns(train: pd.DataFrame, mode: str, seed: int) -> list:
    """A permutation of the columns; `hub_first` = decreasing total mutual
    information with the other columns, `entropy_asc` = increasing marginal entropy
    (numerics quantile-binned into 10 bins first)."""
    import bench as B
    from sklearn.metrics import mutual_info_score

    cols = list(train.columns)
    if mode == "orig":
        return cols
    if mode == "reverse":
        return cols[::-1]
    if mode == "random":
        return list(np.random.default_rng(seed).permutation(cols))
    d = B._discretise(train, train)
    ent = {}
    for c in cols:
        p = d[c].value_counts(normalize=True).to_numpy()
        ent[c] = float(-(p * np.log(p)).sum())
    if mode == "entropy_asc":
        return sorted(cols, key=lambda c: ent[c])
    mi = {c: sum(mutual_info_score(d[c], d[o]) for o in cols if o != c) for c in cols}
    if mode == "hub_first":
        return sorted(cols, key=lambda c: -mi[c])
    raise ValueError(mode)


def make_loss(rtf, eps):
    """Cross-entropy under the SAME per-column token mask sampling uses (each
    position is normalised over only the tokens valid for its column), optionally
    label-smoothed over just those valid tokens. The model no longer has to spend
    capacity learning which vocabulary tokens are legal where."""
    import torch

    from realtabformer.data_utils import SpecialTokens
    from realtabformer.rtf_sampler import ColumnMaskLogitsProcessor

    state = {}

    def loss_fn(outputs, labels, num_items_in_batch=None):
        logits = outputs.logits[:, :-1, :].float()
        tgt = labels[:, 1:]
        if "mask" not in state:
            eos = rtf.vocab["token2id"][SpecialTokens.EOS]
            state["mask"] = ColumnMaskLogitsProcessor(
                rtf.col_idx_ids, eos, logits.shape[-1], rtf.tabular_max_length, logits.device
            ).mask
        m = state["mask"][: logits.shape[1]].unsqueeze(0)
        logp = torch.log_softmax(logits.masked_fill(~m, float("-inf")), dim=-1)
        valid = tgt != -100
        nll = -logp.gather(-1, tgt.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        nll = torch.nan_to_num(nll, posinf=40.0).clamp(max=40.0)
        if eps > 0:
            smooth = -(logp.masked_fill(~m, 0.0)).sum(-1) / m.sum(-1).clamp(min=1)
            nll = (1 - eps) * nll + eps * smooth
        return (nll * valid).sum() / valid.sum().clamp(min=1)

    return loss_fn


class EMAWeights:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = [p.detach().clone() for p in model.parameters()]

    def update(self, model):
        import torch

        with torch.no_grad():
            ps = [p.detach() for p in model.parameters()]
            torch._foreach_mul_(self.shadow, self.decay)
            torch._foreach_add_(self.shadow, ps, alpha=1 - self.decay)

    def swap_in(self, model):
        import torch

        with torch.no_grad():
            self.backup = [p.detach().clone() for p in model.parameters()]
            torch._foreach_copy_([p.data for p in model.parameters()], self.shadow)

    def swap_out(self, model):
        import torch

        with torch.no_grad():
            torch._foreach_copy_([p.data for p in model.parameters()], self.backup)
        self.backup = None


# -------------------------------------------------------------- the study ---
def run_one(job):
    import torch
    from transformers import GPT2Config, TrainerCallback

    import bench as B
    import signals as S
    from realtabformer.data_utils import SpecialTokens
    from realtabformer.realtabformer import REaLTabFormer
    from realtabformer.rtf_analyze import SyntheticDataBench
    from realtabformer.rtf_sampler import ColumnMaskLogitsProcessor, TabularSampler

    seed, device, arm = job["seed"], "cuda", ARMS[job["arm"]]
    df, target, is_cls, pos = B.load_dataset(job["dataset"])
    bench_obj = SyntheticDataBench(
        data=df, target_col=target, categorical=is_cls, target_pos_val=pos,
        test_size=0.2, random_state=seed,
    )
    train = bench_obj.train_df.reset_index(drop=True)
    test = bench_obj.test_df.reset_index(drop=True)
    bench_obj.train_df = train
    cols = list(train.columns)
    order = order_columns(train, arm["order"], seed)
    train_fit, test_fit = train[order], test[order]

    torch.manual_seed(seed)
    np.random.seed(seed)
    gpt2 = GPT2Config(n_layer=6)
    for k, v in arm["gpt2"].items():
        setattr(gpt2, k, v)
    tmp = Path(job["tmp"])
    tmp.mkdir(parents=True, exist_ok=True)
    rtf = REaLTabFormer(
        model_type="tabular", tabular_config=gpt2, epochs=job["epochs"],
        batch_size=arm["batch_size"], random_state=seed, checkpoints_dir=str(tmp / "ckpt"),
        **arm["init"],
    )
    rtf.training_args_kwargs.update(
        eval_strategy="no", save_strategy="no", load_best_model_at_end=False,
        logging_steps=10**6, report_to=[], **arm["train_args"],
    )
    rtf.target_col = None
    rtf.trainer_kwargs = {}  # normally set by fit(), which this protocol bypasses on purpose
    loss_fn = make_loss(rtf, arm["eps"]) if arm["loss"] == "constrained" else None

    t0 = time.time()
    trainer = rtf._fit_tabular(train_fit, device=device, compute_loss_func=loss_fn)

    t2i = rtf.vocab["token2id"]
    pad, eos, bos = t2i[SpecialTokens.PAD], t2i[SpecialTokens.EOS], t2i[SpecialTokens.BOS]
    L = rtf.tabular_max_length
    dev = torch.device(device)
    mask = ColumnMaskLogitsProcessor(rtf.col_idx_ids, eos, len(rtf.vocab["id2token"]), L, dev).mask
    te_ids = S.encode_rows(rtf, test_fit)[:2000]
    sampler = TabularSampler.sampler_from_model(rtf, device=device)
    n_gen = max(1024, len(test) + 64)
    every = job["every"]
    rows = []

    def evaluate(model, epoch, step, weights):
        model.eval()
        torch.manual_seed(seed * 100003 + epoch)
        toks = sampler._generate(
            device=dev, as_numpy=False, constrain_tokens_gen=True,
            inputs=torch.tensor([[bos]], device=dev), do_sample=True, max_length=L,
            num_return_sequences=n_gen, bos_token_id=bos, pad_token_id=pad,
            eos_token_id=eos, suppress_tokens=None, forced_decoder_ids=None,
        )
        r = dict(epoch=epoch, step=int(step), weights=weights,
                 nll_test=float(S.row_nll(model, te_ids, mask, pad).mean()))
        try:
            synth = sampler._processes_sample(
                sample_outputs=toks.cpu().numpy(), vocab=rtf.vocab, validator=None, column_order=None,
            )[cols].dropna().reset_index(drop=True)
            r["n_valid"] = int(len(synth))
            s_tr = synth.iloc[: len(train)]
            r.update(B.fidelity_metrics(train, s_tr))
            r["disc_auc"] = B.discriminator_auc(train, s_tr, target, seed)
            r.update(B.utility_metrics(train, s_tr, test, target, is_cls, pos))
            r["dcr_share"] = B.dcr_share(bench_obj, synth, seed) if len(synth) >= len(test) else float("nan")
        except Exception as e:
            r["decode_error"] = f"{type(e).__name__}: {str(e)[:80]}"
        model.train()
        return r

    class CB(TrainerCallback):
        def on_train_begin(self, args, state, control, model=None, **kw):
            spe = max(1, state.max_steps // max(1, int(state.num_train_epochs)))
            self.emas = {h: EMAWeights(model, float(np.exp(-1.0 / (h * spe)))) for h in arm["ema"]}

        def on_step_end(self, args, state, control, model=None, **kw):
            for e in self.emas.values():
                e.update(model)

        def on_epoch_end(self, args, state, control, model=None, **kw):
            ep = int(round(state.epoch))
            if ep != 1 and ep % every:
                return
            rows.append(evaluate(model, ep, state.global_step, "raw"))
            for h, e in self.emas.items():
                e.swap_in(model)
                try:
                    rows.append(evaluate(model, ep, state.global_step, f"ema{h}"))
                finally:
                    e.swap_out(model)

    trainer.add_callback(CB())
    trainer.train()
    n_params = sum(p.numel() for p in rtf.model.parameters())
    return dict(job=job, n_train=len(train), n_test=len(test), n_params=int(n_params),
                fit_s=time.time() - t0, order=order, rows=rows)


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
                jobs.append(dict(dataset=d, arm=arm, seed=s, epochs=a.epochs, every=a.every, out_json=str(oj),
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
    r.add_argument("--arms", nargs="+", default=["wk"])
    r.add_argument("--datasets", nargs="+", default=["diabetes", "insurance", "abalone"])
    r.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    r.add_argument("--workers", type=int, default=4)
    r.add_argument("--gpus", nargs="+", type=int, default=[0, 1])
    r.add_argument("--epochs", type=int, default=100)
    r.add_argument("--every", type=int, default=10)
    c = sub.add_parser("child")
    c.add_argument("job")
    a = ap.parse_args()
    run_matrix(a) if a.cmd == "run" else child_main(a.job)
