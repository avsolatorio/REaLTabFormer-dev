"""Training-throughput micro-benchmark (H13). Interleaved round-robin over
configurations, fixed optimizer-step count, per-step timing with CUDA sync, medians
over repetitions. Rows/s uses the EFFECTIVE batch (batch_size x grad accumulation).

    python research/bench_train_speed.py --dataset adult5k --model wk --reps 3
"""
import argparse, statistics, sys, time, warnings
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "src", REPO / "cusum_validation", REPO / "research"): sys.path.insert(0, str(p))
warnings.filterwarnings("ignore")

CONFIGS = {  # name: (batch, accum, extra TrainingArguments)
    "bs8xga4_fp16 (current)": (8, 4, {}),
    "bs32xga1_fp16": (32, 1, {}),
    "bs32xga1_bf16": (32, 1, {"fp16": False, "bf16": True}),
    "bs32xga1_fp16_fusedAdam": (32, 1, {"optim": "adamw_torch_fused"}),
    "bs32xga1_bf16_fusedAdam": (32, 1, {"fp16": False, "bf16": True, "optim": "adamw_torch_fused"}),
    "bs32xga1_bf16_compile": (32, 1, {"fp16": False, "bf16": True, "torch_compile": True}),
    "bs64xga1_bf16 (eff. 64)": (64, 1, {"fp16": False, "bf16": True}),
}
MODELS = {"wk": {"n_embd": 128, "n_head": 4, "n_layer": 3}, "big": {}}

def one(cfg, model_name, df, steps, warm):
    import torch
    from transformers import GPT2Config, TrainerCallback
    from realtabformer.realtabformer import REaLTabFormer
    bs, ga, extra = CONFIGS[cfg]
    g = GPT2Config(n_layer=6)
    for k, v in MODELS[model_name].items(): setattr(g, k, v)
    m = REaLTabFormer(model_type="tabular", tabular_config=g, epochs=10**4, batch_size=bs, random_state=0,
                      checkpoints_dir="/tmp/bts_ckpt", unk_dropout=0.0, oov_strategy="random")
    m.training_args_kwargs.update(eval_strategy="no", save_strategy="no", load_best_model_at_end=False,
                                  logging_steps=10**6, report_to=[], gradient_accumulation_steps=ga,
                                  max_steps=steps, **extra)
    m.target_col = None; m.trainer_kwargs = {}
    tr = m._fit_tabular(df, device="cuda")
    ts = []
    class T(TrainerCallback):
        def on_step_end(self, a, s, c, **kw):
            torch.cuda.synchronize(); ts.append(time.perf_counter())
    tr.add_callback(T()); tr.train()
    d = np.diff(ts)[warm:]
    return bs * ga / statistics.median(d), statistics.median(d) * 1e3

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", default="adult5k"); ap.add_argument("--model", default="wk")
    ap.add_argument("--reps", type=int, default=3); ap.add_argument("--steps", type=int, default=60); ap.add_argument("--warm", type=int, default=15)
    ap.add_argument("--only", nargs="*"); a = ap.parse_args()
    import bench as B
    df, *_ = B.load_dataset(a.dataset); df = df.sample(frac=1, random_state=0).reset_index(drop=True)
    names = a.only or list(CONFIGS); res = {n: [] for n in names}
    for r in range(a.reps):
        for n in names:
            try: res[n].append(one(n, a.model, df, a.steps, a.warm))
            except Exception as e: res[n].append((float("nan"), float("nan"))); print(f"  {n}: {type(e).__name__}: {str(e)[:90]}", flush=True)
    base = statistics.median([x[0] for x in res[names[0]]])
    print(f"\nmodel={a.model} dataset={a.dataset} reps={a.reps}  (rows/s at the effective batch; ms per optimizer step)")
    for n in names:
        rs = [x[0] for x in res[n]]; ms = [x[1] for x in res[n]]
        print(f"  {n:34s} {statistics.median(rs):9.0f} rows/s  [{min(rs):.0f}-{max(rs):.0f}]  {statistics.median(ms):7.1f} ms/step   x{statistics.median(rs)/base:4.2f}")
if __name__ == "__main__": main()
