"""M5: model size x checkpoint-selection rule, in the REAL default regime.

One sensitivity-regime fit per (dataset, arm, seed) leaves four checkpoints on disk
(mean_best = what `load_from_best_mean_sensitivity=True` loads; best_disc = latest
acceptable; not_best; last_epoch). Each is loaded into the same fitted pipeline and
scored on the same synthetic-sample protocol as bench.py, so model size and the
checkpoint rule can be compared on identical trajectories.

    python research/m5.py run --name m5 --arms default wk --datasets diabetes insurance \
        --seeds 0 1 2 --workers 6
Legacy settings spelled out (unk_dropout 0, oov random, top_k 50), teacher-forced target,
like M1-M3. Arms: default = GPT2Config(n_layer=6); wk = 128d/4h/3L, lr 3e-4, 5% warmup.
"""
import argparse, json, os, queue, shutil, subprocess, sys, time, traceback, warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[1]; EXP = REPO / "research"
for p in (REPO / "src", REPO / "cusum_validation", EXP): sys.path.insert(0, str(p))
warnings.filterwarnings("ignore")

ARMS = {"default": dict(gpt2={}, train_args={}),
        "wk": dict(gpt2={"n_embd": 128, "n_head": 4, "n_layer": 3}, train_args={"learning_rate": 3e-4, "warmup_steps": 0.05})}
RULES = ("mean_best_disc_model", "best_disc_model", "not_best_disc_model", "last_epoch_model")

def run_one(job):
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel
    import bench as B, run_experiment as rx
    from realtabformer.data_utils.constants import TabularArtefact
    from realtabformer.realtabformer import REaLTabFormer
    from realtabformer.rtf_analyze import SyntheticDataBench
    seed, arm = job["seed"], ARMS[job["arm"]]
    df, target, is_cls, pos = B.load_dataset(job["dataset"])
    bn = SyntheticDataBench(data=df, target_col=target, categorical=is_cls, target_pos_val=pos, test_size=0.2, random_state=seed)
    train, test = bn.train_df.reset_index(drop=True), bn.test_df.reset_index(drop=True); bn.train_df = train
    cols = list(train.columns); n_need = len(train) + len(test)
    torch.manual_seed(seed); np.random.seed(seed)
    g = GPT2Config(n_layer=6)
    for k, v in arm["gpt2"].items(): setattr(g, k, v)
    tmp = Path(job["tmp"]); tmp.mkdir(parents=True, exist_ok=True)
    m = REaLTabFormer(model_type="tabular", tabular_config=g, epochs=300, batch_size=8, random_state=seed,
                      checkpoints_dir=str(tmp / "ckpt"), unk_dropout=0.0, oov_strategy="random")
    m.training_args_kwargs.update(arm["train_args"])
    spe = max(1, len(train) // (8 * 4)); t0 = time.time()
    tr = m.fit(train, device="cuda", n_critic=5, n_critic_stop=2, num_bootstrap=rx.default_sensitivity_num_bootstrap(len(train)),
               sensitivity_cache_dir=str(EXP / "cache"), sensitivity_bootstrap_n_jobs=8,
               load_from_best_mean_sensitivity=True, gen_kwargs={"gen_batch": 512}, target_col=target)
    res = dict(job=job, fit_s=time.time() - t0, stop_epoch=tr.state.global_step / spe, checkpoints={})
    ck = Path(m.checkpoints_dir)
    for name in RULES:
        d = ck / getattr(TabularArtefact, name)
        if not ((d / "model.safetensors").exists() or (d / "pytorch_model.bin").exists()): res["checkpoints"][name] = None; continue
        m.model = GPT2LMHeadModel.from_pretrained(d.as_posix()).to("cuda")
        ep = json.loads((d / "trainer_state.json").read_text())["global_step"] / spe
        torch.manual_seed(seed)
        s = m.sample(n_samples=int(n_need * 1.15) + 50, device="cuda", gen_batch=1024, top_k=50)
        s = s[cols].dropna().reset_index(drop=True).iloc[:n_need]; s_tr = s.sample(n=len(train), random_state=seed).reset_index(drop=True)
        r = dict(epoch=ep); r.update(B.fidelity_metrics(train, s_tr)); r.update(B.utility_metrics(train, s_tr, test, target, is_cls, pos))
        r["disc_auc"] = B.discriminator_auc(train, s_tr, target, seed); r["dcr_share"] = B.dcr_share(bn, s, seed)
        res["checkpoints"][name] = r
    return res

def child_main(path):
    job = json.loads(Path(path).read_text())
    try: res = run_one(job)
    except Exception: res = dict(job=job, error=traceback.format_exc())
    finally: shutil.rmtree(job["tmp"], ignore_errors=True)
    Path(job["out_json"]).write_text(json.dumps(res, indent=1, default=float))

def run_matrix(a):
    out = EXP / "results" / a.name; out.mkdir(parents=True, exist_ok=True); jobs = []
    for d in a.datasets:
        for s in a.seeds:
            for arm in a.arms:
                oj = out / f"{d}__{arm}__s{s}.json"
                if oj.exists() and "error" not in json.loads(oj.read_text()): continue
                jobs.append(dict(dataset=d, arm=arm, seed=s, out_json=str(oj), tmp=str(EXP / "tmp" / f"{a.name}_{d}_{arm}_{s}")))
    print(f"{len(jobs)} jobs, {a.workers} workers", flush=True)
    slots = queue.Queue()
    for w in range(a.workers): slots.put(a.gpus[w % len(a.gpus)])
    def work(ij):
        i, job = ij; gpu = slots.get(); jp = out / f".job_{i}.json"; jp.write_text(json.dumps(job))
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="8", TOKENIZERS_PARALLELISM="false")
        (out / "logs").mkdir(exist_ok=True); t0 = time.time()
        with open(out / "logs" / f"{Path(job['out_json']).stem}.log", "w") as lf:
            try: subprocess.run([sys.executable, __file__, "child", str(jp)], env=env, stdout=lf, stderr=subprocess.STDOUT)
            finally: slots.put(gpu)
        jp.unlink(missing_ok=True); r = json.loads(Path(job["out_json"]).read_text())
        print(f"[{time.time()-t0:6.0f}s] {Path(job['out_json']).stem} {'FAILED' if 'error' in r else 'ok'}", flush=True)
    with ThreadPoolExecutor(a.workers) as ex: list(ex.map(work, enumerate(jobs)))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("--name", required=True); r.add_argument("--arms", nargs="+", default=["default", "wk"])
    r.add_argument("--datasets", nargs="+", default=["diabetes", "insurance", "abalone", "adult5k", "wilt", "churn2"])
    r.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2]); r.add_argument("--workers", type=int, default=6)
    r.add_argument("--gpus", nargs="+", type=int, default=[0, 1])
    c = sub.add_parser("child"); c.add_argument("job"); a = ap.parse_args()
    run_matrix(a) if a.cmd == "run" else child_main(a.job)
