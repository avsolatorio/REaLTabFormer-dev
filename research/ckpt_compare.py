"""Compare the checkpoints one default-recipe fit leaves behind, on the SAME trajectory:
  mean_closest  = what `load_from_best_mean_sensitivity=True` loads (the recipe's recommendation)
  best_disc     = latest checkpoint whose critic sensitivity was still under the threshold
  not_best      = closest-to-threshold checkpoint after it was breached
  last_epoch    = the weights when training stopped
    python research/ckpt_compare.py diabetes 0 /tmp/le_diabetes_0 out.json
"""
import json, sys, warnings
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "src", REPO / "cusum_validation", REPO / "research"): sys.path.insert(0, str(p))
warnings.filterwarnings("ignore")
import torch
from transformers import GPT2LMHeadModel
import bench as B
from realtabformer.realtabformer import REaLTabFormer
from realtabformer.data_utils.constants import TabularArtefact
from realtabformer.rtf_analyze import SyntheticDataBench

ds, seed, ck, out = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
df, target, is_cls, pos = B.load_dataset(ds)
bn = SyntheticDataBench(data=df, target_col=target, categorical=is_cls, target_pos_val=pos, test_size=0.2, random_state=seed)
train, test = bn.train_df.reset_index(drop=True), bn.test_df.reset_index(drop=True); bn.train_df = train
rtf = REaLTabFormer.load_from_dir(ck / "sensitivity_best")
spe = max(1, len(train) // (8 * 4)); cols = list(train.columns); n_need = len(train) + len(test)
res = dict(dataset=ds, seed=seed, checkpoints={})
for name in ("mean_best_disc_model", "best_disc_model", "not_best_disc_model", "last_epoch_model"):
    art = getattr(TabularArtefact, name, None)
    d = ck / art if art else None
    if d is None or not d.exists() or not ((d / "model.safetensors").exists() or (d / "pytorch_model.bin").exists()):
        res["checkpoints"][name] = None; continue
    rtf.model = GPT2LMHeadModel.from_pretrained(d.as_posix()).to("cuda")
    ep = json.loads((d / "trainer_state.json").read_text())["global_step"] / spe
    torch.manual_seed(seed)
    synth = rtf.sample(n_samples=int(n_need * 1.15) + 50, device="cuda", gen_batch=1024, top_k=50)
    synth = synth[cols].dropna().reset_index(drop=True).iloc[:n_need]
    s_tr = synth.sample(n=len(train), random_state=seed).reset_index(drop=True)
    r = dict(epoch=ep); r.update(B.fidelity_metrics(train, s_tr)); r.update(B.utility_metrics(train, s_tr, test, target, is_cls, pos))
    r["disc_auc"] = B.discriminator_auc(train, s_tr, target, seed); r["dcr_share"] = B.dcr_share(bn, synth, seed)
    res["checkpoints"][name] = r
Path(out).write_text(json.dumps(res, indent=1, default=float))
for k, v in res["checkpoints"].items():
    print(f"{ds} s{seed} {k:26s}", "MISSING" if v is None else f"epoch {v['epoch']:5.1f} marg {v['marg_mean']:.3f} disc_auc {v['disc_auc']:.3f} tstr {v['tstr']:.3f} dcr_share {v['dcr_share']:.3f}")
