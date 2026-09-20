"""Which epoch does the default recipe's `load_from_best_mean_sensitivity=True` actually
load, and how good is that model? (M1 recorded only where training STOPPED.)
    python research/loaded_epoch.py diabetes 0 out.json
Legacy settings spelled out (unk_dropout 0, oov random, top_k 50) so it means what M1 meant."""
import json, sys, time, warnings
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "src", REPO / "cusum_validation", REPO / "research"): sys.path.insert(0, str(p))
warnings.filterwarnings("ignore")
import torch
from transformers import GPT2Config
import bench as B, run_experiment as rx
from realtabformer.realtabformer import REaLTabFormer
from realtabformer.rtf_analyze import SyntheticDataBench

ds, seed, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
df, target, is_cls, pos = B.load_dataset(ds)
bn = SyntheticDataBench(data=df, target_col=target, categorical=is_cls, target_pos_val=pos, test_size=0.2, random_state=seed)
train, test = bn.train_df.reset_index(drop=True), bn.test_df.reset_index(drop=True); bn.train_df = train
torch.manual_seed(seed); np.random.seed(seed)
m = REaLTabFormer(model_type="tabular", tabular_config=GPT2Config(n_layer=6), epochs=300, batch_size=8, random_state=seed,
                  checkpoints_dir=f"/tmp/le_{ds}_{seed}", unk_dropout=0.0, oov_strategy="random")
spe = max(1, len(train) // (8 * 4))
tr = m.fit(train, device="cuda", n_critic=5, n_critic_stop=2, num_bootstrap=rx.default_sensitivity_num_bootstrap(len(train)),
           sensitivity_cache_dir=str(REPO / "research" / "cache"), sensitivity_bootstrap_n_jobs=8,
           load_from_best_mean_sensitivity=True, gen_kwargs={"gen_batch": 512}, target_col=target)
res = dict(dataset=ds, seed=seed, stop_epoch=tr.state.global_step / spe,
           loaded_epoch=m.trainer_state["global_step"] / spe, loaded_step=m.trainer_state["global_step"], steps_per_epoch=spe)
n_need = len(train) + len(test)
synth = m.sample(n_samples=int(n_need * 1.15) + 50, device="cuda", gen_batch=1024, top_k=50)
cols = list(train.columns); synth = synth[cols].dropna().reset_index(drop=True).iloc[:n_need]
s_tr = synth.sample(n=len(train), random_state=seed).reset_index(drop=True)
res.update(B.fidelity_metrics(train, s_tr)); res["disc_auc"] = B.discriminator_auc(train, s_tr, target, seed)
res["dcr_share"] = B.dcr_share(bn, synth, seed)
Path(out).write_text(json.dumps(res, indent=1, default=float)); print(res)
