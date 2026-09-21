"""Named experiment configs for bench.py.

Each config is a dict of overrides on the tool's own defaults:
  gpt2          -> attributes set on the GPT2Config (default: n_layer=6)
  init          -> extra REaLTabFormer(...) constructor kwargs
  train_args    -> merged into model.training_args_kwargs (HF TrainingArguments)
  fit           -> extra model.fit(...) kwargs
  epochs / batch_size / teacher_force
  sample_variants -> {name: generate kwargs}; every variant is scored on
                     the SAME trained model.
"""

_SAMPLING = {
    "default": {},  # library default; was HF top_k=50 before 2026-09-20, now top_k=0
    "topk50": {"top_k": 50},  # HF's old implicit default, for comparison with earlier results
    "topk0": {"top_k": 0},
    "topk0_t09": {"top_k": 0, "temperature": 0.9},
    "topk0_t11": {"top_k": 0, "temperature": 1.1},
    "topp95": {"top_k": 0, "top_p": 0.95},
}

CONFIGS = {
    # Tool defaults. Scored under every sampling variant, so `base/default`
    # is the reference arm and `base/topk0` etc. are the H1/H2 tests on the
    # very same trained model (a paired, zero-training-noise comparison).
    "base": {"sample_variants": _SAMPLING},
    "qenc": {"init": {"numeric_quantile_encoding": True}, "sample_variants": {"default": {}, "topk0": {"top_k": 0}}},

    # ---- M2a: model size (H3), learning rate + warmup (H4), grad accum (H7).
    # NOTE `warmup_steps=0.05` (float => ratio) not `warmup_ratio`: the
    # installed transformers (5.16.1) rejects `warmup_ratio`, and the
    # library's _build_training_args would silently drop it.
    "small": {"gpt2": {"n_embd": 256, "n_head": 8, "n_layer": 4}},
    "tiny": {"gpt2": {"n_embd": 128, "n_head": 4, "n_layer": 3}},
    "lr3e4": {"train_args": {"learning_rate": 3e-4, "warmup_steps": 0.05}},
    "lr1e4": {"train_args": {"learning_rate": 1e-4, "warmup_steps": 0.05}},
    "ga1": {"train_args": {"gradient_accumulation_steps": 1}},

    # ---- M3: is the M2 size effect about size or about training length?
    # `tiny` (128d/4h/3L) almost always ran to the 300-epoch ceiling in M2, so
    # size and epochs were confounded. Cap it at base's ~30 epochs, extend it,
    # go smaller, and let it use a higher learning rate.
    # NOTE on meaning after 2026-09-20: the library defaults changed
    # (unk_dropout 0.03 + oov_strategy "unk", TabularSampler top_k=0), so `{}`
    # no longer reproduces the earlier matrices. `b0` keeps its historical
    # meaning explicitly; a sampling variant of `{}` now means top_k=0 (use
    # "topk50" for HF's old implicit default).
    "b0": {"init": {"unk_dropout": 0.0, "oov_strategy": "random", "ema_horizon": 0.0}},
    "tiny_e30": {"epochs": 30, "gpt2": {"n_embd": 128, "n_head": 4, "n_layer": 3}},
    "tiny_e600": {"epochs": 600, "gpt2": {"n_embd": 128, "n_head": 4, "n_layer": 3}},
    "micro_e600": {"epochs": 600, "gpt2": {"n_embd": 64, "n_head": 2, "n_layer": 2}},
    "tiny_lr3e4": {"gpt2": {"n_embd": 128, "n_head": 4, "n_layer": 3},
                   "train_args": {"learning_rate": 3e-4, "warmup_steps": 0.05}},

    # ---- OOV cost check (H8): does input-side UNK dropout hurt ordinary
    # generation? (run before those became the defaults, against b0)
    "unkd03": {"init": {"unk_dropout": 0.03, "oov_strategy": "unk"}},
    "unkd10": {"init": {"unk_dropout": 0.10, "oov_strategy": "unk"}},

    # ---- M4: direct memorisation check (dcr_share) on the arms M2/M3 make
    # interesting. Legacy settings are spelled out so these mean the same thing
    # whatever the library defaults are (see the provenance note in
    # HYPOTHESES.md); sampling uses HF's old top_k=50 like M1-M3.
    "m4_b0": {"init": {"unk_dropout": 0.0, "oov_strategy": "random", "ema_horizon": 0.0},
              "sample_variants": {"default": {"top_k": 50}}},
    "m4_tiny": {"gpt2": {"n_embd": 128, "n_head": 4, "n_layer": 3},
                "init": {"unk_dropout": 0.0, "oov_strategy": "random", "ema_horizon": 0.0},
                "sample_variants": {"default": {"top_k": 50}}},
    "m4_tiny_lr3e4": {"gpt2": {"n_embd": 128, "n_head": 4, "n_layer": 3},
                      "init": {"unk_dropout": 0.0, "oov_strategy": "random", "ema_horizon": 0.0},
                      "train_args": {"learning_rate": 3e-4, "warmup_steps": 0.05},
                      "sample_variants": {"default": {"top_k": 50}}},

    # ---- Before/after: the tool exactly as it was at the start of the utility-optimisation work
    # (commit e9e6c96, checked out as a separate worktree) vs. the current code, each called with ITS
    # OWN defaults. `_default` uses the library's default checkpoint rule (load_from_best_mean_sensitivity=False);
    # `_recipe` uses the previously recommended True. The new arms pass batch_size=None so the new
    # tabular default (32 x 1) applies; the old arms pass the old default 8 (accumulation 4).
    "old_default": {"src": "/home/jupyter-wb536061/WBG/REaLTabFormer-dev-base/src",
                    "fit": {"load_from_best_mean_sensitivity": False}},
    "old_recipe": {"src": "/home/jupyter-wb536061/WBG/REaLTabFormer-dev-base/src"},
    "new_default": {"batch_size": None, "fit": {"load_from_best_mean_sensitivity": False}},
    "new_recipe": {"batch_size": None},
}
