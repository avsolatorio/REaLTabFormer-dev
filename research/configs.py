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
    "default": {},  # HF defaults: top_k=50 (verified), temperature=1
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

    # ---- OOV cost check (H8): does input-side UNK dropout hurt ordinary
    # generation? b0 is the reference trained in this same worktree.
    "b0": {},
    "unkd03": {"init": {"unk_dropout": 0.03, "oov_strategy": "unk"}},
    "unkd10": {"init": {"unk_dropout": 0.10, "oov_strategy": "unk"}},
}
