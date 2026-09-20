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

CONFIGS = {
    # The tool's defaults + the DECISION_LOG "recommended" training regime
    # (sensitivity stopping, load best checkpoint) -- harness-level, not a
    # per-config choice; see bench.run_job.
    "base": {},
    "qenc": {"init": {"numeric_quantile_encoding": True}},
    # Sampling-only variants on a default-trained model.
    "samp": {
        "sample_variants": {
            "default": {},
            "topk0": {"top_k": 0},
            "topk0_t09": {"top_k": 0, "temperature": 0.9},
            "topk0_t11": {"top_k": 0, "temperature": 1.1},
            "topp95": {"top_k": 0, "top_p": 0.95},
        }
    },
}
