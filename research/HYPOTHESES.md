# Hypothesis & experiment registry

Living index of every idea under test in the `exp/utility-optimization`
branch. **Predictions are written before the experiment runs** so a result
can't be rationalised after the fact. `notes/lab_notebook.md` holds the
append-only dated narrative of each concluded experiment; this file holds
the current status of each idea and points at the raw data.

## How experiments are run and scored

- Harness: `research/bench.py` (multi-seed; one job = dataset x config x
  seed; results as JSON under `research/results/<matrix>/`). Summaries:
  `research/summarize.py <results dir> --ref <arm>`. Configs are named in
  `research/configs.py`.
- Training regime for all arms (the tool's own recommended one, per
  DECISION_LOG "Recommended configuration"): sensitivity-based stopping
  (`n_critic=5`) with `load_from_best_mean_sensitivity=True`, teacher-forced
  target. Stopping epoch is therefore an *outcome*, recorded per run.
- Dev datasets (exploration): diabetes (768), insurance (1,338), abalone
  (4,177), adult5k (5,000-row Adult subsample). Holdout datasets (used only
  to confirm a finding chosen on dev): wilt (4,839), churn2 (10,000).
- Seeds: 0, 1, 2 (train/test split and model init both vary with the seed).
- Metrics (lower is better unless noted): `marg_mean` (mean per-column KS for
  numeric / TVD for categorical vs. train), `tail_err` (99th-percentile
  error, numeric cols), `assoc_diff` (mean |delta Cramer's V| over all column
  pairs), `tstr` (higher better; gradient-boosting trained on synthetic,
  scored on real held-out data), `abs_gap` (|TRTR - TSTR|), `disc_dev`
  (|real-vs-synthetic discriminator AUC - 0.5|), and privacy: `frac_suspicious`
  (synthetic rows closer to train than the 5th percentile of held-out rows),
  `exact_dup` (synthetic rows exactly equal to a training row), `dcr_ratio`
  (mean synth-to-train DCR / mean test-to-train DCR; ~1 is healthy, <<1 is
  copying).
- Rule from DECISION_LOG: **a quality gain that comes with worse privacy
  metrics is not a win.**
- Judging: paired deltas on (dataset, seed) with standard errors. A difference
  smaller than ~2 s.e. is reported as "no detectable effect", not as a win.

## Status board

| ID | Idea | Status |
|----|------|--------|
| H0 | Harness sanity + seed noise floor (base, 3 seeds) | done (M1): headroom large (disc AUC ~0.69); abalone seed noise larger than predicted |
| H1 | `top_k=50` HF default silently truncates sampling | CONFIRMED on hicard (tvd_mean 0.142 -> 0.104, 3/3 seeds, no privacy change); no effect on bundled data. Implemented + tested on `exp/topk0-default` (06f37a7), not merged |
| H2 | Sampling temperature / nucleus | done (M1): T=0.9 and top_p=0.95 clearly worse; T=1.1 mild hint of gain, needs a finer test |
| H3 | Default GPT2 (768d x 6L) is oversized for small tables | running (M2a) |
| H4 | Default LR (5e-5, no warmup) under-trains; higher LR + warmup helps | running (M2a) |
| H5 | Re-check quantile encoding win with multiple seeds | done (M1): no overall gain; helps skewed marginals, worsens `frac_suspicious` on all 4 datasets -- not recommended by default; artifact-vs-copying question open |
| H6 | Fewer tokens per numeric column (`numeric_nparts=2`) | planned |
| H7 | `gradient_accumulation_steps=4` default hurts small data | running (M2a) |
| H8 | OOV: random substitution vs UNK vs UNK + input dropout | 5 seeds done: unk + dropout beats random 5/5, sits at the unconditional floor; random is worse than ignoring the seed 5/5. Unconditional-quality cost check running (`oovcost`) |

## Hypotheses

### H0 — Harness sanity and the seed noise floor
- **Why:** every prior result is single-seed; without a noise floor no delta
  can be judged.
- **Prediction:** across 3 seeds, `marg_mean` varies by a few thousandths
  and `tstr` by a few hundredths on small datasets; effects smaller than that
  are undetectable at n=3.
- **Test:** `base` config, 4 dev datasets x 3 seeds.

### H1 — `top_k=50` truncates sampling
- **Why:** `model.generate` is called with `do_sample=True` and no explicit
  `top_k`; HF's default generation config uses `top_k=50`, applied *after*
  the token-constraint mask, so any column with >50 admissible tokens (e.g.
  high-cardinality categoricals, 100-way numeric chunks) has its tail cut off
  and renormalised. Needs verification that the installed transformers
  actually applies it.
- **Prediction:** little effect on these datasets (low cardinalities), so
  expect "no detectable effect" on dev data; effect appears only on
  high-cardinality columns. Cheap to test, worth ruling out.
- **Test:** `samp` config variants `default` vs `topk0` on one trained model.

### H2 — Temperature / nucleus sampling
- **Why:** temperature 1 is the model's calibrated distribution; deviations
  trade diversity for precision.
- **Prediction:** T<1 may raise `tstr` slightly but worsens `marg_mean` and
  `frac_suspicious` (mode-seeking => copying); T>1 the reverse. No free win;
  T=1 stays.
- **Test:** `samp` variants `topk0_t09`, `topk0_t11`, `topp95`.

### H3 — Model size
- **Why:** `GPT2Config(n_layer=6)` inherits `n_embd=768, n_head=12` (~50M
  params) for tables of 768-5,000 rows; likely over-parameterised, slow, and
  prone to earlier memorisation.
- **Prediction:** a smaller model (e.g. 256d x 4L) matches fidelity/utility
  within noise, trains faster, and does not raise copying metrics.
- **Test:** `gpt2` overrides across 3 sizes, 4 dev datasets x 3 seeds.

### H4 — Learning rate and warmup
- **Why:** HF default LR 5e-5 with no warmup is a fine-tuning default, not a
  from-scratch one; sensitivity stopping may be firing on a slow-converging
  model.
- **Prediction:** LR 2e-4 with warmup reaches equal-or-better fidelity in
  fewer epochs; too high (1e-3) destabilises.
- **Test:** `train_args` LR sweep.

### H5 — Quantile encoding, multi-seed
- **Why:** DECISION_LOG calls it the one clean win but on single seeds.
- **Prediction:** improves `marg_mean`/`tail_err` on skewed numerics
  (insurance charges, adult capital-gain); neutral elsewhere.
- **Test:** `qenc` vs `base`.

### H6 — Fewer tokens per numeric column
- **Why:** `numeric_nparts=1` makes every digit a token (4 tokens per
  quantile-encoded column); `nparts=2` halves the sequence and the number of
  autoregressive steps where errors compound.
- **Prediction:** with quantile encoding, `nparts=2` is neutral-to-slightly
  positive on fidelity and clearly faster.
- **Test:** `qenc` vs `qenc_np2` (needs verification that nparts=2 works with
  quantile encoding).

### H7 — `gradient_accumulation_steps`
- **Why:** status doc records that the default of 4 silently multiplies the
  effective batch to 32 -- for 768-row data that is 24 optimizer steps per
  epoch.
- **Prediction:** `gradient_accumulation_steps=1` gives more updates per epoch
  and reaches the stopping point in fewer epochs; effect on final quality
  uncertain (earlier wilt ablation found it interacts badly with other knobs).
- **Test:** `train_args` gradient_accumulation_steps 1 vs 4.

### H8 — OOV handling (owner delegated this to me on 2026-09-20)
- **Why:** `data_utils/dataset.py` replaces an out-of-vocabulary value with a
  uniform random draw over that column's tokens (`oov_options` = all of the
  column's token ids) -- it defeats seed conditioning on an unseen value, is
  non-deterministic, and over-weights rare levels relative to their true
  frequency. Deterministic UNK is principled only if UNK's embedding is
  trained; otherwise it is an untrained random vector.
- **Prediction:** (a) random substitution gives conditionals biased towards
  rare-level behaviour vs the marginal; (b) plain UNK gives erratic output
  from an untrained embedding; (c) UNK + input-side token dropout during
  training gives output close to the marginal of the remaining columns, at
  negligible cost to ordinary generation quality.
- **Test:** hold a category level out of training entirely, seed with
  held-out rows carrying it, compare the generated remaining columns against
  the true conditional and the marginal; separately check unconditional
  quality cost of dropout with the standard harness. Both code paths
  (`get_token_id` and `_vectorized_column_token_ids`) must change together.

### H9 (new, from M1) -- constrained decoding cost
- **Why:** sampling took ~1 min per 1,024 rows in M1 under load; the per-row
  `prefix_allowed_tokens_fn` callback is a suspect.
- **Result (profile, not a matrix):** vectorised logits mask 0.22s vs callback
  76.5s for 1,024 rows, identical tokens for the same seed (measured on a busy
  shared box, so the factor overstates an idle-box gain). Implemented and
  tested, incl. any-order, on `exp/fast-constrained-decoding`.

### H10 (new, from M1) -- is quantile encoding's higher `frac_suspicious` real copying or a value-grid artifact?
- **Prediction:** an artifact of snapping to the 1,000-point training grid; a
  DCR computed on rank-transformed values, and a nearest-neighbour check that
  ignores the snapped columns, should show no excess.

## Ideas parked (not yet hypotheses)
- Numeric OOV: snap to the nearest in-vocab digit token rather than a random one.
- Row-level augmentation via column-order permutation on v1 (any_order is v2-only).
- Category-frequency smoothing / rare-level grouping in `process_categorical_data`.
- Tail handling for quantile encoding beyond the observed range.
