# REaLTabFormer — Lab Notebook

Created 2026-09-19, backfilled from the visible portion of an ongoing
session on `feat/support-seed-input`. Entries below are dated to when
the work actually happened, not to today.

Two untracked files already sitting in this repo, `DECISION_LOG.md` and
`OPTIMAL_STOPPING_RESEARCH.md`, contain substantial earlier lab-notebook-
equivalent content (the quantile-encoding work, the `any_order`/
`shared_numeric_vocab` decoupling fix, the original CUSUM detector
design) from before this file existed. Not yet merged in — ask before
treating this file as the complete record.

This notebook is append-only. A later entry that supersedes an earlier
one gets a new dated entry plus a forward-pointer on the old one; the
old entry's content is never rewritten.

---

## 2026-09-06 — CUSUM final-defaults validation across 5 datasets resolves the abalone open question

**Question:** With the fully-evolved CUSUM detector (`cusum_statistic="median"`,
delta ensemble, `patience=1`, cooldown auto-cap), does the previously-reported
abalone utility-gap regression (R² gap 0.190 vs. sensitivity's ~0.024) still hold?

**What was done:** Analyzed `cusum_validation/results/{abalone,adult,diabetes,
insurance,wilt}_cusum_ep300_*_summary.json` (committed in `217149b`) against
each dataset's already-committed `*_sensitivity_base_ep300_*` counterpart.

**Result:** abalone's utility gap dropped 0.190 → 0.022 (8.6x reduction),
`frac_suspicious` statistically tied with sensitivity (0.078 vs. 0.077),
1.7x faster. On the other 4 datasets CUSUM matched or beat sensitivity's
`frac_suspicious` in every case and ran 1.7x-8.2x faster; utility gaps
tracked closely except wilt, whose gap (0.089 vs. sensitivity's 0.004) is
now the largest in the comparison.

**Implication:** abalone question closed — likely explained by the
median statistic's outlier-robust Delta (not re-isolated to confirm
causally). wilt flagged as a new, smaller instance of the same failure
shape, motivating the ablation below.

---

## 2026-09-07 — wilt ablation: isolating gradient_accumulation_steps vs. cusum_confirm_with_sensitivity

**Question:** Which of `{gradient_accumulation_steps=1,
cusum_confirm_with_sensitivity, cusum_confirm_patience=2}` actually closes
wilt's utility gap?

**What was done:** 4 wilt runs at `cusum_statistic=median`, `epochs=300`,
varying exactly one knob at a time from the baseline
(`frac_suspicious=0.055`, gap=0.089, 13.3 effective epochs, 86.3s).

**Result:**
- `grad_accum=1` alone: gap unchanged (0.089→0.088); `frac_suspicious`
  dropped to 0.039 but effective epochs collapsed to 4.3 — undertraining
  artifact, not a real fix.
- `cusum_confirm_with_sensitivity` (patience=1): gap 0.089→0.012 (-86%),
  `frac_suspicious` 0.062, 16.7 effective epochs, 108.4s.
- confirm + `patience=2`: gap 0.016 (worse than patience=1), 21.3
  effective epochs, 143.0s — patience increment adds cost, no benefit.
- `grad_accum=1` + confirm + `patience=2` combined: gap 0.049 — worse
  than confirm alone; the two knobs interact negatively, not additively.

**Implication:** for wilt, the fix is `cusum_confirm_with_sensitivity`
alone (default `patience=1`); do not combine with
`gradient_accumulation_steps=1`. Recommended default for datasets
showing this failure shape. Not yet validated on other datasets.

---

## 2026-09-07 — hard-cohort tracker was silently non-functional: two real bugs found and fixed

**Question:** Why did `hard_cohort_history`/`hard_cohort_warmup_history`
stay completely empty across all 4 wilt ablation runs above, despite
`track_hard_cohort=True`?

**What was done:** Traced `rtf_cusum.py`'s `maybe_check`/
`_check_hard_cohort`; added a `hard_cohort_last_pool_size` diagnostic
(commit `9cfadab`) and reran (`wilt_ablate_all_hardcohort_recheck2`).

**Result:** real eligibility pool size averaged 77.7/193 (40%) against
a 96-row threshold, range 55-107, crossed the threshold on only 2 of 51
checks — well below a back-of-envelope estimate of ~67% steady-state
cooled fraction; root cause of that specific gap not fully explained.
Separately found and fixed two real bugs (`763d5eb`, `80776ac`):
(1) `_check_hard_cohort` was gated behind the MAIN tracker's own
var/se validity, unrelated to the hard cohort's own (independently
computed) data; (2) even when eligibility passed (steps 940, 1420), the
trajectory logger dropped the `hard_cohort_warmup_history` entry
because the main `post_calibration` branch always won the if/elif
chain first.

**Implication:** hard-cohort diagnostic is now correctly wired
end-to-end (verified via unit test + direct trajectory replay against
the real `attach_trajectory_logger` wrapper), but remains rarely
triggered for wilt specifically at the default `hard_cohort_frac=0.05`.
Why the real eligibility rate runs well below the naive estimate is
still open.

---

## 2026-09-11/12 — OOV fallback (random substitution vs. deterministic UNK): reasoned through, not yet reverted

**Question:** Does `data_utils`'s OOV handling
(`random.choice(oov_options)` instead of the UNK token) make sense?

**What was done:** Git-archaeology'd to commit `2183504` ("Add field
weights option", 2026-01-10) — no comment or docstring anywhere
explains the change. Searched past session transcripts
(`search_session_transcripts`) for the original design discussion —
zero REaLTabFormer-related sessions exist in the retained index; the
discussion is unrecoverable from available tools.

**Result:** reasoned it does not hold up on the merits — it silently
defeats the seed-input feature's purpose (a user's explicit
out-of-distribution conditioning value gets swapped for an unrelated
random real value, no warning); it's non-deterministic (draws from
Python's global `random`, not the file's own locally-seeded RNG used
elsewhere in the same module); and if the real motivation was "UNK is
never trained since the vocab is built from the training data itself,"
the standard, principled fix is UNK-dropout during training, not
silent substitution at inference. Separately confirmed OOV cannot occur
during ordinary training-set encoding (vocab is built from that exact
data), so this bug does not retroactively taint any `cusum_validation`
result generated so far — it's scoped to seed-input/relational-transfer
paths.

**Implication:** recommended reverting to deterministic UNK. Not done
yet — open, pending a decision.

---

## 2026-09-11 — Full codebase audit: 10 confirmed bugs via 8-angle review + independent verification

**Question:** What real bugs/semantic issues exist across the whole
`feat/support-seed-input` branch (`main...HEAD`, ~15.6k lines of real
code+test diff, `cusum_validation/` data and results excluded from the
angle prompts)?

**What was done:** 8 parallel finder-angle agents (line-by-line,
removed-behavior, cross-file, reuse, simplification, efficiency,
altitude, conventions) against `main...HEAD`, then one independent
verifier agent per surviving candidate (CONFIRMED/PLAUSIBLE/REFUTED).

**Result:** 10 of 11 verified candidates came back CONFIRMED (the 11th
self-refuted as stale before even reaching a verifier: "
`hard_cohort_warmup_history` never read" — false, `cusum_validation/
run_experiment.py` reads it, the finder agent's grep just didn't cover
that directory). Ranked by severity:
1. CUSUM's `_get_rows` indexes `train_dataset` positionally while the
   monitor tracks rows by stable `idx` — diverges whenever
   `train_size<1.0` (confirmed dormant for every `cusum_validation` run
   to date, since none override the `train_size=1` default).
2. OOV random substitution (see entry above).
3. `fit()` reusing `experiment_id` on a second call — risks silently
   overwriting a prior run's saved model under `allow_overwrite=True`.
4. v2's independently-rewritten `get_experiment_id` raises on a
   resumed `fit()` call that hits a periodic checkpoint save (v1
   guards this case explicitly; the v2 rewrite doesn't).
5. `save_full_every_epoch` default silently changed 5→0 — periodic
   checkpointing is off by default now, looks unintentional (traced to
   an unrelated commit).
6. `cusum` training path drops `field_weights`/`digit_entropy_weighting`/
   `predict_fields`/`compute_loss_func` — pure missed-forwarding gap.
7. v2's `make_dataset` calls omit `seed=self.random_state` — breaks
   reproducibility specifically for v2.
8. Hard-cohort warmup pacing gap (rhymes with the entry above; the
   fast-calibration-pace mechanism only covers the main tracker).
9. `_fit_relational` silently ignores `numeric_categorical_threshold`/
   `numeric_quantile_encoding`/`numeric_quantile_bins`.
10. `grokfast_args` silently no-ops for `model_type="relational"`.

**Implication:** none of these fixed yet except the two hard-cohort
bugs (already fixed the day before, entry above) and the Python 3.9
import crash (next entry). The other 8 remain open.

---

## 2026-09-16/19 — REaLTabFormer2 uninstantiable on Python 3.8/3.9: found, fixed, verified

**Question (found incidentally, while building an isolated diff branch
to scope a cloud code review to just `realtabformer2.py` and its real
dependencies):** does `realtabformer2.py` actually import cleanly?

**What was done:** `PYTHONPATH=src python3 -c "import
realtabformer.realtabformer2"` — first on the constructed review
branch, then confirmed the same failure reproduces on the real
`feat/support-seed-input` branch directly (ruling out an artifact of
the constructed branch).

**Result:** `TypeError: unsupported operand type(s) for |:
'_CallableType' and 'NoneType'` — a `Callable | None` (PEP 604) type
hint with no `from __future__ import annotations`, needs Python 3.10+,
but `pyproject.toml` declares `python >= 3.8`. `__init__.py` doesn't
import `realtabformer2`, so the base `realtabformer` package was
unaffected; only `REaLTabFormer2` itself was unusable below 3.10.

**Implication:** fixed by adding `from __future__ import annotations`
(commit `d1e5d01`, pushed to `origin/feat/support-seed-input`). Verified
the import succeeds on Python 3.9.6 and the full `test_realtabformer2.py`
suite still passes (28/28) — no code in the file inspects annotations
at runtime (no `get_type_hints`/`.annotation` usage), so this is a pure
compatibility fix with no behavior change.

---

## 2026-09-19 — Full method-by-method diff of REaLTabFormer (v1) vs. REaLTabFormer2 (v2): what's genuinely shared vs. genuinely different vs. accidentally drifted

**Question:** Before refactoring `realtabformer2.py` to reuse `realtabformer.py`'s
code where relevant (motivated by `get_experiment_id`'s bug — see the
2026-09-11 audit entry — being a direct consequence of duplication
instead of reuse), what exactly differs between the two files' ~24
parallel methods, and which differences are load-bearing vs. accidental?

**What was done:** Extracted and diffed every method that exists under
the same name in both `src/realtabformer/realtabformer.py` and
`src/realtabformer/realtabformer2.py` (`_normalize_gpt2_state_dict`,
`_validate_get_device`, `_build_training_args`, `__init__`,
`_invalid_model_type`, `_init_tabular`, `_init_relational`,
`_extract_column_info`, `_generate_vocab`, `_check_model`,
`_split_train_eval_dataset`, `fit`, `_train_with_sensitivity`,
`_train_with_objective`, `get_full_save_dir`, `get_experiment_id`,
`_set_up_relational_coder_configs`, `_fit_relational`, `_fit_tabular`,
`_build_tabular_trainer`, `sample`, `predict`, `save`,
`load_from_dir`), line by line, via `diff` on the extracted source
ranges — not by re-reading from memory or assuming similarity.

**Result:**
- **10 methods are byte-for-byte identical**: `_normalize_gpt2_state_dict`,
  `_validate_get_device`, `_invalid_model_type`, `_extract_column_info`,
  `_generate_vocab`, `_check_model`, `_split_train_eval_dataset`,
  `get_full_save_dir`, `sample`, `predict`.
- **Cosmetic-only differences** (zero behavior change): `_build_training_args`
  (docstring wording only — v2's own docstring literally says "duplicated
  here rather than imported, matching this file's existing pattern");
  `Optional[Callable]` (v1) vs. `Callable | None` (v2) showing up in
  several signatures (pure PEP 604 syntax choice).
- **Legitimate, intentional differences that must be preserved**: every
  difference in `_init_tabular`, `_init_relational`,
  `_set_up_relational_coder_configs`, `_fit_relational`, `save`,
  `load_from_dir` traces to v2's backbone-generality feature
  (`GPT2Config`/`GPT2LMHeadModel` hardcoding in v1 vs. `AutoConfig`/
  `AutoModelForCausalLM`/`CONFIG_MAPPING`-based generic reconstruction
  in v2, plus the renamed `parent_gpt2_*` → `parent_encoder_*`
  attributes). Every difference in `__init__`, `_fit_tabular`,
  `_build_tabular_trainer` traces to `any_order`/`shared_numeric_vocab`,
  confirmed v2-only (zero references to either in `realtabformer.py`
  at all). `fit()`'s ~25-parameter gap is the entire CUSUM parameter
  surface, absent because v2 has no CUSUM support whatsoever (no
  `_train_with_cusum`/`_build_cusum_confirm_fn`/
  `_build_cusum_diagnostic_fn` exist in the file) — and `_build_tabular_trainer`/
  `_fit_tabular` in v2 have no `trainer_cls`/`add_row_idx` plumbing either,
  so CUSUM couldn't be wired into v2 without that groundwork first.
- **4 confirmed accidental bugs from independent reimplementation** (2
  already known, 2 newly found by this diff pass):
  1. `get_experiment_id` (known, 2026-09-11 audit) — v1 checks
     `epoch is not None` first, unconditionally; v2 checks
     `self.experiment_id is not None` first and raises if `epoch` is
     also set. Confirmed at the source: v1's ordering is the one that
     avoids crashing on a resumed `fit()`'s periodic checkpoint save.
  2. `make_dataset`/`make_dataset_with_column_types` missing
     `seed=self.random_state` (known, 2026-09-11 audit) — confirmed
     directly in `_fit_tabular`'s diff: v1's single `make_dataset(...)`
     call has `seed=self.random_state`; neither of v2's two calls do.
  3. **New**: `_train_with_objective` wipes existing checkpoints
     *unconditionally* in v2. v1 has `if not resume_from_checkpoint:`
     guarding the wipe, with a comment explaining exactly why
     ("deleting them first would make `resume_from_checkpoint` always
     find nothing and silently restart from epoch 0"). v2 lost the
     guard entirely — `REaLTabFormer2().fit(resume_from_checkpoint=True,
     ...)` on the objective-callback path deletes the very checkpoint
     it's about to resume from, then silently restarts from epoch 0.
  4. **New**: v2's `_train_with_sensitivity` is missing the
     `shared_preprocessor` optimization (fit once, reuse across every
     periodic check instead of refitting each time — a benchmarked
     ~2.2x speedup on the preprocessing step at Adult-like scale) and
     the `sensitivity_cache_dir`/`sensitivity_bootstrap_n_jobs` params
     (disk caching + configurable bootstrap parallelism). Not a
     correctness bug — a silent performance/capability regression for
     anyone using v2's sensitivity-training path.

**Implication:** the case for refactoring is now empirically grounded,
not just architectural intuition — 10 methods can be unified with zero
risk today, `_build_training_args` and the `Callable | None` syntax
choice with near-zero risk, and unifying `get_experiment_id` around
v1's (correct) logic fixes a real bug as a side effect rather than
requiring a separate patch. The backbone-generality and `any_order`/
`shared_numeric_vocab` differences must NOT be merged away — they're
the actual reason v2 exists. Bugs #3 and #4 are real but sit inside
methods that are NOT safe to unify wholesale (both have substantial
legitimate v1/v2-specific logic alongside the drifted piece) — they
need targeted, standalone fixes rather than being resolved by the
duplication-removal refactor itself.

---

## 2026-09-19 — Executed the safe first slice of the refactor: extracted 10 identical methods into a shared mixin, fixing `get_experiment_id` as a consequence

**Question:** Having mapped exactly which methods are safe to share
(the entry above), does actually extracting them work cleanly, and
does it fix `get_experiment_id` as a side effect rather than requiring
a separate patch?

**What was done:** Created `src/realtabformer/rtf_shared.py` containing
the 3 identical module-level functions (`_normalize_gpt2_state_dict`,
`_validate_get_device`, `_build_training_args`) and a new
`SharedModelMixin` class with the 8 identical instance methods
(`_invalid_model_type`, `_extract_column_info`, `_generate_vocab`,
`_check_model`, `_split_train_eval_dataset`, `get_full_save_dir`,
`sample`, `predict`) plus `get_experiment_id` — using v1's version
specifically, since v2's independently-reimplemented copy was the
confirmed bug. Made both `REaLTabFormer` and `REaLTabFormer2` inherit
from `SharedModelMixin`, removed the now-duplicate method bodies from
both files (replaced with a one-line comment noting the inheritance),
removed the resulting unused imports (verified via `pyflakes`, zero
warnings across all three files afterward), and verified backbone
generality / `any_order` / `shared_numeric_vocab` / CUSUM-support
differences were untouched (they live in methods this refactor
deliberately did not touch).

**Result:** `REaLTabFormer.__mro__` and `REaLTabFormer2.__mro__` both
confirm `SharedModelMixin` is now in the chain. Directly verified the
bug fix: `REaLTabFormer2.get_experiment_id(epoch=5)` with
`experiment_id` already set now returns `"full_model_epoch_005"`
instead of raising — the epoch-branch wins first, matching v1's
(correct) precedence, with no separate patch needed. Full test suite
before vs. after the refactor, both runs on the same machine/Python
3.9.6:
- `test_realtabformer.py`: 7 passed, 1 failed both before and after
  (`test_default_init`, asserts `epochs == 100` against the real
  constructor default of `1000` — confirmed via `git stash` to be
  pre-existing, unrelated to this change, not yet investigated
  further).
- `test_realtabformer2.py`: 28 passed both before and after.
- Full suite (`tests/realtabformer/`): 166 passed, 2 failed both
  before and after (`test_default_init` above, plus
  `test_rtf_sampler.py::test_TabularSampler` — a `ValueError: Input X
  contains NaN` inside `TruncatedSVD.fit_transform` during the
  sensitivity-threshold bootstrap, also confirmed via `git stash` to
  be pre-existing and unrelated).
- `black`/`isort` on all three touched/new files: zero new
  disagreements beyond the same pre-existing installed-tool-version
  drift already documented elsewhere in this branch's history (verified
  by running both checks on the pre-refactor files too).

**Implication:** zero regressions, one real bug fixed as a structural
consequence of removing its root cause rather than patched in place.
Surfaced two previously-undocumented-in-this-thread pre-existing test
failures (`test_default_init`'s stale `epochs` expectation,
`test_TabularSampler`'s NaN-in-bootstrap) — real, but out of scope for
this refactor; worth a look later. Remaining candidates for a *second*
slice, not done here: `_build_training_args`'s docstring-only diff and
the `Optional[Callable]`/`Callable | None` syntax choice (both
near-zero risk); the two newly-found bugs from the 2026-09-19 diff
entry (`_train_with_objective`'s unconditional checkpoint wipe,
`_train_with_sensitivity`'s missing `shared_preprocessor`/caching
optimization) need standalone fixes, not extraction, since they sit
inside methods with substantial legitimate v1/v2 differences alongside
the drift.

---

## 2026-09-19 — Fixed the two remaining bugs from the diff pass as standalone patches

**Question:** Do the two remaining bugs found in the earlier diff pass
(`_train_with_objective`'s unconditional checkpoint wipe,
`_train_with_sensitivity`'s missing `shared_preprocessor`/caching
optimization) actually hold up as real fixes to apply, given they sit
inside methods with substantial legitimate v1/v2 differences alongside
the drift (so they needed standalone patches, not extraction into the
shared mixin)?

**What was done:** For `_train_with_objective`: added back v1's
`if not resume_from_checkpoint:` guard (with the same explanatory
comment) around v2's checkpoint-wipe loop — `resume_from_checkpoint`
was already a parameter, just not used for this. For
`_train_with_sensitivity`: added `sensitivity_cache_dir`/
`sensitivity_bootstrap_n_jobs` params to both `fit()` and
`_train_with_sensitivity` itself (v2's `fit()` didn't even accept
them before — the params existing only on `_train_with_sensitivity`
wouldn't have been reachable), threaded them into the
`compute_sensitivity_threshold` call as `cache_dir=`/`n_jobs=`, and
added the `shared_preprocessor = SyntheticDataBench.
_maybe_fit_shared_preprocessor(...)` call plus `preprocessor=
shared_preprocessor` in both `compute_sensitivity_metric` call sites,
matching v1 exactly.

**Result:** Direct `diff` of both methods against v1's current versions
afterward: `_train_with_sensitivity` differs by exactly one harmless
comment-wording line ("cusum path" vs. "v1 sensitivity/cusum paths");
`_train_with_objective` differs only by the already-known
`Optional[Callable]`/`Callable | None` cosmetic choice and an extra
explanatory comment left over from the earlier `get_experiment_id`
removal. Verified the new params are actually reachable end to end
(`inspect.signature(REaLTabFormer2.fit)` includes
`sensitivity_cache_dir`, not just the internal method). `black`/`isort`
diff line counts identical before and after (31/46 lines respectively,
both pre-existing installed-tool-version drift, confirmed via direct
comparison against the pre-fix file) — no new formatting issues.
Full test suite: 166 passed, same 2 pre-existing failures as every
other check this session, both times — zero regressions.

**Implication:** both fixes verified correct and safe, not just
"probably fine." All 4 bugs found in the 2026-09-19 diff-and-refactor
work (this entry, the mixin-extraction entry, and the original diff
entry) are now resolved. Remaining open items are the ones from the
2026-09-11 audit not touched by this thread: the CUSUM row-index bug,
the OOV substitution question (needs a decision), `fit()`'s own
`experiment_id`-reuse design question, `save_full_every_epoch`'s
default, `cusum`'s missed field_weights/digit_entropy forwarding, and
`_fit_relational`/`grokfast_args`'s relational-mode gaps.

---

## 2026-09-20 — Utility-optimization program started: harness built, plus four findings from reading/probing the code (no experiment results yet)

**Question:** Where can the data-processing, generation and model-design
choices be improved to raise synthetic-data utility without raising
copying? First step: what evidence standard can the existing tooling
support, and what does the code actually do at generation time?

**What was done:** Read `data_utils/{process,transform,dataset,vocab}.py`,
the model/trainer construction in `realtabformer.py`, and the generation
and decode path in `rtf_sampler.py`. Built `research/bench.py` (multi-seed;
scores fidelity, gradient-boosting TSTR utility, discriminator AUC and
privacy together; scores several sampling variants on one trained model;
includes a real-vs-real "oracle" noise floor), `research/summarize.py`
(paired deltas on dataset x seed with standard errors) and
`research/HYPOTHESES.md` (predictions pre-registered before running).
Verified the four items below directly.

**Result:**
1. **HF's default `top_k=50` is applied to REaLTabFormer sampling.**
   Probed with a 200-token toy model, 5,000 first-token draws: 50 distinct
   tokens with default kwargs, 200 with `top_k=0`. The token-constraint mask
   (`prefix_allowed_tokens_fn`) runs first, so any column with >50
   admissible tokens has its tail cut and renormalised: categorical columns
   with >50 levels, and numeric chunks when `numeric_nparts>=2` (100-way
   chunks). None of the six bundled datasets has such a column (widest:
   41 levels), so this bug is invisible on them; a synthetic 300-level Zipf
   dataset (`hicard`) was added to test it. Not yet measured.
2. **The default model is large for these tables.** `GPT2Config(n_layer=6)`
   inherits GPT-2's 768-wide, 12-head shape: 43.5M parameters, fit for a
   614-row training set (measured in the harness smoke run, diabetes).
3. **`mask_rate` is static.** The training set is built once with
   `Dataset.map`, so the [RMASK] positions are drawn once and are identical
   in every epoch; labels are also built from the already-masked ids, so
   the model is trained to predict [RMASK], which generation then suppresses.
   Not the regularisation its docstring implies. Not yet measured.
4. **Correction to the 2026-09-11/12 OOV entry.** That entry argued random
   OOV substitution is non-deterministic because it draws from Python's
   global `random`. That holds for the scalar `get_token_id` path only.
   Seed inputs go through `make_dataset`, which uses the vectorised path with
   a fresh `np.random.default_rng(self.random_state)` on every call, so on
   the seed path the substitution is deterministic (always the same
   arbitrary level for a given model) -- arbitrary, not random. The other
   arguments in that entry (it defeats seed conditioning; it draws uniformly
   over levels, so it over-weights rare levels relative to their true
   frequency) are unaffected. Also: both encoders share the substitution,
   so a revert must change `get_token_id` AND `_vectorized_column_token_ids`.
   Also found while setting up: the repo `.gitignore` already ignores
   `experiments/`, which would have silently kept the harness out of git;
   the harness lives in `research/` instead.

**Implication:** Every comparison from here on uses >=3 seeds and reports
privacy alongside utility. Items 1 and 3 become hypotheses H1 and (new)
mask-rate; item 4 refines the H8 test design. See `research/HYPOTHESES.md`.

---

## 2026-09-20 04:22 UTC — M1: baseline noise floor, sampling variants (H1, H2) and quantile encoding (H5) on 4 datasets x 3 seeds

Code/data: commit `b7d559e` (raw JSON in `research/results/m1/`). The jobs
actually ran from `68f7122` plus uncommitted edits to `research/configs.py`
(the M1 arm definitions) and, mid-run, to `research/bench.py` (HGB
categorical cap and GPU assignment -- neither changes any metric for the
four datasets below). `research/results/m1h` and `m2a` partial results are in
the same commit; they are not analysed here.

**Question:** Against a real noise floor, (H0) how far is default synthetic
data from real, and how big is seed noise? (H1) Does HF's default `top_k=50`
matter? (H2) Do temperature / nucleus sampling give a free win? (H5) Does
quantile encoding still win with several seeds *when privacy is scored too*?

**What was done:** diabetes, insurance, abalone, adult5k x seeds 0,1,2 x
{`base`, `qenc`}; `base` scored under 5 sampling variants on one trained
model. Sensitivity stopping + best-checkpoint loading, teacher-forced target.
12 paired (dataset, seed) units per arm. hicard (the H1 test that can
actually show an effect) is NOT in this entry: all 6 of its jobs failed in my
metric code (HistGradientBoosting rejects >255-level categoricals) after
training and sampling had finished; rerun as `m1h`, pending. Because ~50
paired comparisons were made, single results near 2 s.e. are treated as
hints; only effects that are large (~3+ s.e.) or that point the same way on
every dataset are called findings.

**Result:**
- **H0 noise floor.** Default synthetic data is 1.5-3x the real-vs-real
  floor on marginal distance (`marg_mean` 0.062-0.098 vs floor 0.023-0.065
  by dataset) and is easily told apart from real: discriminator AUC
  0.68-0.70 on all four datasets (floor 0.50). Large headroom. Seed noise is
  bigger than I predicted on abalone: SD across seeds 0.033 on `marg_mean`
  and 0.036 on TSTR (adult5k: 0.005 and 0.005). Sensitivity stopping lands
  at epoch 28-33 on average (SD 3-7.6 epochs).
- **H1 (`top_k=0` vs default 50), bundled datasets:** no detectable effect,
  as predicted. `marg_mean` +0.0003 +-0.0034 (6 better/6 worse), TSTR -0.005
  +-0.007. One hint: `frac_suspicious` +0.010 +-0.004 (2.5 s.e., 3/8) --
  not established. The real test is hicard, still pending.
- **H2 temperature/nucleus.** T=0.9: clearly worse -- `marg_mean` +0.014
  +-0.003 (1 better/11 worse), discriminator distance from 0.5 +0.031 +-0.006
  (1/11), `frac_suspicious` +0.020 +-0.004 (0/11); TSTR -0.003 +-0.009, so the
  small utility gain I predicted for T<1 did not appear. `top_p=0.95`:
  clearly worse -- `tail_err` +0.060 +-0.017 (0/12), discriminator distance
  +0.049 +-0.008 (0/12): nucleus truncation cuts real tails. T=1.1: small,
  same-direction improvements (`marg_mean` -0.007 +-0.004, 8/4; discriminator
  distance -0.013 +-0.006, 8/4; `frac_suspicious` -0.007 +-0.003, 7/5), TSTR
  -0.008 +-0.007. About 2 s.e. each -- a hint that the trained model is
  slightly over-confident, not a finding.
- **H5 quantile encoding vs default (both `default` sampling).** No
  detectable overall gain: `marg_mean` -0.009 +-0.006 (6/6), TSTR -0.002
  +-0.009. Per dataset it helps the marginals where predicted -- insurance
  -0.023, abalone -0.013 -- and hurts adult5k (+0.009). `assoc_diff` is worse
  (+0.0034 +-0.0015, 3 better/9 worse). **`frac_suspicious` is worse on all
  four datasets** (+0.011 to +0.035; mean +0.019 +-0.005, 10 of 12 units
  worse). `exact_dup` is 0 for every arm, so this is closeness, not copied
  rows. Wall-clock: `qenc` fit ~510 s faster on average (+-158), but jobs
  ran at different machine loads, so this is not attributable to the encoding.

**Implication:** (1) Do not use `top_p`; do not use T<1. (2) DECISION_LOG's
"quantile encoding is the one clean win" does not survive multi-seed,
privacy-scored testing on these four datasets: it improves marginals on
skewed columns but raises the suspicious-closeness rate on every dataset, and
under the standing rule (quality gain with worse privacy is not a win) it is
not recommended by default on this evidence. Open question, untested: the
extra closeness may be an artifact -- quantile decoding snaps values to a
1,000-point training grid, which lowers a value-space DCR without any
memorisation -- rather than genuine copying. (3) T slightly above 1 is worth a
proper test (finer grid, more seeds). (4) Finish H1 on hicard before
concluding anything about `top_k`. Hypotheses updated in
`research/HYPOTHESES.md`.

---

## 2026-09-20 07:29 UTC — H1 confirmed on hicard: HF's default `top_k=50` measurably degrades a 300-level categorical column

Code/data: `research/results/m1h/` at commit `b33e8de` (harness as of M1 plus
the >255-level HGB fix). Seeds 0,1,2; hicard is the synthetic table (4,000
rows, Zipf-distributed 300-level `city`, `income` driven by city, `segment`
derived from income) built because no bundled dataset has a column wide enough
for `top_k=50` to bite.

**Question:** Does the default `top_k=50` truncation degrade columns with >50
admissible tokens (H1, deferred from the M1 entry)?

**What was done:** The M1 arms rerun on hicard (jobs had failed in M1 in my
metric code and were rerun as `m1h`): `base` under 5 sampling variants and
`qenc` under 2, 3 seeds, all scored on one trained model per seed.

**Result:** `top_k=0` vs the default, paired by seed (mean of 3 seeds; +- is the
SD across seeds, n=3):
- Categorical fidelity `tvd_mean` 0.1416 -> 0.1042 (-0.037 +-0.008), better in
  all 3 seeds (0.158->0.120, 0.126->0.081, 0.141->0.112).
- `assoc_diff` 0.0415 -> 0.0166 (-0.025 +-0.004); `marg_mean` 0.1147 -> 0.0937;
  discriminator AUC 0.651 -> 0.597 (real-vs-real 0.503).
- Privacy unchanged: `frac_suspicious` 0.0633 -> 0.0621 (-0.001 +-0.018),
  `exact_dup` 0. TSTR is uninformative here (0.9996 in every arm: `segment` is a
  deterministic function of `income`).
- Within `top_k=0`, temperature 1 is best on `tvd_mean` (0.104) vs T=0.9 (0.132),
  T=1.1 (0.119), `top_p=0.95` (0.108). So the M1 hint that T=1.1 helps is NOT
  corroborated on categorical fidelity (T=1.1 is 0.015 worse than T=1 here;
  its lower `frac_suspicious`, 0.047 vs 0.062, has SD ~0.02).
- `qenc` on hicard: no distinguishable gain (`tvd_mean` -0.006 +-0.026), and
  `frac_suspicious` again in the worse direction (+0.020 +-0.013).
- Not measured: per-column numbers for `city` (which cities appear, how many
  distinct levels) -- the synthetic tables were not saved, so the mechanism
  (tail levels cut and renormalised) is inferred from the design, not
  observed directly.

**Implication:** The default `top_k=50` silently degrades high-cardinality
categorical columns; it is invisible on low-cardinality data (M1: no
detectable effect either way). Recommend the sampler pass `top_k=0` unless the
caller sets one. That is a behaviour change to a default, so it is proposed
here, not yet made. Caveats: one synthetic dataset, 3 seeds; the size of the
effect on real high-cardinality data is unmeasured.

---

## 2026-09-20 07:29 UTC — H8 OOV handling: deterministic UNK + input-side UNK dropout beats random substitution on 5/5 seeds; unconditional-quality cost check still running

Code: library change `b8597a8` (`oov_strategy`, `unk_dropout`, collator) on
`exp/oov-unk-dropout`, merged with the fast-decoding branch (`f864173`);
raw results `research/results/oov/` at commit `f90b338` on that branch.
Single-process experiment script `research/oov_bench.py`.

**Question:** When a `seed_input` carries a category value never seen in
training, which handling of that value behaves best: the current random
substitution, deterministic [UNK], or [UNK] made meaningful by training with
input-side [UNK] dropout? (Owner delegated the OOV decision to exploration on
2026-09-20; nothing has been merged into `feat/support-seed-input`.)

**What was done:** adult5k, column `occupation` (moved to first position so a
v1 seed can be a prefix). Per seed, one level with 3-10% frequency is held OUT
of the training data entirely, so it is truly OOV (seeds 0,2,3 drew
Transport-moving, seed 1 Machine-op-inspct, seed 4 Tech-support -- only three
distinct levels, not five). Trained with dropout in {0, 0.03, 0.10}; each model
seeded with {occupation: <held-out level>} under both policies (`random` =
current, `unk`) by toggling `oov_strategy` at encoding time, 6 x 300 rows per
arm. Measured: mean per-column KS/TVD of the OTHER columns against (a) the
true conditional (the held-out rows, 174-279 of them) and (b) the training
marginal; references: an unconditional sample (= "ignore the seed") and, as a
control, conditioning on known levels. Seed 2 failed once on a CUDA
out-of-memory error (shared GPU) and was rerun; its failed file is kept under
`failed_oom/`.

**Result (mean over 5 seeds; lower is closer):**

| arm | distance to marginal | distance to true conditional |
|---|---|---|
| `random`, no dropout (current) | 0.119 (range 0.067-0.168) | 0.172 |
| `unk`, no dropout (UNK untrained) | 0.075 | 0.137 |
| `unk` + 3% dropout | 0.047 | 0.134 |
| `unk` + 10% dropout | 0.041 | 0.124 |
| ignore the seed (unconditional sample) | 0.039 | 0.129 |

- `random` is *worse than ignoring the seed* on the true conditional in 5 of 5
  seeds (0.172 vs 0.129), and its distance from the marginal swings with which
  arbitrary level it happens to pick (0.067 to 0.168).
- `unk` + dropout is closer than `random` to both references in 5 of 5 seeds
  (3%: -0.072 / -0.038 on average; 10%: -0.078 / -0.048). It lands at the
  unconditional floor (0.041-0.047 vs 0.039): an unknown value behaves like
  "no information". At 10% it is better than ignoring the seed on the true
  conditional in 4 of 5 seeds, but the mean gain (0.124 vs 0.129) is small; I do
  not claim that.
- Plain `unk` without dropout helps on average but is erratic (seed 3: 0.098
  from the marginal, worse than seeds with dropout).
- Conditioning on KNOWN levels still helps by the same amount in every arm
  (0.080-0.084), so dropout did not visibly weaken real conditioning.

**Implication:** Random substitution should not stay the default: it silently
gives worse-than-no-conditioning output. The fix that works is deterministic
[UNK] plus input dropout at training time; [UNK] alone is not enough.
Recommendation: `oov_strategy="unk"` with `unk_dropout` of about 0.03-0.10 as
the default. Not yet known: (1) whether dropout costs ordinary (unseeded)
generation quality -- running now as matrix `oovcost` (b0 vs 3% vs 10%, 4
datasets x 3 seeds); (2) behaviour on numeric OOV values, on other datasets,
and on v2/any-order (the collator is v1 only); (3) only three distinct held-out
levels were tested. Decision to change the default is left until (1) is in.

---

## 2026-09-20 07:39 UTC — `top_k=0` default for tabular sampling implemented and validated (owner approved implementing it; not merged into `feat/support-seed-input`)

Code: `06f37a7` on `exp/topk0-default`, stacked on `exp/fast-constrained-decoding`
(`bf249ff`), itself off `exp/utility-optimization`.

**Question:** Can the H1 finding (2026-09-20 07:29 UTC entry) be shipped as a
default without changing anything it has no evidence for?

**What was done:** `TabularSampler.default_top_k = 0` (new class attribute on
the base sampler, default `None` = leave HF alone); `_generate` applies it
only when the caller passed no `top_k` (or `None`) and is actually sampling.
`RelationalSampler` keeps HF's default. Tests: a spy on `model.generate`
(default -> `top_k=0`; explicit 25 -> 25; `None` -> 0; greedy -> no `top_k`;
relational default is `None`), and a behavioural test on a 200-level first
column of a near-uniform 1-epoch model (default sampling yields >50 distinct
levels; `top_k=50` yields <=50). Mutation-checked: with `default_top_k=None`
the behavioural test fails.

**Result:** Full suite 171 passed, 2 failed -- the same two failures that
pre-date this work (`test_default_init`, `test_TabularSampler`). No
regressions.

**Implication:** Ready for review. Effect sizes are from one synthetic
dataset with 3 seeds (see the H1 entry); the change is neutral on the
bundled low-cardinality datasets (M1). It alters default sampling output for
any model with a column wider than 50 tokens, including `numeric_nparts>=2`.
Merging into `feat/support-seed-input` is left to the owner.

---

## 2026-09-20 15:43 UTC — H8 cost check: input-side UNK dropout does not measurably hurt unseeded generation at 3%; and the fast decoder reproduces M1's baseline to 4 decimals

Code/data: `research/results/oovcost/` at `9935ab2` (branch
`exp/oov-unk-dropout`, library change `b8597a8` merged with fast decoding
`f864173`). 4 datasets x 3 seeds, sensitivity stopping, same regime as M1.

**Question:** Does training with [UNK] dropout (needed to make OOV -> [UNK] work,
see the 07:29 UTC H8 entry) cost ordinary, unseeded generation quality or
privacy?

**What was done:** arms `b0` (no dropout, reference), `unkd03`, `unkd10`;
paired on (dataset, seed), 12 units per arm.

**Result (arm - b0, mean +-s.e.; b0 in brackets):**
- `unk_dropout=0.03`: `marg_mean` -0.0019 +-0.0025 [0.0808], `tail_err` -0.0010
  +-0.0039, `assoc_diff` +0.0009 +-0.0008, TSTR -0.0022 +-0.0090, discriminator
  distance -0.0085 +-0.0095 [0.187], `frac_suspicious` +0.0063 +-0.0044 [0.0477],
  `exact_dup` 0. Stops 3.0 +-1.3 epochs later (30.7 -> 33.6). No detectable cost.
- `unk_dropout=0.10`: `marg_mean` -0.0068 +-0.0059, `tail_err` -0.0123 +-0.0075
  (9 better/3 worse), discriminator distance -0.0457 +-0.0152 (9/3, ~3 s.e.),
  TSTR +0.0017 +-0.0058, but `frac_suspicious` +0.0142 +-0.0065 (3 better/9
  worse, ~2.2 s.e.) and stops 9.4 +-1.5 epochs later (30.7 -> 40.1).
- **Full-pipeline equivalence of the fast decoder:** `b0/default` here (trained
  and sampled with the vectorised constraint) reproduces M1's `base/default`
  (trained and sampled with the per-row callback) to four decimals on every
  metric (e.g. `marg_mean` 0.0808, `assoc_diff` 0.0212, TSTR 0.7529,
  `frac_suspicious` 0.0477) and in mean stopping epoch (30.6888). This is a
  much stronger check than the unit tests: same seeds, 12 complete
  train-stop-sample-score pipelines, identical results. Only wall-clock differs
  (mean `fit_s` 851 -> 326, but the two ran at different machine loads, so that
  ratio is not a controlled benchmark).

**Implication:** At 3% the OOV fix costs nothing measurable and captures almost
all of its benefit (distance to marginal 0.047 vs 0.041 at 10%, floor 0.039). At
10% dropout acts as a regulariser that delays sensitivity stopping by ~9 epochs
and comes with a hint (2.2 s.e., one of ~20 comparisons) of more suspiciously
close rows; not recommended as a default. Recommendation to the owner:
`oov_strategy="unk"` with `unk_dropout=0.03`. Not merged; the change alters
training for every model (adds a collator), so it stays a proposal.

---

## 2026-09-20 15:45 UTC — M2: a much smaller model gives a large fidelity gain at flat utility and flat privacy metrics; higher learning rate is worse; grad-accum 1 gives no quality gain

**Partly superseded -- see the 2026-09-20 21:01 UTC entry below: the default model's LOADED checkpoint was epoch 5-16, not its stopping epoch, so the size comparison below was against an under-trained baseline. Original content unchanged.**

Code/data: `research/results/{m2a,m2b,m2c}/` at `dc326e8` (paired against M1's
`base/default`, same 12 dataset x seed units). Provenance: the 41 `m2a` jobs
ran on the old per-row-callback sampler; `m2b`/`m2c` (19 jobs) on the
vectorised decoder -- shown identical in the entry above. Sampling uses HF's
default `top_k=50` throughout (comparable with M1). GPT2 config differs only in
the listed fields; LR arms use `warmup_steps=0.05` (the installed transformers
rejects `warmup_ratio`, which the library would silently have dropped).

**Question:** H3 (is the default 768-wide x 6-layer GPT2, 43.5M parameters, too
big for these tables?), H4 (does a higher learning rate + warmup help?), H7 (does
`gradient_accumulation_steps=1` help?).

**What was done:** 5 arms x 4 datasets x 3 seeds under the standard sensitivity
regime with a 300-epoch ceiling. `small` = 256d/8 heads/4 layers, `tiny` =
128d/4/3, `lr3e4`, `lr1e4`, `ga1`.

**Result (arm - base, mean +-s.e., wins/losses of 12; base in brackets):**
- **H3 model size -- large, consistent effect.** `tiny`: `marg_mean` 0.0808 ->
  0.0288 (-0.0520 +-0.0064, 12/0), `tail_err` -0.0355 +-0.0103 (12/0), discriminator
  distance from 0.5 0.187 -> 0.032 (-0.155 +-0.019, 12/0; AUC ~0.53 vs ~0.69),
  `assoc_diff` -0.0034 +-0.0017 (8/4). `small`: `marg_mean` -0.0325 +-0.0055
  (12/0), discriminator distance -0.113 +-0.021 (11/1). Downstream TSTR
  unchanged (`tiny` +0.009 +-0.009, `small` +0.005 +-0.006). Privacy metrics
  flat: `frac_suspicious` `tiny` -0.002 +-0.004 (5/6), `small` +0.008 +-0.007;
  `exact_dup` 0; DCR ratio ~1.00.
- **The caveat that matters:** smaller models train far longer before the
  sensitivity rule stops them -- `small` 114 epochs, `tiny` 293 on average
  against a 300-epoch ceiling, i.e. `tiny` almost always ran to the ceiling
  (base: 31). So (a) size and training length are confounded in this matrix;
  (b) the tool's overfitting protection essentially never fired for `tiny`, yet
  the DCR-based privacy metrics stayed flat; (c) `tiny`'s `marg_mean` (0.029) is
  *below* the real-held-out-vs-train floor (mean 0.043): its output is closer to
  the training data than unseen real data is -- not by itself evidence of
  copying (exact duplicates 0, DCR ratio 1.007) but a reason for care; (d) it
  costs wall-clock (`fit_s` roughly 4x, confounded by machine load).
- **H4 learning rate -- prediction refuted.** `lr3e4`+warmup: `marg_mean` +0.0084
  +-0.0076 (3 better/9 worse), `assoc_diff` +0.0029 +-0.0014, TSTR -0.0176 +-0.0107,
  `frac_suspicious` +0.0109 +-0.0050; `lr1e4`: `marg_mean` +0.0144 +-0.0062 (3/9),
  `assoc_diff` +0.0042 +-0.0016 (1/11). The default 5e-5 is not under-training
  these models; higher LR is slightly worse and did not stop earlier.
- **H7 `gradient_accumulation_steps=1` -- no quality gain.** `marg_mean` -0.0034
  +-0.0059, `tail_err` -0.0133 +-0.0170 (neither detectable), `assoc_diff` +0.0021
  +-0.0015 (2/10 worse), TSTR -0.0094 +-0.0104; `frac_suspicious` +0.0164 +-0.0070
  (2 better/9 worse, ~2.3 s.e.). It stops ~6 epochs earlier (30.7 -> 24.7) since
  it takes 4x more updates per epoch. This does not support the earlier
  suggestion (status doc) that accumulation, via batch size, is why "small
  batches seem to work better" for quality.

**Implication:** Model size is the first lever in this program with a large,
uniform effect (12/0 on the headline fidelity metrics), at flat downstream
utility and flat privacy proxies. Not yet safe to recommend as a default:
(1) size vs training-length must be separated (planned M3: `tiny` capped at 30
epochs, `tiny` at 600, a smaller `micro`, `tiny` + higher LR); (2) it must be
confirmed on the two held-out datasets (wilt, churn2) that were kept back for
exactly this; (3) the memorisation caution above needs a direct check beyond DCR
(e.g. nearest-neighbour rank against held-out rows). Learning rate and
gradient accumulation stay at their defaults on this evidence.

---

## 2026-09-20 18:28 UTC — Adopted on `feat/support-seed-input`: vectorised constrained decoding, `top_k=0` for tabular sampling, `oov_strategy="unk"` + `unk_dropout=0.03`, and a fix for how seeded OOV values are returned

Code: merge `933e95c` (pushed), containing `b4d452d`/`bf249ff` (vectorised
decoding + any-order test), `06f37a7` (`top_k=0`), `120e4e5` (OOV defaults +
restore). Full suite on the merged result: 178 passed, 2 failed -- the same two
failures that pre-date this work (`test_default_init`, `test_TabularSampler`).
Owner approved the merge, the full test run, and the push explicitly.

**Question:** What changed on the integration branch, on what evidence, and what
does that do to the meaning of earlier results?

**What was done / Result:**
1. **Vectorised constrained decoding** (all tabular sampling, incl. any-order):
   0.22 s vs 76.5 s for 1,024 rows in an interleaved profile on a busy box; and
   identical results end to end (see the 15:43 UTC entry: 12 complete pipelines
   reproduce M1's baseline to 4 decimals). Relational sampling untouched.
2. **`TabularSampler.default_top_k = 0`** (was HF's implicit 50). Evidence:
   hicard 3/3 seeds, categorical error 0.142 -> 0.104, no privacy change; no
   detectable effect on the bundled low-cardinality data.
3. **`oov_strategy="unk"` + `unk_dropout=0.03` for tabular models.** Evidence:
   5 seeds, a held-out level; random substitution worse than ignoring the seed
   on 5/5, `unk`+dropout closer on 5/5 and at the unconditional floor; 3%
   dropout costs nothing detectable on unseeded generation (12 units).
4. **A bug the experiments did not show, found while making OOV a default:**
   with `unk`, a seeded value the model had never seen was decoded into the
   returned table as a literal `[UNK]` (categorical) or a corrupted `'[UNK]9'`
   string that turned a numeric column into `object`. (The old random
   substitution returned a plausible but wrong value instead.) `TabularSampler`
   now returns the caller's own value for those cells: per row for
   `sample_tabular_with_seed`; for the shuffled `sample_tabular` only when the
   caller gave one distinct value for the column, else missing. Tested for
   categorical, same-width numeric (dtype preserved), in-vocab (unchanged) and
   multi-row seeds; mutation-checked. Separately found and NOT fixed: a numeric
   seed wider than the training format (e.g. 9999 for a two-digit column) raises
   `KeyError: '0___NUMERIC___b_02'` under either strategy -- a pre-existing
   width problem, unrelated to [UNK].

**Provenance warning -- earlier numbers used different defaults.** Every M1/M2
result above was produced with `unk_dropout=0`, `oov_strategy="random"` and HF's
`top_k=50`. After this merge `{}` means the new defaults, so re-running an old
config no longer reproduces its number. `research/configs.py` keeps the old
meaning explicit: `b0` = `{unk_dropout: 0, oov_strategy: "random"}`, a `topk50`
sampling variant, and a note on the `default` variant. `base`, `qenc`, `small`,
`tiny`, `lr*`, `ga1` are NOT redefined -- to reproduce M1/M2 exactly, pass those
init/sample settings explicitly. (v2 is unchanged: no [UNK] dropout there, and
saved models keep `random` because the flag lives in the vocab.)

**Implication:** Model size (M2) remains a finding, not a default: M3 is still
running to separate size from training length and to confirm on held-out data.
Known limits of what was adopted: OOV was tested on one column of one dataset
with three distinct held-out levels; the [UNK]-dropout collator is v1 only; with
`train_size < 1` eval loss is computed with dropout too (documented in the
collator).

---

## 2026-09-20 19:48 UTC — M3: the M2 effect is "a small model trained to convergence", not size alone; it replicates on held-out wilt/churn2; direct memorisation check (M4) now running

**Partly superseded -- see the 2026-09-20 21:01 UTC entry below: the default model's LOADED checkpoint was epoch 5-16, not its stopping epoch, so the size comparison below was against an under-trained baseline. Original content unchanged.**

Code/data: `research/results/m3` (48 jobs) and `m3h` (18 jobs) at `c3876f5`. Provenance:
both ran with the OLD library settings (`unk_dropout=0`, `oov_strategy="random"`,
HF `top_k=50`) on the vectorised decoder, i.e. comparable with M1/M2; in `m3h`
`b0` is the legacy default model. Paired against M1's `base/default` (dev, 12
units) and against `b0` (held-out, 6 units).

**Question:** Is the M2 fidelity gain about model size, or about how long the small
model trains (it ran to the 300-epoch ceiling while the default stops near epoch
30)? And does it hold on datasets kept back for exactly this?

**What was done:** Dev, 4 datasets x 3 seeds: `tiny_e30` (128d/3L capped at ~30
epochs, matching base), `tiny_e600`, `micro_e600` (64d/2L), `tiny_lr3e4`
(lr 3e-4, 5% warmup). Held-out wilt and churn2 x 3 seeds: `b0`, `small`, `tiny`.

**Result (dev; arm - base, mean +-s.e., wins/losses of 12; M2's `tiny` for reference):**
- **Equal epochs, small model is far worse.** `tiny_e30`: TSTR 0.318 vs 0.753,
  `assoc_diff` 0.109 vs 0.021, `tail_err` 0.162 vs 0.059, discriminator distance
  0.394 vs 0.187 (AUC ~0.89), DCR ratio 1.90 -- under-trained, and far from the
  training data (`frac_suspicious` 0.012), not private-by-quality. Only
  `marg_mean` improves (0.055 vs 0.081, 11/1). So size alone does not explain M2.
- **More training than `tiny` at 300 does not help.** `tiny_e600` (sensitivity
  stopping fired at epoch 429 on average): `marg_mean` 0.0325 vs `tiny`'s 0.0288,
  discriminator distance 0.037 vs 0.032; `frac_suspicious` +0.0116 +-0.0043 vs
  base (1 better/9 worse, ~2.7 s.e.) -- a little more closeness with longer training.
- **Too small loses ground.** `micro_e600` (hit the 600 ceiling): best `marg_mean`
  (0.0240, -0.057 +-0.006, 12/0) but `assoc_diff` +0.0066 +-0.0029 (4/8),
  TSTR -0.012 +-0.008, discriminator distance 0.125 (`tiny`: 0.032).
- **A higher learning rate gets most of the way in a third of the epochs.**
  `tiny_lr3e4`: 109 epochs (`tiny`: 293), `marg_mean` 0.0471 (-0.034 +-0.006, 12/0),
  discriminator distance 0.068 (-0.119 +-0.021, 12/0), `assoc_diff` -0.0026 +-0.0014,
  TSTR +0.006, `frac_suspicious` +0.0025 +-0.0032; mean `fit_s` 490 vs base 851
  (load-confounded). Less good than `tiny` (0.029) but cheaper.
- **Held-out confirmation (wilt, churn2; 6 units, vs `b0`).** `tiny`: `marg_mean`
  0.0424 -> 0.0187 (-0.0237 +-0.0061, 6/0), `assoc_diff` 0.0165 -> 0.0096
  (-0.0069 +-0.0023, 5/1), discriminator distance 0.103 -> 0.013 (-0.090 +-0.016,
  6/0; AUC ~0.51), TSTR +0.007 +-0.004 (5/1), `frac_suspicious` -0.004 +-0.004,
  `exact_dup` 0; ran to ~301 epochs, `fit_s` +2,488 +-354. `small`: `marg_mean`
  -0.0075 +-0.0050 (5/1), discriminator distance -0.044 +-0.021 (5/1), rest
  undetectable. `tail_err` unchanged for both. `tiny` reaches the real-vs-real
  floor on `marg_mean` (churn2 0.017 vs floor 0.017; wilt 0.020 vs 0.025).

**Implication:** The mechanism reads as follows: the default large model is stopped
by the sensitivity rule near epoch 30 -- when its memorisation signal fires --
while its data is still easy to tell from real (AUC ~0.69 dev, ~0.60 held-out); a
small model learns slowly, needs hundreds of epochs, never trips that rule, and
ends with much better fidelity at unchanged downstream utility and unchanged
distance-based privacy proxies. This replicates out of sample. Costs: roughly
3-4x wall-clock for `tiny` (less for the higher-LR variant). Still open, and the
reason this is not yet a recommendation: (1) the stopping rule that protects the
default model is not what limits the small one, so privacy rests on proxies
(`frac_suspicious`, DCR ratio, exact duplicates) -- M4 adds a direct check,
`dcr_share` (share of synthetic rows whose nearest real neighbour is a training
row rather than an equal-size held-out set; calibrated: a copy of the training
rows scores 1.000, fresh real rows ~0.5); (2) 6 datasets, all small (768-10,000
rows). If M4 is clean the natural form is an opt-in preset (`tabular_config` +
epochs guidance), not a change to the default model.

---

## 2026-09-20 21:01 UTC — CORRECTION: the default recipe loads an epoch 5-16 checkpoint, far earlier than where it stops; against a fair checkpoint most of M2/M3's "small model" discriminator advantage disappears (6 runs; M5 running to settle it)

Code/data: `research/loaded_epoch.py`, `research/ckpt_compare.py`,
`research/results/loaded/` (6 fits: diabetes and insurance x seeds 0-2), pushed on
`exp/utility-optimization`. Legacy settings (`unk_dropout=0`, `oov_strategy="random"`,
`top_k=50`), teacher-forced target, as M1-M3.

**Question:** M1-M3 recorded where training STOPPED (~epoch 30 for the default
model). M3's interpretation was that the default model "is stopped by the
sensitivity rule near epoch 30 while its data is still easy to tell from real
(AUC ~0.69)". But the recommended recipe also passes
`load_from_best_mean_sensitivity=True`, which does not load the stopping weights.
Which checkpoint does it load, and how good is it?

**What was done:** Read the critic loop: that option loads the checkpoint whose
critic sensitivity is CLOSEST TO THE MEAN of the bootstrap null (`mean_best`), out of
the checkpoints saved every 5 epochs. One fit leaves four checkpoints on disk; I
loaded each into the same fitted pipeline and scored it with the bench protocol.

**Result (6 runs; each row is the mean over runs; paired s.e. vs `mean_best`):**

| checkpoint | mean epoch | `marg_mean` | discriminator AUC | TSTR | `dcr_share` |
|---|---|---|---|---|---|
| `mean_best` (what the recipe loads) | 9.5 | 0.090 | 0.691 | 0.812 | 0.517 |
| `best_disc` (latest under threshold) | 20.8 | 0.077 | 0.579 | 0.830 | 0.543 |
| `last_epoch` (weights at stopping) | 31.2 | 0.062 | 0.499 | 0.829 | 0.545 |

- The recipe loaded epochs 5.3, 5.3, 15.8, 10.3, 10.3, 10.3 while training stopped at
  epochs 31.6, 26.3, 31.6, 36.1, 30.9, 30.9.
- `last_epoch` vs `mean_best`, paired: `marg_mean` -0.029 +-0.008 (5 better/1 worse),
  discriminator distance from 0.5 -0.156 +-0.029 (6/0), TSTR +0.017 +-0.011,
  `assoc_diff` +0.008 +-0.003 (1 better/5 worse), `dcr_share` +0.028 +-0.026 (1/5).
- So the discriminator gap I attributed to the default model in M2/M3 (AUC ~0.69) is
  essentially the effect of the checkpoint the recipe loads; the same trajectory at
  its own stopping epoch is at AUC ~0.50.

**Implication:** (1) M2/M3's headline "small model reaches AUC ~0.53 vs the default's
0.69" compared a small model that trained to convergence against a default model
handicapped by its own checkpoint rule; the size effect on the DISCRIMINATOR is
therefore overstated, and the remaining size effect (marginals, associations) must be
re-measured against a fair baseline. The same rule also chose the small model's
checkpoint in M2/M3, so those arms are not clean either. (2) The recommended recipe
(`load_from_best_mean_sensitivity=True`, per DECISION_LOG) trades a lot of fidelity
for a lower `dcr_share` (0.517 vs 0.545, within noise at 154-268 test rows) -- the
privacy proxy is not measurably better while the fidelity cost is large. (3) M5
(`research/m5.py`: default vs small model x all four checkpoint rules, 4 dev + 2
held-out datasets x 3 seeds, 36 fits) is running to settle size vs selection rule.
Not yet a recommendation; the checkpoint rule may be the cheapest large win in the
whole program, but it rests on 6 runs from 2 datasets so far.

---

## 2026-09-20 21:59 UTC — c1 (fixed-epoch learning curves): EMA weights are a free gain (H16 confirmed); constraint-aware loss trains ~2x faster to the same ceiling; label smoothing minor; batch 32 = batch 8 x 4 on quality

Code/data: `research/curves.py`, `research/results/c1` at `141973b`. Protocol: plain training (no stopping rule)
to 100 epochs, checkpoint every 10, on diabetes / insurance / abalone x seeds 0-2 (9 paired units), small GPT2
(128d/4h/3L, lr 3e-4, 5% warmup), legacy settings, no teacher forcing. Reference arm `wk`. The EMA copies are
evaluated on the SAME trajectory as the raw weights (horizons ~1 and ~4 epochs).
Why fixed epochs rather than the sensitivity regime: the default sensitivity path silently dropped
`compute_loss_func` (fixed on `exp/constrained-loss`), and stopping noise would otherwise sit on every comparison.

**Question:** H16 (does weight averaging improve the model at a given step?), H17 (does computing the loss under the
per-column token mask sampling uses speed up learning?), H12 (label smoothing), H13 (does batch 32 x accum 1 match
batch 8 x accum 4 on quality?).

**Result (paired on 9 units; mean +-s.e.; wins/losses; lower better except TSTR):**
- **H16 EMA (horizon ~1 epoch) vs raw weights, same step.** epoch 30: `marg_mean` -0.0179 +-0.0033 (8/1),
  discriminator distance -0.0495 +-0.0080 (9/0), held-out NLL -0.160 (9/0), `assoc_diff` +0.0022 +-0.0012 (2 better/7 worse);
  epoch 50: `marg_mean` -0.0177 +-0.0033 (9/0), NLL -0.140 (9/0); epoch 100: no difference (converged). The EMA copy reaches
  the raw weights' FINAL marginal error at a median of epoch 30 vs 100 (in 89% of runs); best `marg_mean` along the curve
  0.0229 vs 0.0255. `dcr_share` within +-0.008 at every epoch. Horizon ~4 epochs: same gain from epoch 30 on but much worse
  at epoch 10 (TSTR -0.227 +-0.083, 0/9): the average lags while weights are still moving fast. Costs no extra training.
- **H17 constraint-aware loss vs standard.** epoch 10: `assoc_diff` -0.0297 +-0.0023 (9/0), discriminator distance -0.155
  +-0.014 (9/0), TSTR +0.145 +-0.044 (7/2), NLL -0.92 (9/0); epoch 30: discriminator distance -0.034 (9/0), `marg_mean`
  -0.008 (6/3); epoch 50: `marg_mean` -0.013 +-0.003 (9/0) but discriminator distance +0.021 (3/6), held-out NLL +2.1 (0/9).
  Reaches the reference's final `marg_mean` at a median of epoch 50 vs 100; best along curve 0.0240 vs 0.0255. The ceiling
  is not raised and it overfits sooner. Adding label smoothing (0.05) to it hurts TSTR from epoch 30 (-0.065 to -0.082, 1/8): dropped.
- **H12 HF label smoothing 0.05 alone.** small, consistent mid-curve gains (epoch 10 discriminator distance -0.034, 9/0;
  epoch 50 `marg_mean` -0.006, 8/1; `dcr_share` +0.024 +-0.010 at epoch 50); none at epoch 100. Not a lever.
- **H13 batch 32 x accum 1 vs batch 8 x accum 4** (same effective batch): no difference in any metric at any epoch
  (`marg_mean` within +-0.001). The speed side needs the throughput benchmark (`research/bench_train_speed.py`), not yet run.

**Implication:** Two independent, cheap ways to reach a given fidelity in fewer epochs: EMA (~3x, free) and the
constrained loss (~2x, opt-in, needs a stopping rule because it overfits sooner); neither raises the ceiling. They are
untested in combination and untested in the real sensitivity regime, where the checkpoint the recipe loads (see the 21:01 UTC
correction) matters more than the training length. Next: c2 (the remaining Program-2 ideas, each arm with its EMA copy),
and the real-regime M6 for the loss. Caveats: 3 small datasets; marginal error is the metric that moves, associations and
TSTR mostly do not; EMA at very early epochs is harmful for long horizons.

---

## 2026-09-21 00:04 UTC — M5 (final, 36 fits): model size and checkpoint rule are separate levers; the recipe's checkpoint rule costs a lot of fidelity; the small model's advantage is real but modest and costs 2-4x training time

Code/data: `research/m5.py`, `research/results/m5` at `639c476`. Same protocol as the 21:01 UTC correction: real sensitivity
regime, one fit leaves four checkpoints, each scored on the same sample protocol. Default GPT2 vs small GPT2
(128d/4h/3L, lr 3e-4), all six datasets x seeds 0-2 (18 paired fits; wilt and churn2 held-out). Legacy settings
(unk_dropout 0, oov random, top_k 50), teacher-forced target. Deterministic: the default arm reproduced the earlier
six-run comparison exactly for the same seeds.

**Question:** Once each model is taken at a sensible checkpoint, how much of M2/M3's small-model advantage remains,
and how much is the recipe's own checkpoint rule?

**Result (paired on 18 fits; mean +-s.e., wins/losses; lower better except TSTR; `dcr_share` 0.5 = no memorisation):**
- **Checkpoint rule, default model, vs the recipe's `mean_best` (loaded epoch ~12):** `last_epoch` (~33): discriminator
  distance -0.098 +-0.019 (16 better/2), `marg_mean` -0.016 +-0.006 (12/6), TSTR +0.010 +-0.004 (13/5), `assoc_diff`
  +0.004 +-0.002 (5 better/13 worse), `dcr_share` +0.031 +-0.010. `best_disc` (~23; what `load_from_best_mean_sensitivity=False`,
  the library default, loads): discriminator distance -0.066 +-0.015 (15/2), TSTR +0.011 +-0.004 (15/2), `marg_mean` -0.010
  +-0.005 (11/6), `dcr_share` +0.022 +-0.008. Mean `dcr_share`: 0.520 (`mean_best`), 0.542 (`best_disc`), 0.551 (`last_epoch`)
  -- fresh real data scores 0.50-0.535. So it is a trade: later checkpoints buy fidelity and utility at a modest, measurable rise in the
  closeness proxy. The small model shows the same direction with smaller sizes.
- **Model size (small - default) at each rule:** `marg_mean` -0.025 +-0.005 (18/0) at `mean_best`, -0.025 +-0.004 (17/1) at
  `best_disc`, -0.019 +-0.004 (15/3) at `last_epoch`; `assoc_diff` -0.004 to -0.006 (13/5 to 16/2); discriminator distance
  -0.098 +-0.014 (18/0) at `mean_best`, -0.068 +-0.009 (18/0) at `best_disc`, -0.029 +-0.011 (12/6) at `last_epoch`; TSTR ~0 at every
  rule; `dcr_share` within +-0.015. On the two held-out datasets alone (6 fits) every rule gives `marg_mean` -0.016 to -0.030 (all
  better), `assoc_diff` -0.006 (6/0), discriminator distance -0.06 (6/0).
- **Cost:** stopping epoch 32 vs 120 (median); wall-clock median 559 s vs 1,301 s on all datasets, 1,337 s vs 5,179 s on the
  held-out pair (load-confounded, but 2-4x).

**Implication:** (1) M2/M3's discriminator headline (-0.15) was mostly the recipe's checkpoint rule; against a fair checkpoint
the small model's discriminator advantage is -0.03 to -0.07, and it disappears for some datasets at the last epoch. (2) What survives,
including on held-out data, is a consistent modest gain in marginal error (~-0.02) and association error (~-0.005) at no change
in downstream utility or in the privacy proxies, for 2-4x more training. That supports an OPT-IN small-model preset for users who
want the fidelity and can pay the time; it does not support a default change. (3) The checkpoint rule is the larger lever, and it is a
fidelity-vs-closeness trade the owner should choose knowingly: DECISION_LOG recommends `load_from_best_mean_sensitivity=True`
(lowest closeness, lowest fidelity); the library default (`False`) gives ~-0.07 discriminator distance and +0.011 TSTR for
+0.022 `dcr_share`. (4) M4's calibrated memorisation check (with the recipe's checkpoint) found no detectable increase for the
small model (`dcr_share` 0.531 vs 0.523, +0.006 +-0.009, 14 units); M5 shows later checkpoints of it are closer (+0.027 vs
`mean_best`), so that reassurance applies to the recipe's checkpoint. Not tested: larger datasets than 10,000 rows.

---

## 2026-09-21 00:09 UTC — H15 / H15b closed: the self-referential likelihood-gap signals do not track overfitting as predicted, and the pre-registered stopping rule R* is FALSIFIED on held-out data

Code/data: `research/signals.py`, `signals_analyze.py`, `signals_rules.py`, `signals_prereg.py`; raw curves in
`research/results/{s1,s1h_d,s1h_w}` at `8696004`. Plain training with NO stopping rule, checkpoint every 5 (default) / 10 (small)
epochs; dev = 4 datasets x 3 seeds x {default, small GPT2} (24 runs, 120 / 300 epochs); held-out = wilt, churn2 x 3 seeds x 2 arms
(12 runs, 60 / 150 epochs -- shortened to save compute, see the flaw below). No teacher forcing, legacy settings.

**Question:** Can overfitting be detected WITHOUT held-out data by comparing, under the model itself, the per-row NLL of its training
rows with that of its own samples (`srlg_mean`, `srlg_ks`, `srlg_tail`), and can that drive a stopping rule?

**Result:**
- **The pre-registered H15 prediction is false.** Within-run Spearman of each signal with the TRUE generalisation gap (held-out minus
  train NLL) was predicted > 0.6 in most runs. Observed (dev, median [q25,q75]): `srlg_ks` -0.50 [-0.59,-0.39] for the default model
  (0% of runs > 0.6) and +0.47 [+0.43,+0.60] for the small model (25% > 0.6); `srlg_mean` -0.28 / +0.45; `srlg_tail` -0.01 / -0.28.
  With memorisation (`dcr_share`): `srlg_ks` -0.21 / +0.41. The sign flips between models, so it is not a usable monotone signal.
- **Why (a hypothesis, not tested):** once the model collapses onto its training rows its samples ARE training rows, so "training rows vs
  samples" shows no difference exactly when memorisation is worst; the signals rise then fall (e.g. diabetes default: `srlg_ks` 0.07 at
  epoch 5, 0.88 at 30, 0.11 at 120, while the true gap grew to 76 nats).
- **Retracted:** an early observation that `srlg_ks`'s minimum falls at the same epoch as held-out NLL's did not replicate on a second dataset.
- **Held-out likelihood is the wrong thing to stop on for synthetic data.** Its minimum is at epoch 5-8 (default) / ~40 (small); sample
  quality is best much later: discriminator-optimal epoch ~35-52 (default) / ~100 (small); marginal error is still improving at the end of
  every run. Stopping at the held-out-NLL minimum gave discriminator distance ~0.20 (dev, default) / ~0.12 (held-out, default).
- **H15b: R\* (stop when `srlg_ks` >= running min + 0.25) is FALSIFIED by its own pre-registered criteria** (held-out, 12 runs): (i) fires
  12/12 PASS; (ii) stop within x2 of the discriminator-optimal epoch in 50% (needed 75%) FAIL; (iii-a) mean discriminator distance at stop
  0.043 vs 0.024 at the last epoch FAIL; (iii-b) 0.043 vs 0.086 at the held-out-NLL minimum PASS; (iv) `dcr_share` 0.517 at stop vs 0.525 at
  last (needed >= 0.03 lower) FAIL. Falsification condition "(iii) fails" is met. By arm: default stops at a median epoch 20 vs a
  discriminator-optimal ~52 (ratios 0.36-0.45: too early), discriminator distance 0.065 at stop vs 0.031 at last vs 0.117 at the NLL minimum;
  small model stops at 80 vs ~115, distance 0.021 vs 0.017 vs 0.054, `dcr_share` 0.511 vs 0.510.
- **A flaw in my pre-registration that limits how (iii-a) and (iv) should be read:** I shortened the held-out runs (60 / 150 epochs), so
  "last epoch" there is close to the optimum, whereas on the 120 / 300-epoch dev runs it had already degraded (discriminator distance 0.13-0.14).
  The criteria were therefore easier for "last" than they were on dev. This does not rescue R\*: (ii) is a criterion about R\* itself and it also
  failed, and R\* stops early for the default model; it only means the comparison with "last epoch" was less informative than intended.
  Also: delta was tuned on the dev runs (scan over 0.05-0.5), and the dev advantage (e.g. discriminator distance 0.056 vs a hindsight-chosen fixed
  epoch's 0.050) did not carry over.

**Implication:** The label-free likelihood-gap idea does not deliver a stopping rule better than what exists, and I do not recommend it. What the
study did establish, on both dev and held-out data: (1) held-out-likelihood early stopping is inappropriate for synthetic data; (2) likelihood
overfitting, sample-quality overfitting and memorisation begin at very different epochs (5-8, ~35-100, and later), so "overfitting" has to be
specified before a detector can be judged; (3) the checkpoint the tool loads matters more than where it stops (M5). The tool's existing
label-free-of-held-out-data mechanisms (bootstrap sensitivity, CUSUM) remain the baselines; I have not compared R* with them in the real trainer
and no longer plan to. A new rule would need fresh seeds for validation, since all 36 curves have now been seen.

---

## 2026-09-21 01:34 UTC — M6 + M7 (real regime, 18 paired units each): weight averaging (EMA) is a robust, free improvement that matches the small model's fidelity gain; the constraint-aware loss is a minor lever

Code/data: M6 raw results at `429dc8d` on `exp/constrained-loss` (library `a0f20e9`); M7 at `e777d8d` on `exp/ema-weights` (library `f1da71c`, 183 tests
passed / 2 known failures). Protocol = M5: real sensitivity regime, four checkpoints per fit, all six datasets x seeds 0-2, default and small GPT2, legacy
settings, teacher-forced target; each arm paired with M5's identical-seed run of the same model without the option (the default arm reproduced earlier runs
exactly, so the baseline is deterministic). M6 = `constrained_loss=True`; M7 = `ema_horizon=1.0`. (Two earlier problems: a first M6 attempt crashed after
training because its worktree lacked a metric -- fixed and smoke-tested before the rerun -- and some jobs were lost to GPU out-of-memory from over-subscription.)

**Question:** Do the two fixed-epoch efficiency findings (c1: constrained loss ~2x faster to the same ceiling; EMA reaches the raw weights' final marginal error
~3x sooner) survive the real stopping regime and the checkpoint the tool actually loads?

**Result (paired on 18 units; mean +-s.e., wins/losses; lower better except TSTR):**
- **M7 EMA, default model, vs no EMA:** `marg_mean` improves at EVERY checkpoint rule: `mean_best` -0.039 +-0.005 (18/0), `best_disc` -0.029 +-0.004 (16/0),
  `not_best` -0.030 +-0.003 (18/0), `last_epoch` -0.027 +-0.003 (18/0). Discriminator distance: -0.128 +-0.012 (18/0) at `mean_best`, -0.072 +-0.012 (16/0) at
  `best_disc`, -0.052 (14/4) at `not_best`, -0.015 +-0.017 (11/7, not significant) at `last_epoch`. `assoc_diff` -0.002 to -0.004 (10-14 of 16-18 better); TSTR within
  +-0.010; `dcr_share` within +-0.010; stopping epoch 30 vs 32. **Held-out wilt + churn2 (6 fits): `marg_mean` -0.021 to -0.031 (6/0 at every rule), discriminator
  distance -0.074 to -0.092 (6/0 at every rule), `assoc_diff` -0.002 to -0.005.**
- **M7, small model:** `marg_mean` -0.011 to -0.020 (16-18 better at every rule), discriminator distance -0.036 +-0.009 (16/2) at `mean_best` but +0.009 to +0.011
  (not significant, 6/12) at later checkpoints; TSTR and `dcr_share` unchanged. (An earlier read at n=7 suggested EMA hurts the late-checkpoint discriminator score;
  at n=18 that is not supported for the default model and only a non-significant trend for the small one.)
- **Default+EMA vs the small model without EMA:** `marg_mean` -0.013 +-0.004 (15/3) at `mean_best`, -0.005 (11/5) at `best_disc`, -0.005 (13/3) at `not_best`, -0.008
  (14/4) at `last_epoch` -- i.e. the big model with free averaging is as good or better on marginals than a model that needs 2-4x the training; the small
  model keeps a slight association-error edge (`assoc_diff` +0.001 to +0.003 for default+EMA). **Small+EMA vs default:** `marg_mean` -0.030 to -0.045 (18/0 at every
  rule), `assoc_diff` -0.003 to -0.007 (13-16 of 17-18), discriminator distance -0.058 +-0.013 (15/3) at `best_disc`; TSTR and `dcr_share` unchanged.
- **M6 constrained loss, default model:** `mean_best` `marg_mean` -0.022 +-0.005 (17/1), discriminator distance -0.072 +-0.011 (17/1); `best_disc` -0.014 (15/3),
  -0.031 (14/4) but TSTR -0.013 +-0.005 (3 better/15 worse); `last_epoch` `marg_mean` -0.009 (14/4), `assoc_diff` -0.004 (13/5), others n.s. Stopping epoch 30 vs 32.
  Small model: `marg_mean` -0.003 to -0.008, TSTR -0.005 to -0.008 (4-8 better of 16-18), stops at epoch 82 vs 120 (~30% fewer epochs).

**Implication:** EMA is the strongest cost-benefit finding of the program: one line of configuration, no extra training (stopping epoch unchanged; per-step
overhead is one fused multiply-add over the parameters), a consistent marginal-fidelity gain at every checkpoint rule that replicates on held-out data, and no
measurable cost to utility or the privacy proxies. It is worth adopting; that is a change to a default and is the owner's call (`exp/ema-weights`, opt-in for
now, `ema_horizon=1.0`). The constraint-aware loss is a niche efficiency option (fewer epochs for the small model) with a small utility cost at
`best_disc`; leave it opt-in. Untested: EMA together with the constrained loss; horizons other than ~1 epoch in the real regime (4 epochs lagged badly in c1 at epoch
10); datasets above 10,000 rows; the wall-clock overhead (per-step cost is small, but matrix wall-clock is load-confounded, so no timing claim is made).

---

## 2026-09-21 01:41 UTC — c2 (117 runs) and M4 final (54 fits): column order, numeric representation and regularisation do not beat the defaults; EMA replicates across 14 training variants; the small model shows no detectable memorisation increase

Code/data: `research/results/c2`, `research/results/m4` at `cae36fb` (all jobs complete; 5 `c2` jobs and 1 M4 job that had failed on GPU out-of-memory were rerun
with identical arguments). `c2`: fixed-epoch learning-curve protocol (see the c1 entry), 13 arms x 3 datasets x 3 seeds, 100 epochs, every arm also records
an EMA copy (horizon ~1 epoch); reference = c1's `wk` (same seeds, splits, LR schedule). M4: real sensitivity regime, `b0` (default GPT2) vs `tiny`
(128d/3L) vs `tiny` at lr 3e-4, all six datasets x 3 seeds, legacy settings, scored with the calibrated `dcr_share`.

**Result -- c2 raw arms vs `wk` at epoch 100 (9 units; mean +-s.e., wins/losses):**
- **H11 column order (pre-registered: hub-first / entropy-ascending improve `assoc_diff`): not supported.** reverse, hub-first, entropy-ascending, random:
  `marg_mean` within +-0.0025 of the original order (all not significant); `assoc_diff` +0.001 to +0.004 (hub-first +0.0034 +-0.0027, random +0.0042 +-0.0016, 2/7);
  TSTR -0.008 to -0.018 (reverse -0.018 +-0.008, entropy-ascending -0.014 +-0.006, both 2 better/7 worse). No order helps; a few slightly hurt utility.
- **H14 numeric representation (pre-registered: neutral-to-better): refuted.** quantile encoding: `marg_mean` +0.0060 +-0.0019 (0 better/9 worse), `assoc_diff` +0.0048
  (2/7), TSTR -0.0215 +-0.0068 (2/7); precision 3: +0.0099 +-0.0021 (1/8), TSTR -0.0216; `numeric_nparts=2` (only reachable since `top_k=0`): +0.0117 +-0.0024 (0/9),
  `assoc_diff` +0.0150 +-0.0029 (1/8), TSTR -0.0815 +-0.0310. `numeric_categorical_threshold=20`: no detectable effect (only 3 of 9 units have a column it changes).
- **H12 regularisation:** weight decay 0.05: nothing (`marg_mean` +0.0001 +-0.0003). Dropout acts as a speed/overfitting dial: 0.0 learns faster (discriminator
  distance -0.032 +-0.008 at epoch 30, 8/0) but ends worse (+0.040 +-0.009 at 100, 0/9); 0.2 is slower early (epoch 30: discriminator +0.051, TSTR -0.043, 0/8) and better
  late (-0.027 +-0.008, 8/1, at 100; `marg_mean` +0.002). The default 0.1 is a sensible middle.
- **H16 EMA, pooled over ALL 14 arms of c1+c2 (126 runs; EMA minus raw weights on the same trajectory):** `marg_mean` -0.0144 +-0.0009 at epoch 30 (better in 88% of runs) and
  -0.0143 +-0.0009 at epoch 50 (92%); improves at epoch 30 in 14 of 14 arms (per-arm -0.006 to -0.025); discriminator distance -0.032 +-0.003 at epoch 30 (83% of runs);
  no effect at epoch 100 (converged); `dcr_share` unchanged. The cost is confined to the first ~10 epochs, where the average lags: `assoc_diff` +0.0062 +-0.0007 and TSTR -0.042
  +-0.008 at epoch 10.
- **M4 final (18 paired fits, all six datasets), vs default:** `tiny`: `marg_mean` -0.0420 +-0.0057 (18/0), discriminator distance -0.1345 +-0.0153 (18/0), `assoc_diff`
  -0.0043 +-0.0013 (13/5), TSTR +0.0082 +-0.0059 (13/5), `exact_dup` 0, **`dcr_share` 0.522 vs 0.518 (+0.0038 +-0.0079)**, stop epoch 296 vs 33, `fit_s` 3,613 vs 652 (load-confounded).
  `tiny` at lr 3e-4: `marg_mean` -0.0269 (17/1), discriminator distance -0.0991 (17/1), `dcr_share` 0.531 vs 0.518 (+0.0125 +-0.0064, 1.9 s.e.), stop epoch 140, `fit_s` 2,184.

**Implication:** Of everything tried in Program 2 except EMA, none of column order, numeric re-encoding, label smoothing, dropout, weight decay or the loss variants gives a robust gain over
the defaults; several of my predictions (H11 hub-first, H14 neutral-to-better) were wrong, and quantile/coarser numeric encodings are worse at convergence on these datasets
(consistent with M1's mixed quantile-encoding result). EMA is the exception and now replicates across every training variant tried. The memorisation caution about the small
model is answered for the recipe's checkpoint: the calibrated direct measure shows no detectable increase (both readings within the 0.50-0.535 range fresh real data scores);
M5 showed later checkpoints are closer, so that statement is about the recipe's checkpoint. Limits: fixed-epoch protocol with a small GPT2 on three small datasets; `c2` arms were
not run in the real stopping regime.

---

## 2026-09-21 02:14 UTC — H13 confirmed: batch 32 x accumulation 1 trains 3.2-3.5x faster and fits 1.9-2.7x faster end to end, with no consistent quality difference

Code/data: `research/bench_train_speed.py` (log `research/speed_bench.log`), `research/m8.py`, `research/results/m8` (32 fits, none failed) at `827fea0`; quality
also from c1 (`wk_bs32`, 9 units). All at the SAME effective batch (32) and the same number of optimizer steps per epoch; only the split between per-step batch and
accumulation differs.

**Question:** The default trains with batch 8 x gradient accumulation 4 (effective 32) on a GPU these models under-fill. Does batch 32 x accumulation 1 give the same
model faster, and how much of a whole fit does that save?

**What was done:** (a) Controlled training-throughput micro-benchmark on adult5k: 7 configurations interleaved round-robin, 3 repetitions each, 60 optimizer steps (first 15
excluded), CUDA-synchronised timing, on a quiet box. (b) M8: the real sensitivity regime end to end, 4 datasets x 2 seeds x {default, small} x {8x4, 32x1}; the four arms of each
(dataset, seed) group ran together on ONE GPU so they shared the same contention, and the comparison is within groups (16 paired groups).

**Result:**
- **Training throughput (rows/s; small / default GPT2):** 8x4 fp16 (current) 715 / 558; **32x1 fp16 2,473 (3.46x) / 1,781 (3.19x)**; 32x1 bf16 3.59x / 3.11x; 32x1 fp16 + fused AdamW
  3.83x / 3.30x; 32x1 bf16 + fused AdamW 4.20x / 3.61x; 32x1 bf16 + `torch.compile` 6.20x / 5.26x; batch 64 (a different effective batch) 7.51x / 6.23x. Repetition ranges are tight
  (e.g. 2,444-2,498 for 32x1 fp16 small).
- **End to end (M8), fit time 8x4 / 32x1 (geometric mean over paired groups):** default model **1.85x** (n=8, range 1.52-2.30); small model **2.71x** (n=8, range 2.53-3.04);
  all groups 2.24x; total wall-clock 4,485 s vs 1,882 s (2.38x). Below the training-only 3.2-3.5x because the critic rounds (sampling + bootstrap) are unchanged.
- **Quality, 32x1 minus 8x4 (mean +-s.e. over 16 groups; better/worse for 32x1):** `marg_mean` at the recipe's checkpoint -0.0055 +-0.0040 (10/6), at `best_disc` +0.0023 +-0.0030 (7/9), at the last
  epoch +0.0043 +-0.0024 (6/10); discriminator distance -0.0189 +-0.0099 (9/7) / +0.0029 +-0.0073 (7/9) / -0.0076 +-0.0051 (9/7); TSTR -0.0009 (9/7) / -0.0067 +-0.0037 (10/6); `dcr_share` -0.0059 +-0.0076.
  Mixed signs, all within ~2 s.e.: no consistent difference. The stopping epoch differs by +2.1 epochs on average and is identical in 50% of groups -- the two settings are not numerically
  identical (different RNG streams and reduction order), so the noisy critic sometimes stops at a different round, in either direction. c1's fixed-epoch curves (9 units, no stopping rule) agree:
  `wk_bs32` within +-0.001 on `marg_mean` at every epoch.

**Implication:** `batch_size=32` with `gradient_accumulation_steps=1` is a free 1.9-2.7x wall-clock reduction for a fit, with no measurable quality change; it needs ~4x the per-step activation
memory (so a smaller GPU may need the old split), which is why it is a user setting and not a silent change. `bf16` and fused AdamW add ~10-20% each on top. `torch.compile` gives the largest
extra gain in the micro-benchmark but is NOT recommended without care: the critic loop rebuilds the Trainer every `n_critic` epochs and may recompile each time (untested end to end), and the
compile warm-up was excluded from the timing. Not measured: datasets above 10,000 rows, multi-GPU, and the load-free absolute times (M8 groups shared a busy box, so only ratios are meaningful).

---

## 2026-09-21 02:39 UTC — Adopted on `feat/support-seed-input`: weight averaging on by default, tabular batch 32 x 1, the sensitivity-path fix, opt-in constrained loss, and the research harness (owner: "merge all that improves the solution")

Code: merges of `exp/utility-optimization`, `exp/constrained-loss` (`a0f20e9`) and `exp/ema-weights` (`f1da71c`), then the default changes. Full suite on the result: 189 passed, 2 failed
(`test_default_init`, whose assertions were already stale -- `evaluation_strategy` was renamed `eval_strategy` and it expects 100 epochs -- and `test_TabularSampler`'s NaN failure).
The owner approved the merges, EMA as the default, and batch 32 x 1 as the default, explicitly.

**What changed and why (evidence in the entries above):**
1. **Sensitivity-path fix (bug):** `fit()` forwards `field_weights`, `compute_loss_func`, `predict_fields`, `digit_entropy_weighting` to the default training path (they were silently dropped unless
   `n_critic=0`); a test fails with `KeyError` on the old code. Anyone who passed those arguments under the default regime got no effect before this.
2. **`ema_horizon` defaults to 1.0 for tabular models** (M7: `marg_mean` -0.027 to -0.039 at every checkpoint rule on the default GPT2, 18/0; held-out 6/0; no utility or privacy-proxy change;
   c1/c2: 14 of 14 training variants). Safeguards: switched off automatically with `overfitting_detection_method="cusum"` or an `objective_callback`, and only an EXPLICIT request there raises.
3. **Tabular `batch_size` defaults to 32 with `gradient_accumulation_steps=max(1, round(32/batch_size))`** (M8: 1.85x-2.71x faster fits, no consistent quality difference). Effective batch stays ~32:
   an explicit `batch_size=8` still gives 8 x 4; relational keeps 8 x 4 (not measured); `REaLTabFormer2` is unchanged. Needs ~4x the per-step activation memory -- lower `batch_size` on a small GPU.
4. **Opt-in `constrained_loss`** (a minor lever; off by default) and the research harness, notebook and raw results.

**Provenance / what this does to earlier numbers:** all results above this entry were produced with weight averaging OFF and batch 8 x 4; the research scripts now pin those explicitly (see the
provenance note in `research/HYPOTHESES.md`). Behaviour change for users: models trained with the defaults now return the AVERAGED weights (a different model from before), and the default
tabular per-step memory is ~4x higher.

**Known limits:** the evidence covers datasets up to 10,000 rows and single-GPU training; EMA together with the constrained loss is untested; `torch.compile`/`bf16` were measured but not adopted.
