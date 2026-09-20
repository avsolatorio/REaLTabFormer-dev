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
