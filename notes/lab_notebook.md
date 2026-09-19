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
