# REaLTabFormer Optimization — Decision Log

Goal (user directive): iterate toward the most optimal REaLTabFormer
configuration for synthetic tabular data generation. Every decision below
must be grounded in (a) theory — a stated mechanism for why it should help
— and (b) experiment — real measurements on the real UCI Adult Income
dataset (not synthetic toy data), using the project's own evaluation
methodology (`rtf_analyze.py::SyntheticDataBench`: KS statistic,
Wasserstein distance, categorical TVD, correlation preservation,
discriminator two-sample test, TSTR ML efficiency).

All work happens in worktrees under
`/private/tmp/claude-502/.../scratchpad/*/worktree`, gets committed, then
merged into the shared worktree at `scratchpad/repro/seed_input_merged`
(currently at commit `93ddffb`, branch `feat/support-seed-input`, 3 commits
ahead of and pushed to `origin/feat/support-seed-input` as of this log's
start). Nothing gets pushed without explicit user confirmation — that
policy stands even while iterating autonomously.

Adult dataset harness (reused across all experiments for comparability):
3000-row stratified-by-index subsample of the full 45,222-row cleaned
Adult dataset (dropped `education` as redundant with `education-num`,
dropped `native-country` for cardinality), 2400 train / 600 test split via
`SyntheticDataBench`, `random_state=1029` throughout.

---

## D1 — `numeric_quantile_encoding`: CDF-based numeric representation

**Decision:** Represent numeric column values by their empirical quantile
position (`q = F(x)`, via `sklearn.preprocessing.QuantileTransformer`
breakpoints, `np.interp` for all forward/inverse mapping) instead of
fixed-decimal-precision formatting.

**Theoretical grounding:** Probability integral transform — `F(X)` is
exactly `Uniform(0,1)` for continuous `X`, regardless of `X`'s shape. Fixed
absolute-precision formatting manufactures near-constant leading digit
chunks for heavy-tailed columns (magnitude-alignment padding); quantile
encoding is provably shape-agnostic in the resulting per-position entropy.
Compared against magnitude+mantissa encoding beforehand (log-shape
dependent, worse on bimodal data) and rejected that alternative on
theoretical grounds before implementing either.

**Experimental grounding:**
- Per-position entropy verified through the real `process_data` pipeline:
  fractional digit positions reach H_norm=0.986–1.000 vs 0.002 for the
  untransformed leading digit of the old encoding.
- Adult dataset, `fnlwgt` (heavy-tailed, continuous): KS 0.041→0.021,
  Wasserstein/mean 0.052→0.013 vs fixed-width baseline.
- `age` (already well-behaved): no regression (0.017→0.014).
- Discriminator AUC and TSTR: both improved slightly.

**Status:** VALIDATED, shipped (commit `c3ca91b`), pushed.

---

## D2 — Boundary-precision fix for quantile encoding

**Decision:** Floor every fitted quantile-position breakpoint to the
`numeric_precision` decimal grid before storing, then apply a monotonic
non-decreasing pass.

**Why this surfaced:** Found via the explicit discipline of testing on a
*real* dataset, not just synthetic data built to exercise the feature —
`capital-loss` (95.5% exact zeros) decoded almost every zero-valued row to
a wrong nonzero value (KS 0.954, essentially total failure), while every
other column in the same fit worked correctly.

**Root cause (proven, not guessed):** A value that recurs across many rows
collapses a run of fitted quantile positions onto one `np.interp`-assigned
position (the run's right edge). That raw float
(`0.954954954954955` in the reproduction) isn't exactly representable at
4 decimal digits; formatting-then-parsing rounds it to `"0.9550"`, which
lands *past* the true boundary (`0.955955955955956`) into the next
segment. Confirmed by hand: decoding the rounded value gives 0.585
instead of 0.0; decoding the exact unrounded boundary gives 0.0 correctly.

**Alternatives considered and rejected (with reasoning, not just listed):**
1. Explicit zero-inflation split (presence flag + quantile-encode
   remainder) — the "textbook" hurdle-model fix, but a materially larger
   change (new generation step, touches tokenization/sampling/seed_input
   in both v1 and v2) for what was, at that point, purely a correctness
   bug. Deferred, not rejected outright — became D3.
2. Detect-and-warn only — doesn't fix anything, insufficient alone.
3. **Chosen: global floor-to-grid + monotonic clamp.** Smallest blast
   radius (one function, fit-time only, decode code untouched), most
   general (fixes *any* number of point masses / mid-distribution ties /
   top-coded ceilings with no extra logic, not just the one case that
   happened to surface it), and directly targets the proven mechanism
   rather than a symptom.

**Experimental grounding:** Reproduction case: zero-row exact-decode rate
0%→100%. Full Adult rerun: `capital-loss` Wasserstein/mean 0.056
(baseline) → 0.265 (broken) → 0.012 KS / correct zero-fraction (fixed).
Regression test added, confirmed to fail against pre-fix code and pass
against the fix (not a vacuous test).

**Status:** VALIDATED, shipped (commit `35eee52`), pushed.

---

## D3 — Point-mass resolution reclaim

**Decision:** When a single value recurs in ≥5% of a column's rows
(`_POINT_MASS_THRESHOLD`), excise it from the `QuantileTransformer` fit
entirely and reserve it one breakpoint sized to its true frequency and
rank, instead of letting it consume a proportional share of the finite
`n_quantiles` breakpoint budget.

**Theoretical grounding:** D2 made point-mass columns *correct* but not
*efficient* — a 95.5%-zero column still spent 955 of 1000 breakpoints
re-describing the same repeated value, leaving ~45 for the part of the
distribution that actually varies. Literature check (WebSearch, mid-session):
this is structurally the same idea as the classical hurdle model
(Bernoulli gate + zero-truncated distribution for the positive part) —
confirms the direction is sound, implemented here as a token-level
variant with no new generation step, rather than the architecturally
heavier "new column" framing of hurdle models in the DL literature.

**Experimental grounding:** Reproduction: point-mass breakpoints
955/1000→1/109; nonzero-tail round-trip relative error ~0 (from a real,
measurable gap) since resolution now exceeds the remainder's own
cardinality. Adult dataset, `capital-loss` Wasserstein/mean:
0.265 (D2-only) → **0.042 (better than the 0.056 fixed-width baseline)**.
`capital-gain` (also zero-inflated) improved too, without any
column-specific code — the detection is generic.

**Known, out-of-scope finding surfaced along the way:** a *fully* constant
numeric column (every value identical) produces a malformed formatted
string (`"1."` instead of `"1.0000"`) under quantile encoding. Confirmed
pre-existing (reproduces against the commit before D1 even). Not something
a real point-mass column hits (there's always some variation in the
remainder) — a user with a literally-constant column should route it
through `numeric_categorical_threshold` instead. Logged as a known gap,
not fixed (would be scope creep against the actual ask); a
does-not-crash regression test pins current behavior.

**Status:** VALIDATED, shipped (commit `ceb84d3`), pushed.

---

## D4 — `any_order` × `digit_entropy_weighting` composition test

**Decision:** Add an end-to-end test fitting a model with both flags on.

**Why:** Neither feature's test suite exercised the other. The low-level
mechanism (does `token_weights` survive the any-order collator's
per-batch column permutation correctly?) was *already* unit-tested from
`any_order`'s own development — but "the pieces look orthogonal reading
the code" is exactly the kind of claim this session's own track record
says to verify, not trust (this session's own D2 was a bug of precisely
that shape: correct-looking pieces, never run together).

**Result:** Passed on first run — chunk_significance_weights compute
correctly, unconditional sampling works, arbitrary-subset seeding
(the feature's actual point) preserves seeded values correctly under
both entropy weighting and any-order permutation at once.

**Status:** VALIDATED (mechanically — see D5/D7 for whether either
feature's *effect* is actually good), shipped as a test (commit
`93ddffb`), pushed.

---

## D5 — `digit_entropy_weighting` quality evaluation (fixed-width numeric
encoding) — NEGATIVE FINDING

**What was tested:** Adult dataset, REaLTabFormer v1, `numeric_nparts=2`
(multi-chunk so there's something to weight), `numeric_quantile_encoding=
False` (fixed-width — the representation entropy weighting was designed
to help), baseline (uniform loss weight) vs treatment
(`digit_entropy_weighting=True`), 15 epochs, otherwise identical.

**Result: net negative on this test.** Most marginal-fidelity metrics got
*worse* under entropy weighting, not better: `fnlwgt` KS 0.072→0.081,
`capital-loss` KS 0.008→0.014 (Wasserstein 0.172→0.323), `education-num`
KS 0.041→0.053. Discriminator AUC went from 0.582→0.596 (more
detectable). TSTR dropped slightly (0.890→0.878 vs 0.905 ceiling). One
real improvement: `corr(age, hours-per-week)` error dropped 0.130→0.085.
The mechanism itself worked exactly as designed
(`fnlwgt` chunk weights: `[0.708, 1.095, 1.096, 1.101]` — leading chunk
correctly down-weighted relative to trailing) — the *design*, not the
implementation, is what's in question.

**Working hypothesis (flagged as hypothesis, being tested in the
in-flight bake-off — see below):** entropy (information content) and
cost-of-error aren't the same thing for positional digit chunks under
fixed-width formatting. A leading (high place-value) chunk is low-entropy
*because* it's almost always the same digit — but when the model gets it
wrong, that's an order-of-magnitude error in the decoded value, much
larger than a trailing-chunk mistake. Down-weighting its loss removes
gradient pressure from exactly the chunk whose mistakes are most
expensive. Entropy weighting may be measuring the wrong thing for
*this specific representation* (fixed-width, place-value-coupled
chunks) while being fine or even helpful for a representation where
chunks don't have that place-value asymmetry (quantile-encoded columns,
where every chunk is a digit after a decimal point of broadly similar
scale) — untested until the bake-off below.

**Status:** REAL FINDING, not yet acted on. Do not recommend
`digit_entropy_weighting` for fixed-width columns by default pending
further evidence. Currently being re-tested *combined with* quantile
encoding (config C of the in-flight bake-off) to see if the hypothesis
holds — if entropy weighting is neutral-or-positive there, the practical
recommendation becomes "only combine with quantile encoding, not
fixed-width" rather than "redesign the weighting scheme." If it's still
negative even under quantile encoding, that's a stronger signal the
scheme itself needs the magnitude-aware correction sketched above, which
would then become a real follow-up implementation task, not yet started.

---

## D6 — Fair `any_order` cost/benefit test — CONFOUNDED first attempt

**First attempt (Adult, `shared_numeric_vocab=True`, fixed-width numeric
encoding, 15 epochs, `any_order=False` vs `True`) was invalidated, not
just "inconclusive":** *both* conditions came back catastrophically bad
(KS up to 0.95, discriminator AUC 0.91–0.98) — nowhere near the quality
tier of every other Adult experiment this session (KS 0.01–0.04,
discriminator ~0.55). Final train losses (0.99 fixed-order, 1.81
any-order) were far higher than converged runs elsewhere (~0.6–0.7),
consistent with under-convergence on a harder combined objective
(`shared_numeric_vocab` pooling + fixed-width heavy-tailed columns, and
for the treatment, + any-order column permutation) rather than either
configuration being fundamentally broken. **Decision: do not report this
as an "any_order costs quality" finding — the comparison is confounded by
an under-trained, harder-than-necessary representation, not isolating
`any_order` itself.**

**Re-running now (in flight):** same comparison with
`numeric_quantile_encoding=True` (removes the fixed-width confound) and
`epochs=30` (more convergence budget). This is the theoretically correct
way to isolate `any_order`'s own cost: hold every other variable at its
best-known setting (per D1–D3) and vary only the one flag being evaluated.

**Status:** IN PROGRESS. See results section below as they land.

---

## D7 — Systematic bake-off (in flight)

Four configs, same data/split/architecture/epoch budget (v1
REaLTabFormer, 25 epochs — bumped from 15 for better convergence
confidence, chosen because v1 trains fast enough to afford it):

- **A. BASELINE** — nothing on.
- **B. QUANTILE** — `numeric_quantile_encoding=True` only (includes the
  point-mass reclaim from D3, since it's built into the same flag).
- **C. QUANTILE+ENTROPY** — B + `digit_entropy_weighting=True`. Direct
  test of the D5 hypothesis.
- **D. QUANTILE+CAT_THRESH** — B + `numeric_categorical_threshold=20`
  (demotes only `education-num`, confirmed 16 unique values vs every
  other numeric column's 21–21648).

**Status:** IN PROGRESS. See results section below as they land.

---

## D8 — Contingency plan for D5's follow-up (not yet triggered)

Written now, before config C's result is known, so the decision rule is
fixed in advance rather than rationalized after seeing the number.

**If config C (QUANTILE+ENTROPY) is neutral-to-positive** relative to
config B (QUANTILE alone): the D5 hypothesis is confirmed —
entropy-weighting's problem is specifically the place-value asymmetry of
*fixed-width* chunks, and quantile-encoded chunks don't have that
asymmetry (they're all digits after a decimal point of similar scale).
**Chosen action: a policy/documentation recommendation** ("only enable
`digit_entropy_weighting` alongside `numeric_quantile_encoding`, not
fixed-width formatting"), not a code change. This is the *more optimal
and efficient* resolution when it's sufficient — it requires no new
mechanism, no new failure surface, and is fully explained by evidence
already in hand. Building a magnitude-aware weighting scheme (sketched
below) would be solving a problem that no longer exists once the simpler
fix (always pair with quantile encoding) removes it.

**If config C is still negative:** the problem isn't fixed-width-specific
after all, and a real code fix becomes justified. Sketch, not yet
implemented: weight chunks by
`entropy_normalized_i * place_value_i^alpha` instead of pure entropy,
so a chunk that's low-entropy *and* high place-value (rare-but-expensive
errors) doesn't get its gradient pressure removed. Requires computing
each chunk's place value from its column's `mx_sig`/`zfill`/`ljust`/
`numeric_nparts` (already in `transform_data`), a real implementation
task (not attempted unless config C's result actually requires it).

**Efficiency principle applied here, stated explicitly per the user's
directive:** prefer the cheapest change that's fully explained by the
evidence in hand; only reach for a more complex mechanism when a simpler
one demonstrably isn't sufficient. Don't build D8's fallback unless D5/C
proves the simpler policy fix wouldn't have worked.

---

## Results as they land

### [2026-08-31 00:xx] D6 deconfounded any_order result — MAJOR FINDING, redirected investigation

The deconfounded rerun (`numeric_quantile_encoding=True`, `epochs=30`,
otherwise identical: `shared_numeric_vocab=True`, `any_order` False vs
True) did **not** clear up — it's still catastrophically bad, in some
columns *worse* than the first (confounded, 15-epoch, fixed-width) run:

| | FIXED-ORDER | ANY-ORDER |
|---|---|---|
| `age` KS | 0.405 | 0.850 |
| `fnlwgt` KS | 0.500 | 0.781 |
| `education-num` KS | 0.770 | 0.834 |
| `capital-gain` KS | 0.069 | 0.070 |
| `capital-loss` KS | 0.048 | 0.028 |
| `hours-per-week` KS | 0.797 | 0.832 |
| discriminator AUC | 0.977 | 0.952 |
| TSTR ratio | 0.724 | 0.804 |

**Key observation that redirects the whole investigation:** `capital-gain`
and `capital-loss` are fine (KS ~0.03–0.07) in *both* conditions, while
`age`, `fnlwgt`, `education-num`, `hours-per-week` are badly broken in
*both*. That's not noise — capital-gain/loss are the two columns D3's
point-mass reclaim turns into an easy, near-degenerate target (a single
constant string >90% of the time); every other numeric column here is
genuinely continuous/variable. An undertrained model trivially "solves"
the degenerate columns and fails the real ones — consistent across both
`any_order` settings.

**Comparing across ALL experiments run so far, the actual isolated
variable becomes obvious:** the bake-off's config A and B (below) — v1
REaLTabFormer, **no** `shared_numeric_vocab`, same 2400-row Adult subset,
same 6 numeric + 6 categorical columns, only 25 epochs — converged
cleanly (KS 0.01–0.03, discriminator ~0.53–0.55). Every
`shared_numeric_vocab=True` run in this session, regardless of
`any_order`, quantile encoding, or epoch budget (15 or 30), has been
catastrophic. **The common factor across every broken run is
`shared_numeric_vocab=True`, not `any_order`.** Every existing
`shared_numeric_vocab`/`any_order` test in the repo (from prior session
phases, before this one) uses tiny synthetic data — 40–60 rows, 2
numeric columns. This is the first time either feature has been run on
real, moderately complex data (2400 rows, 6 numeric columns), and it
does not hold up.

**Decision: redirect from "does any_order cost quality" to "does
shared_numeric_vocab itself scale beyond toy data" — the prior question
can't be answered until this one is.** `any_order` requires
`shared_numeric_vocab=True` by construction (`realtabformer2.py:293-296`
raises if not), so any_order's quality can't be fairly assessed while
its required dependency is the thing failing.

**Mechanism hypothesis (grounded, not yet proven):**
`shared_numeric_vocab` pools digit-token embeddings across *all*
numeric/datetime columns into one shared `wte` range, disambiguated only
by `token_type_ids`'s additive embedding contribution
(`build_pooled_vocab` docstring, `data_utils/vocab.py`). With 2 numeric
columns (every existing test), that's a 2-way disambiguation job. With 6
numeric columns of quite different per-position digit distributions all
sharing one embedding pool (Adult), it's a 6-way job for the same
additive mechanism — plausibly a much harder optimization problem, not
solved by more epochs alone within a reasonable budget.

**Diagnostic launched to test this directly:** `adult_shared_vocab_diagnostic.py`
— same 2400-row Adult subset, same `shared_numeric_vocab=True`,
`numeric_quantile_encoding=True`, `epochs=30`, but cut down to 2 numeric
columns (`age`, `fnlwgt`) + 2 categoricals (`sex`, `workclass`) + target.
If this converges normally, it confirms the column-count mechanism —
`shared_numeric_vocab` needs either more capacity/training or a design
change to handle many pooled numeric columns, a real, useful, previously
unknown limitation of a beta feature. If it's *still* broken with only 2
numeric columns, the real row count (2400 vs the 40–60 every prior test
used) is implicated instead, which would point to a different mechanism
entirely (worth a follow-up on its own). Result pending — see below.

**Relevance beyond this session:** this is directly relevant to the
earlier "is REaLTabFormerV2 sufficiently novel for a paper" conversation.
`any_order` was flagged there as the strongest candidate contribution,
*contingent on* it holding up under real benchmarking rather than toy
data — this is exactly that check, and it currently doesn't hold up, for
a reason (shared_numeric_vocab, its dependency) that's more fundamental
than any_order itself. A paper claim about any_order would need this
resolved first, and the resolution (if the column-count hypothesis holds)
is itself a legitimate, reportable finding about the pooled-vocab design.

### [2026-08-31 00:xx] D7 bake-off, configs A/B complete, C/D in progress

Configs A (BASELINE) and B (QUANTILE) done, both v1 REaLTabFormer, no
`shared_numeric_vocab`:

| | A_BASELINE | B_QUANTILE |
|---|---|---|
| `age` KS / Wass | 0.015 / 0.010 | 0.013 / 0.007 |
| `fnlwgt` KS / Wass | 0.029 / 0.031 | 0.023 / 0.031 |
| `education-num` KS / Wass | 0.025 / 0.017 | 0.015 / 0.008 |
| `capital-gain` KS / Wass | 0.025 / 0.354 | 0.015 / 0.106 |
| `capital-loss` KS / Wass | 0.004 / 0.099 | 0.006 / 0.142 |
| `hours-per-week` KS / Wass | 0.014 / 0.010 | 0.023 / 0.010 |
| mean categorical TVD | 0.032 | 0.028 |
| `corr(age,hours)` diff | 0.065 | 0.013 |
| discriminator AUC | 0.545 | 0.528 |
| TSTR ratio | 0.986 | (0.986 real, 0.892 synth → 0.986) |

Consistent with D1's original finding: quantile encoding wins clearly on
`capital-gain` (Wasserstein 0.354→0.106, a 70% reduction) and
`education-num`/`age`, roughly ties elsewhere, and `corr_diff` improves
substantially (0.065→0.013). One column bucks the trend —
`capital-loss` Wasserstein is *worse* under quantile encoding (0.099→0.142)
even with the D3 point-mass fix active, worth a closer look once D and C
land alongside it for full context, not in isolation.

Configs C (QUANTILE+ENTROPY, the direct test of D5's hypothesis) and D
(QUANTILE+CAT_THRESH) still running. Per D8's pre-committed rule: once C
lands, act on it immediately (policy recommendation if neutral/positive,
real implementation work only if still negative) rather than re-litigating.

### [2026-08-31 01:xx] Shared-vocab column-count diagnostic — PARTIAL confirmation, points to a broader issue

2 numeric columns (age, fnlwgt) instead of 6, same 2400 rows, same 30
epochs, `shared_numeric_vocab=True`, `numeric_quantile_encoding=True`:

| | 6-numeric-col run | 2-numeric-col run | non-shared baseline (bake-off A/B) |
|---|---|---|---|
| `age` KS | 0.405–0.850 | 0.078 | 0.013–0.015 |
| `fnlwgt` KS | 0.500–0.781 | 0.223 | 0.023–0.029 |
| discriminator AUC | 0.952–0.977 | 0.723 | 0.528–0.545 |

**Reading this honestly:** cutting numeric-column count from 6→2 produced
a real, large improvement (discriminator AUC 0.95→0.72), which is
consistent with — but does not on its own prove — the pooled-embedding
disambiguation-difficulty hypothesis. It is *not* a full explanation:
0.72 AUC is still far from the ~0.53 a properly-converged model gets, and
the *categorical* columns also degraded badly in this run (`sex` TVD
0.40, `workclass` TVD 0.29) even though categorical values aren't pooled
across columns the way numeric digits are — only their `token_type_ids`
contribution is shared machinery. That implicates something broader
than numeric-vocab pooling specifically: possibly the additive
`token_type_ids` embedding mechanism itself, or the
`make_dataset_with_column_types` data path, converges slower than v1's
plain disjoint-vocab approach in general, independent of how many
numeric columns are involved.

**Important process note, not just a technical one:** every existing
`shared_numeric_vocab` test in the repo checks *functional* correctness
(doesn't crash, a seeded value round-trips) — none of them measure
*distributional quality* against a real baseline the way this session's
Adult experiments do. It's plausible this feature has never actually had
its synthetic-data quality rigorously verified before now, at any scale
— the existing tests would pass regardless of whether the underlying
distribution is well-matched, since they don't check that.

**Decision: run the properly controlled version of this test** — same
5-column subset, same 30 epochs, `shared_numeric_vocab=False` this time
— which isolates shared_numeric_vocab's true cost even at the small
scale every prior test used, rather than comparing across two different
column counts (which conflates "column count" with "vocab sharing" as
explanations). Launched as `adult_shared_vocab_controlled.py`. Result
pending, ~92% through as of this entry.

### [2026-08-31 01:2x] D7 bake-off COMPLETE — decisive result, and D8's fallback hypothesis is FALSIFIED

Full 4-config comparison, v1 REaLTabFormer, 25 epochs, same Adult split:

| Config | ks_mean | wass_mean | tvd_mean | corr_diff | discriminator AUC | TSTR |
|---|---|---|---|---|---|---|
| A_BASELINE | 0.0188 | 0.0869 | 0.0324 | 0.0650 | 0.545 | 0.8926 |
| **B_QUANTILE** | 0.0158 | **0.0506** | 0.0279 | **0.0127** | **0.528** | 0.8922 |
| C_QUANTILE_ENTROPY | 0.0195 | 0.0941 | 0.0295 | 0.0316 | 0.541 | 0.8854 |
| D_QUANTILE_CATTHRESH | **0.0146** | 0.0730 | 0.0302 | 0.0518 | 0.566 | 0.8928 |

**B_QUANTILE (quantile encoding alone, nothing else) wins on 3 of 6
aggregate metrics outright (Wasserstein, correlation preservation,
discriminator) and is competitive on the rest.** Confirms D1 again, at a
different epoch count (25 vs 15) and architecture (same v1 REaLTabFormer)
— the finding is not an artifact of one specific run.

**Config C (QUANTILE+ENTROPY) is worse than B on every single one of the
six aggregate metrics** — not mixed, not "slightly worse on one thing,
better on another." ks_mean, wass_mean, tvd_mean, corr_diff,
discriminator AUC, and TSTR all move the wrong direction. This is a
*more* decisive negative result than D5's fixed-width test, not a milder
one.

**This falsifies D8's stated hypothesis, and that matters more than the
result itself.** D8 predicted config C would be neutral-to-positive,
reasoning that quantile-encoded chunks don't have the place-value
asymmetry that (D8 argued) explains why entropy weighting hurt under
fixed-width formatting. That reasoning was wrong, or at least
incomplete — entropy weighting is *actively worse* even where the
mechanism predicted "no reason for it to be." **Per D8's own
pre-committed rule ("if config C is still negative... a real code fix
becomes justified"), this should trigger implementing the magnitude-aware
weighting scheme — but doing that now would mean building a fix
justified by a hypothesis this same experiment just disproved. That is
not sound grounding, so I am not doing it.** The honest position is: I
don't currently have a validated mechanism for *why* digit_entropy_weighting
hurts quality on real data. Speculating a second hypothesis and building
another fix around it, without new evidence, would repeat the same
mistake at one remove.

**Revised, evidence-only conclusion (no unproven mechanism attached):**
`digit_entropy_weighting`, as currently implemented, does not improve —
and consistently degrades — synthetic data quality on real data, under
both fixed-width (D5) and quantile-encoded (D7-C) numeric representation.
Two independent, decisive negative results. **Recommendation: do not
enable `digit_entropy_weighting` by default; do not recommend it in
current form.** A real investigation into *why* (loss-floor
sensitivity, interaction with the optimizer/LR schedule, or something
else entirely) would need to start from fresh diagnostic evidence — e.g.
comparing training loss *curves* (not just endpoints) between B and C,
or an ablation on the `chunk_significance_weight_floor` value — not
from the place-value hypothesis this entry falsifies. Not attempted
here; flagged as a legitimate open question, correctly scoped as
"unknown" rather than "probably place-value" now that the evidence says
otherwise.

**Config D (QUANTILE+CATTHRESH) is also not a clean win over B**: best
on ks_mean (0.0146 vs 0.0158) but *worse* on wass_mean (0.073 vs 0.051),
corr_diff (0.052 vs 0.013, 4x worse), and has the single worst
discriminator AUC of all four configs (0.566). Demoting `education-num`
(16 levels) to categorical does not clearly help on top of quantile
encoding, at least not at this cardinality — no code action taken, just
noted as a negative result for this specific combination. This doesn't
mean `numeric_categorical_threshold` is bad in general (it fixes a real,
different problem — truly low-cardinality columns colliding under
digit-chunking); it means *combining* it with quantile encoding for a
column already well-served by quantile encoding isn't obviously
additive.

**Current best-known configuration, updated:** `numeric_quantile_encoding=
True` alone (which already includes the D2 boundary fix and D3 point-mass
reclaim). No `digit_entropy_weighting`. No `numeric_categorical_threshold`
unless a column's cardinality is low enough that quantile encoding's own
collision warning fires (not tested here — Adult's lowest-cardinality
numeric column, education-num at 16, did NOT benefit). `shared_numeric_vocab`/
`any_order` status still being determined (see above) — not currently
recommended given the unresolved convergence issue.

### [2026-08-31 01:2x] Controlled shared_numeric_vocab comparison COMPLETE — conclusive, severe finding

Same 5-column Adult subset, same 2400 rows, same 30 epochs,
`numeric_quantile_encoding=True` in both — only `shared_numeric_vocab`
varies:

| Metric | `shared_numeric_vocab=False` | `shared_numeric_vocab=True` |
|---|---|---|
| `age` KS | 0.019 | 0.078 (4x worse) |
| `fnlwgt` KS | 0.015 | 0.223 (**15x worse**) |
| `sex` TVD | 0.036 | 0.40 (**11x worse**) |
| `workclass` TVD | 0.014 | 0.29 (**21x worse**) |
| discriminator AUC | 0.542 | 0.723 |
| TSTR ratio | 0.999 | 0.967 |

**Conclusive.** This isolates `shared_numeric_vocab` itself, at exactly
the scale (5 columns) every prior test in the repo used, holding
row count, column count, encoding choice, and epoch budget all fixed. It
is dramatically worse on every metric — not a marginal regression. This
rules out every alternative explanation considered earlier (column
count, real-vs-toy row count, `any_order`, fixed-width vs quantile
encoding): the common factor really is `shared_numeric_vocab=True`
itself.

**Verdict on the `any_order` investigation this whole thread started
from:** the original question ("does `any_order` cost quality") cannot
be answered in isolation, because `any_order` requires
`shared_numeric_vocab=True`, and that dependency is independently,
severely broken. Not a mark against `any_order`'s own design — its
actual mechanism (per-batch column-block permutation, verified correct
at the unit level in D4) was never the problem. It's a mark against
`shared_numeric_vocab`, a *different* beta feature `any_order` happens
to depend on.

**Root cause not investigated further — a deliberate scope decision, not
an oversight.** A plausible mechanism was sketched earlier (pooled
digit-token embeddings force the model to disambiguate columns purely
through the additive `token_type_ids` term, a harder representation-
learning problem than the free disambiguation disjoint per-column
vocabularies get), but confirming it would mean profiling embedding
gradients or running much-longer-epoch training (100+) to see if it's
"slow to converge" vs "structurally capped" — a genuinely separate
debugging project from "find the best working configuration," which is
what this session's investigation is actually for. Recording it as a
well-scoped open question, not chasing it further right now.

**Practical recommendation: do not use `shared_numeric_vocab=True` or
`any_order=True` in production until this is root-caused and fixed.**
Both remain correctly marked `beta` in the docstrings — that caveat is
justified by this evidence, more so than anyone previously knew.

---

## FINAL SYNTHESIS (as of this entry) — current best-known REaLTabFormer configuration

Grounded in every experiment above, all on the same real Adult Income
dataset, all using the project's own `SyntheticDataBench` metrics:

**Recommended:**
- `numeric_quantile_encoding=True` (includes D2's boundary-precision fix
  and D3's point-mass resolution reclaim automatically — both are
  internal to this one flag). Validated repeatedly, at two different
  epoch budgets (15, 25), on both v1 and v2, on both a hand-built
  synthetic heavy-tailed dataset and real Adult data. Consistently wins
  or ties on marginal fidelity, correlation preservation, and
  discriminability; never regresses categorical columns (untouched by
  the flag).

**Not recommended (evidence-backed, not assumed):**
- `digit_entropy_weighting` — negative in two independent, decisive
  tests (fixed-width in D5, quantile-encoded in D7-C). Root cause
  genuinely unknown (D5's place-value hypothesis is falsified by D7-C);
  do not enable pending real investigation.
- `shared_numeric_vocab` / `any_order` — severe, conclusively-isolated
  quality cost (D7 controlled comparison above). Not a fault of
  `any_order`'s own permutation mechanism (verified correct, D4) —
  its required dependency is what's broken.
- `numeric_categorical_threshold` combined with quantile encoding, at
  least for moderate cardinality (~16 levels, D7-D) — not clearly
  additive on top of quantile encoding at that cardinality. Its
  standalone value for genuinely low-cardinality columns (2-5 levels,
  where quantile encoding can't help as much) was established in an
  earlier session phase, before this investigation, and isn't
  contradicted by this — just not shown to *stack* with quantile
  encoding at 16 levels specifically.

**Next planned step:** verify whether the winning configuration
(`numeric_quantile_encoding=True` alone) is actually converged at 25
epochs or would improve further with more training budget — the one
remaining un-investigated axis before calling this configuration
settled. See below.

### [2026-08-31 01:3x] Epoch-scaling experiment launched

`adult_epoch_scaling_experiment.py`: same B_QUANTILE setup (v1
REaLTabFormer, `numeric_quantile_encoding=True`, same Adult split), run
at 50 and 75 epochs, compared against the already-completed 25-epoch
result (ks_mean=0.0158, wass_mean=0.0506, tvd_mean=0.0279,
corr_diff=0.0127, discriminator_auc=0.5283, tstr_auc=0.8922) rather than
rerun. Tests whether more training budget is a genuinely available lever
for improving the current-best configuration further, or whether 25
epochs already captures most of the achievable quality on this dataset
size. (Process note: first launch attempt didn't set
`run_in_background`, timed out at the tool's 2-minute limit, and the
child process was killed with it — confirmed via `ps aux` before
relaunching correctly, not assumed.) Pending.

### [2026-08-31 02:0x] Epoch-scaling result — real headroom, but non-monotonic (single-run caveat)

| Epochs | ks_mean | wass_mean | tvd_mean | corr_diff | discriminator AUC | TSTR ratio |
|---|---|---|---|---|---|---|
| 25 (bake-off B) | 0.0158 | 0.0506 | 0.0279 | 0.0127 | 0.528 | 0.9862* |
| 50 | 0.0175 | 0.0872 | 0.0292 | 0.1008 | 0.553 | 0.991 |
| **75** | **0.0112** | **0.0253** | **0.0211** | 0.0570 | **0.509** | **0.996** |

(*TSTR ratio at 25 epochs recomputed here as tstr_auc/real_auc =
0.8922/0.9051 = 0.9862, for direct comparability with the ratio format
used at 50/75 — same underlying number as before, just expressed as a
ratio.)

**Not a clean monotonic trend.** 50 epochs is *worse* than 25 on
wass_mean, corr_diff, and discriminator AUC — all by a real margin, not
noise-sized. 75 epochs is the best result on 5 of 6 metrics, some
substantially (wass_mean roughly halves again, 0.051→0.025;
discriminator AUC nearly hits the 0.5 ideal). **Honest interpretation:
each epoch count here is a single training run, not averaged over
seeds — the 50-epoch dip is plausible optimization noise (a
different effective LR schedule per total-epoch count, no early-stopping
checkpoint averaging), not a real "quality gets worse then better" law.**
The trustworthy read is the endpoints: 75 clearly beats 25 on most
metrics; I would not claim a smooth epoch-vs-quality curve from n=1 per
setting.

**What this does NOT yet verify, and needs to before recommending 75
epochs as part of the final configuration:** whether the quality gain is
coming from genuinely better generalization, or from memorization —
TSTR/KS/discriminator can all look great for a model that's just copying
training rows back out, which would be a privacy failure dressed up as
a quality win. REaLTabFormer's own `SyntheticDataBench.get_dcr()` /
data-copying test exists exactly to check this (the paper's own core
contribution — see `[[user_realtabformer_author]]` memory note) and
hasn't been run in any experiment this session. Launching a check now
before finalizing the epoch recommendation.

### [2026-08-31 02:1x] DCR/memorization check launched

`adult_dcr_check.py`: retrains the 75-epoch candidate, then compares
synthetic-vs-train DCR (Manhattan distance, preprocessed space, via
`SyntheticDataBench.get_dcr(is_test=False)`) against the real held-out
test set's own natural vs-train DCR (`is_test=True`) — the honest
baseline for "how close does *genuinely unseen* data sit to training
data by chance." If synthetic DCR is systematically smaller than that,
it's a copying signal, not just a good fit. This is the first time any
experiment this session has actually exercised the project's own
privacy-checking mechanism, despite it being one of the two core
contributions of the original REaLTabFormer paper (the other being
constrained decoding) per `[[user_realtabformer_author]]`. ~39 min
estimated (75-epoch training dominates). Pending.

### [2026-08-31 02:5x] DCR result — POSITIVE for copying, AND it exposes a session-wide methodology gap

| | synthetic-vs-train | real-test-vs-train (honest baseline) |
|---|---|---|
| mean DCR | 1.647 | 2.422 |
| median DCR | 1.271 | 2.141 |
| p5 DCR | 0.0007 | 0.443 |
| min DCR | **0.0000** | 0.070 |

Synthetic data sits systematically *closer* to training data than
genuinely unseen real test data does, on every summary statistic — not
a small effect. Minimum synthetic DCR is exactly 0.0000: at least one
generated row is an exact duplicate of a training row in preprocessed
feature space. 18% of synthetic rows are closer to training data than
95% of real held-out data ever gets. **This is a real, meaningful
copying signal, not noise.**

**Root cause, verified in the source, not assumed:** every experiment
this entire session used `n_critic=0` in every `.fit()` call. Checked
`realtabformer.py:540`: `if n_critic <= 0:` routes to the plain
`_fit_tabular` path — trains for the full fixed epoch count with **no**
DCR-bootstrap early stopping. `_train_with_sensitivity` — the method
implementing the paper's actual overfitting/privacy protection, one of
REaLTabFormer's two headline original contributions alongside
constrained decoding (`[[user_realtabformer_author]]`) — is only
reachable when `n_critic > 0`. **None of this session's experiments
used it.** `n_critic=0` was chosen early on purely to keep experiments
fast and simple for feature-correctness testing (D1-D4's bug-hunting
work), which was a reasonable choice for *that* purpose — but it silently
carried forward into every quality-comparison experiment since, including
the ones this log used to declare winners.

**What this invalidates and what it doesn't, reasoned through rather
than blanket-discarded:**
- **Retracted: "75 epochs is better than 25."** This conclusion is
  confounded by unprotected overfitting — more epochs with no early
  stopping plausibly means more memorization, which mechanically
  improves KS/discriminator/TSTR for the wrong reason (the synthetic
  "test-like" rows are closer to being copies, not better draws from
  the true distribution). The DCR check proves the mechanism that would
  explain this is actually present. Do not recommend 75 epochs, or any
  specific higher epoch count, without re-running under
  `_train_with_sensitivity`'s actual protection.
- **Probably still sound: the bake-off's relative rankings (A/B/C/D) and
  D1-D3's quantile-encoding findings.** Every config compared within a
  given experiment used the *same* fixed epoch count and the *same*
  lack of protection, so a systematic confound affecting all arms
  roughly equally is a weaker threat to the *relative* comparison than
  to an *absolute* "more epochs = better" claim across different epoch
  counts. Not proven equally sound — just a different, lesser-risk
  category, worth stating honestly rather than either defending or
  discarding without evidence either way.
- **The `shared_numeric_vocab` finding is unaffected.** That comparison
  was so severe (categorical TVDs off by 11-21x) that unprotected
  overfitting cannot plausibly explain it — overfitting would make
  synthetic data look *more* like a (copied) subset of training data,
  not produce wildly wrong marginal category frequencies.

**Corrective action:** launching the properly-configured version of the
epoch-scaling question — fit the winning config (`numeric_quantile_encoding=
True` alone) with `n_critic>0`, letting `_train_with_sensitivity`'s own
DCR-bootstrap decide when to stop, rather than picking an epoch count by
hand and hoping. This is the scientifically correct way to answer "what's
the best training budget" for this tool — it's literally what the
mechanism is for — not a new idea, just the thing that should have been
used from the start of the epoch-count question specifically (D1-D4's
correctness-testing use of `n_critic=0` remains a reasonable choice for
that different purpose).

Launched as `adult_proper_training.py`: same winning config, `epochs=100`
(generous ceiling), default `n_critic=5`/`n_critic_stop=2` (i.e. simply
*not* overriding them to 0, unlike every prior experiment). Computes its
own sensitivity threshold first (500 bootstrap rounds), then trains in
5-epoch increments with a DCR check after each, stopping once 2
consecutive rounds show no improvement. Reports the same quality metrics
as before plus DCR, so it's directly comparable to both the unprotected
75-epoch run and the 25-epoch bake-off baseline. This is the run whose
result should actually anchor the final epoch/training-budget
recommendation. Pending — likely 30-60+ min given the added bootstrap
and periodic generation overhead on top of training itself.

### [2026-08-31 03:2x] The run CRASHED — and found a real, high-value, pre-existing bug

`_train_with_sensitivity` (the sensitivity training path this entry
just launched) failed after its first 5-epoch critic round:

```
TypeError: REaLTabFormer.sample() argument after ** must be a
mapping, not NoneType
```

**Root cause, verified in source, not assumed:**
`_train_with_sensitivity`'s `gen_kwargs: Optional[Dict[str, Any]] = None`
parameter was unpacked directly as `self.sample(..., **gen_kwargs)`
(realtabformer.py:837, and the identical pattern in
realtabformer2.py:1003) — crashing for any caller who doesn't happen to
pass `gen_kwargs` explicitly. **`fit()`'s own default `n_critic=5` is
what reaches this code path** — meaning the ordinary, default way of
calling `.fit(df)` with no other overrides crashes on the very method
implementing REaLTabFormer's own DCR-bootstrap overfitting-protection
mechanism, one of the two headline contributions of the original paper.
Confirmed pre-existing via `git blame`: commit `73f23964`, authored by
the tool's own author, predating every change in this session and this
branch's earliest commits.

**Why this wasn't caught earlier, honestly:** every experiment this
entire session used `n_critic=0` (chosen early on, for D1-D4's
feature-correctness bug-hunting, where fast/simple training was the
right call) — which routes to a completely different, working code
path (`_fit_tabular` directly) that never touches this line. This bug
was invisible to everything done in this log until the exact moment the
investigation tried to use the tool's real, intended training mode for
the first time.

**Fixed:** `**(gen_kwargs or {})` in both `realtabformer.py` and
`realtabformer2.py`. Two regression tests added (v1 and v2), each
confirmed to fail against the pre-fix code with the exact TypeError
above and pass against the fix — not vacuous. Full suite: 97 passed,
same 2 known pre-existing unrelated failures (and `test_TabularSampler`'s
remaining failure confirmed to be a *different*, unrelated bug — a
missing constructor argument in the test file itself, not this code
path). Committed (`b721085`), merged into the local
`feat/support-seed-input` worktree. Not pushed yet.

**This is arguably the single highest-value finding of the entire
session** — not because the other findings aren't real, but because this
one blocked the *tool's own core safety mechanism* from working at all
under its own default configuration, for anyone who didn't happen to
work around it. Relaunched as `adult_proper_training_v2.py` (task
`bbl9kmlbn`), confirmed running past the sensitivity-threshold bootstrap
without crashing. Pending.

### [2026-08-31 04:1x] Sensitivity-protected result — nuanced, and it reframes how quality was measured all session

**The mechanism worked as designed.** Sensitivity threshold computed as
0.01078. `val_sensitivity` per critic round: round 5 = -0.0057, round
10 = -0.0015, round 15 = 0.0004, round 20 = 0.0053, round 25 = 0.0049,
round 30 = 0.0089, round 35 = 0.0092, round 40 = 0.0091 (all safely
below threshold) → round 45 = 0.0124, round 50 = 0.0141 (both *above*
threshold). Two consecutive above-threshold rounds triggered
`n_critic_stop=2`'s early stop at **epoch 50**. This is a real,
principled, data-driven stopping decision — not a number I picked.

**Resulting quality, compared against the unprotected 25-epoch bake-off
baseline:**

| Metric | Unprotected, 25 epochs | Sensitivity-protected, stopped at 50 |
|---|---|---|
| ks_mean | 0.0158 | 0.0367 (worse) |
| wass_mean | 0.0506 | 0.1784 (much worse) |
| tvd_mean | 0.0279 | 0.0568 (worse) |
| corr_diff | 0.0127 | 0.0507 (worse) |
| discriminator AUC | 0.528 | 0.593 (worse) |
| TSTR ratio | 0.986 | 0.988 (~tied) |

**Every distributional-similarity metric looks worse under proper
protection.** At first glance this reads as "the safety mechanism makes
quality worse" — but the DCR numbers tell a different, more important
story:

| | synthetic-vs-train DCR | fraction suspiciously close |
|---|---|---|
| Unprotected, 75 epochs | mean 1.647, min **0.0000** | 0.180 |
| Sensitivity-protected, 50 epochs | mean 1.979, min **0.0000** | **0.092** |
| Real test-vs-train (honest baseline) | mean 2.422, min 0.070 | — |

Protection roughly halved the copying-risk fraction (0.18→0.09) and
moved the mean DCR meaningfully closer to the honest real-test baseline
(1.65→1.98, vs 2.42) — genuine improvement on the privacy axis. It did
**not** fully eliminate copying (min DCR is still exactly 0.0000, an
exact duplicate persists even in the protected run).

**The reframing, and it matters for everything logged before this
entry:** KS/Wasserstein/discriminator/TSTR against a real held-out test
set are not a "pure" quality signal — they're inflated by memorization.
A model that copies training rows more closely will *mechanically* score
better on "how similar is synthetic data to the true distribution"
metrics, for the wrong reason, right up until copying becomes extreme
enough for a discriminator to start flagging near-duplicates as
suspicious. **This means every quality comparison logged earlier in this
document (D1-D3's quantile-encoding wins, D5/D7's entropy-weighting
losses, the D7 bake-off rankings) was measured this same
memorization-contaminated way** — all used `n_critic=0`. This doesn't
necessarily invalidate their *relative* rankings (every arm of each
comparison had the same confound, so a comparison between two configs at
the same fixed epoch count is on more equal footing than an absolute
number is) — but it means none of those numbers should be read as
"how good is this model, absolutely." Flagging this honestly rather
than either quietly ignoring it or retroactively discarding everything
without evidence either way.

**A concrete, promising, not-yet-tried lever surfaced by reading the
`fit()` signature just now:** `load_from_best_mean_sensitivity: bool =
False`. Default is off. This run used the *final* checkpoint (epoch 50,
2 rounds *after* sensitivity first crossed the threshold at epoch 45,
because `n_critic_stop=2` waits for 2 consecutive bad rounds before
stopping — a reasonable debounce against noise, but it means the
checkpoint actually used is not necessarily the safest one seen).
Loading the *best* (lowest-sensitivity) checkpoint instead of the last
one is directly available and untested. Worth one more run before
finalizing.

Launched: same script, `load_from_best_mean_sensitivity=True` added
(task `bvv5po61b`, log `adult_proper_training_v3_bestckpt.log`).
Everything else identical, so directly comparable to the round-50 result
above. Expectation, stated before seeing the result: should land closer
to round 40's checkpoint (last round safely below the sensitivity
threshold) rather than round 50's, plausibly with quality between the
unprotected-25-epoch numbers and the round-50 numbers, and a copying-risk
profile at least as good as round 50's 0.092. Pending, ~35-45 min
estimated.

### [2026-08-31 04:5x] `load_from_best_mean_sensitivity=True` result — best privacy profile of the entire session

Training trajectory (sensitivity per critic round) is identical to the
round-50 run, as expected — `load_from_best_mean_sensitivity` only
changes which checkpoint gets loaded *after* training stops, not the
training itself. Same stop trigger (rounds 45 and 50 both above the
0.01115 threshold).

| Metric | Unprotected, 25ep | Protected, last ckpt (round 50) | Protected, **best ckpt** |
|---|---|---|---|
| ks_mean | 0.0158 | 0.0367 | 0.0374 |
| wass_mean | 0.0506 | 0.1784 | **0.0966** |
| tvd_mean | 0.0279 | 0.0568 | 0.0630 |
| corr_diff | 0.0127 | 0.0507 | **0.0280** |
| discriminator AUC | 0.528 | 0.593 | 0.618 |
| TSTR ratio | 0.986 | 0.988 | 0.983 |
| DCR mean | not measured | 1.979 | **2.210** |
| DCR min | not measured | 0.0000 | **0.0145** |
| fraction suspiciously close | 0.18 (75ep, unprotected) | 0.092 | **0.058** |

**Best privacy result of the whole session.** `fraction suspiciously
close` at 0.058 is close to the ~0.05 rate you'd expect from a model
that isn't systematically copying at all (recall this fraction is
defined against the real test set's own 5th percentile, so ~5% is
roughly the "no signal" floor). **DCR min is no longer exactly 0.0000
— the first run all session with no exact training-row duplicate in
the synthetic output.** wass_mean and corr_diff both improve
substantially over the last-checkpoint run (wass_mean nearly halves,
corr_diff nearly halves) — genuinely closer to the unprotected
baseline's quality on those two metrics specifically.

**Discriminator AUC is nominally the worst of the three (0.618) — read
carefully, not just at face value.** A lower discriminator AUC in the
*other* two runs plausibly benefited from the same effect the DCR
numbers exposed: rows close enough to being copies are, almost by
construction, hard for a discriminator to distinguish from real data.
The best-checkpoint run's higher, more honest discriminator AUC likely
reflects a *smaller* true distributional gap being measured *without*
that memorization crutch propping the number up artificially, not a
larger true gap. Stated as interpretation, not proven — I don't have a
clean way to fully separate "genuinely worse fit" from "no longer
benefiting from the memorization crutch" with the metrics computed here.

**Practical conclusion:** `load_from_best_mean_sensitivity=True` is
worth recommending alongside the sensitivity mechanism itself — it
gives a materially safer model (no exact duplicates, lowest copying
signal) for a real but bounded quality cost on some metrics, and outright
improves two others (wass_mean, corr_diff) relative to just taking the
final checkpoint. This is the actual mechanism REaLTabFormer provides for
navigating the quality/privacy tradeoff — and until the `gen_kwargs` fix
earlier in this session, it was completely inaccessible via the tool's
own default configuration.

This closes the last well-motivated open question in this investigation.
Full synthesis below.

---

## D9 — Reopened: is `shared_numeric_vocab`'s D6 failure actually explained by the same `n_critic=0` gap?

**User's question, and it's a sharp one:** every `shared_numeric_vocab`
experiment in D6 (the diagnostic, the controlled comparison) used
`n_critic=0` — necessarily, since the `gen_kwargs` crash (fixed later
in this document, after D6) made `_train_with_sensitivity` completely
unusable at the time D6 was investigated. It's entirely possible D6's
"severe, structural failure" verdict is itself an artifact of blind,
unprotected, fixed-epoch training — exactly the same methodology gap
that made the epoch-scaling question unreliable before the DCR
investigation. D6 was never actually retested with the tool's real
training mode.

**This could overturn a major conclusion, so it's being tested directly
rather than argued about.** Rerunning D7's controlled comparison (5-column
Adult subset: age, fnlwgt, sex, workclass, income) — `shared_numeric_vocab`
True vs False — with sensitivity-based training on *both* sides this
time (`n_critic` default, `load_from_best_mean_sensitivity=True`,
`epochs=100` ceiling), for a fully apples-to-apples comparison against
the unprotected numbers already on record:

- `shared_numeric_vocab=False`, unprotected, 30ep: age KS=0.019, fnlwgt
  KS=0.015, sex TVD=0.036, workclass TVD=0.014, discriminator=0.542,
  TSTR ratio=0.999.
- `shared_numeric_vocab=True`, unprotected, 30ep: age KS=0.078, fnlwgt
  KS=0.223, sex TVD=0.40, workclass TVD=0.29, discriminator=0.723,
  TSTR ratio=0.967.

Launched as `adult_shared_vocab_sensitivity.py` (task `bnwivk3nn`).
v2 + distilgpt2 + sensitivity training's per-round generation overhead
makes this slow — likely 1.5-2 hours for both sides. Pending.

### [2026-08-31 09:5x] Result — partially confirmed: the gap shrinks a lot, but doesn't disappear

Full four-way comparison:

| Metric | False, unprotected | False, **protected** | True, unprotected | True, **protected** | Gap ratio, unprotected | Gap ratio, protected |
|---|---|---|---|---|---|---|
| age KS | 0.019 | 0.058 | 0.078 | 0.080 | 4.1x | **1.4x** |
| fnlwgt KS | 0.015 | 0.093 | 0.223 | 0.203 | 14.9x | **2.2x** |
| sex TVD | 0.036 | 0.131 | 0.40 | 0.353 | 11.1x | **2.7x** |
| workclass TVD | 0.014 | 0.070 | 0.29 | 0.179 | 20.7x | **2.6x** |
| discriminator AUC | 0.542 | 0.586 | 0.723 | 0.690 | +0.181 | **+0.104** |
| TSTR ratio | 0.999 | 0.998 | 0.967 | 0.951 | -0.032 | -0.047 |

**Your hypothesis was substantially right, and it materially changes the
picture.** A large fraction of D6's "severe, up to 21x worse" verdict
was indeed inflated by the same unprotected-training confound the DCR
investigation found elsewhere — under fair, sensitivity-protected
training on both sides, the gap shrinks from roughly 4-21x down to
roughly 1.4-2.7x across every metric. That's not noise; it's a real,
substantial narrowing.

**But it doesn't fully disappear.** `shared_numeric_vocab=True` is still
consistently, non-trivially worse than `False` on *every single metric*
under identical, fair, properly-protected training — not by an order of
magnitude anymore, but by a real, repeatable margin. **D6's directional
verdict holds (don't use `shared_numeric_vocab` without more work); its
severity claim was overstated and is hereby corrected.**

**A genuinely useful side-finding from the DCR numbers, worth stating
because it points at a real mechanism rather than leaving this as "still
somewhat worse, unclear why":** `shared_numeric_vocab=False`'s protected
DCR (mean 0.1868) sits almost exactly on the real-test-vs-train baseline
(0.1912) — a clean, unbiased fit. `shared_numeric_vocab=True`'s protected
DCR (mean 0.3009) is *higher* than that baseline, not lower — and its
`frac_suspicious` (0.035) is actually *lower* than `False`'s (0.048).
**This rules out memorization/copying as the explanation for
`shared_numeric_vocab=True`'s remaining gap** — if it were overfitting
more than `False`, its DCR would be systematically *closer* to training
data, not farther. The remaining 1.4-2.7x quality gap looks like a
genuine, non-privacy-related *fit* problem — consistent with the
disambiguation-difficulty mechanism sketched in D6 (pooled digit-token
embeddings needing the additive `token_type_ids` term alone to tell
columns apart), now with the memorization confound ruled out rather than
just hypothesized.

**Updated conclusion for the FINAL SYNTHESIS:** `shared_numeric_vocab`/
`any_order` remain not recommended for production, but the finding is
now sharper and better-grounded — a real, ~2x-scale quality/fit cost
under fair training conditions, not the ~10-20x catastrophe the
unprotected comparison suggested, and demonstrably not a privacy/copying
issue. Worth real investment to fix (the gap is meaningful but not
prohibitive), more so than D6 alone would have suggested.

---

## TRUE FINAL SYNTHESIS — complete, supersedes the interim synthesis earlier in this document

Everything below is grounded in the experiments and evidence logged
above; nothing here is asserted without a corresponding entry. All
experiments used the real UCI Adult Income dataset (3000-row subsample,
2400/600 train/test split, `random_state=1029`), REaLTabFormer's own
`SyntheticDataBench` metrics, and (from the DCR investigation onward)
its own DCR/sensitivity mechanism.

### The one methodology fact that reframes everything: measure privacy alongside quality, always

The single most important finding of this session isn't a feature
verdict — it's this: **KS/Wasserstein/discriminator/TSTR against a real
held-out test set are not trustworthy quality signals on their own.**
An unprotected model that trains longer scores *better* on all of them,
right up until it's copying training rows outright — proven directly:
the unprotected unprotected 75-epoch run had an exact training-row
duplicate in its synthetic output (DCR min = 0.0000) and 18% of rows
closer to training data than 95% of genuinely unseen real data ever
gets, while its KS/discriminator numbers looked like across-the-board
"improvement" over 25 epochs. Every quality-only comparison in this
document (D1-D3, D5, D7's bake-off) used `n_critic=0` (no DCR
protection) — a reasonable choice for the fast, iterative
feature-correctness testing those comparisons were built for, but it
means their *absolute* numbers should be read as "how similar is this to
the training data," not unambiguously "how good is this model." Their
*relative* rankings (quantile encoding beats baseline; entropy weighting
loses to quantile-alone) are on firmer ground, since every arm of a
given comparison shared the same confound roughly equally — but even
that is a "probably fine" not a "proven identical."

**Practical rule going forward, stated once so it doesn't need
re-deriving:** any synthetic-data quality claim for this tool should be
paired with a DCR/copying check, not reported alone.

### Recommended configuration

1. **`numeric_quantile_encoding=True`.** The one clean, repeatedly
   validated win — across a hand-built synthetic heavy-tailed dataset
   and real Adult data, at multiple epoch counts, on both v1 and v2.
   Includes the boundary-precision fix (D2) and point-mass resolution
   reclaim (D3) automatically; both are internal to this one flag. No
   caveat needed beyond the general privacy-measurement rule above (not
   specifically tested against a DCR check in isolation from the epoch
   question, but nothing in its own validation depended on unprotected
   overtraining — D1's core entropy claim is architectural, not an
   overfitting artifact).

2. **Train with `n_critic`'s actual default (do not override to 0) and
   `load_from_best_mean_sensitivity=True`.** This is the tool's own
   mechanism for the quality/privacy tradeoff, and letting it run is
   better-grounded than hand-picking an epoch count — confirmed
   directly: the sensitivity-protected run with best-checkpoint loading
   had the best privacy profile of the entire session (no exact
   duplicates, lowest suspicious-copying fraction) while landing within
   a bounded, real quality cost on some metrics and outright improving
   others (wass_mean, corr_diff) relative to the naive last-checkpoint
   version of the same protected run. **This mechanism was completely
   broken for any default `.fit(df)` call until the `gen_kwargs` fix in
   this session** (see below) — so "use the sensitivity mechanism" was
   not actually available advice before this session's work, regardless
   of what any prior documentation said.

3. **No specific epoch count is "the" answer, and that's the correct
   conclusion, not a gap.** Unprotected epoch-scaling (25/50/75) gave a
   non-monotonic, unreliable signal precisely because more unprotected
   training trades quality-metric-inflation for privacy risk in a way
   that isn't visible without a DCR check. The sensitivity mechanism
   exists to make this tradeoff automatic and data-driven rather than
   something a user should hand-tune by watching KS statistics alone.

4. **`any_order=True, shared_numeric_vocab=False`, when arbitrary-subset
   conditioning is needed.** Originally coupled to `shared_numeric_vocab`
   for reasons that turned out to be implementation convenience, not
   necessity (D11) — decoupled and shipped this session. Not merely
   "works": under sensitivity-protected training on the same Adult
   subset, it **beat the plain non-any_order baseline** on both numeric
   columns (age KS 0.058→0.026, fnlwgt KS 0.093→0.028) and had the best
   discriminator AUC (0.544, closest to the 0.5 ideal) of any
   configuration tested this entire session — on top of unlocking a real
   capability (conditioning on any column subset, not just a prefix) the
   baseline can't do at all. One categorical metric (`workclass` TVD)
   was worse and the copying-risk signal was mildly elevated (still
   within the same order of magnitude as the natural floor) — not a
   clean sweep, reported as such, but a strong, genuinely recommended
   result. Only reachable via a fix shipped in this session
   (`fd9b6fc`) — the combination this recommendation depends on
   (`any_order` without `shared_numeric_vocab`) did not work at all
   before this session.

### Not recommended, with evidence

- **`digit_entropy_weighting`**: negative in two independent, decisive
  tests — fixed-width numeric encoding (D5) and quantile-encoded (D7-C,
  worse on *all six* aggregate metrics, disconfirming the place-value
  hypothesis D5 proposed). Root cause genuinely unknown; do not enable
  pending real investigation (comparing training loss curves, or an
  ablation on the loss-floor hyperparameter — not attempted here, and
  should start from fresh evidence, not the falsified place-value
  theory).
- **`shared_numeric_vocab`** (on its own, or combined with `any_order`):
  real, repeatable quality cost — but smaller than first measured, and
  now known to be `shared_numeric_vocab`'s own cost specifically, not
  `any_order`'s. D6's initial controlled comparison (unprotected,
  `n_critic=0`) found up to 21x worse categorical fidelity; D9 reran the
  identical comparison with proper sensitivity-based training on both
  sides and found the gap shrinks to roughly 1.4-2.7x across every
  metric — still real, but D6's severity was substantially inflated by
  the unprotected-training confound. D9's DCR check rules out
  memorization as the explanation for the remaining gap — the residual
  ~2x cost looks like a genuine fit/learning problem (pooled digit-token
  embeddings needing the additive `token_type_ids` term alone to tell
  columns apart, D9's mechanism). **D10/D11 went further and confirmed a
  second, additive, `any_order`-specific mechanism on top of this**:
  under permutation, `token_type_ids` alone can't disambiguate a
  multi-chunk numeric column's own within-column chunk position either
  (position, which normally does that job, becomes unreliable once
  columns move to different absolute positions each batch) — this
  degrades numeric columns substantially further (fnlwgt KS
  0.203→0.324) while categorical columns (no within-column structure to
  lose) are unaffected or improve. **D11 decoupled `any_order` from
  `shared_numeric_vocab` entirely, sidestepping this second mechanism by
  construction rather than fixing it in place** — see the Recommended
  section above. `shared_numeric_vocab` *alone* (`any_order=False`)
  remains not recommended pending its own root-cause fix; its cost is
  real but moderate (~2x), not the near-unusable failure D6 alone would
  have suggested, and is no longer entangled with `any_order`, which now
  has a clean, positive recommendation of its own.
- **`numeric_categorical_threshold` stacked on top of quantile encoding**,
  at least at moderate cardinality (~16 levels) — not clearly additive;
  its own standalone value for genuinely low-cardinality columns (2-5
  levels) from an earlier session phase is unaffected by this, just not
  shown to *stack*.

### Bugs found and fixed this session (chronological, all committed, all merged into the local `feat/support-seed-input` worktree)

1. `numeric_quantile_encoding`'s core implementation (D1) — shipped and
   pushed.
2. Boundary-precision decode failure on zero-inflated columns (D2) —
   shipped and pushed.
3. Point-mass resolution waste (D3) — shipped and pushed.
4. `any_order` × `digit_entropy_weighting` composition test (D4, test
   coverage only, no code bug) — committed locally, **not yet pushed**.
5. **`gen_kwargs=None` crash in `_train_with_sensitivity`** (both v1 and
   v2) — the highest-value find of the session: this made REaLTabFormer's
   own default training configuration (`n_critic=5`) crash on its first
   critic round for any caller not explicitly passing `gen_kwargs`,
   silently disabling the paper's own core overfitting-protection
   mechanism. Fixed, regression-tested (confirmed to fail pre-fix, pass
   post-fix), committed locally, **not yet pushed**.

### What's ready to push, and what's still open

**Pushed** (user confirmed): commits `93ddffb`, `b721085`, `fd9b6fc` are
now on `origin/feat/support-seed-input`.

**(Historical — was "ready to push," now done, kept for the record):**
- Commits `93ddffb` (any_order × entropy-weighting composition test),
  `b721085` (the `gen_kwargs` fix), and `fd9b6fc` (decoupling `any_order`
  from `shared_numeric_vocab`) are sitting on top of the already-pushed
  `ceb84d3` on the local `feat/support-seed-input` worktree
  (`scratchpad/repro/seed_input_merged`). All tested, lint-clean, and
  low-risk — each is purely additive (existing behavior unchanged,
  confirmed by the full pre-existing test suite passing unmodified in
  every case). The `gen_kwargs` fix is worth prioritizing most — it's a
  correctness fix for a feature that was silently unusable at its own
  documented default. `fd9b6fc` is the one with a genuinely positive,
  empirically-validated result attached (D11) — it doesn't just fix a
  bug, it delivers a configuration that beat the baseline.

**Still open, not attempted (either out of scope for "find the best
configuration" specifically, or needing a decision from you before
proceeding):**
- `shared_numeric_vocab`'s root cause — would need real debugging time,
  a separate effort from this investigation.
- `digit_entropy_weighting`'s root cause — same; the place-value
  hypothesis this session proposed is now known to be wrong, and no
  replacement hypothesis has evidence behind it yet.
- The fully-constant-numeric-column formatting bug found in passing
  during D3 (produces `"1."` instead of `"1.0000"`) — logged as
  out-of-scope there, still true here.
- A rigorous multi-seed replication of any of these findings — everything
  in this log is single-seed, single-dataset (Adult). Directionally
  consistent findings replicated across different epoch counts/
  architectures within Adult (quantile encoding won at 15, 25, and with
  the point-mass fix; entropy weighting lost twice independently) give
  more confidence than a single run would, but a real paper-grade claim
  would want multiple seeds and multiple datasets, as already flagged in
  the earlier "is this publishable" conversation this session had.

---

## D10 — User's architectural question: can within-column chunk tokens even distinguish their own position under any_order?

**Question, verbatim in substance:** `token_type_ids` marks which
*column* a token belongs to, but does anything distinguish a column's
*own* digit-chunk sub-positions from each other (e.g. "2nd chunk" vs
"5th chunk" of the same numeric column)?

**Verified, not assumed, in three steps:**
1. `column_type_ids[c] = name_to_type_id[_original_column_name(c)]`
   (`vocab.py:327`) — every digit-chunk sub-column of the same original
   column gets the *identical* type id. Confirmed.
2. HF GPT2's own source (`modeling_gpt2.py:865-907`): position embeddings
   are always computed (`position_embeds = self.wpe(position_ids)`,
   added to token embeddings unconditionally); `token_type_embeds` is a
   separate, purely additive term on top when provided. So under
   **fixed column order** (`any_order=False`), absolute position still
   reliably distinguishes chunk index within a column — same mechanism
   a non-shared-vocab model relies on, unaffected by `shared_numeric_vocab`
   alone.
3. `AnyOrderColumnCollator` physically reorders `input_ids` (and
   `token_type_ids`, `labels`, `token_weights`) via `torch.gather` every
   batch. Grepped the codebase for custom `position_ids` handling
   anywhere in `realtabformer2.py`/the collator/dataset code — none
   exists. HF's default `position_ids = arange(seq_len)` therefore
   applies to the *already-reordered* sequence. **Under `any_order=True`,
   the same semantic chunk (e.g. "fnlwgt's 2nd digit") lands at a
   different absolute position every batch, so position embeddings can
   no longer reliably encode "which chunk index within this column"
   the way they do under fixed order.**

**A significant correction surfaced while verifying this:** this
session's own memory notes (and therefore several earlier log entries'
framing) stated `numeric_nparts=1` (used throughout every
quantile-encoding experiment, including all of D1-D3, D7, D9) means "no
chunking." **This is wrong, confirmed empirically** —
`tokenize_numeric_col`'s `nparts` is characters *per* chunk, not number
of chunks; `nparts=1` is *maximum* granularity (every character its own
processed sub-column), and only `nparts >= the full string length`
collapses to one unsplit column. A 6-character quantile-encoded value
("0.4732") under `nparts=1` produces 6 separate single-character
sub-columns. **This means the within-column multi-chunk-disambiguation
question has been architecturally in play in every quantile-encoded
experiment this entire session, not a corner case.** Corrected in the
persistent memory file
(`realtabformer_architecture_notes.md`) so this doesn't mislead a future
session.

**The gap this exposes in D6/D9's own methodology:** both of those
investigations held `any_order=False` throughout and only varied
`shared_numeric_vocab`. **`any_order`'s own incremental cost — on top of
whatever `shared_numeric_vocab` alone costs — was never actually
isolated.** D9's mechanism hypothesis (pooled embeddings need
`token_type_ids` alone to disambiguate columns) is a real, still-standing
explanation for `shared_numeric_vocab` alone's residual ~2x gap — but
the user's concern points at a *separate, additional* mechanism specific
to `any_order`'s permutation, layered on top.

**Testing directly:** `adult_any_order_isolation.py` (task `buvj9xoxk`)
reruns D9's `shared_numeric_vocab=True` condition (same 5-column Adult
subset, same sensitivity-protected/best-checkpoint training) with
`any_order=True` added, to compare against D9's existing
`shared_numeric_vocab=True, any_order=False` result directly:
age KS=0.0804, fnlwgt KS=0.2026, sex TVD=0.3531, workclass TVD=0.1790,
discriminator=0.6900, TSTR ratio=0.9508, DCR mean=0.3009,
frac_suspicious=0.0350. If `any_order=True` is meaningfully worse than
this on top of `shared_numeric_vocab`'s own cost, that confirms the
user's hypothesis as a second, additive mechanism. Pending.

### [2026-08-31 10:3x] D10 result — precisely confirms the hypothesis, and only where it should apply

| Metric | `shared_vocab=False` (baseline) | `shared_vocab=True`, fixed order (D9) | `shared_vocab=True`, **any_order=True** (D10) |
|---|---|---|---|
| age KS | 0.058 | 0.080 | **0.121** (worse) |
| fnlwgt KS | 0.093 | 0.203 | **0.324** (much worse) |
| sex TVD | 0.131 | 0.353 | **0.135** (much better) |
| workclass TVD | 0.070 | 0.179 | **0.028** (much better) |
| discriminator AUC | 0.586 | 0.690 | 0.651 (mild improvement) |
| TSTR ratio | 0.998 | 0.951 | **0.835** (notably worse) |
| DCR mean | 0.187 | 0.301 | 0.214 (closer to honest baseline) |
| frac_suspicious | 0.048 | 0.035 | 0.047 |

**The two multi-chunk numeric columns get substantially worse under
`any_order`** (age KS 0.080→0.121, fnlwgt KS 0.203→0.324 — both directly
on top of `shared_numeric_vocab`'s own already-measured cost) **while
the two single-token categorical columns get substantially better**
(sex TVD 0.353→0.135, workclass TVD 0.179→0.028). DCR shows no
memorization story (any_order's copying-risk profile is fine, similar
to or better than fixed-order's).

**This is exactly the pattern the proposed mechanism predicts, not a
coincidence.** Categorical columns are single-token — there is no
"within-column chunk position" to disambiguate, so `any_order`'s
position-scrambling has nothing to break for them, and they may even
benefit from the more order-invariant representation any-order training
forces the model to learn overall. Multi-chunk numeric columns are
exactly where the mechanism bites: `token_type_ids` alone can't tell the
model which chunk-index-within-column it's generating once absolute
position no longer reliably encodes that, under permutation.

**Verdict: the user's architectural hypothesis is confirmed as a real,
distinct, second mechanism — not a replacement for D9's
embedding-pooling-difficulty explanation, but an additional, additive
cost layered on top, specific to `any_order` and specific to multi-chunk
numeric columns.** This sharpens D6/D9's finding further: `shared_numeric_vocab`
alone costs ~2x on marginal fidelity (D9); `any_order` on top of it costs
meaningfully more, but only on the numeric side, and the categorical
side actually improves.

**A concrete, well-motivated fix idea falls out of this diagnosis**
(not implemented, a real next step if this is worth pursuing further):
give multi-chunk numeric columns an explicit within-block relative
position signal that survives `any_order`'s permutation — e.g. a small
additional embedding table indexed by offset-within-chunk-group (0, 1,
2, ... within a column's own digit sequence, independent of where the
whole block landed in the sequence), added the same way `token_type_ids`
already is. This is a scoped, targeted fix (touches only the numeric
partition-column path, leaves the already-fine categorical handling
alone) rather than a redesign — but it's a real implementation task, not
attempted here without checking whether it's worth the further time
investment.

---

## D11 — Decoupling `any_order` from `shared_numeric_vocab`: built, tested, shipped

**User's design question, confirmed correct by tracing the code:** the
`any_order=True requires shared_numeric_vocab=True` constructor check
existed purely because `any_order` reused `shared_numeric_vocab`'s
`token_type_ids` machinery for column identity, not because `any_order`
mechanically needs it. Every training-time mechanism
(`compute_column_blocks`, `AnyOrderColumnCollator`) was already
unconditional on `shared_numeric_vocab` — the only real coupling was the
constructor's explicit `raise` and `_build_order_masks` (sampler)
unconditionally reading `vocab["column_type_ids"]`, a key that only
exists when `shared_numeric_vocab` built the vocab.

**Why this matters beyond cleanliness, tying directly back to D10:** a
non-pooled column's vocabulary is already fully disjoint *per column and
per digit-chunk position* (`encode_column_values` prefixes every value
with its exact processed column name, partition suffix included, before
tokenizing) — every token is self-identifying regardless of where
`any_order`'s permutation moves it. `shared_numeric_vocab`'s pooling is
what removes that self-identification, forcing reliance on
`token_type_ids` + position — and D10 showed position is exactly what
`any_order`'s permutation breaks for multi-chunk numeric columns.
Decoupling doesn't patch the bug D10 found; it sidesteps it by
construction for anyone who doesn't specifically need
`shared_numeric_vocab`'s embedding-pooling benefit.

**Built:**
1. `realtabformer2.py`: constructor now validates `any_order` against
   `model_type=='tabular'` directly (what it actually needs, previously
   reached only transitively via requiring `shared_numeric_vocab`) instead
   of raising when `shared_numeric_vocab` is off. Docstring rewritten.
2. `rtf_sampler.py::_build_order_masks`: `token_type_ids_seq` is `None`
   when the vocab has no `column_type_ids`, instead of `KeyError`. Traced
   every downstream consumer to confirm `None` already propagates
   correctly (`_generate`'s override handling already treated `None` as
   "no override" — this was already correct, just needed the crash
   upstream of it removed).
3. **A real, separate pre-existing bug found while testing this** (not
   introduced by this change): `col_type_ids_seq` never had a default
   value in `__init__` — unlike every other optional/beta attribute on
   the class, it only got assigned inside `_fit_tabular`'s
   `shared_numeric_vocab` branch, so `model.col_type_ids_seq` raised
   `AttributeError` (not `None`) on any model that never used
   `shared_numeric_vocab`. Caught immediately by my own smoke test, not
   found by reading. Fixed with a default; one existing test
   (`assert not hasattr(...)`) was asserting this broken behavior
   directly and got corrected to assert the right default instead.

**Tested:** 4 new tests mirror the existing `shared_numeric_vocab=True`
`any_order` test suite exactly (fit+unconditional sample, seed on last
column with both values, seed on a middle column with a gap, save/load
roundtrip) — all pass. Full suite: 102 passed, only the 2 known
pre-existing unrelated failures. Every pre-existing `any_order` test
(the `shared_numeric_vocab=True` combination) still passes unchanged —
purely additive, nothing removed. Committed (`fd9b6fc`), merged into the
local `feat/support-seed-input` worktree. Not pushed.

**Empirical verification launched:** does the decoupled combination
actually deliver the promised quality benefit, not just avoid crashing?
Rerunning the exact D10 setup (same 5-column Adult subset,
sensitivity-protected/best-checkpoint training) with
`any_order=True, shared_numeric_vocab=False` this time, to compare
directly against:
- D9 baseline (`shared_numeric_vocab=False, any_order=False`): age
  KS=0.058, fnlwgt KS=0.093, sex TVD=0.131, workclass TVD=0.070,
  discriminator=0.586, TSTR ratio=0.998.
- D10 (`shared_numeric_vocab=True, any_order=True`): age KS=0.121,
  fnlwgt KS=0.324, sex TVD=0.135, workclass TVD=0.028,
  discriminator=0.651, TSTR ratio=0.835.

Expectation, stated before seeing the result: numeric columns (age,
fnlwgt) should land much closer to the D9 baseline than D10's numbers,
since the decoupled version has no pooled-vocab position-ambiguity
problem to pay for. Launched as `adult_any_order_decoupled_verify.py`
(task `bvpgkx0kg`). Pending.

### [2026-08-31 12:0x] D11 verification result — the fix works, and exceeds expectations on most metrics

| Metric | D9 baseline (no shared_vocab, no any_order) | D10 (shared_vocab + any_order, coupled) | **D11 (any_order, decoupled)** |
|---|---|---|---|
| age KS | 0.058 | 0.121 | **0.026** (better than D9) |
| fnlwgt KS | 0.093 | 0.324 | **0.028** (better than D9) |
| sex TVD | 0.131 | 0.135 | **0.008** (much better than both) |
| workclass TVD | 0.070 | 0.028 | 0.109 (worse than both) |
| discriminator AUC | 0.586 | 0.651 | **0.544** (best of the three) |
| TSTR ratio | 0.998 | 0.835 | 0.987 |
| DCR mean | 0.187 | 0.301 | 0.251 |
| frac_suspicious | 0.048 | 0.035 | 0.072 |

**Not just "avoids D10's bug" — the decoupled combination beats the
plain non-any_order baseline (D9) on both numeric columns, the
discriminator, and one of the two categoricals.** age KS more than
halves relative to D9 (0.058→0.026) and drops 4.7x relative to D10
(0.121→0.026); fnlwgt KS drops 3.3x relative to D9 and 11.6x relative to
D10. Discriminator AUC (0.544) is the closest to the 0.5 ideal of any
run in this entire investigation thread.

**Plausible explanation, stated as a hypothesis, not proven:** `any_order`
training exposes the model to many different column orderings of the
same data — effectively a form of data augmentation. On a small dataset
(2400 rows), that could genuinely improve generalization beyond what
fixed-order training gets, independent of the arbitrary-conditioning
capability being the point of the feature. This would explain beating
the fixed-order baseline, not just matching it.

**Not a clean sweep — reported honestly:** `workclass` TVD is worse
(0.109 vs D9's 0.070) and `frac_suspicious` (0.072) is somewhat elevated
versus both other runs (0.048, 0.035) — still the same order of
magnitude as the "natural," no-signal ~0.05 floor, not alarming, but
worth noting rather than glossing over. Single run, single seed, single
dataset, same caveat as everything else in this log.

**Conclusion: the decoupling fix is validated, not just functionally but
empirically.** `any_order=True, shared_numeric_vocab=False` is now a
genuinely strong, recommended configuration for arbitrary-subset
conditioning — better evidenced than the plain quantile-encoding-only
baseline on several axes, on top of unlocking a real capability
(conditioning on any column subset) that baseline doesn't have at all.
`shared_numeric_vocab` (with or without `any_order`) remains not
recommended pending its own, separate fix.

**FINAL SYNTHESIS updated accordingly** (see below) — the recommended
configuration now includes `any_order=True, shared_numeric_vocab=False`
as a strong option specifically when arbitrary-subset conditioning is
needed, not merely "usable" but empirically ahead of the plain baseline
in this test.
