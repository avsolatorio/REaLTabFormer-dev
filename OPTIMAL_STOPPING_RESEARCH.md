# Optimal Stopping for Overfitting Detection — Research Log

Companion document to `DECISION_LOG.md` (that file covers REaLTabFormer
configuration optimization; this one is a focused research thread on
replacing/improving the DCR-bootstrap overfitting-detection mechanism
with something faster, more principled, and — per the user's explicit
ask — provably optimal and written up rigorously enough to be
publishable).

## 0. The ask, precisely

Replace or improve `_train_with_sensitivity`'s bootstrap-DCR early
stopping with a method that:
1. Requires no additional held-out data (confirmed in conversation: the
   *existing* method already doesn't use external hold-out — it
   bootstraps from the training data itself — so the bar is to be
   **better**, not just to also avoid hold-out).
2. Doesn't require expensive generation calls (the current method's
   real cost driver — autoregressive `.generate()` of ~1000+ samples
   every `n_critic` epochs).
3. Works at any training-budget scale, including single-epoch or
   sub-epoch training on very large datasets (the current method's
   granularity is denominated in whole epochs via `n_critic`, and
   literally cannot check mid-epoch — confirmed by reading
   `_train_with_sensitivity`'s loop: `for p_epoch in range(last_epoch,
   self.epochs, n_critic)` collapses to a single iteration when
   `epochs=1`).
4. Has a genuine, citable optimality property, not just "seems to work
   empirically."

## 1. Reframing: this is a change-point detection problem, not a fixed-sample hypothesis test

My first pass in conversation reached for Wald's SPRT (Sequential
Probability Ratio Test) — provably optimal (Wald & Wolfowitz, 1948) in
the sense of minimizing *expected sample size* for choosing between two
*fixed* hypotheses at bounded error rates. That's the right tool for "is
this fixed batch of data drawn from distribution A or B." **It is not
quite the right tool for what's actually needed here.**

The real question is different: training is an *ongoing process*, and
somewhere during it — at an *unknown* point — the model's behavior on
fresh vs. already-seen data transitions from "learning the true
distribution" to "memorizing specific rows." That's a **change-point
detection** problem: detect, as quickly as possible, the moment an
ongoing stream's statistics shift from one regime to another, subject to
a bounded false-alarm rate. The classical, and *also* provably optimal,
tool for exactly this is **Page's CUSUM (cumulative sum) test** (Page,
1954), and its optimality is even more directly relevant here: **Lorden's
theorem (1971)** proves CUSUM asymptotically minimizes the *worst-case
expected detection delay* among *all* possible stopping rules, for a
given false-alarm-rate (average-run-length) constraint. That is a
precise, textbook mathematical answer to "optimal stopping" as the user
posed it — not a metaphor.

**Decision: use CUSUM as the primary stopping rule for the ongoing
training-time monitor, and keep (a restarting/repeated form of) SPRT
available as a secondary, fixed-sample diagnostic** — e.g. for validating
that the calibrated null hypothesis is actually accurate at a given
checkpoint, a genuinely different question ("is my assumed p0 correct
right now") from "has the process changed regime yet."

## 2. The test statistic

Per the conversation: for a row, teacher-forced argmax exact-match
(does `argmax(logits)` at every position reproduce the true row exactly)
gives a strict Bernoulli event, computable from a forward pass that's
already happening (loss computation) with zero extra generation. Define,
for a batch of size `m` at training step `t`:

- `X_i in {0,1}`, i = 1..m: row i's whole-row exact-match indicator.
- Two populations per check: **fresh** (rows never yet included in a
  completed gradient step) and **seen** (a small rotating reference pool
  of previously-trained rows, re-scored at current weights).

## 3. Calibrating the null hypothesis from data, not a guess

A well-calibrated (non-memorizing) model's Bayes-optimal single guess at
any position is the *mode* of its learned conditional distribution, and
its expected argmax accuracy at that position is exactly the
**probability mass of the mode** under the true distribution — not
`1/K` in general (that's only a good approximation for a near-uniform
position, which happens to describe quantile-encoded fractional digits
specifically, by design — see D1 in `DECISION_LOG.md`). The correct,
general quantity is directly estimable from the training data itself,
no theory needed: `p0_c = value_counts(normalize=True).max()` for each
processed column/chunk `c`. Under a conditional-independence
approximation across positions (an approximation, not exact — flagged
honestly, correctable empirically, see below), `p0_row = prod(p0_c)`.

**This connects directly back to this session's earlier entropy work**:
`numeric_quantile_encoding` was built specifically to push per-position
entropy toward the maximum, i.e. to push `p0_c` toward `1/K` (its lowest
possible value for K outcomes) — meaning quantile-encoded columns give
the *cleanest*, most well-separated null hypothesis for this test (a
low, well-characterized `p0`), while non-quantile-encoded heavy-tailed
columns (with their near-constant leading chunks) would give a
*high*, uninformative `p0_c` for those positions — the test's power is
itself a beneficiary of D1's fix, not an independent concern.

**Self-correction for the independence approximation:** rather than
trust the theoretical product-of-marginals formula blindly, use it only
as a sanity-check prior; the *actual* calibration source is the
empirically observed fresh-batch exact-match rate during an initial
warm-up window (e.g. the first few hundred steps), before memorization
has had any chance to set in. If dependence between columns makes the
true joint mode probability higher than the product of marginals, the
empirical estimate will reflect that automatically. Theory gives
plausibility and an initial estimate; data gives the real threshold.

## 4. The alternative hypothesis and the CUSUM statistic

Choose `p1` as the smallest rate worth caring about — not "fully
memorized" (rate to 1), but a meaningful floor above `p0` (e.g. a fixed
multiple, tunable). Page-CUSUM on the log-likelihood ratio:

```
llr(X) = X * log(p1/p0) + (1-X) * log((1-p1)/(1-p0))
S_0 = 0
S_n = max(0, S_{n-1} + llr(X_n))
alarm when S_n >= h
```

`h` controls the false-alarm rate (average run length under H0);
standard CUSUM design tables / Monte Carlo calibration on this exact
model's own null-hypothesis simulations give the right `h` for a chosen
tolerance — to be calibrated empirically against the existing
DCR-bootstrap method's own false-positive behavior as a sanity check,
not assumed.

**Run this on the *gap* between seen and fresh populations, not on
either alone** — feed `S_n` from the `seen` population's exact-match
stream, but with `p0` calibrated from the *fresh* population's
concurrent empirical rate (not a fixed constant) — this makes the test
self-normalizing against any drift in the fresh-rate baseline itself
(e.g. from a learning-rate schedule change), isolating specifically the
seen-vs-fresh divergence that memorization actually produces, which is
a real design choice worth stating explicitly and testing.

## 5. Positioning against prior art (found via literature check, not assumed absent)

- Wald & Wolfowitz (1948): SPRT optimality — foundation, not what's used
  as the primary rule here.
- Page (1954); Lorden (1971): CUSUM and its minimax optimality for
  change-point detection — the actual foundation for the stopping rule.
- Toneva et al. (ICLR 2019, "forgetting events"), Swayamdipta et al.
  (EMNLP 2020, "Dataset Cartography"): per-example training-dynamics
  signals for memorization/difficulty — same family of intrinsic signal
  as the argmax exact-match statistic here, established in the
  supervised-classification setting.
- [Data Cartography for Detecting Memorization Hotspots and Guiding Data
  Interventions in Generative Models](https://arxiv.org/html/2509.00083)
  (2025) — closest prior art found: per-example difficulty + forget-event
  memorization scores for *generative* models, with a uniform-stability
  generalization-gap argument. Does not use sequential change-point
  detection or a formal stopping rule as far as this search surfaced —
  that combination, applied to autoregressive tabular generation
  specifically as a *training-time, generation-free replacement* for a
  DCR-bootstrap early-stopping mechanism, looks like the actual novel
  contribution here, not the individual pieces.
- Loss-based membership inference (Yeom et al. 2018; Carlini et al.
  2022) and sequential-metric MIA (SeqMIA) establish the general
  validity of training-dynamics-based membership/memorization signals
  and hypothesis-testing framings for MIA — relevant grounding for why
  the underlying statistic (argmax exact-match as a memorization proxy)
  is a reasonable choice, distinct from this document's specific
  contribution (the CUSUM-based training-time stopping rule built on it,
  calibrated via digit-entropy).
- [Synth-MIA](https://arxiv.org/pdf/2509.18014) (2025): a testbed
  specifically for tabular-synthesis privacy leakage — worth using as an
  external validation reference if this work matures toward a real
  writeup, not consulted further tonight given time constraints.

## 6. Implementation and validation plan

1. Prototype (standalone script first, not deep library integration yet
   — validate the idea before committing to an API surface): a
   lightweight training loop instrumented to record, every N steps,
   fresh-batch and seen-batch argmax exact-match rates, run the CUSUM
   statistic, and log when it would fire.
2. **Calibration check**: does the empirically observed fresh-batch rate
   early in training match the theoretical product-of-marginals
   estimate from the training data's own per-column mode frequencies?
   A real, falsifiable prediction, not assumed.
3. **Efficiency check**: real wall-clock comparison against the existing
   bootstrap-DCR mechanism on the same Adult setup used throughout this
   session — get actual numbers, not just complexity-order claims.
4. **Concordance check**: does the CUSUM-based alarm point roughly agree
   with where DCR-based sensitivity crosses its own threshold, on the
   same training run? And does it agree with the *actual* DCR/copying
   profile of the resulting model (exact-duplicate rate, `frac_suspicious`)
   — using this session's own D9-D11 measurement infrastructure as
   ground truth?
5. **Single-epoch / large-dataset stress test**: directly addressing the
   user's original concern — construct a scenario with a much larger
   row count and a small number of epochs (or sub-epoch step budget),
   confirm the method still fires appropriately using step-level (not
   epoch-level) granularity, which the existing mechanism structurally
   cannot do.
6. If validated, a real library integration (a new `Trainer` callback,
   threaded through `.fit()` as an alternative/complementary
   `overfitting_detection_method`) is a further, separate step — not
   attempted until the core statistical claim is empirically confirmed.

## Results log

### Prototype build (standalone script, `_fit_tabular` direct call)

Built `cusum_memorization_prototype.py` against the real REaLTabFormer
model+dataset+trainer (via `model._fit_tabular(df, device="cpu")`,
bypassing `.fit()`'s dispatch, with a custom instrumented training
loop). Two bugs fixed before any real run:

1. `_fit_tabular` alone doesn't set `model.trainer_kwargs` (normally set
   inside `.fit()` before dispatch) — fixed by setting it manually
   (`{}`) before calling `_fit_tabular` directly.
2. `_fit_tabular` alone doesn't place the model on a device — the
   trainer's model stayed on the default device while manually-built
   tensors defaulted to CPU, causing a device-mismatch error. Fixed with
   `net = trainer.model.to("cpu")`.

Verified bit-for-bit that a manual `shift_logits`/`shift_labels` +
`F.cross_entropy` replicates HF's internal `out.loss` exactly
(`4.165428161621094 == 4.165428161621094`), establishing the custom
loop's forward-pass math is correct before building the exact-match
statistic on top of it.

### Statistic design: whole-row exact match is unobservable, needed a probe column

Whole-row (13-column) product-of-marginals mode probability came out to
~1.68e-13 — essentially never observable across 2400 total fresh
exposures. Redesigned around a single "probe column" whose own
digit-chunk positions carry the exact-match test, restricted via
`probe_positions` (verified the position-indexing: `processed_columns[i]`
maps to shifted-tensor index `i`, no extra offset, via careful
derivation from the shift-by-one convention).

### Probe-column selection: independence assumption fails for low-cardinality numeric columns

Initial candidate `hours-per-week` was chosen using the
product-of-marginals *theoretical* p0 (~0.058, computed assuming
independence across digit-chunk positions). A first smoke test's
*empirical* fresh-batch rate came back at 0.4229 — 7x higher than
predicted. A standalone diagnostic (comparing the TRUE joint mode-match
rate — literally checking what fraction of rows' full digit-chunk
string equals the single most common combination — against the
independence-product estimate) confirmed this is real, not a bug:

| probe column | true joint mode-match rate | independence-product estimate | unique values |
|---|---|---|---|
| fnlwgt | 0.0017 | 1.184e-4 | 2299 |
| hours-per-week | 0.4704 | 5.673e-2 | 71 |
| age | 0.0329 | 2.554e-4 | 63 |
| capital-gain | 0.9154 | 7.189e-1 | 58 |

Interpretation: low-cardinality, heavily-clustered numeric columns
(`hours-per-week`, `age`) have real positive dependence between
digit-chunk positions — many rows literally share the identical
digit-chunk string — which the independence assumption misses entirely.
`capital-gain` is dominated by its zero point-mass (already
near-saturated, useless as a null). `fnlwgt` (2299 unique values across
2400 rows, essentially all-distinct) has the least dependence-inflation
in absolute terms and by far the lowest, most discriminative true rate
— selected as the probe column going forward.

**Design consequence**: since the true joint mode-match rate is
computable directly and exactly from the training data itself (no
independence assumption, no sampling noise), it should be — and now is
— the actual CUSUM calibration source (`p0_calibrated`), computed once
upfront before training starts, rather than estimated noisily from an
early-training warmup window. This is strictly better for a rare-event
probe column like `fnlwgt` (true rate 0.17% means only ~4 expected
successes across all 2400 one-time fresh exposures — far too few to
calibrate reliably from warmup alone). The warmup-window fresh-rate is
kept only as a diagnostic (does the model's own early-training behavior
converge toward the data-derived value, as theory predicts for a
well-calibrated model), not as a calibration input.

### CUSUM threshold calibration: effect-size sensitivity, and a degenerate-threshold failure mode found and fixed

Two bugs found via direct Monte Carlo simulation before committing to a
real run:

1. **Rate-as-single-observation LLR bug**: the batch log-likelihood
   ratio was initially computed by treating the batch's aggregate
   success *rate* as a single Bernoulli observation
   (`llr = x*log(p1/p0) + (1-x)*log((1-p1)/(1-p0))`), silently
   discarding the actual sample size and understating true evidence by
   a factor of `n`. Fixed by switching to the correct batch-level
   (binomial) LLR using `k` (observed successes) and `n` (batch size)
   directly.
2. **Degenerate threshold from an overly large effect size**: with
   `p1 = 10 * p0`, Monte Carlo calibration of `h` (99th percentile of
   the null's max-CUSUM-path over many simulated runs) collapsed to
   exactly 0 — the null hypothesis never produced a single
   positive-LLR batch at `n_seen=256`, so *any* real fluctuation would
   fire, providing zero real false-alarm protection despite superficially
   "passing" calibration. A sweep over effect sizes (1.5x, 2.0x, 3.0x,
   5.0x, 10.0x) at `p0=0.058` (the original hours-per-week estimate)
   found 1.5x well-behaved (non-degenerate `h≈8.74`, median ~5-check
   detection delay for a true shift to exactly `p1`).

Re-verified the same sweep at fnlwgt's much lower, data-derived
`p0=0.0017` (291-check horizon, matching the full 75-epoch plan:
`2400/32=75 steps/epoch × 75 epochs / 20-step check interval + 10`):

| effect (p1/p0) | p1 | h (99th pct null max) | detect rate / 300 checks | median detection delay |
|---|---|---|---|---|
| 1.5x | 0.00255 | 6.761 | 0.980 | 112 checks |
| 2.0x | 0.00340 | 7.853 | 1.000 | 41 checks |
| 3.0x | 0.00510 | 7.839 | 1.000 | 13 checks |
| 5.0x | 0.00850 | 8.763 | 1.000 | 5 checks |
| 10.0x | 0.01700 | 7.636 | 1.000 | 2 checks |

No degeneracy at any effect size at this lower p0 (`frac_null_zero=0.000`
throughout) — confirms the earlier degenerate-threshold failure was
specifically an interaction between the *large* p0=0.058 and the
*extreme* 10x effect size, not a general property of large effect
sizes. Kept 1.5x for consistency with the earlier design decision and
because it gives the most conservative (highest-power-per-false-alarm)
non-degenerate design of the options tested.

`calibrate_cusum_threshold()` runs this simulation dynamically in the
prototype against the actual data-derived p0/p1/horizon at runtime,
rather than hardcoding a constant — the calibration is now fully
data-driven and reproducible, no manual tuning.

### v1 status (binary probe-column statistic)

Prototype updated to use `PROBE_COLUMN="fnlwgt"` with upfront
data-derived calibration (`p0_calibrated`, `p1=1.5×p0_calibrated`,
`CUSUM_H` from Monte Carlo simulation against the real 291-check
horizon). A 5-epoch smoke test confirmed non-degenerate calibration
(`CUSUM_H=3.294`) and no spurious alarm. Superseded in practice by v2/v3
below (whole-row, no probe-column restriction), but kept as a working,
validated reference implementation.

### v2: whole-row continuous statistic (per-example teacher-forced NLL)

Prompted by the user directly questioning the whole-row-unobservability
workaround: "why not use sum of logs to compute the full row statistic
to prevent the extremely low probability value?" Correct diagnosis --
v1's 1.68e-13 whole-row probability isn't a numerical underflow problem
(doubles handle down to ~1e-308), it's a **statistical power** problem:
binarizing whole-row correctness makes it the AND of ~15-20 per-position
events, collapsing the probability multiplicatively so the event
essentially never occurs even once across 2400 fresh exposures, and
discarding all partial-confidence information in the process.

Built `cusum_loglik_prototype.py` (v2): per-row score is the mean
per-token teacher-forced NLL over ALL of the row's valid target
positions (not just one probe column) -- computed for free from the
same forward pass, using `F.cross_entropy(..., reduction="none")`
instead of the batch-mean `out.loss`. Summing/averaging logs doesn't
collapse multiplicatively, so no probe-column restriction is needed --
restores the full whole-row signal v1 had to give up. This is also the
same statistic underlying established loss-based membership inference
(Yeom et al. 2018; Carlini et al. 2022, already cited) -- v1's binary
exact-match was a weaker approximation to this.

CUSUM reformulated from the binomial LLR to the classical Gaussian
mean-shift CUSUM (Page 1954) on a two-sample z-score between fresh-row
and seen-row mean loss per check:
```
Delta_t = mean(fresh_loss) - mean(seen_loss)
Z_t = Delta_t / SE(Delta_t)          (two-sample standard error)
llr(Z) = delta*Z - delta^2/2         (target shift delta)
```
Simpler to calibrate than the binomial version -- depends only on
`delta` (target effect size in z-units) and the check horizon, not on
`p0`/`p1`/`n_seen`. Swept `delta` at the real 291-check horizon:

| delta | h (99th pct null max) | detect rate/300 | median delay |
|---|---|---|---|
| 0.25 | 6.347 | 0.894 | 148 |
| 0.5 | 7.453 | 1.000 | 50 |
| 0.75 | 8.280 | 1.000 | 27 |
| 1.0 | 8.507 | 1.000 | 16 |
| 1.5 | 8.498 | 1.000 | 8 |
| 2.0 | 8.723 | 1.000 | 5 |

No degeneracy at any effect size. Chose `delta=0.5` (100% detection,
sensitive without being trigger-happy).

**Bug found via smoke test, not assumed away: the post-gradient-step
recency confound.** A 5-epoch smoke test fired a CUSUM alarm at step 20
(epoch 0.27 -- implausibly early; the reference D-series run showed
sensitivity crossing around epoch 45-50) with Z=20.98, fresh_loss=2.034,
seen_loss=0.949. Diagnosis: at step 20, the "seen" reference pool
consisted entirely of rows from the last <20 steps -- their lower loss
reflects the ordinary post-gradient-step fit from SGD (a row just
updated on fits it directly; that fit decays over subsequent steps
unless reinforced by repeated exposure), not accumulated memorization.
Comparing "ever-seen" against "never-seen" this early conflates normal
learning progress with overfitting.

**Fix: a cooldown window.** Exclude rows from the seen-reference pool if
touched within the last full epoch (`COOLDOWN_STEPS = steps_per_epoch`),
so the seen measurement reflects *retained* fit from at least one epoch
ago rather than a transient post-update spike. Applied to both v1 and
v2/v3 for consistency (v1 didn't happen to trigger the bug in its own
5-epoch smoke test, but has the identical structural vulnerability).
Re-ran the 10-epoch smoke test with the fix: no spurious alarm, `no
alarm fired` after 10 epochs (589.8s wall clock) -- confirms the fix
resolves the confound without needing to wait unreasonably long before
the CUSUM can start accumulating real evidence.

### v3: gated (Winsorized) per-token statistic

User's follow-up question, verbatim: "What do you think about using the
argmax to mask which align with the actual token, then if argmax is the
actual token, use the log of the probability, while if the argmax is
not the token, then it should be some value." Confirmed sound: v2's raw
per-token NLL is unbounded in the "wrong" regime -- a single genuinely
hard/ambiguous row where the model confidently misses can inject an
arbitrarily large outlier into whichever population it lands in at a
given check, degrading the Gaussian-CUSUM approximation (which assumes
well-behaved per-check noise) via heavy-tailed contamination.

Built `cusum_gated_loglik_prototype.py` (v3):
```
score(token) = log P(true_token)     if argmax(logits) == true_token
             = log(1 / vocab_size)   if argmax(logits) != true_token
```
The wrong-branch floor is a fixed "no better than a uniform guess over
the model's own vocabulary" penalty -- a principled, simple choice that
doesn't require per-position calibration. This Winsorizes the statistic:
continuous, informative credit exactly where it's diagnostic of
memorization (already correct -- how confidently?), a flat capped
penalty everywhere else, since degree-of-wrongness beyond "not the top
choice" isn't strongly diagnostic of memorization either way. Same
additive, whole-row aggregation as v2 (no probe-column restriction);
same Gaussian-CUSUM/cooldown machinery, unchanged (the delta sweep
depends only on the z-unit target shift, not the underlying statistic's
scale, so v2's calibration table carries over directly). Sign
convention flips to log-likelihood (higher = better): `Delta =
seen_score - fresh_score`, positive = memorization signal.

10-epoch smoke test launched to confirm behavior before scaling up.

v3's 10-epoch smoke test completed cleanly (no spurious alarm,
591.7s wall clock, essentially identical to v2's cost -- the gating adds
no measurable overhead). Launched the full 75-epoch v3 validation run
(including the DCR ground-truth comparison) as the primary validation.

### v4: a serious bug found by the user's direct question, and the within-row redesign it required

While the full v3 run was in progress, the user asked directly: "how do
you identify fresh when the epoch is more than 1? Isn't it the case
that all data has already been seen at this point? What's fresh and
seen?" This caught a real, previously-unnoticed bug, confirmed by
tracing the code (not assumed):

`seen_indices` is a Python `set()` that only grows
(`seen_indices.update(batch_idx)` every step) and is never reset.
"Fresh" means `idx not in seen_indices`. Since each epoch is a full
permutation over ALL 2400 rows, by the end of epoch 1 every row index is
already in `seen_indices`. From epoch 2 onward, `fresh_mask` is always
all-False -- `fresh_scores`/`fresh_losses` never gets a new entry again,
and since it's unconditionally reset to `[]` after every check window
regardless of whether it had any entries, `fresh_mean` is `nan` at
*every* check from epoch 2 onward. The CUSUM update was gated on
`not np.isnan(fresh_mean)`, so **the CUSUM statistic silently froze**
at whatever value it reached during epoch 1's first ~3 check windows
(check_every=20, steps_per_epoch=75, so only steps 20/40/60 fall inside
epoch 1) and never updated again for the remaining 74 epochs. The
already-running 75-epoch v3 validation job was killed once this was
understood -- its result past epoch 1 would have been meaningless (a
guaranteed "no alarm fired" or a fluke early alarm, neither informative
about the actual epoch-45-50 question).

Root cause, stated plainly: "fresh" (never-yet-trained) is a ONE-TIME,
exhaustible resource on a small, fixed-size, many-epoch dataset (2400
rows, 75 epochs) -- it runs out after a single pass. Comparing two
*populations* (a fresh set vs a seen set) at the same point in time
doesn't work once one of those populations is empty for the rest of
training. This is exactly the "large dataset / few epochs" vs "small
dataset / many epochs" distinction raised earlier in this research
thread (Section 0) -- the fresh/seen population design is well-suited to
the former (fresh data is continuously available throughout) but
structurally broken for the latter (REaLTabFormer's typical use case).

**Fix (v4, `cusum_baseline_prototype.py`): compare each row against
itself across TIME, not against a different population at the same
time.** During epoch 1 (each row's one-time first exposure, before its
own first gradient step -- genuinely prequential, zero extra cost),
record `baseline_score[row_idx]`, the row's per-token gated score (same
statistic as v3). This is a snapshot stored once per row, not a
resampled population, so it never runs out. At every later check (any
epoch, 1 through 75), sample the same cooled seen-pool as v2/v3, and for
each sampled row compute the PAIRED difference against its own baseline:
`improvement_i = current_score_i - baseline_score_i`. This is also
statistically better than the old unpaired comparison, not just a
workaround: it controls for row-to-row difficulty directly (a hard row
is compared only to its own hard baseline).

Ordinary (non-memorizing) training keeps improving every row's score for
a while, purely from general model convergence -- expected, not
suspicious on its own. Calibrate what "normal" improvement looks like
(`mu0`, `sigma0`) from an early window of `WARMUP_CHECKS=10` checks
(spanning roughly epochs 2-4, well before the reference D-series run's
~epoch 45-50 memorization onset). Unlike v1's `p0` (which had an exact,
closed-form, data-only calibration via the true joint mode-match rate),
this calibration genuinely needs to be model/data-driven -- there's no
closed-form "expected improvement from ordinary learning" computable
from the static data distribution alone, since it depends on the
model's own convergence trajectory. Stated honestly as a real
limitation, not glossed over. From then on: `Z_t = (Delta_t - mu0) /
sqrt(var_t/n_t + sigma0^2)`, fed into the same Gaussian mean-shift CUSUM
as v2/v3 (`delta=0.5`, unchanged -- the calibration sweep depends only
on the target z-unit shift, not on what statistic feeds Z).

10-epoch smoke test confirmed the fix works: baseline scores populate
during epoch 1, warmup calibration triggers with sensible values
(`mu0=0.7359`, `sigma0=0.5786`, from checks at steps 80-260, i.e.
epochs ~1.1-3.5), the CUSUM keeps accumulating (non-frozen) evidence
throughout all 10 epochs, and no spurious alarm fires (expected/correct
this early). Wall clock 645.9s for 10 epochs vs v2/v3's ~590s (~10%
overhead from the added baseline-dict bookkeeping and paired-diff
arithmetic -- acceptable).

Launched the full 75-epoch v4 validation run (including the DCR
ground-truth comparison), expected ~80-90 minutes based on the smoke
test's per-epoch cost.

**Result: the CUSUM alarm fired at step 2980 (epoch 39.7), S=8.04,
Z=0.73, Delta=1.1606 (mu0=0.7359).** This is the first genuine
concordance evidence for the whole approach: it lines up closely with
the independent reference finding from earlier in this session (the
DCR-bootstrap sensitivity mechanism crossing its own threshold around
epoch 45-50 on the same Adult setup), measured via a completely
different method (expensive `.generate()`-based DCR bootstrap vs. this
generation-free, training-time statistic). Two independent measurement
approaches landing in the same region is real evidence the statistic
tracks actual memorization risk, not an artifact of one specific
implementation. Full concordance check (comparing against this same
run's own DCR ground truth, computed after epoch 75 completes) still
pending -- run in progress.

### Reframing: epoch-adaptive stopping instead of monitoring a fixed schedule

User's follow-up, verbatim: "why tie it to epochs? Can't this measure be
used to amortize and decide whether another epoch should be added? ...
instead of defining the total epochs upfront, use this condition to
decide if the model should be trained one epoch more." This is a better
framing than what v4 does (passively monitoring inside a pre-fixed
`epochs=75` schedule) -- it also directly resolves the dataset-size
dependency found in the previous exchange, without needing to retune
`COOLDOWN_STEPS` to some other unit: if the decision is made **once per
completed epoch** rather than every N steps, "epoch" is already a
dataset-size-invariant unit, so none of the step-count tuning is
needed. Proposed design:
```
train epoch 1   (records per-row baselines, as v4 already does)
train epoch 2   (first re-measurement -- mandatory minimum, see below)
loop:
    compare this epoch's performance on already-trained rows against
    their baseline (v4's paired-improvement statistic, aggregated once
    per epoch instead of every 20 steps)
    feed into the CUSUM
    if alarm: stop, roll back to the last checkpoint before the climb
    else: train one more epoch
```
Total epoch count is no longer a required upfront input -- it emerges
from the procedure. Calibration of "normal" epoch-over-epoch
improvement comes from the first couple of epoch transitions instead of
an arbitrary step-count warmup window.

**Inherent limitation, stated honestly, not glossed over**: any
reactive stopping rule needs at least a little data past the true
change point before it can detect it -- this design needs a minimum of
~2 completed epochs before its first decision is even possible (epoch
1's baseline and epoch 1's "current" are the same measurement). This
isn't a flaw specific to this design; it's exactly what Lorden's
theorem is about -- CUSUM *minimizes* expected detection delay, it
doesn't eliminate it. Standard mitigation: keep the last checkpoint from
before the CUSUM started climbing, not just the checkpoint at alarm
time, so the unavoidable delay costs monitoring time, not model
quality.

**What this does and doesn't solve for the large-dataset/single-epoch
case**: if a dataset can only afford exactly one epoch of training,
no reactive method (this one included) can tell you in advance whether
that one epoch already overfit -- there's no second data point to
compare against. But for the case that actually motivated the original
concern (large dataset, *maybe* only a few epochs needed, not
necessarily exactly one), this design handles it correctly: it stops
itself at epoch 2-3 if that's genuinely where risk starts, rather than
committing to a large fixed budget. The complete picture needs BOTH
mechanisms operating at their natural granularities: v2/v3's
within-epoch fresh-vs-seen comparison (catches risk during a single
long pass on a huge dataset, since fresh data is continuously available
within one epoch) and this new between-epoch comparison (catches risk
across repeated passes on a smaller dataset) -- not one replacing the
other.

**Epoch-level calibration verified via lightweight simulation** (no
model retraining needed -- pure Monte Carlo, ran alongside the ongoing
v4 full-run without CPU contention): non-degenerate at any reasonable
effect size (`delta>=0.5`) across horizons from 10 to 75 epoch-level
checks. Honest tradeoff: checking only once per epoch (vs. every 20
steps in v4) means less frequent evidence accumulation, so small effect
sizes (`delta=0.25`) lose real power at short horizons (5% detection
rate at a 10-epoch horizon) -- an inherent cost of coarser monitoring,
not a bug. Reassuring cross-check: at `delta=0.5` on a 75-epoch
horizon, simulated median detection delay is ~36 epochs; if the true
shift starts around epoch 3 (right after the mandatory 2-epoch warmup),
3+36=39 lines up closely with the actual v4 run's empirically observed
alarm at epoch 39.7 -- two independently-derived numbers (a real
model run's result, and a from-scratch calibration simulation)
landing in the same place.

**v4 full run completed. DCR ground truth: `frac_suspicious=0.8900`,
`dcr_synth mean=0.2314` (vs `dcr_test mean=2.4215`) at epoch 75** --
severe memorization if trained to the full, unprotected 75-epoch
schedule (consistent with, and slightly worse than, what the earlier
unprotected D-series reference run found). The CUSUM alarm fired at
epoch 39.7, well before this end state -- real advance warning, not a
signal that only appears after the damage is done. Honest limitation:
this run monitored throughout but never actually stopped, so this
confirms the alarm precedes catastrophic memorization, not that
stopping AT the alarm produces a materially safer model -- that
requires actually halting training there and measuring DCR on that
exact checkpoint, which is a different (and more decisive) experiment.
Launched `cusum_stop_at_alarm.py` (breaks out of training the instant
the alarm fires, then runs the same DCR comparison on that checkpoint)
to test this directly -- deterministic same seed/config, so it retraces
the same trajectory to step 2980 and should cost roughly half the full
run's compute.

### User's follow-up: "shouldn't this track continuously, not just at epoch boundaries?"

Correct instinct, and it caught a mistake in the epoch-adaptive proposal
above: epoch-only *checking* was solving a problem that only needed a
narrower fix. The real issue was `COOLDOWN_STEPS = steps_per_epoch`
being dataset-size-dependent -- not that monitoring itself needed to be
epoch-grained. Decoupling the two: cooldown just needs to be "long
enough for the post-gradient recency bump to fade," which is a property
of the optimizer/loss-landscape dynamics, not dataset size, so a fixed,
small, absolute step count should work regardless of how large the
dataset is -- and if so, v4 can go back to checking continuously every
20 steps (matching the user's "seamless tracking" instinct) with no
epoch-boundary gating needed at all.

**Tested directly**: a 12-epoch smoke test with `COOLDOWN_STEPS=40`
(fixed, ~0.53 epochs on Adult, not scaled to `steps_per_epoch`) ran
clean -- no false alarm, unlike the original no-cooldown version's
implausible step-20 alarm. This is real evidence the recency-bump decay
is a fixed-step-count phenomenon, not an epoch-scale one -- meaning a
properly-chosen fixed cooldown would be satisfied *within* epoch 1 even
on a huge dataset (a few hundred steps into a 100,000-step epoch),
resolving the large-dataset generalization concern more cleanly than
either of the two previous proposals (rescaling the constant, or
falling back to a separate v2/v3 mechanism for that regime).

### User's follow-up: "is there theory to compute the optimal lag?"

Two real anchors, both incomplete on their own:

1. **Adam's own momentum memory (exact, citable)**: the EMA decay
   constant `beta1` (0.9, PyTorch AdamW default) gives an effective
   memory window of `1/(1-beta1) = 10` steps -- a **hard lower bound**.
   Below this, the cooldown isn't isolating "has the model's fit to
   this row eroded," it's still partly measuring "has Adam's own
   internal momentum state forgotten this gradient" -- a different,
   purely mechanical effect that has nothing to do with memorization.
2. **NTK/loss-landscape-curvature reasoning (the deeper, less tractable
   story)**: the actual recency-bump decay is dominated by how much
   *other* rows' subsequent gradients interfere with vs. reinforce the
   fit to this row -- formally, a kernel-similarity effect between
   training examples -- which is not closed-form-computable for a real
   transformer (the relevant Hessian/kernel isn't tractable). Theory
   tells us the right variable to reason about (optimizer/curvature
   properties, not dataset size) but not an exact number.

Given neither anchor gives an exact number, the rigorous resolution --
consistent with how every other free parameter in this design was
calibrated (Monte Carlo for the CUSUM threshold, data-derived `p0`
rather than assumed) -- is to **measure the forgetting curve directly**
rather than guess candidate cooldown values one smoke test at a time:
train normally to a realistic mid-training regime, then train a batch
of probe rows exactly once, and track their own score at every
subsequent step (without retraining them again) to trace the actual
decay back toward baseline. Built `forgetting_curve_probe.py`
(5-epoch warmup to reach realistic dynamics, then a 150-step decay
trace on 128 probe rows) -- running now, in parallel with the
stop-at-alarm experiment above.

**First forgetting-curve probe (5-epoch warmup) result: `+0.0167`
immediate jump, then noise (`±0.02`) with no visible decay pattern
over 150 steps.** Too small to be informative on its own -- and a
mismatch with where the recency-bump bug actually appeared (step 20,
near initialization, not after 5 epochs of warmup). Reran with
`WARM_EPOCHS=0` (near-init) to match the regime that actually matters.

**Near-init probe (flawed design, caught before drawing conclusions
from it): raw probe-row score climbed continuously for the full 150
steps** (baseline -5.6310 -> immediately-after -4.6233 -> +150 steps
-1.2980, i.e. another +3.3 of "improvement" despite never retraining
these rows again). This is NOT a slowly-decaying recency bump -- it's
the general, extremely fast early-training improvement that benefits
*every* row (probe or not) near initialization, uncontrolled-for. The
probe measured the raw trajectory instead of isolating the row-specific
effect, exactly the confound v4's `mu0` subtraction exists to remove --
this diagnostic simply didn't include that control, so its raw numbers
overstate the effect enormously and shouldn't be used to set a cooldown
directly. Caught before acting on it.

**Fix: `forgetting_curve_probe_v2.py`** adds a same-sized CONTROL group
of rows never trained during the trace window, evaluated at the exact
same steps as the probe rows. `probe_score(t) - control_score(t)`
cancels the shared general-improvement trend, isolating the genuine
row-specific recency effect -- this is the actual forgetting curve.
Running now (near-init regime, matching where the original bug
appeared).

**Corrected probe result: isolated gap = 0.0067 immediately after
training, rising to a stable plateau of ~0.08-0.12 by step 10-20, and
staying roughly flat (not decaying toward 0) through step 150.** This
is genuinely informative, though not in the shape originally expected
(spike-then-decay). Two things follow:
1. The magnitude (~0.10 nats, gated statistic) is far smaller than the
   original v2 bug's raw-NLL gap (1.085 nats) -- direct evidence the
   gated/Winsorized statistic (v3) already suppresses most of what made
   that original bug so dramatic (the raw statistic's unbounded penalty
   for wrong tokens was the dominant driver, not a fundamental property
   of the recency effect itself).
2. Since it doesn't decay to zero, the cooldown's actual job is
   narrower than originally framed: it's not waiting out a large,
   decaying transient -- it's avoiding the first ~10-20 steps of extra
   volatility right at the moment of training. The steady-state
   "one-time-training benefit" that remains afterward is already
   absorbed into `mu0` (calibrated as the average paired-improvement
   during early checks, which includes this same effect for every row
   in the calibration sample, not treated as anomalous).

**Decision: `COOLDOWN_STEPS=40` (fixed, dataset-size-independent)**,
grounded by both the Adam-memory floor (10 steps, so 40 gives 4x
margin) and this direct measurement (isolated gap stabilizes by step
10-20, so 40 gives a further ~2x margin past stabilization). Already
validated clean in the 12-epoch smoke test. This is the constant going
into the real library implementation.

### Final design (v5, going into the library implementation)

Merging every validated piece from this thread:
- Per-row statistic: gated/Winsorized log-likelihood (v3) -- continuous
  and whole-row (no probe-column restriction), bounded contribution from
  incorrect tokens.
- Reference: within-row paired comparison against each row's own
  one-time first-exposure baseline (v4) -- survives fresh-population
  exhaustion in multi-epoch training, controls for row difficulty
  directly.
- Cooldown: fixed `COOLDOWN_STEPS=40` (this section), not epoch-scaled
  -- dataset-size-independent, satisfied within epoch 1 even on a huge
  dataset.
- Checking: continuous, every `check_every` steps (default 20) --
  seamless throughout training, no epoch-boundary gating (per the "why
  tie it to epochs" exchange).
- Calibration: `mu0`/`sigma0` from the first `WARMUP_CHECKS=10` valid
  post-cooldown checks; CUSUM threshold `h` via Monte Carlo simulation
  against the actual check horizon.
- Stopping rule: classical Gaussian mean-shift CUSUM (Page 1954),
  `delta=0.5`, alarm triggers `should_training_stop`.
- Known, stated limitation: needs a minimum of ~2 epochs of data before
  any decision is possible (inherent to any reactive method); cannot
  help decide whether a single, non-repeatable epoch already overfit.

Now implementing this as a real library feature (new module, Trainer
subclass override, callback, threaded through `.fit()` as an opt-in
`overfitting_detection_method="cusum"`), built in an isolated git
worktree with tests and a full test-suite run before considering it
done. Not committed to `main` or pushed without explicit user review.

### Decisive result: does stopping at the alarm actually help?

Built `cusum_stop_at_alarm.py` -- identical seed/config to the full
75-epoch v4 run, but breaks out of training the instant the CUSUM fires
and runs the DCR ground-truth comparison on that exact checkpoint,
instead of continuing to epoch 75. Deterministically retraced the same
trajectory (confirmed: alarm at step 2980, epoch 39.73, identical to
the monitoring-only run) at roughly half the compute cost.

**Result:**

| | Stop-at-alarm (epoch 39.7) | Full 75 epochs |
|---|---|---|
| `dcr_synth` mean | 1.1750 | 0.2314 |
| `frac_suspicious` | 0.4450 | 0.8900 |

Stopping at the alarm roughly halves the fraction of synthetic samples
landing suspiciously close to a real training row, and more than
quadruples the average distance-to-closest-record. This is the
decisive validation for the whole approach: not just that the detector
fires at a plausible-looking point relative to an independent reference
(the earlier concordance check against the D-series bootstrap-DCR
result), but that *acting* on it produces a measurably, substantially
safer model than training to a fixed, uninformed schedule -- using a
generation-free, training-time signal that cost a fraction of what the
bootstrap-DCR mechanism's periodic `.generate()` calls would have.

### Library implementation

Built in an isolated worktree (branch `feat/cusum-overfitting-detection`,
not merged/pushed):

- `src/realtabformer/rtf_cusum.py`: `compute_gated_row_scores` (the
  per-row statistic), `calibrate_gaussian_cusum_threshold` (Monte Carlo
  threshold calibration), `CUSUMOverfittingMonitor` (all detector
  state and logic), `CUSUMTrainer` (a `ResumableTrainer` subclass that
  feeds per-row scores to the monitor for free during `compute_loss`),
  `CUSUMEarlyStoppingCallback` (runs the periodic check via HF's
  standard `TrainerCallback`/`TrainerControl.should_training_stop`
  mechanism -- the same hook `EarlyStoppingCallback` uses).
- `realtabformer.py`: `_fit_tabular` gains an opt-in `add_row_idx`
  param (adds a stable per-row `idx` column, needed to track baselines
  across steps) and `trainer_cls` (lets the cusum path swap in
  `CUSUMTrainer`); `.fit()` gains `overfitting_detection_method`
  (`"sensitivity"` default -- byte-for-bit the existing behavior,
  `"cusum"` -- the new path, `"none"` -- equivalent to `n_critic=0`)
  and `cusum_*` tunables; new `_train_with_cusum` method wires it
  together and calls `trainer.train()` normally (no manual epoch-
  chunking loop needed, unlike `_train_with_sensitivity`).

**Two real bugs found and fixed while integration-testing against the
actual library architecture (not just the scratchpad prototype), both
via direct verification rather than assumption:**
1. HF's `Trainer` defaults to `remove_unused_columns=True`, which
   inspects the model's `forward()` signature and silently strips any
   dataset column not in it -- including the new `idx` column, before
   it ever reached `compute_loss`. Confirmed directly (`record_batch`
   was never being called at all); fixed by setting
   `remove_unused_columns=False`, scoped only to the cusum path.
2. The dataset's label column is named `label_ids` internally; HF's
   `default_data_collator` auto-renames it to `labels` *during
   collation*, but the detector's callback reads rows directly from the
   dataset, bypassing the collator -- so it saw the pre-rename name and
   raised `KeyError: 'labels'`. Fixed with an explicit rename on a
   separate dataset reference used only by the callback (the trainer's
   own copy is untouched, still works via the collator's existing
   auto-rename).

Both were caught by actually running the integrated code end-to-end
against the real `REaLTabFormer.fit()` API, not by reasoning about the
architecture in the abstract -- consistent with this whole session's
discipline of verifying rather than assuming.

**Testing**: 14 new tests in `tests/realtabformer/test_rtf_cusum.py`
(the gated-score function's edge cases -- perfect prediction, all-wrong,
padding-masking; threshold calibration determinism and horizon-scaling;
the monitor's cooldown filtering, one-time baseline recording,
calibration-then-accumulation, and firing behavior against a
deterministic mock; end-to-end integration through the real
`REaLTabFormer.fit()` API including sampling from an early-stopped
model; confirmation the default path is completely unaffected). All 14
pass. Full existing suite run for regressions: 2 pre-existing failures
confirmed present on the clean, unmodified base branch too (one is the
already-known `gen_kwargs=None` bug with a dedicated fix branch
elsewhere; verified via `git stash` that neither failure is caused by
this change) -- everything else passes, no regressions introduced.

End-to-end smoke tests against the real `.fit()` API (not just the
scratchpad prototype) confirmed: calibration completes with sensible
values, the CUSUM accumulates evidence and can fire, training actually
stops early when it does (confirmed on a toy dataset: alarm at step 10
of a possible ~120, i.e. ~2.6 of 30 allotted epochs), and sampling from
the early-stopped model works correctly.

Linting: `isort`/`black`/`flake8` all clean on the new
`rtf_cusum.py` (fully self-contained, safe to auto-format in full).
For the existing `realtabformer.py`, confirmed via `git stash` that
every flake8 warning (`C901` complexity on `_train_with_sensitivity`,
`E722` bare except) and every black reformatting suggestion already
exists on the clean, unmodified base branch (same warnings, just
shifted line numbers) -- not introduced by this change, and
deliberately not "fixed" as a drive-by, since that would touch large
amounts of unrelated pre-existing code. Manually verified the actual
diff (148 insertions, 2 deletions, purely additive) is well-formatted.

**Final large-scale validation launched**: the real Adult dataset
(same 2400-row/75-epoch setup used throughout this research) through
the actual public `REaLTabFormer.fit(..., overfitting_detection_method
="cusum")` API with every `cusum_*` parameter left at its library
default (not hand-tuned scratchpad overrides) -- the strongest
remaining confirmation that the shipped defaults work sensibly at
realistic scale through the real API, not just in the scratchpad
prototype or toy-scale smoke tests. Running in the background; result
to be appended here once complete.

### The realistic-scale library run surfaced one more real bug: cooldown vs. gradient accumulation

The first full-defaults library validation on Adult (75 epochs, real
`.fit(overfitting_detection_method="cusum")` API, no scratchpad
overrides) came back with `alarm_step=None, mu0=None` -- calibration
never even completed across the *entire* 75-epoch run, despite up to
71 possible checks. Traced directly (not guessed): HF's `TrainingArguments`
defaults to `gradient_accumulation_steps=4`, so one optimizer step
covers 4x the rows a naive `batch_size`-based estimate would suggest --
on this dataset, `steps_per_epoch` drops from the scratchpad's 75 (no
accumulation) to ~19. With `cooldown_steps=40` (over 2 epochs' worth of
*optimizer* steps), the training loop cycles back through the whole
dataset and re-touches every row before any row can ever satisfy the
cooldown -- the reference pool stays permanently empty. Not a
scratchpad-vs-library discrepancy in the statistic itself, a genuine
structural bug: **any** cooldown_steps >= steps_per_epoch breaks this
way, for any dataset/batch/accumulation combination.

**Fix**: `CUSUMOverfittingMonitor.adjust_cooldown_for_steps_per_epoch`,
called from `on_train_begin` once `steps_per_epoch` is computable from
the real `TrainingArguments` (`per_device_train_batch_size *
gradient_accumulation_steps`) and the actual dataset size -- caps
`cooldown_steps` at half an epoch's worth of steps, with a `UserWarning`
explaining why whenever it actually adjusts anything. No-op on any
dataset where the configured cooldown already fits (the common case).
Re-verified directly on the same Adult/library setup that exposed the
bug: `cooldown_steps` auto-capped 40 -> 9, cooled pool immediately
non-empty (1456+ rows from the first check), calibration completed
cleanly at step 220 (`mu0=0.718, sigma0=0.075`), no false alarm within
the 15-epoch test window (correct -- real onset isn't expected until
~epoch 40).

### Branch reconciliation

Separately, the user asked to verify all of this session's feature
branches were correctly merged into `feat/support-seed-input`. Checked
every branch's ancestry directly (`git merge-base --is-ancestor`, not
assumed): all of them were -- except `refactor/data-utils`, the branch
this CUSUM feature was originally built on top of, which turned out to
be an independently-redone, earlier/simpler version of the same
`data_utils.py` package-split refactor that `feat/support-seed-input`
had already extended further (confirmed via direct diff: every
overlapping piece of functionality, including two specific
improvements -- a `TypedDict` schema doc and a mask-hoisting perf
tweak -- already existed independently on both branches). Merged
`refactor/data-utils` into `feat/support-seed-input` (all conflicts
resolved in favor of the more advanced seed-input versions, verified
line-by-line that nothing unique was being discarded), then rebuilt
this entire CUSUM feature on top of the consolidated branch so it
coexists with quantile encoding, shared vocab, any-order training,
digit-entropy weighting, and everything else developed this session.

### Final status

Implemented, tested, and committed (not pushed) on
`feat/cusum-overfitting-detection`, based on `feat/support-seed-input`:

- `src/realtabformer/rtf_cusum.py`: the full detector (gated per-row
  statistic, Monte Carlo threshold calibration, the monitor's state
  machine, the `CUSUMTrainer` subclass, the `CUSUMEarlyStoppingCallback`).
- `realtabformer.py`: `.fit(overfitting_detection_method=...)` --
  `"sensitivity"` (default, byte-for-byte unchanged existing behavior),
  `"cusum"` (this feature), `"none"` (equivalent to `n_critic=0`).
- 16 unit + integration tests, all passing; full existing suite run
  four times across this branch and clean `feat/support-seed-input`
  during verification, consistently reproducing only two known
  pre-existing failures (confirmed via direct comparison against the
  clean branch, not assumed) plus one intermittent, unrelated flake in
  `test_sensitivity_training_does_not_crash_with_default_gen_kwargs`
  (passed in isolation every time it was tried, and passed on a clean
  full-suite rerun -- doesn't reproduce reliably, and this feature's
  code doesn't touch that code path at all).
- Linting (`isort`/`black`/`flake8`) clean on the new module; the
  modified `realtabformer.py` shows zero new warnings versus the clean
  baseline (directly diffed, not assumed).
- Real, substantial validation: stopping training at the detector's
  alarm point produced a measurably safer model than training to a
  fixed schedule (frac_suspicious 0.445 vs 0.890; see above).

Not pushed, per standing session policy -- ready for review.
