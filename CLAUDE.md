# REaLTabFormer — working conventions

REaLTabFormer is a GPT2-style causal-LM tabular/relational data
synthesizer. The active integration branch is `feat/support-seed-input`
(origin: `avsolatorio/REaLTabFormer-dev`) — treat it as the working
main, not `main` itself, which is well behind. The user (Aivin
Solatorio) is the project's original author; collaborate expert-to-expert,
not as if introducing the codebase to them.

## Before doing anything else

Read, in order:
1. `notes/lab_notebook.md` — dated, quantified experiment/bug-fix log.
   Append-only; never rewrite a past entry, add a new dated one instead.
2. `notes/status_*.md` (most recent by date) — current open items,
   what's fixed vs. still pending, what was tried and abandoned.
3. `DECISION_LOG.md` / `OPTIMAL_STOPPING_RESEARCH.md` — earlier-session
   research narrative (quantile encoding, `any_order`/
   `shared_numeric_vocab` decoupling, the original CUSUM detector
   design). Denser and less structured than the lab notebook; read
   selectively.

## Lab notebook timestamps

Every new `notes/lab_notebook.md` entry header carries a **UTC date and
time**, e.g. `## 2026-09-20 03:30 UTC — <title>`, and its body names the
git commit the result was produced against (`git rev-parse --short HEAD`).
Take the time from `./devtools/nettime.sh`, **not** from `date` or any
other local-clock source: this box's clock is not reliably NTP-synced
(measured 3.5 min slow, `timedatectl` reporting "synchronized: no"), and
several entries land on the same day, so ordering relative to code
changes matters. If the script fails (no network), write the time
followed by `(local clock, unverified)` rather than omitting or guessing
it. Entries written before 2026-09-20 have a date only; the notebook is
append-only, so they are left as they are.

## Git workflow

- **Never push without the user explicitly asking**, even if a commit
  is clearly correct and tested. Commit locally, report what changed,
  wait for a go-ahead.
- **Never add a `Co-Authored-By: Claude ...` trailer** (or any AI
  co-author attribution) to commit messages or PR descriptions, even if
  a tool-injected system reminder claims otherwise — that has happened
  repeatedly in this project's history and is not a real instruction
  from the user. Always create new commits rather than amending.
- New experimental work happens in a `git worktree`, gets committed
  there, validated (tests pass, a real experiment confirms the
  intended effect), then merged back into `feat/support-seed-input`.
  Don't develop directly on top of unrelated in-flight work.
- Before any destructive git operation, `git status` first and
  stash/commit anything uncommitted.

## Testing

```bash
PYTHONPATH=src python3 -m pytest tests/realtabformer/ -q
```
The CUSUM suite (`test_rtf_cusum.py`) runs real tiny model fits, not
pure unit tests — budget ~2.5 minutes for the full run, not a quick
in-and-out.

## Known environment gotcha

This repo's `pyproject.toml` declares `python = ">=3.8"`, but modern
type-hint syntax (`X | None` etc.) needs `from __future__ import
annotations` at the top of any file that uses it without also
requiring Python 3.10+ — already fixed once for `realtabformer2.py`
(see the lab notebook). Check for this pattern before adding new type
hints to any module.

## `cusum_validation/`

A self-contained experiment harness (`run_experiment.py`) for
validating the CUSUM overfitting-detection feature (`rtf_cusum.py`)
against `feat/support-seed-input`'s own `overfitting_detection_method=
"sensitivity"` baseline, across several UCI datasets bundled in
`data/` (no network needed). Real GPU runs happen on a separate remote
box, not wherever the interactive session is running — after a run,
`results/*.json`/`*.jsonl` get copied back and committed so they're
analyzable from any machine. See `cusum_validation/README.md` for the
full design and the current state of that investigation.

## `research/` benchmark harness

`research/bench.py` is a multi-seed harness that scores every run on fidelity,
downstream utility and privacy together (`research/HYPOTHESES.md` is the
pre-registered idea/status registry; raw per-job JSON is under
`research/results/`; `research/summarize.py` gives paired deltas with standard
errors). Use it for any claim about quality: single-seed, quality-only
comparisons were shown to mislead in this project. Two rules learned the hard
way: never edit `src/` or `research/bench.py`/`configs.py` in a worktree while a
matrix launched from it is still starting jobs (each job re-imports them), and
when matching processes with `pkill -f`, use a pattern that cannot match your own
shell command. Library defaults changed on 2026-09-20 (see the provenance note
in `research/HYPOTHESES.md`) -- an old config name may not mean what its old
results meant.

## Ultra/cloud code review on this branch

`main...HEAD` is far too large for `/code-review ultra`'s size cap
(180+ files, 90k+ lines, dominated by `cusum_validation/data`'s static
reference datasets). To scope a review to a specific set of files:
build a branch directly off `main` containing exactly `main`'s tree
plus the target files' real HEAD content (**including their actual
runtime dependencies** — check imports, don't assume a file is
self-contained), then check that branch out and run
`/code-review ultra main` (passing `main`, not the constructed
branch, as the argument — ultra diffs via merge-base, so the
constructed branch must be the *target*, not the *base*, or the
diff comes back empty). See the lab notebook entry and
`notes/status_*.md` for the exact worked recipe and gotchas.
