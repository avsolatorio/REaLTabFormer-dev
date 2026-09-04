"""
Runs the full numeric_quantile_encoding on/off x stopping-method
comparison grid, across multiple datasets, in one call -- meant for a GPU
server, the same way run_all_experiments.py is (see its own docstring).

The three "stopping methods" this compares are all reachable through
run_experiment.py already, just via different flag combinations, not
three different --mode values:
  - "sensitivity": --mode sensitivity (the existing bootstrap-DCR
    mechanism, the pre-CUSUM baseline).
  - "cusum": --mode cusum (no confirmation step).
  - "hybrid": --mode cusum --cusum-confirm-with-sensitivity (a fired
    CUSUM alarm gets one sensitivity-style confirmation check before
    actually stopping).

Crossed with numeric_quantile_encoding off/on (run_experiment.py's own
--numeric-quantile-encoding flag), that's up to 6 runs per dataset. Each
runs as its own subprocess (a fresh Python process per run -- no state,
random seeds, or monkeypatches leaking across runs), and a failure in one
run is logged and skipped rather than aborting the whole grid, so a crash
partway through doesn't cost you the results already collected.

Every run's summary.json records its own numeric_quantile_encoding and
(for cusum runs) cusum_confirm_with_sensitivity value directly -- see
run_experiment.py's run_cusum/run_full/run_sensitivity -- so the
run_id naming below (which also encodes both) is for human
readability/log-file naming only; analyze_quantile_comparison.py reads
the authoritative fields from each summary.json itself, not the filename.

Usage:
    python run_quantile_comparison.py --epochs 300 --device cuda
    python run_quantile_comparison.py --datasets diabetes abalone \
        --methods cusum hybrid --epochs 300 --device cuda
    python run_quantile_comparison.py --epochs 300 --device cuda --paper-metrics

After the grid finishes (or partway through, if you want a first look),
run analyze_quantile_comparison.py to aggregate every summary.json under
--output-dir into a single comparison table.
"""

import argparse
import itertools
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RUN_EXPERIMENT = SCRIPT_DIR / "run_experiment.py"

# Smallest first -- fast feedback before committing to the longer runs.
DEFAULT_DATASETS = ["diabetes", "insurance", "abalone", "wilt", "churn2", "adult"]
DEFAULT_METHODS = ["sensitivity", "cusum", "hybrid"]

# method -> the run_experiment.py --mode/--cusum-confirm-with-sensitivity
# combination that reaches it. "hybrid" is --mode cusum plus the confirm
# flag, not a separate --mode value -- see module docstring.
_METHOD_MODE_FLAGS = {
    "sensitivity": (["--mode", "sensitivity"], False),
    "cusum": (["--mode", "cusum"], False),
    "hybrid": (["--mode", "cusum"], True),
}


def run_one(dataset, method, quantile_encoding, args, log_dir: Path) -> dict:
    qtag = "qenc" if quantile_encoding else "base"
    run_id = f"{dataset}_{method}_{qtag}_ep{args.epochs}_{int(time.time())}"
    log_path = log_dir / f"{run_id}.log"

    mode_flags, confirm = _METHOD_MODE_FLAGS[method]
    cmd = [
        sys.executable,
        str(RUN_EXPERIMENT),
        "--dataset",
        dataset,
        *mode_flags,
        "--epochs",
        str(args.epochs),
        "--device",
        args.device,
        "--batch-size",
        str(args.batch_size),
        "--output-dir",
        args.output_dir,
        "--run-id",
        run_id,
    ]
    if confirm:
        cmd += ["--cusum-confirm-with-sensitivity"]
        if args.cusum_confirm_num_bootstrap is not None:
            cmd += [
                "--cusum-confirm-num-bootstrap",
                str(args.cusum_confirm_num_bootstrap),
            ]
    if quantile_encoding:
        cmd += ["--numeric-quantile-encoding"]
    if args.gen_batch:
        cmd += ["--gen-batch", str(args.gen_batch)]
    if args.paper_metrics:
        cmd += ["--paper-metrics"]
        cmd += ["--eval-n-jobs", str(args.eval_n_jobs)]
    cmd += ["--sensitivity-bootstrap-n-jobs", str(args.sensitivity_bootstrap_n_jobs)]
    if args.sensitivity_cache_dir:
        cmd += ["--sensitivity-cache-dir", args.sensitivity_cache_dir]
    if args.no_sensitivity_cache:
        cmd += ["--no-sensitivity-cache"]

    print(
        f"\n{'=' * 70}\nSTARTING {dataset} / {method} / {qtag} (run_id={run_id})\n"
        f"Log: {log_path}\n{'=' * 70}",
        flush=True,
    )

    t0 = time.time()
    with open(log_path, "w") as logf:
        # Stream live to the console AND save to the log file -- same
        # reasoning as run_all_experiments.py's own run_one.
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        )
        for line in proc.stdout:
            print(line, end="", flush=True)
            logf.write(line)
        proc.wait()
    elapsed = time.time() - t0

    ok = proc.returncode == 0
    status = "OK" if ok else f"FAILED (exit {proc.returncode})"
    print(f"\n{dataset} / {method} / {qtag}: {status} in {elapsed:.1f}s\n", flush=True)
    return dict(
        dataset=dataset,
        method=method,
        quantile_encoding=quantile_encoding,
        run_id=run_id,
        ok=ok,
        status=status,
        elapsed_s=elapsed,
        log=str(log_path),
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help=f"Datasets to run, in order. Default: {DEFAULT_DATASETS}.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=list(DEFAULT_METHODS),
        default=DEFAULT_METHODS,
        help=f"Stopping methods to compare. Default: {DEFAULT_METHODS} (all "
        "three). See module docstring for how each maps to run_experiment.py "
        "flags.",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        default=False,
        help="Only run the numeric_quantile_encoding=True side of the grid "
        "-- e.g. you already have the baseline (off) runs from a previous "
        "invocation and just want to add the quantile-encoding comparison.",
    )
    parser.add_argument(
        "--skip-quantile",
        action="store_true",
        default=False,
        help="Only run the numeric_quantile_encoding=False (baseline) side "
        "of the grid.",
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--gen-batch",
        type=int,
        default=None,
        help="Passed through to run_experiment.py -- see its own --help.",
    )
    parser.add_argument(
        "--paper-metrics",
        action="store_true",
        default=False,
        help="Passed through to run_experiment.py -- adds the paper-matching "
        "MLE (CatBoost)/DM (RandomForest discriminator) metrics to every run "
        "in the grid. Requires `pip install catboost`; meaningfully slower "
        "per run, and this grid already runs up to 6 configurations per "
        "dataset -- consider leaving this off for a first pass and only "
        "re-running the configurations you actually want paper metrics for.",
    )
    parser.add_argument("--eval-n-jobs", type=int, default=-1)
    parser.add_argument("--sensitivity-bootstrap-n-jobs", type=int, default=-1)
    parser.add_argument("--sensitivity-cache-dir", default=None)
    parser.add_argument("--no-sensitivity-cache", action="store_true", default=False)
    parser.add_argument(
        "--cusum-confirm-num-bootstrap",
        type=int,
        default=None,
        help="Passed through to run_experiment.py's own "
        "--cusum-confirm-num-bootstrap, only used for 'hybrid' runs. Leave "
        "unset to reuse each dataset's own --sensitivity-num-bootstrap "
        "value.",
    )
    parser.add_argument("--output-dir", default=str(SCRIPT_DIR / "results"))
    parser.add_argument(
        "--log-dir",
        default=str(SCRIPT_DIR / "results" / "logs"),
        help="Where each run's full stdout/stderr is saved.",
    )
    args = parser.parse_args()

    if args.skip_baseline and args.skip_quantile:
        parser.error("--skip-baseline and --skip-quantile together skip everything")

    quantile_variants = []
    if not args.skip_baseline:
        quantile_variants.append(False)
    if not args.skip_quantile:
        quantile_variants.append(True)

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    grid = list(itertools.product(args.datasets, args.methods, quantile_variants))
    print(
        f"Running {len(grid)} configurations: {len(args.datasets)} dataset(s) x "
        f"{len(args.methods)} method(s) x {len(quantile_variants)} "
        "quantile_encoding variant(s).",
        flush=True,
    )

    results = []
    batch_t0 = time.time()
    for dataset, method, quantile_encoding in grid:
        results.append(run_one(dataset, method, quantile_encoding, args, log_dir))
    batch_elapsed = time.time() - batch_t0

    print(f"\n\n{'=' * 70}\nGRID SUMMARY ({batch_elapsed:.1f}s total)\n{'=' * 70}")
    for r in results:
        qtag = "qenc" if r["quantile_encoding"] else "base"
        print(
            f"  {r['dataset']:12s} {r['method']:12s} {qtag:5s} "
            f"{r['status']:20s} {r['elapsed_s']:8.1f}s  {r['log']}"
        )

    failed = [r for r in results if not r["ok"]]
    if failed:
        print(
            f"\n{len(failed)} of {len(results)} run(s) failed. Check their logs "
            "above for the actual error.",
            flush=True,
        )
        sys.exit(1)

    print(
        f"\nAll {len(results)} runs done. Commit results/ (the new "
        "summary.json/trajectory.jsonl files, not the gitignored checkpoint "
        "dirs or --log-dir) to share them, then run "
        "analyze_quantile_comparison.py to aggregate them into a comparison "
        "table.",
        flush=True,
    )


if __name__ == "__main__":
    sys.exit(main())
