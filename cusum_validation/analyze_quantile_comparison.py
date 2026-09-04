"""
Aggregates the summary.json files produced by run_quantile_comparison.py
(or by hand-run run_experiment.py invocations using its
--numeric-quantile-encoding flag) into one comparison table: dataset x
stopping method x numeric_quantile_encoding on/off.

Deliberately does NOT parse dataset/method/quantile_encoding from run_id
filenames as its primary source of truth -- run_id is a human-readable
label, not a guaranteed-parseable format (a hand-run --run-id could be
anything). Instead reads the authoritative fields each run_* function in
run_experiment.py already writes into its own summary.json:
  - `numeric_quantile_encoding`: recorded directly in whichever of
    cusum_training/full_training/sensitivity_training is present.
  - method: sensitivity_training present -> "sensitivity"; cusum_training
    present -> "hybrid" if its own cusum_confirm_with_sensitivity field is
    true, else "cusum"; full_training present -> "full" (not one of the
    three methods run_quantile_comparison.py compares, but included here
    too in case a --mode all/full run is mixed into the same results
    directory).
`dataset` genuinely isn't recorded in summary.json anywhere (no run_*
function stores it), so that one field IS taken from the run_id/filename
prefix, matched against the known dataset list -- the one piece this
script can't get more robustly than that.

Usage:
    python analyze_quantile_comparison.py
    python analyze_quantile_comparison.py --results-dir /path/to/results --csv-out comparison.csv
"""

import argparse
import json
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent

# Longest-name-first so e.g. "churn2" doesn't get shadowed by a shorter
# false-prefix match against some other dataset name.
KNOWN_DATASETS = sorted(
    ["diabetes", "insurance", "abalone", "wilt", "churn2", "adult"],
    key=len,
    reverse=True,
)

METHOD_LABEL = {
    "sensitivity": "sensitivity_stop",
    "cusum": "cusum_stop_at_alarm",
    "hybrid": "cusum_stop_at_alarm",
    "full": "full_schedule",
}


def _infer_dataset(run_id: str) -> str:
    # Matched against underscore-delimited tokens (not a raw substring
    # search) so e.g. a "diabetes" dataset name can't accidentally match
    # inside some unrelated word, and matched anywhere in run_id (not
    # just as a strict prefix) so a custom --run-id prefix (e.g. a
    # smoke-test label) doesn't defeat the match as long as the dataset
    # name appears somewhere in it -- run_quantile_comparison.py's own
    # run_ids always start with it, but a hand-run run_experiment.py
    # invocation's --run-id might not.
    tokens = run_id.split("_")
    for name in KNOWN_DATASETS:
        if name in tokens:
            return name
    return "unknown"


def _infer_method_and_quantile(data: dict):
    if "sensitivity_training" in data:
        t = data["sensitivity_training"]
        return "sensitivity", t.get("numeric_quantile_encoding", False), t
    if "cusum_training" in data:
        t = data["cusum_training"]
        method = "hybrid" if t.get("cusum_confirm_with_sensitivity") else "cusum"
        return method, t.get("numeric_quantile_encoding", False), t
    if "full_training" in data:
        t = data["full_training"]
        return "full", t.get("numeric_quantile_encoding", False), t
    return None, None, None


def _row_from_summary(path: Path) -> dict:
    data = json.loads(path.read_text())
    run_id = path.name[: -len("_summary.json")]
    method, quantile_encoding, training = _infer_method_and_quantile(data)
    if method is None:
        return None

    label = METHOD_LABEL[method]
    dcr = data.get(label, {})
    utility = data.get(f"{label}_utility", {})
    mle = data.get(f"{label}_mle", {})
    dm = data.get(f"{label}_dm", {})

    row = dict(
        run_id=run_id,
        dataset=_infer_dataset(run_id),
        method=method,
        quantile_encoding=quantile_encoding,
        elapsed_s=training.get("elapsed_s"),
        effective_epochs=training.get(
            "effective_epochs", training.get("stopped_epoch")
        ),
        epochs_ceiling=training.get("epochs_ceiling"),
        frac_suspicious=dcr.get("frac_suspicious"),
        dcr_synth_mean=dcr.get("dcr_synth_mean"),
        dcr_test_mean=dcr.get("dcr_test_mean"),
        utility_metric=utility.get("metric"),
        utility_trtr=utility.get("trtr"),
        utility_tstr=utility.get("tstr"),
        utility_gap=utility.get("gap"),
    )
    if mle:
        row.update(
            mle_metric=mle.get("metric"),
            mle_synthetic_mean=mle.get("synthetic_mean"),
            mle_real_mean=mle.get("real_mean"),
        )
    if dm:
        row.update(dm_mean=dm.get("dm_mean"))
    return row


def build_table(results_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(results_dir.glob("*_summary.json")):
        row = _row_from_summary(path)
        if row is not None:
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df.sort_values(["dataset", "method", "quantile_encoding"]).reset_index(
        drop=True
    )
    return df


def print_comparison(df: pd.DataFrame):
    """For each dataset x method pair with BOTH a baseline and a
    quantile-encoding run present, print them side by side plus the
    deltas that actually answer "did quantile encoding help": utility_gap
    (closer to 0 is better -- synthetic data as useful as real) and
    frac_suspicious (lower is better -- less memorization/privacy risk).
    Datasets/methods with only one side present are listed separately so
    a partially-finished grid is still readable, not silently dropped.
    """
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)

    paired_rows = []
    unpaired = []
    for (dataset, method), g in df.groupby(["dataset", "method"]):
        base = g[~g["quantile_encoding"]]
        qenc = g[g["quantile_encoding"]]
        if len(base) >= 1 and len(qenc) >= 1:
            # If a config was re-run more than once, take the most
            # recently-written summary (sorted by run_id above, which
            # embeds a timestamp) rather than silently averaging/dropping.
            b, q = base.iloc[-1], qenc.iloc[-1]
            paired_rows.append(
                dict(
                    dataset=dataset,
                    method=method,
                    base_frac_suspicious=b["frac_suspicious"],
                    qenc_frac_suspicious=q["frac_suspicious"],
                    base_utility_gap=b["utility_gap"],
                    qenc_utility_gap=q["utility_gap"],
                    delta_frac_suspicious=(
                        q["frac_suspicious"] - b["frac_suspicious"]
                        if pd.notna(q["frac_suspicious"])
                        and pd.notna(b["frac_suspicious"])
                        else None
                    ),
                    delta_abs_utility_gap=(
                        abs(q["utility_gap"]) - abs(b["utility_gap"])
                        if pd.notna(q["utility_gap"]) and pd.notna(b["utility_gap"])
                        else None
                    ),
                )
            )
        else:
            unpaired.extend(g["run_id"].tolist())

    if paired_rows:
        paired_df = pd.DataFrame(paired_rows)
        print("\n=== Paired comparison (quantile encoding vs baseline) ===")
        print(
            "Negative delta_frac_suspicious / delta_abs_utility_gap means "
            "quantile encoding IMPROVED that metric.\n"
        )
        print(paired_df.to_string(index=False))
    else:
        print(
            "\nNo dataset/method pair has both a baseline and a "
            "quantile-encoding run yet -- nothing to compare."
        )

    if unpaired:
        print(
            f"\n({len(unpaired)} run(s) still missing their other-variant "
            f"counterpart, not shown above: {unpaired})"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        default=str(SCRIPT_DIR / "results"),
        help="Directory to glob *_summary.json from.",
    )
    parser.add_argument(
        "--csv-out",
        default=None,
        help="If set, write the full (unpaired) table to this CSV path too.",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    df = build_table(results_dir)
    if df.empty:
        print(f"No summary.json files found under {results_dir}.")
        return

    print(f"Found {len(df)} run(s) under {results_dir}.\n")
    print("=== Full table ===")
    print(df.to_string(index=False))

    print_comparison(df)

    if args.csv_out:
        df.to_csv(args.csv_out, index=False)
        print(f"\nWrote full table to {args.csv_out}")


if __name__ == "__main__":
    main()
