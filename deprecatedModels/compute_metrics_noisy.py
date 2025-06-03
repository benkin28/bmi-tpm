#!/usr/bin/env python3
"""
summarise_accuracy.py

Usage:
    python summarise_accuracy.py results.json
"""
from pathlib import Path
import json
import statistics
import pandas as pd


def summarise(json_path: str | Path) -> None:
    # ---------- load ---------------------------------------------------------
    data = json.loads(json_path.read_text()) if isinstance(json_path, Path) \
           else json.loads(Path(json_path).read_text()) \
           if Path(json_path).exists() else json.loads(json_path)

    # collect all threshold keys (as strings)
    thresholds = sorted({thr for run in data for thr in run["accuracy"].keys()})

    # ---------- build per-run table -----------------------------------------
    rows = []
    for idx, run in enumerate(data):
        row = {
            "run": idx,
            "timestamp": run["timestamp"],
            **{thr: run["accuracy"][thr] for thr in thresholds},
        }
        row["mean_acc"] = statistics.mean(row[thr] for thr in thresholds)
        rows.append(row)

    df = pd.DataFrame(rows)

    # ---------- aggregates ---------------------------------------------------
    # 1) per-run summary
    print("\nPer-run mean accuracy")
    print(df[["run", "timestamp", "mean_acc"]])

    # 2) average accuracy per threshold over all runs
    print("\nAverage accuracy at each threshold")
    print(df[thresholds].mean().to_frame("avg_accuracy"))

    # 3) best run overall
    best = df.loc[df["mean_acc"].idxmax()]
    print("\nBest run:")
    print(best[["run", "timestamp", "mean_acc"]])

    # optional: save CSV
    csv_path = Path("accuracy_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nDetailed table written to {csv_path.resolve()}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "json_source",
        help="Path to the JSON file (or paste the array directly as a string).",
    )
    args = parser.parse_args()
    summarise(Path(args.json_source))
