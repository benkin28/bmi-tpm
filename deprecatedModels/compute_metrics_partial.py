#!/usr/bin/env python3
"""
summarise_partial_data.py

Usage:
    python summarise_partial_data.py accuracy_vs_partial_data.json
"""
from pathlib import Path
import json
import statistics
import pandas as pd


def summarise(json_source: str | Path) -> None:
    # --------- load JSON (either filepath or raw JSON string) ---------------
    if isinstance(json_source, Path) or Path(json_source).exists():
        data = json.loads(Path(json_source).read_text())
    else:
        data = json.loads(json_source)

    # collect all fraction keys (e.g. "1%", "5%", ...) and sort numerically
    fractions = sorted(
        {frac for run in data for frac in run["accuracy"].keys()},
        key=lambda s: float(s.rstrip('%'))
    )

    # --------- build per-run table ------------------------------------------
    rows = []
    for idx, run in enumerate(data):
        row = {
            "run": idx,
            "timestamp": run["timestamp"],
            **{f: run["accuracy"][f] for f in fractions},
        }
        # mean accuracy across all fractions for this run
        row["mean_acc"] = statistics.mean(row[f] for f in fractions)
        rows.append(row)

    df = pd.DataFrame(rows)

    # --------- print summaries ----------------------------------------------
    # 1) per-run mean accuracy
    print("\nPer‐run mean accuracy:")
    print(df[["run", "timestamp", "mean_acc"]].to_string(index=False))

    
        # 2) average accuracy per fraction over all runs
    print("\nAverage accuracy at each fraction:")
    avg_frac = df[fractions].mean().to_frame("avg_accuracy")
    print(avg_frac)

    # 2b) standard deviation at 100%
    if "100%" in df.columns:
        std_100 = df["100%"].std()
        print(f"\nStandard deviation at 100%: {std_100:.4f}")


    # 3) best run overall
    best = df.loc[df["mean_acc"].idxmax()]
    print("\nBest run overall:")
    print(f"  run       : {best['run']}")
    print(f"  timestamp : {best['timestamp']}")
    print(f"  mean_acc  : {best['mean_acc']:.4f}")

    # --------- optional: save detailed table -------------------------------
    csv_path = Path("partial_data_accuracy_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nDetailed table written to {csv_path.resolve()}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Summarise accuracy_vs_partial_data.json"
    )
    parser.add_argument(
        "json_source",
        help="Path to the JSON file (or paste the JSON array directly)."
    )
    args = parser.parse_args()
    summarise(Path(args.json_source))
