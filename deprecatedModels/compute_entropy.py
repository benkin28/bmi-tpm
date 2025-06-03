#!/usr/bin/env python3
"""
summarise_entropy.py

Usage:
    python summarise_entropy.py entropy_data.json
"""

import json
import sys
from pathlib import Path
import statistics


def summarise_entropy(json_path: str | Path) -> None:
    path = Path(json_path)
    if not path.exists():
        print(f"Error: File '{json_path}' not found.")
        sys.exit(1)

    with open(path, "r") as f:
        data = json.load(f)

    entropies = [entry["mean_entropy"] for entry in data]

    avg = statistics.mean(entropies)
    std = statistics.stdev(entropies)

    print(f"Number of entries     : {len(entropies)}")
    print(f"Average entropy       : {avg:.6f}")
    print(f"Standard deviation    : {std:.6f}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python summarise_entropy.py entropy_data.json")
        sys.exit(1)

    summarise_entropy(sys.argv[1])
