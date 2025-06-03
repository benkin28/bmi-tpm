#!/usr/bin/env python3
"""
merge_and_summarise_metrics.py

This script reads all experiment metrics from the 'experiments/' folder:
- metrics_loop3.json
- entropy_history.json
- accuracy_vs_noise_data.json
- accuracy_vs_partial_data.json

It computes aggregated statistics (mean, std, etc.) and prints a summary.
"""

import json
import numpy as np
import pandas as pd
import statistics
from pathlib import Path

# Paths to metrics files
metrics_files = dict(
    mse_relL2_time='experiments/metrics_loop3.json',
    entropy='experiments/entropy_history.json',
    accuracy_noise='experiments/accuracy_vs_noise_data.json',
    accuracy_partial='experiments/accuracy_vs_partial_data.json'
)

summary = {}

# 1. MSE, Rel_L2, Time
with open(metrics_files['mse_relL2_time'], 'r') as f:
    data = json.load(f)

mse_values = [entry["mse"] for entry in data if "mse" in entry]
rel_l2_values = [entry["rel_L2"] for entry in data if "rel_L2" in entry]
time_values = [entry["time"] for entry in data if "time" in entry]

summary['MSE_mean'] = np.mean(mse_values)
summary['MSE_std'] = np.std(mse_values)
summary['rel_L2_mean'] = np.mean(rel_l2_values)
summary['rel_L2_std'] = np.std(rel_l2_values)
if time_values:
    summary['time_mean'] = np.mean(time_values)
    summary['time_std'] = np.std(time_values)

# 2. Entropy
with open(metrics_files['entropy'], 'r') as f:
    entropy_data = json.load(f)

entropy_values = [entry["mean_entropy"] for entry in entropy_data]
summary['entropy_mean'] = np.mean(entropy_values)
summary['entropy_std'] = np.std(entropy_values)

# 3. Accuracy vs Noise
with open(metrics_files['accuracy_noise'], 'r') as f:
    noise_data = json.load(f)

thresholds = sorted({thr for run in noise_data for thr in run["accuracy"].keys()})
noise_rows = []
for run in noise_data:
    row = {thr: run["accuracy"][thr] for thr in thresholds}
    row["mean_acc"] = statistics.mean(row[thr] for thr in thresholds)
    noise_rows.append(row)
noise_df = pd.DataFrame(noise_rows)
summary['acc_noise_mean'] = {k: v for k, v in noise_df[thresholds].mean().to_dict().items()}
summary['acc_noise_std'] = {k: v for k, v in noise_df[thresholds].std().to_dict().items()}

# 4. Accuracy vs Partial Data
with open(metrics_files['accuracy_partial'], 'r') as f:
    partial_data = json.load(f)

fractions = sorted(
    {frac for run in partial_data for frac in run["accuracy"].keys()},
    key=lambda s: float(s.rstrip('%'))
)
partial_rows = []
for run in partial_data:
    row = {f: run["accuracy"][f] for f in fractions}
    row["mean_acc"] = statistics.mean(row[f] for f in fractions)
    partial_rows.append(row)
partial_df = pd.DataFrame(partial_rows)
summary['acc_partial_mean'] = {k: v for k, v in partial_df[fractions].mean().to_dict().items()}
summary['acc_partial_std'] = {k: v for k, v in partial_df[fractions].std().to_dict().items()}

# ========== Print Final Summary ==========
print("\n==================== Summary Metrics ====================")
for key, value in summary.items():
    if isinstance(value, dict):
        print(f"\n{key}:")
        for subkey, subvalue in value.items():
            print(f"  {subkey}: {subvalue:.4f}")
    else:
        print(f"{key}: {value:.4f}")
print("=========================================================\n")
