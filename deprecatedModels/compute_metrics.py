import json
import numpy as np

# Load JSON data
with open('experiments/metrics_loop3.json', 'r') as f:
    data = json.load(f)

# Extract metrics (some entries may not have "time")
mse_values = [entry["mse"] for entry in data if "mse" in entry]
rel_l2_values = [entry["rel_L2"] for entry in data if "rel_L2" in entry]
time_values = [entry["time"] for entry in data if "time" in entry]

# Compute means and standard deviations
mean_mse = np.mean(mse_values)
std_mse = np.std(mse_values)

mean_rel_l2 = np.mean(rel_l2_values)
std_rel_l2 = np.std(rel_l2_values)

if time_values:
    mean_time = np.mean(time_values)
    std_time = np.std(time_values)
else:
    mean_time = std_time = None

# Print results
print(f"Average MSE: {mean_mse:.3f} ± {std_mse:.3f}")
print(f"Average Relative L2: {mean_rel_l2:.3f} ± {std_rel_l2:.3f}")
if mean_time is not None:
    print(f"Average Runtime (s): {mean_time:.2f} ± {std_time:.2f}")
else:
    print("No runtime data available.")
