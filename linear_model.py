import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import json
import time

# Create directory for output plots if it doesn't exist
os.makedirs('./logReg', exist_ok=True)

# Load dataset from a JSON file
with open('data.json', 'r') as file:
    data = json.load(file)

# Select computation device: GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load fine-grid and coarse-grid data as tensors on the selected device
fineGrids_all = torch.tensor(data['x'], dtype=torch.float32, device=device)
coarseGrids_all = torch.tensor(data['XCG'], dtype=torch.float32, device=device)
N_all = len(fineGrids_all)  # Number of data samples

# Preprocess fine-grid labels:
# Convert values from {0.1, 1.0} to binary {-1, +1} (0.1 -> -1, 1.0 -> +1)
x_all = torch.where(fineGrids_all == 0.1, -1, 1)
X_all = coarseGrids_all.view(N_all, -1)  # Flatten coarse grid samples
x_all = x_all.view(N_all, -1)            # Flatten fine grid samples

# Split data into training (90%) and test (10%) sets
split_index = int(0.9 * N_all)
X_train_full, X_test = X_all[:split_index], X_all[split_index:]
x_train_full, x_test = x_all[:split_index], x_all[split_index:]

# Select a validation sample (the last test sample)
validation_idx = -1
X_val_sample = X_test[validation_idx].view(1, -1)  # Shape (1, m)
x_val_sample = x_test[validation_idx].view(65, 65)  # Shape (65, 65) for visualization

# Define different training set sizes to compare performance
training_sizes = [1000, 5000, 10000]

# Dictionary to store predictions for each training size
predictions = {}

# Loop over each training size and train the logistic regression model
for selectionIndex in training_sizes:
    print(f"\nTraining with {selectionIndex} samples...")

    # Select a subset of training data
    X_train = X_train_full[:selectionIndex]
    x_train = x_train_full[:selectionIndex]

    m = X_train.shape[1]  # Input dimension (flattened coarse grid size)
    d = x_train.shape[1]  # Output dimension (flattened fine grid size)

    # Initialize logistic regression weights and biases
    W = torch.randn((d, m), device=device, requires_grad=True)
    b = torch.randn((d,), device=device, requires_grad=True)

    def neg_log_likelihood(W, b, X, x):
        """
        Compute the negative log-likelihood loss for logistic regression.
        - W: weight matrix of shape (d, m)
        - b: bias vector of shape (d,)
        - X: input features of shape (N, m)
        - x: binary labels of shape (N, d)
        """
        f = X @ W.T + b  # Linear model output
        nll = torch.logaddexp(torch.tensor(0.0, device=device), -x * f)  # log(1 + exp(-x*f))
        return torch.sum(nll)

    # Use LBFGS optimizer for training (second-order optimizer)
    optimizer = optim.LBFGS([W, b], lr=0.1, max_iter=1000)

    def closure():
        """Closure function required by LBFGS optimizer to recompute gradients."""
        optimizer.zero_grad()
        loss = neg_log_likelihood(W, b, X_train, x_train)
        loss.backward()
        return loss

    # Train the model by optimizing W and b
    optimizer.step(closure)

    # Generate predictions for the validation sample (single test point)
    with torch.no_grad():
        f_val = X_val_sample @ W.T + b
        x_pred = torch.sign(f_val).view(65, 65)  # Convert logits to {-1, +1} predictions
        predictions[selectionIndex] = x_pred.detach().cpu()  # Store prediction

# Convert ground truth validation sample to CPU for plotting
x_val_sample_cpu = x_val_sample.detach().cpu()

# Plot and compare predictions against ground truth
fig, axes = plt.subplots(1, len(training_sizes)+1,
                         figsize=(4*(len(training_sizes)+1), 4))

# Plot ground truth
axes[0].imshow(torch.flip(x_val_sample_cpu, dims=[0]).numpy(),
               cmap='viridis', vmin=-1, vmax=1)
axes[0].set_title('Ground Truth (flipped)')

# Plot predictions for each training size
for i, selectionIndex in enumerate(training_sizes):
    axes[i+1].imshow(torch.flip(predictions[selectionIndex], dims=[0]).numpy(),
                     cmap='viridis', vmin=-1, vmax=1)
    axes[i+1].set_title(f'Predicted {selectionIndex} (flipped)')

# Remove axis ticks and labels
for ax in axes:
    ax.axis('off')

plt.suptitle("Ground Truth vs Predictions — flipped about x-axis", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig('logReg/comparison_prediction_grid.png', dpi=300)
plt.close()
print("Saved flipped figure to logReg/comparison_prediction_grid.png")
