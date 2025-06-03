import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import json
import time
import logging

# Ensure the dataPairs directory exists
os.makedirs('./dataPairs', exist_ok=True)

# Read data from data.json
with open('data.json', 'r') as file:
    data = json.load(file)

# Define GPU device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load all data
fineGrids_all = data['x']
coarseGrids_all = data['XCG']
N_all = len(fineGrids_all)

# Convert all to tensors
fineGrids_all = torch.tensor(fineGrids_all, dtype=torch.float32, device=device)
coarseGrids_all = torch.tensor(coarseGrids_all, dtype=torch.float32, device=device)

# Convert labels from {0.1, ...} to {-1, 1}
x_all = torch.where(fineGrids_all == 0.1, -1, 1)
X_all = coarseGrids_all.view(N_all, -1)
x_all = x_all.view(N_all, -1)

# Fixed train/test split
split_index = int(0.9 * N_all)
X_train_full, X_test = X_all[:split_index], X_all[split_index:]
x_train_full, x_test = x_all[:split_index], x_all[split_index:]

# Subset size for training
selectionIndex = 5000
X_train = X_train_full[:selectionIndex]
x_train = x_train_full[:selectionIndex]

# Define dimensions
m = X_train.shape[1]
print(m)
d = x_train.shape[1]
print(d)

# Initialize weights and bias
W = torch.randn((d, m), device=device, requires_grad=True)
b = torch.randn((d,), device=device, requires_grad=True)

# Negative log-likelihood function
def neg_log_likelihood(W, b, X, x):
    f = X @ W.T + b  # Compute f(X)
    nll = torch.logaddexp(torch.tensor(0.0, device=device), -x * f)  # Log(1 + exp(-x * f))
    return torch.sum(nll)

# Optimizer setup
optimizer = optim.LBFGS([W, b], lr=0.1, max_iter=10e4)

# Training loop
start_time = time.time()

def closure():
    optimizer.zero_grad()
    nll = neg_log_likelihood(W, b, X_train, x_train)
    nll.backward()
    print(f'Current NLL: {nll.item()}')
    return nll

print("Starting optimization...")
optimizer.step(closure)

end_time = time.time()
print("Optimization finished in", end_time - start_time, "seconds")

# Evaluation on test data
with torch.no_grad():
    f_test = X_test @ W.T + b
    preds = torch.sign(f_test)
    accuracy = torch.mean((preds == x_test).float())
    print("Testing Accuracy:", accuracy.item())

# Visualize predictions for 10 test samples
H, W_grid = 65, 65  # Adjust based on your data

def unpack_params(params):
    W = params[:d*m].view(d, m)
    b = params[d*m:]
    return W, b

def sum_of_squares_error(params, X, x):
    W, b = unpack_params(params)
    f = X @ W.T + b
    x_pred = torch.sign(f)
    error = torch.sum((x - x_pred) ** 2) / len(x)
    return error

params_opt = torch.cat([W.view(-1), b])

# Compute sum of squares error for the testing data
sse = sum_of_squares_error(params_opt, X_test, x_test)
print("Sum of Squares Error on testing data:", sse)
plot_dir = f'logReg/{selectionIndex}'
os.makedirs(plot_dir, exist_ok=True)

for n in range(100):
    testCoarseGrid = X_test[n].view(1, -1)
    f_pred = testCoarseGrid @ W.T + b
    x_pred = torch.sign(f_pred).view(H, W_grid)
    x_actual = x_test[n].view(H, W_grid)

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(x_actual.detach().cpu().numpy(), cmap='bwr', vmin=-1, vmax=1)
    axes[0].set_title('Actual')
    axes[1].imshow(x_pred.detach().cpu().numpy(), cmap='bwr', vmin=-1, vmax=1)
    axes[1].set_title('Predicted')
    plt.savefig(f'{plot_dir}/actual_vs_predicted_sample_{n}.png', dpi=300)
    plt.close(fig)
