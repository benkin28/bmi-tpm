import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import json
import time
from sklearn.metrics import confusion_matrix

os.makedirs('logReg', exist_ok=True)

# Load data
with open('data.json', 'r') as file:
    data = json.load(file)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Try with these dataset sizes
data_sizes = [1000, 5000, 10000]

results = {}

def compute_f1_scores(y_true, y_pred):
    """Compute combined F1 score as described: average F1 for both high and low permeability."""
    y_true = y_true.view(-1).cpu()
    y_pred = y_pred.view(-1).cpu()

    def f1(positive_val):
        TP = ((y_true == positive_val) & (y_pred == positive_val)).sum().item()
        FP = ((y_true != positive_val) & (y_pred == positive_val)).sum().item()
        FN = ((y_true == positive_val) & (y_pred != positive_val)).sum().item()

        if TP + FP == 0 or TP + FN == 0:
            return 0.0  # Avoid divide-by-zero

        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        return 2 * precision * recall / (precision + recall)

    f1_high = f1(1)
    f1_low = f1(-1)
    return 0.5 * (f1_high + f1_low)

for selectionIndex in data_sizes:
    fineGrids = data['x'][:selectionIndex]
    coarseGrids = data['XCG'][:selectionIndex]
    N = len(fineGrids)

    fineGrids = torch.tensor(fineGrids, dtype=torch.float32, device=device)
    coarseGrids = torch.tensor(coarseGrids, dtype=torch.float32, device=device)

    x_data = torch.where(fineGrids == 0.1, -1, 1)
    X_data = coarseGrids.view(N, -1)
    x_data = x_data.view(N, -1)

    split_index = int(0.9 * N)
    X_train, X_test = X_data[:split_index], X_data[split_index:]
    x_train, x_test = x_data[:split_index], x_data[split_index:]

    m = X_train.shape[1]
    d = x_train.shape[1]

    W = torch.randn((d, m), device=device, requires_grad=True)
    b = torch.randn((d,), device=device, requires_grad=True)

    def neg_log_likelihood(W, b, X, x):
        f = X @ W.T + b
        nll = torch.logaddexp(torch.tensor(0.0, device=device), -x * f)
        return torch.sum(nll)

    optimizer = optim.LBFGS([W, b], lr=1.0, max_iter=1000, tolerance_grad=1e-7, tolerance_change=1e-10)

    losses = []
    accs = []
    max_track_iters = 1000

    def closure():
        optimizer.zero_grad()
        nll = neg_log_likelihood(W, b, X_train, x_train)
        nll.backward()

        if closure.iterations < max_track_iters:
            losses.append(nll.item())
            with torch.no_grad():
                preds = torch.sign(X_test @ W.T + b)
                acc = torch.mean((preds == x_test).float()).item()
                accs.append(acc)
                
        closure.iterations += 1
        return nll

    closure.iterations = 0

    print(f"\nTraining on {selectionIndex} samples...")
    start_time = time.time()
    optimizer.step(closure)
    end_time = time.time()
    runtime = end_time - start_time

    with torch.no_grad():
        f_test = X_test @ W.T + b
        preds = torch.sign(f_test)
        accuracy = torch.mean((preds == x_test).float()).item()
        f1_score = compute_f1_scores(x_test, preds)

    # Compute confusion matrix
    true_flat = x_test.view(-1).cpu().numpy()
    pred_flat = preds.view(-1).cpu().numpy()
    confmat = confusion_matrix(true_flat, pred_flat, labels=[1, -1])
    confmat_list = confmat.tolist()  # convert to list for JSON serialization

    results[selectionIndex] = {
        "runtime_sec": runtime,
        "accuracy": accuracy,
        "f1_combined": f1_score,
        "confusion_matrix": {
            "labels": [1, -1],
            "matrix": confmat_list
        }
    }

    print(f"Done training on {selectionIndex} samples:")
    print(f"- Runtime: {runtime:.2f}s")
    print(f"- Accuracy: {accuracy:.4f}")
    print(f"- Combined F1 Score: {f1_score:.4f}")

    # Save plot for first 1000 iterations
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss (log scale)', color='tab:red')
    ax1.plot(losses, color='tab:red')
    ax1.set_yscale('log')
    ax1.tick_params(axis='y', labelcolor='tab:red')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy', color='tab:blue')
    epsilon = 1e-6
    safe_accs = [max(a, epsilon) for a in accs]
    ax2.plot(safe_accs, color='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    plt.title(f'Training Loss & Accuracy – {selectionIndex} Samples')
    fig.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space at the top for the title
    plt.savefig(f'logReg/loglog_plot_{selectionIndex}.png', dpi=300)
    plt.close()


# Save summary results to JSON
with open('logReg/summary_metrics.json', 'w') as f:
    json.dump(results, f, indent=2)
