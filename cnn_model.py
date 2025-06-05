#!/usr/bin/env python3
# train_cnn_predictor.py
# ---------------------------------------------------------------
# End-to-end training + evaluation of a coarse-to-fine CNN predictor
# ---------------------------------------------------------------

import os, json, time, copy
import numpy as np
import matplotlib.pyplot as plt

import torch
print(torch.__version__)
print(torch.__file__)

import torch.nn as nn
import torch.optim as optim
import torch.utils.data as tud
from sklearn.metrics import f1_score, confusion_matrix

# ================================================================
# Configuration
# ================================================================

# Select computation device: GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("▶ Using device:", device)

# Output directories for results
os.makedirs("cNN",      exist_ok=True)   # For plots & metrics
os.makedirs("dataPairs", exist_ok=True)  # For compatibility with previous scripts

# Hyperparameters
selectionIndex = 10000          # Number of (x, X) pairs to use
batch_size      = 128
lr              = 1e-3
weight_decay    = 1e-5           # L2 regularization coefficient
max_epochs      = 10000
patience        = 100            # Early stopping patience (in epochs)
log_interval    = 1              # Print metrics every epoch

# Grid dimensions (fine-grid resolution)
H, W_grid = 65, 65               # Target resolution for fine-grid predictions

# ================================================================
# Data loading & preprocessing
# ================================================================

# Load coarse and fine grid data
with open("data.json", "r") as f:
    raw = json.load(f)

# Convert data to tensors
fineGrids   = torch.tensor(raw["x"]  [:selectionIndex],
                           dtype=torch.float32, device=device)  # (N,65,65)
coarseGrids = torch.tensor(raw["XCG"][:selectionIndex],
                           dtype=torch.float32, device=device)  # (N,17,17)

# Convert fine-scale values {0.1, ???} → {-1, +1}
x_data = torch.where(fineGrids == 0.1, -1.0, 1.0)

# Add channel dimension (1 channel for input/output)
x_data = x_data.unsqueeze(1)           # (N,1,65,65)
X_data = coarseGrids.unsqueeze(1)      # (N,1,17,17)

# Split data into training (90%) and validation (10%) sets
split = int(0.9 * selectionIndex)
X_train, X_val = X_data[:split], X_data[split:]
y_train, y_val = x_data[:split], x_data[split:]

# Create PyTorch DataLoaders
train_loader = tud.DataLoader(tud.TensorDataset(X_train, y_train),
                              batch_size=batch_size, shuffle=True)
val_loader   = tud.DataLoader(tud.TensorDataset(X_val,   y_val),
                              batch_size=batch_size, shuffle=False)

# ================================================================
# Model definition
# ================================================================
class CNNModel(nn.Module):
    """
    A simple convolutional neural network (CNN) model for coarse-to-fine prediction.
    Architecture: 3 convolutional layers with ReLU + bilinear upsampling + 1 output conv layer.
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, padding=1)
        self.upsample   = nn.Upsample((H, W_grid), mode="bilinear",
                                      align_corners=False)
        self.conv_final = nn.Conv2d(32, 1, 3, padding=1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = self.upsample(x)
        return self.conv_final(x)        # Output logits (N,1,65,65)

model = CNNModel().to(device)

# ================================================================
# Loss function, optimiser, and learning rate scheduler
# ================================================================

def neg_log_likelihood(f, y):
    """
    Compute the logistic negative log-likelihood (NLL) loss.
    f: predicted logits, y: target labels.
    """
    f_flat = f.view(f.size(0), -1)
    y_flat = y.view(y.size(0), -1)
    return torch.logaddexp(torch.zeros_like(f_flat), -y_flat * f_flat).sum()

optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=10, verbose=True
)

# ================================================================
# Training loop with full validation tracking
# ================================================================

best_val_loss = float("inf")
best_state    = None
val_losses, val_accs, val_f1s = [], [], []
epochs_no_improve = 0

t0 = time.time()
for epoch in range(1, max_epochs + 1):
    # ---- Training phase -----------------------------------------
    model.train()
    train_loss = 0.0
    for Xb, yb in train_loader:
        optimizer.zero_grad()
        loss = neg_log_likelihood(model(Xb), yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    # ---- Validation phase ---------------------------------------
    model.eval()
    v_loss = 0.0
    y_true_batches, y_pred_batches = [], []
    with torch.no_grad():
        for Xb, yb in val_loader:
            fb    = model(Xb)
            v_loss += neg_log_likelihood(fb, yb).item()
            preds  = torch.sign(torch.tanh(fb))
            y_true_batches.append(yb.view(-1).cpu())
            y_pred_batches.append(preds.view(-1).cpu())
    v_loss /= len(val_loader)

    # Compute metrics
    y_true = torch.cat(y_true_batches).numpy()
    y_pred = torch.cat(y_pred_batches).numpy()
    v_acc  = (y_true == y_pred).mean()
    v_f1   = f1_score(y_true, y_pred, average="macro")

    val_losses.append(v_loss)
    val_accs  .append(v_acc)
    val_f1s  .append(v_f1)

    scheduler.step(v_loss)

    if epoch % log_interval == 0:
        print(f"Epoch {epoch:3d}/{max_epochs} | "
              f"Train NLL={train_loss:.4f} | "
              f"Val NLL={v_loss:.4f} | "
              f"Acc={v_acc:.4f} | F1={v_f1:.4f}")

    # ---- Early stopping ----------------------------------------
    if v_loss < best_val_loss:
        best_val_loss = v_loss
        best_state    = copy.deepcopy(model.state_dict())
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break

runtime = time.time() - t0
print(f"✔ Optimisation finished in {runtime:.2f} s")

# ================================================================
# Restore best model weights & final evaluation
# ================================================================
model.load_state_dict(best_state)
torch.save(best_state, "best_model_state.pth")

# Compute final metrics on validation set
model.eval()
all_true, all_pred = [], []
with torch.no_grad():
    for Xb, yb in val_loader:
        fb   = model(Xb)
        pred = torch.sign(torch.tanh(fb))
        all_true.append(yb.view(-1).cpu())
        all_pred.append(pred.view(-1).cpu())

y_true = torch.cat(all_true).numpy()
y_pred = torch.cat(all_pred).numpy()

final_acc = (y_true == y_pred).mean()
final_f1  = f1_score(y_true, y_pred, average="macro")
confmat   = confusion_matrix(y_true, y_pred, labels=[1, -1])

# Print summary metrics
print("\n================ FINAL METRICS ================")
print(f"Runtime (s):      {runtime:.2f}")
print(f"Pixel accuracy:   {final_acc:.4f}")
print(f"Combined F1-score:{final_f1:.4f}")
print("Confusion matrix [[TP_high FN_high],[FP_high TN_high]]:")
print(confmat)
print("================================================\n")

# ================================================================
# Save plots and metrics
# ================================================================

# Plot validation loss and accuracy curves
fig, ax1 = plt.subplots()
ax1.plot(val_losses[:300], color="tab:red", label="Val NLL")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss (log scale)", color="tab:red")
ax1.set_yscale("log")
ax1.tick_params(axis="y", labelcolor="tab:red")

ax2 = ax1.twinx()
ax2.plot(val_accs[:300], color="tab:blue", label="Accuracy")
ax2.set_ylabel("Accuracy", color="tab:blue")
ax2.tick_params(axis="y", labelcolor="tab:blue")

plt.title("Training Loss & Accuracy - 10000 samples")
fig.tight_layout()
plt.savefig("cNN/cnn_loglog_val_plot.png", dpi=300)
plt.close()

# Save metrics to JSON
with open("cNN/cnn_validation_metrics.json", "a") as f:
    metrics = dict(
        runtime_sec       = runtime,
        final_accuracy    = float(final_acc),
        final_f1_combined = float(final_f1),
        val_losses        = val_losses,
        val_accuracies    = val_accs,
        val_f1_scores     = val_f1s,
        confusion_matrix  = confmat.tolist(),
    )
    f.write(json.dumps(metrics, indent=2) + "\n")

# Plot demo outputs
plt.figure(figsize=(15,5))
plt.subplot(1,3,1); plt.title("Probability(+1)"); plt.imshow(prob_demo, cmap="viridis"); plt.axis("off")
plt.subplot(1,3,2); plt.title("Variance");         plt.imshow(var_demo,  cmap="viridis"); plt.axis("off")
plt.subplot(1,3,3); plt.title("Hard prediction");  plt.imshow(pred_demo, cmap="gray");    plt.axis("off")
plt.tight_layout(); plt.savefig("cNN/demo_visualisation.png", dpi=300); plt.close()

print("🎉 All done — plots & metrics saved in the 'cNN/' folder.")
