import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import matplotlib.pyplot as plt
import os
import json
import time
import copy

# ================================================================
# Configuration
# ================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

os.makedirs('./dataPairs', exist_ok=True)

# Hyperparameters
selectionIndex = 10000
batch_size = 128
lr = 0.001
weight_decay = 1e-5  # L2 regularization strength
max_epochs = 200
patience = 200  # for early stopping

# Dimensions of the fine grid
H, W_grid = 65, 65  # Adjust based on your data

# ================================================================
# Data Loading and Preparation
# ================================================================
with open('data.json', 'r') as file:
    data = json.load(file)

with open('posterior_X_mean.json', 'r') as file2:
    posterior_X_mean = json.load(file2)
    # print("posterior_X_mean:", posterior_X_mean.size())    
    posterior_X_mean = posterior_X_mean['X_mean']
    posterior_X_mean = torch.tensor(posterior_X_mean, dtype=torch.float32, device=device)

fineGrids = data['x'][:selectionIndex]     # (N, H_fg, W_fg)
coarseGrids = data['XCG'][:selectionIndex] # (N, H_cg, W_cg)
print(len(coarseGrids))
N = len(fineGrids)

fineGrids = torch.tensor(fineGrids, dtype=torch.float32, device=device)
coarseGrids = torch.tensor(coarseGrids, dtype=torch.float32, device=device)

# Convert fineGrids from {0.1, something else} to {-1, 1}
x_data = torch.where(fineGrids == 0.1, -1.0, 1.0)

# Add channel dimension: (N, 1, H, W)
x_data = x_data.unsqueeze(1)             # (N, 1, H_fg, W_fg)
X_data = coarseGrids.unsqueeze(1)        # (N, 1, H_cg, W_cg)

# Train-test split (90% train, 10% test)
split_index = int(0.9 * N)
X_train, X_val = X_data[:split_index], X_data[split_index:]
x_train, x_val = x_data[:split_index], x_data[split_index:]

# Create datasets and dataloaders
train_dataset = torch.utils.data.TensorDataset(X_train, x_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = torch.utils.data.TensorDataset(X_val, x_val)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# ================================================================
# Model Definition (CNN)
# ================================================================
# Example CNN: You can adjust this architecture as needed.
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        # Input: (N, 1, 17, 17)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)

        # This will keep the spatial size at 17x17 so far.
        # Now we need to upsample to 65x65.
        self.upsample = nn.Upsample((65, 65), mode='bilinear', align_corners=False)
        
        # After upsampling to 65x65, apply a final convolution if needed:
        self.conv_final = nn.Conv2d(32, 1, kernel_size=3, padding=1)

    def forward(self, x):
        # x: (N, 1, 17, 17)
        x = torch.relu(self.conv1(x))   # (N, 16, 17, 17)
        x = torch.relu(self.conv2(x))   # (N, 32, 17, 17)
        x = torch.relu(self.conv3(x))   # (N, 32, 17, 17)

        # Upsample to (65, 65)
        x = self.upsample(x)            # (N, 32, 65, 65)

        # Final conv to get single-channel output
        f = self.conv_final(x)          # (N, 1, 65, 65)
        return f


model = CNNModel().to(device)

# ================================================================
# Loss Function
# ================================================================
def neg_log_likelihood(f, y):
    # f: (N, 1, H_fg, W_fg), y: (N, 1, H_fg, W_fg)
    # Flatten them
    f_flat = f.view(f.shape[0], -1) # (N, H_fg*W_fg)
    y_flat = y.view(y.shape[0], -1)
    # NLL = sum over all elements of log(1 + exp(-y*f))
    return torch.sum(torch.logaddexp(torch.zeros_like(f_flat), -y_flat * f_flat))

# ================================================================
# Optimizer & Scheduler
# ================================================================
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

# ================================================================
# Training Loop with Early Stopping
# ================================================================
best_val_loss = float('inf')
best_model_state = None
epochs_without_improvement = 0

start_time = time.time()

for epoch in range(200):
    model.train()
    train_loss = 0.0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        f = model(X_batch)    # (N, 1, H_fg, W_fg)
        loss = neg_log_likelihood(f, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            f = model(X_batch)
            loss = neg_log_likelihood(f, y_batch)
            val_loss += loss.item()
    val_loss /= len(val_loader)

    scheduler.step(val_loss)

    print(f"Epoch {epoch+1}/{max_epochs}: Train NLL={train_loss:.4f}, Val NLL={val_loss:.4f}")

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = copy.deepcopy(model.state_dict())
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement > patience:
            print("Early stopping triggered.")
            break

end_time = time.time()
print("Optimization finished in", end_time - start_time, "seconds")

# Load best model

model.load_state_dict(best_model_state)
torch.save(best_model_state, 'best_model_state.pth')
# ================================================================
# Evaluation on Test Data
# ================================================================
with torch.no_grad():
    f_val = model(X_val)
    # Convert to predictions
    # Use torch.sign(torch.tanh(f)) to map logits to {-1, 1}
    preds = torch.sign(torch.tanh(f_val))  # collapses probabilities to {-1, 1}
    accuracy = torch.mean((preds == x_val).float())
    print("Testing Accuracy:", accuracy.item())

# Compute sum of squares error for the test data
def sum_of_squares_error(model, X, x):
    with torch.no_grad():
        f = model(X)
        x_pred = torch.sign(torch.tanh(f))
        error = torch.sum((x - x_pred)**2) / x.numel()
        return error

sse = sum_of_squares_error(model, X_val, x_val)
print("Sum of Squares Error on testing data:", sse.item())

# ================================================================
# Visualization
# ================================================================
plot_dir = f'cNN_full_data/{selectionIndex}'
os.makedirs(plot_dir, exist_ok=True)

# ================================================================
# Visualization with Probability Density and Coarse Grid
# ================================================================
plot_dir = f'cNN/{selectionIndex}'
os.makedirs(plot_dir, exist_ok=True)

model.eval()
# ================================================================
# Visualization with Probability Density, Variance Map, and Coarse Grid
# ================================================================
plot_dir = f'cNN/{selectionIndex}'
os.makedirs(plot_dir, exist_ok=True)

model.eval()
with torch.no_grad():
    for n in range(100):
        # Extract the nth coarse grid and the corresponding ground truth
        testCoarseGrid = torch.tensor(coarseGrids[-n], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)  # shape: (1, 1, H_cg, W_cg)
        testCoarseGrid = torch.flip(testCoarseGrid, dims=[2])
        testFineGrid = torch.tensor(fineGrids[-n], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        testFineGrid = torch.flip(testFineGrid, dims=[2])

        f_pred = model(testCoarseGrid)          # shape: (1, 1, H_fg, W_fg) (logits)
        
        # Probability map p in [0,1] for the pixel being +1
        p_val = torch.sigmoid(f_pred)           # shape: (1, 1, H_fg, W_fg)
        
        # Bernoulli variance = p(1-p), interpreted as uncertainty
        variance_map = p_val * (1 - p_val)      # shape: (1, 1, H_fg, W_fg)
        
        # Hard predictions in {-1,+1}
        x_pred = torch.sign(torch.tanh(f_pred)).squeeze(0).squeeze(0)  # shape: (H_fg, W_fg)
        # Plotting
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot the probability map
        axs[0].imshow(p_val.squeeze().cpu().numpy(), cmap='viridis')
        axs[0].set_title('Probability Map')
        axs[0].axis('off')
        
        # Plot the variance map
        axs[1].imshow(testFineGrid.squeeze().cpu().numpy(), cmap='viridis')
        axs[1].set_title('Variance Map')
        axs[1].axis('off')
        
        # Plot the hard predictions
        axs[2].imshow(x_pred.cpu().numpy(), cmap='viridis')
        axs[2].set_title('Hard Predictions')
        axs[2].axis('off')
        
        plt.tight_layout()
        plt.savefig( f"cNN/img/{n}.png")
        plt.show()
        # Actual label grid
        