import torch
import matplotlib.pyplot as plt
import numpy as np
from pyro.infer import Predictive

# ============================
# (1) Obtain Posterior Samples for X
# ============================
# Here we use the Bayesian inverse model defined above.
# Assume that bayes_inv_model and guide are already defined and trained,
# and that y_star is loaded as in your code.
# Also, note that the temperature parameter is used in the transformation from Z to X.
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import matplotlib.pyplot as plt
import os
import json
import time
import copy

import json
import torch
from pathlib import Path

# ---------- 1) SAVE ---
# -------------------------------------------------------
import matplotlib.pyplot as plt
import torch
import numpy as np
from pathlib import Path

# ---------- 1) PLOT & SAVE ---------------------------------------------------
def plot_y_vs_uinit(y_tensor: torch.Tensor,
                    u_grid: np.ndarray,
                    fname: str | Path = "y_vs_uinit.png",
                    vmin: float | None = None,
                    vmax: float | None = None) -> None:
    """
    Display and save y (model prediction) and u_init_grid (FEM truth) side‑by‑side.
    Args
    ----
    y_tensor : torch.Tensor  – shape (H, W) or (1, H, W) or (H*W,)
    u_grid   : np.ndarray    – shape (H, W)
    fname    : output image filename
    vmin/vmax: pass explicit color limits if you want identical scales
    """
    # --- 1.1) Put y on CPU and reshape to 2‑D -------------------------------
    y_np = y_tensor.detach().cpu().numpy()
    if y_np.ndim == 3 and y_np.shape[0] == 1:             # (1, H, W)
        y_np = y_np[0]
    elif y_np.ndim == 1:                                  # (H*W,)
        side = int(np.sqrt(y_np.size))
        y_np = y_np.reshape(side, side)

    # --- 1.2) Decide on colour limits --------------------------------------
    if vmin is None or vmax is None:
        all_vals = np.concatenate([y_np.ravel(), u_grid.ravel()])
        vmin, vmax = all_vals.min(), all_vals.max()

    # --- 1.3) Make the figure ----------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), constrained_layout=True)
    im0 = axes[0].imshow(y_np, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0].set_title("Predicted $y$")
    axes[0].axis("off")
    fig.colorbar(im0, ax=axes[0], fraction=0.045)

    im1 = axes[1].imshow(u_grid, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[1].set_title(r"Reference $u_{\mathrm{init}}$")
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.045)

    # --- 1.4) Save & optionally display ------------------------------------
    plt.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[✓] Plot saved to {fname}")


# ---------- 2) CALL RIGHT AFTER y_test IS READY -----------------------------


def save_y_tensors_to_json(y_test: torch.Tensor,
                           y_star: torch.Tensor,
                           path: str | Path = "y_test_y_star.json") -> None:
    """
    Write y_test and y_star to <path> in JSON format.
    Tensors are detached, moved to CPU, converted to list‑of‑lists, and dumped.
    """
    data = {
        "y_test": y_test.detach().cpu().numpy().tolist(),
        "y_star": y_star.detach().cpu().numpy().tolist(),
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)          # indent=2 purely for readability
    print(f"[✓] Saved y_test & y_star to {path}")

# call immediately after you have both tensors

# ---------- 2) LOAD (if/when you need them again) ----------------------------
def load_y_tensors_from_json(path: str | Path = "y_test_y_star.json",
                             device: str | torch.device = "cuda:0") -> tuple[torch.Tensor, torch.Tensor]:
    """
    Read <path> and rebuild y_test, y_star as float32 tensors on <device>.
    """
    with open(path) as f:
        data = json.load(f)

    y_test  = torch.tensor(data["y_test"], dtype=torch.float32, device=device)
    y_star  = torch.tensor(data["y_star"], dtype=torch.float32, device=device)
    return y_test, y_star

# Example usage later:
# y_test_loaded, y_star_loaded = load_y_tensors_from_json()


# ================================================================
# Configuration
# ================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


with open('data.json', 'r') as file:
    data = json.load(file)

with open('posterior_X_mean.json', 'r') as file2:
    posterior_X_mean = json.load(file2)
    # print("posterior_X_mean:", posterior_X_mean.size())    
    posterior_X_mean = posterior_X_mean['X_mean']
    posterior_X_mean = torch.tensor(posterior_X_mean, dtype=torch.float32, device=device)

fineGrids = data['x'][:100]     # (N, H_fg, W_fg)
coarseGrids = data['XCG'][:100] # (N, H_cg, W_cg)
print(len(coarseGrids))
N = len(fineGrids)

fineGrids = torch.tensor(fineGrids, dtype=torch.float32, device=device)

os.makedirs('./dataPairs', exist_ok=True)
from dolfin import UnitSquareMesh, FunctionSpace, Function, Constant, \
                   TrialFunction, TestFunction, DirichletBC, solve, grad, dot, dx
import numpy as np

def solve_darcy(permeability_field: np.ndarray):
    """
    Solve  −∇·(k ∇u) = 100  on the unit square, u = 0 on ∂Ω,
    for a binary permeability on a 65×65 Cartesian grid.
    """
    nx = ny = permeability_field.shape[0]          # = 65
    mesh = UnitSquareMesh(nx - 1, ny - 1)
    V    = FunctionSpace(mesh, "P", 1)

    # --- piece-wise constant k(x) from the 65×65 table ---------------
    k_fun  = Function(V)
    coords = V.tabulate_dof_coordinates().reshape(-1, 2)
    ix = np.minimum((coords[:, 0] * (nx - 1)).round().astype(int), nx - 1)
    iy = np.minimum((coords[:, 1] * (ny - 1)).round().astype(int), ny - 1)
    k_fun.vector()[:] = permeability_field[iy, ix]

    # --- variational form -------------------------------------------
    u = TrialFunction(V);  v = TestFunction(V)
    a = dot(k_fun * grad(u), grad(v)) * dx
    L = Constant(100.0) * v * dx
    bc = DirichletBC(V, Constant(0.0), "on_boundary")

    u_sol = Function(V)
    solve(a == L, u_sol, bc, solver_parameters={"linear_solver": "cg"})
    return u_sol

# def fem_solution_to_regular_grid(u_sol, nx=65, ny=65):
#     """
#     Map the FEM nodal solution to a (ny, nx) NumPy array that follows the
#     same row-major convention as the original permeability grid and then
#     mirror it about the x-axis (flip the rows top↔bottom).
#     """
#     # Nodal coordinates and values
#     coords  = u_sol.function_space().tabulate_dof_coordinates().reshape(-1, 2)
#     values  = u_sol.vector().get_local()

#     # Discrete pixel indices
#     ix = np.minimum((coords[:, 0] * (nx - 1)).round().astype(int), nx - 1)
#     iy = np.minimum((coords[:, 1] * (ny - 1)).round().astype(int), ny - 1)

#     # Row-major ordering of DOFs
#     order = np.argsort(iy * nx + ix)
#     grid  = values[order].reshape(ny, nx)

#     # Mirror about the x-axis  →  flip rows
#     grid_flipped = grid[::-1, :]

#     return grid_flipped
def fem_solution_to_regular_grid(u_sol, nx=65, ny=65):
    """
    Map the DOF vector to a (ny, nx) NumPy array whose (row, col) indices
    match the original Cartesian grid, with correct orientation for plotting.
    """
    coords = u_sol.function_space().tabulate_dof_coordinates().reshape(-1, 2)
    values = u_sol.vector().get_local()

    # integer raster indices
    ix = np.minimum((coords[:, 0] * (nx - 1)).round().astype(int), nx - 1)
    iy = np.minimum((coords[:, 1] * (ny - 1)).round().astype(int), ny - 1)

    # order DOFs by (iy, ix) so that a simple reshape produces row-wise order
    order = np.argsort(iy * nx + ix)
    grid = values[order].reshape(ny, nx)

    # Flip vertically and return a copy to ensure PyTorch compatibility
    return grid.copy()
# Hyperparameters
selectionIndex = 10000
batch_size = 128
lr = 0.001
weight_decay = 1e-5  # L2 regularization strength
max_epochs = 10
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

x_test=x_data[0]
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
max_epochs = 100
start_time = time.time()

# for epoch in range(max_epochs):
#     model.train()
#     train_loss = 0.0
#     for X_batch, y_batch in train_loader:
#         optimizer.zero_grad()
#         f = model(X_batch)    # (N, 1, H_fg, W_fg)
#         loss = neg_log_likelihood(f, y_batch)
#         loss.backward()
#         optimizer.step()
#         train_loss += loss.item()

#     train_loss /= len(train_loader)

#     # Validation
#     model.eval()
#     val_loss = 0.0
#     with torch.no_grad():
#         for X_batch, y_batch in val_loader:
#             f = model(X_batch)
#             loss = neg_log_likelihood(f, y_batch)
#             val_loss += loss.item()
#     val_loss /= len(val_loader)

#     scheduler.step(val_loss)

#     print(f"Epoch {epoch+1}/{max_epochs}: Train NLL={train_loss:.4f}, Val NLL={val_loss:.4f}")

#     # Early stopping
#     if val_loss < best_val_loss:
#         best_val_loss = val_loss
#         best_model_state = copy.deepcopy(model.state_dict())
#         epochs_without_improvement = 0
#     else:
#         epochs_without_improvement += 1
#         if epochs_without_improvement > patience:
#             print("Early stopping triggered.")
#             break

# end_time = time.time()
# print("Optimization finished in", end_time - start_time, "seconds")

# Load best model

# model.load_state_dict(best_model_state)
# torch.save(best_model_state, 'best_model_state.pth')
model.load_state_dict(torch.load('best_model_state.pth'))
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
    # for n in range(100):
        # Extract the nth coarse grid and the corresponding ground truth
        testCoarseGrid = torch.tensor(posterior_X_mean, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)  # shape: (1, 1, H_cg, W_cg)
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
        axs[1].imshow(variance_map.squeeze().cpu().numpy(), cmap='viridis')
        axs[1].set_title('Variance Map')
        axs[1].axis('off')
        
        # Plot the hard predictions
        axs[2].imshow(x_pred.cpu().numpy(), cmap='gray')
        axs[2].set_title('Hard Predictions')
        axs[2].axis('off')
        
        plt.tight_layout()
        plt.savefig( 'visualization.png')
        plt.show()
        # Actual label grid
import torch
import pyro
import pyro.distributions as dist
import gpytorch
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from pyro.nn import PyroModule, PyroParam
from pyro.infer.autoguide import AutoNormal
import matplotlib.pyplot as plt

import torch
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS, HMC
import gpytorch
import numpy as np
import json
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import pyro
import pyro.distributions as dist
import pyro.infer.mcmc as mcmc
import pyro.infer.mcmc.api as mcmc_api
from pyro.infer import NUTS, MCMC
import numpy as np
import gpytorch
import torch
import pyro
import pyro.distributions as dist
import pyro.infer.mcmc as mcmc
from pyro.infer import MCMC, NUTS, HMC
from pyro.infer.autoguide import AutoMultivariateNormal
import gpytorch
import numpy as np
import json
import matplotlib.pyplot as plt
### Importing Libraries ###
from matplotlib.patheffects import Normal
import numpy as np
import torch
import os
import time
### Importing Manual Modules ###
from model.ProbModels.cgCnnMvn import probabModel
from model.pde.pdeForm2D import pdeForm
from utils.PostProcessing import postProcessing
from utils.tempData import storingData
from input import *
from utils.variousFunctions import calcRSquared, calcEpsilon, makeCGProjection, setupDevice, createFolderIfNotExists, memoryOfTensor, list_tensors
from utils.saveEvaluationDataset import saveDatasetAll, importDatasetAll
from model.pde.pdeTrueSolFenics import solve_pde
import warnings
import json
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt

device = setupDevice(cudaIndex, device, dataType)

### Constructing Post Processing Instance ###
post = postProcessing(path='./results/data/', displayPlots=display_plots)

# Print internals of postProcessing instance
print(f"PostProcessing Path: {post.path}")
print(f"Display Plots: {post.displayPlots}")

# Save some data and print confirmation
post.save([['numOfTestSamples', np.array(numOfTestSamples)]])
print("Saved number of test samples.")

### Reading Existing Dataset (if it exists) ###
if (not createNewCondField) and not importDatasetOption:
    sampSol, sampSolFenics, sampCond, sampX, sampYCoeff = post.readTestSample(numOfTestSamples)
    print("Read existing test samples.")
    torch.save(sampX, './model/pde/RefSolutions/condFields/' + 'sampX.dat')
    print("Saved sampX to file.")

### Importing Dataset ###
if importDatasetOption and not createNewCondField:
    if os.path.exists(saveDatasetName):
        print("Dataset:" + saveDatasetName + " exists. Importing...")
        sampCond, sampSolFenics, sampX, sampYCoeff, sampSol, gpEigVals, gpEigVecs = importDatasetAll(datapath=saveDatasetName, device=device)
        createFolderIfNotExists('./model/pde/RefSolutions/condFields/')
        torch.save(sampX, './model/pde/RefSolutions/condFields/' + 'sampX.dat')
        torch.save(gpEigVals, './model/pde/RefSolutions/condFields/' + 'gpEigVals.dat')
        torch.save(gpEigVecs, './model/pde/RefSolutions/condFields/' + 'gpEigVecs.dat')
        print("Imported and saved dataset.")
    else:
        raise ValueError("Dataset:" + saveDatasetName + " does not exist. Ignoring the command")

### Definition and Form of the PDE ###
pde = pdeForm(nele, shapeFuncsDim, mean_px, sigma_px, sigma_r, Nx_samp, createNewCondField, device, post, rhs=rhs, reducedDim=reducedDim, options=options)

# Print some internals of the pdeForm instance
print(f"Shape Functions Dimension: {pde.shapeFuncsDim}")
print(f"Number of Elements: {pde.nele}")
print(f"Grid Shape: {pde.grid.shape}")
print(f"Grid first 10: {pde.grid}")
print(f"GridW Shape: {pde.gridW.shape}")
print(f"GridW2 Shape: {pde.gridW2.shape}")
print(f"Number of Shape Functions: {pde.NofShFuncs}")
print(f"Number of Shape Functions W: {pde.NofShFuncsW}")
print(f"Node Coordinates: {pde.node_corrs.shape}")
print(f"Options: {pde.options}")
print(f"Integration Points: {pde.intPoints}")
print(f"Mean px: {pde.mean_px}")
print(f"Sigma px: {pde.sigma_px}")
print(f"Sigma r: {pde.sigma_r}")
print(f"Number of Samples: {pde.Nx_samp}")
print(f"RHS: {pde.rhs}")
print(f"Boundary Condition: {pde.uBc}")
print(f"Length Scale: {pde.lengthScale}")
print(f"GP Sigma: {pde.gpSigma}")
print(f"Fraction: {pde.fraction}")
print(f"Phase High: {pde.phaseHigh}")
print(f"Phase Low: {pde.phaseLow}")


### Creating instance of the Probabilistic Model ###
samples = probabModel(pde, stdInit=stdInit, lr=lr, sigma_r=sigma_r, yFMode=yFMode, randResBatchSize=randResBatchSize, reducedDim=reducedDim)

print(saveFileName)
samples.neuralNet.load_state_dict(torch.load(f"./utils/trainedNNs/trainedCGdim{saveFileName}.pth"))
samples.neuralNet.pde = pde
Navg = 100
samples.globalSigma = torch.load(f"./utils/trainedNNs/trainedCGglobalSigma_dim{saveFileName}.pth").to(device)
samples.V = torch.load(f"./utils/trainedNNs/trainedCGV_dim{saveFileName}.pth").to(device)
samples.neuralNet.V = samples.V
samples.neuralNet.globalSigma = samples.globalSigma
FR = options['volumeFractionOutOfDistribution']
samples.neuralNet.eval()
print("Model loaded successfully.")
import torch
import pyro
import pyro.distributions as dist
import pyro.infer.mcmc as mcmc
from pyro.infer import MCMC, NUTS
import gpytorch
import numpy as np
import json
import matplotlib.pyplot as plt
def get_py_given_x_distribution(x, neuralNet, V, globalSigma):
    mean = (neuralNet.forward(x))
    V_exp = torch.pow(10, V)
    cov_lowrank = V_exp @ V_exp.T
    sigma_scalar = torch.pow(10, globalSigma)
    dimOut = cov_lowrank.size(0)
    cov_matrix = cov_lowrank + sigma_scalar * torch.eye(dimOut, device=V.device)
    return dist.MultivariateNormal(mean, covariance_matrix=cov_matrix)
def get_py_given_X_distribution(x, neuralNet, V, globalSigma):
    mean = (neuralNet.forward_fromX(x))
    V_exp = torch.pow(10, V)
    cov_lowrank = V_exp @ V_exp.T
    sigma_scalar = torch.pow(10, globalSigma)
    dimOut = cov_lowrank.size(0)
    cov_matrix = cov_lowrank + sigma_scalar * torch.eye(dimOut, device=V.device)
    return dist.MultivariateNormal(mean, covariance_matrix=cov_matrix)
# Load observed y values from JSON
for i in range(99,100):
    with open("Y_samples.json", "r") as f:
        # y_samples1 = torch.tensor(json.load(f)["ys"]).reshape(1, 1,32*32)
        # print("y_samples1:",y_samples1.shape)
        sampSol, sampSolFenics, sampCond, sampX, sampYCoeff = pde.produceTestSample(Nx=numOfTestSamples, post=post)
        sampX=sampX[:1].repeat(numOfTestSamples, 1)
        x_testing_samples = sampX[:1].repeat(numOfTestSamples, 1)
        xKLE = x_testing_samples[:1].view(1, 1, 24) #original small x
        x = pde.gpExpansionExponentialParallel(xKLE) #add noise (sigma + random draw from standard normal distribution)
        # # Reshape xKLE and repeat for the specified number of test samples
        xKLE = torch.reshape(xKLE, [xKLE.size(dim=0), 1, -1]) \
            .repeat(numOfTestSamples, 1, 1) \
            .to('cuda:0')

        # Remove unnecessary dimensions from x
        # x = x.squeeze(1).to(device)

        # Generate XCG using the neural network's xtoXCnn function
        # XCG = samples.neuralNet.xtoXCnn(xKLE).squeeze(1)
        # XCG = XCG[:1]

        # x_init_grid = x[0].detach().cpu().numpy().reshape(65, 65)   # adjust reshape if needed
        x_init_grid = fineGrids[-i].detach().cpu().numpy().reshape(65, 65)   # adjust reshape if needed

        # # 2) Map {-1,+1} flags back to physical permeabilities {0.1, 1.0}
        # if x_init_grid.max() <= 1.0 and x_init_grid.min() < 0.0:
        #     k_init = np.where(x_init_grid > 0, 1.0, 0.1)
        # else:                                   # already in physical units
        #     k_init = x_init_grid.copy()

        # k_init = x[0].detach().cpu().numpy().reshape(65, 65)   # adjust reshape if needed
        k_init = x_init_grid.copy()

        # 3) High-fidelity Darcy solve
        u_init      = solve_darcy(k_init)
        u_init_grid = fem_solution_to_regular_grid(u_init)          # shape (65, 65)

        # 4) Convert the pressure field to a torch tensor
        rbfRefSol = torch.tensor(u_init_grid, dtype=torch.float32, device=device)

        # 5) Get RBF coefficients  →  this *is* the new y★
        #    findRbfCoeffs expects (batch, H, W); add batch dim if needed
        rbf_coeffs = pde.shapeFunc.findRbfCoeffs(rbfRefSol.unsqueeze(0)).squeeze(0)

        # 6) Over-write y_star for the rest of the pipeline
        y_star = rbf_coeffs.detach().reshape(1, 1024)  # reshape to [1, 1024]
        print("y_star", y_star)
        y_samples = samples.neuralNet.forward(xKLE.to(device)).detach()

        # Apply the trial solution transformation
        # y = pde.shapeFunc.cTrialSolutionParallel(y.to(device)).cpu()

        # # Reshape y to match the desired dimensions
        # y = torch.reshape(y, [y.size(dim=0), pde.sgrid.size(dim=1), pde.sgrid.size(dim=2)]).detach().cpu()

        # yys, yyMean= samples.samplePosteriorMvn(sampX, Nx=numOfTestSamples, Navg=Navg)

        # print("yys", yys.shape)

        # yvalue= pde.shapeFunc.cTrialSolutionParallel(yys[0].to(device)).cpu()
        
        # Draw 100 ys for the current XCG
        # print("y", yys.shape)

        # y_samples = torch.tensor(samples.samplePosteriorMvn(x_test))

        y = pde.shapeFunc.cTrialSolutionParallel(y_samples)# Reshape y to match the desired dimensions
        y = torch.reshape(y, [y.size(dim=0), pde.sgrid.size(dim=1), pde.sgrid.size(dim=2)]).detach().cpu()    
        # plt.imshow(y_samples[0], cmap='viridis')
        # plt.colorbar()
        # plt.title("Observed y*")
        # plt.savefig("observed_y.png")
        y_test=y[0].to(device)
        print("y_test", y_test)
        save_y_tensors_to_json(y_test, y_star)
        plot_y_vs_uinit(
            y[0],          # ← the network prediction tensor
            u_init_grid      # ← the 65×65 FEM pressure field (numpy array)
            # You can add vmin=..., vmax=... here if you want fixed colour limits
        )

        # print("shape:",y_samples.shape)  # Shape: (1000, 32, 32)

    # Initialize X as a 17x17 grid with ones
    X_init = torch.full((17, 17), 0.5, requires_grad=True)
    # # Draw one sample y from the distribution p(y | X)
    # y_draw = get_py_given_X_distribution(X_init, samples.neuralNet, samples.V, samples.globalSigma).sample()

    y_sample=pde.shapeFunc.cTrialSolutionParallel(y_star.to(device)).cpu()           # Reshape y to match the desired dimensions
    # y_sample = torch.reshape(y_sample, [y_sample.size(dim=0), pde.sgrid.size(dim=1), pde.sgrid.size(dim=2)]).detach().cpu()
    # Reshape and visualize the sampled y
    y_sample_reshaped = torch.reshape(y_sample, [1, pde.sgrid.size(dim=1), pde.sgrid.size(dim=2)]).detach().cpu()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ### Load observed y_star from your dataset
    # with open("Y_samples.json", "r") as f:
    #     y_samples = torch.tensor(json.load(f)["ys"]).reshape(1, 1, 32 * 32).to(device)
    #     y_star = y_samples[0]

    ### Define the Gaussian Process Prior for X ###
    import torch
    import pyro
    import pyro.distributions as dist
    import gpytorch
    from pyro.infer import SVI, Trace_ELBO
    from pyro.optim import Adam
    from pyro.nn import PyroModule, PyroParam
    import numpy as np
    import matplotlib.pyplot as plt

    # Ensure computation is done on the correct device
    import torch
    import pyro
    import pyro.distributions as dist
    import gpytorch
    from pyro.infer import SVI, Trace_ELBO
    from pyro.optim import Adam
    from pyro.nn import PyroModule, PyroParam
    import matplotlib.pyplot as plt
    import numpy as np
    import json
    import os

    # Ensure computation is done on the correct device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Assume all your other modules (pdeForm, probabModel, etc.) and setup are already imported/defined ---
    import torch
    import pyro
    import pyro.distributions as dist
    import gpytorch
    from pyro.infer import SVI, Trace_ELBO, Predictive
    from pyro.optim import Adam
    from pyro.infer.autoguide import AutoMultivariateNormal
    from pyro.nn import PyroModule
    from torch.distributions import constraints
    import matplotlib.pyplot as plt
    import numpy as np

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Build 2D Coordinates for GP Prior ---
    x_coords = torch.linspace(0, 1, 17)
    y_coords = torch.linspace(0, 1, 17)
    X1, X2 = torch.meshgrid(x_coords, y_coords, indexing='ij')
    train_x = torch.stack([X1.reshape(-1), X2.reshape(-1)], dim=-1).to(device)

    latent_dim = 17 * 17
    train_y = torch.zeros(latent_dim, device=device)
    # =============================================================================
    # 1. Define the GP Prior (no changes needed here)
    # =============================================================================
    class AnisotropicGP_Prior(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super(AnisotropicGP_Prior, self).__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            # Using ARD: learn different lengthscales for x and y.
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=2)
            )
            self.likelihood.noise = torch.tensor(1e-3, device=device)

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            jitter = 1e-4 * torch.eye(covar_x.shape[0], device=device)
            covar_x = covar_x + jitter
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    likelihood_gp = gpytorch.likelihoods.GaussianLikelihood().to(device)
    gp_prior = AnisotropicGP_Prior(train_x, train_y, likelihood_gp).to(device)

    # --- Optionally, check GP prior statistics ---
    with torch.no_grad():
        gp_out = gp_prior(train_x)
        print("GP Prior Mean (first 5):", gp_out.mean[:5])
        print("GP Prior Variance (first 5):", gp_out.covariance_matrix.diag()[:5])
    print("y_star shape:", y_star.shape)
    # --- Bayesian Inverse Model with Sigmoid Transformation and Penalty ---
    class BayesianInverseModel(PyroModule):
        def __init__(self, neuralNet, V, globalSigma, temperature=3.0, lambda_reg=10.0):
            super().__init__()
            self.neuralNet = neuralNet
            self.V = V
            self.globalSigma = globalSigma
            self.temperature = temperature
            self.lambda_reg = lambda_reg

        def model(self, y_star):
            # Sample the unconstrained latent variable Z from the 2D GP prior.
            gp_dist = gp_prior(train_x)
            mean_z = gp_dist.mean
            cov_matrix = gp_dist.covariance_matrix + 1e-4 * torch.eye(gp_dist.covariance_matrix.shape[0], device=device)
            Z = pyro.sample("Z", dist.MultivariateNormal(mean_z, covariance_matrix=cov_matrix))
            
            # Transform Z to get X in (0,1) using a steep sigmoid.
            X = torch.sigmoid(self.temperature * Z)
            X=X.reshape(-1,17,17)
            
            # Penalty for intermediate values.
            penalty = -self.lambda_reg * torch.sum(X * (1 - X))
            pyro.factor("bimodal_penalty", penalty)
            
            # Compute likelihood p(y|X) using the forward model.
            p_y_given_X = get_py_given_X_distribution(X, self.neuralNet, self.V, self.globalSigma)
            pyro.sample("y_obs", p_y_given_X, obs=y_star)
        
        def guide(self, y_star):
            # We use an automatic guide for the latent variable Z.
            pass  # Will be provided externally.

    # --- Use an AutoMultivariateNormal Guide ---
    # (Assuming samples.neuralNet, samples.V, samples.globalSigma are loaded from your existing code.)
    from pyro.infer.autoguide import AutoLowRankMultivariateNormal
    bayes_inv_model = BayesianInverseModel(samples.neuralNet, samples.V, samples.globalSigma,
                                            temperature=10.0, lambda_reg=10.0).to(device)
    guide = AutoMultivariateNormal(bayes_inv_model.model)

    optimizer = Adam({"lr": 0.01})
    svi = SVI(bayes_inv_model.model, guide, optimizer, loss=Trace_ELBO())

    # # --- Load y_star as before ---
    # with open("Y_samples.json", "r") as f:
    #     y_samples = torch.tensor(json.load(f)["ys"]).reshape(1, 1, 32*32).to(device)
    #     y_star = y_samples[0]

    num_steps = 10
    losses = []
    for step in range(num_steps):
        loss = svi.step(y_star)
        losses.append(loss)
        if step % 100 == 0:
            print(f"Step {step} | ELBO: {loss:.3f}")

    # --- Posterior Sampling and Visualization ---
    num_samples = 500
    predictive = Predictive(bayes_inv_model.model, guide=guide, num_samples=1, parallel=False)
    posterior_dict = predictive(y_star)
    Z_samples = posterior_dict["Z"]  # shape: (num_samples, latent_dim)
    # Transform to X:
    X_samples = torch.sigmoid(bayes_inv_model.temperature * Z_samples)
    posterior_X_samples = X_samples.detach().cpu().numpy()
    posterior_X_mean = np.mean(posterior_X_samples, axis=0).reshape(17, 17)


    plt.figure()
    plt.imshow(posterior_X_mean, cmap="viridis")
    plt.colorbar()
    plt.title("Estimated X (Posterior Mean)")
    plt.savefig("posterior_X.png")
    plt.show()

    with open("posterior_X_mean.json", "w") as f:
        json.dump({"X_mean": posterior_X_mean.tolist()}, f)

    num_posterior_samples = 100  # number of samples to draw from p(X|y)
    # We use Predictive to sample from the guide/model.
    predictive = Predictive(bayes_inv_model.model, guide=guide, num_samples=num_posterior_samples, parallel=False)
    posterior_samples = predictive(y_star)

    # Extract latent variable samples Z (shape: [num_samples, latent_dim])
    Z_samples = posterior_samples["Z"]  # latent_dim should equal 17*17

    # Transform to X using the sigmoid with the chosen temperature.
    # (You may already have this in your code; adjust the temperature as needed.)
    X_samples = torch.sigmoid(bayes_inv_model.temperature * Z_samples)
    # Reshape each sample into a 2D coarse grid of shape (17, 17)
    X_samples = X_samples.view(num_posterior_samples, 1, 17, 17)

    print(f"Obtained {num_posterior_samples} posterior samples for X with shape: {X_samples.shape}")

    # ============================
    # (2) For Each Sample X, Generate a High-Resolution x Sample via the CNN
    # ============================
    # We assume your trained CNN is called "model" (loaded earlier) and is in evaluation mode.
    import torch
    import numpy as np
    import json
    import matplotlib.pyplot as plt
    from pyro.infer import Predictive

    # -----------------------------------------------------------------------------
    # Assume `model`, `guide`, `svi`, `bayes_inv_model`, `y_star`, `x`, `device` are
    # already defined in the surrounding scope
    # -----------------------------------------------------------------------------

    # Ensure we are in evaluation mode for deterministic layers like Dropout/BatchNorm
    import torch
    import numpy as np
    import json
    import matplotlib.pyplot as plt
    from pyro.infer import Predictive

    # -----------------------------------------------------------------------------
    # Assume `model`, `guide`, `svi`, `bayes_inv_model`, `y_star`, `x`, `device` are
    # already defined in the surrounding scope
    # -----------------------------------------------------------------------------

    import os
    import json
    import datetime as _dt

    import numpy as np
    import matplotlib.pyplot as plt
    import torch
    from pyro.infer import Predictive

    # -----------------------------------------------------------------------------
    # Assume `model`, `guide`, `svi`, `bayes_inv_model`, `y_star`, `x`, `device` are
    # already defined in the surrounding scope
    # -----------------------------------------------------------------------------

    model.eval()  # inference‑mode for BatchNorm/Dropout etc.

    # -----------------------------------------------------------------------------
    # 1) Relative noise levels ------------------------------------------------------
    # -----------------------------------------------------------------------------
    # noise_levels = [0, 0.01, 0.05, 0.10, 0.20, 0.50]  # 0 % → 50 %
    noise_levels =[0]

    # Storage containers -----------------------------------------------------------
    run_accuracy  = {}  # {str(σ_rel): float}
    x_post_means  = []  # list[np.ndarray] per σ_rel
    runtime = {}

    # -----------------------------------------------------------------------------
    # 2) Loop over each noise level -------------------------------------------------
    # -----------------------------------------------------------------------------
    for σ_rel in noise_levels:
        # (a) Noisy observation
        y_noisy = y_star + torch.randn_like(y_star) * (σ_rel * torch.abs(y_star))

        start_time = time.time()

        # (b) Re‑optimise guide via SVI
        for step in range(2000):
            elbo = svi.step(y_noisy)
            if step % 100 == 0:
                print(f"σ_rel={σ_rel:.2f} | step={step:4d} | ELBO={elbo: .3f}")

        # (c) Posterior sampling Z → X  (coarse 17×17 grid)
        S    = 500
        pred = Predictive(bayes_inv_model.model, guide=guide, num_samples=S, parallel=False)
        Z    = pred(y_noisy)["Z"]                                  # (S,289)
        X    = torch.sigmoid(bayes_inv_model.temperature * Z)
        X    = X.view(S, 1, 17, 17).to(device, dtype=torch.float32)  # (S,1,17,17)

        # (d) Monte‑Carlo CNN upscale 17×17 → 65×65
        x_samples = []
        with torch.no_grad():
            for Xi in X:
                logits = model(Xi.unsqueeze(0))         # (1,1,65,65)
                p      = torch.sigmoid(logits)
                b      = torch.bernoulli(p)
                x_samples.append((2*b - 1).squeeze().cpu().numpy())
        x_MC = np.stack(x_samples)                      # (S,65,65)

        end_time = time.time()

        runtime[f"{σ_rel:.2f}"] = end_time - start_time
        # (e) Posterior mean + accuracy
        mean_map  = x_MC.mean(axis=0)
        plt.figure()
        plt.imshow(mean_map, cmap='viridis', vmin=-1, vmax=1)
        plt.title("Posterior Mean")
        plt.colorbar(label="Mean x")
        plt.savefig(f"posterior_mean_plot_acc_{σ_rel*100:.2f}.png", dpi=200)
        plt.close()
        x_post_means.append(mean_map)
        x_post_means.append(mean_map)
        hard_map = np.where(mean_map > 0, 1, 0.1)

        print(f"Posterior mean shape: {hard_map.shape}")

        gt_np = fineGrids[-i].cpu().numpy()
        gt_np = np.where(gt_np == 1, 1, 0.1)

        print (f"Ground truth shape: {gt_np.shape}")
        acc   = (hard_map == gt_np).mean()

        mask_wrong = (hard_map != gt_np)           # shape (65, 65) boolean

        run_accuracy[f"{σ_rel:.2f}"] = float(acc)
        print(f"Pixel‑wise accuracy @ σ_rel={σ_rel:.2f}: {acc*100:.2f}%")

        # ---------------------------------------------------------------
        # Sanity‑check: which discrete values are actually present?
        # ---------------------------------------------------------------
        print("Unique values in x_gt (ground truth) :", np.unique(gt_np))
        print("Unique values in hard_map (prediction):", np.unique(hard_map))

        # and, if you want the counts as well:
        gt_vals, gt_counts   = np.unique(gt_np,   return_counts=True)
        pred_vals, pred_cnts = np.unique(hard_map, return_counts=True)
        print("\nGround‑truth counts :", dict(zip(gt_vals,   gt_counts)))
        print("Prediction counts    :", dict(zip(pred_vals, pred_cnts)))


    # -----------------------------------------------------------------------------
    # 3) Persist accuracy results ---------------------------------------------------
    # -----------------------------------------------------------------------------
    json_path  = "accuracy_vs_noise.json"
    run_stamp  = _dt.datetime.now().isoformat(timespec="seconds")
    run_record = {"timestamp": run_stamp, "accuracy": run_accuracy, "runtime": runtime}

    if os.path.isfile(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            previous = json.load(f)
            if isinstance(previous, list):
                history = previous
            else:  # backward‑compat: single‑dict format → wrap in list
                history = [previous]
    else:
        history = []

    history.append(run_record)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"✓ Appended results to '{json_path}'. Total runs stored: {len(history)}")

    # -----------------------------------------------------------------------------
    # 4) Visual figure -------------------------------------------------------------
    # -----------------------------------------------------------------------------
    fig, axs = plt.subplots(1, len(noise_levels) + 1,
                        figsize=(4*(len(noise_levels)+1), 5))

    # (a) Ground truth
    axs[0].imshow(gt_np, cmap="viridis", vmin=-1, vmax=1)
    axs[0].set_title("Ground truth")
    axs[0].axis("off")

    # (b) Posterior means
    for ax, xm, σ in zip(axs[1:], [mask_wrong], noise_levels):
        im = ax.imshow(xm, cmap="viridis", vmin=-1, vmax=1)
        ax.set_title(f"σ_rel = {int(100*σ)}%")
        ax.axis("off")

    fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.6, label="E[x | y_noisy]",
                orientation="horizontal", pad=0.02)

    plt.tight_layout()
    fig.savefig("fine_grid_posterior_vs_noise_and_gt.png", dpi=200)
    plt.show()
