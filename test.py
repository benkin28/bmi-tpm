import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.distributions import transforms
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import matplotlib.pyplot as plt
import os
import json
import time
from dolfin import UnitSquareMesh, FunctionSpace, Function, Constant, \
                   TrialFunction, TestFunction, DirichletBC, solve, grad, dot, dx
import torch
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule
import math
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.optim import Adam

# ================================================================
# 1) Convolutional surrogate ─────────────────────────────────────
# ================================================================
class CNNModel(nn.Module):
    """3-layer CNN that upsamples 17×17 → 65×65 and outputs a single channel."""
    def __init__(self):
        super().__init__()

        # Encoder on the 17×17 coarse grid
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)   # (N,16,17,17)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # (N,32,17,17)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)  # (N,32,17,17)

        # Upsample to the 65×65 fine grid
        self.upsample    = nn.Upsample((65, 65), mode="bilinear", align_corners=False)
        self.conv_final  = nn.Conv2d(32, 1, kernel_size=3, padding=1)  # (N,1,65,65)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = self.upsample(x)
        f = self.conv_final(x)
        return f   # (N, 1, 65, 65)


# ================================================================
# 2) Bayesian inverse model  ─────────────────────────────────────
# ================================================================
class BayesianInverseModel(PyroModule):
    """
    Latent field X ~ LogisticNormal; likelihood from physics-informed NN.
    Auto-guide is attached externally.
    """
    def __init__(self, neuralNet, V, globalSigma):
        super().__init__()
        self.neuralNet   = neuralNet
        self.V           = V
        self.globalSigma = globalSigma

    def model(self, y_star, mask):
        # 1) Prior: Z ∼ MVN(0, Σ) → X = σ(τ Z)
        base_dist = dist.MultivariateNormal(mean_vec, covariance_matrix=cov_matrix)
        X = pyro.sample(
            "X",
            dist.TransformedDistribution(base_dist, logistic_normal_transform),
        ).view(-1, 17, 17)

        # 2) Likelihood p(y | X)
        p_y = get_py_given_X_distribution(X, self.neuralNet, self.V, self.globalSigma)

        # 3) Observe only masked coefficients
        mean_flat = p_y.mean.view(-1)
        var_flat  = p_y.covariance_matrix.diagonal(dim1=-2, dim2=-1).view(-1)

        y_flat    = y_star.view(-1)
        mask_flat = mask.view(-1).bool()

        mean_obs  = mean_flat[mask_flat]
        std_obs   = torch.sqrt(var_flat[mask_flat] + 1e-6)
        y_obs     = y_flat[mask_flat]

        pyro.sample("y_obs", dist.Normal(mean_obs, std_obs).to_event(1), obs=y_obs)

    def guide(self, y_star, mask):
        pass  # handled automatically by pyro.infer.autoguide.*


# ================================================================
# 3) PDE tools ───────────────────────────────────────────────────
# ================================================================
def solve_darcy(permeability_field: np.ndarray):
    """
    Solve −∇·(k ∇u) = 100 on the unit square with u = 0 on ∂Ω
    for a binary 65×65 permeability field.
    """
    nx = ny = permeability_field.shape[0]              # 65
    mesh = UnitSquareMesh(nx - 1, ny - 1)
    V    = FunctionSpace(mesh, "P", 1)

    # --- map 65×65 table → piece-wise constant k(x)
    k_fun  = Function(V)
    coords = V.tabulate_dof_coordinates().reshape(-1, 2)
    ix = np.minimum((coords[:, 0] * (nx - 1)).round().astype(int), nx - 1)
    iy = np.minimum((coords[:, 1] * (ny - 1)).round().astype(int), ny - 1)
    k_fun.vector()[:] = permeability_field[iy, ix]

    # --- variational form
    u = TrialFunction(V)
    v = TestFunction(V)
    a = dot(k_fun * grad(u), grad(v)) * dx
    L = Constant(100.0) * v * dx
    bc = DirichletBC(V, Constant(0.0), "on_boundary")

    u_sol = Function(V)
    solve(a == L, u_sol, bc, solver_parameters={"linear_solver": "cg"})
    return u_sol


def fem_solution_to_regular_grid(u_sol, nx: int = 65, ny: int = 65) -> np.ndarray:
    """
    Map the FEniCS solution back to a (ny, nx) NumPy array with the
    Cartesian orientation expected by downstream plotting.
    """
    coords = u_sol.function_space().tabulate_dof_coordinates().reshape(-1, 2)
    values = u_sol.vector().get_local()

    ix = np.minimum((coords[:, 0] * (nx - 1)).round().astype(int), nx - 1)
    iy = np.minimum((coords[:, 1] * (ny - 1)).round().astype(int), ny - 1)

    order = np.argsort(iy * nx + ix)     # row-major ordering
    grid  = values[order].reshape(ny, nx)

    return grid.copy()                   # robust for torch conversion


# ================================================================
# 4) Probabilistic helper functions ──────────────────────────────
# ================================================================
def get_py_given_x_distribution(x, neuralNet, V, globalSigma):
    """Return p(y | x) with x on the fine grid (65×65)."""
    mean = neuralNet.forward(x)
    Vexp = torch.pow(10, V)
    cov  = Vexp @ Vexp.T + torch.pow(10, globalSigma) * torch.eye(V.size(0), device=V.device)
    return dist.MultivariateNormal(mean, covariance_matrix=cov)


def get_py_given_X_distribution(x, neuralNet, V, globalSigma):
    """Return p(y | X) with X on the coarse grid (17×17)."""
    mean = neuralNet.forward_fromX(x)
    Vexp = torch.pow(10, V)
    cov  = Vexp @ Vexp.T + torch.pow(10, globalSigma) * torch.eye(V.size(0), device=V.device)
    return dist.MultivariateNormal(mean, covariance_matrix=cov)


def neg_log_likelihood(f, y):
    """
    Binary-cross-entropy in log-space for {-1, +1} labels.
    f : logits, y : labels.
    """
    f_flat = f.view(f.size(0), -1)
    y_flat = y.view(y.size(0), -1)
    return torch.sum(torch.logaddexp(torch.zeros_like(f_flat), -y_flat * f_flat))


def rbf_covariance_matrix(size: int, lengthscale: float = 0.2):
    """RBF kernel on a 17×17 lattice, returned as (size, size) tensor."""
    coords = torch.stack(
        torch.meshgrid(
            torch.linspace(0, 1, 17),
            torch.linspace(0, 1, 17),
            indexing="ij",
        ),
        dim=-1,
    ).reshape(-1, 2)  # (289, 2)

    dists = torch.cdist(coords, coords, p=2)
    return torch.exp(-dists ** 2 / (2 * lengthscale ** 2))


def make_uniform_mask(grid_h: int, grid_w: int, k: int, *, device: str = "cpu"):
    """
    Return a (1, grid_h*grid_w) mask with ~k ones laid out on a uniform lattice.
    `k` is rounded up to the nearest perfect square for easy tiling.
    """
    k_sq  = math.ceil(k ** 0.5) ** 2
    side  = int(math.sqrt(k_sq))

    if side == 1:
        rows = cols = torch.tensor([0], device=device)
    else:
        rows = torch.round(torch.linspace(0, grid_h - 1, side, device=device)).long()
        cols = torch.round(torch.linspace(0, grid_w - 1, side, device=device)).long()

    mask = torch.zeros(grid_h, grid_w, device=device, dtype=torch.float32)
    rr, cc = torch.meshgrid(rows, cols, indexing="ij")
    mask[rr, cc] = 1.0
    return mask.view(1, -1)


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
# Dimensions of the fine grid
W_grid = 65, 65  # Adjust based on your data

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



model = CNNModel().to(device)

# ================================================================
# Loss Function
# ================================================================


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

from model.ProbModels.cgCnnMvn import probabModel
from model.pde.pdeForm2D import pdeForm
from utils.PostProcessing import postProcessing
from input import *
from utils.variousFunctions import setupDevice

device = setupDevice(cudaIndex, device, dataType)

post = postProcessing(path='./results/data/', displayPlots=display_plots)
post.save([['numOfTestSamples', np.array(numOfTestSamples)]])
print("Saved number of test samples.")


### Definition and Form of the PDE ###
pde = pdeForm(nele, shapeFuncsDim, mean_px, sigma_px, sigma_r, Nx_samp, createNewCondField, device, post, rhs=rhs, reducedDim=reducedDim, options=options)


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

# Load observed y values from JSON

for i in range(100):
    with open("Y_samples.json", "r") as f:
        x_init_grid = fineGrids[-i].detach().cpu().numpy().reshape(65, 65) 
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


    # Initialize X as a 17x17 grid with ones
    X_init = torch.full((17, 17), 0.5, requires_grad=True)
    # # Draw one sample y from the distribution p(y | X)
    # y_draw = get_py_given_X_distribution(X_init, samples.neuralNet, samples.V, samples.globalSigma).sample()

    y_sample=pde.shapeFunc.cTrialSolutionParallel(y_star.to(device)).cpu()           # Reshape y to match the desired dimensions
    # y_sample = torch.reshape(y_sample, [y_sample.size(dim=0), pde.sgrid.size(dim=1), pde.sgrid.size(dim=2)]).detach().cpu()
    # Reshape and visualize the sampled y
    y_sample_reshaped = torch.reshape(y_sample, [1, pde.sgrid.size(dim=1), pde.sgrid.size(dim=2)]).detach().cpu()


    # Ensure computation is done on the correct device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Build 2D Coordinates for GP Prior ---
    x_coords = torch.linspace(0, 1, 17)
    y_coords = torch.linspace(0, 1, 17)
    X1, X2 = torch.meshgrid(x_coords, y_coords, indexing='ij')
    train_x = torch.stack([X1.reshape(-1), X2.reshape(-1)], dim=-1).to(device)

    latent_dim = 17 * 17
    train_y = torch.zeros(latent_dim, device=device)  # Initialize with zeros
    latent_dim = 17 * 17   # 289

    # 1) Build a GP-style covariance matrix Σ for the latent Gaussian

    # Precompute covariance matrix
    cov_matrix = rbf_covariance_matrix(latent_dim, lengthscale=0.1).to(device)
    cov_matrix += 1e-4 * torch.eye(latent_dim, device=device)  # ⬅️ THIS LINE ADDS STABILITY

    mean_vec   = torch.zeros(latent_dim, device=device)

    temperature = 3.0
    sigmoid_transform = transforms.SigmoidTransform()
    scale_transform   = transforms.AffineTransform(loc=0., scale=temperature)
    logistic_normal_transform = transforms.ComposeTransform([scale_transform, sigmoid_transform])

    # flatten
    y_flat = y_star.view(1, -1)
    grid_h, grid_w = 32, 32
    n_pixels = grid_h * grid_w
    # define the fractions you want
    fractions=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1]
    n_pixels = y_flat.size(1)
    masks = {}

    for f in fractions:
        label = f"{int(f*100)}%"
        # how many obs pixels we need
        k = int(round(f * n_pixels))
        mask = make_uniform_mask(grid_h, grid_w, k, device=device)
        masks[label] = mask



    # ---------------------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------------------
    num_svi_steps = 3000
    num_posterior = 50
    num_mc        = 50
    accuracy_results = {}

    # ---------------------------------------------------------------------
    # Loop over masking fractions
    # ---------------------------------------------------------------------
    for frac_label, mask in masks.items():
        print(f"\n=== Fraction {frac_label} ===")
        pyro.clear_param_store()

        # Import kept inside the loop (original behaviour)
        from pyro.infer.autoguide import AutoDiagonalNormal   # or AutoLowRankMVN

        # -----------------------------------------------------------------
        # Bayesian inverse model + guide + SVI
        # -----------------------------------------------------------------
        bim = BayesianInverseModel(
            samples.neuralNet,
            samples.V,
            samples.globalSigma,
        ).to(device)

        guide = AutoDiagonalNormal(bim.model)        # inverse transform for MVN → (0, 1)
        svi   = SVI(
            bim.model,
            guide,
            Adam({"lr": 1e-2}),
            loss=Trace_ELBO(),
        )

        # 1) Run SVI
        start_time = time.time()
        for step in range(1, num_svi_steps + 1):
            loss = svi.step(y_star, mask)
            if step == 1 or step % 500 == 0:
                print(f"[{frac_label}] step {step:4d} | ELBO = {loss:.1f}")

        # 2) Draw posterior Z → X
        pred = Predictive(
            bim.model,
            guide=guide,
            num_samples=num_posterior,
            parallel=False,
        )
        post = pred(y_star, mask)
        Xs   = post["X"].view(num_posterior, 1, 17, 17).to(device)   # (S, C, H, W)

        # 3) Monte-Carlo forward pass through CNN
        model.eval()
        x_mc: list[np.ndarray] = []
        with torch.no_grad():
            for _ in range(num_mc):
                idx   = torch.randint(0, num_posterior, (1,)).item()
                X_i   = Xs[idx : idx + 1].to(torch.float32)
                f     = model(X_i)
                p     = torch.sigmoid(f)
                b     = torch.bernoulli(p)
                x_s   = (2 * b - 1).squeeze().cpu().numpy()
                x_mc.append(x_s)

        x_mc = np.stack(x_mc, axis=0)

        # 4) Posterior stats
        x_mean = x_mc.mean(0)
        x_std  = x_mc.std(0)

        # 5) Ground-truth vs posterior-mean
        x_gt     = x_init_grid.copy()
        x_gt     = np.where(x_gt == 1, 1, 0.1)
        hard_map = np.where(x_mean > 0, 1, 0.1)

        end_time = time.time()
        acc      = (hard_map == x_gt).mean()
        accuracy_results[frac_label] = float(acc)

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        # Ground truth
        axs[0].imshow(
            x_gt.reshape(65, 65), cmap="bwr",
            vmin=0.1, vmax=1, origin="lower"
        )
        axs[0].set_title("Ground Truth")
        axs[0].axis("off")

        # Prediction
        axs[1].imshow(
            hard_map.reshape(65, 65), cmap="bwr",
            vmin=0.1, vmax=1, origin="lower"
        )
        axs[1].set_title(f"Prediction (Accuracy = {acc:.2f})")
        axs[1].axis("off")

        plt.tight_layout()
        plt.savefig(f"gt_vs_prediction_{frac_label}.png", dpi=150)

        # -----------------------------------------------------------------
        # Posterior / mask visualisation
        # -----------------------------------------------------------------
        mask_img = mask.squeeze(0).cpu().numpy().reshape(32, 32)

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        axs[0].imshow(x_gt, cmap="viridis", vmin=0.1, vmax=1, origin="lower")
        axs[0].set_title("Ground Truth x")
        axs[0].axis("off")

        im1 = axs[1].imshow(x_mean, cmap="viridis", origin="lower")
        axs[1].set_title(f"Posterior Mean x — {frac_label}")
        axs[1].axis("off")
        fig.colorbar(im1, ax=axs[1], fraction=0.046)

        axs[2].imshow(mask_img, cmap="gray", origin="lower")
        axs[2].set_title(f"Mask — {frac_label}")
        axs[2].axis("off")

        plt.tight_layout()
        os.makedirs(f"loop/images/binary/{frac_label}", exist_ok=True)
        plt.savefig(f"loop/images/binary/{frac_label}/{i}.png", dpi=150)
        plt.show()

    # ---------------------------------------------------------------------
    # Persist accuracy history
    # ---------------------------------------------------------------------
    import os, json, datetime as _dt

    json_path = "loop/accuracy_vs_partial_data.json"
    run_record = {
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "accuracy": accuracy_results,
    }

    if os.path.isfile(json_path):
        with open(json_path, "r") as f:
            prev = json.load(f)
            history = prev if isinstance(prev, list) else [prev]
    else:
        history = []

    history.append(run_record)

    with open(json_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"Appended partial-data accuracies to '{json_path}'.")
    print("Done all fractions.")

    # ---------------------------------------------------------------------
    # Solve Darcy + error metrics
    # ---------------------------------------------------------------------
    u_fem      = solve_darcy(hard_map)
    u_fem_grid = fem_solution_to_regular_grid(u_fem)   # (65, 65)

    target_grid = u_init_grid

    mse = np.mean((u_fem_grid - target_grid) ** 2)
    print(f"MSE(FEM solution ‖ surrogate target) = {mse:.4e}")

    δ      = u_fem_grid - target_grid
    rel_L2 = np.linalg.norm(δ) / np.linalg.norm(target_grid)
    print(f"Relative L2 error = {rel_L2:.4%}")

    # Side-by-side plot
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)

    im0 = ax[0].imshow(target_grid, origin="lower", cmap="viridis")
    ax[0].set_title("Surrogate target $y_{\\mathrm{sample}}$")
    ax[0].axis("off")
    plt.colorbar(im0, ax=ax[0], shrink=0.75)

    im1 = ax[1].imshow(u_fem_grid, origin="lower", cmap="viridis")
    ax[1].set_title("High-fidelity FEM $u_{\\mathrm{FEM}}$")
    ax[1].axis("off")
    plt.colorbar(im1, ax=ax[1], shrink=0.75)

    os.makedirs(f"loop/images/fem/{frac_label}", exist_ok=True)
    plt.suptitle(f"Darcy flow – MSE = {mse:.4e}", y=0.98)
    plt.savefig(f"loop/images/fem/{frac_label}/{i}.png", dpi=300)

    # ---------------------------------------------------------------------
    # Save metrics
    # ---------------------------------------------------------------------
    metrics_path = "loop/metrics_loop3.json"
    from datetime import datetime

    record = {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "mse":   float(mse),
        "rel_L2": float(rel_L2),
        "time":  float(end_time - start_time),
    }

    if os.path.isfile(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
            if not isinstance(data, list):
                data = [data]
    else:
        data = []

    data.append(record)

    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)

    print(f"Saved metrics to {metrics_path}")
