
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

for i in range(100):
    with open("Y_samples.json", "r") as f:
        # # y_samples1 = torch.tensor(json.load(f)["ys"]).reshape(1, 1,32*32)
        # # print("y_samples1:",y_samples1.shape)
        # sampSol, sampSolFenics, sampCond, sampX, sampYCoeff = pde.produceTestSample(Nx=numOfTestSamples, post=post)
        # sampX=sampX[:1].repeat(numOfTestSamples, 1)
        # x_testing_samples = sampX[:1].repeat(numOfTestSamples, 1)
        # xKLE = x_testing_samples[:1].view(1, 1, 24) #original small x
        # x = pde.gpExpansionExponentialParallel(xKLE) #add noise (sigma + random draw from standard normal distribution)
        # # Reshape xKLE and repeat for the specified number of test samples
        # xKLE = torch.reshape(xKLE, [xKLE.size(dim=0), 1, -1]) \
        #     .repeat(numOfTestSamples, 1, 1) \
        #     .to('cuda:0')

        # # Remove unnecessary dimensions from x
        # x = x.squeeze(1).to(device)

        # # Generate XCG using the neural network's xtoXCnn function
        # XCG = samples.neuralNet.xtoXCnn(xKLE).squeeze(1)
        # XCG = XCG[:1]

        # x_init_grid = x[0].detach().cpu().numpy().reshape(65, 65)   # adjust reshape if needed

        # # 2) Map {-1,+1} flags back to physical permeabilities {0.1, 1.0}
        # if x_init_grid.max() <= 1.0 and x_init_grid.min() < 0.0:
        #     k_init = np.where(x_init_grid > 0, 1.0, 0.1)
        # else:                                   # already in physical units
        #     k_init = x_init_grid.copy()

        # # 3) High-fidelity Darcy solve
        # u_init      = solve_darcy(k_init)
        # u_init_grid = fem_solution_to_regular_grid(u_init)          # shape (65, 65)

        # # 4) Convert the pressure field to a torch tensor
        # rbfRefSol = torch.tensor(u_init_grid, dtype=torch.float32, device=device)

        # # 5) Get RBF coefficients  →  this *is* the new y★
        # #    findRbfCoeffs expects (batch, H, W); add batch dim if needed
        # rbf_coeffs = pde.shapeFunc.findRbfCoeffs(rbfRefSol.unsqueeze(0)).squeeze(0)

        # # 6) Over-write y_star for the rest of the pipeline
        # y_star = rbf_coeffs.detach()       
        #      # keep on the same device as before
        
        # print("y_star shape:", y_star.shape)     # keep on the same device as before
        # y_samples = samples.neuralNet.forward(xKLE.to(device)).detach().cpu()
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

        # Apply the trial solution transformation
        # y = pde.shapeFunc.cTrialSolutionParallel(y.to(device)).cpu()

        # # Reshape y to match the desired dimensions
        # y = torch.reshape(y, [y.size(dim=0), pde.sgrid.size(dim=1), pde.sgrid.size(dim=2)]).detach().cpu()

        # yys, yyMean= samples.samplePosteriorMvn(sampX, Nx=numOfTestSamples, Navg=Navg)

        # yvalue= pde.shapeFunc.cTrialSolutionParallel(yys[0].to(device)).cpu()
        
        # Draw 100 ys for the current XCG
        # print("y", yys.shape)

        # y_samples = torch.tensor(samples.samplePosteriorMvn(x_test))

        # y = pde.shapeFunc.cTrialSolutionParallel(y_samples).reshape(1, 1, 32 * 32)# Reshape y to match the desired dimensions
        # y = torch.reshape(y, [y.size(dim=0), pde.sgrid.size(dim=1), pde.sgrid.size(dim=2)]).detach().cpu()    
        # plt.imshow(y_samples[0], cmap='viridis')
        # plt.colorbar()
        # plt.title("Observed y*")
        # plt.savefig("observed_y.png")
        # y_star=y_samples[0].to(device)
        
        # print("kek shape:",y_samples.shape)  # Shape: (1000, 32, 32)
        # print("y_star shape:", y_star.shape)  # Shape: (1000, 32, 32)

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

    import math
    import torch
    import math

    import math

    def make_uniform_mask(grid_h, grid_w, k, device="cpu"):

        # round up to next perfect square
        k_sq = math.ceil(k**0.5)**2
        side = int(math.sqrt(k_sq))

        # evenly spaced positions in [0, grid_h-1]
        if side == 1:
            rows = cols = torch.tensor([0], device=device)
        else:
            rows = torch.round(torch.linspace(0, grid_h-1, steps=side, device=device)).long()
            cols = torch.round(torch.linspace(0, grid_w-1, steps=side, device=device)).long()

        mask = torch.zeros(grid_h, grid_w, device=device, dtype=torch.float32)

        # fill all k_sq = side*side points
        rr, cc = torch.meshgrid(rows, cols, indexing='ij')
        mask[rr, cc] = 1.0

        return mask.view(1, -1)





    # --- Optionally, check GP prior statistics ---
    with torch.no_grad():
        gp_out = gp_prior(train_x)
        print("GP Prior Mean (first 5):", gp_out.mean[:5])
        print("GP Prior Variance (first 5):", gp_out.covariance_matrix.diag()[:5])
    print("y_star shape:", y_star.shape)
    # --- Bayesian Inverse Model with Sigmoid Transformation and Penalty ---

    import torch
    import pyro
    import pyro.distributions as dist
    from pyro.infer import SVI, Trace_ELBO, Predictive
    from pyro.infer.autoguide import AutoLowRankMultivariateNormal
    from torch.distributions import constraints

    # -----------------------------------------------------------------------------
    # (A) Helper to build a random mask of flat‐indices
    # -----------------------------------------------------------------------------
    import torch
    import pyro
    import pyro.distributions as dist

    # … keep all your previous imports and code up to the definition of BayesianInverseModel …

    # class BayesianInverseModel(PyroModule):
    #     def __init__(self, neuralNet, V, globalSigma, temperature=3.0, lambda_reg=10.0):
    #         super().__init__()
    #         self.neuralNet   = neuralNet
    #         self.V           = V
    #         self.globalSigma = globalSigma
    #         self.temperature = temperature
    #         self.lambda_reg  = lambda_reg

    #     def model(self, y_star, mask):
    #         # 1) GP prior on Z
    #         gp_dist = gp_prior(train_x)
    #         mean_z  = gp_dist.mean
    #         cov_z   = gp_dist.covariance_matrix + 1e-4 * torch.eye(len(mean_z), device=device)
    #         Z       = pyro.sample("Z", dist.MultivariateNormal(mean_z, covariance_matrix=cov_z))

    #         # 2) sigmoid→X and penalty
    #         X       = torch.sigmoid(self.temperature * Z).view(-1, 17, 17)
    #         pyro.factor("bimodal_penalty",
    #                     -self.lambda_reg * torch.sum(X * (1 - X)))

    #         # 3) full Gaussian y|X
    #         p_y     = get_py_given_X_distribution(X, self.neuralNet, self.V, self.globalSigma)

    #         # 4) flatten mean & variance to 1-D of length 1024
    #         mean_flat = p_y.mean.view(-1)  # torch.Size([1024])
    #         var_flat  = p_y.covariance_matrix.diagonal(dim1=-2, dim2=-1).view(-1)  # torch.Size([1024])

    #         # 5) flatten observations and mask to 1-D
    #         y_flat    = y_star.view(-1)            # torch.Size([1024])
    #         mask_flat = mask.view(-1).bool()       # torch.Size([1024])

    #         # 6) index only the observed entries
    #         mean_obs = mean_flat[mask_flat]                # torch.Size([#obs])
    #         std_obs  = torch.sqrt(var_flat[mask_flat] + 1e-6)  # add small eps for stability
    #         y_obs    = y_flat[mask_flat]                   # torch.Size([#obs])

    #         # 7) independent Normal likelihood on observed pixels
    #         pyro.sample("y_obs",
    #                     dist.Normal(mean_obs, std_obs).to_event(1),
    #                     obs=y_obs)



    #     def guide(self, y_star, mask):
    #         # automatic guide will register Z
    #         pass
    # ----------------------------------------------------------------------
# ❶  Replace EVERYTHING from “# --- Build 2D Coordinates for GP Prior ---”
#     down to (and including) the first two lines inside BayesianInverseModel.model
#     with the block below.  Nothing else in your script needs to change.
# ----------------------------------------------------------------------
# ───────────────────────────────────────────────────────────────────────
# ⮕  DROP-IN PATCH: correlated logistic-normal prior
#     Replaces: old BayesianInverseModel and GP prior construction
# ───────────────────────────────────────────────────────────────────────
    import pyro
    import pyro.distributions as dist
    from pyro.nn import PyroModule
    from torch.distributions import constraints, transforms

    latent_dim = 17 * 17   # 289

    # 1) Build a GP-style covariance matrix Σ for the latent Gaussian
    def rbf_covariance_matrix(size, lengthscale=0.2):
        coords = torch.stack(torch.meshgrid(
            torch.linspace(0, 1, 17),
            torch.linspace(0, 1, 17),
            indexing='ij'
        ), dim=-1).reshape(-1, 2)  # (289, 2)

        dists = torch.cdist(coords, coords, p=2)
        cov = torch.exp(-dists ** 2 / (2 * lengthscale ** 2))
        return cov

    # Precompute covariance matrix
    cov_matrix = rbf_covariance_matrix(latent_dim, lengthscale=0.1).to(device)
    cov_matrix += 1e-4 * torch.eye(latent_dim, device=device)  # ⬅️ THIS LINE ADDS STABILITY

    mean_vec   = torch.zeros(latent_dim, device=device)

    # Define transform: sigmoid(temperature * Z)
    temperature = 3.0
    sigmoid_transform = transforms.SigmoidTransform()
    scale_transform   = transforms.AffineTransform(loc=0., scale=temperature)
    logistic_normal_transform = transforms.ComposeTransform([scale_transform, sigmoid_transform])

    # Define the Bayesian model
    class BayesianInverseModel(PyroModule):
        def __init__(self, neuralNet, V, globalSigma):
            super().__init__()
            self.neuralNet   = neuralNet
            self.V           = V
            self.globalSigma = globalSigma

        def model(self, y_star, mask):
            # 1) Z ~ MVN(0, Σ) and transform → X = sigmoid(τ Z)
            base_dist = dist.MultivariateNormal(mean_vec, covariance_matrix=cov_matrix)
            X = pyro.sample(
                "X",
                dist.TransformedDistribution(base_dist, logistic_normal_transform)
            ).view(-1, 17, 17)

            # 2) Physics-informed likelihood
            p_y = get_py_given_X_distribution(X, self.neuralNet, self.V, self.globalSigma)

            # 3) Only compute on observed RBF coefficients
            mean_flat = p_y.mean.view(-1)
            var_flat  = p_y.covariance_matrix.diagonal(dim1=-2, dim2=-1).view(-1)

            y_flat    = y_star.view(-1)
            mask_flat = mask.view(-1).bool()

            mean_obs = mean_flat[mask_flat]
            std_obs  = torch.sqrt(var_flat[mask_flat] + 1e-6)
            y_obs    = y_flat[mask_flat]

            pyro.sample("y_obs",
                        dist.Normal(mean_obs, std_obs).to_event(1),
                        obs=y_obs)

        def guide(self, y_star, mask):
            pass  # handled by auto guide



    # ────────────────────────────────────────────────
    # just after you load y_star (shape [1,1024]):

    # flatten
    y_flat = y_star.view(1, -1)
    grid_h, grid_w = 32, 32
    n_pixels = grid_h * grid_w
    # define the fractions you want
    # fractions = [1, 0.75, 0.50, 0.25, 0.10, 0.05, 0.01] 
    fractions=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1]
    # fractions =[0.01 ,0.1, 0.5, 1]
    # fractions =[1]
    n_pixels = y_flat.size(1)
    masks = {}

    for f in fractions:
        label = f"{int(f*100)}%"
        # how many obs pixels we need
        k = int(round(f * n_pixels))
        mask = make_uniform_mask(grid_h, grid_w, k, device=device)
        masks[label] = mask




    # now you can loop over all your masks:
    num_svi_steps = 3000
    num_posterior = 50
    num_mc = 50
    accuracy_results = {}

    for frac_label, mask in masks.items():

        print(f"\n=== Fraction {frac_label} ===")
        pyro.clear_param_store()


        # Rebuild B-I model + guide + SVI
        # bim = BayesianInverseModel(
        #     samples.neuralNet, samples.V, samples.globalSigma,
        #     temperature=10.0, lambda_reg=10.0
        # ).to(device)
        # guide = AutoLowRankMultivariateNormal(bim.model, rank=40)
        # svi = SVI(bim.model, guide, Adam({"lr": 1e-2}), loss=Trace_ELBO())
        from pyro.infer.autoguide import AutoDiagonalNormal   # or AutoLowRankMVN

        # ───────────────────────────────────────────────────────────────────────
        # ⮕  SVI setup (replace your old AutoLowRank → SVI block with this)
        # ───────────────────────────────────────────────────────────────────────
        from pyro.infer.autoguide import AutoDiagonalNormal  # works with transforms

        bim = BayesianInverseModel(samples.neuralNet,
                                samples.V,
                                samples.globalSigma).to(device)

        guide = AutoDiagonalNormal(bim.model)  # applies inverse transform for MVN→(0,1)
        svi   = SVI(bim.model, guide,
                    Adam({"lr": 1e-2}),
                    loss=Trace_ELBO())

        # ───────────────────────────────────────────────────────────────────────
        # Now leave the rest of your SVI loops, Predictive, plotting, etc. unchanged.



        # 1) run SVI
        start_time = time.time()

        for step in range(1, num_svi_steps + 1):
            loss = svi.step(y_star, mask)
            if step % 500 == 0 or step == 1:
                print(f"[{frac_label}] step {step} ELBO={loss:.1f}")

        # 2) draw posterior Z→X
        pred = Predictive(bim.model, guide=guide,
                        num_samples=num_posterior, parallel=False)
        post = pred(y_star, mask)
        # Zs = post["Z"]                                    # (num_posterior, D)
        # Xs = torch.sigmoid(bim.temperature * Zs)   
               # (num_posterior, D)
        Xs = post["X"]                                    # (num_posterior, D)
        Xs = Xs.view(num_posterior, 1, 17, 17).to(device)

        # 3) MC forward through CNN
        model.eval()
        x_mc = []
        with torch.no_grad():
            for _ in range(num_mc):
                idx = torch.randint(0, num_posterior, (1,)).item()
                X_i = Xs[idx:idx+1].to(torch.float32)
                f   = model(X_i)
                p   = torch.sigmoid(f)
                b   = torch.bernoulli(p)
                x_s = (2*b - 1).squeeze().cpu().numpy()
                x_mc.append(x_s)
        x_mc = np.stack(x_mc, axis=0)

        # 4) posterior mean & std
        x_mean = x_mc.mean(0)
        x_std  = x_mc.std(0)

        # 5) ground truth vs posterior-mean plot
        x_gt = x_init_grid.copy()
        x_gt = np.where(x_gt == 1, 1, 0.1)

        # Create binary prediction from posterior mean
        hard_map = np.where(x_mean > 0, 1, 0.1)
        end_time = time.time()
        acc = (hard_map == x_gt).mean()
        accuracy_results[frac_label] = float(acc)

        # Plot ground truth vs binary prediction
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        # Ground truth visualization
        axs[0].imshow(x_gt.reshape(65, 65), cmap='bwr', vmin=0.1, vmax=1, origin='lower')
        axs[0].set_title("Ground Truth")
        axs[0].axis('off')

        # Binary prediction from posterior mean
        axs[1].imshow(hard_map.reshape(65, 65), cmap='bwr', vmin=0.1, vmax=1, origin='lower')
        axs[1].set_title(f"Prediction (Accuracy: {acc:.2f})")
        axs[1].axis('off')

        plt.tight_layout()
        plt.savefig(f"gt_vs_prediction_{frac_label}.png", dpi=150)


        # right after you’ve computed `mask` in the loop:

        # turn your 1×289 mask into a plain 17×17 array
        mask_img = mask.squeeze(0)                  # → shape (289,)
        mask_img = mask_img.cpu().numpy().reshape(32, 32)

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        # Ground truth
        axs[0].imshow(x_gt, cmap='viridis', vmin=0.1, vmax=1, origin='lower')
        axs[0].set_title("Ground Truth x")
        axs[0].axis('off')

        # Posterior mean
        im1 = axs[1].imshow(x_mean, cmap='viridis', origin='lower')
        axs[1].set_title(f"Posterior Mean x — {frac_label}")
        axs[1].axis('off')
        fig.colorbar(im1, ax=axs[1], fraction=0.046)

        # Mask visualization
        axs[2].imshow(mask_img, cmap='gray', origin='lower')
        axs[2].set_title(f"Mask — {frac_label}")
        axs[2].axis('off')

        plt.tight_layout()
        os.makedirs(f"experiments/images/binary/{frac_label}", exist_ok=True)

        plt.savefig(f"experiments/images/binary/{frac_label}/{i}.png", dpi=150)
        plt.show()

    import os, json, datetime as _dt

    # --- (1) suppose you filled this inside the loop) ---
    # accuracy_results = {"1%": 0.42, "5%": 0.55, …}
    # You should have built accuracy_results[frac_label] = computed_accuracy

    # --- (2) now append to JSON history file ---
    json_path = "experiments/accuracy_vs_partial_data.json"
    run_record = {
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "accuracy": accuracy_results
    }

    # load existing history (if any)
    if os.path.isfile(json_path):
        with open(json_path, "r") as f:
            prev = json.load(f)
            # make sure it’s a list
            history = prev if isinstance(prev, list) else [prev]
    else:
        history = []

    history.append(run_record)

    # write back
    with open(json_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"Appended partial‐data accuracies to '{json_path}'.")

    print("Done all fractions.")


    u_fem      = solve_darcy(hard_map)
    u_fem_grid = fem_solution_to_regular_grid(u_fem)   # (65, 65)

    target_grid = u_init_grid

    mse = np.mean((u_fem_grid - target_grid) ** 2)
    print(f"MSE(FEM solution ‖ surrogate target) = {mse:.4e}")
    # ------------------------------------------------------------------
    # Relative L2 error ‖e‖₂ / ‖u_target‖₂
    # ------------------------------------------------------------------
    δ       = u_fem_grid - target_grid
    rel_L2  = np.linalg.norm(δ) / np.linalg.norm(target_grid)
    print(f"Relative L2 error = {rel_L2:.4%}")


    # ---- side-by-side plot --------------------------------------------
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)

    im0 = ax[0].imshow(target_grid, origin="lower", cmap="viridis")
    ax[0].set_title("Surrogate target $y_{\\mathrm{sample}}$")
    ax[0].axis("off"); plt.colorbar(im0, ax=ax[0], shrink=0.75)

    im1 = ax[1].imshow(u_fem_grid, origin="lower", cmap="viridis")
    ax[1].set_title("High-fidelity FEM $u_{\\mathrm{FEM}}$")
    ax[1].axis("off"); plt.colorbar(im1, ax=ax[1], shrink=0.75)

    # Create directory for this fraction if it doesn't exist
    os.makedirs(f"experiments/images/fem/{frac_label}", exist_ok=True)

    # Add super title with MSE information
    plt.suptitle(f"Darcy flow – MSE = {mse:.4e}", y=0.98)
    plt.savefig(f"experiments/images/fem/{frac_label}/{i}.png", dpi=300) 

    metrics_path = "experiments/metrics_loop3.json"
    from datetime import datetime
    # build one record (add a timestamp so you can tell runs apart)
    record = {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "mse": float(mse),         # make sure numpy scalars are cast to Python floats
        "rel_L2": float(rel_L2),
        "time": float(end_time - start_time),
    }

    # read-modify-write workflow
    if os.path.isfile(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
            if not isinstance(data, list):          # safety guard
                data = [data]
    else:
        data = []

    data.append(record)                              # append this run

    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)

    print(f"Saved metrics to {metrics_path}")