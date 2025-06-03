import numpy as np
from autograd import grad
from autograd import numpy as anp
from scipy.optimize import minimize
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

selectionIndex=10000# After reading data
fineGrids = data['x'][:selectionIndex]   # Assume shape (N, H_fg, W_fg)
coarseGrids = data['XCG'][:selectionIndex]  # Assume shape (N, H_cg, W_cg)
N = len(fineGrids)

fineGrids = np.array(fineGrids)       # shape (N, H_fg, W_fg)
coarseGrids = np.array(coarseGrids)   # shape (N, H_cg, W_cg)

# Convert fineGrids from {0.1, something else} to {-1, 1}
x_data = np.where(fineGrids == 0.1, -1, 1)

# Flatten the grids
X_data = coarseGrids.reshape(N, -1)  # Now shape is (N, H_cg*W_cg)
x_data = x_data.reshape(N, -1)       # Now shape is (N, H_fg*W_fg)

# Split the data into 90% training and 10% testing
split_index = int(0.9 * N)
X_train, X_test = X_data[:split_index], X_data[split_index:]
x_train, x_test = x_data[:split_index], x_data[split_index:]

# Now we can define m and d from the shapes
m = X_train.shape[1]
d = x_train.shape[1]

# Redefine W and b initialization and proceed
W_true = np.random.randn(d, m)
b_true = np.random.randn(d)

# The rest of the code remains the same
f_true = X_train.dot(W_true.T) + b_true[None, :]
# Assign x_data based on sign of f_true

# Parameter vector: Flatten W and b into a single vector:
# length of params = d*m + d
def unpack_params(params):
    W = params[:d*m].reshape(d, m)
    b = params[d*m:]
    return W, b

# Define the negative log-likelihood
def neg_log_likelihood(params, X, x):
    W, b = unpack_params(params)
    # Compute f(X) = W X + b
    # f will have shape (N, d)
    f = anp.dot(X, W.T) + b[None, :]
    # Compute the elementwise NLL contribution: log(1 + exp(-x_ij * f_ij))
    # x and f have shape (N, d)
    # We can vectorize: nll = sum over i,j of log(1 + exp(-x[i,j]*f[i,j]))
    exponent = -x * f
    # Use a stable computation: log(1+exp(z)) = softplus(z)
    nll_terms = anp.logaddexp(0, exponent)
    return anp.sum(nll_terms)

# Initialize parameters randomly
params_init = np.random.randn(d*m + d)*0.01

# Compute gradient using autograd
nll_grad = grad(neg_log_likelihood)

# Start the timer
start_time = time.time()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Callback function to log progress
def callback(params):
    current_nll = neg_log_likelihood(params, X_train, x_train)
    print(f'Current NLL: {current_nll}')

# Optimize using scipy
res = minimize(fun=neg_log_likelihood, x0=params_init, 
               args=(X_train, x_train), 
               jac=nll_grad, 
               method='L-BFGS-B')

# Stop the timer
end_time = time.time()

params_opt = res.x
print("Optimization success:", res.success)
print("Final NLL:", res.fun)
print("Time taken for optimization:", end_time - start_time, "seconds")

# Compute training error estimate
def compute_error(params, X, x):
    W, b = unpack_params(params)
    f = X.dot(W.T) + b[None, :]
    preds = np.sign(f)
    accuracy = np.mean(preds == x)
    # Negative log-likelihood on training data
    nll = neg_log_likelihood(params, X, x)
    return accuracy, nll

accuracy, nll = compute_error(params_opt, X_train, x_train)
print("Training accuracy:", accuracy)
print("Training NLL:", nll)

# After training, we have params_opt
W_opt = params_opt[:d*m].reshape(d, m)
b_opt = params_opt[d*m:]

# Suppose we know the original grid shape of the fine grid
H, W = 65, 65  # Adjust based on your actual dimensions
assert H*W == d, "The product of H and W should equal d."

# Plot all 10 testing sets

def sum_of_squares_error(params, X, x):
        W, b = unpack_params(params)
        f = X.dot(W.T) + b[None, :]
        x_pred = np.sign(f)
        error = np.sum((x - x_pred) ** 2)/len(x)
        return error

# Compute sum of squares error for the testing data
sse = sum_of_squares_error(params_opt, X_test, x_test)
print("Sum of Squares Error on testing data:", sse)

for n in range(100):
    # Compute predicted f and x_pred
    testCoarseGrid_flat = X_test[n].reshape(1, -1)
    f_pred = testCoarseGrid_flat.dot(W_opt.T) + b_opt
    x_pred = np.sign(f_pred)

    x_actual = x_test[n].reshape(1, -1)

    # Reshape into the original grid form
    x_pred_grid = x_pred.reshape(H, W)
    x_actual_grid = x_actual.reshape(H, W)

    x_diff = x_pred_grid - x_actual_grid

    # Now we can plot them side-by-side
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    im0 = axes[0].imshow(x_actual_grid, cmap='bwr', vmin=-1, vmax=1)
    axes[0].set_title('Actual x')
    axes[0].axis('off')
    im1 = axes[1].imshow(x_pred_grid, cmap='bwr', vmin=-1, vmax=1)
    axes[1].set_title('Predicted x')
    axes[1].axis('off')
    im2 = axes[2].imshow(x_diff, cmap='bwr', vmin=-1, vmax=1)
    axes[2].set_title('Difference')
    axes[2].axis('off')

    plt.colorbar(im1, ax=axes, fraction=0.046, pad=0.04)
    plt.suptitle(f'Actual vs Predicted x for sample #{n}')
    plt.tight_layout()

    # Save the figure to a file
    plt.savefig(f'/plotsBA/actual_vs_predicted_sample_{n}.png', dpi=300)

    plt.show()
    # Function to compute sum of squares error

    print("done")