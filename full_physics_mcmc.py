"""
mcmc_block_darcy.py
-------------------
Metropolis–Hastings sampler for a binary Darcy‐flow permeability field
using *block* proposals (idea #5: “Bigger moves, fewer solves”).

Requires:  FEniCS 2019.1.x (or 2018.x), NumPy, Matplotlib
Run with:  $ python mcmc_block_darcy.py
"""
import json, time, random
import numpy as np
import matplotlib.pyplot as plt
from dolfin import *


# ------------------------------------------------------------
# 1.  Forward solver helpers (reuse mesh & solver for speed)
# ------------------------------------------------------------
def make_darcy_context(nx=65, ny=65):
    mesh = UnitSquareMesh(nx - 1, ny - 1)
    V    = FunctionSpace(mesh, "P", 1)

    # Variational forms
    u = TrialFunction(V); v = TestFunction(V)
    k_fun = Function(V, name="kappa")
    a_form = dot(k_fun * grad(u), grad(v)) * dx
    L_form = Constant(100.0) * v * dx

    # Pre-allocate linear algebra objects
    A = PETScMatrix(); b = PETScVector()
    bc = DirichletBC(V, Constant(0.0), "on_boundary")

    solver = PETScKrylovSolver("cg", "hypre_amg")
    solver.parameters.update(
        {"relative_tolerance": 1e-8,
         "absolute_tolerance": 1e-12,
         "maximum_iterations": 400})

    # Map DOF coordinates → array indices once
    coords = V.tabulate_dof_coordinates().reshape(-1, 2)
    idx_map = (np.minimum((coords[:, 0]*(nx-1)).astype(int), nx-1),
               np.minimum((coords[:, 1]*(ny-1)).astype(int), ny-1))

    return dict(V=V, k=k_fun, a=a_form, L=L_form,
                A=A, b=b, bc=bc, solver=solver, idx_map=idx_map)

CTX = make_darcy_context()                 # build once

def solve_darcy(k_field, ctx=CTX):
    """Solve −∇·(k∇u)=100 with Dirichlet u=0."""
    ix, iy = ctx["idx_map"]
    ctx["k"].vector()[:] = k_field[iy, ix]  # update coefficient

    assemble(ctx["a"], tensor=ctx["A"])
    assemble(ctx["L"], tensor=ctx["b"])
    ctx["bc"].apply(ctx["A"], ctx["b"])

    u = Function(ctx["V"])
    ctx["solver"].solve(ctx["A"], u.vector(), ctx["b"])
    return u

def fem_solution_to_vector(u):
    return u.vector().get_local()

# ------------------------------------------------------------
# 2.  Probability helpers
# ------------------------------------------------------------
def log_likelihood(y_hat, y_mod, sigma):
    diff = y_hat - y_mod
    return -0.5 * np.dot(diff, diff) / sigma**2

def log_prior(x_field, p=0.5):
    mask = (x_field == 1)
    return np.sum(mask*np.log(p) + (~mask)*np.log(1-p))

def log_posterior(x_field, y_hat, sigma, lambda_smooth=1.0):
    u = solve_darcy(x_field)
    return (log_likelihood(y_hat, fem_solution_to_vector(u), sigma)
            + log_prior(x_field, lambda_smooth))

# ------------------------------------------------------------
# 3.  Block-proposal Metropolis–Hastings
# ------------------------------------------------------------
def propose_block(current_x, block_size):
    """Flip every entry in one random block from 1→0.1 or vice-versa."""
    proposal = current_x.copy()
    nrow, ncol = current_x.shape
    i0 = random.randint(0, nrow - block_size)
    j0 = random.randint(0, ncol - block_size)
    block = proposal[i0:i0+block_size, j0:j0+block_size]
    block[:] = np.where(block == 1, 0.1, 1)    # in-place flip
    return proposal

# ------------------------------------------------------------
# 3.  Block‑proposal Metropolis–Hastings  ➜ now also tracks accuracy
# ------------------------------------------------------------
def mcmc_sampler_block(x_init, y_hat, sigma, *,
                       true_x, p=1, block_size=8,
                       n_samples=1000, burn_in=200):
    current_x = x_init.copy()
    current_logpost = log_posterior(current_x, y_hat, sigma, p)
    samples, n_acc = [], 0
    total_iter = burn_in + n_samples

    acc_history, time_history = [], []

    no_change_count = 0

    t_start = time.time()
    for it in range(total_iter):
        # ---------- propose / accept ----------
        prop_x = propose_block(current_x, block_size)
        prop_lp = log_posterior(prop_x, y_hat, sigma, p)

        # Ensure both log posteriors are scalars before comparison
        if np.log(random.random()) < np.sum(prop_lp - current_logpost):
            current_x, current_logpost = prop_x, prop_lp
            n_acc += 1
            no_change_count = 0
        else:
            no_change_count += 1

        # ---------- bookkeeping ----------
        acc_history.append(np.mean(current_x == true_x))
        time_history.append(time.time() - t_start)


        if it >= burn_in:
            samples.append(current_x.copy())

        # if (it + 1) % 50 == 0:
        #     print(f"iter {it+1:5d}/{total_iter}  acc={n_acc/(it+1):.3f}")

        if no_change_count >= 10_000:
            print(f"No accepted proposals in 10 000 iterations; breaking at {it+1}")
            break

    return samples, n_acc / (it + 1), acc_history, time_history



# ------------------------------------------------------------
# 4.  Main experiment
# ------------------------------------------------------------
if __name__ == "__main__":
    wall_start = time.time()

    # -- synthetic “truth” ----------------------------------------------------
    with open("data.json") as f:
        true_x = np.array(json.load(f)["x"][0])

    u_true = solve_darcy(true_x)
    y_true = fem_solution_to_vector(u_true)

    sigma_noise = 0.01 * true_x
    y_hat = y_true + np.random.normal(0, sigma_noise.flatten(), y_true.shape)

    x_init = np.ones_like(true_x) * 1.0

    print("\n▶  Running block-proposal MCMC …")
    samples, acc_rate, acc_hist, t_hist = mcmc_sampler_block(
        x_init, y_hat, sigma_noise,
        true_x=true_x,
        p=0.5, block_size=1,
        n_samples=50_000, burn_in=1_000)

    print(f"\nAcceptance rate      : {acc_rate:.3f}")
    print(f"Total wall-clock time: {time.time() - wall_start:.1f} s")

    # -- plot accuracy vs time ------------------------------------------------
    threshold = 0.88
    exceed_idx = next(i for i, a in enumerate(acc_hist) if a >= threshold)

    plt.figure(figsize=(7, 4))
    plt.plot(t_hist, acc_hist, lw=1)
    plt.axvline(t_hist[999],  color="gray", ls="--", label="end of burn-in")
    plt.axvline(t_hist[exceed_idx], color="red", ls="--",
                label="accuracy ≥ 88%")
    print(f"Elapsed time for 88% accuracy: {t_hist[exceed_idx]:.1f} s")

    plt.xlabel("Elapsed time (s)")
    plt.ylabel("Prediction accuracy")
    plt.ylim(0, 1)

    # Put 10 major ticks and light minor ticks on the x-axis <<<
    import matplotlib.ticker as mticker
    ax = plt.gca()
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=10, prune=None))
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())

    plt.legend()
    plt.tight_layout()

    plt.savefig("accuracy_over_time.png", dpi=300)
    plt.show()

    # -- quick visual check ---------------------------------------------------
    x_post = samples[-1]
    u_post = solve_darcy(x_post)

    plt.figure(figsize=(6, 5))
    plt.imshow(x_post, origin="lower", cmap="viridis")
    plt.title("Posterior Sample: Permeability Field")
    plt.colorbar(label="Permeability")
    plt.savefig("posterior_sample_permeability.png", dpi=300)
    plt.show()

    plt.figure(figsize=(6, 5))
    p = plot(u_post)
    plt.title("FEM Solution for Posterior Sample")
    plt.colorbar(p)
    plt.savefig("posterior_sample_solution.png", dpi=300)
    plt.show()

    print(f"final accuracy: {np.mean(x_post == true_x):.3f}")
