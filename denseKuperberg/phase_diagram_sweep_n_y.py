import numpy as np
import json
import os
import time
import argparse
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigs
from scipy.sparse.linalg import LinearOperator
from scipy.linalg import eigvals
from denseKuperberg.arnoldi import KuperbergArnoldiSolver

def fit_f_L(L_vals, lam_vals):
    f_L_vals = []
    for L, lam in zip(L_vals, lam_vals):
        f_L = -np.log(abs(lam)) / L
        f_L_vals.append(f_L)

    if len(L_vals) >= 3:
        L_arr = np.array(L_vals)
        y = np.array(f_L_vals)
        X = np.column_stack([np.ones_like(L_arr), 1.0/L_arr, 1.0/(L_arr**2)])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            A, B, C = coeffs
            return C, A, B
        except Exception:
            return None, None, None
    return None, None, None

def sweep():
    parser = argparse.ArgumentParser(description="Sweep x and n to build phase diagram")
    parser.add_argument('--x_start', type=float, default=1)
    parser.add_argument('--x_stop', type=float, default=2.5)
    parser.add_argument('--x_step', type=float, default=0.1)
    parser.add_argument('--n_start', type=float, default=0.2)
    parser.add_argument('--n_stop', type=float, default=1.4)
    parser.add_argument('--n_step', type=float, default=0.1)
    parser.add_argument('--L_max', type=int, default=6)
    args = parser.parse_args()

    L_list = [3, 4, 5, 6]
    # filter L_list based on L_max
    L_list = [L for L in L_list if L <= args.L_max]

    x_vals = np.arange(args.x_start, args.x_stop + args.x_step/2, args.x_step)
    n_vals = np.arange(args.n_start, args.n_stop + args.n_step/2, args.n_step)

    order = 'staggered'

    results = {}
    C_matrix = np.zeros((len(n_vals), len(x_vals)))

    out_dir = "experiment_outputs/denseKuperberg"
    os.makedirs(out_dir, exist_ok=True)

    print(f"Starting sweep: {len(n_vals)} n values, {len(x_vals)} x values.")

    for i, n in enumerate(n_vals):
        for j, x_val in enumerate(x_vals):
            lam_dict = {}
            for L in L_list:
                solver = KuperbergArnoldiSolver(L, 0, 0, 'T(x)', order, n, x_value=x_val)
                if solver.dim > 0:
                    if solver.dim <= 5:
                        H_matrix, _ = solver.arnoldi_iteration(solver.dim)
                        if H_matrix.shape[0] > 0:
                            evs = eigvals(H_matrix)
                            evs = [ev for ev in evs if abs(ev) > 1e-10]
                            if evs:
                                evs.sort(key=lambda v: abs(v), reverse=True)
                                lam_dict[L] = evs[0]
                    else:
                        k_actual = min(10, solver.dim - 2)
                        if k_actual > 0:
                            def matvec(v):
                                return solver.apply_T(v)
                            A = LinearOperator((solver.dim, solver.dim), matvec=matvec, dtype=complex)
                            try:
                                evs, _ = eigs(A, k=k_actual, which='LM')
                                evs = [ev for ev in evs if abs(ev) > 1e-10]
                                if evs:
                                    evs.sort(key=lambda v: abs(v), reverse=True)
                                    lam_dict[L] = evs[0]
                            except Exception:
                                pass

            # Fit
            L_fit = sorted(lam_dict.keys())
            lam_fit = [lam_dict[L] for L in L_fit]

            C, A, B = fit_f_L(L_fit, lam_fit)
            if C is not None:
                C_matrix[i, j] = C
                key = f"n={n:.2f}_x={x_val:.2f}"
                results[key] = {"C": C, "A": A, "B": B, "L": L_fit, "lam": [abs(l) for l in lam_fit]}

            print(f"n={n:.2f}, x={x_val:.2f} -> C={C:.4f}" if C is not None else f"n={n:.2f}, x={x_val:.2f} -> C=None")

    # Plot
    plt.figure(figsize=(8, 6))
    X_grid, N_grid = np.meshgrid(x_vals, n_vals)
    c_plot = plt.contourf(X_grid, N_grid, C_matrix, levels=20, cmap='viridis')
    plt.colorbar(c_plot, label='Coefficient C (proportional to central charge)')
    plt.title(f"Phase Diagram for T(x) | Order={order}")
    plt.xlabel("Boltzmann weight x")
    plt.ylabel("Loop weight n")

    #plt.savefig(os.path.join(out_dir, "phase_diagram_C.png"), dpi=150)
    #plt.close()
    filename = f"phase_diagram_C_x_{args.x_start:.2f}-{args.x_stop:.2f}_n_{args.n_start:.2f}-{args.n_stop:.2f}.png"
    plt.savefig(os.path.join(out_dir, filename), dpi=150)
    plt.close()

    with open(os.path.join(out_dir, "phase_diagram_data.json"), 'w') as f:
        json.dump(results, f, indent=2)

    print("Sweep complete. Data saved.")

if __name__ == "__main__":
    sweep()
