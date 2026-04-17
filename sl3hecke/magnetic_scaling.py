
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import json
import datetime
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from sl3_hecke import Sl3HeckeArnoldi

def fit_quadratic(L_values, f_values):
    def model(L, A, B, C):
        return A + B/L + C/(L**2)
    try:
        popt, pcov = curve_fit(model, L_values, f_values)
        return popt
    except Exception as e:
        print(f"Fit failed: {e}")
        return None

def analyze_magnetic_scaling(S_values=[3, 4, 5, 6], n_range=np.arange(0.8, 1.3, 0.2)):
    print("Magnetic Modules Scaling Analysis")
    print("=" * 60)

    out_dir = "experiment_outputs"
    os.makedirs(out_dir, exist_ok=True)

    results_all = {}
    results_sectors = {}

    # For distribution of leading eigenvalues of entire transfer matrix
    dist_data = []

    for n_val in n_range:
        n_val = round(n_val, 2)
        print(f"\n--- Analyzing n = {n_val} ---")

        # 1. Analyze entire matrix T_all for distribution
        for S in S_values:
            solver_all = Sl3HeckeArnoldi(L=None, n_value=n_val, is_magnetic=True, m=S, use_all_valid=True)
            if solver_all.dim == 0: continue

            k = min(50, solver_all.dim)
            h_mat = solver_all.arnoldi_iteration(k=k, operator='T', random_dense_start=True)
            eigenvalues = np.linalg.eigvals(h_mat)
            eigenvalues = sorted(eigenvalues, key=abs, reverse=True)

            # Record for distribution plot
            dist_data.append({"S": S, "n": n_val, "evals": [complex(e) for e in eigenvalues]})

        # Scaling for Entire Space
        f_L_all = []
        valid_S_all = []
        # Store block data for scaling
        block_data = {} # (x,y) -> {"S": [], "f_L": []}

        for S in S_values:
            # 2. Analyze specific block sectors
            solver_all_dummy2 = Sl3HeckeArnoldi(L=None, n_value=n_val, is_magnetic=True, m=S, use_all_valid=True)
            endpoints = set()
            for string in solver_all_dummy2.basis_strings:
                x = string.count(1) - string.count(0)
                y = string.count(0) - string.count(-1)
                endpoints.add((x, y))

            max_abs_lambda_entire = 0.0

            for (x, y) in endpoints:
                solver_block = Sl3HeckeArnoldi(L=None, n_value=n_val, is_magnetic=True, m=S, x=x, y=y)
                if solver_block.dim == 0: continue

                k = min(50, solver_block.dim)
                h_mat = solver_block.arnoldi_iteration(k=k, operator='T')
                evals = sorted(np.linalg.eigvals(h_mat), key=abs, reverse=True)

                if evals and abs(evals[0]) > 1e-10:
                    Lambda = evals[0]
                    abs_lambda = abs(Lambda)
                    f_L = np.real(np.log(abs_lambda)) / S

                    if abs_lambda > max_abs_lambda_entire:
                        max_abs_lambda_entire = abs_lambda

                    if (x, y) not in block_data:
                        block_data[(x, y)] = {"S": [], "f_L": []}
                    block_data[(x, y)]["S"].append(S)
                    block_data[(x, y)]["f_L"].append(f_L)

            # The leading eigenvalue of the entire space is the maximum over all blocks
            if max_abs_lambda_entire > 1e-10:
                f_L_entire = np.real(np.log(max_abs_lambda_entire)) / S
                f_L_all.append(f_L_entire)
                valid_S_all.append(S)
            else:
                print(f"    S={S} entirely skipped (Lambda zero or missing across blocks)")

        # Fit Entire Space
        if len(valid_S_all) >= 3:
            popt = fit_quadratic(np.array(valid_S_all), np.array(f_L_all))
            if popt is not None:
                A, B, C = popt
                results_all[n_val] = {"f_inf": A, "f_sur": B/2.0, "vFc": -24*C/np.pi, "S": valid_S_all, "f_L": f_L_all}
                print(f"    Entire space fit: v_F*c = {-24*C/np.pi:.4f}")

        # Fit Block Sectors
        results_sectors[n_val] = {}
        for (x, y), data in block_data.items():
            valid_S = data["S"]
            valid_f_L = data["f_L"]

            sector_key = f"{x},{y}"
            if len(valid_S) >= 3:
                popt = fit_quadratic(np.array(valid_S), np.array(valid_f_L))
                if popt is not None:
                    A, B, C = popt
                    results_sectors[n_val][sector_key] = {
                        "f_inf": A, "f_sur": B/2.0, "vFc": -24*C/np.pi,
                        "S": valid_S, "f_L": valid_f_L
                    }
                    print(f"    Block sector ({x},{y}) fit: v_F*c = {-24*C/np.pi:.4f}")
            else:
                # Just log the raw data if not enough points to fit
                results_sectors[n_val][sector_key] = {
                    "f_inf": None, "f_sur": None, "vFc": None,
                    "S": valid_S, "f_L": valid_f_L
                }

    # Plot Distribution of leading eigenvalues for entire matrix (for max S, n=1.0)
    target_dist = [d for d in dist_data if d["S"] == max(S_values) and d["n"] == 1.0]
    if target_dist:
        evals = target_dist[0]["evals"]
        plt.figure(figsize=(8, 8))
        plt.scatter(np.real(evals), np.imag(evals), marker='.', alpha=0.7)
        plt.xlabel(r'Re($\lambda$)')
        plt.ylabel(r'Im($\lambda$)')
        plt.title(f'Eigenvalue Distribution Entire T (S={max(S_values)}, n=1.0)')
        plt.grid(True)
        plt.axis('equal')
        plt.savefig(os.path.join(out_dir, f'magnetic_dist_S{max(S_values)}_n1.png'))

    # Log results
    log_output = {
        "entire_space": results_all,
        "block_sectors": results_sectors
    }
    with open(os.path.join(out_dir, 'magnetic_scaling_log.json'), 'w') as f:
        json.dump(log_output, f, indent=4)

    print(f"Analysis complete. Results saved to {out_dir}/")

if __name__ == "__main__":
    # We need a wider range of S values to ensure a specific sector appears at least 3 times.
    # Since S = 3N1 + 2x + y mod 3 -> S = 2x + y mod 3
    # A sector (x,y) appears when S = 2x + y + 3k.
    # To get 3 points, we need a span of 9. Let's use S_values=[3, 4, 5, 6, 7, 8, 9]
    # Computation might take longer but necessary for block scaling.
    analyze_magnetic_scaling(S_values=[3, 4, 5, 6, 7, 8, 9])
