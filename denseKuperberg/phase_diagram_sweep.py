import numpy as np
import json
import os
import argparse
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigs
from scipy.sparse.linalg import LinearOperator
from scipy.linalg import eigvals
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from denseKuperberg.arnoldi import KuperbergArnoldiSolver

def fit_f_L(L_vals, lam_vals, operator='T'):
    f_L_vals = []
    for L, lam in zip(L_vals, lam_vals):
        f_L = -np.log(abs(lam)) / L if operator == 'T' else -lam.real / L
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_val', type=float, default=1.0)
    parser.add_argument('--z_val', type=float, default=1.0)
    parser.add_argument('--x_start', type=float, default=0.1)
    parser.add_argument('--x_stop', type=float, default=2.0)
    parser.add_argument('--x_step', type=float, default=0.2)
    parser.add_argument('--y_start', type=float, default=0.1)
    parser.add_argument('--y_stop', type=float, default=2.0)
    parser.add_argument('--y_step', type=float, default=0.2)
    parser.add_argument('--L_max', type=int, default=6)
    parser.add_argument('--operator', type=str, default='T')
    args = parser.parse_args()

    L_list = [3, 4, 5, 6]
    L_list = [L for L in L_list if L <= args.L_max]

    x_vals = np.arange(args.x_start, args.x_stop + args.x_step/2, args.x_step)
    y_vals = np.arange(args.y_start, args.y_stop + args.y_step/2, args.y_step)
    n = args.n_val

    order = 'staggered'
    types = ['T(x,y,z)']

    out_dir = "experiment_outputs/denseKuperberg"
    os.makedirs(out_dir, exist_ok=True)

    for type_str in types:
        results = {}
        C_matrix = np.zeros((len(y_vals), len(x_vals)))

        print(f"\nStarting sweep for {type_str}: n={n:.2f}, z={args.z_val:.2f}, {len(x_vals)} x vals, {len(y_vals)} y vals, operator={args.operator}.")

        for j, y_val in enumerate(y_vals):
            for i, x_val in enumerate(x_vals):
                lam_dict = {}
                for L in L_list:
                    solver = KuperbergArnoldiSolver(L, 0, 0, type_str, order, n, x_value=x_val, y_value=y_val, z_value=args.z_val, operator=args.operator)
                    if solver.dim > 0:
                        if solver.dim <= 5:
                            H_matrix, _ = solver.arnoldi_iteration(solver.dim)
                            if H_matrix.shape[0] > 0:
                                evs = eigvals(H_matrix)
                                evs = [ev for ev in evs if abs(ev) > 1e-10]
                                if evs:
                                    if args.operator == 'H':
                                        evs.sort(key=lambda v: v.real, reverse=True)
                                    else:
                                        evs.sort(key=lambda v: abs(v), reverse=True)
                                    lam_dict[L] = evs[0]
                        else:
                            k_actual = min(10, solver.dim - 2)
                            if k_actual > 0:
                                def matvec(v):
                                    if args.operator == 'H':
                                        return solver.apply_H(v)
                                    else:
                                        return solver.apply_T(v)
                                A = LinearOperator((solver.dim, solver.dim), matvec=matvec, dtype=complex)
                                try:
                                    evs, _ = eigs(A, k=k_actual, which='LM')
                                    evs = [ev for ev in evs if abs(ev) > 1e-10]
                                    if evs:
                                        if args.operator == 'H':
                                            evs.sort(key=lambda v: v.real, reverse=True)
                                        else:
                                            evs.sort(key=lambda v: abs(v), reverse=True)
                                        lam_dict[L] = evs[0]
                                except Exception:
                                    pass

                L_fit = sorted(lam_dict.keys())
                lam_fit = [lam_dict[L] for L in L_fit]

                C, A, B = fit_f_L(L_fit, lam_fit, args.operator)
                if C is not None:
                    C_matrix[j, i] = C
                    key = f"x={x_val:.2f}_y={y_val:.2f}"
                    results[key] = {"C": C, "A": A, "B": B, "L": L_fit, "lam": [float(abs(l)) if args.operator=='T' else float(l.real) for l in lam_fit]}

                print(f"y={y_val:.2f}, x={x_val:.2f} -> C={C:.4f}" if C is not None else f"y={y_val:.2f}, x={x_val:.2f} -> C=None")

        plt.figure(figsize=(8, 6))
        X_grid, Y_grid = np.meshgrid(x_vals, y_vals)
        c_plot = plt.contourf(X_grid, Y_grid, C_matrix, levels=20, cmap='viridis')

        # Scale to central charge
        # T: c = -24C/pi, H: c = -24C/(pi * vF). We assume vF=1.
        c_charge_matrix = -24 * C_matrix / np.pi

        c_plot_2 = plt.contourf(X_grid, Y_grid, c_charge_matrix, levels=20, cmap='inferno')
        plt.colorbar(c_plot_2, label='Extrapolated Central Charge c')
        plt.title(f"Phase Diagram for {args.operator}(x,y,z={args.z_val:.1f}) | n={n:.2f}")
        plt.xlabel("Boltzmann weight x")
        plt.ylabel("Boltzmann weight y")

        filename = f"phase_diagram_C_{args.operator}_n{n:.2f}_z{args.z_val:.1f}_x{args.x_start}-{args.x_stop}_y{args.y_start}-{args.y_stop}.png"
        plt.savefig(os.path.join(out_dir, filename), dpi=150)
        plt.close()

        with open(os.path.join(out_dir, f"phase_diagram_data_{args.operator}_n{n:.2f}_z{args.z_val:.1f}.json"), 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Sweep complete for {type_str}. Data saved.")

if __name__ == "__main__":
    sweep()
