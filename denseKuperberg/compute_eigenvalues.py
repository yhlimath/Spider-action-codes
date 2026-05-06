import numpy as np
import json
import os
import time
import argparse
from scipy.sparse.linalg import eigs
from scipy.sparse.linalg import LinearOperator
from scipy.linalg import eigvals
from denseKuperberg.arnoldi import KuperbergArnoldiSolver

def get_n_values(config):
    n_vals = set()
    if 'specific_values' in config:
        n_vals.update(config['specific_values'])
    if 'sweep' in config:
        start = config['sweep'].get('start', 0.5)
        stop = config['sweep'].get('stop', 2.0)
        step = config['sweep'].get('step', 0.1)
        sweep_vals = np.arange(start, stop + step/2, step)
        n_vals.update(np.round(sweep_vals, 4))
    return sorted(list(n_vals))

def compute_and_log():
    parser = argparse.ArgumentParser()
    parser.add_argument('--operator', type=str, default='T', choices=['T', 'H'])
    parser.add_argument('--L_max', type=int, default=6)
    parser.add_argument('--x_val', type=float, default=1.0)
    parser.add_argument('--y_val', type=float, default=1.0)
    parser.add_argument('--z_val', type=float, default=1.0)
    args = parser.parse_args()

    L_list = [2, 3, 4, 5, 6]
    L_list = [L for L in L_list if L <= args.L_max]

    config = {
        'specific_values': [1.4142, 1.7321], # sqrt(2) and sqrt(3) as special points
        'sweep': {
            'start': 0.5,
            'stop': 2.0,
            'step': 0.1
        }
    }

    n_values = get_n_values(config)
    types = ['T(x,y,z)']
    orders = ['sequential', 'staggered']
    x, y = 0, 0
    extract_top_k = 50

    out_dir = "experiment_outputs/denseKuperberg"
    os.makedirs(out_dir, exist_ok=True)

    logs = {t: {o: {str(n): {} for n in n_values} for o in orders} for t in types}

    for L in L_list:
        print(f"\n--- Computing for L={L} ---")
        for type_str in types:
            for order_str in orders:
                print(f"Type: {type_str}, Order: {order_str}")
                for n in n_values:
                    start_time = time.time()
                    n_str = str(n)

                    solver = KuperbergArnoldiSolver(
                        L, x, y, type_str, order_str, n,
                        x_value=args.x_val, y_value=args.y_val, z_value=args.z_val, operator=args.operator
                    )

                    top_eigenvalues = []
                    if solver.dim > 0:
                        if solver.dim <= extract_top_k:
                            H_matrix, _ = solver.arnoldi_iteration(solver.dim)
                            if H_matrix.shape[0] > 0:
                                evs = eigvals(H_matrix)
                        else:
                            k_actual = min(extract_top_k, solver.dim - 2)
                            if k_actual > 0:
                                def matvec(v):
                                    if args.operator == 'H':
                                        return solver.apply_H(v)
                                    else:
                                        return solver.apply_T(v)
                                A = LinearOperator((solver.dim, solver.dim), matvec=matvec, dtype=complex)

                                # For Hamiltonian we usually want LR (largest real) if looking for ground states,
                                # but for consistency with modulus let's use LM (largest magnitude)
                                which_eig = 'LM'
                                try:
                                    evs, _ = eigs(A, k=k_actual, which=which_eig)
                                except Exception:
                                    evs = []
                            else:
                                evs = []

                        if 'evs' in locals() and len(evs) > 0:
                            evs = [ev for ev in evs if abs(ev) > 1e-10]
                            if evs:
                                # if operator == H, sort by real descending
                                if args.operator == 'H':
                                    evs.sort(key=lambda v: v.real, reverse=True)
                                else:
                                    evs.sort(key=lambda v: abs(v), reverse=True)

                                for l in evs[:extract_top_k]:
                                    top_eigenvalues.append({
                                        "real": float(l.real),
                                        "imag": float(l.imag),
                                        "abs": float(abs(l))
                                    })

                    if top_eigenvalues:
                        logs[type_str][order_str][n_str][L] = top_eigenvalues

                    elapsed = time.time() - start_time
                    l0_str = f"{top_eigenvalues[0]['abs']:.4f}" if top_eigenvalues else "None"
                    print(f"  n={n:.4f}: dim={solver.dim}, elapsed={elapsed:.2f}s, top {len(top_eigenvalues)} evs, lambda_0={l0_str}")

    with open(os.path.join(out_dir, f"eigenvalue_logs_top_k_{args.operator}.json"), 'w') as f:
        json.dump(logs, f, indent=2)

if __name__ == "__main__":
    compute_and_log()
