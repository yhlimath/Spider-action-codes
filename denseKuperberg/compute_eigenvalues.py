import numpy as np
import json
import time

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from scipy.sparse.linalg import eigs
from scipy.sparse.linalg import LinearOperator
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
    L_list = [ 5, 6, 7, 8]

    config = {
        'specific_values': [1.4142],  # sqrt(2) and golden ratio
        'sweep': {
            'start': 0,
            'stop': -1,
            'step': 0.25
        }
    }

    n_values = get_n_values(config)
    types = ['E+H'] #'E+H+H2', 'E+H', 'H2'
    orders = ['sequential'] #, 'staggered'
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

                    solver = KuperbergArnoldiSolver(L, x, y, type_str, order_str, n)

                    top_eigenvalues = []
                    if solver.dim > 0:
                        if solver.dim <= extract_top_k:
                            # For small dims, use standard dense eig
                            from scipy.linalg import eigvals
                            H_matrix, _ = solver.arnoldi_iteration(solver.dim)
                            if H_matrix.shape[0] > 0:
                                evs = eigvals(H_matrix)
                        else:
                            # Use Implicitly Restarted Arnoldi for true extreme eigenvalues
                            # We need ncv > k. scipy defaults to 2*k+1 or similar.
                            # k cannot be >= dim-1.
                            k_actual = min(extract_top_k, solver.dim - 2)
                            if k_actual > 0:
                                def matvec(v):
                                    return solver.apply_T(v)
                                A = LinearOperator((solver.dim, solver.dim), matvec=matvec, dtype=complex)
                                evs, _ = eigs(A, k=k_actual, which='LM')
                            else:
                                evs = []

                        if 'evs' in locals() and len(evs) > 0:
                            evs = [ev for ev in evs if abs(ev) > 1e-10]
                            if evs:
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

    with open(os.path.join(out_dir, "eigenvalue_logs_top_k.json"), 'w') as f:
        json.dump(logs, f, indent=2)

if __name__ == "__main__":
    compute_and_log()
