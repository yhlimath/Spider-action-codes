import numpy as np
import json
import os
import time
import argparse
from scipy.sparse.linalg import eigs
from scipy.sparse.linalg import LinearOperator
from scipy.linalg import eigvals
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from denseKuperberg.zigzag import ZigzagArnoldiSolver

def compute_and_log():
    parser = argparse.ArgumentParser()
    parser.add_argument('--L_max', type=int, default=11)
    args = parser.parse_args()

    L_list = [5, 8, 11]
    #L_list = [L for L in L_list if L <= args.L_max]

    n_values = [0.01 * i for i in range(1, 200)]
    
    x, y = 0, 0
    extract_top_k = 1

    out_dir = "experiment_outputs/denseKuperberg"
    os.makedirs(out_dir, exist_ok=True)

    logs = {"Zigzag": {"symmetric": {str(n): {} for n in n_values}}}

    for L in L_list:
        print(f"\n--- Computing Zigzag for L={L} ---")
        for n in n_values:
            start_time = time.time()
            n_str = str(n)

            solver = ZigzagArnoldiSolver(L, x, y, n)

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
                            return solver.apply_T(v)
                        A = LinearOperator((solver.dim, solver.dim), matvec=matvec, dtype=complex)

                        try:
                            evs, _ = eigs(A, k=k_actual, which='LM')
                        except Exception:
                            evs = []
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
                logs["Zigzag"]["symmetric"][n_str][L] = top_eigenvalues

            elapsed = time.time() - start_time
            l0_str = f"{top_eigenvalues[0]['abs']:.4f}" if top_eigenvalues else "None"
            print(f"  n={n:.4f}: dim={solver.dim}, elapsed={elapsed:.2f}s, top {len(top_eigenvalues)} evs, lambda_0={l0_str}")

    with open(os.path.join(out_dir, f"eigenvalue_logs_zigzag.json"), 'w') as f:
        json.dump(logs, f, indent=2)

if __name__ == "__main__":
    compute_and_log()
