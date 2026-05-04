import numpy as np
import json
import os
import time
from scipy.linalg import eigvals
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from denseKuperberg.arnoldi import KuperbergArnoldiSolver

def get_n_values(config):
    n_vals = set()
    if 'specific_values' in config:
        n_vals.update(config['specific_values'])
    if 'sweep' in config:
        start = config['sweep'].get('start', 0.5)
        stop = config['sweep'].get('stop', 2.0)
        step = config['sweep'].get('step', 0.1)
        # Using np.arange could cause float precision issues, so we round
        sweep_vals = np.arange(start, stop + step/2, step)
        n_vals.update(np.round(sweep_vals, 4))
    return sorted(list(n_vals))

def compute_and_log():
    L_list = [3,4,5,6,7,8,9]#4, 5, 6, 7, 8

    config = {
        'specific_values': [1.41421356237095, 0.8660254037844386, 0.7071067811865476, 1.7320508075688772], # sqrt(2) and 1/sqrt(2)
        'sweep': {
            'start': -1.0,
            'stop': 2.5,
            'step': 0.1
        }
    }

    n_values = get_n_values(config)
    types = ['E+H'] # 'H2' 'E+H+H2'
    orders = ['sequential'] # 'staggered'
    x, y = 0, 0
    arnoldi_k = 50

    out_dir = "experiment_outputs/denseKuperberg"
    os.makedirs(out_dir, exist_ok=True)

    # Structure: logs[type][order][n][L] = lambda_0
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

                    if solver.dim == 0:
                        lambda_0 = None
                    else:
                        H_matrix, _ = solver.arnoldi_iteration(arnoldi_k)
                        if H_matrix.shape[0] == 0:
                            lambda_0 = None
                        else:
                            evs = eigvals(H_matrix)
                            evs = [ev for ev in evs if abs(ev) > 1e-10]
                            if not evs:
                                lambda_0 = None
                            else:
                                evs.sort(key=lambda x: abs(x), reverse=True)
                                l0 = evs[0]
                                lambda_0 = {"real": float(l0.real), "imag": float(l0.imag), "abs": float(abs(l0))}

                    if lambda_0:
                        logs[type_str][order_str][n_str][L] = lambda_0

                    elapsed = time.time() - start_time
                    print(f"  n={n:.4f}: dim={solver.dim}, elapsed={elapsed:.2f}s, lambda_0={lambda_0['abs'] if lambda_0 else 'None'}")

    # Save to JSON
    out_file = os.path.join(out_dir, "eigenvalue_logs.json")
    with open(out_file, 'w') as f:
        json.dump(logs, f, indent=2)
    print(f"\nSaved logs to {out_file}")

if __name__ == "__main__":
    compute_and_log()
