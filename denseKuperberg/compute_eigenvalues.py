import os
import sys


import numpy as np
import json
import os
import time
from scipy.linalg import eigvals
from transfer_matrix import build_transfer_matrix

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def compute_and_log():
    L_list = [ 4, 5, 6]
    n_values = [0.5, 0.7, 0.9, 1.0, 1.2, 1.41421356, 1.5, 1.8, 2.0]
    types = ['E+H+H2', 'E+H', 'H2']
    orders = ['sequential', 'staggered'] #
    x, y = 0, 0

    out_dir = "experiment_outputs/denseKuperberg"
    os.makedirs(out_dir, exist_ok=True)

    # Structure: logs[type][order][n][L] = lambda_0
    logs = {t: {o: {n: {} for n in n_values} for o in orders} for t in types}

    for L in L_list:
        print(f"\n--- Computing for L={L} ---")
        for type_str in types:
            for order_str in orders:
                print(f"Type: {type_str}, Order: {order_str}")
                for n in n_values:
                    start_time = time.time()
                    M, paths = build_transfer_matrix(L, x, y, type_str, order_str, n)

                    if M.shape[0] == 0:
                        lambda_0 = None
                    else:
                        evs = eigvals(M)
                        # Remove zero eigenvalues (threshold)
                        evs = [ev for ev in evs if abs(ev) > 1e-10]
                        if not evs:
                            lambda_0 = None
                        else:
                            # Sort by absolute value descending
                            evs.sort(key=lambda x: abs(x), reverse=True)

                            # Keep complex format for logging
                            l0 = evs[0]
                            lambda_0 = {"real": float(l0.real), "imag": float(l0.imag), "abs": float(abs(l0))}

                    if lambda_0:
                        logs[type_str][order_str][n][L] = lambda_0

                    elapsed = time.time() - start_time
                    print(f"  n={n:.1f}: dim={M.shape[0]}, elapsed={elapsed:.2f}s, lambda_0={lambda_0['abs'] if lambda_0 else 'None'}")

    # Save to JSON
    out_file = os.path.join(out_dir, "eigenvalue_logs.json")
    with open(out_file, 'w') as f:
        json.dump(logs, f, indent=2)
    print(f"\nSaved logs to {out_file}")

if __name__ == "__main__":
    compute_and_log()
