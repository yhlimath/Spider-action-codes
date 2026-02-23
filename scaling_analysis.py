import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sl3_hecke import Sl3HeckeArnoldi
import time
import json
import datetime

def analyze_scaling(L_values, n_value, output_filename="eigenvalues_log.json"):
    print(f"Starting Finite-Size Scaling Analysis for n = {n_value}")
    print("=" * 60)

    # Data structure to hold results
    results_data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "n_value": n_value,
        "operator": "H = sum(e_i)",
        "L_values": L_values,
        "scaling_data": [],
        "central_charge_estimate": None
    }

    leading_eigenvalues = []
    f_L_values = []
    inv_L_squared_values = []

    eigenvalues_by_L = {}

    for L in L_values:
        print(f"\nProcessing L = {L} (System size 3L = {3*L})...")
        start_time = time.time()

        try:
            # Instantiate Solver
            solver = Sl3HeckeArnoldi(L=L, n_value=n_value)
            print(f"  Dimension of space: {solver.dim}")

            k_arnoldi = 50
            if k_arnoldi < 1: k_arnoldi = 1

            # Use custom Arnoldi iteration
            hessenberg_mat = solver.arnoldi_iteration(k=k_arnoldi)

            # Compute eigenvalues of Hessenberg matrix
            eigenvalues = np.linalg.eigvals(hessenberg_mat)
            eigenvalues_by_L[L] = eigenvalues

            # Sort by magnitude descending
            eigenvalues_sorted = sorted(eigenvalues, key=lambda x: abs(x), reverse=True)
            Lambda_L = eigenvalues_sorted[0]

            print(f"  Leading eigenvalue (LM): {Lambda_L}")

            # Calculate f_L
            val_log = np.log(Lambda_L)
            f_L = val_log / (3 * L)

            f_L_real = np.real(f_L)

            leading_eigenvalues.append(Lambda_L)
            f_L_values.append(f_L_real)
            inv_L_squared_values.append(1.0 / (L**2))

            print(f"  f_L (real part): {f_L_real}")
            print(f"  Time elapsed: {time.time() - start_time:.2f}s")

            # Store data for this L
            results_data["scaling_data"].append({
                "L": L,
                "dim": solver.dim,
                "leading_eigenvalue": {
                    "real": float(np.real(Lambda_L)),
                    "imag": float(np.imag(Lambda_L)),
                    "magnitude": float(abs(Lambda_L))
                },
                "f_L_real": float(f_L_real),
                "top_k_eigenvalues": [
                    {"real": float(np.real(e)), "imag": float(np.imag(e))}
                    for e in eigenvalues_sorted[:min(10, len(eigenvalues_sorted))]
                ]
            })

        except Exception as e:
            print(f"  Error processing L={L}: {e}")
            import traceback
            traceback.print_exc()


        # Save to JSON file
        with open(output_filename, 'w') as f:
            json.dump(results_data, f, indent=4)
        print(f"Saved eigenvalues and results to '{output_filename}'")

if __name__ == "__main__":
    L_range = [2,3,4,5,6]
    n_val = 1.0

    analyze_scaling(L_range, n_val)
