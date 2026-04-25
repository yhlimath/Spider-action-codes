import sys
import os
import argparse
import numpy as np
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sl3hecke.conjecture_test import build_sl3_magnetic_matrix

def compute_all(L, n_val):
    print(f"Computing all non-zero eigenvalues for sl3 magnetic modules of length L={L}, n={n_val}")

    results = {
        "L": L,
        "n": n_val,
        "modules": {}
    }

    total_valid_modules = 0
    total_dimension = 0

    for x in range(L + 1):
        for y in range(L + 1):
            if 2*x + y > L: continue
            if (L + 2*x + y) % 3 != 0: continue

            print(f"\nEvaluating sector (x,y) = ({x},{y})...")
            T, basis = build_sl3_magnetic_matrix(L, x, y, n_val=n_val, n_sym=None)

            dim = len(basis) if basis else 0
            if dim == 0:
                print("  Module is empty.")
                continue

            total_valid_modules += 1
            total_dimension += dim
            print(f"  Dimension: {dim}")

            T_num = np.array(T, dtype=complex)
            eigs = np.linalg.eigvals(T_num)

            eigs_abs = np.abs(eigs)
            eigs = eigs[eigs_abs > 1e-10]

            if len(eigs) > 0:
                if np.all(np.abs(np.imag(eigs)) < 1e-8):
                    eigs_sorted = np.sort(np.real(eigs))
                else:
                    eigs_sorted = eigs[np.argsort(np.abs(eigs))]
            else:
                eigs_sorted = np.array([])

            print(f"  Non-zero eigenvalues found: {len(eigs_sorted)}")

            eigs_export = [float(e) if np.abs(e.imag) < 1e-10 else {"re": float(e.real), "im": float(e.imag)} for e in eigs_sorted]

            results["modules"][f"x={x}, y={y}"] = {
                "dimension": dim,
                "eigenvalues": eigs_export
            }

    print(f"\nDone! Evaluated {total_valid_modules} modules. Total matrix dimension spanned: {total_dimension}")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute all explicit non-zero eigenvalues for valid sl3 magnetic modules V^{L, (x,y)}")
    parser.add_argument("-L", type=int, required=True, help="System size L")
    parser.add_argument("-n", "--n_val", type=float, default=1.414213562373095, help="Numeric loop weight n (default 1.414213562373095)")

    args = parser.parse_args()

    data = compute_all(args.L, args.n_val)

    os.makedirs("experiment_outputs", exist_ok=True)
    out_file = f"experiment_outputs/all_eigenvalues_L{args.L}.json"
    with open(out_file, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Exported data to {out_file}")
