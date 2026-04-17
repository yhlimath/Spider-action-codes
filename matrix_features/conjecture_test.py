import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import sympy
import json

from dilute_temperley_lieb.dtl_transfer_matrix import construct_dtl_transfer_matrix
from sl3hecke.magnetic_modules import generate_constrained_strings, ed
from sl3hecke.sl3_hecke import Polynomial

def eval_poly(poly, n_val):
    if hasattr(poly, 'evaluate'):
        return poly.evaluate(n_val)
    return poly

def compute_sl3_magnetic_eigenvalues(L, x, y, n_val):
    """Compute eigenvalues of T on V^{L, (x,y)}."""
    basis = generate_constrained_strings(L, x, y)
    dim = len(basis)
    if dim == 0:
        return [], 0

    string_to_idx = {tuple(s): i for i, s in enumerate(basis)}
    matrix = np.zeros((dim, dim), dtype=complex)

    num_generators = L - 1

    def apply_seq(states, indices):
        curr = states
        for k in indices:
            next_curr = []
            for coeff, s in curr:
                res = ed([(coeff, s)], k, L, x, y)
                next_curr.extend(res)
            cons = {}
            for c, s in next_curr:
                t = tuple(s)
                if t not in cons:
                    cons[t] = Polynomial.constant(0)
                cons[t] = cons[t] + c
            curr = [(c, list(s)) for s, c in cons.items() if not c.is_zero()]
        return curr

    for k, s in enumerate(basis):
        curr = [(Polynomial.constant(1), s)]
        odd_indices = range(1, num_generators + 1, 2)
        even_indices = range(2, num_generators + 1, 2)

        curr = apply_seq(curr, odd_indices)
        curr = apply_seq(curr, even_indices)

        for coeff, end_s in curr:
            t = tuple(end_s)
            if t in string_to_idx:
                r_idx = string_to_idx[t]
                val = eval_poly(coeff, n_val)
                matrix[r_idx, k] += val

    eigvals = np.linalg.eigvals(matrix)
    return np.sort(np.real(eigvals)), dim

def analyze_dtl(L, j, n_val):
    T_dtl, states_dtl = construct_dtl_transfer_matrix(L, j)
    if len(states_dtl) == 0:
        return {}

    q_val = (n_val + np.sqrt(n_val**2 - 4 + 0j)) / 2

    dim_dtl = len(states_dtl)
    T_dtl_num = np.zeros((dim_dtl, dim_dtl), dtype=complex)
    q_sym = sympy.Symbol('q')
    for r in range(dim_dtl):
        for c in range(dim_dtl):
            val = T_dtl[r][c]
            if val != 0:
                val_c = complex(val.subs(q_sym, q_val))
                T_dtl_num[r, c] = val_c

    p_groups = {}
    for i, s in enumerate(states_dtl):
        p = s.count(0)
        if p not in p_groups: p_groups[p] = []
        p_groups[p].append(i)

    dtl_data = {}
    for p, indices in p_groups.items():
        sub_T = T_dtl_num[np.ix_(indices, indices)]
        eigs = np.sort(np.real(np.linalg.eigvals(sub_T)))
        eigs = eigs[np.abs(eigs) > 1e-10]
        dtl_data[p] = {
            "dimension": len(indices),
            "eigenvalues": [float(e) for e in eigs]
        }
    return dtl_data

def evaluate_conjecture_and_export(L, n_val=1.372):
    print(f"Generating data for L={L}, n={n_val}...")

    output_data = {
        "L": L,
        "n": n_val,
        "dtl_modules": {},
        "sl3_modules": {}
    }

    # 1. Gather sl3 data
    print("  Computing sl3 modules...")
    for x in range(L + 1):
        for y in range(L + 1):
            if 2*x + y > L: continue
            if (L + 2*x + y) % 3 != 0: continue

            eigs, dim = compute_sl3_magnetic_eigenvalues(L, x, y, n_val)
            if dim > 0:
                key = f"L={L}, x={x}, y={y}"
                eigs_filtered = eigs[np.abs(eigs) > 1e-10]
                output_data["sl3_modules"][key] = {
                    "dimension": dim,
                    "eigenvalues": [float(e) for e in eigs_filtered]
                }

    # 2. Gather dTL data
    print("  Computing dTL modules...")
    for j in range(L + 1):
        try:
            dtl_res = analyze_dtl(L, j, n_val)
            for p, data in dtl_res.items():
                key = f"L={L}, j={j}, p={p}"
                output_data["dtl_modules"][key] = data
        except Exception as e:
            pass

    # Export
    os.makedirs("experiment_outputs", exist_ok=True)
    out_path = f"experiment_outputs/eigenvalue_mapping_L{L}.json"
    with open(out_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"  Data exported to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-L", type=int, default=6)
    parser.add_argument("-n", "--n_val", type=float, default=1.372)
    parser.add_argument("--symbolic", action="store_true", help="Enable symbolic computation of eigensystems")
    args = parser.parse_args()

    evaluate_conjecture_and_export(args.L, args.n_val, args.symbolic)
