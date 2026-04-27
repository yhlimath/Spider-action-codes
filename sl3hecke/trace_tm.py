import sys
import os
import argparse
import numpy as np
import sympy
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from matrix_features.conjecture_test import build_sl3_magnetic_matrix
from sl3hecke.sl3_hecke import Sl3HeckeArnoldi

def compute_trace(L, M, x, y, n_val=None, symbolic=False):
    if symbolic:
        print(f"Building full symbolic matrix T for L={L}, (x,y)=({x},{y})...")
        q_sym = sympy.Symbol('q')
        n_sym = q_sym + 1/q_sym
        T, basis = build_sl3_magnetic_matrix(L, x, y, n_val=None, n_sym=n_sym)

        if T is None or len(basis) == 0:
            print(f"Module V^{{L={L}, ({x},{y})}} is empty or invalid.")
            return None, []

        dim = len(basis)
        print(f"Matrix dimension: {dim}")
        print(f"Computing Tr(T^{M}) symbolically (this may be slow)...")

        T_sym = sympy.Matrix(T)
        T_pow = T_sym ** M
        trace_val = T_pow.trace()
        trace_simplified = sympy.simplify(trace_val)
        print(f"\nSymbolic Trace (in terms of q, where n = q + 1/q):\n{trace_simplified}")
        return str(trace_simplified), []

    else:
        print(f"Computing dense numeric trace for exactness...")
        T, basis = build_sl3_magnetic_matrix(L, x, y, n_val=n_val, n_sym=None)
        if T is None or len(basis) == 0:
            print(f"Module V^{{L={L}, ({x},{y})}} is empty or invalid.")
            return None, []

        dim = len(basis)
        print(f"Matrix dimension: {dim}")

        T_num = np.array(T, dtype=complex)
        eigs = np.linalg.eigvals(T_num)

        # Clean and sort eigenvalues
        eigs_abs = np.abs(eigs)
        eigs = eigs[eigs_abs > 1e-10]

        if len(eigs) > 0:
            if np.all(np.abs(np.imag(eigs)) < 1e-8):
                eigs_sorted = np.sort(np.real(eigs))
            else:
                eigs_sorted = eigs[np.argsort(np.abs(eigs))]
        else:
            eigs_sorted = np.array([])

        trace_val = np.sum(eigs_sorted ** M) if len(eigs_sorted) > 0 else 0.0

        if np.abs(trace_val.imag) < 1e-10:
            trace_clean = float(trace_val.real)
        else:
            trace_clean = complex(trace_val)

        print(f"\nNumeric Trace (n={n_val}):\n{trace_clean}")

        eigs_export = [float(e) if np.abs(e.imag) < 1e-10 else {"re": float(e.real), "im": float(e.imag)} for e in eigs_sorted]
        return trace_clean, eigs_export

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute Tr(T^M) on sl3 magnetic module V^{L, (x,y)}")
    parser.add_argument("-L", type=int, required=True, help="System size L")
    parser.add_argument("-M", type=int, required=True, help="Power M of the transfer matrix")
    parser.add_argument("-x", type=int, required=True, help="Magnetic index x")
    parser.add_argument("-y", type=int, required=True, help="Magnetic index y")
    parser.add_argument("-n", "--n_val", type=float, default=1.372, help="Numeric loop weight n (default 1.372)")
    parser.add_argument("--symbolic", action="store_true", help="Compute trace symbolically")
    parser.add_argument("--export", action="store_true", help="Export result to experiment_outputs/")

    args = parser.parse_args()

    # Validate constraints (L = m + 2x + y, where m is string length and must be m=L, wait.
    # For sl3, the length of the string is L, and valid endpoints depend on it)
    if 2*args.x + args.y > args.L:
        print(f"Error: 2x+y ({2*args.x + args.y}) cannot exceed L ({args.L})")
        sys.exit(1)
    if (args.L + 2*args.x + args.y) % 3 != 0:
        print(f"Error: (L + 2x + y) must be divisible by 3")
        sys.exit(1)

    trace_res = compute_trace(args.L, args.M, args.x, args.y, args.n_val, args.symbolic)

    if args.export and trace_res is not None:
        os.makedirs("experiment_outputs", exist_ok=True)
        mode = "sym" if args.symbolic else "num"
        out_file = f"experiment_outputs/trace_L{args.L}_M{args.M}_x{args.x}_y{args.y}_{mode}.json"

        data = {
            "L": args.L,
            "M": args.M,
            "x": args.x,
            "y": args.y,
            "mode": mode,
            "n": "q+1/q" if args.symbolic else args.n_val,
            "trace": trace_res
        }
        with open(out_file, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\nResult exported to {out_file}")
