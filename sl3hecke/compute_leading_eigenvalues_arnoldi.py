import sys
import os
import argparse
import numpy as np
import json
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sl3hecke.sl3_hecke import Polynomial
from sl3hecke.magnetic_modules import generate_constrained_strings, ed

class MagneticArnoldiSolver:
    def __init__(self, m, x, y, n_value):
        self.L = m
        self.x = x
        self.y = y
        self.n_value = n_value
        self.basis_strings = generate_constrained_strings(m, x, y)
        self.dim = len(self.basis_strings)
        self.string_to_idx = {tuple(s): i for i, s in enumerate(self.basis_strings)}
        self.num_generators = m - 1
        self.e_cache = {}

    def _get_e_k_action(self, s, k):
        s_tuple = tuple(s)
        cache_key = (s_tuple, k)
        if cache_key in self.e_cache:
            return self.e_cache[cache_key]

        res_pairs = ed([(Polynomial.constant(1), s)], k, self.L, self.x, self.y)
        action_results = []
        for c, res_s in res_pairs:
            res_s_tuple = tuple(res_s)
            if res_s_tuple in self.string_to_idx:
                val = c.evaluate(self.n_value)
                action_results.append((val, self.string_to_idx[res_s_tuple]))

        self.e_cache[cache_key] = action_results
        return action_results

    def apply_single_generator(self, v, k):
        w = np.zeros_like(v)
        for idx, s in enumerate(self.basis_strings):
            coeff_v = v[idx]
            if abs(coeff_v) < 1e-12: continue

            action_results = self._get_e_k_action(s, k)
            for val, target_idx in action_results:
                w[target_idx] += coeff_v * val
        return w

    def apply_H(self, v):
        w = np.zeros_like(v)
        for k in range(1, self.num_generators + 1):
            w += self.apply_single_generator(v, k)
        return w

    def apply_T(self, v):
        w = v.copy()
        odd_indices = range(1, self.num_generators + 1, 2)
        even_indices = range(2, self.num_generators + 1, 2)

        for k in odd_indices:
            w = self.apply_single_generator(w, k)
        for k in even_indices:
            w = self.apply_single_generator(w, k)
        return w

    def arnoldi_iteration(self, k, operator='T'):
        k = min(k, self.dim)
        Q = np.zeros((self.dim, k + 1), dtype=complex)
        h = np.zeros((k + 1, k), dtype=complex)

        v_start = np.random.rand(self.dim) + 1j * np.random.rand(self.dim)
        v_start = v_start / np.linalg.norm(v_start)
        Q[:, 0] = v_start

        for j in range(k):
            v_j = Q[:, j]

            if operator == 'H':
                w = self.apply_H(v_j)
            else:
                w = self.apply_T(v_j)

            for i in range(j + 1):
                h[i, j] = np.vdot(Q[:, i], w)
                w = w - h[i, j] * Q[:, i]

            h[j + 1, j] = np.linalg.norm(w)
            if h[j + 1, j] < 1e-10:
                return h[:j + 1, :j + 1]

            Q[:, j + 1] = w / h[j + 1, j]

        return h[:k, :k]


def compute_all_arnoldi(L, n_val, top_k, operator='T'):
    print(f"Computing top {top_k} non-zero eigenvalues of operator {operator} via Arnoldi for sl3 magnetic modules of length L={L}, n={n_val}")

    results = {
        "L": L,
        "n": n_val,
        "operator": operator,
        "modules": {}
    }

    total_valid_modules = 0
    total_dimension = 0

    for x in range(1): #Change: only analyze (x,y)=(0,0) sector for now
        for y in range(1):
            if 2*x + y > L: continue
            if (L + 2*x + y) % 3 != 0: continue

            print(f"\nEvaluating sector (x,y) = ({x},{y})...")
            start_time = time.time()
            solver = MagneticArnoldiSolver(m=L, x=x, y=y, n_value=n_val)
            dim = solver.dim

            if dim == 0:
                print("  Module is empty.")
                continue

            total_valid_modules += 1
            total_dimension += dim
            print(f"  Dimension: {dim}")

            # To extract up to top_k, we need the Krylov dimension to be slightly larger
            # to ensure convergence of the top_k eigenvalues, bounded by matrix dim.
            k_arnoldi = min(dim, top_k * 2 + 10, 50)
            if dim <= 50:
                k_arnoldi = dim

            h_mat = solver.arnoldi_iteration(k_arnoldi, operator=operator)
            eigs = np.linalg.eigvals(h_mat)

            eigs_abs = np.abs(eigs)
            eigs = eigs[eigs_abs > 1e-10]

            if len(eigs) > 0:
                if np.all(np.abs(np.imag(eigs)) < 1e-8):
                    eigs_sorted = np.sort(np.real(eigs))[::-1]
                else:
                    eigs_sorted = eigs[np.argsort(np.abs(eigs))[::-1]]
            else:
                eigs_sorted = np.array([])

            if top_k is not None and top_k > 0:
                eigs_sorted = eigs_sorted[:top_k]

            print(f"  Non-zero leading eigenvalues found: {len(eigs_sorted)}")
            print(f"  Time elapsed: {time.time() - start_time:.2f}s")
            
            eigs_export = [float(e) if np.abs(e.imag) < 1e-10 else {"re": float(e.real), "im": float(e.imag)} for e in eigs_sorted]

            results["modules"][f"x={x}, y={y}"] = {
                "dimension": dim,
                "eigenvalues": eigs_export
            }

    print(f"\nDone! Evaluated {total_valid_modules} modules. Total matrix dimension spanned: {total_dimension}")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute top explicit non-zero eigenvalues matrix-free via Arnoldi for valid sl3 magnetic modules V^{L, (x,y)}")
    parser.add_argument("-L", type=int, default=6, help="System size L")
    parser.add_argument("-n", "--n_val", type=float, default=1.414213562373095, help="Numeric loop weight n (default 1.414213562373095)")
    parser.add_argument("-k", "--top_k", type=int, default=50, help="Number of leading eigenvalues to extract per sector (default 10)")
    parser.add_argument("-O", "--operator", type=str, default="H", choices=["T", "H"], help="Operator to diagonalize: 'T' (Transfer Matrix) or 'H' (Hamiltonian)")

    args = parser.parse_args()

    data = compute_all_arnoldi(args.L, args.n_val, args.top_k, args.operator)

    os.makedirs("experiment_outputs", exist_ok=True)
    out_file = f"experiment_outputs/all_eigenvalues_{args.operator}_L{args.L}.json"
    with open(out_file, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Exported data to {out_file}")
