import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import numpy as np
import copy
import os
from sl3hecke.sl3_hecke import Polynomial, e, generate_all_valid_strings
from sl3hecke.magnetic_modules import ed, generate_constrained_strings

class SymbolicHeckeBuilder:
    def __init__(self, L=None, is_magnetic=False, m=None, x=None, y=None, use_all_valid=False):
        self.L = L
        self.is_magnetic = is_magnetic
        self.m = m if m is not None else (3 * L if L is not None else None)
        self.x = x
        self.y = y
        self.use_all_valid = use_all_valid

        if is_magnetic:
            if use_all_valid:
                # generate all valid of length m
                self.basis_strings = self._generate_all_valid_len_m(self.m)
            else:
                self.basis_strings = generate_constrained_strings(self.m, self.x, self.y)
        else:
            self.basis_strings = generate_all_valid_strings(L)

        self.dim = len(self.basis_strings)
        self.string_to_idx = {tuple(s): i for i, s in enumerate(self.basis_strings)}

    def _generate_all_valid_len_m(self, m):
        def backtrack(sequence, count_1, count_0, count_neg1, results):
            if len(sequence) == m:
                results.append(sequence[:])
                return
            backtrack(sequence + [1], count_1 + 1, count_0, count_neg1, results)
            if count_0 < count_1:
                backtrack(sequence + [0], count_1, count_0 + 1, count_neg1, results)
            if count_neg1 < count_0:
                backtrack(sequence + [-1], count_1, count_0, count_neg1 + 1, results)
        results = []
        backtrack([], 0, 0, 0, results)
        return results

    def _apply_e_k(self, s, k):
        """Apply e_k symbolically to a basis string s."""
        if self.is_magnetic:
            if self.use_all_valid:
                curr_x = s.count(1) - s.count(0)
                curr_y = s.count(0) - s.count(-1)
                results = ed([(Polynomial.constant(1), list(s))], k, self.m, curr_x, curr_y)
            else:
                results = ed([(Polynomial.constant(1), list(s))], k, self.m, self.x, self.y)
        else:
            results = e([(Polynomial.constant(1), list(s))], k)
        return results

    def get_H_matrix(self):
        """Build the symbolic matrix for H = sum e_k."""
        matrix = [[Polynomial() for _ in range(self.dim)] for _ in range(self.dim)]

        num_generators = 3 * self.L - 1

        for j, s in enumerate(self.basis_strings):
            for k in range(1, num_generators + 1):
                results = self._apply_e_k(s, k)
                for coeff, new_s in results:
                    idx_i = self.string_to_idx.get(tuple(new_s))
                    if idx_i is not None:
                        matrix[idx_i][j] = matrix[idx_i][j] + coeff

        return matrix

    def get_T_matrix(self):
        """Build the symbolic matrix for T = prod e_even * prod e_odd."""

        num_generators = self.m - 1 if self.is_magnetic else 3 * self.L - 1
        odd_indices = range(1, num_generators + 1, 2)
        even_indices = range(2, num_generators + 1, 2)

        # We construct the matrix column by column.
        # Column j corresponds to T applied to basis vector s_j.

        matrix = [[Polynomial() for _ in range(self.dim)] for _ in range(self.dim)]

        for j, s_start in enumerate(self.basis_strings):
            # Start with vector having 1 at index j: {j: 1}
            current_state = {j: Polynomial.constant(1)}

            # Apply odd generators
            for k in odd_indices:
                next_state = {}
                for idx, poly in current_state.items():
                    s_current = self.basis_strings[idx]

                    results = self._apply_e_k(s_current, k)
                    for res_poly, new_s in results:
                        new_idx = self.string_to_idx.get(tuple(new_s))
                        if new_idx is not None:
                            combined_poly = poly * res_poly

                            if new_idx in next_state:
                                next_state[new_idx] = next_state[new_idx] + combined_poly
                            else:
                                next_state[new_idx] = combined_poly
                current_state = next_state

            # Apply even generators
            for k in even_indices:
                next_state = {}
                for idx, poly in current_state.items():
                    s_current = self.basis_strings[idx]

                    results = self._apply_e_k(s_current, k)
                    for res_poly, new_s in results:
                        new_idx = self.string_to_idx.get(tuple(new_s))
                        if new_idx is not None:
                            combined_poly = poly * res_poly

                            if new_idx in next_state:
                                next_state[new_idx] = next_state[new_idx] + combined_poly
                            else:
                                next_state[new_idx] = combined_poly
                current_state = next_state

            # Populate column j of matrix
            for idx, poly in current_state.items():
                matrix[idx][j] = poly

        return matrix

    def polynomial_to_mathematica(self, poly):
        if poly.is_zero():
            return "0"

        parts = []
        # Sort by power descending
        for power in sorted(poly.coeffs.keys(), reverse=True):
            coeff = poly.coeffs[power]

            # Format term
            term = ""
            abs_coeff = abs(coeff)

            if power == 0:
                term = str(abs_coeff)
            elif power == 1:
                term = "n" if abs_coeff == 1 else f"{abs_coeff}*n"
            else:
                term = f"n^{power}" if abs_coeff == 1 else f"{abs_coeff}*n^{power}"

            # Sign handling
            if coeff > 0:
                if parts:
                    parts.append("+" + term)
                else:
                    parts.append(term)
            else:
                if parts:
                    parts.append("-" + term)
                else:
                    parts.append("-" + term)

        return "".join(parts)

    def matrix_to_mathematica(self, matrix):
        rows = []
        for i in range(self.dim):
            cols = []
            for j in range(self.dim):
                cols.append(self.polynomial_to_mathematica(matrix[i][j]))
            rows.append("{" + ", ".join(cols) + "}")
        return "{\n" + ",\n".join(rows) + "\n}"

    def basis_to_mathematica(self):
        rows = []
        for s in self.basis_strings:
            # Mathematica list: {1, 0, -1}
            s_str = "{" + ", ".join(map(str, s)) + "}"
            rows.append(s_str)
        return "{\n" + ",\n".join(rows) + "\n}"

    def evaluate_matrix(self, matrix, n_val):
        """Evaluate a symbolic matrix at n=n_val to get a numpy array."""
        evaluated = np.zeros((self.dim, self.dim), dtype=float)
        for i in range(self.dim):
            for j in range(self.dim):
                evaluated[i, j] = matrix[i][j].evaluate(n_val)
        return evaluated

    def analyze_structure(self, matrix, name, n_val=1.2345):
        """Analyze rank and sparsity of the matrix."""
        print(f"\n--- Analysis of {name} Matrix (L={self.L}) ---")

        # 1. Rank
        evaluated = self.evaluate_matrix(matrix, n_val)
        rank = np.linalg.matrix_rank(evaluated)
        print(f"Rank (at n={n_val:.4f}): {rank}")

        # 2. Non-zero entries (sparse format)
        non_zero_entries = []
        active_rows = set()
        active_cols = set()

        for i in range(self.dim):
            for j in range(self.dim):
                if not matrix[i][j].is_zero():
                    poly_str = self.polynomial_to_mathematica(matrix[i][j])
                    non_zero_entries.append((i, j, poly_str))
                    active_rows.add(i)
                    active_cols.add(j)

        print(f"Non-zero entries: {len(non_zero_entries)} (Sparsity: {len(non_zero_entries)/(self.dim**2)*100:.1f}%)")

        # 3. Active Vectors (Rows/Cols)
        print(f"Active Row Indices (Image support): {sorted(list(active_rows))}")
        print(f"Active Column Indices (Source support): {sorted(list(active_cols))}")

        # Print list of non-zero entries? If too many, maybe skip.
        # "keep of list of non-zero entries" - user asked for it.
        # Assuming for small L it's fine.
        print("\nNon-zero Entries List (row, col -> value):")
        for r, c, val in non_zero_entries:
            row_basis = self.basis_strings[r]
            col_basis = self.basis_strings[c]
            print(f"  ({r}, {c}) -> {val}")
            # print(f"    Basis[{r}] (Row): {row_basis}")
            # print(f"    Basis[{c}] (Col): {col_basis}")

def generate_symbolic_matrices(L_list=[2, 3]):
    out_dir = "../experiment_outputs" if os.path.basename(os.getcwd()) == "python_implementation" else "experiment_outputs"
    os.makedirs(out_dir, exist_ok=True)

    for L in L_list:
        print(f"\nGenerating symbolic matrices for L={L}...")
        builder = SymbolicHeckeBuilder(L=L)
        print(f"  Dimension: {builder.dim}")

        # Basis
        basis_str = builder.basis_to_mathematica()
        filename_basis = os.path.join(out_dir, f"basis_L{L}.m")
        with open(filename_basis, 'w') as f:
            f.write(f"(* Basis vectors for L={L} *)\n")
            f.write(f"basisL{L} = {basis_str};\n")
        print(f"  Saved basis to {filename_basis}")

        # H Matrix
        H_mat = builder.get_H_matrix()
        H_str = builder.matrix_to_mathematica(H_mat)
        filename_H = os.path.join(out_dir, f"H_matrix_L{L}.m")
        with open(filename_H, 'w') as f:
            f.write(f"(* Hamiltonian Matrix H for L={L} *)\n")
            f.write(f"HL{L} = {H_str};\n")
        print(f"  Saved H matrix to {filename_H}")

        # Analyze H
        builder.analyze_structure(H_mat, "Hamiltonian H", n_val=1.2345)

        # T Matrix
        T_mat = builder.get_T_matrix()
        T_str = builder.matrix_to_mathematica(T_mat)
        filename_T = os.path.join(out_dir, f"T_matrix_L{L}.m")
        with open(filename_T, 'w') as f:
            f.write(f"(* Transfer Matrix T for L={L} *)\n")
            f.write(f"TL{L} = {T_str};\n")
        print(f"  Saved T matrix to {filename_T}")

        # Analyze T
        builder.analyze_structure(T_mat, "Transfer Matrix T", n_val=1.2345)

def generate_magnetic_symbolic_matrices(S_list=[2, 3]):
    out_dir = "../experiment_outputs" if os.path.basename(os.getcwd()) == "python_implementation" else "experiment_outputs"
    os.makedirs(out_dir, exist_ok=True)

    for S in S_list:
        print(f"\nGenerating symbolic magnetic matrices for S={S}...")

        # 1. Entire Space
        builder_all = SymbolicHeckeBuilder(is_magnetic=True, m=S, use_all_valid=True)
        print(f"  Entire space dimension: {builder_all.dim}")

        if builder_all.dim > 0:
            basis_str = builder_all.basis_to_mathematica()
            filename_basis = os.path.join(out_dir, f"basis_magnetic_S{S}_all.m")
            with open(filename_basis, 'w') as f:
                f.write(f"(* Basis vectors for magnetic S={S} (entire) *)\n")
                f.write(f"basisMagS{S}All = {basis_str};\n")

            T_mat_all = builder_all.get_T_matrix()
            T_str_all = builder_all.matrix_to_mathematica(T_mat_all)
            filename_T_all = os.path.join(out_dir, f"T_matrix_magnetic_S{S}_all.m")
            with open(filename_T_all, 'w') as f:
                f.write(f"(* Transfer Matrix T for magnetic S={S} (entire) *)\n")
                f.write(f"TMagS{S}All = {T_str_all};\n")
            print(f"  Saved entire T matrix to {filename_T_all}")
            # Analyze T entire
            builder_all.analyze_structure(T_mat_all, f"Entire Transfer Matrix T (S={S})", n_val=1.2345)

        # 2. Block sectors
        endpoints = set()
        for string in builder_all.basis_strings:
            x = string.count(1) - string.count(0)
            y = string.count(0) - string.count(-1)
            endpoints.add((x, y))

        for (x, y) in sorted(list(endpoints)):
            builder_block = SymbolicHeckeBuilder(is_magnetic=True, m=S, x=x, y=y)
            print(f"  Block sector (x={x}, y={y}): dim = {builder_block.dim}")

            if builder_block.dim > 0:
                basis_str = builder_block.basis_to_mathematica()
                filename_basis = os.path.join(out_dir, f"basis_magnetic_S{S}_x{x}_y{y}.m")
                with open(filename_basis, 'w') as f:
                    f.write(f"(* Basis vectors for magnetic S={S}, x={x}, y={y} *)\n")
                    f.write(f"basisMagS{S}x{x}y{y} = {basis_str};\n")

                T_mat_block = builder_block.get_T_matrix()
                T_str_block = builder_block.matrix_to_mathematica(T_mat_block)
                filename_T_block = os.path.join(out_dir, f"T_matrix_magnetic_S{S}_x{x}_y{y}.m")
                with open(filename_T_block, 'w') as f:
                    f.write(f"(* Transfer Matrix T for magnetic S={S}, x={x}, y={y} *)\n")
                    f.write(f"TMagS{S}x{x}y{y} = {T_str_block};\n")
                print(f"    Saved block T matrix to {filename_T_block}")

if __name__ == "__main__":
    # Generate vacuum module matrices (L=2) to keep tests fast
    # generate_symbolic_matrices(L_list=[2])

    # Generate magnetic module matrices (S=3, 4)
    # S=3 dim=3, S=4 dim=9
    generate_magnetic_symbolic_matrices(S_list=[2])
