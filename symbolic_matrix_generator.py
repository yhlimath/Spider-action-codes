
import numpy as np
import copy
from sl3_hecke import Polynomial, e, generate_all_valid_strings

class SymbolicHeckeBuilder:
    def __init__(self, L):
        self.L = L
        self.basis_strings = generate_all_valid_strings(L)
        self.dim = len(self.basis_strings)
        self.string_to_idx = {tuple(s): i for i, s in enumerate(self.basis_strings)}

    def _apply_e_k(self, s, k):
        """Apply e_k symbolically to a basis string s."""
        # e(S, k) takes list of (coeff, string)
        # Here input is (1, s)
        results = e([(Polynomial.constant(1), s)], k)
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

        num_generators = 3 * self.L - 1
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

def generate_symbolic_matrices(L_list=[2, 3, 4]):
    for L in L_list:
        print(f"\nGenerating symbolic matrices for L={L}...")
        builder = SymbolicHeckeBuilder(L)
        print(f"  Dimension: {builder.dim}")

        # Basis
        basis_str = builder.basis_to_mathematica()
        filename_basis = f"basis_L{L}.m"
        with open(filename_basis, 'w') as f:
            f.write(f"(* Basis vectors for L={L} *)\n")
            f.write(f"basisL{L} = {basis_str};\n")
        print(f"  Saved basis to {filename_basis}")

        # H Matrix
        H_mat = builder.get_H_matrix()
        H_str = builder.matrix_to_mathematica(H_mat)
        filename_H = f"H_matrix_L{L}.m"
        with open(filename_H, 'w') as f:
            f.write(f"(* Hamiltonian Matrix H for L={L} *)\n")
            f.write(f"HL{L} = {H_str};\n")
        print(f"  Saved H matrix to {filename_H}")

        # Analyze H
        builder.analyze_structure(H_mat, "Hamiltonian H", n_val=1.2345)

        # T Matrix
        T_mat = builder.get_T_matrix()
        T_str = builder.matrix_to_mathematica(T_mat)
        filename_T = f"T_matrix_L{L}.m"
        with open(filename_T, 'w') as f:
            f.write(f"(* Transfer Matrix T for L={L} *)\n")
            f.write(f"TL{L} = {T_str};\n")
        print(f"  Saved T matrix to {filename_T}")

        # Analyze T
        builder.analyze_structure(T_mat, "Transfer Matrix T", n_val=1.2345)

if __name__ == "__main__":
    generate_symbolic_matrices()
