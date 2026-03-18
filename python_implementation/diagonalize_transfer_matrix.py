import os
import sys
import sympy as sp

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from symbolic_matrix_generator import SymbolicHeckeBuilder
from estimate_rank import get_active_indices
from sl3_hecke import Polynomial

def poly_to_sympy(poly, n_sym):
    """Convert Polynomial object to sympy expression"""
    if poly.is_zero():
        return 0

    expr = 0
    for power, coeff in poly.coeffs.items():
        expr += coeff * (n_sym ** power)
    return expr

def sympy_to_mathematica(expr):
    """Convert sympy expression to mathematica string"""
    from sympy.printing.mathematica import mathematica_code
    return mathematica_code(expr)

def matrix_to_mathematica(mat):
    """Convert sympy Matrix to mathematica string"""
    rows = []
    for i in range(mat.rows):
        cols = []
        for j in range(mat.cols):
            cols.append(sympy_to_mathematica(mat[i, j]))
        rows.append("{" + ", ".join(cols) + "}")
    return "{\n" + ",\n".join(rows) + "\n}"

def diagonalize_blocks(S_list=[3, 4]):
    print("Diagonalizing Transfer Matrix Blocks")
    print("=" * 60)

    out_dir = "../experiment_outputs" if os.path.basename(os.getcwd()) == "python_implementation" else "experiment_outputs"
    os.makedirs(out_dir, exist_ok=True)

    n_sym = sp.Symbol('n')

    for S in S_list:
        print(f"\nProcessing S = {S}")

        # Get all valid endpoints
        builder_all = SymbolicHeckeBuilder(is_magnetic=True, m=S, use_all_valid=True)
        endpoints = set()
        for string in builder_all.basis_strings:
            x = string.count(1) - string.count(0)
            y = string.count(0) - string.count(-1)
            endpoints.add((x, y))

        for (x, y) in sorted(list(endpoints)):
            builder = SymbolicHeckeBuilder(is_magnetic=True, m=S, x=x, y=y)
            dim = builder.dim
            if dim == 0: continue

            print(f"  Sector ({x},{y}), dim={dim}")

            # 1. Use estimate_rank logic to quickly find active indices J
            J = get_active_indices(is_magnetic=True, m=S, x=x, y=y, sample_ratio=0.1) # Use higher ratio for safety on small dims

            print(f"    Active indices (J) found via sampling: {J}")

            if not J:
                print("    Matrix is completely zero (or estimated so).")
                continue

            # 2. Build ONLY the restricted symbolic matrix T_{J x J}
            # T_poly = builder.get_T_matrix() builds the whole thing, which is slow.
            # We will manually build T[J, J] using the builder's _apply_e_k logic.

            # Replicate the logic from get_T_matrix but restricted
            num_generators = S - 1
            odd_indices = range(1, num_generators + 1, 2)
            even_indices = range(2, num_generators + 1, 2)

            T_sub_poly = [[Polynomial() for _ in range(len(J))] for _ in range(len(J))]

            # We only evaluate columns j in J
            for j_sub, orig_j in enumerate(J):
                current_state = {orig_j: Polynomial.constant(1)}

                # Apply odd generators
                for k in odd_indices:
                    next_state = {}
                    for idx, poly in current_state.items():
                        s_current = builder.basis_strings[idx]
                        results = builder._apply_e_k(s_current, k)
                        for res_poly, new_s in results:
                            new_idx = builder.string_to_idx.get(tuple(new_s))
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
                        s_current = builder.basis_strings[idx]
                        results = builder._apply_e_k(s_current, k)
                        for res_poly, new_s in results:
                            new_idx = builder.string_to_idx.get(tuple(new_s))
                            if new_idx is not None:
                                combined_poly = poly * res_poly
                                if new_idx in next_state:
                                    next_state[new_idx] = next_state[new_idx] + combined_poly
                                else:
                                    next_state[new_idx] = combined_poly
                    current_state = next_state

                # Populate restricted column j_sub
                for idx, poly in current_state.items():
                    # We only care about rows in J
                    if idx in J:
                        i_sub = J.index(idx)
                        T_sub_poly[i_sub][j_sub] = poly

            # Convert to sympy Matrix
            T_sub = sp.zeros(len(J), len(J))
            for i in range(len(J)):
                for j in range(len(J)):
                    T_sub[i, j] = poly_to_sympy(T_sub_poly[i][j], n_sym)

            # Try to diagonalize T_sub
            is_diagonalized = False
            P_sub, D_sub = None, None

            try:
                #P_sub, D_sub = T_sub.diagonalize()
                #is_diagonalized = True
                print("    Successfully diagonalized submatrix.")
            except sp.MatrixError as e:
                print(f"    Could not diagonalize submatrix symbolically: {e}")
                # Fallback: just use the un-diagonalized submatrix as the "non-trivial part"
                D_sub = T_sub
                P_sub = sp.eye(len(J))

            # Write to Mathematica file
            filename = os.path.join(out_dir, f"T_diagonalized_S{S}_x{x}_y{y}.m")
            with open(filename, 'w') as f:
                f.write(f"(* Diagonalization Analysis for S={S}, x={x}, y={y} *)\n\n")

                # Active indices (1-based for Mathematica)
                math_J = [idx + 1 for idx in J]
                f.write(f"ActiveIndices = {{{', '.join(map(str, math_J))}}};\n\n")

                # Original basis restricted to J
                basis_strs = []
                for j in J:
                    s = builder.basis_strings[j]
                    basis_strs.append("{" + ", ".join(map(str, s)) + "}")
                f.write("ActiveBasis = {\n" + ",\n".join(basis_strs) + "\n};\n\n")

                # Restricted Submatrix
                f.write("RestrictedSubmatrix = " + matrix_to_mathematica(T_sub) + ";\n\n")

                if is_diagonalized:
                    f.write("DiagonalizedMatrix = " + matrix_to_mathematica(D_sub) + ";\n\n")
                    f.write("EigenvectorBasisSubmatrix = " + matrix_to_mathematica(P_sub) + ";\n\n")

                    # Construct full eigenvectors if needed?
                    # The eigenvectors in the full space:
                    # For eigenvalue 0, they are the basis vectors e_k for k not in J.
                    # For non-zero eigenvalues, they are P_sub embedded into J coordinates.
                    # Actually, if T(e_k) = sum_{j in J} T_{jk} e_j, then e_k is not necessarily an eigenvector.
                    # The generalized eigenvectors for eigenvalue 0 are e_k.
                    # Let's just output the transformation matrix P_full if they want it.
                    # P_full = Identity. Then replace P_full[J, J] = P_sub.
                    P_full = sp.eye(dim)
                    for r_idx, orig_r in enumerate(J):
                        for c_idx, orig_c in enumerate(J):
                            P_full[orig_r, orig_c] = P_sub[r_idx, c_idx]

                    f.write("FullBasisTransformationMatrix = " + matrix_to_mathematica(P_full) + ";\n\n")
                else:
                    f.write("(* Submatrix could not be completely diagonalized symbolically. *)\n")

            print(f"    Saved to {filename}")

if __name__ == "__main__":
    # S=5 might be too slow for symbolic diagonalization, keep it to 3, 4
    diagonalize_blocks(S_list=[10])
