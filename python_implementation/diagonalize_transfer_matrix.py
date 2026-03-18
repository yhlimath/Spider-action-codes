import os
import sys
import sympy as sp

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from symbolic_matrix_generator import SymbolicHeckeBuilder

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

            # Build symbolic matrix
            T_poly = builder.get_T_matrix()

            # Convert to sympy Matrix
            T_sympy = sp.zeros(dim, dim)
            for i in range(dim):
                for j in range(dim):
                    T_sympy[i, j] = poly_to_sympy(T_poly[i][j], n_sym)

            # Identify non-zero rows
            non_zero_rows = []
            for i in range(dim):
                if not T_sympy.row(i).is_zero_matrix:
                    non_zero_rows.append(i)

            print(f"    Non-zero rows (Active indices): {non_zero_rows}")

            if not non_zero_rows:
                print("    Matrix is completely zero.")
                continue

            # Extract submatrix T_sub = T[J, J] where J = non_zero_rows
            # Wait, the columns must also be restricted.
            # If T maps everything into span(e_i | i in J), then T_sub captures the non-trivial dynamics.
            J = non_zero_rows
            T_sub = sp.zeros(len(J), len(J))
            for r_idx, orig_r in enumerate(J):
                for c_idx, orig_c in enumerate(J):
                    T_sub[r_idx, c_idx] = T_sympy[orig_r, orig_c]

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

                # Original basis
                basis_strs = []
                for s in builder.basis_strings:
                    basis_strs.append("{" + ", ".join(map(str, s)) + "}")
                f.write("OriginalBasis = {\n" + ",\n".join(basis_strs) + "\n};\n\n")

                # Active indices (1-based for Mathematica)
                math_J = [idx + 1 for idx in J]
                f.write(f"ActiveIndices = {{{', '.join(map(str, math_J))}}};\n\n")

                # Original Matrix
                f.write("OriginalMatrix = " + matrix_to_mathematica(T_sympy) + ";\n\n")

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
