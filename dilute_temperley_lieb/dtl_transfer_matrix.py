import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dtl_algebra import generate_dtl_states, apply_E_i
import sympy

def construct_dtl_transfer_matrix(L, j):
    """
    Constructs the exact symbolic Transfer Matrix T for a given module V^{L,j}.
    T = E_odd_product * E_even_product
      = (E_1 E_3 E_5 ...) (E_2 E_4 E_6 ...)  (using 1-based indices from user)
      = (E_0 E_2 E_4 ...) (E_1 E_3 E_5 ...)  (in our 0-based Python indexing)

    Since E operators on non-overlapping sites commute, we can just apply them sequentially.
    The formula says: "apply the oddly indexed operators and then the evenly indexed ones"
    Wait, in 1-based indexing, odd is 1, 3, 5 -> Python 0, 2, 4.
    Even is 2, 4, 6 -> Python 1, 3, 5.
    If T |v> = E_1 E_3 ... E_2 E_4 ... |v>, this means we act on |v> with E_even first, then E_odd.
    """
    states = generate_dtl_states(L, j)
    dim = len(states)

    state_to_idx = {s: i for i, s in enumerate(states)}

    q = sympy.Symbol('q')

    # Initialize T as a zero matrix
    T_matrix = [[0 for _ in range(dim)] for _ in range(dim)]

    # We want to compute T |state_k> for each basis state
    for k, start_state in enumerate(states):
        # We start with state_k having weight 1
        current_superposition = {start_state: 1}

        # 1. Apply even 1-based operators -> odd Python indices: 1, 3, 5, ...
        odd_python_indices = list(range(1, L - 1, 2))
        for i in odd_python_indices:
            next_superposition = {}
            for state, coeff in current_superposition.items():
                res = apply_E_i(state, i, q)
                for new_state, weight in res:
                    if new_state in next_superposition:
                        next_superposition[new_state] += coeff * weight
                    else:
                        next_superposition[new_state] = coeff * weight
            current_superposition = next_superposition

        # 2. Apply odd 1-based operators -> even Python indices: 0, 2, 4, ...
        even_python_indices = list(range(0, L - 1, 2))
        for i in even_python_indices:
            next_superposition = {}
            for state, coeff in current_superposition.items():
                res = apply_E_i(state, i, q)
                for new_state, weight in res:
                    if new_state in next_superposition:
                        next_superposition[new_state] += coeff * weight
                    else:
                        next_superposition[new_state] = coeff * weight
            current_superposition = next_superposition

        # Now current_superposition holds T |start_state>
        for final_state, coeff in current_superposition.items():
            if coeff != 0:
                row_idx = state_to_idx[final_state]
                T_matrix[row_idx][k] = sympy.simplify(coeff)

    return T_matrix, states


def analyze_transfer_matrix(L, j):
    T, states = construct_dtl_transfer_matrix(L, j)
    dim = len(states)

    print(f"==================================================")
    print(f"Analysis of Transfer Matrix T for L={L}, j={j}")
    print(f"Dimension: {dim}")
    print(f"==================================================\n")

    print("Basis states:")
    for i, s in enumerate(states):
        print(f"  |{i}> = {s}")
    print("\nMatrix T (columns are T|k>):")

    # Calculate symbolic rank using sympy Matrix
    sympy_T = sympy.Matrix(T)
    rank = sympy_T.rank()
    print(f"\nExact symbolic rank of T: {rank}\n")

    # Find invariant subspaces (connected components)
    # We create a directed graph where edge k -> i exists if T_{i,k} != 0
    # A vector |k> mixes into |i> under action of T.
    # To find disconnected blocks, we look at the weakly connected components
    # of this graph.
    adj = [[] for _ in range(dim)]
    for i in range(dim):
        for k in range(dim):
            if T[i][k] != 0:
                adj[i].append(k)
                adj[k].append(i) # treat as undirected for block identification

    visited = [False] * dim
    blocks = []

    for i in range(dim):
        if not visited[i]:
            comp = []
            queue = [i]
            visited[i] = True
            while queue:
                curr = queue.pop(0)
                comp.append(curr)
                for neighbor in adj[curr]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)
            blocks.append(sorted(comp))

    print(f"Invariant Subspaces (Blocks): {len(blocks)}")
    for b_idx, block in enumerate(blocks):
        print(f"  Block {b_idx + 1}:")
        for state_idx in block:
            print(f"    |{state_idx}> = {states[state_idx]}")

    # Explicit matrix and Mathematica formatting
    print("\nExplicit Matrix Elements:")
    m_rows = []
    json_rows = []
    for r in range(dim):
        row_str = []
        m_row = []
        for c in range(dim):
            val = T[r][c]
            if val == 0:
                row_str.append("0")
                m_row.append("0")
            else:
                val_str = str(val).replace("**", "^") # Mathematica friendly powers
                row_str.append(str(val))
                m_row.append(val_str)
        print("  [" + ", ".join(row_str) + "]")
        m_rows.append("{" + ", ".join(m_row) + "}")
        json_rows.append(row_str)

    m_matrix_str = "{" + ", \n".join(m_rows) + "}"

    # Save to files
    import os
    import json
    os.makedirs("experiment_outputs", exist_ok=True)

    # JSON output
    out_json = f"experiment_outputs/transfer_matrix_L{L}_j{j}.json"
    json_data = {
        "L": L,
        "j": j,
        "dimension": dim,
        "rank": int(rank),
        "basis": [str(s) for s in states],
        "invariant_subspaces": [[str(states[idx]) for idx in block] for block in blocks],
        "matrix": json_rows
    }
    with open(out_json, "w") as f:
        json.dump(json_data, f, indent=2)

    # Mathematica output
    out_m = f"experiment_outputs/transfer_matrix_L{L}_j{j}.m"
    with open(out_m, "w") as f:
        f.write(f"(* Dilute Temperley-Lieb Transfer Matrix for L={L}, j={j} *)\n")
        f.write(f"TMatrix = {m_matrix_str};\n")
        f.write(f"TRank = {rank};\n")

    print(f"\nSaved analysis outputs to {out_json} and {out_m}")

if __name__ == "__main__":
    L=6; j=3
    if len(sys.argv) > 2:
        L = int(sys.argv[1])
        j = int(sys.argv[2])
    analyze_transfer_matrix(L, j)
