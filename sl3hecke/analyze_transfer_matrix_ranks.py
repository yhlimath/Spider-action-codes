import os
import sys
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from sl3_hecke import Sl3HeckeArnoldi
from estimate_rank import estimate_T_rank

def analyze_magnetic_ranks(S_list=[3, 4, 5, 6, 7], n_val=1.2345):
    print("Analyzing Transfer Matrix Ranks for Magnetic Modules")
    print("=" * 60)

    out_dir = "../experiment_outputs" if os.path.basename(os.getcwd()) == "python_implementation" else "experiment_outputs"
    os.makedirs(out_dir, exist_ok=True)

    results = {}

    for S in S_list:
        print(f"\nProcessing S = {S}")

        # We need to find all valid endpoints (x,y) for length S.
        # We can use Sl3HeckeArnoldi with use_all_valid=True to generate the basis
        # and then extract unique endpoints.
        solver_all = Sl3HeckeArnoldi(L=None, n_value=n_val, is_magnetic=True, m=S, use_all_valid=True)
        if solver_all.dim == 0:
            print(f"  No valid strings for S={S}.")
            continue

        endpoints = set()
        for string in solver_all.basis_strings:
            x = string.count(1) - string.count(0)
            y = string.count(0) - string.count(-1)
            endpoints.add((x, y))

        print(f"  Found {len(endpoints)} unique endpoint sectors (x,y): {endpoints}")

        S_results = {
            "entire_dim": solver_all.dim,
            "sectors": {}
        }

        total_rank = 0
        total_active_subspace = 0

        for (x, y) in sorted(list(endpoints)):
            # estimate_T_rank returns (rank, J_active_indices)
            rank, J = estimate_T_rank(n_val=n_val, is_magnetic=True, m=S, x=x, y=y, sample_ratio=0.1)

            # Dimension of this block
            solver_block = Sl3HeckeArnoldi(L=None, n_value=n_val, is_magnetic=True, m=S, x=x, y=y)
            dim_block = solver_block.dim

            active_size = len(J)

            S_results["sectors"][f"{x},{y}"] = {
                "dim": dim_block,
                "rank": int(rank),
                "active_subspace_size": active_size
            }

            print(f"    Sector ({x},{y}): dim = {dim_block}, active size = {active_size}, Rank = {rank}")

            total_rank += rank
            total_active_subspace += active_size

        S_results["total_rank"] = int(total_rank)
        S_results["total_active_subspace"] = int(total_active_subspace)

        results[str(S)] = S_results

        print(f"  Summary for S={S}:")
        print(f"    Entire space dim: {solver_all.dim}")
        print(f"    Sum of active subspace sizes: {total_active_subspace}")
        print(f"    Sum of block ranks: {total_rank}")

    # Save results
    filename = os.path.join(out_dir, "magnetic_transfer_matrix_ranks.json")
    with open(filename, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"\nSaved rank analysis results to '{filename}'")

if __name__ == "__main__":
    analyze_magnetic_ranks(S_list=[3, 4, 5, 6, 7])
