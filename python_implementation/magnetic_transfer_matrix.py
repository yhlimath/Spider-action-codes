
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from sl3_hecke import Sl3HeckeArnoldi

def analyze_blocks(S_list=[4, 5], n_val=1.2345):
    print("Transfer Matrix Block Analysis")
    print("=" * 60)

    out_dir = "experiment_outputs"
    os.makedirs(out_dir, exist_ok=True)

    for S in S_list:
        print(f"\nAnalyzing S = {S}")

        # 1. Analyze "entire fashion"
        solver_all = Sl3HeckeArnoldi(L=None, n_value=n_val, is_magnetic=True, m=S, use_all_valid=True)
        dim_all = solver_all.dim
        print(f"  Entire space dimension: {dim_all}")

        # Generate full matrix for small S to verify block diagonality
        if dim_all <= 100:
            T_all = np.zeros((dim_all, dim_all), dtype=complex)
            for j in range(dim_all):
                v = np.zeros(dim_all, dtype=complex)
                v[j] = 1.0
                T_all[:, j] = solver_all.apply_T(v)

            #np.save(os.path.join(out_dir, f"T_all_S{S}.npy"), T_all)
            #print(f"  Saved full T matrix to T_all_S{S}.npy")

        # 2. Find all valid (x, y) endpoints
        endpoints = set()
        for string in solver_all.basis_strings:
            count_1 = string.count(1)
            count_0 = string.count(0)
            count_neg1 = string.count(-1)
            x = count_1 - count_0
            y = count_0 - count_neg1
            endpoints.add((x, y))

        print(f"  Found {len(endpoints)} unique endpoint sectors (x,y): {endpoints}")

        # 3. Analyze each block
        total_dim_blocks = 0
        for (x, y) in sorted(list(endpoints)):
            solver_block = Sl3HeckeArnoldi(L=None, n_value=n_val, is_magnetic=True, m=S, x=x, y=y)
            dim_block = solver_block.dim
            total_dim_blocks += dim_block
            print(f"    Sector ({x},{y}): dim = {dim_block}")

            
            T_block = np.zeros((dim_block, dim_block), dtype=complex)
            for j in range(dim_block):
                v = np.zeros(dim_block, dtype=complex)
                v[j] = 1.0
                T_block[:, j] = solver_block.apply_T(v)

            # Check rank
            rank = np.linalg.matrix_rank(T_block, tol=1e-10)
            print(f"      Rank: {rank}")
            #np.save(os.path.join(out_dir, f"T_block_S{S}_x{x}_y{y}.npy"), T_block)

        print(f"  Sum of block dimensions: {total_dim_blocks} (Matches entire space: {total_dim_blocks == dim_all})")

if __name__ == "__main__":
    analyze_blocks(S_list=[12], n_val=2)
