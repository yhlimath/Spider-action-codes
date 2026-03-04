import numpy as np
import random
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from sl3_hecke import Sl3HeckeArnoldi

def estimate_T_rank(L, n_val=2, sample_ratio=0.05):
    print(f"Estimating Rank of Transfer Matrix T for L={L}")
    print(f"Using generic n={n_val} to avoid accidental zeroes.")
    print("=" * 60)

    start_t = time.time()
    solver = Sl3HeckeArnoldi(L=L, n_value=n_val)
    dim = solver.dim
    print(f"Full space dimension: {dim}")

    # 1. Sample index set I
    num_samples = max(1, int(dim * sample_ratio))
    print(f"Sampling {num_samples} vectors (ratio: {sample_ratio}) to find active subspace...")

    # Use fixed seed for reproducibility across small runs, or just sample
    # Let's try multiple samples or a larger ratio to be safe on the estimate
    I = random.sample(range(dim), num_samples)

    # 2. Find active index set J
    J = set()
    for i in I:
        v = np.zeros(dim, dtype=complex)
        v[i] = 1.0
        w = solver.apply_T(v)
        for idx in range(dim):
            if abs(w[idx]) > 1e-12:
                J.add(idx)

    J = sorted(list(J))
    size_J = len(J)
    print(f"Found active set J of size: {size_J}")

    if size_J == 0:
        print("Rank estimate: 0")
        return 0

    # 3. Construct restricted matrix T_{J x J}
    print(f"Constructing restricted matrix of size {size_J} x {size_J}...")
    T_restricted = np.zeros((size_J, size_J), dtype=complex)
    j_to_restricted_idx = {orig_idx: new_idx for new_idx, orig_idx in enumerate(J)}

    for restricted_col_idx, orig_col_idx in enumerate(J):
        v = np.zeros(dim, dtype=complex)
        v[orig_col_idx] = 1.0
        w = solver.apply_T(v)
        for orig_row_idx in J:
            val = w[orig_row_idx]
            if abs(val) > 1e-12:
                restricted_row_idx = j_to_restricted_idx[orig_row_idx]
                T_restricted[restricted_row_idx, restricted_col_idx] = val

    # 4. Compute rank
    rank = np.linalg.matrix_rank(T_restricted, tol=1e-10)

    # Let's also check rank of full matrix for L=4 to be absolutely sure the estimation is good
    # For L=4, dim is 462, building full matrix is fast.
    if L <= 4:
        print("\nVerifying with full matrix construction (since L is small)...")
        T_full = np.zeros((dim, dim), dtype=complex)
        for j in range(dim):
            v = np.zeros(dim, dtype=complex)
            v[j] = 1.0
            w = solver.apply_T(v)
            T_full[:, j] = w
        full_rank = np.linalg.matrix_rank(T_full, tol=1e-10)
        print(f"Actual Rank of T (from full matrix): {full_rank}")

    elapsed = time.time() - start_t
    print(f"\nEstimated Rank of T: {rank}")
    print(f"Time elapsed: {elapsed:.2f}s")
    print("=" * 60)

    return rank

if __name__ == "__main__":
    estimate_T_rank(L=7, n_val=3.14, sample_ratio=0.0005)
