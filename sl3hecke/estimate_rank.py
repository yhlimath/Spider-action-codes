import numpy as np
import random
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from sl3_hecke import Sl3HeckeArnoldi

def estimate_T_rank(L=None, n_val=1.2345, sample_ratio=0.5, is_magnetic=False, m=None, x=None, y=None, use_all_valid=False):
    if is_magnetic:
        if use_all_valid:
            print(f"Estimating Rank of Transfer Matrix T for magnetic entire space (S={m})")
        else:
            print(f"Estimating Rank of Transfer Matrix T for magnetic sector S={m}, x={x}, y={y}")
    else:
        print(f"Estimating Rank of Transfer Matrix T for vacuum L={L}")
    print(f"Using generic n={n_val} to avoid accidental zeroes.")
    print("=" * 60)

    start_t = time.time()
    solver = Sl3HeckeArnoldi(L=L, n_value=n_val, is_magnetic=is_magnetic, m=m, x=x, y=y, use_all_valid=use_all_valid)
    dim = solver.dim
    print(f"Full space dimension: {dim}")

    if dim == 0:
        print("Rank estimate: 0 (Dimension is 0)")
        return 0, []

    # 1. Sample index set I
    # Increase sample ratio to 0.5 or minimum 5 vectors for robust estimation.
    num_samples = max(min(dim, 5), int(dim * sample_ratio))
    print(f"Sampling {num_samples} vectors to find active subspace...")

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
        return 0, []

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

    # Only verify full matrix if dim is reasonably small
    if dim <= 500:
        print("\nVerifying with full matrix construction (since dim is small)...")
        T_full = np.zeros((dim, dim), dtype=complex)
        for j in range(dim):
            v = np.zeros(dim, dtype=complex)
            v[j] = 1.0
            w = solver.apply_T(v)
            T_full[:, j] = w
        full_rank = np.linalg.matrix_rank(T_full, tol=1e-10)
        print(f"Actual Rank of T (from full matrix): {full_rank}")
        if rank != full_rank:
            print(f"WARNING: Estimated rank ({rank}) differs from actual ({full_rank}). Try increasing sample_ratio.")
            # If we missed it, return the actual rank for logging correctness in this small dim case.
            rank = full_rank

    elapsed = time.time() - start_t
    print(f"\nEstimated Rank of T: {rank}")
    print(f"Time elapsed: {elapsed:.2f}s")
    print("=" * 60)

    return rank, J

def get_active_indices(L=None, n_val=1.2345, sample_ratio=0.5, is_magnetic=False, m=None, x=None, y=None, use_all_valid=False):
    """Utility function to just get J efficiently without full printing"""
    solver = Sl3HeckeArnoldi(L=L, n_value=n_val, is_magnetic=is_magnetic, m=m, x=x, y=y, use_all_valid=use_all_valid)
    dim = solver.dim
    if dim == 0: return []
    num_samples = max(min(dim, 5), int(dim * sample_ratio))
    I = random.sample(range(dim), num_samples)
    J = set()
    for i in I:
        v = np.zeros(dim, dtype=complex)
        v[i] = 1.0
        w = solver.apply_T(v)
        for idx in range(dim):
            if abs(w[idx]) > 1e-12:
                J.add(idx)
    return sorted(list(J))

if __name__ == "__main__":
    print("--- Vacuum Sector ---")
    estimate_T_rank(L=4, sample_ratio=0.5)

    print("\n--- Magnetic Sector (Entire Space S=4) ---")
    estimate_T_rank(is_magnetic=True, m=4, use_all_valid=True)

    print("\n--- Magnetic Sector (Block S=5, x=0, y=1) ---")
    estimate_T_rank(is_magnetic=True, m=5, x=0, y=1)
