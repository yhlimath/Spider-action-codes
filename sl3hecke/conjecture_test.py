import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

S = 6
n_val = 1.372

from dilute_temperley_lieb.dtl_transfer_matrix import construct_dtl_transfer_matrix
import sympy

T_dtl, states_dtl = construct_dtl_transfer_matrix(S, 0)
q_val = (n_val + np.sqrt(n_val**2 - 4 + 0j)) / 2

dim_dtl = len(states_dtl)
T_dtl_num = np.zeros((dim_dtl, dim_dtl), dtype=complex)
q_sym = sympy.Symbol('q')
for r in range(dim_dtl):
    for c in range(dim_dtl):
        val = T_dtl[r][c]
        if val != 0:
            val_c = complex(val.subs(q_sym, q_val))
            T_dtl_num[r, c] = val_c

p_groups = {}
for i, s in enumerate(states_dtl):
    p = s.count(0)
    if p not in p_groups: p_groups[p] = []
    p_groups[p].append(i)

dtl_eigs_by_p = {}
for p, indices in p_groups.items():
    sub_T = T_dtl_num[np.ix_(indices, indices)]
    eigs = np.sort(np.real(np.linalg.eigvals(sub_T)))
    eigs = eigs[np.abs(eigs) > 1e-5]
    dtl_eigs_by_p[p] = np.sort(eigs)
    print(f"dTL p={p}: Eigenvalues {dtl_eigs_by_p[p]}")

print("=========================")

from sl3hecke.sl3_hecke import Sl3HeckeArnoldi
solver_all = Sl3HeckeArnoldi(L=None, n_value=n_val, is_magnetic=True, m=S, use_all_valid=True)
endpoints = set()
for string in solver_all.basis_strings:
    count_1 = string.count(1)
    count_0 = string.count(0)
    count_neg1 = string.count(-1)
    x = count_1 - count_0
    y = count_0 - count_neg1
    endpoints.add((x, y))

sl3_eigs_by_xy = {}
for (x, y) in sorted(list(endpoints)):
    solver_block = Sl3HeckeArnoldi(L=None, n_value=n_val, is_magnetic=True, m=S, x=x, y=y)
    dim_block = solver_block.dim
    if dim_block > 0:
        T_block = np.zeros((dim_block, dim_block), dtype=complex)
        for j in range(dim_block):
            v = np.zeros(dim_block, dtype=complex)
            v[j] = 1.0
            T_block[:, j] = solver_block.apply_T(v)

        eigs = np.sort(np.real(np.linalg.eigvals(T_block)))
        eigs = eigs[np.abs(eigs) > 1e-5]
        sl3_eigs_by_xy[(x,y)] = eigs
        print(f"SL3 Eigenvalues for (x,y)=({x},{y}): {eigs}")

print("=========================")
print("Matching...")
for p, dtl_eigs in dtl_eigs_by_p.items():
    matched_xys = []
    remaining_eigs = list(dtl_eigs)

    for xy, sl3_eigs in sl3_eigs_by_xy.items():
        is_subset = True
        temp_rem = list(remaining_eigs)
        for eig in sl3_eigs:
            if not temp_rem:
                is_subset = False; break
            diffs = np.abs(np.array(temp_rem) - eig)
            min_idx = np.argmin(diffs)
            if diffs[min_idx] < 1e-5:
                temp_rem.pop(min_idx)
            else:
                is_subset = False
                break

        if is_subset:
            matched_xys.append(xy)
            remaining_eigs = temp_rem

    if len(remaining_eigs) == 0 and len(dtl_eigs) > 0:
        print(f"  SUCCESS! F^-1(p={p}) = {matched_xys}")
    elif len(dtl_eigs) == 0:
        print(f"  SUCCESS! F^-1(p={p}) = [] (No non-zero eigs)")
    else:
        print(f"  FAILED to perfectly match p={p}. Leftover eigs: {remaining_eigs}, Matched: {matched_xys}")
