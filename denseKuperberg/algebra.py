import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from denseKuperberg.operations import homogenize_H_i, reverse_operations, apply_operations, branch_Y, merge_Y_inv
from sl3hecke.sl3_hecke import e, Polynomial

def map_path_to_sl3_string(path):
    """
    Extracts the J sequence from a homogeneous path (assuming all signs are the same).
    Returns a list of j values.
    """
    return [j for s, j in path]

def map_sl3_string_to_path(string, sign):
    """
    Combines a sign and a sequence of j values to form a path.
    """
    return [(sign, j) for j in string]

def action_E_i(coeff, path, i):
    """
    Applies the E_i operator on a path at index i (0-indexed).
    coeff is the current polynomial coefficient of the path.
    Returns a list of (new_coeff, new_path) pairs.
    """
    s_i = path[i][0]
    s_next = path[i+1][0]

    if s_i != s_next:
        return []

    # Homogenize the string
    homog_path, ops, new_i = homogenize_H_i(path, i)
    target_sign = s_i  # since s_i == s_next, the whole homog_path has sign s_i

    # Map to sl3 string
    sl3_string = map_path_to_sl3_string(homog_path)

    # Apply e_i from sl3hecke.
    # Note: e in sl3hecke takes 1-indexed i for generator, but it operates on S format: [(coeff, string)].
    # The argument to e is 1-indexed. e(S, k) where k is 1-indexed position.
    # Our new_i is 0-indexed, so we pass new_i + 1.
    sl3_results = e([(coeff, sl3_string)], new_i + 1)

    # Apply H_i^{-1} to the results
    final_results = []
    inv_ops = reverse_operations(ops)

    for res_coeff, res_string in sl3_results:
        res_path = map_sl3_string_to_path(res_string, target_sign)
        final_path = apply_operations(res_path, inv_ops)
        final_results.append((res_coeff, final_path))

    return final_results

def action_H_i(coeff, path, i):
    """
    Applies the H_i operator on a path at index i (0-indexed).
    coeff is the current polynomial coefficient.
    Returns a list of (new_coeff, new_path) pairs.
    """
    s_i = path[i][0]
    s_next = path[i+1][0]

    if s_i == s_next:
        return []

    # Homogenize the string. It will choose a target_sign (either 1 or -1) that minimizes length.
    homog_path, ops, new_i = homogenize_H_i(path, i)

    # What is the target sign? The entire homog_path EXCEPT i and i+1 has the target sign.
    # Wait, the homogenization leaves new_i and new_i+1 alone.
    # Let's inspect the sign of the first element not in {new_i, new_i+1} to find target_sign.
    # If the path is only length 2, then target_sign doesn't matter for the rest,
    # but we can deduce it from what we need.
    # Wait, homogenize_H_i already did the work, we just need to see which of {new_i, new_i+1}
    # has a different sign from target_sign.
    # Actually, we know exactly one of {new_i, new_i+1} has sign != target_sign.
    # Let's just find the target_sign by looking at the rest of the string, or default to 1 if length is 2.
    if len(homog_path) == 2:
        # If length is 2, the homogenization didn't do anything because left and right segments were empty.
        # It chose target_sign = 1 in our implementation when lengths are tied.
        target_sign = 1
    else:
        # Just pick an element that is not new_i or new_i+1
        if new_i > 0:
            target_sign = homog_path[0][0]
        else:
            target_sign = homog_path[new_i+2][0]

    # We branch k in {new_i, new_i+1} such that its sign becomes target_sign.
    # If an element has sign -target_sign, branching it makes its children have sign target_sign.
    if homog_path[new_i][0] == -target_sign:
        k = new_i
    else:
        k = new_i + 1

    # Now branch k
    path_branched = homog_path[:k] + branch_Y(homog_path[k][0], homog_path[k][1]) + homog_path[k+1:]

    # Now the path has length len(homog_path) + 1.
    # We act by E_{2i+1-k} ... wait, you said:
    # "branches on k in {i, i+1} with Y ... then act by E_{2i+1-k}, then merge the 2i+1-k and 2i+2-k steps"
    # Wait, the instruction is "act by E_{2i+1-k}". Let's trace the indices.
    # The original pair was at indices {new_i, new_i+1}.
    # If we branched k = new_i:
    # The new elements are at new_i and new_i+1. The other element (originally new_i+1) is now at new_i+2.
    # So the relevant elements to apply E are at {new_i+1, new_i+2}.
    # Let's check 2*new_i + 1 - new_i = new_i + 1. So it matches!
    # If we branched k = new_i+1:
    # The original new_i element is still at new_i. The branched elements are at new_i+1, new_i+2.
    # So the relevant elements are at {new_i, new_i+1}.
    # Let's check 2*new_i + 1 - (new_i+1) = new_i. So it matches!
    # Therefore, the generator index is indeed 2*new_i + 1 - k.
    e_idx = 2 * new_i + 1 - k

    # Map to sl3 string
    sl3_string = map_path_to_sl3_string(path_branched)

    # Apply e. It is 1-indexed.
    sl3_results = e([(coeff, sl3_string)], e_idx + 1)

    final_results = []
    inv_ops = reverse_operations(ops)

    for res_coeff, res_string in sl3_results:
        res_path = map_sl3_string_to_path(res_string, target_sign)

        # Merge the e_idx and e_idx+1 steps (which are the newly acted steps)
        # Wait, the instruction says "merge the 2i+1-k and 2i+1-k steps". Typo in user message?
        # User said "merge the 2i+1-k and 2i+1-k steps" initially, but later confirmed: "merge the 2i+1-k and 2i+2-k steps"
        try:
            merged_step = merge_Y_inv(res_path[e_idx], res_path[e_idx+1])
        except ValueError:
            # If merging is impossible, skip or error? "If merging is impossible... report an error."
            raise ValueError(f"Merging impossible on {res_path[e_idx]} and {res_path[e_idx+1]}")

        merged_path = res_path[:e_idx] + [merged_step] + res_path[e_idx+2:]

        # Finally, apply H_i^{-1}
        final_path = apply_operations(merged_path, inv_ops)
        final_results.append((res_coeff, final_path))

    return final_results
