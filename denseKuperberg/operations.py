def branch_Y(s, j):
    """
    Branches u_{s, j} into u_{-s, j'} and u_{-s, j''} where j' > j'' and j = j' + j''.
    Returns a list of two new steps.
    """
    if j == 1:
        # 1 = 1 + 0, and 1 > 0
        jp, jpp = 1, 0
    elif j == 0:
        # 0 = 1 + (-1), and 1 > -1
        jp, jpp = 1, -1
    elif j == -1:
        # -1 = 0 + (-1), and 0 > -1
        jp, jpp = 0, -1
    else:
        raise ValueError(f"Invalid j for branching: {j}")

    return [(-s, jp), (-s, jpp)]

def merge_Y_inv(step1, step2):
    """
    Merges u_{-s, j'} and u_{-s, j''} to u_{s, j' + j''}.
    Assumes step1 = (-s, j') and step2 = (-s, j'').
    Requires j' > j''.
    """
    s1, jp = step1
    s2, jpp = step2

    if s1 != s2:
        raise ValueError(f"Cannot merge steps with different signs: {step1}, {step2}")

    if jp <= jpp:
        raise ValueError(f"Cannot merge steps unless j' > j'': {step1}, {step2}")

    j = jp + jpp
    if j not in [1, 0, -1]:
        raise ValueError(f"Invalid merged j value: {j} from {jp} + {jpp}")

    s = -s1
    return (s, j)


def homogenize_segment(segment, target_sign):
    """
    Homogenizes a segment of a path to the target sign, returning the minimal possible length segment,
    and the sequence of operations applied.
    Operations are stored as a list of tuples:
    ('merge', index) meaning at `index` we merge segment[index] and segment[index+1].
    ('branch', index) meaning at `index` we branch segment[index].
    Note: The returned segment only contains the modified segment.
    """
    # We want everything in segment to have target_sign.
    # Elements that already have target_sign are kept as is.
    # Elements with -target_sign:
    # 1. We greedily try to merge adjacent (-target_sign, j') and (-target_sign, j'') if j' > j''.
    # 2. Any remaining (-target_sign, j) elements are branched.

    current_segment = segment[:]
    operations = []

    # We apply operations iteratively.
    # First, greedy merging of adjacent -target_sign elements.
    changed = True
    while changed:
        changed = False
        for k in range(len(current_segment) - 1):
            s1, j1 = current_segment[k]
            s2, j2 = current_segment[k+1]
            if s1 == -target_sign and s2 == -target_sign and j1 > j2:
                # Merge!
                merged_step = merge_Y_inv(current_segment[k], current_segment[k+1])
                current_segment = current_segment[:k] + [merged_step] + current_segment[k+2:]
                operations.append(('merge', k))
                changed = True
                break # Restart loop since indices changed

    # Next, branch any remaining -target_sign elements.
    # To keep operations list predictable for reversal, we branch from left to right.
    # Actually, we should branch from right to left so indices of earlier elements aren't affected.
    # But it's fine if we just apply and track. Let's do left to right and track carefully.
    k = 0
    while k < len(current_segment):
        s, j = current_segment[k]
        if s == -target_sign:
            branched_steps = branch_Y(s, j)
            current_segment = current_segment[:k] + branched_steps + current_segment[k+1:]
            operations.append(('branch', k))
            k += 2 # Skip the two newly added elements
        else:
            k += 1

    return current_segment, operations

def reverse_operations(operations):
    """
    Given a list of operations, returns the reverse list of inverse operations.
    If we did 'merge' at k, the inverse is 'branch' at k.
    If we did 'branch' at k, the inverse is 'merge' at k.
    Reversing the order is necessary to undo correctly.
    """
    inv_ops = []
    for op, k in reversed(operations):
        if op == 'merge':
            inv_ops.append(('branch', k))
        elif op == 'branch':
            inv_ops.append(('merge', k))
    return inv_ops

def apply_operations(path, operations):
    """
    Applies a list of operations to a path.
    """
    current_path = path[:]
    for op, k in operations:
        if op == 'merge':
            merged = merge_Y_inv(current_path[k], current_path[k+1])
            current_path = current_path[:k] + [merged] + current_path[k+2:]
        elif op == 'branch':
            s, j = current_path[k]
            branched = branch_Y(s, j)
            current_path = current_path[:k] + branched + current_path[k+1:]
    return current_path

def homogenize_H_i(path, i):
    """
    Applies the sign homogenization map H_i to the path around index i.
    i is 0-indexed here.
    If s_i == s_{i+1}: homogenize all steps except i and i+1 to s_i, minimal length.
    If s_i != s_{i+1}: homogenize all steps except i and i+1 to either + or - (whichever is shorter).
    Returns (new_path, operations_list, shifted_i)
    """
    s_i, j_i = path[i]
    s_next, j_next = path[i+1]

    left_segment = path[:i]
    right_segment = path[i+2:]

    def try_homogenize(target_sign):
        new_left, left_ops = homogenize_segment(left_segment, target_sign)
        # Shift operations for right segment by len(new_left) + 2
        new_right, right_ops_local = homogenize_segment(right_segment, target_sign)

        # Combine
        combined_path = new_left + [path[i], path[i+1]] + new_right

        shift_offset = len(new_left) + 2
        right_ops = [(op, k + shift_offset) for op, k in right_ops_local]

        operations = left_ops + right_ops
        return combined_path, operations, len(new_left)

    if s_i == s_next:
        return try_homogenize(s_i)
    else:
        # Try both + and -
        path_pos, ops_pos, shift_pos = try_homogenize(1)
        path_neg, ops_neg, shift_neg = try_homogenize(-1)

        if len(path_pos) <= len(path_neg):
            return path_pos, ops_pos, shift_pos
        else:
            return path_neg, ops_neg, shift_neg
