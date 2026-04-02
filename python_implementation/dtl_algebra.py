def generate_dtl_states(L, j):
    """
    Generates the basis of the dilute Temperley-Lieb module V^{L,j}.
    States are represented as tuples of 1, 0, -1.
    Constraints:
    - Length of state is L.
    - Partial sums are always non-negative.
    - Total sum is j.
    """
    results = []

    def _generate(current_state, current_sum):
        if len(current_state) == L:
            if current_sum == j:
                results.append(tuple(current_state))
            return

        # We can always add 1
        _generate(current_state + [1], current_sum + 1)

        # We can always add 0
        _generate(current_state + [0], current_sum)

        # We can add -1 only if current_sum > 0 (to keep partial sums >= 0)
        if current_sum > 0:
            _generate(current_state + [-1], current_sum - 1)

    _generate([], 0)
    return results

def apply_identity(state, i):
    """
    Identity operator I_i: acts on sites i, i+1.
    If both sites are occupied (1 or -1), keeps state unchanged.
    Otherwise, yields zero vector.
    """
    if state[i] != 0 and state[i+1] != 0:
        return [(state, 1)]
    return []

def apply_half_vacuum_1(state, i):
    """
    Half-vacuum type 1: maps (±1, 0) to (±1, 0).
    Other inputs yield zero vector.
    """
    if state[i] in [1, -1] and state[i+1] == 0:
        return [(state, 1)]
    return []

def apply_half_vacuum_2(state, i):
    """
    Half-vacuum type 2: maps (0, ±1) to (0, ±1).
    Other inputs yield zero vector.
    """
    if state[i] == 0 and state[i+1] in [1, -1]:
        return [(state, 1)]
    return []

def apply_half_vacuum_3(state, i):
    """
    Half-vacuum type 3: maps (±1, 0) to (0, ±1).
    Other inputs yield zero vector.
    """
    if state[i] in [1, -1] and state[i+1] == 0:
        new_state = list(state)
        new_state[i] = 0
        new_state[i+1] = state[i]
        return [(tuple(new_state), 1)]
    return []

def apply_half_vacuum_4(state, i):
    """
    Half-vacuum type 4: maps (0, ±1) to (±1, 0).
    Other inputs yield zero vector.
    """
    if state[i] == 0 and state[i+1] in [1, -1]:
        new_state = list(state)
        new_state[i] = state[i+1]
        new_state[i+1] = 0
        return [(tuple(new_state), 1)]
    return []

def apply_tl(state, i, n):
    """
    Temperley-Lieb generator e_i: acts on sites i, i+1.
    If it touches a 0, yields zero vector.
    (1, -1) -> (1, -1) with weight n
    (-1, 1) -> (1, -1) with weight 1
    (1, 1) -> finds the matching -1 to the right, changes it to 1, changes second 1 to -1.
    (-1, -1) -> finds the matching 1 to the left, changes it to -1, changes first -1 to 1.
    """
    if state[i] == 0 or state[i+1] == 0:
        return []

    s1, s2 = state[i], state[i+1]

    if s1 == 1 and s2 == -1:
        return [(state, n)]

    elif s1 == -1 and s2 == 1:
        new_state = list(state)
        new_state[i] = 1
        new_state[i+1] = -1
        return [(tuple(new_state), 1)]

    elif s1 == 1 and s2 == 1:
        current_sum = 0
        found_k = -1
        for k in range(i+2, len(state)):
            if current_sum == 0 and state[k] == -1:
                found_k = k
                break
            current_sum += state[k]

        if found_k != -1:
            new_state = list(state)
            new_state[found_k] = 1
            new_state[i+1] = -1
            return [(tuple(new_state), 1)]
        else:
            return []

    elif s1 == -1 and s2 == -1:
        current_sum = 0
        found_k = -1
        for k in range(i-1, -1, -1):
            if current_sum == 0 and state[k] == 1:
                found_k = k
                break
            current_sum += state[k]

        if found_k != -1:
            new_state = list(state)
            new_state[found_k] = -1
            new_state[i] = 1
            return [(tuple(new_state), 1)]
        else:
            return []

    return []

def apply_E_i(state, i, q):
    """
    Operator E_i = TL_i + q * V^(1)_i + (1/q) * V^(2)_i + V^(3)_i + V^(4)_i.
    Uses sympy for symbolic variables q.
    n = q + 1/q is used for the TL loop weight.
    Returns a list of (state, weight).
    """
    import sympy

    # Calculate n
    n = q + 1/q

    results = []

    # Add TL
    results.extend(apply_tl(state, i, n))

    # Add q * V^(1)
    for res_state, weight in apply_half_vacuum_1(state, i):
        results.append((res_state, weight * q))

    # Add (1/q) * V^(2)
    for res_state, weight in apply_half_vacuum_2(state, i):
        results.append((res_state, weight / q))

    # Add V^(3)
    for res_state, weight in apply_half_vacuum_3(state, i):
        results.append((res_state, weight))

    # Add V^(4)
    for res_state, weight in apply_half_vacuum_4(state, i):
        results.append((res_state, weight))

    return results
