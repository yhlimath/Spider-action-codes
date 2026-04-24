from sl3hecke.sl3_hecke import Polynomial, e, bending_power, string_decomposition, form_triplets_with_positions
import numpy as np

def generate_constrained_strings(m, x, y):
    """Generate all valid strings of type (m,x,y)"""
    if (m + 2*x + y) % 3 != 0:
        return []

    count_1s = (m + 2*x + y) // 3
    count_0s = (m - x + y) // 3
    count_minus1s = (m - x - 2*y) // 3

    if count_1s < 0 or count_0s < 0 or count_minus1s < 0:
        return []

    if count_1s + count_0s + count_minus1s != m:
        return []

    def backtrack(sequence, remaining_1s, remaining_0s, remaining_minus1s,
                  current_1s, current_0s, current_minus1s, results):
        if len(sequence) == m:
            if (remaining_1s == 0 and remaining_0s == 0 and remaining_minus1s == 0):
                results.append(sequence[:])
            return

        if remaining_1s > 0:
            backtrack(sequence + [1], remaining_1s - 1, remaining_0s, remaining_minus1s,
                     current_1s + 1, current_0s, current_minus1s, results)

        if remaining_0s > 0:
            new_0s = current_0s + 1
            if current_1s >= new_0s:
                backtrack(sequence + [0], remaining_1s, remaining_0s - 1, remaining_minus1s,
                         current_1s, new_0s, current_minus1s, results)

        if remaining_minus1s > 0:
            new_minus1s = current_minus1s + 1
            if current_1s >= current_0s >= new_minus1s:
                backtrack(sequence + [-1], remaining_1s, remaining_0s, remaining_minus1s - 1,
                         current_1s, current_0s, new_minus1s, results)

    results = []
    backtrack([], count_1s, count_0s, count_minus1s, 0, 0, 0, results)
    return results

def map_constrained_to_balanced(s, x, y):
    """Map constrained string to balanced string"""
    s_prime = s.copy()
    for _ in range(x):
        s_prime.extend([0, -1])
    for _ in range(y):
        s_prime.append(-1)
    return s_prime

def is_valid_constrained_string(string, m, x, y):
    """Check if string satisfies (m,x,y) constraints"""
    if len(string) != m:
        return False

    count_1, count_0, count_neg1 = 0, 0, 0
    for element in string:
        if element == 1:
            count_1 += 1
        elif element == 0:
            count_0 += 1
        elif element == -1:
            count_neg1 += 1

        if not (count_1 >= count_0 >= count_neg1):
            return False

    return True

def ed(S, i, m, x, y, verbose=False):
    """Extended e function for constrained strings"""
    if not S:
        return []

    # Step 1: Map to balanced strings
    balanced_pairs = []
    for coeff, constrained_string in S:
        balanced_string = map_constrained_to_balanced(constrained_string, x, y)
        balanced_pairs.append((coeff, balanced_string))

    # Step 2: Apply e function
    from sl3hecke.sl3_hecke import e
    balanced_results = e(balanced_pairs, i, verbose)

    # Step 3: Truncate and filter
    final_results = []
    for coeff, balanced_result in balanced_results:
        if len(balanced_result) >= m:
            truncated = balanced_result[:m]
            if is_valid_constrained_string(truncated, m, x, y):
                final_results.append((coeff, truncated))

    return final_results
