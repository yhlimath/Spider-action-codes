import numpy as np
from scipy.sparse import dok_matrix, csr_matrix
from scipy.linalg import eig, hessenberg, norm
import random
import time

class Polynomial:
    def __init__(self, coeffs=None):
        self.coeffs = coeffs if coeffs is not None else {}
        self._normalize()

    def _normalize(self):
        self.coeffs = {k: v for k, v in self.coeffs.items() if v != 0}

    @classmethod
    def constant(cls, value):
        return cls({0: value}) if value != 0 else cls({})

    @classmethod
    def variable(cls, power=1):
        return cls({power: 1})

    def __add__(self, other):
        if isinstance(other, (int, float, complex)):
            other = Polynomial.constant(other)
        result = Polynomial()
        result.coeffs = self.coeffs.copy()
        for power, coeff in other.coeffs.items():
            result.coeffs[power] = result.coeffs.get(power, 0) + coeff
        result._normalize()
        return result

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, (int, float, complex)):
            other = Polynomial.constant(other)
        result = Polynomial()
        result.coeffs = self.coeffs.copy()
        for power, coeff in other.coeffs.items():
            result.coeffs[power] = result.coeffs.get(power, 0) - coeff
        result._normalize()
        return result

    def __rsub__(self, other):
         return (Polynomial.constant(other) - self) if isinstance(other, (int, float, complex)) else NotImplemented

    def __mul__(self, other):
        if isinstance(other, (int, float, complex)):
            other = Polynomial.constant(other)
        result = Polynomial()
        for p1, c1 in self.coeffs.items():
            for p2, c2 in other.coeffs.items():
                power = p1 + p2
                result.coeffs[power] = result.coeffs.get(power, 0) + c1 * c2
        result._normalize()
        return result

    def __rmul__(self, other):
        return self.__mul__(other)

    def evaluate(self, n_value):
        result = 0
        for power, coeff in self.coeffs.items():
            result += coeff * (n_value ** power)
        return result

    def __repr__(self):
         return str(self.coeffs)

    def is_zero(self):
        return len(self.coeffs) == 0


def generate_all_valid_strings(n):
    def backtrack(sequence, count_1, count_0, count_neg1, results):
        if len(sequence) == 3 * n:
            results.append(sequence[:])
            return
        if count_1 < n:
            backtrack(sequence + [1], count_1 + 1, count_0, count_neg1, results)
        if count_0 < count_1 and count_0 < n:
            backtrack(sequence + [0], count_1, count_0 + 1, count_neg1, results)
        if count_neg1 < count_0 and count_neg1 < n:
            backtrack(sequence + [-1], count_1, count_0, count_neg1 + 1, results)

    results = []
    backtrack([], 0, 0, 0, results)
    return results

def form_triplets_with_positions(sequence):
    sequence = sequence[:]
    indexed_sequence = list(enumerate(sequence))
    triplets = []

    while indexed_sequence:
        try:
            index_1 = max(i for i, val in indexed_sequence if val == 1)
        except ValueError:
            break
        try:
            index_0 = next(i for i, val in indexed_sequence if val == 0 and i > index_1)
        except StopIteration:
            break
        try:
            index_neg1 = next(i for i, val in indexed_sequence if val == -1 and i > index_0)
        except StopIteration:
            break

        triplet = ((1, index_1), (0, index_0), (-1, index_neg1))
        triplets.append(triplet)

        indexed_sequence = [(i, val) for i, val in indexed_sequence if i not in {index_1, index_0, index_neg1}]

    return triplets

def bend_string(sequence):
    sequence = sequence[:]
    indexed_sequence = list(enumerate(sequence))

    try:
        start_1_index = next(i for i, val in indexed_sequence if val == 1)
    except StopIteration:
        return sequence

    triplets = form_triplets_with_positions(sequence)
    for triplet in triplets:
        if triplet[0][0] == 1 and triplet[0][1] == start_1_index:
            index_1, index_0, index_neg1 = triplet[0][1], triplet[1][1], triplet[2][1]
            break
    else:
        return sequence

    new_sequence = sequence[:]
    new_sequence[index_0] = 1
    new_sequence[index_neg1] = 0
    new_sequence.pop(index_1)
    new_sequence.append(-1)

    return new_sequence

def bending_power(s, p):
    s_bent_p_times = s[:]
    for _ in range(p):
        s_bent_p_times = bend_string(s_bent_p_times)
    return s_bent_p_times

def inverse_bending_power(s, p):
    return bending_power(s, len(s) - p)

def generate_triplet_at_position(s, p):
    if p > len(s):
        return s + [1, 0, -1]
    else:
        return s[:p-1] + [1, 0, -1] + s[p-1:]

def find_last_triplet(s):
    try:
        if 1 not in s: return None
        last_1_index = len(s) - 1 - s[::-1].index(1)

        last_0_index = next(i for i in range(last_1_index + 1, len(s)) if s[i] == 0)
        last_neg1_index = next(i for i in range(last_0_index + 1, len(s)) if s[i] == -1)
        return last_1_index, last_0_index, last_neg1_index
    except (ValueError, StopIteration):
        return None

def string_decomposition(s):
    s_decomposed = s[:]
    operations = []

    while len(s_decomposed) > 6:
        last_triplet_indices = find_last_triplet(s_decomposed)
        if last_triplet_indices is None:
            break

        last_1_index, last_0_index, last_neg1_index = last_triplet_indices

        if last_0_index == last_1_index + 1 and last_neg1_index == last_0_index + 1:
            s_decomposed = s_decomposed[:last_1_index] + s_decomposed[last_neg1_index + 1:]
            operations.append(('t', last_1_index + 1))

        else:
            if all(x == -1 for x in s_decomposed[last_1_index + 1:last_0_index]):
                start_neg1_string = last_1_index + 1
                end_neg1_string = last_0_index - 1
                s_decomposed[last_1_index], s_decomposed[end_neg1_string] = s_decomposed[end_neg1_string], s_decomposed[last_1_index]
                for i in range(start_neg1_string, end_neg1_string + 1):
                    operations.append(('e', i))

            if all(x == 0 for x in s_decomposed[last_0_index + 1:last_neg1_index]):
                    start_0_string = last_0_index + 1
                    end_0_string = last_neg1_index - 1
                    s_decomposed[last_neg1_index], s_decomposed[start_0_string] = s_decomposed[start_0_string] ,s_decomposed[last_neg1_index]
                    for i in range(end_0_string, start_0_string -1, -1):
                        operations.append(('e', i+1))
            else:
                 pass

    return s_decomposed, operations

def e(S, i, verbose=False):
    S_out = []
    for coeff, s in S:
        bent_s = bending_power(s, i-1)
        s_decomposed, operations = string_decomposition(bent_s)

        if s_decomposed in [[1, 0, 1, 0, -1, -1], [1, 0, 1, -1, 0, -1], [1, 0, -1, 1, 0, -1]]:
            if isinstance(coeff, (int, float, complex)):
                new_coeff = Polynomial.constant(coeff) * Polynomial.variable()
            elif isinstance(coeff, Polynomial):
                new_coeff = coeff * Polynomial.variable()
            else:
                new_coeff = coeff
            S_out.append((new_coeff, s))

        elif s_decomposed == [1, 1, 0, -1, 0, -1]:
            triplets = form_triplets_with_positions(bent_s)
            target_triplet = None
            try:
                # Find second 1
                first_one = bent_s.index(1)
                second_one_index = bent_s.index(1, first_one + 1)
            except ValueError:
                second_one_index = -1

            for triplet in triplets:
                if triplet[0][0] == 1 and triplet[0][1] == second_one_index:
                    target_triplet = triplet
                    break

            if target_triplet:
                s_jescon = bent_s[:]
                index_1, index_0, _ = target_triplet[0][1], target_triplet[1][1], target_triplet[2][1]
                s_jescon[index_1], s_jescon[index_0] = s_jescon[index_0], s_jescon[index_1]
                result = inverse_bending_power(s_jescon, i - 1)
                S_out.append((coeff, result))

        elif s_decomposed == [1,1,0,0,-1,-1]:
            new_strings = [[1,0,-1,1,0,-1],[1,0,1,0,-1,-1]]
            for op_type, position in reversed(operations):
                if op_type == 't':
                    temp_new_strings = []
                    for new_s in new_strings:
                        temp_new_strings.append(generate_triplet_at_position(new_s, position))
                    new_strings = temp_new_strings
                elif op_type == 'e':
                    temp_input = [(1, new_s) for new_s in new_strings]
                    temp_result = e(temp_input, position, verbose=False)
                    new_strings = [string for _, string in temp_result]

            final_results = [inverse_bending_power(new_s, i - 1) for new_s in new_strings]
            for result in final_results:
                S_out.append((coeff, result))

        else:
             pass

    return S_out


class Sl3HeckeArnoldi:
    def __init__(self, L, n_value):
        self.L = L
        self.n_value = n_value
        self.basis_strings = generate_all_valid_strings(L)
        self.dim = len(self.basis_strings)
        self.string_to_idx = {tuple(s): i for i, s in enumerate(self.basis_strings)}

        # Caches
        self.e_cache = {} # Cache for individual e_k(s)
        self.H_cache = {} # Cache for H(s) = sum e_k(s)

    def _get_e_k_action(self, s, k):
        s_tuple = tuple(s)
        cache_key = (s_tuple, k)
        if cache_key in self.e_cache:
            return self.e_cache[cache_key]

        res_pairs = e([(1, s)], k)
        action_results = []
        for c, res_s in res_pairs:
            if isinstance(c, Polynomial):
                val = c.evaluate(self.n_value)
            else:
                val = c

            res_idx = self.string_to_idx.get(tuple(res_s))
            if res_idx is not None:
                action_results.append((val, res_idx))

        self.e_cache[cache_key] = action_results
        return action_results

    def _get_H_action(self, s):
        s_tuple = tuple(s)
        if s_tuple in self.H_cache:
            return self.H_cache[s_tuple]

        num_generators = 3 * self.L - 1
        h_s_results = []
        # H = sum e_k
        for k in range(1, num_generators + 1):
            e_k_results = self._get_e_k_action(s, k)
            h_s_results.extend(e_k_results)

        # Optimization: Combine like terms?
        # A basis index might appear multiple times from different e_k
        # If we sum them up here, we save additions later.
        combined_results = {}
        for val, idx in h_s_results:
            combined_results[idx] = combined_results.get(idx, 0) + val

        final_results = [(val, idx) for idx, val in combined_results.items() if val != 0]
        self.H_cache[s_tuple] = final_results
        return final_results

    def apply_H(self, v):
        """
        Apply H = sum e_i to a dense vector v on the fly.
        """
        w = np.zeros(self.dim, dtype=complex)

        for idx, s in enumerate(self.basis_strings):
            coeff_v = v[idx]
            if abs(coeff_v) < 1e-12: continue

            h_s_results = self._get_H_action(s)

            for val, target_idx in h_s_results:
                w[target_idx] += coeff_v * val

        return w

    def apply_T(self, v):
        """
        Apply Transfer Matrix T = prod e_even * prod e_odd to v.
        Application order: Apply odd indices first, then even indices.
        """
        w = v.copy()
        num_generators = 3 * self.L - 1

        # Odd indices: 1, 3, 5, ...
        odd_indices = range(1, num_generators + 1, 2)
        # Even indices: 2, 4, 6, ...
        even_indices = range(2, num_generators + 1, 2)

        # Apply all odd e_k
        for k in odd_indices:
            w = self.apply_single_generator(w, k)

        # Apply all even e_k
        for k in even_indices:
            w = self.apply_single_generator(w, k)

        return w

    def apply_single_generator(self, v, k):
        """
        Apply e_k to v.
        """
        w = np.zeros(self.dim, dtype=complex)

        for idx, s in enumerate(self.basis_strings):
            coeff_v = v[idx]
            if abs(coeff_v) < 1e-12: continue

            action_results = self._get_e_k_action(s, k)
            for val, target_idx in action_results:
                w[target_idx] += coeff_v * val
        return w

    def arnoldi_iteration(self, k, start_vector_idx=None, operator='H'):
        """
        Run Arnoldi iteration manually.
        operator: 'H' for Hamiltonian, 'T' for Transfer Matrix
        """
        print(f"Running custom Arnoldi iteration (k={k}, operator={operator})...")

        Q = np.zeros((self.dim, k + 1), dtype=complex)
        h = np.zeros((k + 1, k), dtype=complex)

        if start_vector_idx is None:
            start_vector_idx = random.randint(0, self.dim - 1)

        print(f"Starting with basis vector index {start_vector_idx}")
        Q[start_vector_idx, 0] = 1.0

        for j in range(k):
            v_j = Q[:, j]

            if operator == 'H':
                w = self.apply_H(v_j)
            elif operator == 'T':
                w = self.apply_T(v_j)
            else:
                raise ValueError("Unknown operator. Use 'H' or 'T'.")

            # Orthogonalize
            for i in range(j + 1):
                h[i, j] = np.vdot(Q[:, i], w)
                w = w - h[i, j] * Q[:, i]

            norm_w = np.linalg.norm(w)
            h[j + 1, j] = norm_w

            if norm_w < 1e-12:
                print("Arnoldi breakdown (invariant subspace found)")
                return h[:j+1, :j+1]

            if j < k:
                Q[:, j + 1] = w / norm_w

        return h[:k, :k]

if __name__ == "__main__":
    L_test = 3
    n_val = 1.0

    solver = Sl3HeckeArnoldi(L=L_test, n_value=n_val)

    # Test H
    print("Testing H operator:")
    h_mat = solver.arnoldi_iteration(k=10, operator='H')
    evals_H = np.linalg.eigvals(h_mat)
    print("Top eigenvalues (H):", sorted(evals_H, key=abs, reverse=True)[:3])

    # Test T
    print("\nTesting T operator:")
    t_mat = solver.arnoldi_iteration(k=10, operator='T')
    evals_T = np.linalg.eigvals(t_mat)
    print("Top eigenvalues (T):", sorted(evals_T, key=abs, reverse=True)[:3])
