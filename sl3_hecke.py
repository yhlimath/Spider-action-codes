
import numpy as np
from scipy.sparse import dok_matrix, csr_matrix
from scipy.sparse.linalg import eigs
import random

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
        self.H_matrix = None

    def build_matrix(self):
        print(f"Building H matrix for L={self.L} (dim={self.dim})...")
        mat = dok_matrix((self.dim, self.dim), dtype=complex)

        # H = sum_{k=1 to 3L-1} e_k
        num_generators = 3 * self.L - 1

        for idx, s in enumerate(self.basis_strings):
            for k in range(1, num_generators + 1):
                res_pairs = e([(1, s)], k)
                for coeff, res_s in res_pairs:
                    if isinstance(coeff, Polynomial):
                        val = coeff.evaluate(self.n_value)
                    else:
                        val = coeff

                    res_idx = self.string_to_idx.get(tuple(res_s))
                    if res_idx is not None:
                        mat[res_idx, idx] += val

        self.H_matrix = mat.tocsr()
        print("Matrix build complete.")

    def run_arnoldi(self, k=6, which='LM'):
        if self.H_matrix is None:
            self.build_matrix()

        print(f"Running Arnoldi method to find {k} eigenvalues...")
        vals, vecs = eigs(self.H_matrix, k=k, which=which)
        return vals

if __name__ == "__main__":
    # Test with L=2 (dim is small) first, then maybe L=3 or 4
    # User asked for "valid strings of length 3L".
    # In notebook: generate_all_valid_strings(4) produced 12 sites (3*4). So L=4 means 12 sites.
    # Notebook ran tests on L=4.

    # Let's try L=3 (9 sites) first as a quick test.
    L_test = 3
    # n value? q + 1/q. Let's pick q = 1, so n = 2.
    # Or q = exp(i*pi/3)?
    # User said "C(q)-vector space". Since we need numerical values, I'll choose a generic q.
    # q = 2 (so n = 2.5) to avoid roots of unity issues unless desired.
    # Let's use n=2 (q=1) for simplicity as a start, or a complex number.
    n_val = 2.0

    solver = Sl3HeckeArnoldi(L=L_test, n_value=n_val)
    eigenvalues = solver.run_arnoldi(k=min(10, solver.dim - 2))

    print(f"Eigenvalues for L={L_test}, n={n_val}:")
    for val in eigenvalues:
        print(val)
