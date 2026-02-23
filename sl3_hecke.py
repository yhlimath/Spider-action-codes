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

        # Caching e operations for performance (optional, but good for "by hand" iteration if repeated)
        # But to be truly matrix-free, we shouldn't precompute matrix entries,
        # but maybe we can cache individual e_k(s) results if memory allows.
        # Given "H is very sparse so we do not need to know all the entries to do Arnoldi method. Rather, let's do Arnoldi 'by hand' without resorting to existing programs",
        # I will implement applying H(v) on the fly.
        self.e_cache = {}

    def apply_H(self, v):
        """
        Apply H = sum e_i to a dense vector v on the fly.
        v: numpy array of shape (dim,)
        Returns: H * v
        """
        # Output vector
        w = np.zeros(self.dim, dtype=complex)

        # H is sum of e_k for k=1..3L-1
        num_generators = 3 * self.L - 1

        # Iterate over non-zero elements of v?
        # v is likely dense in Arnoldi. So iterate over all basis elements i.
        # If v[i] is small, maybe skip? No, explicit loop over basis.

        # Optimization: H is sum of e_k. Applying e_k to basis element s gives a linear combination of basis elements.
        # w = sum_i v[i] * H(s_i)
        #   = sum_i v[i] * sum_k e_k(s_i)

        # This is essentially matrix-vector multiplication where matrix is implicit.
        # To make this fast, we can cache the result of H(s_i).
        # H(s_i) is a sparse vector (list of (coeff, index)).

        for idx, s in enumerate(self.basis_strings):
            coeff_v = v[idx]
            if abs(coeff_v) < 1e-12: continue

            # Apply H to s: sum_k e_k(s)
            # Check cache
            s_tuple = tuple(s)
            if s_tuple in self.e_cache:
                h_s_results = self.e_cache[s_tuple]
            else:
                h_s_results = [] # List of (val, target_idx)
                for k in range(1, num_generators + 1):
                    res_pairs = e([(1, s)], k)
                    for c, res_s in res_pairs:
                        if isinstance(c, Polynomial):
                            val = c.evaluate(self.n_value)
                        else:
                            val = c

                        res_idx = self.string_to_idx.get(tuple(res_s))
                        if res_idx is not None:
                            h_s_results.append((val, res_idx))

                # Cache it? If L is large, cache might grow.
                # For L=5, dim=6006. 6006 entries in cache. Each entry is a list of ~few terms.
                # This is manageable.
                self.e_cache[s_tuple] = h_s_results

            # Add contribution to w
            for val, target_idx in h_s_results:
                w[target_idx] += coeff_v * val

        return w

    def arnoldi_iteration(self, k, start_vector_idx=None):
        """
        Run Arnoldi iteration manually.
        k: Number of Krylov vectors (dimension of Hessenberg matrix)
        start_vector_idx: Index of basis string to start with (default: random or 0)
        """
        print(f"Running custom Arnoldi iteration (k={k})...")

        # Q matrix (orthogonal basis vectors)
        Q = np.zeros((self.dim, k + 1), dtype=complex)
        # H matrix (Hessenberg)
        h = np.zeros((k + 1, k), dtype=complex)

        # Start vector
        if start_vector_idx is None:
            # Use a random basis string as requested, or just index 0
            # "We just begin with a vector v, which is a basis_string"
            # Random basis string index
            start_vector_idx = random.randint(0, self.dim - 1)

        print(f"Starting with basis vector index {start_vector_idx}")
        Q[start_vector_idx, 0] = 1.0 # Normalized since it's a basis vector

        for j in range(k):
            # w = A * q_j
            v_j = Q[:, j]
            w = self.apply_H(v_j)

            # Orthogonalize
            for i in range(j + 1):
                # h_{i,j} = q_i^H * w
                h[i, j] = np.vdot(Q[:, i], w)
                w = w - h[i, j] * Q[:, i]

            # Norm
            norm_w = np.linalg.norm(w)
            h[j + 1, j] = norm_w

            if norm_w < 1e-12: # Breakdown
                print("Arnoldi breakdown (invariant subspace found)")
                return h[:j+1, :j+1]

            if j < k:
                Q[:, j + 1] = w / norm_w

        # Remove the last row of h (which is for the next vector) to get the square Hessenberg matrix
        # Usually eigenvalues are computed from the square k x k part.
        # But we computed k+1 rows.
        # The eigenvalues of the projection are from h[:k, :k].

        return h[:k, :k]

if __name__ == "__main__":
    L_test = 3
    n_val = 1.0

    solver = Sl3HeckeArnoldi(L=L_test, n_value=n_val)

    # Run Arnoldi
    k_arnoldi = min(20, solver.dim)
    hessenberg_mat = solver.arnoldi_iteration(k=k_arnoldi)

    # Compute eigenvalues of Hessenberg matrix
    eigenvalues = np.linalg.eigvals(hessenberg_mat)

    # Sort by magnitude
    eigenvalues = sorted(eigenvalues, key=abs, reverse=True)

    print(f"Top eigenvalues for L={L_test}, n={n_val}:")
    for val in eigenvalues[:10]:
        print(val)
