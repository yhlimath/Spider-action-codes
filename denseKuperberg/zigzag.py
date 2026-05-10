import numpy as np
from scipy.linalg import eigvals, norm
from denseKuperberg.states import step_weight
from denseKuperberg.algebra import action_H_i
from denseKuperberg.transfer_matrix import apply_T_i

def generate_one_minus_paths(L, target_x, target_y):
    """
    Generates all valid paths P = (S, J) of length L with exactly one '-' sign
    and final weight (target_x, target_y).
    """
    results = []

    def backtrack(path, current_x, current_y, minus_count):
        if len(path) == L:
            if current_x == target_x and current_y == target_y and minus_count == 1:
                results.append(path[:])
            return

        remaining_steps = L - len(path)

        possible_signs = []
        if minus_count == 1:
            possible_signs = [1]
        elif minus_count == 0:
            if remaining_steps == 1:
                possible_signs = [-1]
            else:
                possible_signs = [1, -1]
        else:
            return

        for s in possible_signs:
            for j in [1, 0, -1]:
                dx, dy = step_weight(s, j)
                new_x = current_x + dx
                new_y = current_y + dy

                if new_x >= 0 and new_y >= 0:
                    path.append((s, j))
                    backtrack(path, new_x, new_y, minus_count + (1 if s == -1 else 0))
                    path.pop()

    backtrack([], 0, 0, 0)
    return results

class ZigzagArnoldiSolver:
    def __init__(self, L, x, y, n_value):
        self.L = L
        self.n_value = n_value

        self.basis_paths = generate_one_minus_paths(L, x, y)
        self.dim = len(self.basis_paths)
        if self.dim > 0:
            self.path_to_idx = {tuple(p): i for i, p in enumerate(self.basis_paths)}
        else:
            self.path_to_idx = {}

        self.num_generators = L - 1

        indices_forward = list(range(self.num_generators))
        indices_backward = list(reversed(range(self.num_generators)))
        self.sequence = indices_forward + indices_backward

        self.cache = {}

    def _get_H_i_action(self, path_tuple, i):
        key = (path_tuple, i)
        if key in self.cache:
            return self.cache[key]

        from denseKuperberg.transfer_matrix import apply_action

        base_state = [(1.0, list(path_tuple))]
        res = apply_action(base_state, action_H_i, i, self.n_value)

        action_results = []
        for p, c in res.items():
            if abs(c) > 1e-12:
                idx = self.path_to_idx.get(p)
                if idx is not None:
                    action_results.append((c, idx))
                else:
                    pass

        self.cache[key] = action_results
        return action_results

    def apply_T(self, v):
        current_v = v.copy()

        for i in self.sequence:
            next_v = np.zeros(self.dim, dtype=complex)
            for idx in range(self.dim):
                coeff = current_v[idx]
                if abs(coeff) < 1e-12:
                    continue

                path_tuple = tuple(self.basis_paths[idx])
                res = self._get_H_i_action(path_tuple, i)

                for c, target_idx in res:
                    next_v[target_idx] += coeff * c
            current_v = next_v

        return current_v

    def arnoldi_iteration(self, k, start_vector=None):
        if self.dim == 0:
            return np.zeros((0, 0)), np.zeros((0, 0))

        k = min(k, self.dim)

        H = np.zeros((k, k), dtype=complex)
        Q = np.zeros((self.dim, k + 1), dtype=complex)

        if start_vector is None:
            np.random.seed(42)
            start_vector = np.random.rand(self.dim) + 1j * np.random.rand(self.dim)

        v = start_vector / norm(start_vector)
        Q[:, 0] = v

        for j in range(k):
            v_next = self.apply_T(Q[:, j])

            for i in range(j + 1):
                H[i, j] = np.vdot(Q[:, i], v_next)
                v_next = v_next - H[i, j] * Q[:, i]

            h_next = norm(v_next)

            if j < k - 1:
                H[j + 1, j] = h_next
                if h_next > 1e-10:
                    Q[:, j + 1] = v_next / h_next
                else:
                    return H[:j+1, :j+1], Q[:, :j+1]

        return H, Q[:, :-1]
