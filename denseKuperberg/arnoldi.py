import numpy as np
from scipy.linalg import eigvals, norm
from denseKuperberg.states import generate_paths
from denseKuperberg.transfer_matrix import apply_T_i

class KuperbergArnoldiSolver:
    def __init__(self, L, x, y, type_str, order_str, n_value):
        self.L = L
        self.type_str = type_str
        self.order_str = order_str
        self.n_value = n_value

        self.basis_paths = generate_paths(L, x, y)
        self.dim = len(self.basis_paths)
        if self.dim > 0:
            self.path_to_idx = {tuple(p): i for i, p in enumerate(self.basis_paths)}
        else:
            self.path_to_idx = {}

        self.num_generators = L - 1
        if order_str == 'sequential':
            self.layers = [list(range(self.num_generators))]
        elif order_str == 'staggered':
            self.layers = [list(range(1, self.num_generators, 2)), list(range(0, self.num_generators, 2))]
        else:
            raise ValueError(f"Unknown order_str: {order_str}")

        self.cache = {}

    def _get_T_i_action(self, path_tuple, i):
        key = (path_tuple, i)
        if key in self.cache:
            return self.cache[key]

        res = apply_T_i(1.0, list(path_tuple), i, self.type_str, self.n_value)

        action_results = []
        for p, c in res.items():
            if abs(c) > 1e-12:
                idx = self.path_to_idx.get(p)
                if idx is not None:
                    action_results.append((c, idx))

        self.cache[key] = action_results
        return action_results

    def apply_layer(self, v, indices):
        current_v = v.copy()

        for i in indices:
            next_v = np.zeros(self.dim, dtype=complex)
            for idx in range(self.dim):
                coeff = current_v[idx]
                if abs(coeff) < 1e-12:
                    continue

                path_tuple = tuple(self.basis_paths[idx])
                res = self._get_T_i_action(path_tuple, i)

                for c, target_idx in res:
                    next_v[target_idx] += coeff * c
            current_v = next_v

        return current_v

    def apply_T(self, v):
        w = v.copy()
        for layer_indices in self.layers:
            w = self.apply_layer(w, layer_indices)
        return w

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
