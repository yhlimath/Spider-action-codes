import sys
import re

with open('denseKuperberg/arnoldi.py', 'r') as f:
    content = f.read()

content = content.replace("def __init__(self, L, x, y, type_str, order_str, n_value, x_value=None):", "def __init__(self, L, x, y, type_str, order_str, n_value, x_value=None, y_value=None, z_value=None, operator='T'):")
content = content.replace("def __init__(self, L, x, y, type_str, order_str, n_value, x_value=None, y_value=None):", "def __init__(self, L, x, y, type_str, order_str, n_value, x_value=None, y_value=None, z_value=None, operator='T'):")

content = content.replace("self.n_value = n_value\n        self.x_value = x_value", "self.n_value = n_value\n        self.x_value = x_value\n        self.y_value = y_value\n        self.z_value = z_value\n        self.operator = operator")

content = content.replace("res = apply_T_i(1.0, list(path_tuple), i, self.type_str, self.n_value, self.x_value)", "res = apply_T_i(1.0, list(path_tuple), i, self.type_str, self.n_value, self.x_value, self.y_value, self.z_value)")

# Add apply_H method
new_apply_h = """    def apply_H(self, v):
        w = np.zeros(self.dim, dtype=complex)
        indices = list(range(self.num_generators))

        for i in indices:
            for idx in range(self.dim):
                coeff = v[idx]
                if abs(coeff) < 1e-12:
                    continue

                path_tuple = tuple(self.basis_paths[idx])
                res = self._get_T_i_action(path_tuple, i)

                for c, target_idx in res:
                    w[target_idx] += coeff * c
        return w

    def arnoldi_iteration(self, k, start_vector=None):"""

content = content.replace("    def arnoldi_iteration(self, k, start_vector=None):", new_apply_h)

# Update arnoldi loop to call apply_H if operator is H
content = content.replace("v_next = self.apply_T(Q[:, j])", "if self.operator == 'H':\n                v_next = self.apply_H(Q[:, j])\n            else:\n                v_next = self.apply_T(Q[:, j])")

with open('denseKuperberg/arnoldi.py', 'w') as f:
    f.write(content)
