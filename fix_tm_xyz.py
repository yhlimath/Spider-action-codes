import sys
import re

with open('denseKuperberg/transfer_matrix.py', 'r') as f:
    content = f.read()

content = content.replace("action_T1_x_i, action_T2_x_i", "action_T_xyz_i")

new_apply = """def apply_T_i(coeff, path, i, type_str, n_value, x_value=None, y_value=None, z_value=None):
    base_state = [(coeff, path)]
    result = {}

    def add_to_result(action_dict):
        for p, c in action_dict.items():
            result[p] = result.get(p, 0) + c

    if type_str == 'T(x,y,z)':
        res = apply_action(base_state, lambda c, p, idx: action_T_xyz_i(c, p, idx, x_value, y_value, z_value, n_value), i, n_value)
        add_to_result(res)
        return result"""

content = re.sub(r'def apply_T_i\(.*?\).*?return result(?=\n\n    if type_str ==)', new_apply, content, flags=re.DOTALL)

# Update apply_layer
content = content.replace("def apply_layer(state_dict, indices, type_str, n_value, x_value=None):", "def apply_layer(state_dict, indices, type_str, n_value, x_value=None, y_value=None, z_value=None):")
content = content.replace("def apply_layer(state_dict, indices, type_str, n_value, x_value=None, y_value=None):", "def apply_layer(state_dict, indices, type_str, n_value, x_value=None, y_value=None, z_value=None):")
content = content.replace("res = apply_T_i(c, list(p_tup), i, type_str, n_value, x_value)", "res = apply_T_i(c, list(p_tup), i, type_str, n_value, x_value, y_value, z_value)")
content = content.replace("res = apply_T_i(c, list(p_tup), i, type_str, n_value, x_value, y_value)", "res = apply_T_i(c, list(p_tup), i, type_str, n_value, x_value, y_value, z_value)")

# Update build_transfer_matrix
content = content.replace("def build_transfer_matrix(L, x, y, type_str, order_str, n_value, x_value=None):", "def build_transfer_matrix(L, x, y, type_str, order_str, n_value, x_value=None, y_value=None, z_value=None):")
content = content.replace("def build_transfer_matrix(L, x, y, type_str, order_str, n_value, x_value=None, y_value=None):", "def build_transfer_matrix(L, x, y, type_str, order_str, n_value, x_value=None, y_value=None, z_value=None):")
content = content.replace("state = apply_layer(state, layer_indices, type_str, n_value, x_value)", "state = apply_layer(state, layer_indices, type_str, n_value, x_value, y_value, z_value)")
content = content.replace("state = apply_layer(state, layer_indices, type_str, n_value, x_value, y_value)", "state = apply_layer(state, layer_indices, type_str, n_value, x_value, y_value, z_value)")

with open('denseKuperberg/transfer_matrix.py', 'w') as f:
    f.write(content)
