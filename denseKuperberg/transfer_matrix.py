import numpy as np
from denseKuperberg.states import generate_paths
from denseKuperberg.algebra import action_E_i, action_H_i, action_T_xyz_i
from sl3hecke.sl3_hecke import Polynomial

def evaluate_coeff(coeff, n_value):
    if isinstance(coeff, Polynomial):
        return coeff.evaluate(n_value)
    return coeff

def apply_action(paths_with_coeffs, action_func, i, n_value):
    """
    Applies an action_func to a list of (coeff, path) tuples.
    Returns a dictionary of path_tuple -> coeff.
    """
    result = {}
    for coeff, path in paths_with_coeffs:
        res = action_func(coeff, path, i)
        for res_coeff, res_path in res:
            val = evaluate_coeff(res_coeff, n_value)
            if val != 0:
                p_tup = tuple(res_path)
                result[p_tup] = result.get(p_tup, 0) + val
    return result

def apply_T_i(coeff, path, i, type_str, n_value, x_value=None, y_value=None, z_value=None):
    base_state = [(coeff, path)]
    result = {}

    def add_to_result(action_dict):
        for p, c in action_dict.items():
            result[p] = result.get(p, 0) + c

    if type_str == 'T(x,y,z)':
        res = apply_action(base_state, lambda c, p, idx: action_T_xyz_i(c, p, idx, x_value, y_value, z_value, n_value), i, n_value)
        add_to_result(res)
        return result

    if type_str == 'E+H+H2' or type_str == 'E+H':
        res_E = apply_action(base_state, action_E_i, i, n_value)
        add_to_result(res_E)

    if type_str in ['E+H+H2', 'E+H']:
        res_H = apply_action(base_state, action_H_i, i, n_value)
        add_to_result(res_H)

    if type_str == 'E+H+H2' or type_str == 'H2':
        res_H = apply_action(base_state, action_H_i, i, n_value)
        h_states = [(c, list(p)) for p, c in res_H.items()]
        if h_states:
            res_H2 = apply_action(h_states, action_H_i, i, n_value)
            add_to_result(res_H2)

    return result

def apply_layer(state_dict, indices, type_str, n_value, x_value=None, y_value=None, z_value=None):
    """
    Applies T_i for all i in indices sequentially to state_dict.
    state_dict is path_tuple -> coeff.
    """
    current_state = state_dict
    for i in indices:
        next_state = {}
        for p_tup, c in current_state.items():
            res = apply_T_i(c, list(p_tup), i, type_str, n_value, x_value, y_value, z_value)
            for new_p, new_c in res.items():
                if abs(new_c) > 1e-12:
                    next_state[new_p] = next_state.get(new_p, 0) + new_c
        current_state = next_state
    return current_state

def build_transfer_matrix(L, x, y, type_str, order_str, n_value, x_value=None, y_value=None, z_value=None):
    """
    Builds the dense transfer matrix.
    order_str: 'sequential' or 'staggered'
    """
    paths = generate_paths(L, x, y)
    dim = len(paths)
    if dim == 0:
        return np.zeros((0, 0)), paths

    # To optimize, we can map tuple(path) to index
    path_to_idx = {tuple(p): idx for idx, p in enumerate(paths)}
    matrix = np.zeros((dim, dim), dtype=complex)

    num_generators = L - 1

    if order_str == 'sequential':
        indices = list(range(num_generators))
        layers = [indices]
    elif order_str == 'staggered':
        odd_indices = list(range(1, num_generators, 2))
        even_indices = list(range(0, num_generators, 2))
        layers = [odd_indices, even_indices]
    else:
        raise ValueError(f"Unknown order_str: {order_str}")

    for idx, p in enumerate(paths):
        state = {tuple(p): 1.0}
        for layer_indices in layers:
            state = apply_layer(state, layer_indices, type_str, n_value, x_value, y_value, z_value)

        for p_tup, c in state.items():
            if p_tup in path_to_idx:
                target_idx = path_to_idx[p_tup]
                matrix[target_idx, idx] += c
            else:
                pass

    return matrix, paths
