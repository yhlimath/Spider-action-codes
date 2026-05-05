import json
import numpy as np
import os
import argparse
import itertools

def load_data(in_file):
    with open(in_file, 'r') as f:
        return json.load(f)

def compute_h(lam0, lamj, L, vF):
    return (L / (np.pi * vF)) * np.log(abs(lam0) / abs(lamj))

def evaluate_fit(h_vals, L_vals):
    # Fit h_L = h + A/L + B/L^2
    L_arr = np.array(L_vals)
    y = np.array(h_vals)
    X = np.column_stack([np.ones_like(L_arr), 1.0/L_arr, 1.0/(L_arr**2)])

    coeffs, residuals, _, _ = np.linalg.lstsq(X, y, rcond=None)

    if len(residuals) > 0:
        ssr = residuals[0]
    else:
        # Exact fit for 3 points
        y_fit = X @ coeffs
        ssr = np.sum((y - y_fit)**2)

    return coeffs[0], ssr

def extrapolate_for_config(L_dict, vF, num_states=10, max_search_depth=15):
    """
    L_dict: dictionary mapping L_str -> list of dicts {"abs": ...}
    vF: Fermi velocity
    """
    L_vals = sorted([int(L) for L in L_dict.keys() if int(L) >= 3 and L_dict[L] is not None])
    if len(L_vals) < 3:
        return None

    # Pre-calculate candidate h_j values for each L
    # candidate_h[L][j] = h_value
    candidate_h = {}
    for L in L_vals:
        lam_list = L_dict[str(L)]
        lam0 = lam_list[0]['abs']

        # Limit search depth per L to avoid combinatorial explosion
        depth = min(max_search_depth, len(lam_list))
        h_arr = []
        for j in range(depth):
            lamj = lam_list[j]['abs']
            h_arr.append(compute_h(lam0, lamj, L, vF))
        candidate_h[L] = h_arr

    available_indices = {L: set(range(len(candidate_h[L]))) for L in L_vals}

    extracted_states = []

    for state_idx in range(num_states):
        best_fit = None
        best_ssr = float('inf')
        best_indices = None

        # Build generator for all possible combinations of available indices
        # To avoid massive loops, we can iterate over L and pick indices
        # A simpler way: we assume h values don't cross wildly.
        # We can just use itertools.product on the available indices

        # Actually, let's sort available indices by their h value so we prioritize lower h
        lists_of_indices = [sorted(list(available_indices[L])) for L in L_vals]

        for idx_combo in itertools.product(*lists_of_indices):
            h_vals = [candidate_h[L][idx] for L, idx in zip(L_vals, idx_combo)]

            # Simple heuristic: h shouldn't jump by more than ~1.0 between Ls
            # This speeds up search drastically
            if max(h_vals) - min(h_vals) > 2.0:
                continue

            h_extrap, ssr = evaluate_fit(h_vals, L_vals)

            if ssr < best_ssr:
                best_ssr = ssr
                best_fit = h_extrap
                best_indices = idx_combo

        if best_indices is not None:
            extracted_states.append({
                "h_extrap": best_fit,
                "ssr": best_ssr,
                "indices": {L: idx for L, idx in zip(L_vals, best_indices)}
            })
            # Remove used indices
            for L, idx in zip(L_vals, best_indices):
                available_indices[L].remove(idx)
        else:
            break

    return extracted_states

def main():
    parser = argparse.ArgumentParser(description="Extrapolate conformal dimensions for dense Kuperberg")
    parser.add_argument('--vF', type=float, default=1.0, help="Fermi velocity to use for extrapolation")
    parser.add_argument('--type', type=str, default=None, help="Filter by matrix type (e.g. E+H)")
    parser.add_argument('--order', type=str, default=None, help="Filter by order (e.g. staggered)")
    parser.add_argument('--n', type=str, default=None, help="Filter by weight n (e.g. 1.0)")
    parser.add_argument('--num_states', type=int, default=10, help="Number of conformal states to extract")
    args = parser.parse_args()

    in_file = "experiment_outputs/denseKuperberg/eigenvalue_logs_top_k.json"
    if not os.path.exists(in_file):
        print(f"Log file {in_file} not found.")
        return

    logs = load_data(in_file)

    for t in logs:
        if args.type and t != args.type: continue
        for order in logs[t]:
            if args.order and order != args.order: continue
            for n_str in logs[t][order]:
                if args.n and n_str != args.n: continue

                L_dict = logs[t][order][n_str]
                results = extrapolate_for_config(L_dict, args.vF, num_states=args.num_states)

                if results:
                    print(f"\nConfiguration: Type={t}, Order={order}, n={n_str}")
                    print(f"{'State':<6} | {'Extrapolated h':<15} | {'SSR':<12} | {'Indices Used (L=' + ','.join(map(str, sorted([int(x) for x in L_dict.keys() if int(x)>=3]))) + ')'}")
                    print("-" * 75)
                    for i, res in enumerate(results):
                        idx_str = str([res['indices'][L] for L in sorted(res['indices'].keys())])
                        print(f"{i:<6} | {res['h_extrap']:<15.6f} | {res['ssr']:<12.2e} | {idx_str}")

if __name__ == "__main__":
    main()
