import json
import numpy as np
import os
import argparse

def load_data(in_file):
    with open(in_file, 'r') as f:
        return json.load(f)

def compute_h(lam0_dict, lamj_dict, L, vF):
    return (L / (np.pi * vF)) * np.log(abs(lam0_dict['abs']) / abs(lamj_dict['abs']))

def evaluate_fit(h_vals, L_vals):
    L_arr = np.array(L_vals)
    y = np.array(h_vals)
    X = np.column_stack([np.ones_like(L_arr), 1.0/L_arr, 1.0/(L_arr**2)])

    coeffs, residuals, _, _ = np.linalg.lstsq(X, y, rcond=None)

    if len(residuals) > 0:
        ssr = residuals[0]
    else:
        y_fit = X @ coeffs
        ssr = np.sum((y - y_fit)**2)

    return coeffs[0], ssr

def extrapolate_for_config(L_dict, vF, num_states=10):
    L_vals = sorted([int(L) for L in L_dict.keys() if L_dict[L] is not None])
    if len(L_vals) < 3:
        return None

    extracted_states = []

    for j in range(num_states):
        h_vals = []
        valid_Ls = []

        for L in L_vals:
            lam_list = L_dict[str(L)]
            if j < len(lam_list):
                lam0 = lam_list[0]
                lamj = lam_list[j]
                h_vals.append(compute_h(lam0, lamj, L, vF))
                valid_Ls.append(L)

        if len(valid_Ls) >= 3:
            h_extrap, ssr = evaluate_fit(h_vals, valid_Ls)
            extracted_states.append({
                "h_extrap": h_extrap,
                "ssr": ssr,
                "j": j,
                "Ls": valid_Ls
            })

    return extracted_states

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vF', type=float, default=1.0)
    parser.add_argument('--n', type=str, default=None)
    parser.add_argument('--num_states', type=int, default=20)
    args = parser.parse_args()

    in_file = "experiment_outputs/denseKuperberg/eigenvalue_logs_zigzag.json"
    if not os.path.exists(in_file):
        print(f"Log file {in_file} not found.")
        return

    logs = load_data(in_file)

    for t in logs:
        for order in logs[t]:
            for n_str in logs[t][order]:
                if args.n and n_str != args.n: continue

                L_dict = logs[t][order][n_str]
                results = extrapolate_for_config(L_dict, args.vF, num_states=args.num_states)

                if results:
                    print(f"\nConfiguration: Zigzag Transfer Matrix, n={n_str}")
                    print(f"{'Index j':<7} | {'Extrapolated h':<15} | {'SSR':<12} | {'Evaluated Ls'}")
                    print("-" * 65)
                    for res in results:
                        L_str = ",".join(map(str, res['Ls']))
                        print(f"{res['j']:<7} | {res['h_extrap']:<15.6f} | {res['ssr']:<12.2e} | {L_str}")

if __name__ == "__main__":
    main()
