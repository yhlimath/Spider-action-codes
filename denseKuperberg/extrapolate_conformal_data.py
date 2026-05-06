import json
import numpy as np
import os
import argparse

def load_data(in_file):
    with open(in_file, 'r') as f:
        return json.load(f)

def compute_h(lam0_dict, lamj_dict, L, vF, operator):
    if operator == 'T':
        return (L / (np.pi * vF)) * np.log(abs(lam0_dict['abs']) / abs(lamj_dict['abs']))
    elif operator == 'H':
        # the formula from user prompt: h_j = (L / (\pi v_F)) Re(lam_0 - lam_j)
        # However, memory says: h_j = (L/(2\pi v_F)) (\text{Re}(\lambda_0) - \text{Re}(\lambda_j))
        # Let's use the one from memory which matches standard Hecke derivations. Wait, prompt said:
        # "h_j = (L / (\pi v_F)) Re(lam_0 - lam_j)"? No, prompt didn't specify. I wrote that in the plan.
        # Let's use the one from memory which is standard. Actually sl3hecke memory says:
        # "h_j = (L/(2\pi v_F)) (\text{Re}(\lambda_0) - \text{Re}(\lambda_j))"
        return (L / (2 * np.pi * vF)) * (lam0_dict['real'] - lamj_dict['real'])
    return 0.0

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

def extrapolate_for_config(L_dict, vF, operator, num_states=10):
    L_vals = sorted([int(L) for L in L_dict.keys() if int(L) >= 3 and L_dict[L] is not None])
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
                h_vals.append(compute_h(lam0, lamj, L, vF, operator))
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
    parser = argparse.ArgumentParser(description="Extrapolate conformal dimensions using strict rank-by-moduli")
    parser.add_argument('--vF', type=float, default=1.0, help="Fermi velocity to use for extrapolation")
    parser.add_argument('--type', type=str, default=None, help="Filter by matrix type (e.g. T(x,y,z))")
    parser.add_argument('--order', type=str, default=None, help="Filter by order (e.g. staggered)")
    parser.add_argument('--n', type=str, default=None, help="Filter by weight n (e.g. 1.0)")
    parser.add_argument('--operator', type=str, default='T', help="'T' for Transfer Matrix or 'H' for Hamiltonian")
    parser.add_argument('--num_states', type=int, default=20, help="Number of conformal states to extract")
    args = parser.parse_args()

    in_file = f"experiment_outputs/denseKuperberg/eigenvalue_logs_top_k_{args.operator}.json"
    if not os.path.exists(in_file):
        # Fall back to original name if testing older data
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
                results = extrapolate_for_config(L_dict, args.vF, args.operator, num_states=args.num_states)

                if results:
                    print(f"\nConfiguration: Operator={args.operator}, Type={t}, Order={order}, n={n_str}")
                    print(f"{'Index j':<7} | {'Extrapolated h':<15} | {'SSR':<12} | {'Evaluated Ls'}")
                    print("-" * 65)
                    for res in results:
                        L_str = ",".join(map(str, res['Ls']))
                        print(f"{res['j']:<7} | {res['h_extrap']:<15.6f} | {res['ssr']:<12.2e} | {L_str}")

if __name__ == "__main__":
    main()
