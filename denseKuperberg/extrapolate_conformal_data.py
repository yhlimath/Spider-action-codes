import json
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt

def load_data(in_file):
    with open(in_file, 'r') as f:
        return json.load(f)

def compute_h(lam0_dict, lamj_dict, L, vF, operator):
    if operator == 'T':
        return (L / (np.pi * vF)) * np.log(abs(lam0_dict['abs']) / abs(lamj_dict['abs']))
    elif operator == 'H':
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

    return coeffs[0], ssr, coeffs

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
            h_extrap, ssr, coeffs = evaluate_fit(h_vals, valid_Ls)
            extracted_states.append({
                "h_extrap": h_extrap,
                "ssr": ssr,
                "j": j,
                "Ls": valid_Ls,
                "h_vals": h_vals,
                "coeffs": coeffs
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
    parser.add_argument('--plot', action='store_true', help="Plot the extrapolated fits")
    args = parser.parse_args()

    in_file = f"experiment_outputs/denseKuperberg/eigenvalue_logs_top_k_{args.operator}.json"
    if not os.path.exists(in_file):
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

                    if args.plot:
                        plt.figure(figsize=(10, 6))

                    for res in results:
                        L_str = ",".join(map(str, res['Ls']))
                        print(f"{res['j']:<7} | {res['h_extrap']:<15.6f} | {res['ssr']:<12.2e} | {L_str}")

                        if args.plot and res['j'] < 5:  # Plot top 5 to avoid clutter
                            h_vals = res['h_vals']
                            L_vals = res['Ls']
                            A, B, C = res['coeffs']

                            inv_L2 = 1.0 / (np.array(L_vals)**2)
                            p = plt.plot(inv_L2, h_vals, 'o', label=f"State {res['j']} (h={res['h_extrap']:.4f})")

                            L_continuous = np.linspace(min(L_vals)*0.9, max(L_vals)*1.1, 100)
                            h_fit = A + B/L_continuous + C/(L_continuous**2)
                            plt.plot(1.0/(L_continuous**2), h_fit, '-', color=p[0].get_color(), alpha=0.5)

                    if args.plot:
                        plt.title(f"Conformal Dimension Fits | {args.operator} | {t} | n={n_str}")
                        plt.xlabel("$1/L^2$")
                        plt.ylabel("$h_j(L)$")
                        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                        plt.grid(True)
                        plt.tight_layout()
                        safe_t = t.replace('+', '_').replace('(', '').replace(')', '').replace(',', '_')
                        plt.savefig(f"experiment_outputs/denseKuperberg/extrapolate_fit_{args.operator}_{safe_t}_{order}_n{n_str}.png", dpi=150)
                        plt.close()

if __name__ == "__main__":
    main()
