import argparse
import numpy as np
import os
import time
import datetime
from sl3_hecke import Sl3HeckeArnoldi
from scipy.optimize import curve_fit
import json
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description="Sweep over n to estimate central charge for H or T operators.")
    parser.add_argument("-O", "--operator", choices=['H', 'T'], default='H', help="Operator to analyze (H or T)")
    parser.add_argument("-L", "--sizes", type=int, nargs="+", default=[2, 3, 4], help="List of system sizes L")
    parser.add_argument("--n_start", type=float, default=0.8, help="Start value of n")
    parser.add_argument("--n_end", type=float, default=1.3, help="End value of n (exclusive)")
    parser.add_argument("--n_step", type=float, default=0.1, help="Step size for n")
    parser.add_argument("-o", "--out_dir", type=str, default="experiment_outputs", help="Output directory")

    return parser.parse_args()

def compute_eigenvalues(n_range, L_values, operator):
    data_by_n = {}

    for n_val in n_range:
        n_val = round(n_val, 4)
        print(f"\nAnalyzing n = {n_val}")

        Lambda_list = []
        f_L_list = []
        valid_L = []

        for L in L_values:
            print(f"  L = {L}...", end="", flush=True)
            start_t = time.time()
            try:
                solver = Sl3HeckeArnoldi(L=L, n_value=n_val)
                k = min(50, solver.dim)

                h_mat = solver.arnoldi_iteration(k=k, operator=operator)
                eigenvalues = np.linalg.eigvals(h_mat)
                eigenvalues = sorted(eigenvalues, key=abs, reverse=True)

                Lambda = eigenvalues[0]

                if operator == 'T':
                    f_L = np.log(abs(Lambda)) / (3 * L)
                else:
                    f_L = np.real(Lambda) / (3 * L)

                all_eigs = []
                for e in eigenvalues:
                    all_eigs.append({
                        "real": float(np.real(e)),
                        "imag": float(np.imag(e)),
                        "abs": float(abs(e))
                    })

                Lambda_list.append({
                    "L": L,
                    "lambda_0": {"real": float(np.real(Lambda)), "imag": float(np.imag(Lambda)), "abs": float(abs(Lambda))},
                    "f_L": float(f_L),
                    "all_eigenvalues": all_eigs
                })
                f_L_list.append(f_L)
                valid_L.append(L)
                print(f" Done ({time.time()-start_t:.2f}s). f_L={f_L:.5f}")

            except Exception as e:
                print(f" Error: {e}")

        data_by_n[n_val] = {
            "valid_L": valid_L,
            "eigenvalues": Lambda_list,
            "f_L_values": f_L_list
        }

    return data_by_n

def scaling_model(L, A, B, C):
    return A + B/L + C/(L**2)

def fit_scaling(valid_L, f_L_values):
    if len(valid_L) < 3:
        return None

    try:
        popt, pcov = curve_fit(scaling_model, valid_L, f_L_values)
        A, B, C = popt

        f_inf = A
        f_sur = B / 2.0
        vFc = - (24 * C) / np.pi

        return {
            "popt": list(popt),
            "f_inf": float(f_inf),
            "f_sur": float(f_sur),
            "vFc": float(vFc)
        }
    except Exception as e:
        print(f"Fit failed: {e}")
        return None

def analyze_and_export(data_by_n, operator, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = os.path.join(out_dir, f"central_charge_sweep_{operator}_{timestamp}")

    n_plot = []
    vFc_plot = []

    results = {
        "operator": operator,
        "timestamp": timestamp,
        "data": {}
    }

    for n_val, data in data_by_n.items():
        fit_res = fit_scaling(data["valid_L"], data["f_L_values"])

        results["data"][str(n_val)] = {
            "L_values": data["valid_L"],
            "f_L_values": data["f_L_values"],
            "eigenvalues": data["eigenvalues"],
            "fit": fit_res
        }

        if fit_res:
            n_plot.append(n_val)
            vFc_plot.append(fit_res["vFc"])

    # Export JSON
    json_path = f"{prefix}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nExported detailed results to {json_path}")

    # Generate Plot
    if n_plot:
        plt.figure(figsize=(8, 6))
        plt.plot(n_plot, vFc_plot, 'o-', color='b')
        plt.xlabel('n (Loop Weight)')
        plt.ylabel(r'$v_F c$')
        plt.title(f'Estimated Central Charge Parameter vs n (Operator {operator})')
        plt.grid(True)

        plot_path = f"{prefix}.png"
        plt.savefig(plot_path)
        print(f"Exported plot to {plot_path}")

    return prefix, results, n_plot, vFc_plot

def export_mathematica(prefix, n_plot, vFc_plot, data_by_n, operator):
    m_path = f"{prefix}.m"
    with open(m_path, "w") as f:
        f.write(f"(* Central Charge Extrapolation Data for Operator {operator} *)\n\n")

        # Central charge vs n
        pairs = [f"{{{n}, {v}}}" for n, v in zip(n_plot, vFc_plot)]
        f.write(f"vFcData{operator} = {{{', '.join(pairs)}}};\n\n")

        # Detailed scaling data per n
        for n_val, data in data_by_n.items():
            valid_L = data["valid_L"]
            f_L = data["f_L_values"]
            if valid_L:
                pairs_L = [f"{{{L}, {f}}}" for L, f in zip(valid_L, f_L)]
                n_str = str(n_val).replace(".", "p")
                f.write(f"fLData{operator}n{n_str} = {{{', '.join(pairs_L)}}};\n")

    print(f"Exported Mathematica data to {m_path}")

def main():
    args = parse_args()
    print(f"Operator: {args.operator}")
    print(f"Sizes: {args.sizes}")
    n_range = np.arange(args.n_start, args.n_end, args.n_step)
    print(f"n_range: {n_range}")

    data_by_n = compute_eigenvalues(n_range, args.sizes, args.operator)
    prefix, results, n_plot, vFc_plot = analyze_and_export(data_by_n, args.operator, args.out_dir)
    export_mathematica(prefix, n_plot, vFc_plot, data_by_n, args.operator)

if __name__ == "__main__":
    main()
