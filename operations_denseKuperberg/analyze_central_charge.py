import json
import numpy as np
import os
import matplotlib.pyplot as plt

def analyze_logs():
    in_file = "experiment_outputs/denseKuperberg/eigenvalue_logs_zigzag.json"
    if not os.path.exists(in_file):
        print(f"Log file {in_file} not found.")
        return

    with open(in_file, 'r') as f:
        logs = json.load(f)

    out_dir = "experiment_outputs/denseKuperberg"
    os.makedirs(out_dir, exist_ok=True)

    types = list(logs.keys())

    for t in types:
        for order in logs[t]:
            for n_str in logs[t][order]:
                L_dict = logs[t][order][n_str]
                L_vals = []
                f_L_vals = []
                for L_s, lam_list in L_dict.items():
                    if lam_list is not None and len(lam_list) > 0:
                        L = int(L_s)
                        lambda_abs = lam_list[0]['abs']
                        f_L = (2/np.sqrt(3)) * np.log(lambda_abs) / L
                        L_vals.append(L)
                        f_L_vals.append(f_L)

                if len(L_vals) >= 3:
                    L_arr = np.array(L_vals)
                    y = np.array(f_L_vals)

                    X = np.column_stack([np.ones_like(L_arr), 1.0/L_arr, 1.0/(L_arr**2)])

                    try:
                        coeffs, residuals, _, _ = np.linalg.lstsq(X, y, rcond=None)
                        A, B, C = coeffs
                        print(f"Type: {t:8s} | Order: {order:10s} | n={n_str:4s} | A={A:7.4f}, B={B:7.4f}, C={C:7.4f} (from L={min(L_vals)}-{max(L_vals)})")
                    except Exception as e:
                        pass

def plot_central_charge_extrapolations():
    in_file = "experiment_outputs/denseKuperberg/eigenvalue_logs_zigzag.json"
    if not os.path.exists(in_file):
        print(f"Log file {in_file} not found.")
        return
    with open(in_file, 'r') as f:
        logs = json.load(f)

    out_dir = "experiment_outputs/denseKuperberg"
    os.makedirs(out_dir, exist_ok=True)

    for t, orders in logs.items():
        for order, n_dict in orders.items():
            plt.figure(figsize=(10, 6))
            plotted_any = False

            for n_str, L_dict in n_dict.items():
                L_vals = []
                f_L_vals = []
                for L_s, lam_list in L_dict.items():
                    if lam_list is not None and len(lam_list) > 0:
                        L = int(L_s)
                        lambda_abs = lam_list[0]['abs']
                        f_L = - np.log(lambda_abs) / L
                        L_vals.append(L)
                        f_L_vals.append(f_L)

                if len(L_vals) >= 3:
                    L_arr = np.array(L_vals)
                    sort_idx = np.argsort(L_arr)
                    L_arr = L_arr[sort_idx]
                    f_L_vals = np.array(f_L_vals)[sort_idx]
                    inv_L2 = 1.0 / (L_arr**2)
                    y = f_L_vals

                    X = np.column_stack([np.ones_like(L_arr), 1.0/L_arr, 1.0/(L_arr**2)])
                    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                    A, B, C = coeffs

                    p = plt.plot(inv_L2, y, 'o', label=f"n={n_str}")
                    color = p[0].get_color()

                    L_continuous = np.linspace(min(L_vals)*0.9, max(L_vals)*1.1, 100)
                    y_fit = A + B/L_continuous + C/(L_continuous**2)
                    plt.plot(1.0/(L_continuous**2), y_fit, '-', color=color, alpha=0.5)
                    plotted_any = True

            if not plotted_any:
                plt.close()
                continue

            plt.title(f"Scaling $f_L$ vs $1/L^2$ | {t} | {order}")
            plt.xlabel("$1/L^2$")
            plt.ylabel("$f_L$")
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True)
            plt.tight_layout()
            safe_t = t.replace('+', '_').replace('(', '').replace(')', '').replace(',', '_')
            out_file = os.path.join(out_dir, f"cc_fL_vs_L2_{safe_t}_{order}.png")
            plt.savefig(out_file, dpi=200)
            plt.close()
            print(f"Saved {out_file}")

if __name__ == "__main__":
    analyze_logs()
    plot_central_charge_extrapolations()
