import json
import numpy as np
import os
import matplotlib.pyplot as plt

def analyze_logs():
    in_file = "experiment_outputs/denseKuperberg/eigenvalue_logs.json"
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
                # Filter L values where lambda_0 is present
                L_vals = []
                f_L_vals = []
                for L_s, lam_data in L_dict.items():
                    if lam_data is not None:
                        L = int(L_s)
                        lambda_abs = lam_data['abs']
                        f_L = - np.log(lambda_abs) / L
                        L_vals.append(L)
                        f_L_vals.append(f_L)

                if len(L_vals) >= 3:
                    # Fit f_L = A + B/L + C/L^2
                    L_arr = np.array(L_vals)
                    y = np.array(f_L_vals)

                    X = np.column_stack([np.ones_like(L_arr), 1.0/L_arr, 1.0/(L_arr**2)])

                    try:
                        coeffs, residuals, _, _ = np.linalg.lstsq(X, y, rcond=None)
                        A, B, C = coeffs

                        print(f"Type: {t:8s} | Order: {order:10s} | n={n_str:4s} | A={A:7.4f}, B={B:7.4f}, C={C:7.4f} (from L={min(L_vals)}-{max(L_vals)})")

                        # We could plot it but let's just log the parameters.
                    except Exception as e:
                        print(f"Fit failed for {t} {order} n={n_str}: {e}")

if __name__ == "__main__":
    analyze_logs()
def plot_central_charge_extrapolations():
    import json, os, numpy as np
    import matplotlib.pyplot as plt

    in_file = "experiment_outputs/denseKuperberg/eigenvalue_logs.json"
    if not os.path.exists(in_file): return
    with open(in_file, 'r') as f:
        logs = json.load(f)

    out_dir = "experiment_outputs/denseKuperberg"

    # Let's plot for staggered and sequential, E+H+H2 and E+H
    for order in ['sequential', 'staggered']:
        for t in ['E+H+H2', 'E+H']:
            plt.figure(figsize=(10, 6))

            for n_str in logs[t][order]:
                L_dict = logs[t][order][n_str]
                L_vals = []
                f_L_vals = []
                for L_s, lam_data in L_dict.items():
                    if lam_data is not None:
                        L = int(L_s)
                        lambda_abs = lam_data['abs']
                        f_L = - np.log(lambda_abs) / L
                        L_vals.append(L)
                        f_L_vals.append(f_L)

                if len(L_vals) >= 3:
                    L_arr = np.array(L_vals)
                    inv_L2 = 1.0 / (L_arr**2)
                    y = np.array(f_L_vals)

                    X = np.column_stack([np.ones_like(L_arr), 1.0/L_arr, 1.0/(L_arr**2)])
                    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                    A, B, C = coeffs

                    # Plot f_L - B/L against 1/L^2 to see if it's linear with slope C and intercept A
                    # This corrects for the 1/L term which is non-universal boundary effect
                    y_corrected = y - B/L_arr

                    plt.plot(inv_L2, y_corrected, 'o-', label=f"n={n_str} (C={C:.2f})")

            plt.title(f"Scaling $f_L - B/L$ vs $1/L^2$ | {t} | {order}")
            plt.xlabel("$1/L^2$")
            plt.ylabel("$f_L - B/L$")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"cc_{t}_{order}.png"))
            plt.close()

if __name__ == "__main__":
    plot_central_charge_extrapolations()
