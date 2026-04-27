import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sl3hecke.sl3_hecke import Sl3HeckeArnoldi
import time
import json
import datetime

def fit_quadratic(L_values, f_values):
    # Model: f(L) = A + B/L + C/L^2
    # x1 = 1/L, x2 = 1/L^2
    # We use curve_fit

    def model(L, A, B, C):
        return A + B/L + C/(L**2)

    try:
        popt, pcov = curve_fit(model, L_values, f_values)
        return popt # [A, B, C]
    except Exception as e:
        print(f"Fit failed: {e}")
        return None

def analyze_transfer_matrix(L_values=[2, 3, 4], n_range=np.arange(0.8, 1.3, 0.1)):
    print("Starting Transfer Matrix Analysis")
    print("=" * 60)

    results = {
        "timestamp": datetime.datetime.now().isoformat(),
        "L_values": L_values,
        "data_by_n": {}
    }

    # Store aggregated metrics for phase diagram
    n_plot = []
    vFc_plot = []
    f_sur_plot = []
    f_inf_plot = []

    plt.figure(figsize=(10, 7))

    for n_val in n_range:
        # Round n to avoid floating point mess in keys
        n_val = round(n_val, 2)
        print(f"\nAnalyzing n = {n_val}")

        f_L_list = []
        Lambda_list = []

        # 1. Compute Eigenvalues
        for L in L_values:
            print(f"  L = {L}...", end="", flush=True)
            start_t = time.time()
            try:
                solver = Sl3HeckeArnoldi(L=L, n_value=n_val)
                k = min(50, solver.dim)

                # Arnoldi
                h_mat = solver.arnoldi_iteration(k=k, operator='T')
                eigenvalues = np.linalg.eigvals(h_mat)
                eigenvalues = sorted(eigenvalues, key=abs, reverse=True)

                Lambda = eigenvalues[0]
                # f_L = log(Lambda) / (3L)
                # Take real part of log(Lambda) -> log(|Lambda|)
                f_L = np.log(abs(Lambda)) / (3 * L)

                Lambda_list.append({
                    "real": float(np.real(Lambda)),
                    "imag": float(np.imag(Lambda)),
                    "abs": float(abs(Lambda))
                })
                f_L_list.append(f_L)
                print(f" Done ({time.time()-start_t:.2f}s). f_L={f_L:.5f}")

            except Exception as e:
                print(f" Error: {e}")
                Lambda_list.append(None)
                f_L_list.append(None)

        # 2. Fit
        # Filter None
        valid_indices = [i for i, x in enumerate(f_L_list) if x is not None]
        valid_L = [L_values[i] for i in valid_indices]
        valid_f = [f_L_list[i] for i in valid_indices]

        if len(valid_L) >= 3:
            popt = fit_quadratic(np.array(valid_L), np.array(valid_f))
            if popt is not None:
                A, B, C = popt
                f_inf = A
                f_sur = B / 2.0
                vFc = - (24 * C) / np.pi

                results["data_by_n"][str(n_val)] = {
                    "fit_params": {"A": A, "B": B, "C": C},
                    "physical_params": {"f_inf": f_inf, "f_sur": f_sur, "vFc": vFc},
                    "L": valid_L,
                    "f_L": valid_f,
                    "eigenvalues_top": Lambda_list
                }

                n_plot.append(n_val)
                vFc_plot.append(vFc)
                f_sur_plot.append(f_sur)
                f_inf_plot.append(f_inf)

                # 3. Scaling Plot (f_L - 2f_sur/L vs 1/L^2)
                # y = f_L - B/L = A + C/L^2
                # x = 1/L^2

                inv_L2 = [1.0/(l**2) for l in valid_L]
                y_corrected = [f - B/l for f, l in zip(valid_f, valid_L)]

                # Fit line for plot: A + C*x
                x_line = np.linspace(0, max(inv_L2)*1.1, 100)
                y_line = A + C * x_line

                p = plt.plot(inv_L2, y_corrected, 'o', label=f'n={n_val}')
                color = p[0].get_color()
                plt.plot(x_line, y_line, '--', color=color)

        else:
            print("  Not enough data for fit.")

    # Finalize Scaling Plot
    plt.xlabel(r'$1/L^2$')
    plt.ylabel(r'$f_L - 2f_{sur}/L$')
    plt.title('Scaling of Free Energy Density (Transfer Matrix)')
    plt.legend()
    plt.grid(True)
    plt.savefig('tm_scaling_diagram.png')
    print("Saved 'tm_scaling_diagram.png'")

    # Phase Diagram Plots
    if n_plot:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # vFc
        axes[0].plot(n_plot, vFc_plot, 'o-')
        axes[0].set_xlabel('n')
        axes[0].set_ylabel(r'$v_F c$')
        axes[0].set_title('Effective Central Charge Proxy')
        axes[0].grid(True)

        # f_sur
        axes[1].plot(n_plot, f_sur_plot, 's-')
        axes[1].set_xlabel('n')
        axes[1].set_ylabel(r'$f_{sur}$')
        axes[1].set_title('Surface Free Energy')
        axes[1].grid(True)

        # f_inf
        axes[2].plot(n_plot, f_inf_plot, '^-')
        axes[2].set_xlabel('n')
        axes[2].set_ylabel(r'$f_{\infty}$')
        axes[2].set_title('Bulk Free Energy Density')
        axes[2].grid(True)

        plt.tight_layout()
        plt.savefig('tm_phase_diagram.png')
        print("Saved 'tm_phase_diagram.png'")

    # Save log
    with open('transfer_matrix_log.json', 'w') as f:
        json.dump(results, f, indent=4)
    print("Saved 'transfer_matrix_log.json'")

if __name__ == "__main__":
    # Define range
    n_values = np.arange(0.8, 1.25, 0.1) # 0.8, 0.9, 1.0, 1.1, 1.2
    # L values: 2, 3, 4. (L=5 takes too long for interaction loop, ~2.5 mins per point * 5 points = 12 mins. Might be ok? Let's stick to 2,3,4 for speed unless user insists on precision)
    # The user said "I'd like to set also the size L".
    # I will use 2,3,4 for quick response.
    analyze_transfer_matrix(L_values=[2, 3, 4], n_range=n_values)
