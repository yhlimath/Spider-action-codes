import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import sys
import os

def analyze_json(filename):
    print(f"Analyzing data from: {filename}")
    print("=" * 60)

    if not os.path.exists(filename):
        print(f"Error: File '{filename}' not found.")
        return

    with open(filename, 'r') as f:
        data = json.load(f)

    operator = data.get("operator", "Unknown")
    n_val = data.get("n_value", "Unknown")
    scaling_data = data.get("data_by_n", [])

    if not scaling_data:
        print("No scaling data found in JSON.")
        return

    # Extract Data
    L_list = []
    f_L_list = []

    for entry in scaling_data:
        L = entry['L']
        # Use stored f_L_real if available, otherwise compute
        if 'f_L_real' in entry:
            f_L = entry['f_L_real']
        elif 'leading_eigenvalue' in entry:
            # Recompute
            ev = entry['leading_eigenvalue']
            # mag = ev['magnitude'] # or calculate from real/imag
            # f_L = log(mag) / (3L)
            # Let's rely on stored value if possible for consistency
            pass
        else:
            continue

        L_list.append(L)
        f_L_list.append(f_L)

    if len(L_list) < 3:
        print(f"Not enough data points ({len(L_list)}) for quadratic fit.")
        return

    # Sort by L
    sorted_pairs = sorted(zip(L_list, f_L_list))
    L_array = np.array([p[0] for p in sorted_pairs])
    f_L_array = np.array([p[1] for p in sorted_pairs])

    print("Data loaded:")
    print(f"{'L':<5} | {'f_L':<15}")
    print("-" * 25)
    for l, f in zip(L_array, f_L_array):
        print(f"{l:<5} | {f:<15.6f}")

    # Fit Model: f(L) = A + B/L + C/L^2
    def scaling_model(L, A, B, C):
        return A + B/L + C/(L**2)

    try:
        popt, pcov = curve_fit(scaling_model, L_array, f_L_array)
        A, B, C = popt

        f_inf = A
        f_sur = B / 2.0
        vFc = - (24 * C) / np.pi

        print("\n" + "=" * 60)
        print("Finite-Size Scaling Results (Quadratic Fit)")
        print("=" * 60)
        print(f"Model: f(L) = A + B/L + C/L^2")
        print(f"A (f_inf): {f_inf}")
        print(f"B (2*f_sur): {B} => f_sur = {f_sur}")
        print(f"C (-pi*v_F*c/24): {C} => v_F*c = {vFc}")

        # Plots
        # Corrected scaling: y = f_L - B/L vs 1/L^2
        y_corrected = f_L_array - B / L_array
        inv_L_squared = 1.0 / (L_array ** 2)

        plt.figure(figsize=(10, 6))
        plt.plot(inv_L_squared, y_corrected, 'o', label=f'Data corrected ({operator})')

        x_fit = np.linspace(min(inv_L_squared)*0.9, max(inv_L_squared)*1.1, 100)
        # Line is A + C*x
        y_fit = A + C * x_fit

        plt.plot(x_fit, y_fit, '-', label=f'Fit: v_F c={vFc:.4f}')

        plt.xlabel(r'$1/L^2$')
        plt.ylabel(r'$f_L - 2 f_{sur}/L$')
        plt.title(f'Corrected Scaling Analysis from Log (n={n_val}, Op={operator})')
        plt.legend()
        plt.grid(True)

        plot_filename = f"reanalysis_corrected_{operator}.png"
        plt.savefig(plot_filename)
        print(f"\nSaved corrected scaling plot to '{plot_filename}'")

        # Mathematica Output
        mathematica_pairs = []
        for l, f in zip(L_array, f_L_array):
            mathematica_pairs.append(f"{{{l}, {f}}}")

        mathematica_str = "{" + ", ".join(mathematica_pairs) + "}"
        print(f"\nMathematica Data List:\n{mathematica_str}")

    except Exception as e:
        print(f"Fit failed: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        analyze_json(filename)
    else:
        # Default test if no arg
        default_file = "eigenvalues_log_T.json"
        if os.path.exists(default_file):
            analyze_json(default_file)
        else:
            print("Usage: python analyze_from_log.py <json_filename>")
