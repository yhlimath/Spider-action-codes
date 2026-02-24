import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sl3_hecke import Sl3HeckeArnoldi
import time
import json
import datetime

def analyze_scaling(L_values, n_value, operator='H', output_filename="eigenvalues_log.json"):
    print(f"Starting Finite-Size Scaling Analysis for n = {n_value} (Operator: {operator})")
    print("=" * 60)

    # Data structure to hold results
    results_data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "n_value": n_value,
        "operator": operator,
        "L_values": L_values,
        "scaling_data": [],
        "scaling_parameters": None
    }

    leading_eigenvalues = []
    f_L_values = []
    L_array = []

    eigenvalues_by_L = {}

    for L in L_values:
        print(f"\nProcessing L = {L} (System size 3L = {3*L})...")
        start_time = time.time()

        try:
            # Instantiate Solver
            solver = Sl3HeckeArnoldi(L=L, n_value=n_value)
            print(f"  Dimension of space: {solver.dim}")

            k_target = 50
            if solver.dim <= k_target:
                k_arnoldi = solver.dim
            else:
                k_arnoldi = k_target

            print(f"  Using Arnoldi k = {k_arnoldi}")

            # Use custom Arnoldi iteration
            hessenberg_mat = solver.arnoldi_iteration(k=k_arnoldi, operator=operator)

            # Compute eigenvalues of Hessenberg matrix
            eigenvalues = np.linalg.eigvals(hessenberg_mat)
            eigenvalues_by_L[L] = eigenvalues

            # Sort by magnitude descending
            eigenvalues_sorted = sorted(eigenvalues, key=lambda x: abs(x), reverse=True)
            Lambda_L = eigenvalues_sorted[0]

            print(f"  Leading eigenvalue (LM): {Lambda_L}")

            # Calculate f_L
            val_log = np.log(Lambda_L)
            f_L = val_log / (3 * L)

            f_L_real = np.real(f_L)

            leading_eigenvalues.append(Lambda_L)
            f_L_values.append(f_L_real)
            L_array.append(L)

            print(f"  f_L (real part): {f_L_real}")
            print(f"  Time elapsed: {time.time() - start_time:.2f}s")

            # Store data for this L
            results_data["scaling_data"].append({
                "L": L,
                "dim": solver.dim,
                "leading_eigenvalue": {
                    "real": float(np.real(Lambda_L)),
                    "imag": float(np.imag(Lambda_L)),
                    "magnitude": float(abs(Lambda_L))
                },
                "f_L_real": float(f_L_real),
                "top_k_eigenvalues": [
                    {"real": float(np.real(e)), "imag": float(np.imag(e))}
                    for e in eigenvalues_sorted[:min(10, len(eigenvalues_sorted))]
                ]
            })

        except Exception as e:
            print(f"  Error processing L={L}: {e}")
            import traceback
            traceback.print_exc()

    # Scaling Fit
    if len(f_L_values) >= 3:
        def scaling_model(L, A, B, C):
            return A + B/L + C/(L**2)

        popt, pcov = curve_fit(scaling_model, L_array, f_L_values)

        f_inf_fit = popt[0]
        B_fit = popt[1]
        C_fit = popt[2]

        f_sur_fit = B_fit / 2.0
        vF_c_fit = - (24 * C_fit) / np.pi

        print("\n" + "=" * 60)
        print("Finite-Size Scaling Results (Quadratic Fit)")
        print("=" * 60)
        print(f"Model: f(L) = A + B/L + C/L^2")
        print(f"A (f_inf): {f_inf_fit}")
        print(f"B (2*f_sur): {B_fit} => f_sur = {f_sur_fit}")
        print(f"C (-pi*v_F*c/24): {C_fit} => v_F*c = {vF_c_fit}")
        print(f"Variance of fit parameters: {pcov}")

        results_data["scaling_parameters"] = {
            "model": "f_L = A + B/L + C/L^2",
            "A_f_inf": float(f_inf_fit),
            "B_2f_sur": float(B_fit),
            "C_coeff": float(C_fit),
            "f_sur": float(f_sur_fit),
            "vF_c_product": float(vF_c_fit)
        }

        # Save to JSON file (append or overwrite? If running both H and T, append is better but user wants file)
        # I'll use filename based on operator to separate them, or just one file per run.
        # User said "in a file...".
        # I'll prepend operator to filename or user specifies it.
        # Default filename provided, but I can modify it.

        final_output_filename = output_filename.replace(".json", f"_{operator}.json")

        with open(final_output_filename, 'w') as f:
            json.dump(results_data, f, indent=4)
        print(f"Saved eigenvalues and results to '{final_output_filename}'")

        # Plotting Scaling
        plt.figure(figsize=(10, 6))

        inv_L_squared = [1.0/(l**2) for l in L_array]
        inv_L = [1.0/l for l in L_array]

        plt.plot(inv_L_squared, f_L_values, 'o', label=f'Data ({operator})')

        x_fit = np.linspace(min(inv_L_squared)*0.9, max(inv_L_squared)*1.1, 100)
        L_fit = 1.0 / np.sqrt(x_fit)
        y_fit = scaling_model(L_fit, *popt)

        plt.plot(x_fit, y_fit, '-', label=f'Fit: v_F c={vF_c_fit:.4f}')

        plt.xlabel(r'$1/L^2$')
        plt.ylabel(r'$f_L = \log(\Lambda_L)/(3L)$')
        plt.title(f'Finite-Size Scaling (n={n_value}, Operator={operator})')
        plt.legend()
        plt.grid(True)
        plot_filename = f'scaling_fit_{operator}.png'
        plt.savefig(plot_filename)
        print(f"Saved scaling plot to '{plot_filename}'")

        # Plotting Eigenvalue Distribution for largest L
        max_L = max(L_values)
        if max_L in eigenvalues_by_L:
            evals = eigenvalues_by_L[max_L]
            plt.figure(figsize=(8, 8))
            plt.scatter(np.real(evals), np.imag(evals), marker='.', alpha=0.6)
            plt.xlabel(r'Re($\lambda$)')
            plt.ylabel(r'Im($\lambda$)')
            plt.title(f'Eigenvalue Distribution (L={max_L}, Operator={operator})')
            plt.grid(True)
            plt.axis('equal')
            dist_filename = f'eigenvalue_dist_{operator}.png'
            plt.savefig(dist_filename)
            print(f"Saved eigenvalue distribution plot to '{dist_filename}'")

        # Data Table
        print("\nData Table:")
        print(f"{'L':<5} | {'1/L^2':<10} | {'Lambda_L (Leading)':<30} | {'f_L (Real)':<15}")
        print("-" * 70)
        for i, L in enumerate(L_values):
            if i < len(leading_eigenvalues):
                lam = leading_eigenvalues[i]
                f = f_L_values[i]
                inv_l2 = 1.0 / (L**2)
                print(f"{L:<5} | {inv_l2:<10.5f} | {str(lam):<30} | {f:<15.6f}")

    else:
        print("Not enough data points for fit (need at least 3).")

if __name__ == "__main__":
    L_range = [2, 3, 4, 5, 6, 7]
    n_val = 0.5

    # Analyze H
    #analyze_scaling(L_range, n_val, operator='H')

    # Analyze T
    print("\n" + "#" * 80 + "\n")
    analyze_scaling(L_range, n_val, operator='T')
