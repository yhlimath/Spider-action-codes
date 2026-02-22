
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sl3_hecke import Sl3HeckeArnoldi
import time

def analyze_scaling(L_values, n_value):
    print(f"Starting Finite-Size Scaling Analysis for n = {n_value}")
    print("=" * 60)

    leading_eigenvalues = []
    f_L_values = []
    inv_L_squared_values = []

    eigenvalues_by_L = {}

    for L in L_values:
        print(f"\nProcessing L = {L} (System size 3L = {3*L})...")
        start_time = time.time()

        try:
            # Instantiate Solver
            solver = Sl3HeckeArnoldi(L=L, n_value=n_value)
            print(f"  Dimension of space: {solver.dim}")

            k = min(50, solver.dim - 2)
            if k < 1: k = 1

            eigenvalues = solver.run_arnoldi(k=k, which='LM')
            eigenvalues_by_L[L] = eigenvalues

            # Sort by magnitude descending
            eigenvalues = sorted(eigenvalues, key=lambda x: abs(x), reverse=True)
            Lambda_L = eigenvalues[0]

            print(f"  Leading eigenvalue (LM): {Lambda_L}")

            # Calculate f_L
            val_log = np.log(Lambda_L)
            f_L = val_log / (3 * L)

            f_L_real = np.real(f_L)

            leading_eigenvalues.append(Lambda_L)
            f_L_values.append(f_L_real)
            inv_L_squared_values.append(1.0 / (L**2))

            print(f"  f_L (real part): {f_L_real}")
            print(f"  Time elapsed: {time.time() - start_time:.2f}s")

        except Exception as e:
            print(f"  Error processing L={L}: {e}")
            # import traceback
            # traceback.print_exc()

    # Scaling Fit
    if len(f_L_values) >= 2:
        def linear_model(x, a, b):
            return a + b * x

        popt, pcov = curve_fit(linear_model, inv_L_squared_values, f_L_values)
        f_inf_fit = popt[0]
        slope_fit = popt[1]

        # Formula: f_L = f_inf - (pi * c) / (54 * L^2)
        # Slope B = - pi * c / 54
        c_estimated = - (54 * slope_fit) / np.pi

        print("\n" + "=" * 60)
        print("Finite-Size Scaling Results")
        print("=" * 60)
        print(f"Slope (B): {slope_fit}")
        print(f"Intercept (f_inf): {f_inf_fit}")
        print(f"Estimated Central Charge c: {c_estimated}")

        # Plotting Scaling
        plt.figure(figsize=(10, 6))
        plt.plot(inv_L_squared_values, f_L_values, 'o', label='Data')
        x_fit = np.linspace(min(inv_L_squared_values)*0.9, max(inv_L_squared_values)*1.1, 100)
        plt.plot(x_fit, linear_model(x_fit, *popt), '-', label=f'Fit: c={c_estimated:.4f}')
        plt.xlabel(r'$1/L^2$')
        plt.ylabel(r'$f_L = \log(\Lambda_L)/(3L)$')
        plt.title(f'Finite-Size Scaling (n={n_value})')
        plt.legend()
        plt.grid(True)
        plt.savefig('scaling_fit.png')
        print("Saved scaling plot to 'scaling_fit.png'")

        # Plotting Eigenvalue Distribution for largest L
        max_L = max(L_values)
        if max_L in eigenvalues_by_L:
            evals = eigenvalues_by_L[max_L]
            plt.figure(figsize=(8, 8))
            plt.scatter(np.real(evals), np.imag(evals), marker='.', alpha=0.6)
            plt.xlabel(r'Re($\lambda$)')
            plt.ylabel(r'Im($\lambda$)')
            plt.title(f'Eigenvalue Distribution (L={max_L}, top {len(evals)})')
            plt.grid(True)
            plt.axis('equal')
            plt.savefig('eigenvalue_dist.png')
            print("Saved eigenvalue distribution plot to 'eigenvalue_dist.png'")

        # Data Table
        print("\nData Table:")
        print(f"{'L':<5} | {'1/L^2':<10} | {'Lambda_L (Leading)':<30} | {'f_L (Real)':<15}")
        print("-" * 70)
        for i, L in enumerate(L_values):
            if i < len(leading_eigenvalues):
                lam = leading_eigenvalues[i]
                f = f_L_values[i]
                inv_l2 = inv_L_squared_values[i]
                print(f"{L:<5} | {inv_l2:<10.5f} | {str(lam):<30} | {f:<15.6f}")

    else:
        print("Not enough data points for fit.")

if __name__ == "__main__":
    # L=5 added. Note: dim for L=5 is 5810 (found via OEIS A000000 or logic? 462 * ?).
    # Catalan numbers logic?
    # Actually for sl3 it's generalized Catalan.
    # L=5 might be around 6000. It should be fine.
    L_range = [2, 3, 4, 5]
    n_val = 1.0

    analyze_scaling(L_range, n_val)
