
import subprocess
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
import sys

def run_cpp_solver(L, n_val, operator, k_arnoldi=50):
    """
    Run the C++ solver and return the output data.
    """
    cmd = [
        "./cpp_implementation/main_scaling",
        str(L),
        str(np.real(n_val)),
        str(np.imag(n_val)),
        operator,
        str(k_arnoldi)
    ]

    print(f"Running C++: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        # Parse JSON from stdout
        # The C++ program might output debug info to stderr or stdout.
        # I used cout for JSON.
        # I need to ensure only JSON is in stdout or filter it.
        # My C++ code uses cout for "Arnoldi breakdown..." messages which are NOT JSON.
        # I should filter lines.

        lines = result.stdout.splitlines()
        json_str = ""
        capture = False
        brace_count = 0

        for line in lines:
            if line.strip() == "{":
                capture = True

            if capture:
                json_str += line + "\n"
                brace_count += line.count("{") - line.count("}")
                if brace_count == 0:
                    break

        if not json_str:
            print("Error: No JSON output found.")
            print("Stdout:", result.stdout)
            return None

        data = json.loads(json_str)

        # Reconstruct Hessenberg matrix
        h_json = data['hessenberg_matrix']
        h_mat = []
        for row_json in h_json:
            row = []
            for elem in row_json:
                row.append(complex(elem['real'], elem['imag']))
            h_mat.append(row)

        h_mat = np.array(h_mat)

        # Compute eigenvalues
        eigenvalues = np.linalg.eigvals(h_mat)

        data['eigenvalues_computed'] = sorted(eigenvalues, key=abs, reverse=True)
        return data

    except subprocess.CalledProcessError as e:
        print(f"C++ execution failed: {e}")
        print("Stderr:", e.stderr)
        return None
    except json.JSONDecodeError as e:
        print(f"JSON parsing failed: {e}")
        return None

def analyze_scaling_cpp(L_values, n_val, operator='H', output_filename="eigenvalues_log_cpp.json"):
    print(f"Starting Massive Finite-Size Scaling Analysis (C++ Backend) for n = {n_val} (Operator: {operator})")
    print("=" * 80)

    # Compile first
    print("Compiling C++ code...")
    subprocess.run(["make", "-C", "cpp_implementation"], check=True)

    results_data = {
        "timestamp": "",
        "n_value": float(np.real(n_val)) if np.isreal(n_val) else {"real": float(np.real(n_val)), "imag": float(np.imag(n_val))},
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
        # Determine k based on L?
        # For L=6, dim is large. k=50 is fine.
        # For small L, dim is small.
        k = 50

        data = run_cpp_solver(L, n_val, operator, k)

        if data:
            eigenvalues = data['eigenvalues_computed']
            dim = data['dim']
            eigenvalues_by_L[L] = eigenvalues

            if len(eigenvalues) > 0:
                Lambda_L = eigenvalues[0]
                print(f"  Leading eigenvalue (LM): {Lambda_L}")

                # Calculate f_L
                # Use magnitude if complex? Or real part of log?
                # Usually f_L = log(Lambda) / Volume.
                # If Lambda is complex, log is complex.
                # We take real part of f_L.

                val_log = np.log(Lambda_L)
                f_L = val_log / (3 * L)
                f_L_real = np.real(f_L)

                leading_eigenvalues.append(Lambda_L)
                f_L_values.append(f_L_real)
                L_array.append(L)

                print(f"  f_L (real part): {f_L_real}")

                # Store
                results_data["scaling_data"].append({
                    "L": L,
                    "dim": dim,
                    "leading_eigenvalue": {
                        "real": float(np.real(Lambda_L)),
                        "imag": float(np.imag(Lambda_L)),
                        "magnitude": float(abs(Lambda_L))
                    },
                    "f_L_real": float(f_L_real),
                    "top_k_eigenvalues": [
                        {"real": float(np.real(e)), "imag": float(np.imag(e))}
                        for e in eigenvalues[:min(10, len(eigenvalues))]
                    ]
                })
            else:
                print("  No eigenvalues found.")
        else:
            print(f"  Skipping L={L} due to error.")

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

        results_data["scaling_parameters"] = {
            "model": "f_L = A + B/L + C/L^2",
            "A_f_inf": float(f_inf_fit),
            "B_2f_sur": float(B_fit),
            "C_coeff": float(C_fit),
            "f_sur": float(f_sur_fit),
            "vF_c_product": float(vF_c_fit)
        }

        final_output_filename = output_filename.replace(".json", f"_{operator}.json")
        with open(final_output_filename, 'w') as f:
            json.dump(results_data, f, indent=4)
        print(f"Saved results to '{final_output_filename}'")

        # Plotting
        plt.figure(figsize=(10, 6))
        inv_L_squared = [1.0/(l**2) for l in L_array]
        plt.plot(inv_L_squared, f_L_values, 'o', label=f'Data ({operator})')

        x_fit = np.linspace(min(inv_L_squared)*0.9, max(inv_L_squared)*1.1, 100)
        L_fit = 1.0 / np.sqrt(x_fit)
        y_fit = scaling_model(L_fit, *popt)

        plt.plot(x_fit, y_fit, '-', label=f'Fit: v_F c={vF_c_fit:.4f}')
        plt.xlabel(r'$1/L^2$')
        plt.ylabel(r'$f_L = \log(\Lambda_L)/(3L)$')
        plt.title(f'Scaling (C++ Backend, n={n_val}, Op={operator})')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'scaling_fit_cpp_{operator}.png')

        # Distribution plot for max L
        if eigenvalues_by_L:
            max_L = max(eigenvalues_by_L.keys())
            evals = eigenvalues_by_L[max_L]
            plt.figure(figsize=(8, 8))
            plt.scatter(np.real(evals), np.imag(evals), marker='.', alpha=0.6)
            plt.xlabel(r'Re($\lambda$)')
            plt.ylabel(r'Im($\lambda$)')
            plt.title(f'Eigenvalue Distribution (L={max_L}, Op={operator})')
            plt.grid(True)
            plt.axis('equal')
            plt.savefig(f'eigenvalue_dist_cpp_{operator}.png')

if __name__ == "__main__":
    # Test with L up to 6?
    # L=2, 3, 4, 5, 6
    L_range = [2, 3, 4, 5]
    # n can be complex now.
    n_val = 1.0 + 0.0j

    analyze_scaling_cpp(L_range, n_val, operator='H')
    analyze_scaling_cpp(L_range, n_val, operator='T')
