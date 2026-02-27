
import json
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

def analyze_phase_diagram(json_file, n_min=None, n_max=None):
    print(f"Analyzing phase diagram from: {json_file}")
    if n_min is not None and n_max is not None:
        print(f"Filtering for n in range [{n_min}, {n_max}]")
    print("=" * 60)

    if not os.path.exists(json_file):
        print(f"Error: File '{json_file}' not found.")
        return

    with open(json_file, 'r') as f:
        data = json.load(f)

    # The structure of transfer_matrix_log.json is:
    # { "data_by_n": { "0.8": { "physical_params": {...} }, ... } }

    data_by_n = data.get("data_by_n", {})
    if not data_by_n:
        print("No 'data_by_n' found in JSON.")
        return

    n_list = []
    vFc_list = []
    f_sur_list = []
    f_inf_list = []

    # Sort keys as floats
    sorted_keys = sorted(data_by_n.keys(), key=lambda x: float(x))

    for key in sorted_keys:
        n_val = float(key)

        # Filter range
        if n_min is not None and n_val < n_min: continue
        if n_max is not None and n_val > n_max: continue

        entry = data_by_n[key]
        params = entry.get("physical_params", {})

        if "vFc" in params:
            n_list.append(n_val)
            vFc_list.append(params["vFc"])
            f_sur_list.append(params["f_sur"])
            f_inf_list.append(params["f_inf"])

    if not n_list:
        print("No data found in the specified range.")
        return

    # Output Log of Central Charge
    print(f"\nCentral Charge Estimate (v_F * c) vs n:")
    print(f"{'n':<10} | {'v_F * c':<15} | {'f_sur':<15} | {'f_inf':<15}")
    print("-" * 60)

    log_data = []
    mathematica_pairs = []

    for i in range(len(n_list)):
        n = n_list[i]
        c = vFc_list[i]
        print(f"{n:<10.2f} | {c:<15.6f} | {f_sur_list[i]:<15.6f} | {f_inf_list[i]:<15.6f}")
        log_data.append({"n": n, "vFc": c, "f_sur": f_sur_list[i], "f_inf": f_inf_list[i]})
        mathematica_pairs.append(f"{{{n}, {c}}}")

    # Save Log
    log_filename = "central_charge_log.json"
    with open(log_filename, 'w') as f:
        json.dump(log_data, f, indent=4)
    print(f"\nSaved numerical log to '{log_filename}'")

    # Save Mathematica
    m_filename = "central_charge_vs_n.m"
    m_str = "{" + ", ".join(mathematica_pairs) + "}"
    with open(m_filename, 'w') as f:
        f.write("(* Data: {n, v_F * c} *)\n")
        f.write(f"centralChargeData = {m_str};\n")
    print(f"Saved Mathematica data to '{m_filename}'")

    # Plot Phase Diagram
    plt.figure(figsize=(10, 6))
    plt.plot(n_list, vFc_list, 'o-', label=r'$v_F c$')
    plt.xlabel('n')
    plt.ylabel(r'$v_F c$')
    plt.title(f'Phase Diagram: Effective Central Charge vs n')
    plt.grid(True)
    plt.legend()

    plot_filename = "phase_diagram_vFc.png"
    plt.savefig(plot_filename)
    print(f"Saved phase diagram plot to '{plot_filename}'")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Analyze phase diagram from JSON log.')
    parser.add_argument('file', nargs='?', default='transfer_matrix_log.json', help='JSON log file path')
    parser.add_argument('--min', type=float, help='Minimum n value')
    parser.add_argument('--max', type=float, help='Maximum n value')

    args = parser.parse_args()

    analyze_phase_diagram(args.file, args.min, args.max)
