import sys
import os
import json
import argparse
import numpy as np
from scipy.optimize import curve_fit

def parse_complex(val):
    if isinstance(val, dict):
        return complex(val["re"], val["im"])
    return complex(val)

def load_data(file_paths):
    data_by_L = {}
    for path in file_paths:
        if not os.path.exists(path):
            print(f"Warning: File {path} not found. Skipping.")
            continue
        with open(path, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: Could not parse {path} as JSON. Skipping.")
                continue
            L = data.get("L")
            if L is None:
                print(f"Warning: JSON {path} does not contain 'L'. Skipping.")
                continue
            #if L % 3 != 0: # Change: We can still use data from non-multiple-of-3 L for conformal dimension extrapolation, just not for central charge. So we won't skip them entirely.
                #print(f"Warning: L={L} is not a multiple of 3 in {path}. Skipping.")
                #continue
            data_by_L[L] = data
    return data_by_L

def scaling_model(L, A, B, C):
    return A + B/L + C/(L**2)

def linear_model(inv_L, intercept, slope):
    return intercept + slope * inv_L

def extrapolate_central_charge(data_by_L):
    L_list = []
    f_L_list = []
    lambda_0_dict = {}

    for L in sorted(data_by_L.keys()):
        data = data_by_L[L]
        modules = data.get("modules", {})
        if "x=0, y=0" not in modules:
            print(f"Warning: Vacuum sector 'x=0, y=0' not found for L={L}.")
            continue

        vacuum_eigs = [parse_complex(e) for e in modules["x=0, y=0"]["eigenvalues"]]
        if not vacuum_eigs:
            print(f"Warning: No eigenvalues in vacuum sector for L={L}.")
            continue

        # Lambda_0 is the eigenvalue with largest modulus
        lambda_0_cplx = max(vacuum_eigs, key=abs)
        lambda_0 = np.abs(lambda_0_cplx)
        lambda_0_dict[L] = lambda_0

        # Free energy f_L = 1/(3L) * ln |lambda_0| #Change: The log formula is for T
        f_L = np.log(lambda_0) / (3*L)
        #The next formula is for H
        #f_L = np.real(lambda_0) / (3*L)

        L_list.append(L)
        f_L_list.append(f_L)

    results = {
        "L_values": L_list,
        "f_L_values": f_L_list,
        "lambda_0": lambda_0_dict
    }

    if len(L_list) >= 3:
        L_array = np.array(L_list)
        f_L_array = np.array(f_L_list)

        try:
            popt, pcov = curve_fit(scaling_model, L_array, f_L_array)
            A, B, C = popt

            f_inf = A
            f_sur = B / 2.0
            vFc = - (24 * C) / np.pi

            results["fit_success"] = True
            results["f_inf"] = f_inf
            results["f_sur"] = f_sur
            results["vFc"] = vFc
            results["popt"] = list(popt)
        except Exception as e:
            print(f"Failed to fit central charge: {e}")
            results["fit_success"] = False
    else:
        results["fit_success"] = False

    return results

def extrapolate_conformal_dimensions(data_by_L, lambda_0_dict, top_k=5):
    # Calculate h_j(L) for all available points
    h_data = {} # nested dict: h_data[sector][rank][L] = h_j

    L_sorted = sorted(data_by_L.keys())

    for L in L_sorted:
        if L not in lambda_0_dict:
            continue

        lambda_0 = lambda_0_dict[L]
        modules = data_by_L[L].get("modules", {})

        for sector, sector_data in modules.items():
            if sector not in h_data:
                h_data[sector] = {}

            eigs = [parse_complex(e) for e in sector_data.get("eigenvalues", [])]
            # Sort descending by modulus
            eigs_sorted = sorted(eigs, key=abs, reverse=True)

            for rank, eig in enumerate(eigs_sorted[:top_k]):
                lam_j = abs(eig)
                if lam_j < 1e-12:
                    continue # avoid log(0) issues

                h_j = (L / np.pi) * np.log(lambda_0 / lam_j)

                if rank not in h_data[sector]:
                    h_data[sector][rank] = {}
                h_data[sector][rank][L] = h_j

    # Perform fits
    extrapolations = {}
    for sector, ranks in h_data.items():
        extrapolations[sector] = {}
        for rank, L_dict in ranks.items():
            L_vals = sorted(L_dict.keys())
            h_vals = [L_dict[L] for L in L_vals]

            result = {
                "L_vals": L_vals,
                "h_vals": h_vals,
                "h_inf": None,
                "fit_success": False
            }

            if len(L_vals) >= 2:
                # Linear fit vs 1/L
                inv_L = np.array([1.0 / L for L in L_vals])
                h_array = np.array(h_vals)

                try:
                    popt, pcov = curve_fit(linear_model, inv_L, h_array)
                    result["h_inf"] = float(popt[0])
                    result["slope"] = float(popt[1])
                    result["fit_success"] = True
                except Exception as e:
                    pass
            elif len(L_vals) == 1:
                # If only one point, we can't extrapolate, just record the value
                pass

            extrapolations[sector][rank] = result

    return extrapolations

def export_outputs(cc_results, h_extrapolations, output_prefix):
    # Output to JSON
    json_data = {
        "central_charge": {
            "L_values": cc_results["L_values"],
            "f_L_values": cc_results["f_L_values"],
            "fit_success": cc_results["fit_success"]
        },
        "conformal_dimensions": {}
    }

    if cc_results["fit_success"]:
        json_data["central_charge"]["f_inf"] = cc_results["f_inf"]
        json_data["central_charge"]["f_sur"] = cc_results["f_sur"]
        json_data["central_charge"]["vFc"] = cc_results["vFc"]

    for sector, ranks in h_extrapolations.items():
        json_data["conformal_dimensions"][sector] = {}
        for rank, res in ranks.items():
            json_data["conformal_dimensions"][sector][rank] = {
                "L_vals": res["L_vals"],
                "h_vals": res["h_vals"],
                "fit_success": res["fit_success"]
            }
            if res["fit_success"]:
                json_data["conformal_dimensions"][sector][rank]["h_inf"] = res["h_inf"]
                json_data["conformal_dimensions"][sector][rank]["slope"] = res["slope"]

    json_path = f"{output_prefix}.json"
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"\nExported JSON data to {json_path}")

    # Output to Mathematica (.m)
    m_path = f"{output_prefix}.m"
    with open(m_path, "w") as f:
        f.write("(* Extrapolated Conformal Data *)\n\n")

        # Central charge data
        f.write("(* Free Energy f_L vs L *)\n")
        pairs = [f"{{{L}, {f_L}}}" for L, f_L in zip(cc_results["L_values"], cc_results["f_L_values"])]
        f.write(f"fLData = {{{', '.join(pairs)}}};\n")

        if cc_results["fit_success"]:
            f.write(f"fInf = {cc_results['f_inf']};\n")
            f.write(f"fSur = {cc_results['f_sur']};\n")
            f.write(f"vFc = {cc_results['vFc']};\n")

        f.write("\n(* Conformal Dimensions h_j(L) vs L *)\n")
        for sector, ranks in h_extrapolations.items():
            # Create a clean variable name from sector
            sector_name = sector.replace(" ", "").replace("=", "").replace(",", "Y")
            for rank, res in ranks.items():
                pairs = [f"{{{L}, {h}}}" for L, h in zip(res["L_vals"], res["h_vals"])]
                var_name = f"hData{sector_name}Rank{rank}"
                f.write(f"{var_name} = {{{', '.join(pairs)}}};\n")
                if res["fit_success"]:
                    f.write(f"hInf{sector_name}Rank{rank} = {res['h_inf']};\n")

    print(f"Exported Mathematica data to {m_path}")

def main():
    parser = argparse.ArgumentParser(description="Extrapolate central charge and conformal dimensions from eigenvalue JSON files.")
    parser.add_argument("input_jsons", nargs="+", help="Paths to JSON files (e.g., experiment_outputs/all_eigenvalues_L*.json)")
    parser.add_argument("-o", "--output_prefix", type=str, default="extrapolated", help="Prefix for output files")
    parser.add_argument("-k", "--top_k", type=int, default=5, help="Number of eigenvalues to track per sector")

    args = parser.parse_args()

    data_by_L = load_data(args.input_jsons)

    if not data_by_L:
        print("Error: No valid data loaded.")
        sys.exit(1)

    print(f"Loaded data for L = {sorted(list(data_by_L.keys()))}")

    cc_results = extrapolate_central_charge(data_by_L)
    print("\nCentral Charge Extrapolation:")
    print("-----------------------------")
    if not cc_results["L_values"]:
        print("No valid vacuum eigenvalues found.")
        sys.exit(1)

    for L, f_L in zip(cc_results["L_values"], cc_results["f_L_values"]):
        print(f"L={L}: f_L = {f_L:.6f}")

    if cc_results.get("fit_success"):
        print(f"\nf_inf = {cc_results['f_inf']:.6f}")
        print(f"f_sur = {cc_results['f_sur']:.6f}")
        print(f"v_F c = {cc_results['vFc']:.6f}")
    else:
        print("\nNot enough points (need at least 3) to extrapolate central charge.")

    h_extrapolations = extrapolate_conformal_dimensions(data_by_L, cc_results["lambda_0"], top_k=args.top_k)
    print("\nConformal Dimensions Extrapolation:")
    print("-----------------------------------")
    for sector, ranks in sorted(h_extrapolations.items()):
        for rank, res in sorted(ranks.items()):
            if res["fit_success"]:
                print(f"Sector {sector:<10} Rank {rank}: h_inf = {res['h_inf']:.6f}")
            elif len(res["L_vals"]) == 1:
                print(f"Sector {sector:<10} Rank {rank}: h(L={res['L_vals'][0]}) = {res['h_vals'][0]:.6f} (No extrapolation)")

    export_outputs(cc_results, h_extrapolations, args.output_prefix)

if __name__ == "__main__":
    main()
