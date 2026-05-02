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
            if L % 3 != 0:
                print(f"Warning: L={L} is not a multiple of 3 in {path}. Skipping.")
                continue
            data_by_L[L] = data
    return data_by_L

def scaling_model(L, A, B, C):
    return A + B/L + C/(L**2)

def linear_model(inv_L, intercept, slope):
    return intercept + slope * inv_L

def extrapolate_central_charge(data_by_L, operator="T"):
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
        lambda_0_dict[L] = lambda_0_cplx

        # Free energy f_L
        if operator == "T":
            f_L = np.log(lambda_0) / (3 * L)
        else:
            f_L = np.real(lambda_0_cplx) / (3 * L)

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


import itertools

def extrapolate_conformal_dimensions(data_by_L, lambda_0_dict, top_k=5, operator="T", vF=1.0):
    L_sorted = sorted(data_by_L.keys())

    # Pre-calculate all h_j(L) for each sector, L, and rank up to top_k + delta
    delta = 2
    max_k_search = top_k + delta

    # raw_h[sector][L][rank] = h_j value
    raw_h = {}

    for L in L_sorted:
        if L not in lambda_0_dict:
            continue

        lambda_0_cplx = lambda_0_dict[L]
        lambda_0_mod = np.abs(lambda_0_cplx)
        modules = data_by_L[L].get("modules", {})

        for sector, sector_data in modules.items():
            if sector not in raw_h:
                raw_h[sector] = {}
            if L not in raw_h[sector]:
                raw_h[sector][L] = {}

            eigs = [parse_complex(e) for e in sector_data.get("eigenvalues", [])]
            eigs_sorted = sorted(eigs, key=abs, reverse=True)

            for rank, eig in enumerate(eigs_sorted[:max_k_search]):
                if operator == "T":
                    lam_j = abs(eig)
                    if lam_j < 1e-12:
                        continue
                    h_j = (L / (np.pi * vF)) * np.log(lambda_0_mod / lam_j)
                else:
                    h_j = (L / (2 * np.pi * vF)) * (np.real(lambda_0_cplx) - np.real(eig))

                raw_h[sector][L][rank] = h_j

    extrapolations = {}

    for sector, L_data in raw_h.items():
        extrapolations[sector] = {}
        valid_Ls = sorted(L_data.keys())
        if not valid_Ls:
            continue

        # We want to map target ranks 0 to top_k-1
        # For a target rank `target_k`, candidate ranks for each L are in [target_k - delta, target_k + delta]

        # Build list of all combinations up to top_k
        # combination: (rank_L1, rank_L2, ...)
        all_combinations = []

        # Generate all valid candidate sequences for target ranks 0 to top_k-1
        for target_k in range(top_k):
            candidates_per_L = []
            for L in valid_Ls:
                valid_ranks_for_L = []
                for r in range(max(0, target_k - delta), target_k + delta + 1):
                    if r in L_data[L]:
                        valid_ranks_for_L.append(r)
                candidates_per_L.append(valid_ranks_for_L)

            # Cartesian product
            for seq in itertools.product(*candidates_per_L):
                # seq is a tuple of ranks, e.g., (r1, r2, r3)
                # Compute fit for this sequence
                h_vals = [L_data[L][r] for L, r in zip(valid_Ls, seq)]

                if len(valid_Ls) >= 2:
                    inv_L = np.array([1.0 / L for L in valid_Ls])
                    h_array = np.array(h_vals)
                    try:
                        popt, pcov = curve_fit(linear_model, inv_L, h_array)
                        # SSR calculation
                        residuals = h_array - linear_model(inv_L, *popt)
                        ssr = np.sum(residuals**2)

                        all_combinations.append({
                            "target_k": target_k,
                            "seq": seq,
                            "h_vals": h_vals,
                            "L_vals": valid_Ls,
                            "h_inf": float(popt[0]),
                            "slope": float(popt[1]),
                            "ssr": float(ssr),
                            "fit_success": True
                        })
                    except Exception:
                        pass
                elif len(valid_Ls) == 1:
                    all_combinations.append({
                        "target_k": target_k,
                        "seq": seq,
                        "h_vals": h_vals,
                        "L_vals": valid_Ls,
                        "h_inf": None,
                        "slope": None,
                        "ssr": 0.0,
                        "fit_success": False
                    })

        # Sort combinations by SSR
        all_combinations.sort(key=lambda x: x["ssr"])

        assigned_combinations = []
        used_L_ranks = {L: set() for L in valid_Ls}

        # We need to pick one combination for each target_k in 0..top_k-1
        # It's better to iterate target_k and find the best available, or just go greedy globally?
        # Greedy globally is better to ensure the lowest error trajectories are locked in first.

        assigned_target_ks = set()

        for combo in all_combinations:
            if combo["target_k"] in assigned_target_ks:
                continue

            # Check if any rank in this combo is already used
            conflict = False
            for L, r in zip(valid_Ls, combo["seq"]):
                if r in used_L_ranks[L]:
                    conflict = True
                    break

            if not conflict:
                # Accept this combination
                assigned_combinations.append(combo)
                assigned_target_ks.add(combo["target_k"])
                for L, r in zip(valid_Ls, combo["seq"]):
                    used_L_ranks[L].add(r)

            if len(assigned_target_ks) == top_k:
                break

        # Store in extrapolations dictionary
        for combo in assigned_combinations:
            extrapolations[sector][combo["target_k"]] = {
                "L_vals": combo["L_vals"],
                "h_vals": combo["h_vals"],
                "h_inf": combo["h_inf"],
                "slope": combo["slope"],
                "fit_success": combo["fit_success"],
                "seq": combo["seq"],
                "ssr": combo["ssr"]
            }

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
    parser.add_argument("-k", "--top_k", type=int, default=50, help="Number of eigenvalues to track per sector")
    parser.add_argument("-O", "--operator", choices=["H", "T"], default="H", help="Operator to analyze (H or T)")
    parser.add_argument("--vF", type=float, default=1.0, help="Fermi velocity v_F to scale conformal dimensions (default 1.0)")

    args = parser.parse_args()

    data_by_L = load_data(args.input_jsons)

    if not data_by_L:
        print("Error: No valid data loaded.")
        sys.exit(1)

    print(f"Loaded data for L = {sorted(list(data_by_L.keys()))}")

    cc_results = extrapolate_central_charge(data_by_L, args.operator)
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

    h_extrapolations = extrapolate_conformal_dimensions(data_by_L, cc_results["lambda_0"], top_k=args.top_k, operator=args.operator, vF=args.vF)
    print(f"\nConformal Dimensions Extrapolation (vF = {args.vF}):")
    print("-----------------------------------")
    for sector, ranks in sorted(h_extrapolations.items()):
        for rank, res in sorted(ranks.items()):
            if res["fit_success"]:
                print(f"Sector {sector:<10} Rank {rank}: h_inf (scaled) = {res['h_inf']:.6f}")
            elif len(res["L_vals"]) == 1:
                print(f"Sector {sector:<10} Rank {rank}: h(L={res['L_vals'][0]}) (scaled) = {res['h_vals'][0]:.6f} (No extrapolation)")

    export_outputs(cc_results, h_extrapolations, args.output_prefix)

if __name__ == "__main__":
    main()
