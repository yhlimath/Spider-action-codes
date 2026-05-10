import sys
import os
import json
import argparse
import numpy as np

def parse_complex(val):
    if isinstance(val, dict):
        return complex(val["re"], val["im"])
    return float(val)

def extract_conformal_data(json_path, k=10):
    if not os.path.exists(json_path):
        print(f"Error: File {json_path} not found.")
        sys.exit(1)

    with open(json_path, "r") as f:
        data = json.load(f)

    L = data.get("L")
    if L is None:
        print("Error: JSON does not contain 'L'.")
        sys.exit(1)

    if L % 3 != 0:
        print(f"Error: System size L={L} is not divisible by 3. Vacuum sector (0,0) does not exist.")
        sys.exit(1)

    modules = data.get("modules", {})
    if "x=0, y=0" not in modules:
        print("Error: Vacuum sector 'x=0, y=0' not found in data.")
        sys.exit(1)

    # Find lambda_0 from vacuum sector
    vacuum_eigs = [parse_complex(e) for e in modules["x=0, y=0"]["eigenvalues"]]
    if not vacuum_eigs:
        print("Error: Vacuum sector has no non-zero eigenvalues.")
        sys.exit(1)

    lambda_0_cplx = max(vacuum_eigs, key=abs)
    lambda_0 = np.abs(lambda_0_cplx)
    print(f"L = {L}")
    print(f"Vacuum leading eigenvalue |lambda_0| = {lambda_0:.8f}")

    # Pool all eigenvalues
    all_eigenvalues = []
    for sector_key, sector_data in modules.items():
        eigs = [parse_complex(e) for e in sector_data["eigenvalues"]]
        for e in eigs:
            all_eigenvalues.append({
                "sector": sector_key,
                "eigenvalue_cplx": e,
                "modulus": np.abs(e)
            })

    # Sort globally by modulus descending
    all_eigenvalues.sort(key=lambda item: item["modulus"], reverse=True)

    # Extract top k
    top_k = all_eigenvalues[:k]

    results = {
        "L": L,
        "n": data.get("n"),
        "lambda_0": float(lambda_0),
        "k": k,
        "conformal_dimensions": []
    }

    print(f"\nTop {len(top_k)} eigenvalues and conformal dimensions (h_j):")
    print(f"{'Rank':<5} | {'Sector':<10} | {'|lambda_j|':<12} | {'h_j':<12}")
    print("-" * 50)

    for idx, item in enumerate(top_k):
        lam_j = item["modulus"]
        # h_j = (L / pi) * log(|lambda_0| / |lambda_j|)
        h_j = (L / np.pi) * np.log(lambda_0 / lam_j)

        # Serialize complex value
        c_val = item["eigenvalue_cplx"]
        if np.abs(c_val.imag) < 1e-10:
            val_export = float(c_val.real)
        else:
            val_export = {"re": float(c_val.real), "im": float(c_val.imag)}

        results["conformal_dimensions"].append({
            "rank": idx,
            "sector": item["sector"],
            "eigenvalue": val_export,
            "modulus": float(lam_j),
            "h_j": float(h_j)
        })
        print(f"{idx:<5} | {item['sector']:<10} | {lam_j:<12.6f} | {h_j:<12.6f}")

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute conformal data h_j from existing eigenvalue JSON.")
    parser.add_argument("input_json", type=str, help="Path to the JSON file (e.g. experiment_outputs/all_eigenvalues_L6.json)")
    parser.add_argument("-k", type=int, default=15, help="Number of top eigenvalues to extract overall")
    args = parser.parse_args()

    res = extract_conformal_data(args.input_json, args.k)

    out_dir = os.path.dirname(args.input_json)
    L_val = res["L"]
    base_name = os.path.basename(args.input_json).replace(".json", "")
    out_file = os.path.join(out_dir, f"conformal_data_{base_name}.json")

    with open(out_file, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nExported conformal data to {out_file}")
