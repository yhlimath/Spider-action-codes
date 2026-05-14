import json
import matplotlib.pyplot as plt
import os
import argparse

def plot_eigenvalues(logs, t, order, n_str, out_dir):
    L_dict = logs[t][order][n_str]

    plt.figure(figsize=(8, 8))
    colors = ['r', 'g', 'b', 'c', 'm', 'y']
    L_vals = sorted([int(L) for L in L_dict.keys() if L_dict[L] is not None])

    for idx, L in enumerate(L_vals):
        lam_list = L_dict[str(L)]
        real_vals = [lam['real'] for lam in lam_list]
        imag_vals = [lam['imag'] for lam in lam_list]
        color = colors[idx % len(colors)]
        plt.scatter(real_vals, imag_vals, color=color, label=f"L={L}", alpha=0.7, edgecolors='k', s=50)

    plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
    plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
    plt.title(f"Leading Eigenvalues for {t}, order={order}, n={n_str}")
    plt.xlabel("Re(\u03bb)")
    plt.ylabel("Im(\u03bb)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save plot
    filename = f"eigenvalues_plot_{t.replace('+', '_')}_{order}_n{n_str}.png"
    filepath = os.path.join(out_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {filepath}")

def main():
    parser = argparse.ArgumentParser(description="Plot eigenvalues in the complex plane")
    parser.add_argument('--type', type=str, default=None, help="Filter by matrix type (e.g. E+H)")
    parser.add_argument('--order', type=str, default=None, help="Filter by order (e.g. staggered)")
    parser.add_argument('--n', type=str, default=None, help="Filter by weight n (e.g. 1.0)")
    args = parser.parse_args()

    in_file = "experiment_outputs/denseKuperberg/eigenvalue_logs_top_k.json"
    if not os.path.exists(in_file):
        print(f"Log file {in_file} not found.")
        return

    with open(in_file, 'r') as f:
        logs = json.load(f)

    out_dir = "experiment_outputs/denseKuperberg"

    for t in logs:
        if args.type and t != args.type: continue
        for order in logs[t]:
            if args.order and order != args.order: continue
            for n_str in logs[t][order]:
                if args.n and n_str != args.n: continue
                plot_eigenvalues(logs, t, order, n_str, out_dir)

if __name__ == "__main__":
    main()
