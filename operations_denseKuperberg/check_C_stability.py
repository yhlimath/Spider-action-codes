#!/usr/bin/env python3
"""Check finite-size stability of the fitted C coefficient.

This is a companion to phase_diagram_sweep.py.  It focuses on one point
(x, y, z, n), computes or loads the leading eigenvalue for each L, and repeats
the same fit after changing which system sizes are included.
"""

import argparse
import json
import math
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.linalg import eigvals
from scipy.sparse.linalg import LinearOperator, eigs


DEFAULT_PROJECT_ROOT = (
    Path.home()
    / "Library"
    / "Mobile Documents"
    / "com~apple~CloudDocs"
    / "Programming"
    / "Web_transfer_matrix"
)
DEFAULT_OUTPUT_DIR = Path("experiment_outputs") / "denseKuperberg"
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "check-c-stability-matplotlib-cache"),
)


@dataclass
class FitResult:
    label: str
    L_vals: list[int]
    A: float
    B: float
    C: float
    central_charge: float
    rms: float
    condition: float


@dataclass
class ZigzagFitResult:
    label: str
    L_vals: list[int]
    A: float
    B: float
    C: float
    ssr: float
    rms: float
    condition: float


def parse_L_list(value: str) -> list[int]:
    L_vals = [int(part.strip()) for part in value.split(",") if part.strip()]
    if len(L_vals) < 3:
        raise argparse.ArgumentTypeError("Need at least three L values to fit A + B/L + C/L^2.")
    return sorted(set(L_vals))


def add_project_root(project_root: str) -> None:
    root = Path(project_root).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"Project root does not exist: {root}")
    sys.path.insert(0, str(root))


def import_solver(project_root: str):
    add_project_root(project_root)
    from denseKuperberg.arnoldi import KuperbergArnoldiSolver

    return KuperbergArnoldiSolver


def leading_eigenvalue(args, L: int):
    Solver = import_solver(args.project_root)
    solver = Solver(
        L,
        0,
        0,
        args.type_str,
        args.order,
        args.n_val,
        x_value=args.x_val,
        y_value=args.y_val,
        z_value=args.z_val,
        operator=args.operator,
    )

    if solver.dim <= 0:
        return None, solver.dim

    if solver.dim <= args.exact_cutoff:
        hessenberg, _ = solver.arnoldi_iteration(solver.dim)
        if hessenberg.shape[0] == 0:
            return None, solver.dim
        values = eigvals(hessenberg)
    else:
        k_actual = min(args.top_k, solver.dim - 2)
        if k_actual <= 0:
            return None, solver.dim

        def matvec(v):
            if args.operator == "H":
                return solver.apply_H(v)
            return solver.apply_T(v)

        operator = LinearOperator((solver.dim, solver.dim), matvec=matvec, dtype=complex)
        values, _ = eigs(operator, k=k_actual, which=args.which)

    values = [value for value in values if abs(value) > args.eigenvalue_tol]
    if not values:
        return None, solver.dim

    if args.operator == "H":
        values.sort(key=lambda value: value.real, reverse=True)
    else:
        values.sort(key=lambda value: abs(value), reverse=True)
    return values[0], solver.dim


def free_energy_value(L: int, eigenvalue, operator: str) -> float:
    if operator == "T":
        return -math.log(abs(eigenvalue)) / L
    return -float(np.real(eigenvalue)) / L


def fit_C(label: str, L_vals: list[int], eigenvalues: list[complex], operator: str, vF: float) -> FitResult:
    if len(L_vals) < 3:
        raise ValueError("Need at least three points for this fit.")

    L_arr = np.array(L_vals, dtype=float)
    y = np.array([free_energy_value(L, lam, operator) for L, lam in zip(L_vals, eigenvalues)])
    X = np.column_stack([np.ones_like(L_arr), 1.0 / L_arr, 1.0 / (L_arr**2)])
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    A, B, C = [float(value) for value in coeffs]
    residuals = y - X @ coeffs
    rms = float(np.sqrt(np.mean(residuals**2)))
    condition = float(np.linalg.cond(X))
    central_charge = -24.0 * C / (math.pi * vF)
    return FitResult(label, list(L_vals), A, B, C, central_charge, rms, condition)


def build_stability_fits(L_vals: list[int], eigenvalues: list[complex], operator: str, vF: float) -> list[FitResult]:
    pairs = sorted(zip(L_vals, eigenvalues), key=lambda pair: pair[0])
    L_sorted = [pair[0] for pair in pairs]
    eig_sorted = [pair[1] for pair in pairs]

    fits = [fit_C("all", L_sorted, eig_sorted, operator, vF)]

    for start in range(1, len(L_sorted) - 2):
        label = f"L>={L_sorted[start]}"
        fits.append(fit_C(label, L_sorted[start:], eig_sorted[start:], operator, vF))

    if len(L_sorted) >= 4:
        for drop_index, dropped_L in enumerate(L_sorted):
            keep_L = [L for index, L in enumerate(L_sorted) if index != drop_index]
            keep_eig = [lam for index, lam in enumerate(eig_sorted) if index != drop_index]
            fits.append(fit_C(f"drop L={dropped_L}", keep_L, keep_eig, operator, vF))

    return fits


def fit_series(label: str, L_vals: list[int], y_vals: list[float]) -> ZigzagFitResult:
    if len(L_vals) < 3:
        raise ValueError("Need at least three points for this fit.")

    L_arr = np.array(L_vals, dtype=float)
    y = np.array(y_vals, dtype=float)
    X = np.column_stack([np.ones_like(L_arr), 1.0 / L_arr, 1.0 / (L_arr**2)])
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    A, B, C = [float(value) for value in coeffs]
    residuals = y - X @ coeffs
    ssr = float(np.sum(residuals**2))
    rms = float(np.sqrt(np.mean(residuals**2)))
    condition = float(np.linalg.cond(X))
    return ZigzagFitResult(label, list(L_vals), A, B, C, ssr, rms, condition)


def build_series_stability_fits(L_vals: list[int], y_vals: list[float]) -> list[ZigzagFitResult]:
    pairs = sorted(zip(L_vals, y_vals), key=lambda pair: pair[0])
    L_sorted = [pair[0] for pair in pairs]
    y_sorted = [pair[1] for pair in pairs]

    fits = [fit_series("all", L_sorted, y_sorted)]

    for start in range(1, len(L_sorted) - 2):
        label = f"L>={L_sorted[start]}"
        fits.append(fit_series(label, L_sorted[start:], y_sorted[start:]))

    if len(L_sorted) >= 4:
        for drop_index, dropped_L in enumerate(L_sorted):
            keep_L = [L for index, L in enumerate(L_sorted) if index != drop_index]
            keep_y = [y for index, y in enumerate(y_sorted) if index != drop_index]
            fits.append(fit_series(f"drop L={dropped_L}", keep_L, keep_y))

    return fits


def parse_phase_key(key: str):
    match = re.fullmatch(r"x=([-+0-9.eE]+)_y=([-+0-9.eE]+)", key)
    if not match:
        return None
    return float(match.group(1)), float(match.group(2))


def load_from_sweep_json(path: str, x_val: float, y_val: float, tolerance: float):
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    best_key = None
    best_distance = None
    for key in data:
        parsed = parse_phase_key(key)
        if parsed is None:
            continue
        x_key, y_key = parsed
        distance = math.hypot(x_key - x_val, y_key - y_val)
        if best_distance is None or distance < best_distance:
            best_key = key
            best_distance = distance

    if best_key is None:
        raise ValueError("Could not find keys of the form x=<value>_y=<value> in the JSON file.")
    if best_distance is None or best_distance > tolerance:
        raise ValueError(
            f"Closest point is {best_key}, distance {best_distance:.6g}; "
            f"increase --json-point-tol if that is intentional."
        )

    point = data[best_key]
    if "L" not in point or "lam" not in point:
        raise ValueError(f"Point {best_key} does not contain both 'L' and 'lam'.")
    return best_key, [int(L) for L in point["L"]], [complex(lam) for lam in point["lam"]]


def print_eigenvalue_table(L_vals: list[int], eigenvalues: list[complex], operator: str) -> None:
    print("\nFinite-size data")
    print(f"{'L':>5} {'lambda used':>22} {'f_L':>18}")
    print("-" * 49)
    for L, lam in zip(L_vals, eigenvalues):
        lam_used = lam.real if operator == "H" else abs(lam)
        f_L = free_energy_value(L, lam, operator)
        print(f"{L:5d} {lam_used:22.14g} {f_L:18.10g}")


def print_fit_table(fits: list[FitResult]) -> None:
    print("\nC stability refits")
    print(f"{'fit':<14} {'L values':<18} {'C':>15} {'c':>15} {'rms':>12} {'cond(X)':>12}")
    print("-" * 91)
    for fit in fits:
        L_label = ",".join(str(L) for L in fit.L_vals)
        print(
            f"{fit.label:<14} {L_label:<18} "
            f"{fit.C:15.8g} {fit.central_charge:15.8g} "
            f"{fit.rms:12.4g} {fit.condition:12.4g}"
        )


def print_summary(fits: list[FitResult]) -> None:
    C_values = np.array([fit.C for fit in fits], dtype=float)
    c_values = np.array([fit.central_charge for fit in fits], dtype=float)
    full_C = fits[0].C
    max_delta = float(np.max(np.abs(C_values - full_C)))
    print("\nSummary over refits")
    print(f"full-fit C:     {full_C:.10g}")
    print(f"C mean/std:     {float(np.mean(C_values)):.10g} / {float(np.std(C_values)):.4g}")
    print(f"C min/max:      {float(np.min(C_values)):.10g} / {float(np.max(C_values)):.10g}")
    print(f"max |dC|:       {max_delta:.4g}")
    print(f"central c mean: {float(np.mean(c_values)):.10g}")


def safe_filename_piece(value) -> str:
    text = str(value)
    return re.sub(r"[^A-Za-z0-9_.+-]+", "_", text)


def default_plot_path(args) -> str:
    filename = (
        f"C_stability_fit_{args.operator}"
        f"_n{args.n_val:.2f}"
        f"_z{args.z_val:.2f}"
        f"_x{args.x_val:.3g}"
        f"_y{args.y_val:.3g}.png"
    )
    return str(Path(args.out_dir) / safe_filename_piece(filename))


def default_zigzag_log_path(args) -> str:
    return str(Path(args.out_dir) / "eigenvalue_logs_zigzag.json")


def default_zigzag_plot_path(args, config: dict) -> str:
    filename = (
        f"zigzag_stability_fit"
        f"_n{config['n']}"
        f"_{config['type']}"
        f"_{config['order']}.png"
    )
    return str(Path(args.out_dir) / safe_filename_piece(filename))


def plot_path_for_config(args, config: dict, num_configs: int) -> str:
    if args.plot == "auto":
        return default_zigzag_plot_path(args, config)

    path = Path(args.plot)
    if num_configs == 1:
        return str(path)

    suffix = path.suffix or ".png"
    config_suffix = safe_filename_piece(f"{config['type']}_{config['order']}_n{config['n']}")
    return str(path.with_name(f"{path.stem}_{config_suffix}{suffix}"))


def save_json_report(path: str, args, L_vals: list[int], eigenvalues: list[complex], fits: list[FitResult]) -> None:
    payload = {
        "config": {
            "x_val": args.x_val,
            "y_val": args.y_val,
            "z_val": args.z_val,
            "n_val": args.n_val,
            "operator": args.operator,
            "order": args.order,
            "type_str": args.type_str,
            "vF": args.vF,
        },
        "finite_size_data": [
            {
                "L": L,
                "lambda_real": float(np.real(lam)),
                "lambda_imag": float(np.imag(lam)),
                "lambda_abs": float(abs(lam)),
                "f_L": free_energy_value(L, lam, args.operator),
            }
            for L, lam in zip(L_vals, eigenvalues)
        ],
        "fits": [
            {
                "label": fit.label,
                "L_vals": fit.L_vals,
                "A": fit.A,
                "B": fit.B,
                "C": fit.C,
                "central_charge": fit.central_charge,
                "rms": fit.rms,
                "condition": fit.condition,
            }
            for fit in fits
        ],
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def save_plot(path: str, L_vals: list[int], eigenvalues: list[complex], fits: list[FitResult], args) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    L_arr = np.array(L_vals, dtype=float)
    x_data = 1.0 / (L_arr**2)
    y_data = np.array([free_energy_value(L, lam, args.operator) for L, lam in zip(L_vals, eigenvalues)])

    full = fits[0]
    L_smooth = np.linspace(min(L_arr), max(L_arr), 200)
    x_smooth = 1.0 / (L_smooth**2)
    y_smooth = full.A + full.B / L_smooth + full.C / (L_smooth**2)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
    axes[0].plot(x_data, y_data, "o", label="data")

    showed_refit_label = False
    for fit in fits[1:]:
        fit_L = np.linspace(min(fit.L_vals), max(fit.L_vals), 100)
        fit_x = 1.0 / (fit_L**2)
        fit_y = fit.A + fit.B / fit_L + fit.C / (fit_L**2)
        label = "refit curves" if not showed_refit_label else None
        axes[0].plot(fit_x, fit_y, "-", color="0.65", alpha=0.45, linewidth=1.0, label=label)
        showed_refit_label = True

    axes[0].plot(
        x_smooth,
        y_smooth,
        "-",
        linewidth=2.0,
        label=f"all-size fit: C={full.C:.5g}, c={full.central_charge:.5g}",
    )
    axes[0].set_title(
        f"C fit at x={args.x_val:g}, y={args.y_val:g}, z={args.z_val:g}, "
        f"n={args.n_val:g}, operator={args.operator}"
    )
    axes[0].set_xlabel("$1/L^2$")
    axes[0].set_ylabel("$f_L$")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best")

    labels = [fit.label for fit in fits]
    C_values = [fit.C for fit in fits]
    axes[1].plot(range(len(fits)), C_values, "o-")
    axes[1].axhline(full.C, color="0.4", linestyle="--", linewidth=1)
    axes[1].set_xticks(range(len(fits)))
    axes[1].set_xticklabels(labels, rotation=35, ha="right")
    axes[1].set_ylabel("fitted C")
    axes[1].set_title("Sensitivity of C to fit window")
    axes[1].grid(True, alpha=0.3)

    fig.savefig(path, dpi=150)
    plt.close(fig)


def eigen_abs(entry: dict) -> float:
    if "abs" in entry:
        return abs(float(entry["abs"]))
    return abs(complex(float(entry.get("real", 0.0)), float(entry.get("imag", 0.0))))


def zigzag_h_value(lam0: dict, lamj: dict, L: int, vF: float) -> float:
    lam0_abs = eigen_abs(lam0)
    lamj_abs = eigen_abs(lamj)
    if lam0_abs <= 0.0 or lamj_abs <= 0.0:
        raise ValueError("Encountered non-positive eigenvalue magnitude.")
    return float((L / (math.pi * vF)) * math.log(lam0_abs / lamj_abs))


def load_zigzag_logs(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def selected_zigzag_configs(logs: dict, args):
    configs = []
    for type_name in logs:
        if args.zigzag_type and type_name != args.zigzag_type:
            continue
        for order_name in logs[type_name]:
            if args.zigzag_order and order_name != args.zigzag_order:
                continue
            for n_name in logs[type_name][order_name]:
                if args.n and n_name != args.n:
                    continue
                configs.append(
                    {
                        "type": type_name,
                        "order": order_name,
                        "n": n_name,
                        "L_dict": logs[type_name][order_name][n_name],
                    }
                )
    return configs


def extract_zigzag_state_series(L_dict: dict, state_index: int, vF: float, min_L: int):
    L_vals = []
    h_vals = []
    for L in sorted(int(value) for value in L_dict.keys() if int(value) >= min_L):
        lam_list = L_dict.get(str(L), L_dict.get(L))
        if not lam_list or state_index >= len(lam_list):
            continue
        try:
            h_val = zigzag_h_value(lam_list[0], lam_list[state_index], L, vF)
        except ValueError:
            continue
        L_vals.append(L)
        h_vals.append(h_val)
    return L_vals, h_vals


def analyze_zigzag_config(config: dict, args):
    analyses = []
    for state_index in range(args.num_states):
        L_vals, h_vals = extract_zigzag_state_series(
            config["L_dict"], state_index, args.vF, args.min_L
        )
        if len(L_vals) < 3:
            continue
        analyses.append(
            {
                "j": state_index,
                "L_vals": L_vals,
                "h_vals": h_vals,
                "fits": build_series_stability_fits(L_vals, h_vals),
            }
        )
    return analyses


def print_zigzag_summary(config: dict, analyses: list[dict], verbose: bool = False) -> None:
    print(
        f"\nConfiguration: Zigzag extrapolations, "
        f"type={config['type']}, order={config['order']}, n={config['n']}"
    )
    if not analyses:
        print("No states had at least three usable L values.")
        return

    print(
        f"{'j':>4} {'h(all)':>14} {'mean':>14} {'std':>12} "
        f"{'min':>14} {'max':>14} {'max |dh|':>12} {'rms':>11} {'Ls'}"
    )
    print("-" * 113)
    for analysis in analyses:
        fits = analysis["fits"]
        h_values = np.array([fit.A for fit in fits], dtype=float)
        full = fits[0]
        max_delta = float(np.max(np.abs(h_values - full.A)))
        L_label = ",".join(str(L) for L in analysis["L_vals"])
        print(
            f"{analysis['j']:4d} {full.A:14.8g} {float(np.mean(h_values)):14.8g} "
            f"{float(np.std(h_values)):12.4g} {float(np.min(h_values)):14.8g} "
            f"{float(np.max(h_values)):14.8g} {max_delta:12.4g} "
            f"{full.rms:11.4g} {L_label}"
        )

        if verbose:
            print(f"  {'fit':<14} {'L values':<18} {'h extrap':>15} {'SSR':>12} {'cond(X)':>12}")
            for fit in fits:
                fit_L_label = ",".join(str(L) for L in fit.L_vals)
                print(
                    f"  {fit.label:<14} {fit_L_label:<18} "
                    f"{fit.A:15.8g} {fit.ssr:12.4g} {fit.condition:12.4g}"
                )


def save_zigzag_json_report(path: str, args, reports: list[dict]) -> None:
    payload = {
        "config": {
            "vF": args.vF,
            "num_states": args.num_states,
            "min_L": args.min_L,
            "zigzag_json": args.zigzag_json or default_zigzag_log_path(args),
        },
        "reports": [],
    }
    for report in reports:
        payload["reports"].append(
            {
                "type": report["config"]["type"],
                "order": report["config"]["order"],
                "n": report["config"]["n"],
                "states": [
                    {
                        "j": analysis["j"],
                        "L_vals": analysis["L_vals"],
                        "h_vals": analysis["h_vals"],
                        "fits": [
                            {
                                "label": fit.label,
                                "L_vals": fit.L_vals,
                                "h_extrap": fit.A,
                                "B": fit.B,
                                "C": fit.C,
                                "ssr": fit.ssr,
                                "rms": fit.rms,
                                "condition": fit.condition,
                            }
                            for fit in analysis["fits"]
                        ],
                    }
                    for analysis in report["analyses"]
                ],
            }
        )

    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def save_zigzag_plot(path: str, config: dict, analyses: list[dict], args) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    states = analyses[: args.plot_states]
    if not states:
        return

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
    for analysis in states:
        L_vals = np.array(analysis["L_vals"], dtype=float)
        h_vals = np.array(analysis["h_vals"], dtype=float)
        full = analysis["fits"][0]
        inv_L2 = 1.0 / (L_vals**2)
        line = axes[0].plot(
            inv_L2,
            h_vals,
            "o",
            label=f"State {analysis['j']} (h={full.A:.4g})",
        )
        color = line[0].get_color()
        L_continuous = np.linspace(min(L_vals) * 0.9, max(L_vals) * 1.1, 100)
        h_fit = full.A + full.B / L_continuous + full.C / (L_continuous**2)
        axes[0].plot(1.0 / (L_continuous**2), h_fit, "-", color=color, alpha=0.65)

        h_refits = [fit.A for fit in analysis["fits"]]
        axes[1].plot(range(len(h_refits)), h_refits, "o-", label=f"State {analysis['j']}")

    first_labels = [fit.label for fit in states[0]["fits"]]
    axes[0].set_title(
        f"Zigzag Extrapolation Stability | n={config['n']} | "
        f"{config['type']} / {config['order']}"
    )
    axes[0].set_xlabel("$1/L^2$")
    axes[0].set_ylabel("$h_j(L)$")
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Sensitivity of extrapolated h to fit window")
    axes[1].set_ylabel("extrapolated h")
    axes[1].set_xticks(range(len(first_labels)))
    axes[1].set_xticklabels(first_labels, rotation=35, ha="right")
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    axes[1].grid(True, alpha=0.3)

    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_zigzag_stability(args) -> None:
    in_file = args.zigzag_json or default_zigzag_log_path(args)
    if not os.path.exists(in_file):
        raise SystemExit(f"Zigzag log file not found: {in_file}")

    logs = load_zigzag_logs(in_file)
    configs = selected_zigzag_configs(logs, args)
    if not configs:
        raise SystemExit("No zigzag configurations matched the requested filters.")

    reports = []
    for config in configs:
        analyses = analyze_zigzag_config(config, args)
        print_zigzag_summary(config, analyses, verbose=args.verbose_fits)
        reports.append({"config": config, "analyses": analyses})

    if args.out_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
        save_zigzag_json_report(args.out_json, args, reports)
        print(f"\nWrote JSON report: {args.out_json}")

    if args.plot:
        for report in reports:
            plot_path = plot_path_for_config(args, report["config"], len(reports))
            os.makedirs(os.path.dirname(os.path.abspath(plot_path)), exist_ok=True)
            save_zigzag_plot(plot_path, report["config"], report["analyses"], args)
            print(f"Wrote plot: {plot_path}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator", choices=["T", "H"], default="T")
    parser.add_argument("--x_val", type=float, default=0.1)
    parser.add_argument("--y_val", type=float, default=0.1)
    parser.add_argument("--z_val", type=float, default=1.0)
    parser.add_argument("--n_val", type=float, default=1.0)
    parser.add_argument("--L_list", type=parse_L_list, default=parse_L_list("4,6,8"))
    parser.add_argument("--type_str", default="T(x,y,z)")
    parser.add_argument("--order", default="staggered", choices=["sequential", "staggered"])
    parser.add_argument("--vF", type=float, default=1.0, help="Velocity factor used for H central-charge scaling.")
    parser.add_argument("--project-root", default=str(DEFAULT_PROJECT_ROOT))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--which", default="LM", help="ARPACK selector passed to scipy.sparse.linalg.eigs.")
    parser.add_argument("--exact-cutoff", type=int, default=5)
    parser.add_argument("--eigenvalue-tol", type=float, default=1e-10)
    parser.add_argument("--from-json", help="Reuse one point from a phase_diagram_data_*.json file.")
    parser.add_argument("--json-point-tol", type=float, default=5e-3)
    parser.add_argument("--out-json", help="Optional path for a machine-readable stability report.")
    parser.add_argument("--zigzag", action="store_true", help="Check extrapolation stability for extrapolate_zigzag.py logs.")
    parser.add_argument("--zigzag-json", help="Path to eigenvalue_logs_zigzag.json. Defaults to --out-dir/eigenvalue_logs_zigzag.json.")
    parser.add_argument("--n", help="Only analyze one zigzag n value, e.g. --n 1.0.")
    parser.add_argument("--num-states", "--num_states", dest="num_states", type=int, default=20)
    parser.add_argument("--plot-states", type=int, default=5, help="Number of zigzag states to include in each plot.")
    parser.add_argument("--zigzag-type", help="Only analyze this top-level zigzag type, e.g. Zigzag.")
    parser.add_argument("--zigzag-order", help="Only analyze this zigzag order, e.g. symmetric.")
    parser.add_argument("--min-L", dest="min_L", type=int, default=3, help="Minimum L to include in zigzag fits.")
    parser.add_argument("--verbose-fits", action="store_true", help="Print each zigzag refit window for every state.")
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory used when --plot is passed without an explicit filename.",
    )
    parser.add_argument(
        "--plot",
        nargs="?",
        const="auto",
        help="Save a PNG fit diagnostic. With no value, auto-save under --out-dir.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.zigzag or args.zigzag_json:
        run_zigzag_stability(args)
        return

    if args.from_json:
        key, L_vals, eigenvalues = load_from_sweep_json(
            args.from_json, args.x_val, args.y_val, args.json_point_tol
        )
        print(f"Loaded point {key} from {args.from_json}")
    else:
        L_vals = []
        eigenvalues = []
        dims = {}
        for L in args.L_list:
            print(f"Computing L={L} ...")
            try:
                eigenvalue, dim = leading_eigenvalue(args, L)
            except Exception as exc:
                print(f"  failed: {exc}")
                continue
            dims[L] = dim
            if eigenvalue is None:
                print(f"  dim={dim}, no usable eigenvalue")
                continue
            print(f"  dim={dim}, lambda={eigenvalue}")
            L_vals.append(L)
            eigenvalues.append(eigenvalue)

        if dims:
            print("\nDimensions")
            for L in sorted(dims):
                print(f"  L={L}: dim={dims[L]}")

    if len(L_vals) < 3:
        raise SystemExit("Need at least three successful L values to estimate C.")

    fits = build_stability_fits(L_vals, eigenvalues, args.operator, args.vF)
    print_eigenvalue_table(L_vals, eigenvalues, args.operator)
    print_fit_table(fits)
    print_summary(fits)

    if args.out_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
        save_json_report(args.out_json, args, L_vals, eigenvalues, fits)
        print(f"\nWrote JSON report: {args.out_json}")

    if args.plot:
        plot_path = default_plot_path(args) if args.plot == "auto" else args.plot
        os.makedirs(os.path.dirname(os.path.abspath(plot_path)), exist_ok=True)
        save_plot(plot_path, L_vals, eigenvalues, fits, args)
        print(f"Wrote plot: {plot_path}")


if __name__ == "__main__":
    main()
