#!/usr/bin/env python3
"""Plot phase diagrams from phase_diagram_sweep.py JSON output."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import tempfile
from pathlib import Path

if "MPLCONFIGDIR" not in os.environ:
    mpl_cache = Path(tempfile.gettempdir()) / "phase-diagram-matplotlib-cache"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_cache)

if "XDG_CACHE_HOME" not in os.environ:
    xdg_cache = Path(tempfile.gettempdir()) / "phase-diagram-xdg-cache"
    xdg_cache.mkdir(parents=True, exist_ok=True)
    os.environ["XDG_CACHE_HOME"] = str(xdg_cache)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import numpy as np
import colorsys


POINT_KEY_RE = re.compile(
    r"^\s*x=(?P<x>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
    r"_y=(?P<y>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*$"
)

FILENAME_RE = re.compile(
    r"phase_diagram_data_(?P<operator>[^_]+)_n(?P<n>[-+]?\d+(?:\.\d+)?)"
    r"_z(?P<z>[-+]?\d+(?:\.\d+)?)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct the x-y sweep grid from a phase_diagram_sweep.py JSON "
            "file and save a phase-diagram plot."
        )
    )
    parser.add_argument("json_file", type=Path, help="Path to phase_diagram_data_*.json")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output image path. Defaults to <json stem>_<quantity>.png in the current directory.",
    )
    parser.add_argument(
        "--quantity",
        choices=("central_charge", "C", "A", "B"),
        default="central_charge",
        help="Quantity to plot. central_charge uses c = -24 C / (pi * velocity).",
    )
    parser.add_argument(
        "--velocity",
        type=float,
        default=1.0,
        help="Velocity factor used only for --quantity central_charge.",
    )
    parser.add_argument(
        "--style",
        choices=("contour", "heatmap"),
        default="contour",
        help="Use filled contours or a cell heatmap.",
    )
    parser.add_argument("--levels", type=int, default=50, help="Number of contour levels.")
    parser.add_argument("--cmap", help="Matplotlib colormap name.")
    parser.add_argument(
        "--desaturate",
        type=float,
        default=0.45,
        help="Desaturate the colormap by this fraction; 0 keeps original colors, 1 is grayscale.",
    )
    parser.add_argument("--dpi", type=int, default=200, help="Output image DPI.")
    parser.add_argument("--title", help="Custom plot title.")
    parser.add_argument(
        "--critical-levels",
        nargs="*",
        type=float,
        default=(0.0, 0.8),
        help="Central-charge contour levels to emphasize when plotting central_charge.",
    )
    parser.add_argument(
        "--near-zero-c-threshold",
        type=float,
        default=-0.01,
        help="Draw the boundary of near-zero regions |c| <= this value. Use a negative value to disable.",
    )
    parser.add_argument(
        "--high-abs-c-threshold",
        type=float,
        default=20.0,
        help="Draw the boundary of high-|c| regions |c| >= this value. Use a negative value to disable.",
    )
    parser.add_argument(
        "--show-points",
        action="store_true",
        help="Overlay sampled x-y points on top of the phase diagram.",
    )
    parser.add_argument(
        "--figsize",
        nargs=2,
        type=float,
        metavar=("WIDTH", "HEIGHT"),
        default=(8.0, 6.0),
        help="Figure size in inches.",
    )
    return parser.parse_args()


def infer_metadata(path: Path) -> dict[str, str]:
    match = FILENAME_RE.search(path.name)
    return match.groupdict() if match else {}


def load_json_grid(path: Path, field: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object at top level.")

    points: list[tuple[float, float, float]] = []
    warnings: list[str] = []

    for key, entry in data.items():
        match = POINT_KEY_RE.match(key)
        if not match:
            warnings.append(f"Skipping key with unexpected format: {key!r}")
            continue

        if not isinstance(entry, dict) or field not in entry or entry[field] is None:
            warnings.append(f"Skipping {key!r}: missing numeric field {field!r}")
            continue

        try:
            x_value = float(match.group("x"))
            y_value = float(match.group("y"))
            z_value = float(entry[field])
        except (TypeError, ValueError):
            warnings.append(f"Skipping {key!r}: {field!r} is not a finite number")
            continue

        if not math.isfinite(z_value):
            warnings.append(f"Skipping {key!r}: {field!r} is not finite")
            continue

        points.append((x_value, y_value, z_value))

    if not points:
        raise ValueError(f"No plottable points with field {field!r} found in {path}.")

    x_values = np.array(sorted({point[0] for point in points}), dtype=float)
    y_values = np.array(sorted({point[1] for point in points}), dtype=float)
    grid = np.full((len(y_values), len(x_values)), np.nan, dtype=float)

    x_index = {value: index for index, value in enumerate(x_values)}
    y_index = {value: index for index, value in enumerate(y_values)}

    for x_value, y_value, z_value in points:
        grid[y_index[y_value], x_index[x_value]] = z_value

    missing = int(np.isnan(grid).sum())
    if missing:
        warnings.append(f"Grid has {missing} missing x-y cells; those cells are masked.")

    return x_values, y_values, grid, warnings


def centers_to_edges(values: np.ndarray) -> np.ndarray:
    if len(values) == 1:
        return np.array([values[0] - 0.5, values[0] + 0.5], dtype=float)

    edges = np.empty(len(values) + 1, dtype=float)
    edges[1:-1] = 0.5 * (values[:-1] + values[1:])
    edges[0] = values[0] - 0.5 * (values[1] - values[0])
    edges[-1] = values[-1] + 0.5 * (values[-1] - values[-2])
    return edges


def transformed_grid(grid: np.ndarray, quantity: str, velocity: float) -> tuple[np.ndarray, str, str]:
    if quantity == "central_charge":
        if velocity == 0:
            raise ValueError("--velocity must be nonzero for central_charge.")
        return -24.0 * grid / (np.pi * velocity), "Extrapolated central charge c", "YlGnBu"

    labels = {
        "C": "Finite-size fit coefficient C",
        "A": "Thermodynamic-limit coefficient A",
        "B": "1/L correction coefficient B",
    }
    return grid, labels[quantity], "viridis"


def desaturated_cmap(cmap_name: str, amount: float) -> LinearSegmentedColormap:
    amount = min(max(amount, 0.0), 1.0)
    base = plt.get_cmap(cmap_name)
    samples = np.linspace(0.0, 1.0, 256)
    colors = []
    for value in samples:
        red, green, blue, alpha = base(value)
        hue, lightness, saturation = colorsys.rgb_to_hls(red, green, blue)
        red, green, blue = colorsys.hls_to_rgb(hue, lightness, saturation * (1.0 - amount))
        colors.append((red, green, blue, alpha))
    return LinearSegmentedColormap.from_list(f"{cmap_name}_desaturated_{amount:.2f}", colors)


def default_title(path: Path, quantity_label: str) -> str:
    metadata = infer_metadata(path)
    if not metadata:
        return quantity_label

    pieces = [quantity_label]
    if "operator" in metadata:
        pieces.append(f"operator={metadata['operator']}")
    if "n" in metadata:
        pieces.append(f"n={metadata['n']}")
    if "z" in metadata:
        pieces.append(f"z={metadata['z']}")
    return " | ".join(pieces)


def indicate_c_region_boundaries(
    ax: plt.Axes,
    x_values: np.ndarray,
    y_values: np.ndarray,
    grid: np.ndarray,
    near_zero_threshold: float,
    high_abs_threshold: float,
    signed_levels: tuple[float, ...],
) -> dict[str, bool]:
    finite_values = grid[np.isfinite(grid)]
    if finite_values.size == 0:
        return {"near_zero": False, "high_abs": False, "signed": False}

    masked_grid = np.ma.masked_invalid(grid)
    abs_grid = np.ma.masked_invalid(np.abs(grid))
    abs_values = np.abs(finite_values)
    abs_min = float(abs_values.min())
    abs_max = float(abs_values.max())
    drew_near_zero = False
    drew_high_abs = False
    legend_handles: list[Line2D] = []

    if near_zero_threshold >= 0 and abs_min <= near_zero_threshold <= abs_max:
        near_zero_contour = ax.contour(
            x_values,
            y_values,
            abs_grid,
            levels=[near_zero_threshold],
            colors=["#3f7f85"],
            linewidths=1.65,
            linestyles="solid",
            zorder=7,
        )
        ax.clabel(
            near_zero_contour,
            inline=True,
            fontsize=8,
            fmt=lambda value: f"|c|={value:g}",
        )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color="#3f7f85",
                linewidth=1.65,
                label=f"near-zero boundary |c| = {near_zero_threshold:g}",
            )
        )
        drew_near_zero = True

    if high_abs_threshold >= 0 and abs_min <= high_abs_threshold <= abs_max:
        high_abs_contour = ax.contour(
            x_values,
            y_values,
            abs_grid,
            levels=[high_abs_threshold],
            colors=["#8a4b38"],
            linewidths=1.8,
            linestyles="dashed",
            zorder=8,
        )
        ax.clabel(
            high_abs_contour,
            inline=True,
            fontsize=8,
            fmt=lambda value: f"|c|={value:g}",
        )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color="#8a4b38",
                linewidth=1.8,
                linestyle="dashed",
                label=f"high-|c| boundary |c| = {high_abs_threshold:g}",
            )
        )
        drew_high_abs = True

    contour_levels = sorted(
        level for level in signed_levels if finite_values.min() <= level <= finite_values.max()
    )
    if contour_levels:
        contour = ax.contour(
            x_values,
            y_values,
            masked_grid,
            levels=contour_levels,
            colors=["#315d6b"],
            linewidths=1.15,
            zorder=6,
        )
        ax.clabel(contour, inline=True, fontsize=8, fmt=lambda value: f"c={value:g}")

    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper right", frameon=True, framealpha=0.88)

    return {"near_zero": drew_near_zero, "high_abs": drew_high_abs, "signed": bool(contour_levels)}


def plot_phase_diagram(args: argparse.Namespace) -> Path:
    field = "C" if args.quantity == "central_charge" else args.quantity
    x_values, y_values, raw_grid, warnings = load_json_grid(args.json_file, field)
    grid, colorbar_label, default_cmap = transformed_grid(raw_grid, args.quantity, args.velocity)

    output = args.output
    if output is None:
        output = Path.cwd() / f"{args.json_file.stem}_{args.quantity}.png"

    output.parent.mkdir(parents=True, exist_ok=True)

    finite_values = grid[np.isfinite(grid)]
    if finite_values.size == 0:
        raise ValueError("All transformed grid values are NaN or infinite.")

    fig, ax = plt.subplots(figsize=tuple(args.figsize), constrained_layout=True)
    masked_grid = np.ma.masked_invalid(grid)
    cmap = desaturated_cmap(args.cmap or default_cmap, args.desaturate)

    if args.style == "heatmap" or len(x_values) < 2 or len(y_values) < 2:
        x_edges = centers_to_edges(x_values)
        y_edges = centers_to_edges(y_values)
        color_plot = ax.pcolormesh(x_edges, y_edges, masked_grid, shading="auto", cmap=cmap)
        x_limits = (float(x_edges.min()), float(x_edges.max()))
        y_limits = (float(y_edges.min()), float(y_edges.max()))
    else:
        level_count = max(args.levels, 2)
        color_plot = ax.contourf(x_values, y_values, masked_grid, levels=level_count, cmap=cmap)
        x_limits = (float(x_values.min()), float(x_values.max()))
        y_limits = (float(y_values.min()), float(y_values.max()))

    boundary_status = {"near_zero": False, "high_abs": False, "signed": False}
    if args.quantity == "central_charge":
        boundary_status = indicate_c_region_boundaries(
            ax,
            x_values,
            y_values,
            grid,
            args.near_zero_c_threshold,
            args.high_abs_c_threshold,
            tuple(args.critical_levels),
        )

    if args.show_points:
        yy, xx = np.where(np.isfinite(grid))
        ax.scatter(x_values[xx], y_values[yy], s=8, c="#f7f7f7", edgecolors="none", alpha=0.50)

    colorbar = fig.colorbar(color_plot, ax=ax)
    colorbar.set_label(colorbar_label)

    ax.set_xlabel("Boltzmann weight x")
    ax.set_ylabel("Boltzmann weight y")
    ax.set_title(args.title or default_title(args.json_file, colorbar_label))
    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)

    fig.savefig(output, dpi=args.dpi)
    plt.close(fig)

    print(f"Saved {output}")
    print(
        "Grid: "
        f"{len(x_values)} x-values, {len(y_values)} y-values, "
        f"{finite_values.size} finite cells, "
        f"value range [{finite_values.min():.6g}, {finite_values.max():.6g}]"
    )
    if args.quantity == "central_charge":
        print(
            "Region boundaries: "
            f"near_zero={boundary_status['near_zero']}, "
            f"high_abs={boundary_status['high_abs']}, "
            f"signed_contours={boundary_status['signed']}"
        )
    for warning in warnings[:10]:
        print(f"Warning: {warning}")
    if len(warnings) > 10:
        print(f"Warning: {len(warnings) - 10} additional warnings omitted.")

    return output


def main() -> None:
    args = parse_args()
    plot_phase_diagram(args)


if __name__ == "__main__":
    main()
