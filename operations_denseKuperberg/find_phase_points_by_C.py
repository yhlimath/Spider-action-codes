#!/usr/bin/env python3
"""Find sweep points whose fitted C value is close to a target."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path


POINT_KEY_RE = re.compile(
    r"^\s*x=(?P<x>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
    r"_y=(?P<y>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Locate (x, y) sweep points where the estimated C is close to a target value."
    )
    parser.add_argument("json_file", type=Path, help="Path to phase_diagram_data_*.json")
    parser.add_argument("target", type=float, help="Target value for the fitted C coefficient.")
    parser.add_argument(
        "--tolerance",
        type=float,
        help="Only report points with |C - target| <= tolerance. By default, report nearest points.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Maximum number of nearest points to print when no tolerance is given.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="Optional CSV output path for the reported points.",
    )
    parser.add_argument(
        "--field",
        default="C",
        help="JSON field to compare against the target. Defaults to C.",
    )
    return parser.parse_args()


def load_points(path: Path, field: str) -> list[dict[str, float]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object at top level.")

    points: list[dict[str, float]] = []
    skipped = 0

    for key, entry in data.items():
        match = POINT_KEY_RE.match(key)
        if not match or not isinstance(entry, dict) or field not in entry:
            skipped += 1
            continue

        try:
            x_value = float(match.group("x"))
            y_value = float(match.group("y"))
            field_value = float(entry[field])
        except (TypeError, ValueError):
            skipped += 1
            continue

        if not math.isfinite(field_value):
            skipped += 1
            continue

        points.append({"x": x_value, "y": y_value, field: field_value})

    if not points:
        raise ValueError(f"No valid points with field {field!r} found in {path}.")

    if skipped:
        print(f"Skipped {skipped} entries that did not contain usable {field!r} values.")

    return points


def select_points(
    points: list[dict[str, float]],
    field: str,
    target: float,
    tolerance: float | None,
    top: int,
) -> list[dict[str, float]]:
    for point in points:
        point["delta"] = point[field] - target
        point["abs_delta"] = abs(point["delta"])

    points.sort(key=lambda point: (point["abs_delta"], point["x"], point["y"]))

    if tolerance is not None:
        return [point for point in points if point["abs_delta"] <= tolerance]

    return points[: max(top, 1)]


def print_table(points: list[dict[str, float]], field: str, target: float) -> None:
    if not points:
        print(f"No points found close to target {target:g}.")
        return

    print(f"Closest points to {field} = {target:g}")
    print(f"{'rank':>4} {'x':>12} {'y':>12} {field:>16} {'delta':>16} {'|delta|':>16}")
    for rank, point in enumerate(points, start=1):
        print(
            f"{rank:4d} "
            f"{point['x']:12.8g} "
            f"{point['y']:12.8g} "
            f"{point[field]:16.8g} "
            f"{point['delta']:16.8g} "
            f"{point['abs_delta']:16.8g}"
        )


def write_csv(path: Path, points: list[dict[str, float]], field: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["rank", "x", "y", field, "delta", "abs_delta"])
        writer.writeheader()
        for rank, point in enumerate(points, start=1):
            row = {"rank": rank}
            row.update(point)
            writer.writerow(row)
    print(f"Saved {path}")


def main() -> None:
    args = parse_args()
    points = load_points(args.json_file, args.field)
    selected = select_points(points, args.field, args.target, args.tolerance, args.top)
    print_table(selected, args.field, args.target)

    if args.csv:
        write_csv(args.csv, selected, args.field)


if __name__ == "__main__":
    main()
