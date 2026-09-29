#!/usr/bin/env python3
"""Aggregate multi-seed JSONL metrics and render dependency-free SVG plots.

Inputs are one or more ``metrics.jsonl`` files from ``run_grpo.py``.  Outputs
are CSV summaries and SVG figures containing per-group means and 95% normal
confidence bands.  ``--final-per-run`` produces learning-rate sweep plots from
the final matching record in each run.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean, stdev
from typing import Any


COLORS = ["#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed", "#0891b2"]


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", help="metrics.jsonl files")
    parser.add_argument("--metric", action="append", required=True)
    parser.add_argument("--kind", choices=["train", "val"], default="val")
    parser.add_argument("--x-key", default="step")
    parser.add_argument("--group-key", default="algorithm")
    parser.add_argument("--final-per-run", action="store_true")
    parser.add_argument("--output-dir", default="experiments/plots")
    return parser


def read_records(path: Path, kind: str) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            record = json.loads(line)
            if record.get("kind") == kind:
                records.append(record)
    return records


def aggregate(
    log_paths: list[Path],
    *,
    metric: str,
    kind: str,
    x_key: str,
    group_key: str,
    final_per_run: bool,
) -> list[dict[str, Any]]:
    """Aggregate one scalar metric by experiment group and x coordinate."""
    values: dict[tuple[str, float], list[float]] = defaultdict(list)
    for path in log_paths:
        # Older my_grpo_clean.py logs kept run metadata only in config.json.
        # Recover it here so historical standard runs can share plots with
        # learning-rate, prompt, and algorithm ablations.
        config_path = path.with_name("config.json")
        config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
        prompt = config.get("prompt_name") or config.get("prompt")
        if prompt is None and config.get("prompt_path"):
            prompt = Path(config["prompt_path"]).stem
        if prompt == "r1_zero_three_shot_gsm8k":
            prompt = "r1_zero_three_shot"
        defaults = {
            "algorithm": config.get("algorithm", "standard"),
            "prompt_name": prompt,
            "learning_rate": config.get("learning_rate"),
        }
        records = []
        for record in read_records(path, kind):
            record = dict(record)
            for key, value in defaults.items():
                if value is not None:
                    record.setdefault(key, value)
            if metric in record and x_key in record:
                records.append(record)
        if final_per_run and records:
            records = [max(records, key=lambda record: record.get("step", 0))]
        for record in records:
            group = str(record.get(group_key, "all"))
            values[(group, float(record[x_key]))].append(float(record[metric]))

    summary = []
    for (group, x_value), samples in sorted(values.items()):
        sample_std = stdev(samples) if len(samples) > 1 else 0.0
        summary.append(
            {
                "group": group,
                "x": x_value,
                "n": len(samples),
                "mean": fmean(samples),
                "std": sample_std,
                "ci95": 1.96 * sample_std / math.sqrt(len(samples)),
                "min": min(samples),
                "max": max(samples),
            }
        )
    if not summary:
        raise ValueError(f"no {kind!r} records contain metric {metric!r}")
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write aggregated statistics for later analysis or typesetting."""
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render_svg(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    metric: str,
    x_label: str,
) -> None:
    """Render mean curves and confidence bands as a standalone SVG."""
    width, height = 960, 560
    left, right, top, bottom = 90, 30, 45, 75
    plot_width = width - left - right
    plot_height = height - top - bottom
    x_values = [row["x"] for row in rows]
    y_values = [value for row in rows for value in (row["mean"] - row["ci95"], row["mean"] + row["ci95"])]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    if x_min == x_max:
        x_min, x_max = x_min - 0.5, x_max + 0.5
    if y_min == y_max:
        padding = max(abs(y_min) * 0.05, 0.5)
        y_min, y_max = y_min - padding, y_max + padding
    y_padding = (y_max - y_min) * 0.08
    y_min, y_max = y_min - y_padding, y_max + y_padding

    def x_pixel(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_width

    def y_pixel(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_height

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="25" text-anchor="middle" font-family="sans-serif" font-size="18">{html.escape(metric)}</text>',
    ]
    for tick in range(6):
        fraction = tick / 5
        x_value = x_min + fraction * (x_max - x_min)
        x = x_pixel(x_value)
        elements.extend(
            [
                f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_height}" stroke="#e5e7eb"/>',
                f'<text x="{x:.2f}" y="{top + plot_height + 24}" text-anchor="middle" font-family="sans-serif" font-size="12">{x_value:.4g}</text>',
            ]
        )
        y_value = y_min + fraction * (y_max - y_min)
        y = y_pixel(y_value)
        elements.extend(
            [
                f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}" stroke="#e5e7eb"/>',
                f'<text x="{left - 12}" y="{y + 4:.2f}" text-anchor="end" font-family="sans-serif" font-size="12">{y_value:.4g}</text>',
            ]
        )
    elements.extend(
        [
            f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="black"/>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="black"/>',
            f'<text x="{left + plot_width / 2}" y="{height - 20}" text-anchor="middle" font-family="sans-serif" font-size="14">{html.escape(x_label)}</text>',
            f'<text x="20" y="{top + plot_height / 2}" text-anchor="middle" transform="rotate(-90 20 {top + plot_height / 2})" font-family="sans-serif" font-size="14">{html.escape(metric)}</text>',
        ]
    )

    groups = sorted({row["group"] for row in rows})
    for group_index, group in enumerate(groups):
        color = COLORS[group_index % len(COLORS)]
        group_rows = sorted((row for row in rows if row["group"] == group), key=lambda row: row["x"])
        upper = [
            (x_pixel(row["x"]), y_pixel(row["mean"] + row["ci95"])) for row in group_rows
        ]
        lower = [
            (x_pixel(row["x"]), y_pixel(row["mean"] - row["ci95"]))
            for row in reversed(group_rows)
        ]
        polygon = " ".join(f"{x:.2f},{y:.2f}" for x, y in upper + lower)
        mean_line = " ".join(
            f"{x_pixel(row['x']):.2f},{y_pixel(row['mean']):.2f}" for row in group_rows
        )
        elements.extend(
            [
                f'<polygon points="{polygon}" fill="{color}" opacity="0.16"/>',
                f'<polyline points="{mean_line}" fill="none" stroke="{color}" stroke-width="2.5"/>',
            ]
        )
        for row in group_rows:
            elements.append(
                f'<circle cx="{x_pixel(row["x"]):.2f}" cy="{y_pixel(row["mean"]):.2f}" r="3" fill="{color}"/>'
            )
        legend_y = top + 18 * group_index
        elements.extend(
            [
                f'<line x1="{left + 12}" y1="{legend_y}" x2="{left + 38}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>',
                f'<text x="{left + 44}" y="{legend_y + 4}" font-family="sans-serif" font-size="12">{html.escape(group)}</text>',
            ]
        )
    elements.append("</svg>")
    path.write_text("\n".join(elements) + "\n", encoding="utf-8")


def main() -> None:
    args = make_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [Path(path) for path in args.logs]
    for metric in args.metric:
        rows = aggregate(
            paths,
            metric=metric,
            kind=args.kind,
            x_key=args.x_key,
            group_key=args.group_key,
            final_per_run=args.final_per_run,
        )
        stem = f"{args.kind}_{metric}"
        write_csv(output_dir / f"{stem}.csv", rows)
        render_svg(
            output_dir / f"{stem}.svg",
            rows,
            metric=metric,
            x_label=args.x_key,
        )


if __name__ == "__main__":
    main()
