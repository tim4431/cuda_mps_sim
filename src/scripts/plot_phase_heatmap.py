#!/usr/bin/env python3
"""
Plot (J, g) phase-diagram heatmaps of final mid-chain entropy and max bond
dimension from the sweep_mpi strong-scaling CSV.

Usage:
    python3 scripts/plot_phase_heatmap.py <sweep_csv> --out <output.png>

Example:
    python3 scripts/plot_phase_heatmap.py \
        ../lib/minimal_tebd_c/strong_p1.csv \
        --out ../../doc/figs/phase_heatmap.png
"""

import argparse
import csv
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_csv(path):
    rows = []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append({
                "J":             float(row["J"]),
                "g":             float(row["g"]),
                "max_chi":       int(row["max_chi"]),
                "final_entropy": float(row["final_entropy"]),
                "t_wall_s":      float(row["t_wall_s"]),
            })
    return rows


def pivot(rows, key, J_vals, g_vals):
    grid = np.full((len(J_vals), len(g_vals)), np.nan)
    j_idx = {v: i for i, v in enumerate(J_vals)}
    g_idx = {v: i for i, v in enumerate(g_vals)}
    for r in rows:
        i = j_idx.get(r["J"])
        j = g_idx.get(r["g"])
        if i is not None and j is not None:
            grid[i, j] = r[key]
    return grid


def add_heatmap(ax, grid, J_vals, g_vals, title, fmt, cmap):
    im = ax.imshow(grid, cmap=cmap, aspect="auto",
                   origin="lower",
                   extent=[g_vals[0] - 0.25, g_vals[-1] + 0.25,
                           J_vals[0] - 0.25, J_vals[-1] + 0.25])
    ax.set_xticks(g_vals)
    ax.set_yticks(J_vals)
    ax.set_xlabel("g  (transverse field)", fontsize=11)
    ax.set_ylabel("J  (coupling)", fontsize=11)
    ax.set_title(title, fontsize=12)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for i, J in enumerate(J_vals):
        for j, g in enumerate(g_vals):
            val = grid[i, j]
            if not np.isnan(val):
                ax.text(g, J, fmt.format(val),
                        ha="center", va="center",
                        fontsize=9, color="white",
                        fontweight="bold")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="sweep_mpi strong CSV (e.g. strong_p1.csv)")
    parser.add_argument("--out", default="phase_heatmap.png")
    args = parser.parse_args()

    rows = load_csv(args.csv)
    if not rows:
        print("No data in", args.csv, file=sys.stderr)
        sys.exit(1)

    J_vals = sorted(set(r["J"] for r in rows))
    g_vals = sorted(set(r["g"] for r in rows))

    entropy_grid = pivot(rows, "final_entropy", J_vals, g_vals)
    chi_grid     = pivot(rows, "max_chi",       J_vals, g_vals)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    add_heatmap(axes[0], entropy_grid, J_vals, g_vals,
                "Mid-chain entanglement entropy\n(end of evolution)",
                "{:.2f}", "YlOrRd")

    add_heatmap(axes[1], chi_grid, J_vals, g_vals,
                "Max bond dimension $\\chi_a$ reached\n(cap = 64)",
                "{:.0f}", "Blues")

    fig.suptitle(
        "TFIM phase-diagram sweep  ($L=20$, $\\chi_{{\\max}}=64$, 20 steps)",
        fontsize=13, y=1.01)

    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved {args.out}")

    print(f"\n{'J':>5}  {'g':>5}  {'entropy':>10}  {'max_chi':>8}")
    for r in rows:
        print(f"{r['J']:>5.2f}  {r['g']:>5.2f}  "
              f"{r['final_entropy']:>10.6f}  {r['max_chi']:>8d}")


if __name__ == "__main__":
    main()
