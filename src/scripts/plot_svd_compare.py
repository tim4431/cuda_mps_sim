#!/usr/bin/env python3
"""
Plot cuSOLVER vs. custom Jacobi SVD performance and accuracy from bench_svd output.

Expected CSV (from ./bench_svd):
    chi_max,N_steps,t_cusolver_s,t_jacobi_s,ratio,max_err

Usage:
  ./bench_svd > svd_data.csv
  python3 scripts/plot_svd_compare.py svd_data.csv --out svd_compare.png
"""

import argparse
import csv
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def load_csv(path):
    rows = []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append({
                "chi_max":      int(row["chi_max"]),
                "N_steps":      int(row["N_steps"]),
                "t_cusolver_s": float(row["t_cusolver_s"]),
                "t_jacobi_s":   float(row["t_jacobi_s"]),
                "ratio":        float(row["ratio"]),
                "max_err":      float(row["max_err"]),
            })
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Plot cuSOLVER vs. custom Jacobi SVD benchmark results.")
    parser.add_argument("csv", help="CSV output from ./bench_svd")
    parser.add_argument("--out", default="svd_compare.png",
                        help="Output PNG (default: svd_compare.png)")
    args = parser.parse_args()

    rows = load_csv(args.csv)
    if not rows:
        print("No data found in", args.csv, file=sys.stderr)
        sys.exit(1)

    chi_vals  = [r["chi_max"]      for r in rows]
    t_cs      = [r["t_cusolver_s"] for r in rows]
    t_jc      = [r["t_jacobi_s"]   for r in rows]
    ratio     = [r["ratio"]        for r in rows]
    max_err   = [r["max_err"]      for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # ----------------------------------------------------------------
    # Panel 1: Wall time (log scale).
    # ----------------------------------------------------------------
    ax = axes[0]
    ax.plot(chi_vals, t_cs, "o-", color="steelblue", label="cuSOLVER gesvdj")
    ax.plot(chi_vals, t_jc, "s-", color="tomato",    label="Custom Jacobi")
    ax.set_xlabel("χ_max")
    ax.set_ylabel("Wall time (s)")
    ax.set_title("SVD Total Time  (L=20, 20 steps)")
    ax.set_yscale("log")
    ax.set_xscale("log", base=2)
    ax.set_xticks(chi_vals)
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    # ----------------------------------------------------------------
    # Panel 2: Jacobi / cuSOLVER ratio.
    # ----------------------------------------------------------------
    ax = axes[1]
    ax.bar(range(len(chi_vals)), ratio, color="mediumseagreen", alpha=0.8)
    ax.axhline(1.0, linestyle="--", color="gray", alpha=0.6, label="equal")
    ax.set_xticks(range(len(chi_vals)))
    ax.set_xticklabels(chi_vals)
    ax.set_xlabel("χ_max")
    ax.set_ylabel("cuSOLVER / Jacobi")
    ax.set_title("Relative Speed  (>1 = Jacobi faster)")
    ax.legend(fontsize=9)
    for i, r in enumerate(ratio):
        ax.text(i, r + 0.02, f"{r:.2f}", ha="center", va="bottom", fontsize=8)

    # ----------------------------------------------------------------
    # Panel 3: Max Schmidt spectrum error.
    # ----------------------------------------------------------------
    ax = axes[2]
    ax.semilogy(chi_vals, max_err, "D-", color="darkorchid")
    ax.axhline(1e-10, linestyle="--", color="tomato", alpha=0.6,
               label="1e-10 threshold")
    ax.set_xlabel("χ_max")
    ax.set_ylabel("Max |Δσ| in Schmidt spectrum")
    ax.set_title("Accuracy vs. cuSOLVER")
    ax.set_xscale("log", base=2)
    ax.set_xticks(chi_vals)
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved {args.out}")

    # Print summary table.
    print(f"\n{'chi_max':>7}  {'t_cs':>10}  {'t_jc':>10}  {'ratio':>7}  {'max_err':>10}")
    for r in rows:
        print(f"{r['chi_max']:>7}  {r['t_cusolver_s']:>10.4f}  "
              f"{r['t_jacobi_s']:>10.4f}  {r['ratio']:>7.2f}  "
              f"{r['max_err']:>10.2e}")


if __name__ == "__main__":
    main()
