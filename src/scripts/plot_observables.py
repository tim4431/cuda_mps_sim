#!/usr/bin/env python3
"""
Plot (J, g) heatmaps of GPU-computed observables from sweep_mpi output.

Expected CSV columns (L=20):
  rank,J,g,L,chi_max,N_steps,t_wall_s,max_chi,final_entropy,
  sz_0,...,sz_{L-1},sx_0,...,sx_{L-1},xx_0_0,...,xx_{L-1,L-1}

Usage:
    python3 scripts/plot_observables.py <sweep_csv> --out <output.png>

Example:
    python3 scripts/plot_observables.py \
        ../lib/minimal_tebd_c/strong_p4_obs.csv \
        --out ../../doc/figs/observables_heatmap.png
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
            L = int(row["L"])
            sz = np.array([float(row[f"sz_{i}"]) for i in range(L)])
            sx = np.array([float(row[f"sx_{i}"]) for i in range(L)])
            xx = np.array([[float(row[f"xx_{i}_{j}"]) for j in range(L)]
                           for i in range(L)])
            rows.append({
                "J":             float(row["J"]),
                "g":             float(row["g"]),
                "L":             L,
                "max_chi":       int(row["max_chi"]),
                "final_entropy": float(row["final_entropy"]),
                "sz":            sz,
                "sx":            sx,
                "xx":            xx,
            })
    return rows


def pivot_scalar(rows, key, J_vals, g_vals):
    grid = np.full((len(J_vals), len(g_vals)), np.nan)
    j_idx = {v: i for i, v in enumerate(J_vals)}
    g_idx = {v: i for i, v in enumerate(g_vals)}
    for r in rows:
        i = j_idx.get(round(r["J"], 4))
        j = g_idx.get(round(r["g"], 4))
        if i is not None and j is not None:
            grid[i, j] = r[key]
    return grid


def pivot_obs(rows, obs_key, J_vals, g_vals, reduce_fn):
    """Pivot a per-site observable reduced to a scalar by reduce_fn."""
    grid = np.full((len(J_vals), len(g_vals)), np.nan)
    j_idx = {v: i for i, v in enumerate(J_vals)}
    g_idx = {v: i for i, v in enumerate(g_vals)}
    for r in rows:
        i = j_idx.get(round(r["J"], 4))
        j = g_idx.get(round(r["g"], 4))
        if i is not None and j is not None:
            grid[i, j] = reduce_fn(r[obs_key])
    return grid


def add_heatmap(ax, grid, J_vals, g_vals, title, fmt, cmap, vmin=None, vmax=None):
    im = ax.imshow(grid, cmap=cmap, aspect="auto",
                   origin="lower",
                   extent=[g_vals[0] - 0.25, g_vals[-1] + 0.25,
                           J_vals[0] - 0.25, J_vals[-1] + 0.25],
                   vmin=vmin, vmax=vmax)
    ax.set_xticks(g_vals)
    ax.set_yticks(J_vals)
    ax.set_xlabel("g  (transverse field)", fontsize=10)
    ax.set_ylabel("J  (coupling)", fontsize=10)
    ax.set_title(title, fontsize=11)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for i, J in enumerate(J_vals):
        for j, g in enumerate(g_vals):
            val = grid[i, j]
            if not np.isnan(val):
                ax.text(g, J, fmt.format(val),
                        ha="center", va="center",
                        fontsize=9, color="white", fontweight="bold")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="sweep_mpi observable CSV")
    parser.add_argument("--out", default="observables_heatmap.png")
    args = parser.parse_args()

    rows = load_csv(args.csv)
    if not rows:
        print("No data in", args.csv, file=sys.stderr)
        sys.exit(1)

    J_vals = sorted(set(round(r["J"], 4) for r in rows))
    g_vals = sorted(set(round(r["g"], 4) for r in rows))

    # Scalar grids: entropy, mean |sz|, mean |sx|, mid-chain xx correlation.
    entropy_grid  = pivot_scalar(rows, "final_entropy", J_vals, g_vals)
    mean_sz_grid  = pivot_obs(rows, "sz", J_vals, g_vals,
                              lambda v: float(np.mean(np.abs(v))))
    mean_sx_grid  = pivot_obs(rows, "sx", J_vals, g_vals,
                              lambda v: float(np.mean(np.abs(v))))
    # Mid-chain correlator: <sx_{L/2} sx_{L/2+1}> as representative NN pair.
    def mid_corr(xx):
        L = xx.shape[0]
        i = L // 2 - 1
        return xx[i, i + 1]
    midcorr_grid  = pivot_obs(rows, "xx", J_vals, g_vals, mid_corr)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    add_heatmap(axes[0, 0], entropy_grid, J_vals, g_vals,
                "Mid-chain entanglement entropy", "{:.2f}", "YlOrRd")

    add_heatmap(axes[0, 1], mean_sz_grid, J_vals, g_vals,
                r"Mean $|\langle\sigma^z_i\rangle|$  (magnetization)", "{:.3f}",
                "Blues", vmin=0)

    add_heatmap(axes[1, 0], mean_sx_grid, J_vals, g_vals,
                r"Mean $|\langle\sigma^x_i\rangle|$  (transverse magnetization)", "{:.3f}",
                "Greens", vmin=0)

    add_heatmap(axes[1, 1], midcorr_grid, J_vals, g_vals,
                r"$\langle\sigma^x_{L/2}\sigma^x_{L/2+1}\rangle$  (NN correlator)",
                "{:.3f}", "PuOr")

    fig.suptitle(
        r"TFIM observables  ($L=20$, $\chi_{\max}=64$, 20 steps)",
        fontsize=13, y=1.01)

    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved {args.out}")

    # Summary table.
    print(f"\n{'J':>5}  {'g':>5}  {'entropy':>10}  "
          f"{'mean|sz|':>10}  {'mean|sx|':>10}  {'xx_mid':>10}")
    for r in sorted(rows, key=lambda x: (x["J"], x["g"])):
        L = r["L"]
        i = L // 2 - 1
        print(f"{r['J']:>5.2f}  {r['g']:>5.2f}  "
              f"{r['final_entropy']:>10.6f}  "
              f"{np.mean(np.abs(r['sz'])):>10.6f}  "
              f"{np.mean(np.abs(r['sx'])):>10.6f}  "
              f"{r['xx'][i, i+1]:>10.6f}")


if __name__ == "__main__":
    main()
