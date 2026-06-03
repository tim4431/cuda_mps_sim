"""Plot CSV outputs from minimal_tebd_c into PNGs.

Usage:  python3 plot.py [outdir=output]
Produces: <outdir>/figs/{summary,sz_heatmap,sx_heatmap,entropy_heatmap,
                        validate_errors,bench_chi,bench_L}.png
"""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_csv(path):
    with open(path) as f:
        rows = list(csv.reader(f))
    header = rows[0]
    data = np.array([[float(x) for x in r] for r in rows[1:]])
    return header, data


def plot_summary(outdir):
    _, d = read_csv(os.path.join(outdir, "summary.csv"))
    t = d[:, 0]
    s_mid, max_chi, trunc = d[:, 1], d[:, 2], d[:, 3]

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    axes[0].plot(t, s_mid, lw=2)
    axes[0].set_xlabel("t  [1/J]")
    axes[0].set_ylabel(r"mid-bond $S_{vN}$")
    axes[0].set_title("Mid-bond entanglement entropy")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, max_chi, lw=2, color="tab:orange")
    axes[1].set_xlabel("t  [1/J]")
    axes[1].set_ylabel(r"$\max\chi(t)$")
    axes[1].set_title("Bond-dimension growth")
    axes[1].grid(True, alpha=0.3)

    axes[2].semilogy(t, np.maximum(np.abs(trunc), 1e-16), lw=2, color="tab:red")
    axes[2].set_xlabel("t  [1/J]")
    axes[2].set_ylabel(r"$|\sum\epsilon_{trunc}|$")
    axes[2].set_title("Accumulated SVD truncation error")
    axes[2].grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    out = os.path.join(outdir, "figs", "summary.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print("wrote", out)


def plot_heatmap(outdir, basename, label, cmap="viridis", vsymmetric=False):
    path = os.path.join(outdir, basename + ".csv")
    if not os.path.exists(path):
        return
    _, d = read_csv(path)
    t = d[:, 0]
    grid = d[:, 1:].T  # rows=column index (site/bond), cols=time
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    if vsymmetric:
        v = max(abs(grid.min()), abs(grid.max()))
        im = ax.imshow(grid, aspect="auto", origin="lower", cmap=cmap,
                       vmin=-v, vmax=v,
                       extent=[t[0], t[-1], 0, grid.shape[0]])
    else:
        im = ax.imshow(grid, aspect="auto", origin="lower", cmap=cmap,
                       extent=[t[0], t[-1], 0, grid.shape[0]])
    ax.set_xlabel("t  [1/J]")
    ax.set_ylabel("index")
    ax.set_title(label)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    out = os.path.join(outdir, "figs", basename + ".png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print("wrote", out)


def plot_validate(outdir):
    path = os.path.join(outdir, "validate_errors.csv")
    if not os.path.exists(path):
        return
    _, d = read_csv(path)
    t = d[:, 0]
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = [r"$\max_i|\langle\sigma^z_i\rangle|$",
              r"$\max_i|\langle\sigma^x_i\rangle|$",
              r"$\max_{ij}|\langle X_iX_j\rangle|$",
              r"$\max_b|S_b|$"]
    for k, lab in enumerate(labels):
        y = np.maximum(np.abs(d[:, 1 + k]), 1e-16)
        ax.semilogy(t, y, lw=1.6, label=lab)
    ax.set_xlabel("t  [1/J]")
    ax.set_ylabel("max |TEBD - ED|")
    ax.set_title("TEBD vs exact diagonalization (L=8, chi=64, order 4, dt=0.05)")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    out = os.path.join(outdir, "figs", "validate_errors.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print("wrote", out)


def plot_bench(outdir):
    bdir = os.path.join(outdir, "bench")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    p1 = os.path.join(bdir, "timing.csv")
    if os.path.exists(p1):
        _, d = read_csv(p1)
        chi, t = d[:, 0], d[:, 2]
        axes[0].loglog(chi, t, "o-", lw=1.8, ms=8)
        axes[0].set_xlabel(r"$\chi_{max}$")
        axes[0].set_ylabel("wall time [s]")
        axes[0].set_title("Quench cost vs $\\chi_{max}$  (L=20, t=2)")
        axes[0].grid(True, alpha=0.3, which="both")
    p2 = os.path.join(bdir, "timing_L.csv")
    if os.path.exists(p2):
        _, d = read_csv(p2)
        L, t = d[:, 0], d[:, 3]
        axes[1].plot(L, t, "s-", lw=1.8, ms=8, color="tab:green")
        axes[1].set_xlabel("L")
        axes[1].set_ylabel("wall time [s]")
        axes[1].set_title("Quench cost vs L  ($\\chi_{max}=64$, t=1.5)")
        axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    out = os.path.join(outdir, "figs", "bench.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print("wrote", out)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "output"
    os.makedirs(os.path.join(outdir, "figs"), exist_ok=True)
    plot_summary(outdir)
    plot_heatmap(outdir, "entropy_grid",
                 r"Entanglement entropy $S_b(t)$  (L=20)", cmap="viridis")
    plot_heatmap(outdir, "sz_grid",
                 r"$\langle\sigma^z_i(t)\rangle$  (L=20)", cmap="RdBu",
                 vsymmetric=True)
    plot_heatmap(outdir, "sx_grid",
                 r"$\langle\sigma^x_i(t)\rangle$  (L=20)", cmap="RdBu",
                 vsymmetric=True)
    plot_validate(outdir)
    plot_bench(outdir)


if __name__ == "__main__":
    main()
