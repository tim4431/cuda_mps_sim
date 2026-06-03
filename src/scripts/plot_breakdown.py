#!/usr/bin/env python3
"""Per-stage time breakdown of update_bond_gpu vs chi."""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, NullLocator, FuncFormatter


def read_breakdown(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("chi,") or not line[0].isdigit():
                continue
            parts = line.split(",")
            if len(parts) < 8:
                continue
            try:
                rows.append({
                    "chi": int(parts[0]),
                    "h2d": float(parts[1]),
                    "gate": float(parts[2]),
                    "r2c": float(parts[3]),
                    "svd": float(parts[4]),
                    "restore": float(parts[5]),
                    "d2h": float(parts[6]),
                    "total": float(parts[7]),
                })
            except ValueError:
                continue
    return rows


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "bench_breakdown.csv"
    out = sys.argv[2] if len(sys.argv) > 2 else "breakdown.pdf"

    rows = read_breakdown(csv_path)
    rows.sort(key=lambda r: r["chi"])
    chis = [r["chi"] for r in rows]
    cats = [("svd", "cuSOLVER gesvdj", "#f58518", "s"),
            ("h2d", "H2D copy",        "#4c78a8", "o"),
            ("gate", "gate_contract",   "#54a24b", "^"),
            ("restore", "Vidal restore","#72b7b2", "D"),
            ("d2h", "D2H copy",         "#9d755d", "v"),
            ("r2c", "row_to_col",       "#e45756", "P")]

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4))

    # Panel (a): each stage as a line, log-log vs chi.
    ax = axes[0]
    for k, lbl, c, m in cats:
        vals = [r[k] for r in rows]
        ax.loglog(chis, vals, marker=m, color=c, label=lbl, lw=1.2,
                  markersize=5)
    ax.set_xlabel(r"bond dimension $\chi$")
    ax.set_ylabel("per-stage time  [ms]")
    ax.set_title("(a) per-stage time vs $\chi$")
    ax.xaxis.set_major_locator(FixedLocator(chis))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}"))
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.85, ncol=2)

    # Panel (b): SVD's share of total -- one bar per chi.
    ax = axes[1]
    x = np.arange(len(chis))
    svd_share = np.array([100 * r["svd"] / r["total"] for r in rows])
    overhead = 100 - svd_share
    ax.bar(x, svd_share, color="#f58518", edgecolor="white",
           label="cuSOLVER gesvdj")
    ax.bar(x, overhead, bottom=svd_share, color="#4c78a8", edgecolor="white",
           label="all other stages")
    for xi, sv in zip(x, svd_share):
        ax.text(xi, 50, f"{sv:.1f}%", ha="center", va="center",
                color="white", fontsize=9, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in chis])
    ax.set_xlabel(r"bond dimension $\chi$")
    ax.set_ylabel("share of per-bond time  [%]")
    ax.set_title("(b) cuSOLVER SVD share of update_bond_gpu")
    ax.set_ylim(0, 100)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.85)

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    if out.endswith(".pdf"):
        fig.savefig(out[:-4] + ".png", dpi=160, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
