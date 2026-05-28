#!/usr/bin/env python3
"""
Plot MPI strong and weak scaling results from sweep_mpi output.

Strong-scaling CSV (produced by running sweep_mpi with a fixed 16-pair grid
at ranks 1, 2, 4, 8, 16) has columns:
    rank,J,g,L,chi_max,N_steps,t_wall_s,max_chi,final_entropy

Weak-scaling CSV (--weak mode, 4 pairs/rank) has the same columns.

Usage
-----
  # Collect data (run these on the cluster):
  #   mpirun -n 1  ./sweep_mpi --steps 20 --chi 64 > strong_p1.csv
  #   mpirun -n 2  ./sweep_mpi --steps 20 --chi 64 > strong_p2.csv
  #   mpirun -n 4  ./sweep_mpi --steps 20 --chi 64 > strong_p4.csv
  #   mpirun -n 8  ./sweep_mpi --steps 20 --chi 64 > strong_p8.csv
  #   mpirun -n 16 ./sweep_mpi --steps 20 --chi 64 > strong_p16.csv
  #
  #   mpirun -n 1  ./sweep_mpi --weak --steps 20 --chi 64 > weak_p1.csv
  #   mpirun -n 2  ./sweep_mpi --weak --steps 20 --chi 64 > weak_p2.csv
  #   mpirun -n 4  ./sweep_mpi --weak --steps 20 --chi 64 > weak_p4.csv
  #   mpirun -n 8  ./sweep_mpi --weak --steps 20 --chi 64 > weak_p8.csv
  #
  # Then plot:
  #   python3 scripts/plot_scaling.py \
  #       --strong strong_p1.csv strong_p2.csv strong_p4.csv strong_p8.csv strong_p16.csv \
  #       --weak   weak_p1.csv   weak_p2.csv   weak_p4.csv   weak_p8.csv \
  #       --out    scaling.png
"""

import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_total_rank_time(filepath, n_ranks):
    """Return max-over-ranks of (sum of that rank's pair times).

    Rows in the CSV are ordered by rank after MPI_Gatherv, so the first
    n_rows/n_ranks rows belong to rank 0, the next chunk to rank 1, etc.
    """
    times = []
    with open(filepath, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            times.append(float(row["t_wall_s"]))
    per = len(times) // n_ranks
    return max(sum(times[i * per:(i + 1) * per]) for i in range(n_ranks))


def parse_ranks_from_filenames(files):
    """
    Try to infer the rank counts from filename stems like 'strong_p4.csv' -> 4.
    Falls back to 1, 2, 4, ... if pattern not matched.
    """
    ranks = []
    for f in files:
        stem = os.path.splitext(os.path.basename(f))[0]
        # Look for a trailing digit sequence after 'p' or '_p'.
        import re
        m = re.search(r'[_p](\d+)$', stem)
        if m:
            ranks.append(int(m.group(1)))
        else:
            ranks.append(None)
    # Fill None with powers of two if needed.
    if any(r is None for r in ranks):
        default = [2 ** i for i in range(len(files))]
        ranks = [r if r is not None else default[i] for i, r in enumerate(ranks)]
    return ranks


def plot_strong(ax, files, ranks, t1):
    """
    Strong-scaling efficiency plot.
    ax       : matplotlib Axes
    files    : list of CSV paths at ranks[i]
    ranks    : list of int MPI ranks
    t1       : wall time at 1 rank (reference)
    """
    ts = [read_total_rank_time(f, r) for f, r in zip(files, ranks)]
    efficiency = [t1 / (r * t) for r, t in zip(ranks, ts)]
    speedup    = [t1 / t for t in ts]

    ax_twin = ax.twinx()
    ax.plot(ranks, speedup,    "o-", color="steelblue",  label="Speedup")
    ax.plot(ranks, ranks,      "--", color="gray",        label="Ideal speedup", alpha=0.5)
    ax_twin.plot(ranks, [e * 100 for e in efficiency],
                 "s--", color="tomato", label="Efficiency (%)")

    ax.set_xlabel("MPI ranks")
    ax.set_ylabel("Speedup", color="steelblue")
    ax_twin.set_ylabel("Efficiency (%)", color="tomato")
    ax.set_title("Strong Scaling (16 fixed (J,g) pairs)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ranks)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_ylim(bottom=0)
    ax_twin.set_ylim(0, 110)

    # Combine legends.
    lines1, labs1 = ax.get_legend_handles_labels()
    lines2, labs2 = ax_twin.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labs1 + labs2, loc="upper left", fontsize=8)

    # Annotate data points.
    for r, s, e in zip(ranks, speedup, efficiency):
        ax.annotate(f"{s:.1f}x", (r, s), textcoords="offset points",
                    xytext=(4, 4), fontsize=7, color="steelblue")
        ax_twin.annotate(f"{e*100:.0f}%", (r, e * 100),
                         textcoords="offset points", xytext=(4, -10),
                         fontsize=7, color="tomato")


def plot_weak(ax, files, ranks):
    """
    Weak-scaling efficiency plot (constant work per rank = 4 (J,g) pairs).
    Ideal: constant wall time regardless of rank count.
    """
    ts = [read_total_rank_time(f, r) for f, r in zip(files, ranks)]
    t1 = ts[0]
    efficiency = [t1 / t * 100 for t in ts]

    ax.plot(ranks, ts, "o-", color="darkorange", label="Wall time (s)")
    ax.axhline(t1, linestyle="--", color="gray", alpha=0.5, label="Ideal (constant)")
    ax.set_xlabel("MPI ranks")
    ax.set_ylabel("Max wall time (s)")
    ax.set_title("Weak Scaling (4 pairs/rank)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ranks)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.legend(fontsize=8)

    # Second axis: efficiency.
    ax_twin = ax.twinx()
    ax_twin.plot(ranks, efficiency, "s--", color="purple", label="Efficiency (%)")
    ax_twin.set_ylabel("Efficiency (%)", color="purple")
    ax_twin.set_ylim(0, 110)
    lines2, labs2 = ax_twin.get_legend_handles_labels()
    ax.legend(ax.get_legend_handles_labels()[0] + lines2,
              ax.get_legend_handles_labels()[1] + labs2,
              loc="lower left", fontsize=8)

    for r, t, e in zip(ranks, ts, efficiency):
        ax.annotate(f"{t:.2f}s", (r, t), textcoords="offset points",
                    xytext=(4, 4), fontsize=7, color="darkorange")


def main():
    parser = argparse.ArgumentParser(
        description="Plot MPI strong/weak scaling for sweep_mpi results.")
    parser.add_argument("--strong", nargs="+", metavar="CSV",
                        help="CSV files for strong scaling (one per rank count)")
    parser.add_argument("--weak", nargs="+", metavar="CSV",
                        help="CSV files for weak scaling (one per rank count)")
    parser.add_argument("--out", default="scaling.png",
                        help="Output PNG file (default: scaling.png)")
    args = parser.parse_args()

    if not args.strong and not args.weak:
        parser.print_help()
        sys.exit(1)

    n_plots = (1 if args.strong else 0) + (1 if args.weak else 0)
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    import matplotlib.ticker

    idx = 0
    if args.strong:
        strong_ranks = parse_ranks_from_filenames(args.strong)
        ts = [read_total_rank_time(f, r)
              for f, r in zip(args.strong, strong_ranks)]
        t1 = ts[0]
        plot_strong(axes[idx], args.strong, strong_ranks, t1)
        idx += 1

    if args.weak:
        weak_ranks = parse_ranks_from_filenames(args.weak)
        plot_weak(axes[idx], args.weak, weak_ranks)

    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
