#!/usr/bin/env python3
"""Plot CPU vs GPU TEBD benchmarks for the M3 report.

Reads two CSVs produced by bench_gpu and bench_update and emits a single
side-by-side figure summarising the two main scaling axes:
  (a) full-step time vs L at fixed chi_max (latency-dominated regime)
  (b) per-bond update_bond time vs chi (compute-dominated regime)

Usage:
  python plot_bench.py bench_step.csv bench_update.csv [out.pdf]
"""

import sys
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_step_csv(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("g++") \
                    or line.startswith("/data") or line.startswith("L,") \
                    or line.lower().startswith("l="):
                continue
            parts = line.split(",")
            if len(parts) < 7:
                continue
            try:
                rows.append({
                    "L": int(parts[0]),
                    "chi_max": int(parts[1]),
                    "N_steps": int(parts[2]),
                    "t_cpu": float(parts[3]),
                    "t_gpu": float(parts[4]),
                    "speedup": float(parts[5]),
                    "max_chi": int(parts[6]),
                })
            except ValueError:
                continue
    return rows


def read_update_csv(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("chi,") or line.startswith("chi="):
                continue
            parts = line.split(",")
            if len(parts) != 4:
                continue
            try:
                rows.append({
                    "chi": int(parts[0]),
                    "t_cpu": float(parts[1]),
                    "t_gpu": float(parts[2]),
                    "speedup": float(parts[3]),
                })
            except ValueError:
                continue
    return rows


def main():
    step_csv = sys.argv[1] if len(sys.argv) > 1 else "bench_step.csv"
    upd_csv = sys.argv[2] if len(sys.argv) > 2 else "bench_update.csv"
    out = sys.argv[3] if len(sys.argv) > 3 else "bench_M3.pdf"

    step = read_step_csv(step_csv)
    upd = read_update_csv(upd_csv)

    # Sweep over L at fixed chi_max=64.  The CSV contains two test families
    # (chi_max sweep at L=20, and L sweep at chi_max=64); keep only the L
    # sweep (N_steps=15).
    by_L = [r for r in step
            if r["chi_max"] == 64 and r["N_steps"] == 15
            and r["L"] in (12, 20, 28, 40, 60, 80)]
    by_L.sort(key=lambda r: r["L"])
    Ls = [r["L"] for r in by_L]
    t_cpu_L = [r["t_cpu"] for r in by_L]
    t_gpu_L = [r["t_gpu"] for r in by_L]

    # Microbenchmark vs chi.
    upd.sort(key=lambda r: r["chi"])
    chis = [r["chi"] for r in upd]
    t_cpu_c = [r["t_cpu"] * 1e3 for r in upd]
    t_gpu_c = [r["t_gpu"] * 1e3 for r in upd]

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))

    # Panel (a): full TEBD step time vs L.
    ax = axes[0]
    ax.plot(Ls, t_cpu_L, "o-", label="CPU (g++ -O2)", color="#1f77b4")
    ax.plot(Ls, t_gpu_L, "s-", label="GPU (RTX 5090)", color="#d62728")
    ax.set_xlabel("system size L")
    ax.set_ylabel(r"time for 15 TEBD steps  [s]")
    ax.set_title(r"(a) full-step time, $\chi_{\max}{=}64$, $dt{=}0.1$")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")

    # Panel (b): per-bond update_bond time vs chi.
    ax = axes[1]
    ax.loglog(chis, t_cpu_c, "o-", label="CPU update_bond", color="#1f77b4")
    ax.loglog(chis, t_gpu_c, "s-", label="GPU update_bond_gpu", color="#d62728")
    ax.set_xlabel(r"bond dimension $\chi$")
    ax.set_ylabel("per-bond time  [ms]")
    ax.set_title(r"(b) per-bond cost vs $\chi$  (synthetic state, L=16)")
    # The measured chi values (16, 32, ..., 200) fall in a single decade of
    # the log scale; matplotlib's default major ticks (10, 100) hide them.
    # Pin major ticks to the actual sample points so the x-axis is readable.
    from matplotlib.ticker import FixedLocator, NullLocator, FuncFormatter
    ax.xaxis.set_major_locator(FixedLocator(chis))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}"))
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")
    # Also write a PNG for quick inspection.
    if out.endswith(".pdf"):
        fig.savefig(out[:-4] + ".png", dpi=160, bbox_inches="tight")


if __name__ == "__main__":
    main()
