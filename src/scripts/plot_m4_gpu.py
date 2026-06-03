#!/usr/bin/env python3
"""Plot M4 GPU TEBD timing benchmark (CPU / M3 GPU / Persistent / Streamed).

Reads bench_gpu.csv and produces a figure with two panels:
  (a) Wall time vs chi_max  (L=20, 20 steps)
  (b) Speedup vs chi_max    (GPU persistent / CPU)

Usage:
  python3 scripts/plot_m4_gpu.py [bench_gpu.csv] [--out doc/figs/m4_gpu_timing.png]
"""

import argparse
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def load(path, L=20, N_steps=20):
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            if int(r["L"]) == L and int(r["N_steps"]) == N_steps:
                rows.append({
                    "chi":       int(r["chi_max"]),
                    "cpu":       float(r["t_cpu_s"]),
                    "m3":        float(r["t_gpu_s"]),
                    "pers":      float(r["t_persistent_s"]),
                    "streamed":  float(r["t_streamed_s"]),
                })
    rows.sort(key=lambda r: r["chi"])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default="bench_gpu.csv")
    ap.add_argument("--out", default="../../doc/figs/m4_gpu_timing.png")
    args = ap.parse_args()

    rows = load(args.csv)
    chis     = [r["chi"]     for r in rows]
    t_cpu    = [r["cpu"]     for r in rows]
    t_m3     = [r["m3"]      for r in rows]
    t_pers   = [r["pers"]    for r in rows]
    t_str    = [r["streamed"]for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # ── Panel (a): wall time ──────────────────────────────────────────────
    ax = axes[0]
    ax.plot(chis, t_cpu,  "o-",  color="#1f77b4", label="CPU (reference)")
    ax.plot(chis, t_m3,   "s--", color="#d62728", label="M3 GPU (H2D/D2H per bond)")
    ax.plot(chis, t_pers, "^-",  color="#2ca02c", label="M4 persistent MPS")
    ax.plot(chis, t_str,  "D:",  color="#9467bd", label="M4 streamed (4 streams)")
    ax.set_xlabel(r"$\chi_{\max}$")
    ax.set_ylabel("Wall time (s)")
    ax.set_title(r"(a) TEBD wall time vs $\chi_{\max}$  ($L{=}20$, 20 steps)")
    ax.set_xticks(chis)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # ── Panel (b): speedup (CPU / GPU-persistent) ─────────────────────────
    ax = axes[1]
    speedup = [c / p for c, p in zip(t_cpu, t_pers)]
    ax.plot(chis, speedup, "^-", color="#2ca02c", lw=2)
    ax.axhline(1.0, linestyle="--", color="gray", alpha=0.6, label="break-even")
    for chi, sp in zip(chis, speedup):
        ax.annotate(f"{sp:.2f}×", (chi, sp),
                    textcoords="offset points", xytext=(0, 6),
                    ha="center", fontsize=8, color="#2ca02c")
    ax.set_xlabel(r"$\chi_{\max}$")
    ax.set_ylabel("Speedup  (CPU / persistent GPU)")
    ax.set_title("(b) CPU/GPU speedup (persistent MPS)")
    ax.set_xticks(chis)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
