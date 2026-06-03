#!/usr/bin/env python3
"""Generate M4 report plots without third-party Python dependencies.

The course cluster image used for development does not always provide
matplotlib.  This script writes simple PNG figures directly with the Python
standard library so the M4 report plots are reproducible from the benchmark
CSVs.

Usage:
  python3 scripts/plot_m4_simple.py --dir . --out ../../doc/figs
"""

import argparse
import csv
import math
import struct
import zlib
from pathlib import Path


WHITE = (255, 255, 255)
BLACK = (35, 35, 35)
GRID = (225, 225, 225)
BLUE = (45, 105, 190)
ORANGE = (220, 110, 35)
GREEN = (35, 145, 85)
PURPLE = (140, 70, 160)
RED = (210, 70, 55)


FONT = {
    "0": ["111", "101", "101", "101", "111"],
    "1": ["010", "110", "010", "010", "111"],
    "2": ["111", "001", "111", "100", "111"],
    "3": ["111", "001", "111", "001", "111"],
    "4": ["101", "101", "111", "001", "001"],
    "5": ["111", "100", "111", "001", "111"],
    "6": ["111", "100", "111", "101", "111"],
    "7": ["111", "001", "010", "010", "010"],
    "8": ["111", "101", "111", "101", "111"],
    "9": ["111", "101", "111", "001", "111"],
    "A": ["010", "101", "111", "101", "101"],
    "B": ["110", "101", "110", "101", "110"],
    "C": ["111", "100", "100", "100", "111"],
    "D": ["110", "101", "101", "101", "110"],
    "E": ["111", "100", "110", "100", "111"],
    "F": ["111", "100", "110", "100", "100"],
    "G": ["111", "100", "101", "101", "111"],
    "H": ["101", "101", "111", "101", "101"],
    "I": ["111", "010", "010", "010", "111"],
    "J": ["001", "001", "001", "101", "111"],
    "K": ["101", "101", "110", "101", "101"],
    "L": ["100", "100", "100", "100", "111"],
    "M": ["101", "111", "111", "101", "101"],
    "N": ["101", "111", "111", "111", "101"],
    "O": ["111", "101", "101", "101", "111"],
    "P": ["111", "101", "111", "100", "100"],
    "Q": ["111", "101", "101", "111", "001"],
    "R": ["110", "101", "110", "101", "101"],
    "S": ["111", "100", "111", "001", "111"],
    "T": ["111", "010", "010", "010", "010"],
    "U": ["101", "101", "101", "101", "111"],
    "V": ["101", "101", "101", "101", "010"],
    "W": ["101", "101", "111", "111", "101"],
    "X": ["101", "101", "010", "101", "101"],
    "Y": ["101", "101", "010", "010", "010"],
    "Z": ["111", "001", "010", "100", "111"],
    ".": ["000", "000", "000", "000", "010"],
    ",": ["000", "000", "000", "010", "100"],
    ":": ["000", "010", "000", "010", "000"],
    "-": ["000", "000", "111", "000", "000"],
    "/": ["001", "001", "010", "100", "100"],
    "%": ["101", "001", "010", "100", "101"],
    "(": ["001", "010", "010", "010", "001"],
    ")": ["100", "010", "010", "010", "100"],
    "=": ["000", "111", "000", "111", "000"],
    " ": ["000", "000", "000", "000", "000"],
}


class Canvas:
    def __init__(self, w=900, h=760):
        self.w = w
        self.h = h
        self.pix = [bytearray(WHITE * w) for _ in range(h)]

    def setp(self, x, y, c):
        x = int(x)
        y = int(y)
        if 0 <= x < self.w and 0 <= y < self.h:
            i = x * 3
            self.pix[y][i:i + 3] = bytes(c)

    def line(self, x0, y0, x1, y1, c, thick=1):
        x0 = int(round(x0))
        y0 = int(round(y0))
        x1 = int(round(x1))
        y1 = int(round(y1))
        dx = abs(x1 - x0)
        sx = 1 if x0 < x1 else -1
        dy = -abs(y1 - y0)
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        while True:
            for ox in range(-thick, thick + 1):
                for oy in range(-thick, thick + 1):
                    self.setp(x0 + ox, y0 + oy, c)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy

    def rect(self, x0, y0, x1, y1, c, fill=True):
        x0 = int(round(x0))
        y0 = int(round(y0))
        x1 = int(round(x1))
        y1 = int(round(y1))
        if fill:
            for y in range(max(0, y0), min(self.h, y1 + 1)):
                for x in range(max(0, x0), min(self.w, x1 + 1)):
                    self.setp(x, y, c)
        else:
            self.line(x0, y0, x1, y0, c)
            self.line(x1, y0, x1, y1, c)
            self.line(x1, y1, x0, y1, c)
            self.line(x0, y1, x0, y0, c)

    def circle(self, cx, cy, r, c):
        cx = int(round(cx))
        cy = int(round(cy))
        r = int(round(r))
        for y in range(cy - r, cy + r + 1):
            for x in range(cx - r, cx + r + 1):
                if (x - cx) ** 2 + (y - cy) ** 2 <= r * r:
                    self.setp(x, y, c)

    def text(self, x, y, s, c=BLACK, scale=2):
        ox = int(x)
        for ch in s.upper():
            pat = FONT.get(ch, FONT[" "])
            for yy, row in enumerate(pat):
                for xx, bit in enumerate(row):
                    if bit == "1":
                        self.rect(ox + xx * scale, y + yy * scale,
                                  ox + (xx + 1) * scale - 1,
                                  y + (yy + 1) * scale - 1, c)
            ox += 4 * scale

    def axes(self, x0, y0, x1, y1):
        self.line(x0, y1, x1, y1, BLACK)
        self.line(x0, y1, x0, y0, BLACK)

    def write(self, path):
        raw = b"".join(b"\x00" + bytes(row) for row in self.pix)

        def chunk(kind, data):
            return (struct.pack(">I", len(data)) + kind + data +
                    struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff))

        png = (b"\x89PNG\r\n\x1a\n" +
               chunk(b"IHDR", struct.pack(">IIBBBBB", self.w, self.h, 8, 2, 0, 0, 0)) +
               chunk(b"IDAT", zlib.compress(raw, 9)) +
               chunk(b"IEND", b""))
        Path(path).write_bytes(png)


def rows(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def x_positions(values, x0, x1):
    if len(values) == 1:
        return {values[0]: (x0 + x1) / 2}
    return {v: x0 + i * (x1 - x0) / (len(values) - 1) for i, v in enumerate(values)}


def plot_gpu(data_dir, out_dir):
    data = [r for r in rows(data_dir / "bench_gpu.csv")
            if int(r["L"]) == 20 and int(r["N_steps"]) == 20]
    chis = [int(r["chi_max"]) for r in data]
    series = [
        ("CPU", [float(r["t_cpu_s"]) for r in data], BLUE),
        ("M3 GPU", [float(r["t_gpu_s"]) for r in data], ORANGE),
        ("PERSIST", [float(r["t_persistent_s"]) for r in data], GREEN),
        ("STREAM", [float(r["t_streamed_s"]) for r in data], PURPLE),
    ]
    c = Canvas()
    x0, y0, x1, y1 = 90, 80, 850, 660
    ymax = max(max(vals) for _, vals, _ in series) * 1.15
    xs = x_positions(chis, x0, x1)
    c.text(250, 25, "M4 GPU TEBD TIMING VS CHI", scale=3)
    c.axes(x0, y0, x1, y1)
    for tick in [0, 10, 20, 30]:
        y = y1 - (tick / ymax) * (y1 - y0)
        c.line(x0 - 5, y, x1, y, GRID)
        c.text(30, int(y) - 8, str(tick))
    for name, vals, col in series:
        pts = [(xs[chis[i]], y1 - (vals[i] / ymax) * (y1 - y0))
               for i in range(len(chis))]
        for a, b in zip(pts, pts[1:]):
            c.line(a[0], a[1], b[0], b[1], col, thick=1)
        for p in pts:
            c.circle(p[0], p[1], 5, col)
    for chi in chis:
        c.text(int(xs[chi]) - 16, y1 + 18, str(chi))
    c.text(380, 700, "CHI_MAX")
    c.text(10, 70, "SECONDS")
    for j, (name, _, col) in enumerate(series):
        c.rect(610, 95 + 25 * j, 628, 109 + 25 * j, col)
        c.text(638, 95 + 25 * j, name)
    c.write(out_dir / "m4_gpu_timing.png")


def plot_svd(data_dir, out_dir):
    data = rows(data_dir / "bench_svd.csv")
    chis = [int(r["chi_max"]) for r in data]
    cs = [float(r["t_cusolver_s"]) for r in data]
    jc = [float(r["t_jacobi_s"]) for r in data]
    c = Canvas()
    x0, y0, x1, y1 = 90, 80, 850, 660
    ymax = max(jc) * 1.15
    xs = x_positions(chis, x0, x1)
    c.text(210, 25, "CUSOLVER VS CUSTOM JACOBI SVD", scale=3)
    c.axes(x0, y0, x1, y1)
    for tick in [0, 20, 40, 60, 80]:
        y = y1 - (tick / ymax) * (y1 - y0)
        c.line(x0 - 5, y, x1, y, GRID)
        c.text(30, int(y) - 8, str(tick))
    for i, chi in enumerate(chis):
        x = xs[chi]
        ycs = y1 - (cs[i] / ymax) * (y1 - y0)
        yjc = y1 - (jc[i] / ymax) * (y1 - y0)
        c.rect(x - 25, ycs, x - 4, y1, BLUE)
        c.rect(x + 4, yjc, x + 25, y1, RED)
        c.text(int(x) - 16, y1 + 18, str(chi))
    c.text(330, 700, "CHI_MAX")
    c.text(10, 70, "SECONDS")
    c.rect(595, 100, 615, 115, BLUE)
    c.text(625, 100, "CUSOLVER")
    c.rect(595, 130, 615, 145, RED)
    c.text(625, 130, "JACOBI")
    c.text(560, 180, "MAX ERR 4.7E-14 TO 1E-12")
    c.write(out_dir / "svd_compare.png")


def total_rank_time(path, n_ranks):
    """Max over ranks of (sum of that rank's pair times).
    Rows in CSV are in rank order: first n_rows/n_ranks rows belong to rank 0, etc.
    """
    times = [float(r["t_wall_s"]) for r in rows(path)]
    per = len(times) // n_ranks
    return max(sum(times[i * per:(i + 1) * per]) for i in range(n_ranks))


def plot_scaling(data_dir, out_dir):
    rank_counts = [1, 2, 4]
    strong = [(p, total_rank_time(data_dir / f"strong_p{p}.csv", p))
              for p in rank_counts]
    weak = [(p, total_rank_time(data_dir / f"weak_p{p}.csv", p))
            for p in rank_counts]

    t1_strong = strong[0][1]
    speedups = [t1_strong / t for _, t in strong]
    effs = [s / p * 100 for (p, _), s in zip(strong, speedups)]

    c = Canvas(w=900, h=800)
    x0, y0, x1, y1 = 90, 80, 850, 680

    # Left half: strong scaling speedup (ranks on x, speedup on y)
    mx = (x0 + x1) // 2 - 20
    xs = {1: x0 + 30, 2: (x0 + mx) // 2 + 30, 4: mx - 10}

    c.text(160, 25, "MPI STRONG SCALING SPEEDUP", scale=3)
    c.axes(x0, y0, mx, y1)
    ideal_y_max = 4.5
    for tick_val in [0, 1, 2, 3, 4]:
        yy = y1 - (tick_val / ideal_y_max) * (y1 - y0)
        c.line(x0 - 5, yy, mx, yy, GRID)
        c.text(x0 - 60, int(yy) - 8, str(tick_val))
    # ideal line
    for p1, p2 in [(1, 2), (2, 4)]:
        yi1 = y1 - (p1 / ideal_y_max) * (y1 - y0)
        yi2 = y1 - (p2 / ideal_y_max) * (y1 - y0)
        c.line(xs[p1], yi1, xs[p2], yi2, GRID, thick=1)
    pts = [(xs[p], y1 - (s / ideal_y_max) * (y1 - y0))
           for (p, _), s in zip(strong, speedups)]
    for a, b in zip(pts, pts[1:]):
        c.line(a[0], a[1], b[0], b[1], BLUE, thick=1)
    for i, ((p, _), pt) in enumerate(zip(strong, pts)):
        c.circle(pt[0], pt[1], 6, BLUE)
        eff_str = str(int(round(effs[i]))) + "%"
        c.text(int(pt[0]) - 16, int(pt[1]) - 24, eff_str, BLUE)
    for p, x in xs.items():
        c.text(x - 4, y1 + 18, str(p))
    c.text(x0 + 60, 700, "RANKS")
    c.text(x0 - 70, 70, "SPEEDUP")
    c.rect(x0 + 10, y0 + 10, x0 + 28, y0 + 22, BLUE)
    c.text(x0 + 36, y0 + 10, "SPEEDUP")
    c.rect(x0 + 10, y0 + 36, x0 + 28, y0 + 48, GRID)
    c.text(x0 + 36, y0 + 36, "IDEAL")

    # Right half: weak scaling total rank time
    rx0 = mx + 40
    xs2 = {1: rx0 + 30, 2: (rx0 + x1) // 2, 4: x1 - 20}
    weak_times = [t for _, t in weak]
    ymax_w = max(weak_times) * 1.20

    c.text(580, 25, "WEAK SCALING RANK TIME", scale=3)
    c.axes(rx0, y0, x1, y1)
    for tick_val in [0, 25, 50, 75, 100]:
        if tick_val <= ymax_w:
            yy = y1 - (tick_val / ymax_w) * (y1 - y0)
            c.line(rx0 - 5, yy, x1, yy, GRID)
            c.text(rx0 - 54, int(yy) - 8, str(tick_val))
    pts2 = [(xs2[p], y1 - (t / ymax_w) * (y1 - y0)) for p, t in weak]
    for a, b in zip(pts2, pts2[1:]):
        c.line(a[0], a[1], b[0], b[1], ORANGE, thick=1)
    t1w = weak[0][1]
    for i, (p, t) in enumerate(weak):
        c.circle(pts2[i][0], pts2[i][1], 6, ORANGE)
        eff_str = str(int(round(t1w / t * 100))) + "%"
        c.text(int(pts2[i][0]) - 16, int(pts2[i][1]) - 24, eff_str, ORANGE)
    for p, x in xs2.items():
        c.text(x - 4, y1 + 18, str(p))
    c.text(rx0 + 60, 700, "RANKS")
    c.text(rx0 - 54, 70, "SECONDS")
    c.rect(rx0 + 10, y0 + 10, rx0 + 28, y0 + 22, ORANGE)
    c.text(rx0 + 36, y0 + 10, "RANK TOTAL (S)")

    c.write(out_dir / "scaling_m4.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=".", help="minimal_tebd_c directory")
    parser.add_argument("--out", default="../../doc/figs", help="output figure directory")
    args = parser.parse_args()
    data_dir = Path(args.dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_gpu(data_dir, out_dir)
    plot_svd(data_dir, out_dir)
    plot_scaling(data_dir, out_dir)
    print(f"Wrote M4 plots to {out_dir}")


if __name__ == "__main__":
    main()
