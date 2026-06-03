# `scripts/` — plotting and job submission

Post-processing for the benchmark CSVs the [`../apps/`](../apps/) binaries write,
plus the Slurm scripts that produce them on the `gpu-turing` cluster. Not
compiled by the Makefile.

## Plotters (Python)

| Script | Reads | Produces |
|---|---|---|
| `plot_bench.py` | `bench_gpu` + `bench_update` CSVs | M3 figure: full-step time vs `L` (latency regime) and per-bond time vs `χ` (compute regime) |
| `plot_breakdown.py` | `bench_breakdown` CSV | per-stage `update_bond_gpu` time vs `χ`, and SVD's share of the total |
| `plot_m4_gpu.py` | `bench_gpu` CSV | M4 figure: wall time and speedup vs `χ_max` (CPU / M3 / persistent / streamed) |
| `plot_m4_simple.py` | `bench_gpu` CSV | same M4 plots with **no third-party deps** (stdlib-only PNG writer, for cluster images without matplotlib) |
| `plot_scaling.py` | `sweep_mpi` strong/weak CSVs | MPI strong-scaling speedup + weak-scaling efficiency curves |
| `plot_svd_compare.py` | `bench_svd` CSV | cuSOLVER vs custom Jacobi: time ratio and Schmidt-spectrum accuracy |

Most use `matplotlib` with the `Agg` backend (no display needed);
`plot_m4_simple.py` is the dependency-free fallback.

## Slurm / orchestration (bash)

| Script | Role |
|---|---|
| `submit_all.sh` | submit every M4 benchmark job to `gpu-turing` (resolves its own project path) |
| `bench_scaling.sh` | `sbatch` job (4 GPUs, ≤15 min): MPI strong/weak scaling sweep over rank counts |
| `verify.sh` | `sbatch` job: full correctness pass — build + unit tests + `validate_*` + 1/2/4-rank sweep consistency check, in one job |

> Cluster jobs must stay under the 15-minute limit; use small problem sizes for
> development and reserve large `χ`/`L` runs for the final benchmarking pass.
