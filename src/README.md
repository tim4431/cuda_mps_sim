# GPU-Accelerated TEBD for the 1D Transverse-Field Ising Model

CME 213 final project. A from-scratch C++14/CUDA/MPI implementation of
**TEBD** (Time-Evolving Block Decimation) that simulates real-time quench
dynamics of the 1D transverse-field Ising model (TFIM)

```
H = -J Σ_i σˣ_i σˣ_{i+1} - g Σ_i σᶻ_i
```

on a **Matrix Product State** (MPS). The wavefunction of an `L`-site spin-½
chain lives in a `2^L`-dimensional Hilbert space; the MPS compresses it to
`O(L · χ² · d)` numbers, where the bond dimension `χ` tracks the entanglement.
Each Trotter time step applies `L-1` two-site gates, and each gate update is a
small **complex SVD** — the operation that dominates runtime and is the target
of all GPU work here.

## What the code does

- **Input:** `L`, `(J, g)`, an initial product state, time step `dt`, number of
  steps, max bond dimension `χ_max`, Trotter order (1, 2, or 4).
- **Output:** time series of `⟨σᶻ_i⟩`, `⟨σˣ_i⟩`, two-point correlations,
  bipartite entanglement entropy on every cut, the bond-dimension profile, and
  accumulated SVD truncation error (CSV files under `output/`).
- **Assumptions:** open boundaries, local dimension `d = 2`, nearest-neighbour
  bonds, no LAPACK/Eigen — all linear algebra (GEMM, GEMV, SVD, `expm`) is
  hand-written so every kernel is visible and CUDA-portable.

## Layout

| Directory | Contents | README |
|---|---|---|
| [`core/`](core/)   | CPU reference library: tensor, linalg, MPS, model, TEBD engine, exact-diag oracle, validator | [core/README.md](core/README.md) |
| [`gpu/`](gpu/)     | CUDA backend: custom kernels + cuSOLVER/custom Jacobi SVD, persistent device MPS, streams, MPI helpers | [gpu/README.md](gpu/README.md) |
| [`apps/`](apps/)   | Driver and benchmark binaries (`sim`, `sweep_mpi`, `validate_*`, `bench_*`) | [apps/README.md](apps/README.md) |
| [`tests/`](tests/) | Google Test unit suite (54 tests, 6 binaries) | [tests/README.md](tests/README.md) |
| [`scripts/`](scripts/) | Plotting and Slurm job scripts | [scripts/README.md](scripts/README.md) |
| `third_party/googletest/` | Vendored Google Test | — |
| `output/` | Benchmark CSVs and figures committed for the reports | — |

See [ARCHITECTURE.md](ARCHITECTURE.md) for the module dependency graph and the
per-stage GPU `update_bond` pipeline.

## Build

Two Makefiles select the toolchain; targets are identical.

```bash
make                # local/dev build (RTX, sm_120); CUDA apps auto-built if nvcc present
make -f Makefile.local   # course cluster build (gpu-turing, Quadro RTX 6000, sm_75, NVHPC + OpenMPI)

make tests          # build the 6 Google Test binaries
make run_tests      # build + run the unit suite
make bench          # build only the GPU benchmark/validation apps
make clean
```

Header search uses `-Icore -Igpu`, so sources include headers by plain name
(`#include "tensor.h"`, `#include "gpu_tebd.h"`). The app list is `$(wildcard)`-
driven, so the build adapts to whatever exists in `apps/`. Override
`CUDA_HOME` / `CUDA_ARCH` / `CUDA_CCBIN` on the command line for other
toolchains. The only external dependencies are cuBLAS/cuSOLVER (GPU apps),
MPI (`sweep_mpi` only), and the vendored Google Test.

## Parallelism, in one paragraph

A single TEBD chain has limited spatial parallelism (even/odd bonds within a
sub-step are independent, but a spatial decomposition would need one MPI
round-trip per sub-step). So the design is **task-parallel**: each MPI rank
owns one GPU and runs an independent TEBD evolution over `(J, g)` points in the
TFIM phase diagram, with *zero in-loop communication* and only a final
result-gather (`MPI_Gather` of counts + `MPI_Gatherv` of records). On one GPU, the bond update keeps the MPS
device-resident (`GpuMps`) and overlaps independent bonds across CUDA streams.
The cuSOLVER Jacobi SVD is `>99.6%` of GPU kernel time, so the implementation
is **compute-bound on the SVD**; MPI scaling is limited only by load imbalance
across `(J, g)` pairs. Details and measurements are in the milestone reports
under [`../doc/`](../doc/).
