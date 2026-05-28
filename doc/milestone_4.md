# Milestone 4 Progress Report

**CME 213 Final Project — GPU-Accelerated TEBD**  
Chiling Han · Xin Wei · Hercy Shen  
**Due: May 27, 2026**

---

## Summary of M4 Deliverables

All four M4 components are implemented and compile cleanly on the `gpu-turing` nodes (Quadro RTX 6000, sm_75, NVHPC 24.1 + CUDA 12.3 + OpenMPI).

| # | Deliverable | Status | Key files |
|---|---|---|---|
| 1 | Persistent device MPS (`GpuMps`) | ✅ Done | `gpu/gpu_tebd.h`, `gpu/gpu_tebd.cu` |
| 2 | CUDA streams within a sub-step | ✅ Done | `gpu/gpu_tebd.cu`, `apps/bench_gpu.cpp` |
| 3 | MPI parameter sweep (`sweep_mpi`) | ✅ Done | `apps/sweep_mpi.cpp`, `Makefile.local` |
| 4 | Custom Jacobi SVD kernel | ✅ Done | `gpu/gpu_tebd.cu`, `apps/bench_svd.cpp` |

---

## 1. Persistent Device MPS

### Design

The `GpuMps` struct (added to `gpu/gpu_tebd.h`) keeps all B-tensors and Schmidt
spectra device-resident for the entire TEBD evolution:

```cpp
struct GpuMps {
  int L, d, chi_max;
  void** d_Bs;   // [L] device pointers, each chi_max*d*chi_max complex
  void** d_Ss;   // [L] device pointers, each chi_max doubles
  int*   chi;    // [L+1] current bond dims (host side)
  explicit GpuMps(const MPS& mps);  // one-time H2D
  void sync_to_host(MPS& mps) const;  // call at measurement points only
};
```

The `update_bond_gpu_persistent()` function reads and writes B-tensors directly
on device, eliminating per-bond H2D/D2H for the large tensor data.  The gate
(256 bytes) is still sent from host each call, which is negligible.

New CUDA kernels added to support the persistent path:

- **`theta2_gpu_kernel<D>`** — builds the two-site tensor θ entirely on device
  from `d_Bs[site]`, `d_Bs[site+1]`, and `d_Ss[site]`.  This replaces the H2D
  copy of theta from `update_bond_gpu`.
- **`compute_ss_inv_kernel`** — computes 1/λ element-wise on device, replacing
  the H2D copy of the inverse Schmidt spectrum.

The Vidal restore kernels (`vidal_restore_left/right`) are reused unchanged;
they now write directly into `gmps.d_Bs[site]` and `gmps.d_Bs[site+1]`.

### Stream-aware SVD

`update_bond_gpu_persistent` accepts a stream via `ws.stream` and calls
`cusolverDnSetStream` before the SVD, making the entire bond pipeline
(gate-contract → row-to-col → SVD → vidal-restore) asynchronous on that stream.

---

## 2. CUDA Streams Within a Sub-Step

### Implementation

In each Suzuki-Trotter sub-step, bonds `{0, 2, 4, …}` (even) or `{1, 3, 5, …}`
(odd) are completely independent — they share no B-tensors.  With
`GpuMps` (device-resident), different bonds read/write disjoint device memory.

`tebd_step_gpu_persistent(eng, gmps, ws_arr, n_ws)` accepts `n_ws` workspaces
each owning a distinct CUDA stream.  It assigns workspace `ws_arr[bond_idx % n_ws]`
round-robin across bonds within a sub-step, then issues a
`cudaDeviceSynchronize()` barrier after all bonds in a sub-step complete before
proceeding to the next.

```cpp
// Pseudocode for one sub-step
for (int b = start; b < L - 1; b += 2, ++bond_idx) {
  GpuTebdWorkspace& ws = *ws_arr[bond_idx % n_ws];
  update_bond_gpu_persistent(gmps, b, gates[b], ..., ws);
}
cudaDeviceSynchronize();  // sub-step barrier
```

`apps/bench_gpu.cpp` benchmarks four modes:
- **CPU** baseline
- **M3 GPU** (`update_bond_gpu`, H2D/D2H per bond)
- **M4 persistent** (single workspace, null stream)
- **M4 streamed** (4 workspaces, 4 streams)

New CSV columns: `t_persistent_s`, `t_streamed_s`.

### Stream Helper API

To avoid `cudaStream_t` in app `.cpp` files compiled with `nvc++`, opaque helper
functions are exposed from `gpu_tebd.h`:

```cpp
void* gpu_stream_create();        // cudaStreamCreate wrapper
void  gpu_stream_destroy(void*);  // cudaStreamDestroy wrapper
void  gpu_device_sync();          // cudaDeviceSynchronize
int   gpu_device_count();         // cudaGetDeviceCount
void  gpu_set_device(int);        // cudaSetDevice
```

---

## 3. MPI Parameter Sweep

### Design rationale

Single-chain TEBD is not amenable to spatial decomposition: a chain split across
ranks would require one MPI round-trip per Trotter sub-step, costing more than
the computation for L ≤ 80.  Instead, each MPI rank owns one GPU and runs an
independent TEBD evolution for one or more `(J, g)` parameter pairs in the TFIM
phase diagram.  There is **zero in-loop communication**; only
`MPI_Gather`/`MPI_Gatherv` at the boundary.

### `apps/sweep_mpi.cpp`

```
MPI_Init
Each rank: cudaSetDevice(rank % n_gpus)
           Build (J, g) param list (strong: round-robin from 4×4 grid;
                                    weak:   4 pairs per rank)
           Warm-up + timed GPU TEBD (persistent MPS, n_streams per rank)
           Compute mid-chain entanglement entropy at end
MPI_Gatherv all results to rank 0
Rank 0: write CSV, print scaling summary to stderr
MPI_Finalize
```

Command-line options: `--L`, `--chi`, `--dt`, `--steps`, `--order`,
`--n-streams`, `--strong`/`--weak`.

### Scaling modes

| Mode | Work per rank | Fixed quantity |
|------|---|---|
| Strong | ≤ 16 pairs / n_ranks | 16 total (J,g) pairs |
| Weak | 4 pairs / rank | pairs per rank |

Strong-scaling efficiency should approach 100% since there is no inter-rank
communication during the TEBD evolution.  The `scripts/plot_scaling.py` script
reads per-configuration CSV files and plots speedup and efficiency curves.

---

## 4. Custom One-Sided Jacobi SVD Kernel

### Motivation

M2 and M3 identified the SVD (cuSOLVER `Zgesvdj`) as the dominant bottleneck:
99.3–99.99% of per-bond GPU time.  The M4 plan originally specified a hand-written
Jacobi SVD as the primary deliverable, with cuSOLVER as fallback.

### Implementation

Two CUDA kernels in `gpu/gpu_tebd.cu`:

**`jacobi_sweep_kernel`** (one thread block × `JACOBI_BLOCK=128` threads):
- Performs cyclic one-sided Jacobi sweeps on the col-major `m × n` input matrix W.
- For each column pair `(p, q)`:
  1. Parallel reduction across all threads to compute `αpp = ‖Wp‖²`,
     `βqq = ‖Wq‖²`, and `γpq = Wp^H Wq` (complex).
  2. Thread 0 computes the Jacobi rotation `(c, sᵣ, e^{-iφ})` from the 2×2
     sub-matrix eigenvalue problem.
  3. All threads apply the rotation in parallel to rows of W and V.
- Shared memory layout:
  `JACOBI_BLOCK + 10` doubles = 1,104 bytes per block (≪ 48 KB limit).
- V is initialized to `Iₙ` inside the kernel (no host setup needed).
- Convergence criterion: `|γpq| < tol·√(αpp·βqq)`, max 100 sweeps.

**`jacobi_finish_kernel`** (one thread block):
- Computes column norms of W (= singular values `σⱼ`) via parallel reduction.
- Sorts `{σⱼ}` descending using insertion sort on thread 0 (n ≤ 256).
- Writes sorted `d_S`, normalised `d_U = W / σ`, and `d_Vout = V` (first K columns).
- Output format is identical to `cusolverDnZgesvdj`, so downstream Vidal-restore
  kernels are reused unchanged.

**Buffer reuse** (zero extra device allocations):
- W = `d_theta` (already holds col-major A after `row_to_col_kernel`)
- V = `d_Utheta` (free after gate contraction; same size as W = `m_max² × 16B`)

### API

```cpp
double update_bond_gpu_jacobi(MPS*, int site, const Tensor& gate,
                               int chi_max, double svd_min,
                               GpuTebdWorkspace&, GpuBondTimings* = nullptr);
```

`apps/bench_svd.cpp` benchmarks both paths at χ ∈ {16, 32, 48, 64, 96, 128}
and verifies that the Schmidt spectra agree to ≤ 10⁻¹³.

---

## Correctness

- All 54 unit tests (6 suites) pass after M4 changes.
- `apps/validate_gpu` confirms GPU↔CPU agreement to ~10⁻¹³ (requires GPU node).
- `bench_svd` reports max Schmidt-spectrum error between cuSOLVER and custom
  Jacobi paths.

---

## Build Instructions

```bash
# Build everything (requires gpu-turing node or login node for compile only):
make -f Makefile.local

# Run CPU tests:
make -f Makefile.local run_tests

# Run GPU benchmarks (submit to gpu-turing partition):
sbatch --partition=gpu-turing --gres=gpu:1 --wrap="./apps/bench_gpu"
sbatch --partition=gpu-turing --gres=gpu:1 --wrap="./apps/bench_svd"
sbatch --partition=gpu-turing --gres=gpu:1 --wrap="./apps/validate_gpu"

# MPI parameter sweep (multi-GPU node):
sbatch --partition=gpu-turing --gres=gpu:4 --ntasks=4 \
       --wrap="mpirun -n 4 ./apps/sweep_mpi --chi 64 --steps 20"
```

---

## File Index

| File | Role |
|---|---|
| `gpu/gpu_tebd.h` | Public API: `GpuMps`, `GpuTebdWorkspace` (stream field), persistent + streamed functions, stream helpers, Jacobi declaration |
| `gpu/gpu_tebd.cu` | Kernels: `theta2_gpu_kernel`, `compute_ss_inv_kernel`, `jacobi_sweep_kernel`, `jacobi_finish_kernel`; functions: `GpuMps::*`, `update_bond_gpu_persistent`, `tebd_step_gpu_persistent`, `update_bond_gpu_jacobi`, stream helpers |
| `apps/bench_gpu.cpp` | CPU / M3 / M4-persistent / M4-streamed timing comparison |
| `apps/sweep_mpi.cpp` | MPI parameter sweep, strong + weak scaling |
| `apps/bench_svd.cpp` | cuSOLVER vs. custom Jacobi per-χ benchmark |
| `Makefile.local` | Build system: MPI detection, CUDA include path fix, all targets |
| `scripts/plot_scaling.py` | Strong + weak scaling plots from sweep_mpi CSV |
| `scripts/plot_svd_compare.py` | cuSOLVER vs. Jacobi time + accuracy plots |
