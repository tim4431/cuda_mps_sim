# `gpu/` — CUDA backend

`gpu_tebd.{h,cu}` accelerates the TEBD bond update on a single GPU and exposes
the helpers that let the MPI driver ([`../apps/sweep_mpi.cpp`](../apps/sweep_mpi.cpp))
run one independent evolution per rank. It depends only on the
[`core/`](../core/) headers (`mps.h`, `tebd.h`, `tensor.h`) plus cuBLAS and
cuSOLVER. All CUDA-specific types (`cudaStream_t`, handles) are hidden behind
opaque `void*`s so app `.cpp` files never include `cuda_runtime.h`.

## The bond-update pipeline

A two-site update factorises as

```
Θ'_(v_L, i'j', v_R) = Σ_ij  G_(i'j'),(ij) · Θ_(v_L, ij, v_R)     (gate contraction)
Θ'_(χ_L d) × (d χ_R) = U Σ Vᴴ                                     (SVD)
restore to Vidal B-form (truncate to χ_max, divide by left Schmidt values)
```

Stages, with `m = χ_L·d`, `n = d·χ_R`, `K = min(m, n)` (block shapes for `d=2`):

| # | Stage | Kernel / call | Notes |
|---|---|---|---|
| 1 | gate contraction | `gate_contract_kernel<2>` | gate `G` staged in shared memory, `#pragma unroll` |
| 2 | row → column layout | `row_to_col_kernel` | coalesced reads; feeds column-major SVD |
| 3 | SVD | `cusolverDnZgesvdj` (`econ=1`) **[production]** | Jacobi, `K = min(m,n)` |
| 4 | Vidal restore left | `vidal_restore_left_kernel<2>` | `new_Bi[a,p,b] = U_col[a·d+p, b]·S[b] / Λ_a` |
| 5 | Vidal restore right | `vidal_restore_right_kernel<2>` | `new_Bj[c,q,e] = conj(V_col[q·χ_R+e, c])` |

Each stage is bracketed by CUDA events (when a `GpuBondTimings*` is passed) so
[`../apps/bench_breakdown.cpp`](../apps/bench_breakdown.cpp) can report per-stage time.

## Two SVD backends

- **cuSOLVER `gesvdj` (production)** — `update_bond_gpu*`. The
  `lwork` buffer size is re-queried per call because `gesvdj_bufferSize` is
  non-monotone in `(m, n)` (an out-of-bounds bug caught with
  `compute-sanitizer`).
- **Custom one-sided Jacobi (experimental)** — `update_bond_gpu_jacobi`, built
  from `jacobi_sweep_kernel` + `jacobi_finish_kernel`. One block of 128 threads:
  parallel reductions compute the `2×2` Gram sub-matrix `(αpp, βqq, γpq)`,
  thread 0 forms the complex rotation, all threads apply it to rows of `W` and
  `V`; convergence `|γpq| < tol·√(αpp·βqq)`, ≤ 200 sweeps. Wide matrices
  (`m < n`) are handled by decomposing `Aᴴ` (`col_major_adjoint_kernel`) and
  swapping buffers. Output format matches cuSOLVER, so the restore kernels are
  reused unchanged. It agrees with cuSOLVER to `~10⁻¹²` but runs 1.2–2.2× slower,
  so cuSOLVER stays the production backend. Benchmarked by
  [`../apps/bench_svd.cpp`](../apps/bench_svd.cpp).

## Persistent device MPS (`GpuMps`)

M3 staged `theta2` through the host every bond; M4's `GpuMps` keeps all `L`
B-tensors (`χ_max·d·χ_max` complex each) and Schmidt spectra device-resident for
the whole evolution. `update_bond_gpu_persistent` builds `theta2` on-device
(`theta2_gpu_kernel`), inverts the left Schmidt spectrum on-device
(`compute_ss_inv_kernel`), and writes the restored B-tensors straight back into
`gmps.d_Bs`. The **bulk B-tensor transfers are eliminated** — what remains per
bond is only small traffic: the 256-byte gate H2D, a `K`-double singular-value
D2H (needed for host-side truncation, see below), and the normalized Schmidt
spectrum H2D. `sync_to_host()` is called *only* at measurement points. This
removed the M3 bulk transfers (profiled H2D/D2H drops to `~0.2 MB`).

Truncation (how many singular values to keep, renormalization) is done on the
host, so each bond issues a `cudaStreamSynchronize` after the SVD to read the
singular values back. That mid-pipeline sync is why, in practice, bonds run
close to serially even with multiple streams (see below).

## Streams within a Trotter sub-step

Even bonds `{0,2,4,…}` and odd bonds `{1,3,5,…}` within one sub-step touch
disjoint device memory, so they are independent. `tebd_step_gpu_persistent`
takes `n_ws` workspaces, each with its own `cudaStream_t` and its own cuSOLVER
handle (bound to that stream via `cusolverDnSetStream` each bond), and assigns
them round-robin across bonds, then issues one `cudaDeviceSynchronize()` barrier
per sub-step. Measured speedup from 4 streams is negligible — note that each
bond's host-side truncation forces a `cudaStreamSynchronize` mid-pipeline, so
consecutive bonds do not actually overlap across streams in the current
implementation.

## Persistent workspace (`GpuTebdWorkspace`)

Allocated once at engine construction, reused every bond — the device-side
mirror of the host MPS's pre-allocation strategy. Holds all device scratch
(`d_theta`, `d_Utheta`/`A_col`, `d_U`, `d_Vt`, `d_S`, `d_Ss_in`, cuSOLVER
`lwork`), **pinned** host staging buffers (`cudaMallocHost`) for async PCIe, the
cuBLAS+cuSOLVER handle bundle, and an optional `cudaStream_t`. Non-copyable;
store several in a `std::vector<unique_ptr<…>>` for the multi-stream path.

## Public API surface (`gpu_tebd.h`)

```cpp
// Per-bond updates
double update_bond_gpu          (MPS*, site, gate, χ_max, svd_min, ws, timings=nullptr);
double update_bond_gpu_jacobi   (MPS*, site, gate, χ_max, svd_min, ws, timings=nullptr);
double update_bond_gpu_persistent(GpuMps&, site, gate, χ_max, svd_min, ws, timings=nullptr);

// One Trotter step on the GPU
void tebd_step_gpu           (TEBDEngine&, GpuTebdWorkspace&);                 // M3 host-staging
void tebd_step_gpu_persistent(TEBDEngine&, GpuMps&, GpuTebdWorkspace* const*, n_ws);
void tebd_step_gpu_persistent(TEBDEngine&, GpuMps&, GpuTebdWorkspace&);        // single-ws overload

// Opaque stream / device helpers (keep CUDA headers out of app .cpp files)
void* gpu_stream_create();   void gpu_stream_destroy(void*);   void gpu_device_sync();
int   gpu_device_count();    void gpu_set_device(int);
```

## Resolved gotchas (see milestone 3/4 reports)
- cuSOLVER returns `V`, not `Vᴴ`, in column-major — a silent gauge bug that
  diverged after step 1. Fixed by the explicit row→col transpose (stage 2) and
  conjugating inside `vidal_restore_right_kernel`.
- The gate→shared-memory load originally used `if (tid < 16)`, which zero-padded
  the gate when a block had `< 16` threads (small `χ_R`); replaced by a strided
  cooperative loop.
- `gesvdj_bufferSize` non-monotonicity → re-query `lwork` every call.
