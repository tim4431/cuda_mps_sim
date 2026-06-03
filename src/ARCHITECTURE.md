# minimal_tebd_c Architecture

C++14 implementation of TEBD (Time-Evolving Block Decimation) for the 1D
transverse-field Ising model. All linear algebra is hand-written -- no
LAPACK/OpenBLAS -- designed for direct CUDA porting. M3 adds a CUDA
backend in `gpu/` that uses cuSOLVER's Jacobi SVD plus four custom
kernels.

## Directory layout

```
minimal_tebd_c/
├── core/        # CPU library: tensor, linalg, mps, model, tebd,
│                # exact_diag, validate
├── gpu/         # CUDA backend: gpu_tebd.{h,cu}
├── apps/        # driver binaries (sim + validate_gpu + bench_*)
├── tests/       # Google Test binaries (test_*.cpp)
├── scripts/     # plotting / analysis (plot_*.py)
├── third_party/googletest/
├── Makefile     # one Makefile builds all targets; CUDA auto-detected
└── ARCHITECTURE.md
```

The Makefile uses `-Icore -Igpu`, so source files include headers by
plain name (`#include "tensor.h"`, `#include "gpu_tebd.h"`) regardless of
which subdirectory they live in.

## Module dependency graph

```
apps/sim.cpp
  |
  v
core/tebd  -----> core/mps  -----> core/tensor
  |                  |                  ^
  v                  v                  |
core/model      core/linalg ------------+

core/exact_diag  (validation only)
core/validate    (validation only)

gpu/gpu_tebd.cu --(uses)--> core/{mps,tebd,linalg,tensor}.h
                  +
                  cuBLAS + cuSOLVER (Jacobi SVD)
```

## Modules

### tensor.h/cpp

Core data type. Row-major dense tensor with rank 1--4.

- `Cdouble = std::complex<double>` -- layout-compatible with `cuDoubleComplex`.
- Fixed `MAX_NDIM = 4`, explicit `stride[]` array, separate `capacity` vs `size`
  fields to support pre-allocation (avoids `cudaMalloc` during evolution).
- `reshape()` reinterprets without copying; `tensor_transpose()` copies with
  permuted indices.
- Move semantics; deep copy on `operator=`.

### linalg.h/cpp

Hand-written linear algebra routines, each a direct CUDA kernel candidate.

| Function | Purpose | CUDA target |
|---|---|---|
| `zgemm` | Complex matrix multiply (C = aAB + bC) | `cublasZgemm` or tiled kernel |
| `zgemv` | Complex matrix-vector multiply | `cublasZgemv` |
| `svd` | One-sided Jacobi SVD (complex) | Parallel Jacobi rotations |
| `matrix_expm` | Scaling-and-squaring + Taylor series | Pre-computation only |
| `tensor_contract` | N-d contraction via transpose + GEMM | Reduces to `zgemm` |
| `kron` | Kronecker product | Model construction only |

The Jacobi SVD sweeps over column pairs `(p, q)`, applying complex rotations to
orthogonalize the Gram matrix. Non-overlapping pairs are independent -- this maps
to parallel threads on GPU. Typically converges in 5--10 sweeps for the matrix
sizes in TEBD (up to `2*chi_max x 2*chi_max`).

### mps.h/cpp

Right-canonical MPS (Matrix Product State) in Vidal B-form.

```
|psi> = Bs[0] @ Bs[1] @ ... @ Bs[L-1]
```

- `Bs[i]`: rank-3 tensor `(chi_L, d, chi_R)`, right-canonical.
- `Ss[i]`: Schmidt values on the bond *left* of site `i`.
- `theta1(i) = diag(Ss[i]) @ Bs[i]` -- single-site mixed-canonical tensor.
- `theta2(i) = theta1(i) @ Bs[i+1]` -- two-site tensor for TEBD updates.
- Pre-allocated to `chi_max` capacity; shapes track actual bond dimensions.
- Observables: `expect()`, `entropy()`, `corr()`, `to_state_vector()`.
- Pauli matrices defined as file-scope constants (`PAULI_I/X/Y/Z`).

### model.h/cpp

Bond Hamiltonian representation.

- `BondModel`: array of `L-1` rank-4 tensors `H_bonds[b]` with legs
  `(i_out, j_out, i_in, j_in)` acting on sites `(b, b+1)`.
- `tfi_chain(L, J, g)`: constructs the transverse-field Ising chain
  `H = -J sum sigma^x_i sigma^x_{i+1} - g sum sigma^z_i`.

### tebd.h/cpp

TEBD engine -- the hot path.

**Core functions:**

- `update_bond(psi, site, gate, chi_max, svd_min)`: Apply a two-site gate,
  SVD-truncate, restore Vidal B-form. This is the inner loop to port to CUDA.
- `svd_truncate(theta, chi_max, svd_min, A, S, B, eps)`: SVD of the two-site
  tensor `(chi_L, d, d, chi_R)`, truncate to `chi_max`, renormalize.

**TEBDEngine** caches gates and drives the Trotter decomposition:

- Trotter orders 1, 2 (symmetric), and 4 (Forest--Ruth).
- Gates `U_b = expm(-i * frac * dt * H_bond[b])` are pre-computed per unique
  fraction and reused across steps.
- `run()` advances `N_steps * dt`; `step()` executes one Trotter step.

### exact_diag.h/cpp

Full `d^L x d^L` state-vector evolution for validation (small systems only).

- Builds the full Hamiltonian via Kronecker products.
- `ExactEvolution`: holds state vector, computes `expm(-i*dt*H)`, applies via GEMV.
- Single-site and two-site operator embedding in the full Hilbert space.

### validate.h/cpp

Automated TEBD vs exact-diag comparison.

- `validate_tebd_vs_ed()`: Runs both engines on the same initial state, compares
  `<Sz>`, `<Sx>`, entanglement entropy, `<XX>` correlations, and state-vector
  fidelity at each time step.

## Build

```
make              # builds apps/sim (CPU) + apps/{validate_gpu,bench_*} if nvcc is present
make tests       # builds the 6 Google Test binaries in tests/
make run_tests   # builds and runs the test suite
make bench       # builds only the CUDA benchmark/validation apps
make clean
```

Compiler: any C++14 compiler (`g++`) plus optional `nvcc` 12.x for the
CUDA targets. Override `CUDA_HOME` / `CUDA_ARCH` / `CUDA_CCBIN` on the
make command line if your toolchain lives elsewhere. No external
libraries except cuBLAS/cuSOLVER (for the GPU apps) and Google Test
(vendored in `third_party/googletest/`).

## Test suite

| Binary | Tests | What it covers |
|---|---|---|
| `test_tensor` | 9 | Alloc, reshape, transpose, copy/move, scale, identity |
| `test_linalg` | 18 | GEMM, GEMV, SVD (diagonal/complex/rectangular/unitary), expm, contraction, kron |
| `test_mps` | 10 | Product states, theta1/2, entropy, expectation values, correlations, state vector |
| `test_model` | 5 | TFI dimensions, Hermiticity, zero-field limit, field sum rule |
| `test_tebd` | 6 | Norm preservation, Sz conservation, entropy growth, bond dim growth, Trotter order convergence, SVD truncation |
| `test_validate` | 6 | TEBD vs ED for orders 2 and 4, Neel state, entropy, correlations |

## CUDA backend (`gpu/gpu_tebd.{h,cu}`)

The M3 backend implements `update_bond_gpu(psi, site, gate, chi_max, svd_min, ws)`
on top of a persistent `GpuTebdWorkspace` (pre-allocated device buffers,
pinned host staging, cached cuBLAS/cuSOLVER handles). The pipeline per
bond is:

1. custom `gate_contract_kernel<2>` -- 2-site gate * theta on the inner d^2
2. custom `row_to_col_kernel` -- row-major to column-major layout shim
3. `cusolverDnZgesvdj` (econ=1) -- Jacobi SVD, K=min(m,n)
4. custom `vidal_restore_left_kernel<2>` -- new_Bi[a,p,b] = (1/Λ_a) U_col[a*d+p,b] S[b]
5. custom `vidal_restore_right_kernel<2>` -- new_Bj[c,q,e] = conj(V_col[q*chi_R+e,c])

Each stage is bracketed by CUDA events (gated by an optional
`GpuBondTimings*`) so per-stage time is reportable for the
`apps/bench_breakdown` driver.

## Apps (`apps/`)

| Binary             | Purpose                                                    |
|--------------------|------------------------------------------------------------|
| `sim`              | CPU TEBD driver, prints quench CSV (was M2 `main.cpp`).    |
| `validate_gpu`     | Runs CPU and GPU TEBDEngines side-by-side; checks observables. |
| `bench_gpu`        | Full-step CPU vs GPU wall-clock sweep over L and chi_max.  |
| `bench_update`     | Per-call `update_bond` microbench at fixed chi.            |
| `bench_breakdown`  | Per-stage GPU time decomposition via CUDA events.          |

## CUDA porting notes (historical, from M2)

- `Cdouble` is `std::complex<double>`, layout-compatible with `cuDoubleComplex`
  via `reinterpret_cast`.
- All MPS tensors are pre-allocated to `chi_max` capacity -- zero device
  allocations during time evolution.
- `update_bond` is the single hot kernel: gate contraction + SVD + Vidal
  regularisation. Even-parity bonds and odd-parity bonds are independent
  within each Trotter sub-step and can run in parallel (planned for M4).
