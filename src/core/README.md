# `core/` — CPU reference library

Hand-written C++14 TEBD for the 1D transverse-field Ising model. No LAPACK,
BLAS, or Eigen: every numerical routine is written so it maps directly onto a
CUDA kernel (see [`../gpu/`](../gpu/)). This library is both the production CPU
path and the *correctness oracle* against which the GPU backend is validated.

## Dependency order

```
tensor  ←  linalg  ←  mps  ←  model  ←  tebd          (evolution path)
                       ↑
                  exact_diag, validate                 (validation only)
```

## Files

### `tensor.{h,cpp}` — dense complex tensor
Row-major tensor of rank 1–4 holding `Cdouble = std::complex<double>` (layout-
compatible with `cuDoubleComplex` via `reinterpret_cast`). Fixed `MAX_NDIM = 4`,
explicit `stride[]`, and a separate `capacity` vs `size` so tensors can be
**pre-allocated to `χ_max` once** — no `new`/`cudaMalloc` happens during
evolution. `reshape()` reinterprets without copying; `tensor_transpose()` copies
with a permuted index map. Move semantics; deep copy on assignment.

### `linalg.{h,cpp}` — hand-written linear algebra
Each routine is a deliberate CUDA-kernel candidate:

| Function | Purpose | GPU counterpart |
|---|---|---|
| `zgemm` | complex `C = αAB + βC` (row-major) | `cublasZgemm` / tiled kernel |
| `zgemv` | complex matrix–vector | `cublasZgemv` |
| `svd` | **one-sided Jacobi** SVD, `A = U diag(S) Vᴴ`, `S` descending | parallel Jacobi rotations |
| `matrix_expm` | gate exponential via scaling-and-squaring + Taylor | precompute only |
| `tensor_contract` | N-d contraction via transpose + GEMM | reduces to `zgemm` |
| `kron` | Kronecker product | model / ED construction only |

The Jacobi SVD sweeps non-overlapping column pairs `(p, q)`, which are
independent within a sweep — the structure that parallelises on the GPU. It
converges in ~5–10 sweeps for the `≤ 2χ_max × 2χ_max` matrices TEBD produces.

### `mps.{h,cpp}` — right-canonical MPS (Vidal B-form)
`|ψ⟩ = B₀ B₁ … B_{L-1}`, with `Bs[i]` a rank-3 tensor `(χ_L, d, χ_R)` and
`Ss[i]` the Schmidt spectrum on the bond *left* of site `i`. Helpers:
`theta1(i) = diag(Ss[i]) · Bs[i]` (single-site, mixed canonical) and
`theta2(i) = theta1(i) · Bs[i+1]` (two-site tensor consumed by a bond update).
Observables: `expect()`, `entropy()`, `corr()`, `to_state_vector()`. Pauli
matrices are file-scope constants `PAULI_{I,X,Y,Z}`. Pre-allocated to `χ_max`.

### `model.{h,cpp}` — bond Hamiltonian
`BondModel` holds `L-1` rank-4 tensors `H_bonds[b]` with legs
`(i_out, j_out, i_in, j_in)` acting on sites `(b, b+1)`. `tfi_chain(L, J, g)`
builds the TFIM bond Hamiltonians (critical at `J = g = 1`).

### `tebd.{h,cpp}` — the evolution engine (hot path)
- `update_bond(psi, site, gate, χ_max, svd_min)` — apply a two-site gate to
  `theta2`, `svd_truncate`, restore Vidal B-form. **This is the single inner
  loop ported to CUDA.**
- `svd_truncate(theta, …)` — SVD of `(χ_L, d, d, χ_R)`, keep top `χ_max`
  singular values above `svd_min`, renormalise, return truncation error `eps`.
- `TEBDEngine` — caches the Trotter `schedule` and pre-computes gates
  `U_b = expm(-i · frac · dt · H_bond[b])` once per unique fraction. Supports
  Trotter orders 1, 2 (symmetric), and 4 (Forest–Ruth), plus real (`type_evo=0`)
  and imaginary-time (`1`) evolution. `run()` advances `N_steps · dt`; `step()`
  does one Trotter step.

### `exact_diag.{h,cpp}` — full state-vector oracle
Builds the full `d^L × d^L` Hamiltonian via Kronecker products and evolves the
exact state vector with `expm(-i·dt·H)` applied by GEMV. Feasible only for
`L ≲ 14`. Used to certify TEBD physics, not for production.

### `validate.{h,cpp}` — TEBD vs ED comparison
`validate_tebd_vs_ed()` runs both engines from the same product state and
returns max per-step errors in `⟨σᶻ⟩`, `⟨σˣ⟩`, `⟨XX⟩` correlations, entropy,
and state-vector fidelity. The TEBD↔ED agreement floor is the `dt⁴` Trotter
error (`~10⁻⁶` at `dt = 0.05`).

## Design invariants worth keeping
- **Zero allocations in the inner loop.** All tensors are sized to `χ_max` at
  construction; the only per-step heap traffic is the small Schmidt arrays.
- **`Cdouble` is binary-compatible with `cuDoubleComplex`** — host buffers are
  handed to CUDA by `reinterpret_cast`, no element-wise conversion.
- The CPU `svd` and the GPU's cuSOLVER `gesvdj` agree on singular values to
  `~10⁻¹⁴`; singular *vectors* may differ by per-column phase, so validation
  compares observables and `S`, never raw `U`/`V`.
