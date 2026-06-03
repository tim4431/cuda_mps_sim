# `tests/` — Google Test unit suite

54 tests across 6 binaries, covering the [`core/`](../core/) CPU library
bottom-up. Google Test is vendored in [`../third_party/googletest/`](../third_party/googletest/),
so no system install is needed.

```bash
make tests        # build the 6 binaries
make run_tests    # build and run them all
```

| Binary | Tests | Coverage |
|---|---|---|
| `test_tensor`   | 9  | alloc, reshape, transpose, copy/move, scale, identity |
| `test_linalg`   | 18 | GEMM, GEMV, SVD (diagonal / complex / rectangular / unitary), `expm`, contraction, kron |
| `test_mps`      | 10 | product states, `theta1`/`theta2`, entropy, expectation values, correlations, state vector |
| `test_model`    | 5  | TFI dimensions, Hermiticity, zero-field limit, field sum rule |
| `test_tebd`     | 6  | norm preservation, `Sz` conservation, entropy growth, bond-dim growth, Trotter-order convergence, SVD truncation |
| `test_validate` | 6  | TEBD vs exact-diag for orders 2 and 4, Néel state, entropy, correlations |

## Tolerances
- Exact operations: `10⁻¹²`.
- SVD-dependent results: `10⁻¹⁰` (the one-sided Jacobi convergence floor).
- TEBD-vs-ED physics: bounded by the `dt⁴` Trotter error (`~10⁻⁶` at `dt=0.05`).

These are CPU-only tests. The GPU path is verified separately by the
`validate_gpu` / `validate_persistent` apps in [`../apps/`](../apps/), which
compare the GPU engine against this same CPU reference.
