# minimal_tebd

Self-contained TEBD (Time-Evolving Block Decimation) for finite spin chains. Only depends on NumPy and SciPy — no TeNPy import required.

The API mirrors TeNPy's style so code can later be swapped to the full library with minimal changes.

## Modules

| File | Purpose |
|---|---|
| `mps.py` | `SimpleMPS` — right-canonical MPS with expectation values, correlators, and entanglement entropy |
| `model.py` | `BondModel` base class and `TFIChain` (transverse-field Ising) |
| `tebd.py` | `TEBDEngine` — Trotter decomposition (order 1, 2, 4) with SVD truncation |
| `exact_diag.py` | `ExactEvolution` — full state-vector reference for validation (L ≤ 14) |
| `validate.py` | Side-by-side TEBD vs ED comparison with per-observable error reporting |
| `example_notebook.py` | Runs a TFI global quench and collects observables |
| `visualize.py` | 9-panel figure: entropy, magnetization, correlators, bond dimension, truncation error |

## Quick start

```python
from lib.minimal_tebd import SimpleMPS, TFIChain, TEBDEngine

model = TFIChain(L=30, J=1.0, g=1.0)
psi = SimpleMPS.from_product_state(model.L, [0] * model.L, d=model.d)
eng = TEBDEngine(psi, model, {
    'dt': 0.1, 'order': 4, 'N_steps': 1,
    'trunc_params': {'chi_max': 100, 'svd_min': 1e-12},
})

while eng.evolved_time < 5.0:
    eng.run()
    S = eng.psi.entanglement_entropy()
    Sz = eng.psi.expectation_value('Sigmaz')
```

## Running the examples

```bash
# Console output of quench dynamics
python -m lib.minimal_tebd.example_notebook

# 9-panel visualization (saves tebd_visualization.png)
python -m lib.minimal_tebd.visualize

# TEBD vs exact-diag validation
python -m lib.minimal_tebd.validate
```

## Physics

The built-in model is the transverse-field Ising chain:

$$H = -J \sum_{i} \sigma^x_i \sigma^x_{i+1} - g \sum_{i} \sigma^z_i$$

Critical at J = g = 1. The default simulation starts from |↑↑...↑⟩ (the g → ∞ ground state) and time-evolves under the critical Hamiltonian — a global quantum quench.

### Observables measured

- **Entanglement entropy** S(b) = −Tr(ρ_b log ρ_b) at each bond
- **Local magnetization** ⟨σ^z_i⟩ and ⟨σ^x_i⟩
- **Correlation function** ⟨σ^x_i σ^x_j⟩
- **Bond dimension** χ (grows until it saturates at χ_max)
- **Truncation error** ε = 1 − ∏(1 − ε_k) accumulated over all SVD truncations

## MPS conventions

The MPS is stored in right-canonical B-form:

- `Bs[i]` has shape `(χ_L, d, χ_R)` with legs `(vL, phys, vR)`, right-orthonormal
- `Ss[i]` holds the Schmidt values on the bond left of site i
- The mixed-canonical two-site tensor `θ = diag(Ss[i]) @ Bs[i] @ Bs[i+1]` is used for the TEBD update

## Custom models

Subclass `BondModel` or provide any object with `L`, `d`, and `H_bonds` attributes:

```python
from lib.minimal_tebd import BondModel
import numpy as np

# H_bonds[b] has shape (d, d, d, d) with index order (i_out, j_out, i_in, j_in)
H_list = [my_two_site_hamiltonian.reshape(2, 2, 2, 2) for _ in range(L - 1)]
model = BondModel(H_list, L=L, d=2)
```

## CUDA porting

The hot path is `update_bond()` in `tebd.py` — gate contraction + SVD + re-canonicalization. This is the function to port to CUDA for GPU acceleration.
