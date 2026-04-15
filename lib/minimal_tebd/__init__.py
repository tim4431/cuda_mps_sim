"""Minimal, standalone TEBD implementation.

Self-contained: only depends on numpy and scipy. No tenpy import.

The public API intentionally mirrors tenpy's style so that code using this
package can later be swapped to the full tenpy library with minimal changes:

    from minimal_tebd import SimpleMPS, TFIChain, TEBDEngine

    model = TFIChain(L=30, J=1.0, g=1.0)
    psi = SimpleMPS.from_product_state(model.L, [0] * model.L, d=model.d)
    eng = TEBDEngine(psi, model, {
        'dt': 0.1, 'order': 4, 'N_steps': 1,
        'trunc_params': {'chi_max': 100, 'svd_min': 1e-12},
    })
    while eng.evolved_time < 5.0:
        eng.run()

The exact-diagonalization reference (for validation on small L) lives in
``exact_diag.py``; ``validate.py`` runs both side by side and compares
observables.
"""

from .exact_diag import ExactEvolution, bond_model_to_full_hamiltonian
from .model import BondModel, TFIChain
from .mps import PAULI, SimpleMPS
from .tebd import TEBDEngine, TruncationError, update_bond
from .validate import validate_tebd_vs_ed

__all__ = [
    'SimpleMPS',
    'PAULI',
    'TFIChain',
    'BondModel',
    'TEBDEngine',
    'TruncationError',
    'update_bond',
    'ExactEvolution',
    'bond_model_to_full_hamiltonian',
    'validate_tebd_vs_ed',
]
