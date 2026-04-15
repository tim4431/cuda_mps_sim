"""Exact-diagonalisation reference: full ``2**L``-dimensional state evolution.

Only feasible for small systems (L <= ~14 on a laptop), but invaluable as
a correctness check for TEBD.

Two conventions matter and must match ``mps.py``:

1. The basis ordering of the full vector. We use the standard "big-endian"
   convention where the composite index ``k = sum_i s_i * d**(L-1-i)``
   corresponds to the product state ``|s_0 s_1 ... s_{L-1}>``. This matches
   ``np.kron(A, B) = A (x) B`` with ``A`` acting on the *left* site.

2. The bond Hamiltonian layout. ``H_bonds[b]`` has shape ``(d, d, d, d)``
   with indices ``(i_out, j_out, i_in, j_in)``; after ``.reshape(d*d, d*d)``
   it matches ``np.kron`` on the two-site subspace in the same big-endian order.
"""

import numpy as np
from scipy.linalg import expm


def bond_model_to_full_hamiltonian(model):
    """Assemble the full ``d**L x d**L`` Hamiltonian from bond Hamiltonians.

    ``H = sum_b I^{(x)b} (x) H_b (x) I^{(x)(L-b-2)}``
    """
    L, d = model.L, model.d
    dim = d ** L
    H = np.zeros((dim, dim), dtype=complex)
    I_d = np.eye(d)
    for b, h in enumerate(model.H_bonds):
        left = np.eye(d ** b)
        right = np.eye(d ** (L - b - 2))
        h_mat = h.reshape(d * d, d * d)
        H += np.kron(np.kron(left, h_mat), right)
    return H


def _single_site_op_full(op, i, L, d=2):
    """Embed a ``(d, d)`` single-site operator at site ``i`` into the full space."""
    return np.kron(np.kron(np.eye(d ** i), op), np.eye(d ** (L - i - 1)))


def _two_site_op_full(op_A, op_B, i, j, L, d=2):
    """Embed ``op_A_i op_B_j`` (with ``i != j``) into the full space."""
    if i == j:
        return _single_site_op_full(op_A @ op_B, i, L, d)
    # Sort so i < j for the Kron assembly, but keep track of which op goes where.
    if i < j:
        lo, op_lo, hi, op_hi = i, op_A, j, op_B
    else:
        lo, op_lo, hi, op_hi = j, op_B, i, op_A
    I = np.eye(d)
    parts = [I] * L
    parts[lo] = op_lo
    parts[hi] = op_hi
    out = parts[0]
    for k in range(1, L):
        out = np.kron(out, parts[k])
    return out


class ExactEvolution:
    """Full-state-vector real-time (or imaginary-time) evolution.

    Uses the dense Hamiltonian matrix and a cached ``expm(-1j * dt * H)`` so
    repeated ``step()`` calls are cheap after the first.

    Parameters
    ----------
    model : object with ``L``, ``d``, ``H_bonds``
    state_vector : ndarray of length ``d**L``
        Initial state, will be evolved in place by ``step()``.
    dt : float
        Time step.
    type_evo : {'real', 'imag'}
    """

    def __init__(self, model, state_vector, dt, type_evo='real'):
        self.model = model
        self.L = model.L
        self.d = model.d
        self.psi = np.asarray(state_vector, dtype=complex).copy()
        self.dt = float(dt)
        self.type_evo = type_evo
        self.evolved_time = 0.0
        self.H = bond_model_to_full_hamiltonian(model)
        coeff = -1j if type_evo == 'real' else -1.0
        self.U_dt = expm(coeff * self.dt * self.H)

    # ---- time evolution -------------------------------------------------

    def step(self):
        """Advance by one ``dt``."""
        self.psi = self.U_dt @ self.psi
        if self.type_evo == 'imag':
            self.psi /= np.linalg.norm(self.psi)
        self.evolved_time += self.dt

    # ---- constructors ---------------------------------------------------

    @classmethod
    def from_product_state(cls, model, states, dt, type_evo='real'):
        """Initial state ``|states[0] states[1] ... states[L-1]>``."""
        L, d = model.L, model.d
        assert len(states) == L
        idx = 0
        for s in states:
            idx = idx * d + s
        psi = np.zeros(d ** L, dtype=complex)
        psi[idx] = 1.0
        return cls(model, psi, dt, type_evo=type_evo)

    # ---- observables ----------------------------------------------------

    def expectation_value(self, op):
        """Array ``<op_i>`` of length ``L``."""
        from .mps import _resolve_op
        op_mat = _resolve_op(op)
        out = np.empty(self.L, dtype=complex)
        for i in range(self.L):
            O = _single_site_op_full(op_mat, i, self.L, self.d)
            out[i] = np.vdot(self.psi, O @ self.psi)
        return np.real_if_close(out)

    def correlation_function(self, op_A, op_B):
        """Matrix ``C[i, j] = <op_A_i op_B_j>`` of shape ``(L, L)``."""
        from .mps import _resolve_op
        A = _resolve_op(op_A)
        B = _resolve_op(op_B)
        C = np.zeros((self.L, self.L), dtype=complex)
        for i in range(self.L):
            for j in range(self.L):
                O = _two_site_op_full(A, B, i, j, self.L, self.d)
                C[i, j] = np.vdot(self.psi, O @ self.psi)
        return np.real_if_close(C)

    def entanglement_entropy(self):
        """Von-Neumann entropy at each of the ``L - 1`` interior bonds."""
        out = np.zeros(self.L - 1)
        for b in range(1, self.L):
            # Reshape as (d**b, d**(L-b)) and SVD: Schmidt values on bond b.
            M = self.psi.reshape(self.d ** b, self.d ** (self.L - b))
            s = np.linalg.svd(M, compute_uv=False)
            s = s[s > 1e-20]
            s2 = s * s
            out[b - 1] = -np.sum(s2 * np.log(s2))
        return out
