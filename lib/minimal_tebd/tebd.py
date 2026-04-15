"""Minimal TEBD engine.

Adapted from tenpy's toycode ``tenpy_toycodes/c_tebd.py``, with these changes:

- Real-time evolution by default: gates are ``exp(-1j * coeff * dt * H_bond)``.
- Supports Trotter orders 1, 2, 4 (4 = Forest--Ruth).
- The ``TEBDEngine`` class mimics tenpy's interface (``run()``,
  ``evolved_time``, ``trunc_err``) so downstream code written against tenpy
  can be swapped to this package with minimal changes.

The single hot kernel is :func:`update_bond` — this is the piece to
port to CUDA. Everything else is a thin driver.
"""

import numpy as np
from scipy.linalg import expm

# ---- Trotter schedules -----------------------------------------------------


def _trotter_substeps(order):
    """Return the list of ``(parity, dt_fraction)`` pairs for one time step.

    Each sub-step says: apply all gates of the given ``parity`` ('even' = bonds
    0, 2, 4, ... ; 'odd' = bonds 1, 3, 5, ...) for a fraction ``dt_fraction``
    of the full time step ``dt``.
    """
    if order == 1:
        return [('even', 1.0), ('odd', 1.0)]
    if order == 2:
        return [('even', 0.5), ('odd', 1.0), ('even', 0.5)]
    if order == 4:
        # Forest--Ruth 4th-order symplectic integrator, applied to the
        # splitting H = H_even + H_odd (within each group the bond Hamiltonians
        # commute, so we can exponentiate them independently).
        theta = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))
        a1 = theta / 2.0
        a2 = (1.0 - theta) / 2.0
        b1 = theta
        b2 = 1.0 - 2.0 * theta
        return [
            ('even', a1),
            ('odd', b1),
            ('even', a2),
            ('odd', b2),
            ('even', a2),
            ('odd', b1),
            ('even', a1),
        ]
    raise ValueError(f'unsupported order {order}; use 1, 2, or 4')


def _compute_gates(H_bonds, coeff):
    """Exponentiate each bond Hamiltonian: ``U_b = expm(coeff * H_b)``.

    Returned gates have shape ``(d, d, d, d)`` with index order
    ``(i_out, j_out, i_in, j_in)`` so they directly contract with ``theta2``.
    """
    gates = []
    for H in H_bonds:
        d = H.shape[0]
        H_mat = H.reshape(d * d, d * d)
        U = expm(coeff * H_mat)
        gates.append(U.reshape(d, d, d, d))
    return gates


# ---- SVD split + truncate (the inner loop of update_bond) ------------------


def split_truncate_theta(theta, chi_max, svd_min):
    """Split ``theta[vL, i, j, vR]`` by SVD, truncate, renormalise.

    Returns
    -------
    A : ndarray, shape ``(chi_L, d, chi_new)`` -- left-canonical on site i.
    S : ndarray, shape ``(chi_new,)`` -- Schmidt values on the new middle bond.
    B : ndarray, shape ``(chi_new, d, chi_R)`` -- right-canonical on site j.
    eps : float -- discarded weight (``1 - sum kept_S^2``), the bond truncation error.
    """
    chivL, dL, dR, chivR = theta.shape
    mat = theta.reshape(chivL * dL, dR * chivR)
    U, s, Vh = np.linalg.svd(mat, full_matrices=False)
    keep = min(chi_max, int(np.sum(s > svd_min)))
    if keep < 1:
        keep = 1  # never discard everything
    idx = np.argsort(s)[::-1][:keep]
    U = U[:, idx]
    s = s[idx]
    Vh = Vh[idx, :]
    norm = np.linalg.norm(s)
    eps = 1.0 - norm * norm
    S = s / norm
    A = U.reshape(chivL, dL, keep)
    B = Vh.reshape(keep, dR, chivR)
    return A, S, B, eps


# ---- the core kernel -------------------------------------------------------


def update_bond(psi, i, U_bond, chi_max, svd_min):
    """Apply a two-site gate on bond ``(i, i+1)`` in-place, truncate, return eps.

    This is the hot path: gate contraction + SVD + canonicalisation.
    Everything performance-critical happens here.
    """
    # Two-site wave function in mixed canonical form.
    theta = psi.get_theta2(i)  # vL i j vR
    # Contract with the gate on the two physical legs.
    # U_bond has legs (i', j', i, j); theta has (vL, i, j, vR).
    Utheta = np.tensordot(U_bond, theta, axes=([2, 3], [1, 2]))  # i' j' vL vR
    Utheta = np.transpose(Utheta, [2, 0, 1, 3])  # vL i' j' vR
    # SVD and truncate.
    A, S, B, eps = split_truncate_theta(Utheta, chi_max, svd_min)
    # Restore right-canonical B-form at site i:
    #   new_B[i] = diag(1/Ss[i]) @ A @ diag(S)
    # Clamp tiny Ss[i] to avoid blow-ups from stale small Schmidt values on the
    # neighbouring bond; this is the standard Vidal-form regularisation.
    Ss_i = psi.Ss[i]
    Ss_i_inv = np.where(Ss_i > 1e-14, 1.0 / Ss_i, 0.0)
    G_i = np.tensordot(np.diag(Ss_i_inv), A, axes=(1, 0))  # vL i vC
    psi.Bs[i] = np.tensordot(G_i, np.diag(S), axes=(2, 0))  # vL i vC
    psi.Ss[i + 1] = S
    psi.Bs[i + 1] = B
    return eps


# ---- accumulated truncation error -----------------------------------------


class TruncationError:
    """Running accumulator for truncation error, ``eps = 1 - prod_k (1 - eps_k)``.

    Mirrors (a small subset of) ``tenpy.algorithms.truncation.TruncationError``.
    """

    def __init__(self, eps=0.0):
        self.eps = float(eps)

    def add(self, eps_k):
        self.eps = self.eps + eps_k - self.eps * eps_k
        return self

    def __repr__(self):
        return f'TruncationError(eps={self.eps:.3e})'


# ---- TEBDEngine driver -----------------------------------------------------


class TEBDEngine:
    """TEBD time-evolution engine.

    Parameters
    ----------
    psi : SimpleMPS
        State (evolved in place).
    model : object with ``L``, ``d``, ``H_bonds`` attributes
        Typically a ``TFIChain`` or ``BondModel``.
    options : dict
        Recognised keys:

        - ``dt`` (float, default 0.1): time step.
        - ``order`` (int, default 2): Trotter order (1, 2, or 4).
        - ``N_steps`` (int, default 1): steps per ``run()`` call.
        - ``trunc_params`` (dict): ``chi_max`` (default 100) and
          ``svd_min`` (default 1e-12).
        - ``type_evo`` ('real' or 'imag', default 'real'): real-time or
          imaginary-time evolution. Imaginary time is non-unitary; use
          it to relax to a ground state.
    """

    def __init__(self, psi, model, options=None):
        options = dict(options or {})
        self.psi = psi
        self.model = model
        self.dt = float(options.get('dt', 0.1))
        self.order = int(options.get('order', 2))
        self.N_steps = int(options.get('N_steps', 1))
        trunc = options.get('trunc_params', {}) or {}
        self.chi_max = int(trunc.get('chi_max', 100))
        self.svd_min = float(trunc.get('svd_min', 1e-12))
        self.type_evo = options.get('type_evo', 'real')
        assert self.type_evo in ('real', 'imag')
        self.evolved_time = 0.0
        self.trunc_err = TruncationError()
        self._build_gate_cache()

    def _build_gate_cache(self):
        coeff_prefactor = -1j if self.type_evo == 'real' else -1.0
        self._schedule = _trotter_substeps(self.order)
        unique_fracs = sorted({frac for _, frac in self._schedule})
        self._gates = {
            frac: _compute_gates(self.model.H_bonds, coeff_prefactor * frac * self.dt)
            for frac in unique_fracs
        }

    def run(self):
        """Evolve the state by ``N_steps * dt``."""
        for _ in range(self.N_steps):
            self._one_trotter_step()
            self.evolved_time += self.dt

    def _one_trotter_step(self):
        L = self.model.L
        for parity, frac in self._schedule:
            gates = self._gates[frac]
            start = 0 if parity == 'even' else 1
            for b in range(start, L - 1, 2):
                eps = update_bond(self.psi, b, gates[b], self.chi_max, self.svd_min)
                self.trunc_err.add(eps)
