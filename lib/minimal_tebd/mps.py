"""Minimal matrix product state for finite systems.

Adapted from tenpy's toycode ``tenpy_toycodes/a_mps.py``.

Conventions
-----------
We store the MPS in the *right-canonical* B-form:

    |psi> = sum_{j_0,...,j_{L-1}}  B[0] B[1] ... B[L-1]  |j_0 ... j_{L-1}>

- ``Bs[i]`` has shape ``(chi_L, d, chi_R)`` with legs ``vL, i, vR``.
  It is right-canonical:  sum_{j, vR}  B[vL, j, vR] B*[vL', j, vR] = delta_{vL, vL'}.
- ``Ss[i]`` is the vector of Schmidt values on the bond *left* of site ``i``.
  ``Ss[0]`` and (implicitly) the bond right of the last site are both ``[1.0]``.

In this form the tensor ``theta[i] = diag(Ss[i]) @ Bs[i]`` gives the two-legged
Schmidt-weighted wave function on site ``i``, and two-site expectation values
and the TEBD update all work cleanly without awkward inverse-Schmidt factors
on the right boundary of the affected bond.
"""

import numpy as np

PAULI = {
    'Id': np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex),
    'Sigmax': np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
    'Sigmay': np.array([[0.0, -1j], [1j, 0.0]], dtype=complex),
    'Sigmaz': np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex),
}


def _resolve_op(op):
    """Accept either a ``(d, d)`` matrix or one of the strings in ``PAULI``."""
    if isinstance(op, str):
        return PAULI[op]
    return np.asarray(op)


class SimpleMPS:
    """Right-canonical MPS for a finite chain.

    Attributes
    ----------
    Bs : list of ndarray
        ``Bs[i]`` has legs ``(vL, i, vR)``, right-canonical.
    Ss : list of ndarray
        ``Ss[i]`` is the Schmidt spectrum on the bond left of site ``i``.
        ``len(Ss) == L``; the right-most bond is implicitly trivial.
    L : int
        Number of sites.
    """

    def __init__(self, Bs, Ss):
        self.Bs = list(Bs)
        self.Ss = list(Ss)
        self.L = len(Bs)
        assert len(Ss) == self.L, 'expect one Schmidt vector per left-bond'

    @property
    def d(self):
        return self.Bs[0].shape[1]

    def copy(self):
        return SimpleMPS([B.copy() for B in self.Bs], [S.copy() for S in self.Ss])

    # ---- constructors ----------------------------------------------------

    @classmethod
    def from_product_state(cls, L, states, d=2):
        """Product state ``|states[0], states[1], ..., states[L-1]>``.

        Parameters
        ----------
        L : int
            Number of sites.
        states : sequence of int, length ``L``
            Local basis index at each site (``0 <= states[i] < d``).
        d : int
            Local Hilbert-space dimension.
        """
        assert len(states) == L
        Bs = []
        for j in states:
            B = np.zeros((1, d, 1), dtype=complex)
            B[0, j, 0] = 1.0
            Bs.append(B)
        Ss = [np.ones(1) for _ in range(L)]
        return cls(Bs, Ss)

    # ---- tensor accessors -----------------------------------------------

    def get_theta1(self, i):
        """Single-site mixed-canonical tensor ``theta[vL, i, vR] = diag(Ss[i]) @ Bs[i]``."""
        return np.tensordot(np.diag(self.Ss[i]), self.Bs[i], [1, 0])

    def get_theta2(self, i):
        """Two-site mixed-canonical tensor on sites ``i, i+1`` with legs ``(vL, i, j, vR)``."""
        return np.tensordot(self.get_theta1(i), self.Bs[i + 1], [2, 0])

    def bond_dimensions(self):
        """List of (non-trivial) bond dimensions, length ``L - 1``."""
        return [self.Bs[i].shape[2] for i in range(self.L - 1)]

    # ---- observables -----------------------------------------------------

    def entanglement_entropy(self):
        """Von-Neumann entropy at each of the ``L - 1`` interior bonds."""
        out = np.zeros(self.L - 1)
        for b in range(1, self.L):
            s = self.Ss[b]
            s = s[s > 1e-20]
            s2 = s * s
            out[b - 1] = -np.sum(s2 * np.log(s2))
        return out

    def expectation_value(self, op):
        """Array of local expectation values ``<psi| op_i |psi>`` for ``i = 0 ... L-1``."""
        op_mat = _resolve_op(op)
        out = np.empty(self.L, dtype=complex)
        for i in range(self.L):
            theta = self.get_theta1(i)  # vL i vR
            op_theta = np.tensordot(op_mat, theta, axes=(1, 1))  # i' vL vR
            out[i] = np.tensordot(theta.conj(), op_theta, axes=([0, 1, 2], [1, 0, 2]))
        return np.real_if_close(out)

    def correlation_function(self, op_A, op_B):
        """Matrix ``C[i, j] = <psi| op_A_i op_B_j |psi>`` of shape ``(L, L)``.

        Simple ``O(L^3 * chi^3)`` reference implementation — fine for validation
        but not optimized for production use.
        """
        A = _resolve_op(op_A)
        B = _resolve_op(op_B)
        L = self.L
        C = np.zeros((L, L), dtype=complex)
        for i in range(L):
            C[i, i] = np.tensordot(
                self.get_theta1(i).conj(),
                np.tensordot(A @ B, self.get_theta1(i), axes=(1, 1)),
                axes=([0, 1, 2], [1, 0, 2]),
            )
            # i < j: propagate the partial contraction to the right
            theta_i = self.get_theta1(i)
            left = np.tensordot(A, theta_i, axes=(1, 1))  # i' vL vR
            left = np.tensordot(theta_i.conj(), left, axes=([0, 1], [1, 0]))  # vR* vR
            for j in range(i + 1, L):
                Bj = self.Bs[j]
                tmp = np.tensordot(left, Bj, axes=(1, 0))  # vR* i vR
                Oj_Bj = np.tensordot(B, Bj, axes=(1, 1)).transpose(1, 0, 2)
                C[i, j] = np.tensordot(
                    Bj.conj(), np.tensordot(left, Oj_Bj, axes=(1, 0)),
                    axes=([0, 1, 2], [0, 1, 2]),
                )
                # advance `left` past site j (identity operator there)
                left = np.tensordot(Bj.conj(), tmp, axes=([0, 1], [0, 1]))  # vR* vR
            # by symmetry C[j, i] = conj(C[i, j]) only when [A, B] = 0 and operators
            # are Hermitian; for generality recompute with roles swapped:
        for j in range(L):
            theta_j = self.get_theta1(j)
            right = np.tensordot(B, theta_j, axes=(1, 1))
            right = np.tensordot(theta_j.conj(), right, axes=([0, 1], [1, 0]))
            for i in range(j + 1, L):
                Bi = self.Bs[i]
                Oi_Bi = np.tensordot(A, Bi, axes=(1, 1)).transpose(1, 0, 2)
                C[i, j] = np.tensordot(
                    Bi.conj(), np.tensordot(right, Oi_Bi, axes=(1, 0)),
                    axes=([0, 1, 2], [0, 1, 2]),
                )
                tmp = np.tensordot(right, Bi, axes=(1, 0))
                right = np.tensordot(Bi.conj(), tmp, axes=([0, 1], [0, 1]))
        return np.real_if_close(C)

    def norm_squared(self):
        """<psi|psi>. Should stay ~1 up to truncation."""
        # Contract  ...  B*[0] B[0] B*[1] B[1] ...
        E = np.tensordot(self.Bs[0].conj(), self.Bs[0], axes=([0, 1], [0, 1]))  # vR* vR
        for i in range(1, self.L):
            B = self.Bs[i]
            E = np.tensordot(E, B, axes=(1, 0))  # vR* i vR
            E = np.tensordot(B.conj(), E, axes=([0, 1], [0, 1]))  # vR* vR
        return np.real_if_close(np.trace(E))

    # ---- converters ------------------------------------------------------

    def to_state_vector(self):
        """Full state vector of length ``d ** L``. Only feasible for small ``L``."""
        # Build  diag(Ss[0]) B[0] B[1] ... B[L-1]  and flatten physical legs.
        psi = np.tensordot(np.diag(self.Ss[0]), self.Bs[0], [1, 0])  # vL i vR
        for i in range(1, self.L):
            psi = np.tensordot(psi, self.Bs[i], axes=(-1, 0))  # vL i_0 ... i_k vR
        # psi has shape (1, d, d, ..., d, 1)
        return psi.reshape(self.d ** self.L)
