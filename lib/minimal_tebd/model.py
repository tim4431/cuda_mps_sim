"""Models = lists of bond Hamiltonians.

A model just needs three attributes:

- ``L`` : number of sites
- ``d`` : local Hilbert-space dimension
- ``H_bonds`` : list of ``L - 1`` arrays of shape ``(d, d, d, d)``.
  ``H_bonds[b]`` acts on sites ``(b, b+1)`` with index order
  ``(i_out, j_out, i_in, j_in)``, i.e. the matrix reshape
  ``H_bonds[b].reshape(d*d, d*d)`` is a standard 4x4 operator in the basis
  ``|i, j>`` where ``i`` is the left site.

TEBD reads only these three attributes, so to build your own model, either:

1. subclass / construct ``BondModel`` directly with your list of bond
   Hamiltonians, or
2. write a class with the same three attributes.
"""

import numpy as np

from .mps import PAULI


class BondModel:
    """Generic holder for a list of bond Hamiltonians.

    Parameters
    ----------
    H_bonds : list of ndarray
        Each of shape ``(d, d, d, d)``, acting on sites ``(b, b+1)``.
    L : int, optional
        Number of sites. Defaults to ``len(H_bonds) + 1``.
    d : int, optional
        Local dimension. Defaults to ``H_bonds[0].shape[0]``.
    """

    def __init__(self, H_bonds, L=None, d=None):
        self.H_bonds = [np.asarray(h) for h in H_bonds]
        self.L = L if L is not None else len(H_bonds) + 1
        self.d = d if d is not None else self.H_bonds[0].shape[0]
        assert len(self.H_bonds) == self.L - 1
        for h in self.H_bonds:
            assert h.shape == (self.d, self.d, self.d, self.d)


class TFIChain(BondModel):
    r"""Transverse-field Ising chain matching the notebook / tenpy convention:

    .. math::
        H = -J \sum_{i=0}^{L-2} \sigma^x_i \sigma^x_{i+1}
            -g \sum_{i=0}^{L-1} \sigma^z_i

    Critical at ``J = g = 1``. Starting state ``|up ... up>`` (i.e. all sites
    in basis state ``0``, a ``+1`` eigenstate of :math:`\sigma^z`) is the
    ground state of the ``g = \infty`` limit, so evolving it with this ``H``
    realises a global quench.
    """

    def __init__(self, L, J=1.0, g=1.0):
        self.L = L
        self.d = 2
        self.J = J
        self.g = g
        X = PAULI['Sigmax'].real
        Z = PAULI['Sigmaz'].real
        Id = PAULI['Id'].real
        nbonds = L - 1
        H_list = []
        for b in range(nbonds):
            # split the on-site -g Z evenly across its two neighbouring bonds,
            # with full weight on the boundary sites that only touch one bond.
            gL = 0.5 * g if b != 0 else g
            gR = 0.5 * g if b + 1 != L - 1 else g
            H = -J * np.kron(X, X) - gL * np.kron(Z, Id) - gR * np.kron(Id, Z)
            H_list.append(H.reshape(2, 2, 2, 2))
        super().__init__(H_list, L=L, d=2)
