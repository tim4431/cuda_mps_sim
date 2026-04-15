"""Validate TEBD against exact diagonalisation on small systems.

Usage
-----

    from minimal_tebd import TFIChain, validate_tebd_vs_ed
    model = TFIChain(L=10, J=1.0, g=1.0)
    validate_tebd_vs_ed(model, dt=0.05, t_max=2.0, chi_max=64, order=4)

The function runs the *same* initial state through both TEBD and ED, measures
a small set of observables at each time step, and reports the maximum
elementwise error. Small deviations are expected from Trotter splitting and
finite bond-dimension truncation; the errors should scale as ``dt**order``
for ``chi_max`` large enough that truncation is negligible.
"""

import numpy as np

from .exact_diag import ExactEvolution
from .mps import SimpleMPS
from .tebd import TEBDEngine


def validate_tebd_vs_ed(
    model,
    *,
    dt=0.05,
    t_max=1.0,
    chi_max=64,
    svd_min=1e-12,
    order=4,
    initial_states=None,
    verbose=True,
):
    """Compare TEBD and exact-diag evolution of the same initial product state.

    Parameters
    ----------
    model : TFIChain or BondModel
        ``model.L`` must be small enough that ``d**L`` fits in memory
        (L <= ~14 for spin-1/2).
    dt : float
    t_max : float
    chi_max, svd_min : truncation parameters for TEBD.
    order : int, Trotter order (1, 2, 4).
    initial_states : list[int] or None
        Per-site basis index. Default ``[0] * L`` (``|up ... up>``).
    verbose : bool
        If True, prints per-step errors.

    Returns
    -------
    summary : dict
        ``t``: list of times.
        ``err_Sz``, ``err_Sx``, ``err_corr_XX``, ``err_entropy``:
            arrays of max-absolute-error vs time.
        ``trunc_err``: final accumulated truncation error from TEBD.
        ``fidelity_drop``: ``1 - |<psi_TEBD | psi_ED>|`` at the final time.
    """
    L = model.L
    if L > 14:
        raise ValueError(f'validation needs d**L to fit in memory; L={L} is too large')

    if initial_states is None:
        initial_states = [0] * L

    # TEBD
    psi_mps = SimpleMPS.from_product_state(L, initial_states, d=model.d)
    eng = TEBDEngine(
        psi_mps,
        model,
        {
            'dt': dt,
            'order': order,
            'N_steps': 1,
            'trunc_params': {'chi_max': chi_max, 'svd_min': svd_min},
        },
    )

    # ED
    ed = ExactEvolution.from_product_state(model, initial_states, dt=dt)

    ts = [0.0]
    err_Sz, err_Sx, err_XX, err_S = [], [], [], []

    def compare_step():
        Sz_m = eng.psi.expectation_value('Sigmaz')
        Sz_e = ed.expectation_value('Sigmaz')
        Sx_m = eng.psi.expectation_value('Sigmax')
        Sx_e = ed.expectation_value('Sigmax')
        C_m = eng.psi.correlation_function('Sigmax', 'Sigmax')
        C_e = ed.correlation_function('Sigmax', 'Sigmax')
        S_m = eng.psi.entanglement_entropy()
        S_e = ed.entanglement_entropy()
        err_Sz.append(np.max(np.abs(Sz_m - Sz_e)))
        err_Sx.append(np.max(np.abs(Sx_m - Sx_e)))
        err_XX.append(np.max(np.abs(C_m - C_e)))
        err_S.append(np.max(np.abs(S_m - S_e)))

    compare_step()
    if verbose:
        print(
            f'{"t":>6s}  {"max|dSz|":>10s}  {"max|dSx|":>10s}  '
            f'{"max|dXX|":>10s}  {"max|dS|":>10s}  {"chi":>4s}'
        )
        print(f'{0.0:>6.3f}  {err_Sz[-1]:>10.2e}  {err_Sx[-1]:>10.2e}  '
              f'{err_XX[-1]:>10.2e}  {err_S[-1]:>10.2e}  {max(eng.psi.bond_dimensions()):>4d}')

    n_steps = int(round(t_max / dt))
    for _ in range(n_steps):
        eng.run()
        ed.step()
        ts.append(eng.evolved_time)
        compare_step()
        if verbose:
            print(f'{ts[-1]:>6.3f}  {err_Sz[-1]:>10.2e}  {err_Sx[-1]:>10.2e}  '
                  f'{err_XX[-1]:>10.2e}  {err_S[-1]:>10.2e}  '
                  f'{max(eng.psi.bond_dimensions()):>4d}')

    # Final global fidelity as a single-number sanity check.
    psi_full_mps = eng.psi.to_state_vector()
    fidelity = abs(np.vdot(psi_full_mps, ed.psi))

    summary = {
        't': np.array(ts),
        'err_Sz': np.array(err_Sz),
        'err_Sx': np.array(err_Sx),
        'err_corr_XX': np.array(err_XX),
        'err_entropy': np.array(err_S),
        'trunc_err': eng.trunc_err.eps,
        'fidelity_drop': 1.0 - fidelity,
    }
    if verbose:
        print()
        print(f'final truncation error (accumulated): {eng.trunc_err.eps:.3e}')
        print(f'final |<psi_TEBD | psi_ED>|         : {fidelity:.10f}')
        print(f'final fidelity drop                  : {summary["fidelity_drop"]:.3e}')
    return summary


if __name__ == '__main__':
    # Quick smoke test when the module is run directly.
    from .model import TFIChain
    model = TFIChain(L=8, J=1.0, g=1.0)
    validate_tebd_vs_ed(model, dt=0.05, t_max=1.0, chi_max=64, order=4)
