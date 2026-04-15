"""Standalone reproduction of ``notebooks/00_tebd.ipynb``.

Run:  python -m minimal_tebd.example_notebook
or:   python minimal_tebd/example_notebook.py
"""

import numpy as np

from .mps import SimpleMPS
from .model import TFIChain
from .tebd import TEBDEngine


def run(L=30, dt=0.1, order=4, chi_max=100, svd_min=1e-12, t_max=5.0):
    model = TFIChain(L=L, J=1.0, g=1.0)
    psi = SimpleMPS.from_product_state(L, [0] * L, d=model.d)  # |up ... up>
    eng = TEBDEngine(
        psi,
        model,
        {
            'dt': dt,
            'order': order,
            'N_steps': 1,
            'trunc_params': {'chi_max': chi_max, 'svd_min': svd_min},
        },
    )

    keys = ['t', 'entropy', 'Sx', 'Sz', 'corr_XX', 'trunc_err', 'chi_max']
    data = {k: [] for k in keys}

    def measure():
        data['t'].append(eng.evolved_time)
        data['entropy'].append(eng.psi.entanglement_entropy())
        data['Sx'].append(eng.psi.expectation_value('Sigmax'))
        data['Sz'].append(eng.psi.expectation_value('Sigmaz'))
        data['corr_XX'].append(eng.psi.correlation_function('Sigmax', 'Sigmax'))
        data['trunc_err'].append(eng.trunc_err.eps)
        data['chi_max'].append(max(eng.psi.bond_dimensions()))

    measure()
    while eng.evolved_time < t_max - 1e-9:
        eng.run()
        measure()
        print(
            f't={eng.evolved_time:5.2f}  '
            f'max chi={data["chi_max"][-1]:3d}  '
            f'mid-bond S={data["entropy"][-1][L // 2 - 1]:.4f}  '
            f'trunc_err={data["trunc_err"][-1]:.2e}'
        )

    for k in keys:
        data[k] = np.array(data[k])
    return data


if __name__ == '__main__':
    run()
