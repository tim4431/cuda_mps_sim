"""Visualization for the minimal TEBD simulation.

Runs a TFI quench and produces a multi-panel figure showing:
  1. Entanglement entropy (space-time heatmap)
  2. Magnetization <Sz> and <Sx> (space-time heatmaps)
  3. Mid-chain entanglement entropy vs time
  4. Connected correlator <XiXj> - <Xi><Xj> at final time
  5. Bond dimension and truncation error vs time

Usage:  conda run -n claude python -m minimal_tebd.visualize
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from .mps import SimpleMPS
from .model import TFIChain
from .tebd import TEBDEngine


def collect_data(L=30, dt=0.1, order=4, chi_max=100, svd_min=1e-12, t_max=5.0):
    model = TFIChain(L=L, J=1.0, g=1.0)
    psi = SimpleMPS.from_product_state(L, [0] * L, d=model.d)
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
    return data, L


def plot(data, L):
    t = data['t']
    entropy = data['entropy']       # (n_t, L-1)
    Sz = data['Sz']                 # (n_t, L)
    Sx = data['Sx']                 # (n_t, L)
    corr_XX = data['corr_XX']       # (n_t, L, L)
    trunc_err = data['trunc_err']   # (n_t,)
    chi = data['chi_max']           # (n_t,)

    fig = plt.figure(figsize=(16, 12), constrained_layout=True)
    fig.suptitle(
        f'TFI quench  (L={L}, J=g=1, 4th-order TEBD, $\\chi_{{max}}$={int(chi.max())})',
        fontsize=14,
    )
    gs = GridSpec(3, 3, figure=fig)

    # --- 1. Entanglement entropy heatmap ---
    ax1 = fig.add_subplot(gs[0, 0])
    bonds = np.arange(entropy.shape[1])
    im1 = ax1.pcolormesh(bonds, t, entropy, shading='nearest', cmap='inferno')
    ax1.set_xlabel('Bond index')
    ax1.set_ylabel('Time $t$')
    ax1.set_title('Entanglement entropy $S$')
    fig.colorbar(im1, ax=ax1, pad=0.02)

    # --- 2. <Sz> heatmap ---
    ax2 = fig.add_subplot(gs[0, 1])
    sites = np.arange(L)
    im2 = ax2.pcolormesh(sites, t, Sz, shading='nearest', cmap='RdBu', vmin=-1, vmax=1)
    ax2.set_xlabel('Site $i$')
    ax2.set_ylabel('Time $t$')
    ax2.set_title(r'$\langle \sigma^z_i \rangle$')
    fig.colorbar(im2, ax=ax2, pad=0.02)

    # --- 3. <Sx> heatmap ---
    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.pcolormesh(sites, t, Sx, shading='nearest', cmap='RdBu', vmin=-1, vmax=1)
    ax3.set_xlabel('Site $i$')
    ax3.set_ylabel('Time $t$')
    ax3.set_title(r'$\langle \sigma^x_i \rangle$')
    fig.colorbar(im3, ax=ax3, pad=0.02)

    # --- 4. Mid-chain entropy vs time ---
    ax4 = fig.add_subplot(gs[1, 0])
    mid = L // 2 - 1
    ax4.plot(t, entropy[:, mid], 'k-', lw=1.5)
    ax4.set_xlabel('Time $t$')
    ax4.set_ylabel('$S_{L/2}$')
    ax4.set_title('Mid-chain entanglement entropy')

    # --- 5. Entropy profile at selected times ---
    ax5 = fig.add_subplot(gs[1, 1])
    n_t = len(t)
    indices = [0, n_t // 4, n_t // 2, 3 * n_t // 4, n_t - 1]
    for idx in indices:
        ax5.plot(bonds, entropy[idx], label=f't={t[idx]:.1f}')
    ax5.set_xlabel('Bond index')
    ax5.set_ylabel('$S$')
    ax5.set_title('Entropy profile snapshots')
    ax5.legend(fontsize=8)

    # --- 6. <Sz> profile at selected times ---
    ax6 = fig.add_subplot(gs[1, 2])
    for idx in indices:
        ax6.plot(sites, Sz[idx], label=f't={t[idx]:.1f}')
    ax6.set_xlabel('Site $i$')
    ax6.set_ylabel(r'$\langle \sigma^z_i \rangle$')
    ax6.set_title(r'$\langle \sigma^z \rangle$ profile snapshots')
    ax6.legend(fontsize=8)

    # --- 7. Connected XX correlator at final time ---
    ax7 = fig.add_subplot(gs[2, 0])
    Sx_final = Sx[-1]
    connected = corr_XX[-1] - np.outer(Sx_final, Sx_final)
    vmax = np.max(np.abs(connected))
    im7 = ax7.imshow(
        connected, origin='lower', cmap='RdBu_r', vmin=-vmax, vmax=vmax,
        extent=[0, L, 0, L],
    )
    ax7.set_xlabel('Site $j$')
    ax7.set_ylabel('Site $i$')
    ax7.set_title(r'Connected $\langle X_i X_j \rangle_c$ (final)')
    fig.colorbar(im7, ax=ax7, pad=0.02)

    # --- 8. Bond dimension vs time ---
    ax8 = fig.add_subplot(gs[2, 1])
    ax8.plot(t, chi, 'b-o', ms=3, lw=1.2)
    ax8.set_xlabel('Time $t$')
    ax8.set_ylabel('$\\chi_{\\mathrm{max}}$')
    ax8.set_title('Max bond dimension')

    # --- 9. Truncation error vs time ---
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.semilogy(t[1:], trunc_err[1:], 'r-o', ms=3, lw=1.2)
    ax9.set_xlabel('Time $t$')
    ax9.set_ylabel('Accumulated $\\epsilon$')
    ax9.set_title('Truncation error')

    return fig


def main():
    print('Running TEBD simulation...')
    data, L = collect_data()
    print('\nGenerating plots...')
    fig = plot(data, L)
    out = 'tebd_visualization.png'
    fig.savefig(out, dpi=150)
    print(f'Saved to {out}')
    plt.show()


if __name__ == '__main__':
    main()
