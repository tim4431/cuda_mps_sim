// Side-by-side TEBD: CPU baseline vs GPU update_bond.
//
// We run two MPS through identical Trotter schedules, one through the host
// update_bond and one through update_bond_gpu, and compare observables at
// every step.  Tolerances are loose because the CPU SVD is a one-sided Jacobi
// and the GPU SVD is cuSOLVER's gesvdj -- the singular vectors can differ
// by an overall sign/phase, but observables and singular values must match.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <vector>

#include "gpu_tebd.h"
#include "model.h"
#include "mps.h"
#include "tebd.h"

static double max_abs_diff(const std::vector<double>& a,
                           const std::vector<double>& b) {
  double mx = 0.0;
  for (size_t i = 0; i < a.size(); ++i)
    mx = std::max(mx, std::abs(a[i] - b[i]));
  return mx;
}

int main(int argc, char** argv) {
  int L = (argc > 1) ? std::atoi(argv[1]) : 12;
  int chi_max = (argc > 2) ? std::atoi(argv[2]) : 32;
  int N_steps = (argc > 3) ? std::atoi(argv[3]) : 30;
  double dt = 0.05;
  int order = 2;
  double svd_min = 1e-12;

  BondModel model = tfi_chain(L, 1.0, 1.0);
  std::vector<int> init(L, 0);

  MPS psi_cpu = MPS::product_state(L, init.data(), model.d, chi_max);
  TEBDEngine eng_cpu(&psi_cpu, &model, dt, order, 1, chi_max, svd_min);

  MPS psi_gpu = MPS::product_state(L, init.data(), model.d, chi_max);
  TEBDEngine eng_gpu(&psi_gpu, &model, dt, order, 1, chi_max, svd_min);

  GpuTebdWorkspace ws(model.d, chi_max);

  std::vector<double> Sz_cpu(L), Sz_gpu(L);
  std::vector<double> Sx_cpu(L), Sx_gpu(L);
  std::vector<double> ent_cpu(L - 1), ent_gpu(L - 1);

  auto compare = [&](int step) {
    psi_cpu.expect(PAULI_Z, Sz_cpu.data());
    psi_gpu.expect(PAULI_Z, Sz_gpu.data());
    psi_cpu.expect(PAULI_X, Sx_cpu.data());
    psi_gpu.expect(PAULI_X, Sx_gpu.data());
    psi_cpu.entropy(ent_cpu.data());
    psi_gpu.entropy(ent_gpu.data());
    double dSz = max_abs_diff(Sz_cpu, Sz_gpu);
    double dSx = max_abs_diff(Sx_cpu, Sx_gpu);
    double dEnt = max_abs_diff(ent_cpu, ent_gpu);
    std::printf("step=%3d t=%6.3f  max|dSz|=%.2e  max|dSx|=%.2e  max|dEnt|=%.2e\n",
                step, step * dt, dSz, dSx, dEnt);
    return std::max({dSz, dSx, dEnt});
  };

  std::printf("# L=%d chi_max=%d N_steps=%d dt=%g order=%d\n", L, chi_max,
              N_steps, dt, order);
  double max_err = 0.0;
  max_err = std::max(max_err, compare(0));

  for (int s = 0; s < N_steps; ++s) {
    eng_cpu.step();
    tebd_step_gpu(eng_gpu, ws);
    max_err = std::max(max_err, compare(s + 1));
  }
  std::printf("# overall max obs error: %.3e\n", max_err);
  if (max_err > 1e-6) {
    std::fprintf(stderr, "FAIL: max error %.3e exceeds 1e-6\n", max_err);
    return 1;
  }
  std::printf("PASS\n");
  return 0;
}
