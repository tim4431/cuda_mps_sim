// Side-by-side TEBD: CPU baseline vs GPU PERSISTENT (GpuMps) path.
// Closes the gap that validate_gpu only checks the M3 (tebd_step_gpu) path.
// Drives the GPU with tebd_step_gpu_persistent and syncs the device MPS back
// to host each step to compare observables against the CPU engine.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
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
  int L         = (argc > 1) ? std::atoi(argv[1]) : 12;
  int chi_max   = (argc > 2) ? std::atoi(argv[2]) : 32;
  int N_steps   = (argc > 3) ? std::atoi(argv[3]) : 30;
  int n_streams = (argc > 4) ? std::atoi(argv[4]) : 1;
  double dt = 0.05;
  int order = 2;
  double svd_min = 1e-12;

  BondModel model = tfi_chain(L, 1.0, 1.0);
  std::vector<int> init(L, 0);

  MPS psi_cpu = MPS::product_state(L, init.data(), model.d, chi_max);
  TEBDEngine eng_cpu(&psi_cpu, &model, dt, order, 1, chi_max, svd_min);

  MPS psi_gpu = MPS::product_state(L, init.data(), model.d, chi_max);
  TEBDEngine eng_gpu(&psi_gpu, &model, dt, order, 1, chi_max, svd_min);

  // Streams + persistent workspaces (mirror apps/sweep_mpi.cpp).
  std::vector<void*> streams(n_streams, nullptr);
  for (int i = 1; i < n_streams; ++i) streams[i] = gpu_stream_create();
  std::vector<std::unique_ptr<GpuTebdWorkspace>> ws_ptrs;
  ws_ptrs.reserve(n_streams);
  for (int i = 0; i < n_streams; ++i)
    ws_ptrs.emplace_back(new GpuTebdWorkspace(model.d, chi_max, streams[i]));
  std::vector<GpuTebdWorkspace*> ws_raw(n_streams);
  for (int i = 0; i < n_streams; ++i) ws_raw[i] = ws_ptrs[i].get();

  GpuMps gmps(psi_gpu);

  std::vector<double> Sz_c(L), Sz_g(L), Sx_c(L), Sx_g(L), e_c(L - 1), e_g(L - 1);
  auto compare = [&](int step) {
    psi_cpu.expect(PAULI_Z, Sz_c.data());
    psi_gpu.expect(PAULI_Z, Sz_g.data());
    psi_cpu.expect(PAULI_X, Sx_c.data());
    psi_gpu.expect(PAULI_X, Sx_g.data());
    psi_cpu.entropy(e_c.data());
    psi_gpu.entropy(e_g.data());
    double dSz = max_abs_diff(Sz_c, Sz_g);
    double dSx = max_abs_diff(Sx_c, Sx_g);
    double dE  = max_abs_diff(e_c, e_g);
    std::printf("step=%3d t=%6.3f  max|dSz|=%.2e  max|dSx|=%.2e  max|dEnt|=%.2e\n",
                step, step * dt, dSz, dSx, dE);
    return std::max({dSz, dSx, dE});
  };

  std::printf("# persistent path  L=%d chi_max=%d N_steps=%d dt=%g order=%d "
              "n_streams=%d\n",
              L, chi_max, N_steps, dt, order, n_streams);
  double max_err = compare(0);
  for (int s = 0; s < N_steps; ++s) {
    eng_cpu.step();
    tebd_step_gpu_persistent(eng_gpu, gmps, ws_raw.data(), n_streams);
    gmps.sync_to_host(psi_gpu);  // bring device MPS back for observables
    max_err = std::max(max_err, compare(s + 1));
  }
  std::printf("# overall max obs error: %.3e\n", max_err);

  for (int i = 1; i < n_streams; ++i) gpu_stream_destroy(streams[i]);

  if (max_err > 1e-6) {
    std::fprintf(stderr, "FAIL: max error %.3e exceeds 1e-6\n", max_err);
    return 1;
  }
  std::printf("PASS\n");
  return 0;
}
