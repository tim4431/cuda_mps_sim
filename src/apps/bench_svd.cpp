// bench_svd: compare cuSOLVER gesvdj vs. custom Jacobi SVD per chi_max.
//
// For each chi_max, runs N_STEPS TEBD steps using each SVD path,
// compares wall time and verifies that the Schmidt spectra agree.
//
// CSV output to stdout:
//   chi_max,N_steps,t_cusolver_s,t_jacobi_s,ratio,max_err
//
// Usage:
//   ./bench_svd                       # default chi sweep
//   ./bench_svd chi_max N_steps       # single config

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

#include "gpu_tebd.h"
#include "model.h"
#include "mps.h"
#include "tebd.h"

using clk = std::chrono::high_resolution_clock;
static double sec_since(const clk::time_point& t0) {
  using D = std::chrono::duration<double>;
  return std::chrono::duration_cast<D>(clk::now() - t0).count();
}

// Run one full TEBD step using update_bond_gpu (cuSOLVER).
static void step_cusolver(TEBDEngine& eng, GpuTebdWorkspace& ws) {
  int L = eng.model->L;
  for (int s = 0; s < eng.n_schedule; ++s) {
    int start    = eng.schedule[s].parity;
    Tensor* gates = eng.gate_sets[eng.schedule[s].frac_idx];
    for (int b = start; b < L - 1; b += 2)
      update_bond_gpu(eng.psi, b, gates[b], eng.chi_max, eng.svd_min, ws);
  }
}

// Run one full TEBD step using update_bond_gpu_jacobi (custom Jacobi).
static void step_jacobi(TEBDEngine& eng, GpuTebdWorkspace& ws) {
  int L = eng.model->L;
  for (int s = 0; s < eng.n_schedule; ++s) {
    int start    = eng.schedule[s].parity;
    Tensor* gates = eng.gate_sets[eng.schedule[s].frac_idx];
    for (int b = start; b < L - 1; b += 2)
      update_bond_gpu_jacobi(eng.psi, b, gates[b], eng.chi_max, eng.svd_min, ws);
  }
}

// Maximum absolute difference between Schmidt spectra of two MPS.
static double max_spec_err(const MPS& a, const MPS& b) {
  double err = 0.0;
  int L = a.L;
  for (int i = 1; i < L; ++i) {
    int ka = a.chi[i], kb = b.chi[i];
    int k  = std::max(ka, kb);
    for (int j = 0; j < k; ++j) {
      double sa = (j < ka) ? a.Ss[i][j] : 0.0;
      double sb = (j < kb) ? b.Ss[i][j] : 0.0;
      err = std::max(err, std::abs(sa - sb));
    }
  }
  return err;
}

static void run_config(int L, int chi_max, int N_steps, bool csv) {
  const double dt      = 0.1;
  const int    order   = 2;
  const double svd_min = 1e-12;
  BondModel model = tfi_chain(L, 1.0, 1.0);
  std::vector<int> init(L, 0);

  // ----------------------------------------------------------------
  // Warm-up both paths (prime cuSOLVER and cuBLAS caches).
  // ----------------------------------------------------------------
  {
    GpuTebdWorkspace ws_wu(model.d, chi_max);
    MPS psi_wu = MPS::product_state(L, init.data(), model.d, chi_max);
    TEBDEngine eng_wu(&psi_wu, &model, dt, order, 1, chi_max, svd_min);
    step_cusolver(eng_wu, ws_wu);
  }
  {
    GpuTebdWorkspace ws_wu(model.d, chi_max);
    MPS psi_wu = MPS::product_state(L, init.data(), model.d, chi_max);
    TEBDEngine eng_wu(&psi_wu, &model, dt, order, 1, chi_max, svd_min);
    step_jacobi(eng_wu, ws_wu);
  }

  // ----------------------------------------------------------------
  // cuSOLVER timing.
  // ----------------------------------------------------------------
  GpuTebdWorkspace ws_cs(model.d, chi_max);
  MPS psi_cs = MPS::product_state(L, init.data(), model.d, chi_max);
  TEBDEngine eng_cs(&psi_cs, &model, dt, order, 1, chi_max, svd_min);

  auto t0 = clk::now();
  for (int s = 0; s < N_steps; ++s) step_cusolver(eng_cs, ws_cs);
  double t_cs = sec_since(t0);

  // ----------------------------------------------------------------
  // Custom Jacobi timing.
  // ----------------------------------------------------------------
  GpuTebdWorkspace ws_jc(model.d, chi_max);
  MPS psi_jc = MPS::product_state(L, init.data(), model.d, chi_max);
  TEBDEngine eng_jc(&psi_jc, &model, dt, order, 1, chi_max, svd_min);

  auto t1 = clk::now();
  for (int s = 0; s < N_steps; ++s) step_jacobi(eng_jc, ws_jc);
  double t_jc = sec_since(t1);

  double ratio   = t_cs / t_jc;
  double max_err = max_spec_err(psi_cs, psi_jc);

  if (csv) {
    std::printf("%d,%d,%.6f,%.6f,%.3f,%.2e\n",
                chi_max, N_steps, t_cs, t_jc, ratio, max_err);
  } else {
    std::printf("L=%d chi=%d steps=%d\n"
                "  cuSOLVER: %.4fs\n"
                "  Jacobi:   %.4fs  ratio=%.2f\n"
                "  max_spec_err=%.2e %s\n",
                L, chi_max, N_steps,
                t_cs,
                t_jc, ratio,
                max_err, max_err < 1e-10 ? "(PASS)" : "(FAIL - check kernel)");
    std::fflush(stdout);
  }
}

int main(int argc, char** argv) {
  const int L = 20;

  if (argc >= 3) {
    int chi_max = std::atoi(argv[1]);
    int N_steps = std::atoi(argv[2]);
    run_config(L, chi_max, N_steps, /*csv=*/false);
    return 0;
  }

  std::printf("chi_max,N_steps,t_cusolver_s,t_jacobi_s,ratio,max_err\n");
  for (int chi_max : {16, 32, 48, 64, 96, 128}) {
    run_config(L, chi_max, /*N_steps=*/20, /*csv=*/true);
  }
  return 0;
}
