// Benchmark CPU vs GPU TEBD step on a TFI quench.
//
// CSV output to stdout:
//   L,chi_max,N_steps,t_cpu_s,t_gpu_s,speedup,max_chi
//
// Usage:
//   ./bench_gpu                       # default sweep
//   ./bench_gpu L chi_max N_steps     # single config

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iostream>
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

static void run_config(int L, int chi_max, int N_steps, double dt, int order,
                       bool csv) {
  double svd_min = 1e-12;
  BondModel model = tfi_chain(L, 1.0, 1.0);
  std::vector<int> init(L, 0);

  // ---- CPU ----
  MPS psi_cpu = MPS::product_state(L, init.data(), model.d, chi_max);
  TEBDEngine eng_cpu(&psi_cpu, &model, dt, order, 1, chi_max, svd_min);

  auto t0 = clk::now();
  for (int s = 0; s < N_steps; ++s) eng_cpu.step();
  double t_cpu = sec_since(t0);
  int max_chi_cpu = psi_cpu.max_bond_dim();

  // ---- GPU ----
  MPS psi_gpu = MPS::product_state(L, init.data(), model.d, chi_max);
  TEBDEngine eng_gpu(&psi_gpu, &model, dt, order, 1, chi_max, svd_min);
  GpuTebdWorkspace ws(model.d, chi_max);

  // Warm up (first GEMM/SVD allocates extra memory).
  tebd_step_gpu(eng_gpu, ws);
  // Reset for clean timing.
  MPS psi_gpu2 = MPS::product_state(L, init.data(), model.d, chi_max);
  TEBDEngine eng_gpu2(&psi_gpu2, &model, dt, order, 1, chi_max, svd_min);

  auto t1 = clk::now();
  for (int s = 0; s < N_steps; ++s) tebd_step_gpu(eng_gpu2, ws);
  double t_gpu = sec_since(t1);
  int max_chi_gpu = psi_gpu2.max_bond_dim();

  double speedup = t_cpu / t_gpu;
  int max_chi = std::max(max_chi_cpu, max_chi_gpu);
  if (csv) {
    std::printf("%d,%d,%d,%.6f,%.6f,%.3f,%d\n", L, chi_max, N_steps, t_cpu,
                t_gpu, speedup, max_chi);
  } else {
    std::printf("L=%d chi_max=%d N_steps=%d  t_cpu=%.3fs t_gpu=%.3fs "
                "speedup=%.2fx max_chi=%d\n",
                L, chi_max, N_steps, t_cpu, t_gpu, speedup, max_chi);
  }
  std::fflush(stdout);
}

int main(int argc, char** argv) {
  double dt = 0.1;
  int order = 2;

  if (argc >= 4) {
    int L = std::atoi(argv[1]);
    int chi_max = std::atoi(argv[2]);
    int N_steps = std::atoi(argv[3]);
    run_config(L, chi_max, N_steps, dt, order, /*csv=*/false);
    return 0;
  }

  std::printf("L,chi_max,N_steps,t_cpu_s,t_gpu_s,speedup,max_chi\n");

  // Sweep over chi_max at fixed L (entanglement-driven cost).
  for (int chi_max : {16, 32, 48, 64, 96, 128}) {
    run_config(/*L=*/20, chi_max, /*N_steps=*/20, dt, order, /*csv=*/true);
  }
  // Sweep over L at fixed chi_max.
  for (int L : {12, 20, 28, 40, 60, 80}) {
    run_config(L, /*chi_max=*/64, /*N_steps=*/15, dt, order, /*csv=*/true);
  }
  return 0;
}
