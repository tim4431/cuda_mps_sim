// Benchmark CPU vs GPU TEBD step on a TFI quench.
//
// CSV output to stdout:
//   L,chi_max,N_steps,t_cpu_s,t_gpu_s,speedup,t_persistent_s,t_streamed_s,max_chi
//
// t_persistent_s  = GPU with persistent device MPS (no per-bond H2D/D2H)
// t_streamed_s    = GPU with persistent MPS + N_STREAMS CUDA streams
//
// Usage:
//   ./bench_gpu                       # default sweep
//   ./bench_gpu L chi_max N_steps     # single config

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <memory>
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

// Number of CUDA streams for the streamed persistent benchmark.
static const int N_STREAMS = 4;

static void run_config(int L, int chi_max, int N_steps, double dt, int order,
                       bool csv) {
  double svd_min = 1e-12;
  BondModel model = tfi_chain(L, 1.0, 1.0);
  std::vector<int> init(L, 0);

  // ----------------------------------------------------------------
  // CPU baseline
  // ----------------------------------------------------------------
  MPS psi_cpu = MPS::product_state(L, init.data(), model.d, chi_max);
  TEBDEngine eng_cpu(&psi_cpu, &model, dt, order, 1, chi_max, svd_min);

  auto t0 = clk::now();
  for (int s = 0; s < N_steps; ++s) eng_cpu.step();
  double t_cpu = sec_since(t0);
  int max_chi_cpu = psi_cpu.max_bond_dim();

  // ----------------------------------------------------------------
  // M3 GPU (non-persistent: H2D/D2H every bond)
  // ----------------------------------------------------------------
  GpuTebdWorkspace ws_m3(model.d, chi_max);

  // Warm-up.
  {
    MPS psi_wu = MPS::product_state(L, init.data(), model.d, chi_max);
    TEBDEngine eng_wu(&psi_wu, &model, dt, order, 1, chi_max, svd_min);
    tebd_step_gpu(eng_wu, ws_m3);
  }

  MPS psi_m3 = MPS::product_state(L, init.data(), model.d, chi_max);
  TEBDEngine eng_m3(&psi_m3, &model, dt, order, 1, chi_max, svd_min);

  auto t1 = clk::now();
  for (int s = 0; s < N_steps; ++s) tebd_step_gpu(eng_m3, ws_m3);
  double t_gpu = sec_since(t1);
  int max_chi_gpu = psi_m3.max_bond_dim();

  // ----------------------------------------------------------------
  // M4 GPU persistent (single workspace, null stream)
  // ----------------------------------------------------------------
  GpuTebdWorkspace ws_pers(model.d, chi_max);

  // Warm-up.
  {
    MPS psi_wu = MPS::product_state(L, init.data(), model.d, chi_max);
    TEBDEngine eng_wu(&psi_wu, &model, dt, order, 1, chi_max, svd_min);
    GpuMps gmps_wu(psi_wu);
    tebd_step_gpu_persistent(eng_wu, gmps_wu, ws_pers);
  }

  MPS psi_pers = MPS::product_state(L, init.data(), model.d, chi_max);
  TEBDEngine eng_pers(&psi_pers, &model, dt, order, 1, chi_max, svd_min);
  GpuMps gmps_pers(psi_pers);

  auto t2 = clk::now();
  for (int s = 0; s < N_steps; ++s)
    tebd_step_gpu_persistent(eng_pers, gmps_pers, ws_pers);
  double t_persistent = sec_since(t2);

  // ----------------------------------------------------------------
  // M4 GPU streamed (N_STREAMS workspaces, each with its own stream)
  // ----------------------------------------------------------------
  // Use the gpu_stream_create/destroy helpers so this file needs no CUDA headers.
  std::vector<void*> streams(N_STREAMS, nullptr);
  for (int i = 1; i < N_STREAMS; ++i) streams[i] = gpu_stream_create();

  std::vector<std::unique_ptr<GpuTebdWorkspace>> ws_ptrs;
  ws_ptrs.reserve(N_STREAMS);
  for (int i = 0; i < N_STREAMS; ++i)
    ws_ptrs.emplace_back(new GpuTebdWorkspace(model.d, chi_max, streams[i]));

  std::vector<GpuTebdWorkspace*> ws_raw(N_STREAMS);
  for (int i = 0; i < N_STREAMS; ++i) ws_raw[i] = ws_ptrs[i].get();

  // Warm-up.
  {
    MPS psi_wu = MPS::product_state(L, init.data(), model.d, chi_max);
    TEBDEngine eng_wu(&psi_wu, &model, dt, order, 1, chi_max, svd_min);
    GpuMps gmps_wu(psi_wu);
    tebd_step_gpu_persistent(eng_wu, gmps_wu, ws_raw.data(), N_STREAMS);
  }

  MPS psi_str = MPS::product_state(L, init.data(), model.d, chi_max);
  TEBDEngine eng_str(&psi_str, &model, dt, order, 1, chi_max, svd_min);
  GpuMps gmps_str(psi_str);

  auto t3 = clk::now();
  for (int s = 0; s < N_steps; ++s)
    tebd_step_gpu_persistent(eng_str, gmps_str, ws_raw.data(), N_STREAMS);
  double t_streamed = sec_since(t3);

  for (int i = 1; i < N_STREAMS; ++i) gpu_stream_destroy(streams[i]);

  // ----------------------------------------------------------------
  // Report
  // ----------------------------------------------------------------
  double speedup = t_cpu / t_gpu;
  int max_chi = std::max(max_chi_cpu, max_chi_gpu);

  if (csv) {
    std::printf("%d,%d,%d,%.6f,%.6f,%.3f,%.6f,%.6f,%d\n",
                L, chi_max, N_steps, t_cpu, t_gpu, speedup,
                t_persistent, t_streamed, max_chi);
  } else {
    std::printf("L=%d chi=%d steps=%d\n"
                "  CPU:        %.3fs\n"
                "  GPU (M3):   %.3fs  speedup=%.2fx\n"
                "  persistent: %.3fs  speedup_vs_m3=%.2fx\n"
                "  streamed(%d):%.3fs  speedup_vs_m3=%.2fx\n"
                "  max_chi=%d\n",
                L, chi_max, N_steps,
                t_cpu,
                t_gpu, speedup,
                t_persistent, t_gpu / t_persistent,
                N_STREAMS, t_streamed, t_gpu / t_streamed,
                max_chi);
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

  std::printf("L,chi_max,N_steps,t_cpu_s,t_gpu_s,speedup,"
              "t_persistent_s,t_streamed_s,max_chi\n");

  // Sweep over chi_max at fixed L.
  for (int chi_max : {16, 32, 48, 64, 96, 128}) {
    run_config(/*L=*/20, chi_max, /*N_steps=*/20, dt, order, /*csv=*/true);
  }
  // Sweep over L at fixed chi_max.
  for (int L : {12, 20, 28, 40, 60, 80}) {
    run_config(L, /*chi_max=*/64, /*N_steps=*/15, dt, order, /*csv=*/true);
  }
  return 0;
}
