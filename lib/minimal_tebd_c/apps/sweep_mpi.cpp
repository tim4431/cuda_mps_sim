// MPI parameter sweep for TEBD phase diagram computation.
//
// Each MPI rank is assigned one or more (J, g) parameter pairs from a grid
// and runs an independent GPU TEBD evolution.  Zero in-loop MPI communication;
// only MPI_Scatter (params) and MPI_Gather (results) at the boundaries.
//
// CSV output to stdout (rank 0 only):
//   rank,J,g,L,chi_max,N_steps,t_wall_s,max_chi,final_entropy
//
// Usage:
//   mpirun -n <P> ./sweep_mpi [options]
//
// Options:
//   --L <int>        chain length          (default: 20)
//   --chi <int>      max bond dimension    (default: 64)
//   --dt <float>     Trotter time step     (default: 0.1)
//   --steps <int>    number of TEBD steps  (default: 20)
//   --order <int>    Trotter order (2 or 4)(default: 2)
//   --strong         16-pair fixed grid, vary ranks (default)
//   --weak           4 pairs/rank, scale problem with ranks
//   --n-streams <int> streams per rank     (default: 1)

#include <mpi.h>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#ifdef USE_GPU
#  include "gpu_tebd.h"
#endif

#include "model.h"
#include "mps.h"
#include "tebd.h"

using clk = std::chrono::high_resolution_clock;
static double elapsed_sec(clk::time_point t0) {
  return std::chrono::duration<double>(clk::now() - t0).count();
}

// ---------------------------------------------------------------------------
// Parameter record: one (J, g) pair.
// ---------------------------------------------------------------------------
struct Param { double J, g; };

// ---------------------------------------------------------------------------
// Build the 4x4 phase-boundary grid (strong scaling: 16 total pairs).
// ---------------------------------------------------------------------------
static std::vector<Param> make_grid_strong() {
  std::vector<Param> grid;
  const double Jvals[] = {0.5, 1.0, 1.5, 2.0};
  const double gvals[] = {0.5, 1.0, 1.5, 2.0};
  for (double J : Jvals)
    for (double g : gvals)
      grid.push_back({J, g});
  return grid;
}

// ---------------------------------------------------------------------------
// Build per-rank grid for weak scaling: 4 pairs/rank.
// The pairs tile the (J, g) space with offset based on rank.
// ---------------------------------------------------------------------------
static std::vector<Param> make_grid_weak(int rank, int /*n_ranks*/) {
  // Each rank gets 4 pairs centred around a unique point in (J, g) space.
  // We tile a 0.5-step grid starting at (0.25 + rank*0.5, ...) to avoid
  // all ranks coinciding.
  std::vector<Param> grid;
  const double J0 = 0.25 + rank * 0.5;
  const double gvals[] = {0.5, 1.0, 1.5, 2.0};
  for (int gi = 0; gi < 4; ++gi)
    grid.push_back({J0, gvals[gi]});
  return grid;
}

// ---------------------------------------------------------------------------
// Run one (J, g) combination on this rank's GPU, return timing + observables.
// ---------------------------------------------------------------------------
struct RunResult {
  double J, g;
  double t_wall_s;
  int    max_chi;
  double final_entropy;  // mid-chain entropy at the end
};

static RunResult run_param(double J, double g,
                           int L, int chi_max, double dt,
                           int N_steps, int order, int n_streams) {
  double svd_min = 1e-12;
  std::vector<int> init(L, 0);  // all spin-down product state
  BondModel model = tfi_chain(L, J, g);

  RunResult res;
  res.J = J; res.g = g;

  auto t0 = clk::now();

#ifdef USE_GPU
  // ---- GPU path via persistent device MPS ----

  // Create CUDA streams using the helper so this file needs no cuda_runtime.h.
  // stream[0] = default (null), stream[i>0] = new non-null stream.
  std::vector<void*> streams(n_streams, nullptr);
  for (int i = 1; i < n_streams; ++i)
    streams[i] = gpu_stream_create();

  // GpuTebdWorkspace is non-copyable/non-movable, so store via unique_ptr.
  std::vector<std::unique_ptr<GpuTebdWorkspace>> ws_ptrs;
  ws_ptrs.reserve(n_streams);
  for (int i = 0; i < n_streams; ++i)
    ws_ptrs.emplace_back(new GpuTebdWorkspace(model.d, chi_max, streams[i]));

  // Build raw pointer array for tebd_step_gpu_persistent.
  std::vector<GpuTebdWorkspace*> ws_raw(n_streams);
  for (int i = 0; i < n_streams; ++i) ws_raw[i] = ws_ptrs[i].get();

  // Warm-up run (untimed) to let cuSOLVER pre-allocate.
  {
    MPS psi_wu = MPS::product_state(L, init.data(), model.d, chi_max);
    TEBDEngine eng_wu(&psi_wu, &model, dt, order, 1, chi_max, svd_min);
    GpuMps gmps_wu(psi_wu);
    tebd_step_gpu_persistent(eng_wu, gmps_wu, ws_raw.data(), n_streams);
  }

  // Timed run.
  MPS psi = MPS::product_state(L, init.data(), model.d, chi_max);
  TEBDEngine eng(&psi, &model, dt, order, 1, chi_max, svd_min);
  GpuMps gmps(psi);

  t0 = clk::now();
  for (int s = 0; s < N_steps; ++s)
    tebd_step_gpu_persistent(eng, gmps, ws_raw.data(), n_streams);

  // Sync result back to host for observable computation.
  gmps.sync_to_host(psi);

  for (int i = 1; i < n_streams; ++i) gpu_stream_destroy(streams[i]);

#else
  // ---- CPU fallback (no CUDA) ----
  MPS psi = MPS::product_state(L, init.data(), model.d, chi_max);
  TEBDEngine eng(&psi, &model, dt, order, N_steps, chi_max, svd_min);
  eng.run();
  (void)n_streams;
#endif

  res.t_wall_s = elapsed_sec(t0);
  res.max_chi  = psi.max_bond_dim();

  // Mid-chain entanglement entropy.
  std::vector<double> ent(L - 1);
  psi.entropy(ent.data());
  res.final_entropy = ent[(L - 1) / 2];

  return res;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank, n_ranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);

  // Parse arguments.
  int    L         = 20;
  int    chi_max   = 64;
  double dt        = 0.1;
  int    N_steps   = 20;
  int    order     = 2;
  int    n_streams = 1;
  bool   weak_mode = false;

  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--L")        && i+1 < argc) L         = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--chi") && i+1 < argc) chi_max   = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--dt")  && i+1 < argc) dt        = atof(argv[++i]);
    else if (!strcmp(argv[i], "--steps")&&i+1<argc)   N_steps   = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--order")&&i+1<argc)   order     = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--n-streams")&&i+1<argc)n_streams= atoi(argv[++i]);
    else if (!strcmp(argv[i], "--weak")) weak_mode = true;
    else if (!strcmp(argv[i], "--strong")) weak_mode = false;
  }

#ifdef USE_GPU
  // Assign each rank a GPU (round-robin over visible devices).
  {
    int n_gpus = gpu_device_count();
    if (n_gpus > 0) gpu_set_device(rank % n_gpus);
  }
#endif

  // Build this rank's parameter list.
  std::vector<Param> my_params;
  if (weak_mode) {
    my_params = make_grid_weak(rank, n_ranks);
  } else {
    // Strong scaling: distribute the full 16-pair grid across ranks.
    std::vector<Param> full_grid = make_grid_strong();
    int total = (int)full_grid.size();
    // Assign params by round-robin across ranks.
    for (int pi = rank; pi < total; pi += n_ranks)
      my_params.push_back(full_grid[pi]);
  }

  // Run all assigned parameter pairs.
  std::vector<RunResult> my_results;
  my_results.reserve(my_params.size());
  for (const Param& p : my_params)
    my_results.push_back(run_param(p.J, p.g, L, chi_max, dt,
                                   N_steps, order, n_streams));

  // Flatten results for MPI_Gather.
  // Each result contributes 5 doubles: J, g, t_wall_s, max_chi, entropy.
  const int FIELDS = 5;
  std::vector<double> send_buf;
  send_buf.reserve(my_results.size() * FIELDS);
  for (const auto& r : my_results) {
    send_buf.push_back(r.J);
    send_buf.push_back(r.g);
    send_buf.push_back(r.t_wall_s);
    send_buf.push_back(double(r.max_chi));
    send_buf.push_back(r.final_entropy);
  }
  int my_count = (int)send_buf.size();

  // Gather counts to rank 0.
  std::vector<int> counts(n_ranks), displs(n_ranks);
  MPI_Gather(&my_count, 1, MPI_INT, counts.data(), 1, MPI_INT, 0,
             MPI_COMM_WORLD);

  std::vector<double> recv_buf;
  if (rank == 0) {
    displs[0] = 0;
    for (int i = 1; i < n_ranks; ++i) displs[i] = displs[i-1] + counts[i-1];
    int total_elems = displs[n_ranks-1] + counts[n_ranks-1];
    recv_buf.resize(total_elems);
  }

  MPI_Gatherv(send_buf.data(), my_count, MPI_DOUBLE,
              recv_buf.data(), counts.data(), displs.data(), MPI_DOUBLE,
              0, MPI_COMM_WORLD);

  // Rank 0: print CSV.
  if (rank == 0) {
    std::printf("rank,J,g,L,chi_max,N_steps,t_wall_s,max_chi,final_entropy\n");
    // We don't know which rank sent which result for the combined buffer.
    // Since counts differ, emit without rank attribution.
    int n_total = (int)recv_buf.size() / FIELDS;
    for (int i = 0; i < n_total; ++i) {
      double J   = recv_buf[i * FIELDS + 0];
      double g   = recv_buf[i * FIELDS + 1];
      double tw  = recv_buf[i * FIELDS + 2];
      int    mc  = int(recv_buf[i * FIELDS + 3]);
      double ent = recv_buf[i * FIELDS + 4];
      std::printf("-1,%.2f,%.2f,%d,%d,%d,%.6f,%d,%.6f\n",
                  J, g, L, chi_max, N_steps, tw, mc, ent);
    }
    // Also print scaling summary: max wall time across all results.
    double t_max = 0.0;
    for (int i = 0; i < n_total; ++i)
      t_max = std::max(t_max, recv_buf[i * FIELDS + 2]);
    std::fprintf(stderr,
                 "# scaling: n_ranks=%d total_params=%d "
                 "t_max_s=%.4f mode=%s\n",
                 n_ranks, n_total, t_max, weak_mode ? "weak" : "strong");
    std::fflush(stdout);
  }

  MPI_Finalize();
  return 0;
}
