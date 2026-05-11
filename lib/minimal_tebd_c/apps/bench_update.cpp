// Microbenchmark: time a single update_bond at a fixed bond dimension,
// CPU vs GPU.  This isolates the kernel/SVD/restore costs from the rest of
// TEBD bookkeeping and avoids the variable-bond-dimension noise of the
// full-step benchmark.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <random>
#include <vector>

#include "gpu_tebd.h"
#include "linalg.h"
#include "model.h"
#include "mps.h"
#include "tebd.h"

using clk = std::chrono::high_resolution_clock;

static double sec_since(const clk::time_point& t0) {
  using D = std::chrono::duration<double>;
  return std::chrono::duration_cast<D>(clk::now() - t0).count();
}

// Build a synthetic MPS with bond dim chi at every interior bond.  Random
// values with norm = 1 (approximately).  Schmidt values are a uniform 1/sqrt(chi).
static MPS make_random_mps(int L, int d, int chi) {
  MPS psi(L, d, chi);
  std::mt19937_64 rng(42);
  std::normal_distribution<double> N(0.0, 1.0);
  int cap = chi * d * chi;
  auto bond_dim = [&](int i) {
    if (i == 0 || i == L) return 1;
    return chi;
  };
  for (int i = 0; i < L; ++i) {
    int chiL = bond_dim(i);
    int chiR = bond_dim(i + 1);
    int sh[3] = {chiL, d, chiR};
    Tensor B(3, sh, cap);
    // Fill with random complex, then normalise so the right contraction is
    // approximately the identity (right-canonical form).
    for (int a = 0; a < chiL; ++a)
      for (int p = 0; p < d; ++p)
        for (int b = 0; b < chiR; ++b)
          B(a, p, b) = Cdouble(N(rng), N(rng));
    // Crude renormalisation: divide by sqrt(d * chiR) so that
    // sum_{p, b} |B|^2 ~ chiL.
    double scale = 1.0 / std::sqrt(double(d) * chiR);
    for (int k = 0; k < B.size; ++k) B.data[k] *= scale;
    psi.Bs[i] = std::move(B);
    psi.Ss[i] = new double[chi];
    std::fill(psi.Ss[i], psi.Ss[i] + chi, 0.0);
    double sval = 1.0 / std::sqrt(double(chiL));
    for (int k = 0; k < chiL; ++k) psi.Ss[i][k] = sval;
    psi.chi[i] = chiL;
  }
  psi.chi[L] = 1;
  return psi;
}

static Tensor make_random_gate(int d) {
  // Build a unitary by applying expm(-i*H) for a random Hermitian H on d*d.
  int d2 = d * d;
  std::vector<Cdouble> H(d2 * d2);
  std::mt19937_64 rng(7);
  std::normal_distribution<double> N(0.0, 1.0);
  for (int i = 0; i < d2; ++i)
    for (int j = i; j < d2; ++j) {
      Cdouble v = (i == j) ? Cdouble(N(rng), 0.0)
                            : Cdouble(N(rng), N(rng));
      H[i * d2 + j] = v;
      H[j * d2 + i] = std::conj(v);
    }
  // Scale to keep ||H|| small so expm is close to identity.
  for (auto& z : H) z *= Cdouble(0.0, -0.05);
  std::vector<Cdouble> U(d2 * d2);
  matrix_expm(H.data(), d2, U.data());
  int sh[4] = {d, d, d, d};
  Tensor G(4, sh);
  std::copy(U.begin(), U.end(), G.data);
  return G;
}

static void run_chi(int chi, int iters, bool csv) {
  const int L = 16;
  const int d = 2;
  const int chi_max = chi;
  const double svd_min = 1e-12;

  Tensor gate = make_random_gate(d);

  // CPU timing.
  double t_cpu = 0;
  for (int it = 0; it < iters; ++it) {
    MPS psi = make_random_mps(L, d, chi);
    int site = 7;  // middle bond
    auto t0 = clk::now();
    update_bond(&psi, site, gate, chi_max, svd_min);
    t_cpu += sec_since(t0);
  }
  t_cpu /= iters;

  // GPU timing.  Persistent workspace; warm-up first.
  GpuTebdWorkspace ws(d, chi_max);
  {
    MPS psi = make_random_mps(L, d, chi);
    update_bond_gpu(&psi, 7, gate, chi_max, svd_min, ws);
  }
  double t_gpu = 0;
  for (int it = 0; it < iters; ++it) {
    MPS psi = make_random_mps(L, d, chi);
    int site = 7;
    auto t0 = clk::now();
    update_bond_gpu(&psi, site, gate, chi_max, svd_min, ws);
    t_gpu += sec_since(t0);
  }
  t_gpu /= iters;

  double speedup = t_cpu / t_gpu;
  if (csv) {
    std::printf("%d,%.6e,%.6e,%.3f\n", chi, t_cpu, t_gpu, speedup);
  } else {
    std::printf("chi=%4d  CPU=%8.3f ms  GPU=%8.3f ms  speedup=%.2fx\n", chi,
                t_cpu * 1e3, t_gpu * 1e3, speedup);
  }
  std::fflush(stdout);
}

int main(int argc, char** argv) {
  if (argc >= 3) {
    int chi = std::atoi(argv[1]);
    int iters = std::atoi(argv[2]);
    run_chi(chi, iters, /*csv=*/false);
    return 0;
  }
  std::printf("chi,t_cpu_s,t_gpu_s,speedup\n");
  for (int chi : {16, 32, 48, 64, 96, 128, 160, 200, 256}) {
    int iters = (chi <= 64) ? 50 : (chi <= 160 ? 20 : 10);
    run_chi(chi, iters, /*csv=*/true);
  }
  return 0;
}
