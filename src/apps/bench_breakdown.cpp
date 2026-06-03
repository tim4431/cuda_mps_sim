// Time breakdown of update_bond_gpu by sub-stage, at a sweep of chi.
//
// CSV: chi,h2d_ms,gate_ms,r2c_ms,svd_ms,restore_ms,d2h_ms,total_ms

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <random>
#include <vector>

#include "gpu_tebd.h"
#include "linalg.h"
#include "mps.h"
#include "tebd.h"

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
    int chiL = bond_dim(i), chiR = bond_dim(i + 1);
    int sh[3] = {chiL, d, chiR};
    Tensor B(3, sh, cap);
    for (int a = 0; a < chiL; ++a)
      for (int p = 0; p < d; ++p)
        for (int b = 0; b < chiR; ++b)
          B(a, p, b) = Cdouble(N(rng), N(rng));
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
  int d2 = d * d;
  std::vector<Cdouble> H(d2 * d2);
  std::mt19937_64 rng(7);
  std::normal_distribution<double> N(0.0, 1.0);
  for (int i = 0; i < d2; ++i)
    for (int j = i; j < d2; ++j) {
      Cdouble v = (i == j) ? Cdouble(N(rng), 0.0) : Cdouble(N(rng), N(rng));
      H[i * d2 + j] = v;
      H[j * d2 + i] = std::conj(v);
    }
  for (auto& z : H) z *= Cdouble(0.0, -0.05);
  std::vector<Cdouble> U(d2 * d2);
  matrix_expm(H.data(), d2, U.data());
  int sh[4] = {d, d, d, d};
  Tensor G(4, sh);
  std::copy(U.begin(), U.end(), G.data);
  return G;
}

int main() {
  const int L = 16, d = 2;
  std::vector<int> chis = {16, 32, 64, 96, 128, 160, 200};
  std::printf("chi,h2d_ms,gate_ms,r2c_ms,svd_ms,restore_ms,d2h_ms,total_ms\n");
  for (int chi : chis) {
    Tensor gate = make_random_gate(d);
    GpuTebdWorkspace ws(d, chi);
    // Warm up.
    {
      MPS psi = make_random_mps(L, d, chi);
      update_bond_gpu(&psi, 7, gate, chi, 1e-12, ws);
    }
    GpuBondTimings tim;
    int iters = (chi <= 64) ? 30 : 12;
    for (int it = 0; it < iters; ++it) {
      MPS psi = make_random_mps(L, d, chi);
      update_bond_gpu(&psi, 7, gate, chi, 1e-12, ws, &tim);
    }
    double total = tim.h2d_ms + tim.gate_kernel_ms + tim.row_to_col_ms +
                   tim.svd_ms + tim.restore_ms + tim.d2h_ms;
    double inv = 1.0 / iters;
    std::printf("%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n", chi,
                tim.h2d_ms * inv, tim.gate_kernel_ms * inv,
                tim.row_to_col_ms * inv, tim.svd_ms * inv,
                tim.restore_ms * inv, tim.d2h_ms * inv, total * inv);
    std::fflush(stdout);
  }
  return 0;
}
