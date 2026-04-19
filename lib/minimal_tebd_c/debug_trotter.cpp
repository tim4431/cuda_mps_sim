// Debug: apply Trotter substeps one at a time, comparing MPS vs full-space
// after each substep to see where the divergence starts.

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "exact_diag.h"
#include "linalg.h"
#include "model.h"
#include "mps.h"
#include "tebd.h"

static std::vector<Cdouble> embed_gate(const Cdouble* U_mat, int b, int L,
                                       int d) {
  int dim = 1;
  for (int i = 0; i < L; ++i) dim *= d;
  int d2 = d * d;
  int dim_left = 1;
  for (int i = 0; i < b; ++i) dim_left *= d;
  int dim_right = 1;
  for (int i = 0; i < L - b - 2; ++i) dim_right *= d;
  std::vector<Cdouble> I_left(dim_left * dim_left, 0.0);
  for (int i = 0; i < dim_left; ++i) I_left[i * dim_left + i] = 1.0;
  std::vector<Cdouble> I_right(dim_right * dim_right, 0.0);
  for (int i = 0; i < dim_right; ++i) I_right[i * dim_right + i] = 1.0;
  int t1_r = dim_left * d2, t1_c = dim_left * d2;
  std::vector<Cdouble> temp(t1_r * t1_c);
  kron(I_left.data(), dim_left, dim_left, U_mat, d2, d2, temp.data());
  std::vector<Cdouble> full(dim * dim);
  kron(temp.data(), t1_r, t1_c, I_right.data(), dim_right, dim_right,
       full.data());
  return full;
}

int main() {
  int L = 3;
  int d = 2;
  int dim = 1;
  for (int i = 0; i < L; ++i) dim *= d;

  double J = 1.0, g = 1.0;
  double dt = 0.1;
  int order = 2;
  BondModel model = tfi_chain(L, J, g);

  std::vector<int> init(L, 0);
  std::vector<Cdouble> psi_full(dim, 0.0);
  {
    int idx = 0;
    for (int i = 0; i < L; ++i) idx = idx * d + init[i];
    psi_full[idx] = 1.0;
  }

  MPS psi_mps = MPS::product_state(L, init.data(), d, 40);

  // Order 2 schedule: even(0.5), odd(1.0), even(0.5)
  int parities[] = {0, 1, 0};
  double fracs[] = {0.5, 1.0, 0.5};
  int n_sub = 3;

  Cdouble coeff_prefactor(0, -1);

  printf("L=%d, dt=%.3f, order=%d\n", L, dt, order);
  printf("Schedule: ");
  for (int s = 0; s < n_sub; ++s)
    printf("(%s, %.3f) ", parities[s] == 0 ? "even" : "odd", fracs[s]);
  printf("\n\n");

  for (int s = 0; s < n_sub; ++s) {
    int start = parities[s];
    Cdouble coeff = coeff_prefactor * fracs[s] * dt;

    printf("--- Substep %d: %s bonds, frac=%.3f ---\n", s,
           start == 0 ? "even" : "odd", fracs[s]);

    for (int b = start; b < L - 1; b += 2) {
      printf("  Updating bond %d\n", b);

      // Full-space gate application.
      int d2 = d * d;
      std::vector<Cdouble> H_scaled(d2 * d2);
      for (int i = 0; i < d2 * d2; ++i)
        H_scaled[i] = coeff * model.H_bonds[b].data[i];
      std::vector<Cdouble> U_mat(d2 * d2);
      matrix_expm(H_scaled.data(), d2, U_mat.data());
      auto U_full_mat = embed_gate(U_mat.data(), b, L, d);
      std::vector<Cdouble> new_psi(dim, 0.0);
      zgemv(U_full_mat.data(), dim, psi_full.data(), new_psi.data(), dim, dim);
      psi_full = std::move(new_psi);

      // MPS gate application.
      Tensor gate(4, (int[]){d, d, d, d});
      std::copy(U_mat.begin(), U_mat.end(), gate.data);
      update_bond(&psi_mps, b, gate, 40, 1e-12);
    }

    // Compare state vectors after this substep.
    Tensor sv = psi_mps.to_state_vector();

    printf("  Full-space state vector:\n    ");
    for (int i = 0; i < dim; ++i)
      printf("(%.6f,%.6f) ", psi_full[i].real(), psi_full[i].imag());
    printf("\n");

    printf("  MPS state vector:\n    ");
    for (int i = 0; i < dim; ++i)
      printf("(%.6f,%.6f) ", sv(i).real(), sv(i).imag());
    printf("\n");

    Cdouble overlap = 0.0;
    for (int i = 0; i < dim; ++i)
      overlap += std::conj(psi_full[i]) * sv(i);
    double infid = 1.0 - std::abs(overlap);
    printf("  Infidelity (full vs MPS): %.3e\n\n", infid);

    // Print MPS internal state.
    printf("  MPS internals:\n");
    for (int i = 0; i < L; ++i) {
      printf("    Bs[%d] shape=(%d,%d,%d), Ss[%d]=[", i, psi_mps.Bs[i].shape[0],
             psi_mps.Bs[i].shape[1], psi_mps.Bs[i].shape[2], i);
      int chi_l = psi_mps.Bs[i].shape[0];
      for (int k = 0; k < chi_l; ++k) {
        if (k > 0) printf(", ");
        printf("%.6f", psi_mps.Ss[i][k]);
      }
      printf("]\n");
    }
    printf("\n");
  }

  return 0;
}
