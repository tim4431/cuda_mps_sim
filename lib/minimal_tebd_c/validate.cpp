#include "validate.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <vector>

#include "exact_diag.h"
#include "mps.h"
#include "tebd.h"

ValidationSummary validate_tebd_vs_ed(const BondModel& model, double dt,
                                      double t_max, int chi_max,
                                      double svd_min, int order,
                                      const int* initial_states,
                                      bool verbose) {
  int L = model.L, d = model.d;
  assert(L <= 14);

  std::vector<int> init(L, 0);
  if (initial_states) {
    for (int i = 0; i < L; ++i) init[i] = initial_states[i];
  }

  // TEBD setup.
  MPS psi_mps = MPS::product_state(L, init.data(), d, chi_max);
  TEBDEngine eng(&psi_mps, const_cast<BondModel*>(&model), dt, order, 1,
                 chi_max, svd_min);

  // ED setup.
  ExactEvolution ed =
      ExactEvolution::from_product_state(model, init.data(), dt);

  ValidationSummary summary;
  summary.t.push_back(0.0);

  std::vector<double> Sz_m(L), Sz_e(L), Sx_m(L), Sx_e(L);
  std::vector<double> C_m(L * L), C_e(L * L);
  std::vector<double> S_m(L - 1), S_e(L - 1);

  auto compare_step = [&]() {
    psi_mps.expect(PAULI_Z, Sz_m.data());
    ed.expectation_value(PAULI_Z, Sz_e.data());
    psi_mps.expect(PAULI_X, Sx_m.data());
    ed.expectation_value(PAULI_X, Sx_e.data());
    psi_mps.corr(PAULI_X, PAULI_X, C_m.data());
    ed.correlation_function(PAULI_X, PAULI_X, C_e.data());
    psi_mps.entropy(S_m.data());
    ed.entanglement_entropy(S_e.data());

    double e_Sz = 0, e_Sx = 0, e_XX = 0, e_S = 0;
    for (int i = 0; i < L; ++i) {
      e_Sz = std::max(e_Sz, std::abs(Sz_m[i] - Sz_e[i]));
      e_Sx = std::max(e_Sx, std::abs(Sx_m[i] - Sx_e[i]));
    }
    for (int i = 0; i < L * L; ++i)
      e_XX = std::max(e_XX, std::abs(C_m[i] - C_e[i]));
    for (int i = 0; i < L - 1; ++i)
      e_S = std::max(e_S, std::abs(S_m[i] - S_e[i]));

    summary.err_Sz.push_back(e_Sz);
    summary.err_Sx.push_back(e_Sx);
    summary.err_corr_XX.push_back(e_XX);
    summary.err_entropy.push_back(e_S);
  };

  compare_step();

  if (verbose) {
    std::cout << std::setw(6) << "t" << "  " << std::setw(10) << "max|dSz|"
              << "  " << std::setw(10) << "max|dSx|" << "  " << std::setw(10)
              << "max|dXX|" << "  " << std::setw(10) << "max|dS|" << "  "
              << std::setw(4) << "chi" << std::endl;
    std::cout << std::fixed << std::setprecision(3) << std::setw(6) << 0.0
              << "  " << std::scientific << std::setprecision(2)
              << std::setw(10) << summary.err_Sz.back() << "  "
              << std::setw(10) << summary.err_Sx.back() << "  "
              << std::setw(10) << summary.err_corr_XX.back() << "  "
              << std::setw(10) << summary.err_entropy.back() << "  "
              << std::setw(4) << psi_mps.max_bond_dim() << std::endl;
  }

  int n_steps = static_cast<int>(std::round(t_max / dt));
  for (int step = 0; step < n_steps; ++step) {
    eng.run();
    ed.step();
    summary.t.push_back(eng.evolved_time);
    compare_step();

    if (verbose) {
      std::cout << std::fixed << std::setprecision(3) << std::setw(6)
                << eng.evolved_time << "  " << std::scientific
                << std::setprecision(2) << std::setw(10)
                << summary.err_Sz.back() << "  " << std::setw(10)
                << summary.err_Sx.back() << "  " << std::setw(10)
                << summary.err_corr_XX.back() << "  " << std::setw(10)
                << summary.err_entropy.back() << "  " << std::setw(4)
                << psi_mps.max_bond_dim() << std::endl;
    }
  }

  // Final fidelity check.
  Tensor psi_full = psi_mps.to_state_vector();
  int dim = psi_full.size;
  Cdouble overlap = 0.0;
  for (int i = 0; i < dim; ++i)
    overlap += std::conj(psi_full(i)) * ed.psi[i];
  double fidelity = std::abs(overlap);

  summary.trunc_err = eng.trunc_err_eps;
  summary.fidelity_drop = 1.0 - fidelity;

  if (verbose) {
    std::cout << std::endl;
    std::cout << "final truncation error (accumulated): " << std::scientific
              << std::setprecision(3) << summary.trunc_err << std::endl;
    std::cout << "final |<psi_TEBD | psi_ED>|         : " << std::fixed
              << std::setprecision(10) << fidelity << std::endl;
    std::cout << "final fidelity drop                  : " << std::scientific
              << std::setprecision(3) << summary.fidelity_drop << std::endl;
  }

  return summary;
}
