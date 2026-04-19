#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

#include "mps.h"
#include "model.h"
#include "tebd.h"

int main() {
  const int L = 30;
  const double dt = 0.1;
  const int order = 4;
  const int chi_max = 100;
  const double svd_min = 1e-12;
  const double t_max = 5.0;

  BondModel model = tfi_chain(L, 1.0, 1.0);
  std::vector<int> init(L, 0);
  MPS psi = MPS::product_state(L, init.data(), model.d, chi_max);
  TEBDEngine eng(&psi, &model, dt, order, 1, chi_max, svd_min);

  std::vector<double> entropy(L - 1);
  std::vector<double> Sx(L), Sz(L);

  auto measure = [&]() {
    psi.entropy(entropy.data());
    psi.expect(PAULI_X, Sx.data());
    psi.expect(PAULI_Z, Sz.data());
  };

  // CSV header.
  std::cout << "# t,S_mid,max_chi,trunc_err" << std::endl;

  auto print_row = [&]() {
    int mid = L / 2 - 1;
    std::cout << std::fixed << std::setprecision(2) << eng.evolved_time << ","
              << std::setprecision(4) << entropy[mid] << ","
              << psi.max_bond_dim() << "," << std::scientific
              << std::setprecision(2) << eng.trunc_err_eps << std::endl;
  };

  measure();
  print_row();

  while (eng.evolved_time < t_max - 1e-9) {
    eng.run();
    measure();
    print_row();

    // Progress to stderr.
    int mid = L / 2 - 1;
    std::cerr << "t=" << std::fixed << std::setprecision(2)
              << eng.evolved_time << "  max chi=" << std::setw(3)
              << psi.max_bond_dim() << "  mid-bond S=" << std::setprecision(4)
              << entropy[mid] << "  trunc_err=" << std::scientific
              << std::setprecision(2) << eng.trunc_err_eps << std::endl;
  }

  return 0;
}
