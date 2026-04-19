#ifndef VALIDATE_H_
#define VALIDATE_H_

#include <vector>

#include "model.h"

struct ValidationSummary {
  std::vector<double> t;
  std::vector<double> err_Sz, err_Sx, err_corr_XX, err_entropy;
  double trunc_err;
  double fidelity_drop;
};

// Compare TEBD and exact-diag evolution of the same initial product state.
// model.L must be small enough that d^L fits in memory (L <= ~14).
ValidationSummary validate_tebd_vs_ed(
    const BondModel& model, double dt = 0.05, double t_max = 1.0,
    int chi_max = 64, double svd_min = 1e-12, int order = 4,
    const int* initial_states = nullptr,  // default: all zeros
    bool verbose = true);

#endif  // VALIDATE_H_
