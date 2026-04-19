#ifndef MODEL_H_
#define MODEL_H_

#include "tensor.h"

// A model is a list of bond Hamiltonians.
// H_bonds[b] has shape (d, d, d, d) acting on sites (b, b+1)
// with index order (i_out, j_out, i_in, j_in).
struct BondModel {
  int L, d;
  Tensor* H_bonds;  // array of L-1 rank-4 tensors

  BondModel();
  ~BondModel();
  BondModel(BondModel&& other) noexcept;
  BondModel& operator=(BondModel&& other) noexcept;
  BondModel(const BondModel&) = delete;
  BondModel& operator=(const BondModel&) = delete;
};

// Transverse-field Ising chain:
//   H = -J sum_i sigma^x_i sigma^x_{i+1} - g sum_i sigma^z_i
// Critical at J = g = 1.
BondModel tfi_chain(int L, double J, double g);

#endif  // MODEL_H_
