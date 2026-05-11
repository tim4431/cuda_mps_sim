#ifndef MPS_H_
#define MPS_H_

#include "tensor.h"

// Pauli matrices stored as flat 2x2 row-major arrays.
extern const Cdouble PAULI_I[4];
extern const Cdouble PAULI_X[4];
extern const Cdouble PAULI_Y[4];
extern const Cdouble PAULI_Z[4];

// Right-canonical MPS for a finite chain.
//
// Bs[i] has legs (chi_L, d, chi_R), right-canonical.
// Ss[i] is the Schmidt spectrum on the bond left of site i.
// chi[i] = length of Ss[i] = left bond dimension at site i (length L+1).
struct MPS {
  int L, d;
  Tensor* Bs;    // array of L tensors
  double** Ss;   // array of L double* vectors
  int* chi;      // bond dimensions, length L+1
  int chi_max;

  MPS(int L, int d, int chi_max);
  ~MPS();

  // Disable copy (use explicit clone instead).
  MPS(const MPS&) = delete;
  MPS& operator=(const MPS&) = delete;

  // Move.
  MPS(MPS&& other) noexcept;
  MPS& operator=(MPS&& other) noexcept;

  // Construct product state |states[0], states[1], ..., states[L-1]>.
  static MPS product_state(int L, const int states[], int d, int chi_max);

  // Single-site tensor: theta1 = diag(Ss[i]) @ Bs[i], shape (chi_L, d, chi_R).
  Tensor theta1(int i) const;

  // Two-site tensor: theta2 on sites (i, i+1), shape (chi_L, d, d, chi_R).
  Tensor theta2(int i) const;

  // Bond dimensions array, length L-1.
  void bond_dims(int out[]) const;
  int max_bond_dim() const;

  // Von-Neumann entropy at each of the L-1 interior bonds.
  void entropy(double out[]) const;

  // Local expectation values <op_i> for i = 0..L-1.
  // op is a 2x2 matrix in row-major (d x d).
  void expect(const Cdouble* op, double out[]) const;

  // Correlation function C[i*L + j] = <op_A_i op_B_j>.
  // out must have length L*L.
  void corr(const Cdouble* op_A, const Cdouble* op_B, double out[]) const;

  // Full state vector of length d^L. Only feasible for small L.
  Tensor to_state_vector() const;
};

#endif  // MPS_H_
