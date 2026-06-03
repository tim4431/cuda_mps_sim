#ifndef EXACT_DIAG_H_
#define EXACT_DIAG_H_

#include "model.h"
#include "tensor.h"

// Build full d^L x d^L Hamiltonian from bond Hamiltonians.
void bond_model_to_full_hamiltonian(const BondModel& model, Cdouble* H_full);

// Embed single-site operator at site i into full d^L space.
void single_site_op_full(const Cdouble* op, int i, int L, int d,
                         Cdouble* out);

// Embed two-site operator A_i B_j into full d^L space.
void two_site_op_full(const Cdouble* opA, const Cdouble* opB, int i, int j,
                      int L, int d, Cdouble* out);

// Full state-vector evolution for validation.
struct ExactEvolution {
  int L, d, dim;       // dim = d^L
  Cdouble* psi;        // state vector, length dim
  Cdouble* H;          // full Hamiltonian, dim x dim
  Cdouble* U_dt;       // expm(coeff * dt * H), dim x dim
  double dt;
  double evolved_time;
  int type_evo;        // 0 = real, 1 = imag

  ExactEvolution(const BondModel& model, const Cdouble* state_vector,
                 double dt, int type_evo = 0);
  ~ExactEvolution();

  ExactEvolution(const ExactEvolution&) = delete;
  ExactEvolution& operator=(const ExactEvolution&) = delete;
  ExactEvolution(ExactEvolution&& other) noexcept;
  ExactEvolution& operator=(ExactEvolution&&) = delete;

  // Construct from product state.
  static ExactEvolution from_product_state(const BondModel& model,
                                           const int states[], double dt,
                                           int type_evo = 0);

  // Advance by one dt.
  void step();

  // Observables.
  void expectation_value(const Cdouble* op, double out[]) const;
  void correlation_function(const Cdouble* opA, const Cdouble* opB,
                            double out[]) const;
  void entanglement_entropy(double out[]) const;
};

#endif  // EXACT_DIAG_H_
