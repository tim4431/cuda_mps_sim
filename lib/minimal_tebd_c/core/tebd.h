#ifndef TEBD_H_
#define TEBD_H_

#include "model.h"
#include "mps.h"

struct TrotterStep {
  int parity;    // 0 = even, 1 = odd
  int frac_idx;  // index into unique_fracs
};

struct TEBDEngine {
  MPS* psi;
  BondModel* model;
  double dt;
  int order;
  int N_steps;
  int chi_max;
  double svd_min;
  int type_evo;  // 0 = real, 1 = imag
  double evolved_time;
  double trunc_err_eps;

  // Gate cache.
  int n_schedule;
  TrotterStep* schedule;
  int n_unique_fracs;
  double* unique_fracs;
  Tensor** gate_sets;  // gate_sets[frac_idx][bond]

  TEBDEngine(MPS* psi, BondModel* model, double dt, int order, int N_steps,
             int chi_max, double svd_min, int type_evo = 0);
  ~TEBDEngine();

  TEBDEngine(const TEBDEngine&) = delete;
  TEBDEngine& operator=(const TEBDEngine&) = delete;

  void run();   // Advance by N_steps * dt.
  void step();  // One Trotter step.

 private:
  void build_gate_cache();
};

// Core update: apply gate on bond (site, site+1), truncate, return eps.
double update_bond(MPS* psi, int site, const Tensor& gate, int chi_max,
                   double svd_min);

// SVD + truncation of a two-site tensor theta (chi_L, d, d, chi_R).
void svd_truncate(const Tensor& theta, int chi_max, double svd_min, Tensor& A,
                  double* S, Tensor& B, double& eps);

#endif  // TEBD_H_
