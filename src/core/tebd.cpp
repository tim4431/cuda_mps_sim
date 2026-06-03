#include "tebd.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <set>
#include <vector>

#include "linalg.h"

// ===========================================================================
// Trotter schedules
// ===========================================================================

static int trotter_substeps(int order, int parities[], double fracs[]) {
  if (order == 1) {
    parities[0] = 0;
    fracs[0] = 1.0;
    parities[1] = 1;
    fracs[1] = 1.0;
    return 2;
  }
  if (order == 2) {
    parities[0] = 0;
    fracs[0] = 0.5;
    parities[1] = 1;
    fracs[1] = 1.0;
    parities[2] = 0;
    fracs[2] = 0.5;
    return 3;
  }
  if (order == 4) {
    double theta = 1.0 / (2.0 - std::pow(2.0, 1.0 / 3.0));
    double a1 = theta / 2.0;
    double a2 = (1.0 - theta) / 2.0;
    double b1 = theta;
    double b2 = 1.0 - 2.0 * theta;
    int p[] = {0, 1, 0, 1, 0, 1, 0};
    double f[] = {a1, b1, a2, b2, a2, b1, a1};
    for (int i = 0; i < 7; ++i) {
      parities[i] = p[i];
      fracs[i] = f[i];
    }
    return 7;
  }
  assert(false && "unsupported order; use 1, 2, or 4");
  return 0;
}

// ===========================================================================
// Compute gates: U_b = expm(coeff * H_bond_b)
// ===========================================================================

static void compute_gates(const BondModel& model, Cdouble coeff,
                          Tensor* gates) {
  int d = model.d;
  int d2 = d * d;
  std::vector<Cdouble> H_scaled(d2 * d2);
  std::vector<Cdouble> U_mat(d2 * d2);
  for (int b = 0; b < model.L - 1; ++b) {
    // Scale: H_scaled = coeff * H_bonds[b] (reshaped to d2 x d2).
    for (int i = 0; i < d2 * d2; ++i)
      H_scaled[i] = coeff * model.H_bonds[b].data[i];
    matrix_expm(H_scaled.data(), d2, U_mat.data());
    int sh[4] = {d, d, d, d};
    gates[b] = Tensor(4, sh);
    std::copy(U_mat.begin(), U_mat.end(), gates[b].data);
  }
}

// ===========================================================================
// SVD + truncation
// ===========================================================================

void svd_truncate(const Tensor& theta, int chi_max, double svd_min, Tensor& A,
                  double* S, Tensor& B, double& eps) {
  int chivL = theta.shape[0];
  int dL = theta.shape[1];
  int dR = theta.shape[2];
  int chivR = theta.shape[3];
  int m = chivL * dL;
  int n = dR * chivR;
  int mn = std::min(m, n);

  // Reshape theta to 2D (m x n) for SVD.
  // theta is already row-major (chivL, dL, dR, chivR), so reshape is free.
  std::vector<Cdouble> U_full(m * mn);
  std::vector<double> s_full(mn);
  std::vector<Cdouble> Vt_full(mn * n);

  // Handle m < n case: SVD requires m >= n.
  if (m >= n) {
    svd(theta.data, m, n, n, U_full.data(), s_full.data(), Vt_full.data());
  } else {
    // Transpose, SVD, then un-transpose.
    // A^T = V S U^T => A = U S V^T
    std::vector<Cdouble> At(n * m);
    for (int i = 0; i < m; ++i)
      for (int j = 0; j < n; ++j) At[j * m + i] = theta.data[i * n + j];
    std::vector<Cdouble> U_t(n * mn);
    std::vector<Cdouble> Vt_t(mn * m);
    svd(At.data(), n, m, m, U_t.data(), s_full.data(), Vt_t.data());
    // A^T = U_t S Vt_t, so A = Vt_t^T S U_t^T => U = Vt_t^T, Vt = U_t^T.
    U_full.resize(m * mn);
    Vt_full.resize(mn * n);
    for (int i = 0; i < m; ++i)
      for (int j = 0; j < mn; ++j)
        U_full[i * mn + j] = Vt_t[j * m + i];
    for (int i = 0; i < mn; ++i)
      for (int j = 0; j < n; ++j)
        Vt_full[i * n + j] = U_t[j * mn + i];
  }

  // Determine how many to keep.
  int keep = 0;
  for (int i = 0; i < mn; ++i) {
    if (s_full[i] > svd_min) ++keep;
  }
  keep = std::min(keep, chi_max);
  if (keep < 1) keep = 1;

  // s_full is already sorted descending by our SVD.
  // Compute truncation error.
  double norm_sq = 0.0;
  for (int i = 0; i < mn; ++i) norm_sq += s_full[i] * s_full[i];
  double kept_sq = 0.0;
  for (int i = 0; i < keep; ++i) kept_sq += s_full[i] * s_full[i];
  eps = 1.0 - kept_sq;  // before renormalization

  // Renormalize.
  double norm = std::sqrt(kept_sq);
  for (int i = 0; i < keep; ++i) S[i] = s_full[i] / norm;

  // A: shape (chivL, dL, keep)
  int shA[3] = {chivL, dL, keep};
  A = Tensor(3, shA);
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < keep; ++j) A.data[i * keep + j] = U_full[i * mn + j];

  // B: shape (keep, dR, chivR)
  int shB[3] = {keep, dR, chivR};
  B = Tensor(3, shB);
  for (int i = 0; i < keep; ++i)
    for (int j = 0; j < n; ++j) B.data[i * n + j] = Vt_full[i * n + j];
}

// ===========================================================================
// update_bond
// ===========================================================================

double update_bond(MPS* psi, int site, const Tensor& gate, int chi_max,
                   double svd_min) {
  int d = psi->d;

  // theta2: (chi_L, d, d, chi_R)
  Tensor theta = psi->theta2(site);
  int chivL = theta.shape[0];
  int chivR = theta.shape[3];

  // Apply gate: U(i',j',i,j) * theta(vL,i,j,vR) -> Utheta(i',j',vL,vR)
  // Then transpose to (vL, i', j', vR).
  int sh_out[4] = {chivL, d, d, chivR};
  Tensor Utheta(4, sh_out);
  for (int vL = 0; vL < chivL; ++vL) {
    for (int ip = 0; ip < d; ++ip) {
      for (int jp = 0; jp < d; ++jp) {
        for (int vR = 0; vR < chivR; ++vR) {
          Cdouble s = 0.0;
          for (int i = 0; i < d; ++i)
            for (int j = 0; j < d; ++j)
              s += gate(ip, jp, i, j) * theta(vL, i, j, vR);
          Utheta(vL, ip, jp, vR) = s;
        }
      }
    }
  }

  // SVD and truncate.
  Tensor A_out, B_out;
  std::vector<double> S_buf(chi_max > 0 ? chi_max : 1);
  double eps;
  svd_truncate(Utheta, chi_max, svd_min, A_out, S_buf.data(), B_out, eps);

  int keep = A_out.shape[2];

  // Restore right-canonical B-form at site i.
  // new_Bs[i] = diag(1/Ss[i]) @ A @ diag(S)
  int chi_L = psi->Bs[site].shape[0];
  int sh_Bi[3] = {chi_L, d, keep};
  Tensor new_Bi(3, sh_Bi, psi->Bs[site].capacity);
  for (int a = 0; a < chi_L; ++a) {
    double ss_inv =
        (psi->Ss[site][a] > 1e-14) ? 1.0 / psi->Ss[site][a] : 0.0;
    for (int p = 0; p < d; ++p) {
      for (int b = 0; b < keep; ++b) {
        new_Bi(a, p, b) = ss_inv * A_out(a, p, b) * S_buf[b];
      }
    }
  }

  psi->Bs[site] = std::move(new_Bi);

  // Update Ss[site+1] and Bs[site+1].
  delete[] psi->Ss[site + 1];
  psi->Ss[site + 1] = new double[psi->chi_max];
  std::fill(psi->Ss[site + 1], psi->Ss[site + 1] + psi->chi_max, 0.0);
  for (int i = 0; i < keep; ++i) psi->Ss[site + 1][i] = S_buf[i];
  psi->chi[site + 1] = keep;

  // B_out already has correct shape (keep, d, chivR). Set with capacity.
  int shB[3] = {keep, d, chivR};
  Tensor new_Bj(3, shB, psi->Bs[site + 1].capacity);
  std::copy(B_out.data, B_out.data + B_out.size, new_Bj.data);
  psi->Bs[site + 1] = std::move(new_Bj);

  return eps;
}

// ===========================================================================
// TEBDEngine
// ===========================================================================

TEBDEngine::TEBDEngine(MPS* psi, BondModel* model, double dt, int order,
                       int N_steps, int chi_max, double svd_min, int type_evo)
    : psi(psi),
      model(model),
      dt(dt),
      order(order),
      N_steps(N_steps),
      chi_max(chi_max),
      svd_min(svd_min),
      type_evo(type_evo),
      evolved_time(0.0),
      trunc_err_eps(0.0),
      n_schedule(0),
      schedule(nullptr),
      n_unique_fracs(0),
      unique_fracs(nullptr),
      gate_sets(nullptr) {
  assert(type_evo == 0 || type_evo == 1);
  build_gate_cache();
}

TEBDEngine::~TEBDEngine() {
  delete[] schedule;
  delete[] unique_fracs;
  if (gate_sets) {
    for (int i = 0; i < n_unique_fracs; ++i) delete[] gate_sets[i];
    delete[] gate_sets;
  }
}

void TEBDEngine::build_gate_cache() {
  int parities[7];
  double fracs[7];
  n_schedule = trotter_substeps(order, parities, fracs);
  schedule = new TrotterStep[n_schedule];

  // Collect unique fractions.
  std::set<double> frac_set;
  for (int i = 0; i < n_schedule; ++i) frac_set.insert(fracs[i]);
  n_unique_fracs = frac_set.size();
  unique_fracs = new double[n_unique_fracs];
  int idx = 0;
  for (double f : frac_set) unique_fracs[idx++] = f;

  // Map schedule entries to frac indices.
  for (int i = 0; i < n_schedule; ++i) {
    schedule[i].parity = parities[i];
    for (int k = 0; k < n_unique_fracs; ++k) {
      if (std::abs(unique_fracs[k] - fracs[i]) < 1e-15) {
        schedule[i].frac_idx = k;
        break;
      }
    }
  }

  // Compute gate sets.
  Cdouble coeff_prefactor = (type_evo == 0) ? Cdouble(0, -1) : Cdouble(-1, 0);
  gate_sets = new Tensor*[n_unique_fracs];
  for (int k = 0; k < n_unique_fracs; ++k) {
    gate_sets[k] = new Tensor[model->L - 1];
    Cdouble coeff = coeff_prefactor * unique_fracs[k] * dt;
    compute_gates(*model, coeff, gate_sets[k]);
  }
}

void TEBDEngine::run() {
  for (int s = 0; s < N_steps; ++s) {
    step();
    evolved_time += dt;
  }
}

void TEBDEngine::step() {
  int L = model->L;
  for (int s = 0; s < n_schedule; ++s) {
    int start = schedule[s].parity;  // 0 = even, 1 = odd
    Tensor* gates = gate_sets[schedule[s].frac_idx];
    for (int b = start; b < L - 1; b += 2) {
      double eps = update_bond(psi, b, gates[b], chi_max, svd_min);
      trunc_err_eps = trunc_err_eps + eps - trunc_err_eps * eps;
    }
  }
}
