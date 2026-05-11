#include "exact_diag.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <vector>

#include "linalg.h"

// ===========================================================================
// Kronecker product helpers for building full-space operators
// ===========================================================================

// In-place Kronecker: out = A kron B.
// A is (ra x ca), B is (rb x cb), out is (ra*rb x ca*cb).
static void kron_inplace(const Cdouble* A, int ra, int ca, const Cdouble* B,
                         int rb, int cb, Cdouble* out) {
  kron(A, ra, ca, B, rb, cb, out);
}

// Identity matrix of size n, heap-allocated.
static std::vector<Cdouble> eye(int n) {
  std::vector<Cdouble> I(n * n, 0.0);
  for (int i = 0; i < n; ++i) I[i * n + i] = 1.0;
  return I;
}

// Kronecker product returning a new vector.
static std::vector<Cdouble> kron_vec(const Cdouble* A, int ra, int ca,
                                      const Cdouble* B, int rb, int cb) {
  int rr = ra * rb, cc = ca * cb;
  std::vector<Cdouble> out(rr * cc);
  kron(A, ra, ca, B, rb, cb, out.data());
  return out;
}

// ===========================================================================
// Build full Hamiltonian
// ===========================================================================

void bond_model_to_full_hamiltonian(const BondModel& model, Cdouble* H_full) {
  int L = model.L, d = model.d;
  int dim = 1;
  for (int i = 0; i < L; ++i) dim *= d;
  std::memset(H_full, 0, dim * dim * sizeof(Cdouble));

  for (int b = 0; b < L - 1; ++b) {
    // H_bond reshaped to (d*d x d*d).
    const Cdouble* h_mat = model.H_bonds[b].data;
    int d2 = d * d;

    // left = I_{d^b}
    int dim_left = 1;
    for (int i = 0; i < b; ++i) dim_left *= d;
    // right = I_{d^(L-b-2)}
    int dim_right = 1;
    for (int i = 0; i < L - b - 2; ++i) dim_right *= d;

    auto I_left = eye(dim_left);
    auto I_right = eye(dim_right);

    // temp = kron(I_left, h_mat)
    auto temp = kron_vec(I_left.data(), dim_left, dim_left, h_mat, d2, d2);
    int tr = dim_left * d2;
    int tc = dim_left * d2;

    // full = kron(temp, I_right)
    auto full = kron_vec(temp.data(), tr, tc, I_right.data(), dim_right,
                         dim_right);

    // Add to H_full.
    for (int i = 0; i < dim * dim; ++i) H_full[i] += full[i];
  }
}

// ===========================================================================
// Single-site and two-site operators in full space
// ===========================================================================

void single_site_op_full(const Cdouble* op, int i, int L, int d,
                         Cdouble* out) {
  int dim = 1;
  for (int k = 0; k < L; ++k) dim *= d;

  int dim_left = 1;
  for (int k = 0; k < i; ++k) dim_left *= d;
  int dim_right = 1;
  for (int k = 0; k < L - i - 1; ++k) dim_right *= d;

  auto I_left = eye(dim_left);
  auto I_right = eye(dim_right);
  auto temp = kron_vec(I_left.data(), dim_left, dim_left, op, d, d);
  int tr = dim_left * d;
  auto full =
      kron_vec(temp.data(), tr, tr, I_right.data(), dim_right, dim_right);
  std::copy(full.begin(), full.end(), out);
}

void two_site_op_full(const Cdouble* opA, const Cdouble* opB, int i, int j,
                      int L, int d, Cdouble* out) {
  int dim = 1;
  for (int k = 0; k < L; ++k) dim *= d;

  if (i == j) {
    // A @ B at site i.
    Cdouble AB[4];
    for (int p = 0; p < d; ++p)
      for (int q = 0; q < d; ++q) {
        AB[p * d + q] = 0;
        for (int r = 0; r < d; ++r) AB[p * d + q] += opA[p * d + r] * opB[r * d + q];
      }
    single_site_op_full(AB, i, L, d, out);
    return;
  }

  // General case: build as tensor product with identities.
  int lo = std::min(i, j), hi = std::max(i, j);
  const Cdouble* op_lo = (i < j) ? opA : opB;
  const Cdouble* op_hi = (i < j) ? opB : opA;

  auto I_d = eye(d);
  // Build the chain of kron products: I^lo x op_lo x I^(hi-lo-1) x op_hi x I^(L-hi-1)
  // Start with identity of size d^0 = 1.
  std::vector<Cdouble> result = {1.0};
  int cur_dim = 1;
  for (int k = 0; k < L; ++k) {
    const Cdouble* mat;
    if (k == lo)
      mat = op_lo;
    else if (k == hi)
      mat = op_hi;
    else
      mat = I_d.data();
    auto next = kron_vec(result.data(), cur_dim, cur_dim, mat, d, d);
    cur_dim *= d;
    result = std::move(next);
  }
  std::copy(result.begin(), result.end(), out);
}

// ===========================================================================
// ExactEvolution
// ===========================================================================

ExactEvolution::ExactEvolution(const BondModel& model,
                               const Cdouble* state_vector, double dt,
                               int type_evo)
    : L(model.L), d(model.d), dt(dt), type_evo(type_evo), evolved_time(0.0) {
  dim = 1;
  for (int i = 0; i < L; ++i) dim *= d;

  psi = new Cdouble[dim];
  std::copy(state_vector, state_vector + dim, psi);

  H = new Cdouble[dim * dim];
  bond_model_to_full_hamiltonian(model, H);

  Cdouble coeff = (type_evo == 0) ? Cdouble(0, -1) : Cdouble(-1, 0);
  std::vector<Cdouble> scaled_H(dim * dim);
  for (int i = 0; i < dim * dim; ++i) scaled_H[i] = coeff * dt * H[i];

  U_dt = new Cdouble[dim * dim];
  matrix_expm(scaled_H.data(), dim, U_dt);
}

ExactEvolution::~ExactEvolution() {
  delete[] psi;
  delete[] H;
  delete[] U_dt;
}

ExactEvolution::ExactEvolution(ExactEvolution&& other) noexcept
    : L(other.L),
      d(other.d),
      dim(other.dim),
      psi(other.psi),
      H(other.H),
      U_dt(other.U_dt),
      dt(other.dt),
      evolved_time(other.evolved_time),
      type_evo(other.type_evo) {
  other.psi = nullptr;
  other.H = nullptr;
  other.U_dt = nullptr;
}

ExactEvolution ExactEvolution::from_product_state(const BondModel& model,
                                                   const int states[],
                                                   double dt, int type_evo) {
  int L = model.L, d = model.d;
  int dim = 1;
  for (int i = 0; i < L; ++i) dim *= d;

  // Compute index: k = sum_i states[i] * d^(L-1-i)
  int idx = 0;
  for (int i = 0; i < L; ++i) idx = idx * d + states[i];

  std::vector<Cdouble> psi(dim, 0.0);
  psi[idx] = 1.0;

  return ExactEvolution(model, psi.data(), dt, type_evo);
}

void ExactEvolution::step() {
  // psi = U_dt @ psi
  std::vector<Cdouble> new_psi(dim, 0.0);
  zgemv(U_dt, dim, psi, new_psi.data(), dim, dim);
  std::copy(new_psi.begin(), new_psi.end(), psi);

  if (type_evo == 1) {
    // Normalize for imaginary-time evolution.
    double nrm = 0.0;
    for (int i = 0; i < dim; ++i) nrm += std::norm(psi[i]);
    nrm = std::sqrt(nrm);
    for (int i = 0; i < dim; ++i) psi[i] /= nrm;
  }
  evolved_time += dt;
}

void ExactEvolution::expectation_value(const Cdouble* op, double out[]) const {
  std::vector<Cdouble> O_full(dim * dim);
  for (int i = 0; i < L; ++i) {
    single_site_op_full(op, i, L, d, O_full.data());
    // <psi| O |psi> = psi^H @ O @ psi
    std::vector<Cdouble> O_psi(dim, 0.0);
    zgemv(O_full.data(), dim, psi, O_psi.data(), dim, dim);
    Cdouble val = 0.0;
    for (int k = 0; k < dim; ++k) val += std::conj(psi[k]) * O_psi[k];
    out[i] = std::real(val);
  }
}

void ExactEvolution::correlation_function(const Cdouble* opA,
                                          const Cdouble* opB,
                                          double out[]) const {
  std::vector<Cdouble> O_full(dim * dim);
  for (int i = 0; i < L; ++i) {
    for (int j = 0; j < L; ++j) {
      two_site_op_full(opA, opB, i, j, L, d, O_full.data());
      std::vector<Cdouble> O_psi(dim, 0.0);
      zgemv(O_full.data(), dim, psi, O_psi.data(), dim, dim);
      Cdouble val = 0.0;
      for (int k = 0; k < dim; ++k) val += std::conj(psi[k]) * O_psi[k];
      out[i * L + j] = std::real(val);
    }
  }
}

void ExactEvolution::entanglement_entropy(double out[]) const {
  for (int b = 1; b < L; ++b) {
    int dim_left = 1;
    for (int i = 0; i < b; ++i) dim_left *= d;
    int dim_right = dim / dim_left;

    // Reshape psi as (dim_left x dim_right) matrix and SVD.
    int m = dim_left, n = dim_right;
    int mn = std::min(m, n);

    std::vector<Cdouble> U(m * mn);
    std::vector<double> S(mn);
    std::vector<Cdouble> Vt(mn * n);

    if (m >= n) {
      svd(psi, m, n, n, U.data(), S.data(), Vt.data());
    } else {
      // Transpose psi matrix.
      std::vector<Cdouble> psi_t(n * m);
      for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j) psi_t[j * m + i] = psi[i * n + j];
      std::vector<Cdouble> U_t(n * mn);
      std::vector<Cdouble> Vt_t(mn * m);
      svd(psi_t.data(), n, m, m, U_t.data(), S.data(), Vt_t.data());
    }

    double ent = 0.0;
    for (int k = 0; k < mn; ++k) {
      double s = S[k];
      if (s > 1e-20) {
        double s2 = s * s;
        ent -= s2 * std::log(s2);
      }
    }
    out[b - 1] = ent;
  }
}
