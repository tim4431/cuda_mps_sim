#include "linalg.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <vector>

// ===========================================================================
// GEMM: C = alpha * A @ B + beta * C
// ===========================================================================

void zgemm(const Cdouble* A, int lda, const Cdouble* B, int ldb, Cdouble* C,
           int ldc, int m, int n, int k, Cdouble alpha, Cdouble beta) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      C[i * ldc + j] *= beta;
    }
    // i,p,j loop order for cache-friendly B access.
    for (int p = 0; p < k; ++p) {
      Cdouble a_ip = alpha * A[i * lda + p];
      for (int j = 0; j < n; ++j) {
        C[i * ldc + j] += a_ip * B[p * ldb + j];
      }
    }
  }
}

// ===========================================================================
// GEMV: y = alpha * A @ x + beta * y
// ===========================================================================

void zgemv(const Cdouble* A, int lda, const Cdouble* x, Cdouble* y, int m,
           int n, Cdouble alpha, Cdouble beta) {
  for (int i = 0; i < m; ++i) {
    Cdouble s = 0.0;
    for (int j = 0; j < n; ++j) {
      s += A[i * lda + j] * x[j];
    }
    y[i] = alpha * s + beta * y[i];
  }
}

// ===========================================================================
// SVD: One-sided Jacobi
// ===========================================================================

// Helper: compute complex dot product x^H * y for columns of length m.
static Cdouble col_dot(const Cdouble* W, int ldw, int m, int p, int q) {
  Cdouble s = 0.0;
  for (int i = 0; i < m; ++i) {
    s += std::conj(W[i * ldw + p]) * W[i * ldw + q];
  }
  return s;
}

// Helper: apply complex Jacobi rotation to columns p, q.
// Rotation: R = [[c, s_r * phase_conj],
//                [-s_r, c * phase_conj]]
// where phase_conj = e^{-i*phi} removes the phase of gamma.
static void apply_jacobi_rotation(Cdouble* M, int ldm, int rows, int p, int q,
                                  double c, double s_r, Cdouble phase_conj) {
  for (int i = 0; i < rows; ++i) {
    Cdouble mp = M[i * ldm + p];
    Cdouble mq = M[i * ldm + q];
    M[i * ldm + p] = c * mp + s_r * (phase_conj * mq);
    M[i * ldm + q] = -s_r * mp + c * (phase_conj * mq);
  }
}

void svd(const Cdouble* A, int m, int n, int lda, Cdouble* U, double* S,
         Cdouble* Vt) {
  assert(m >= n);
  const double tol = 1e-15;
  const int max_sweeps = 100;

  // W = copy of A (m x n), row-major with leading dim n.
  std::vector<Cdouble> W(m * n);
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < n; ++j) W[i * n + j] = A[i * lda + j];

  // V = I (n x n).
  std::vector<Cdouble> V(n * n, 0.0);
  for (int i = 0; i < n; ++i) V[i * n + i] = 1.0;

  for (int sweep = 0; sweep < max_sweeps; ++sweep) {
    double off_norm = 0.0;
    for (int p = 0; p < n; ++p) {
      for (int q = p + 1; q < n; ++q) {
        // Gram matrix elements for columns p, q.
        double alpha_pp = std::real(col_dot(W.data(), n, m, p, p));
        double beta_qq = std::real(col_dot(W.data(), n, m, q, q));
        Cdouble gamma_pq = col_dot(W.data(), n, m, p, q);

        double g = std::abs(gamma_pq);
        off_norm += g * g;

        if (g < tol * std::sqrt(alpha_pp * beta_qq)) continue;

        // Compute Jacobi rotation that zeros gamma_pq.
        // We diagonalize the 2x2 Hermitian Gram sub-matrix
        // [[alpha, gamma], [conj(gamma), beta]].
        // First, remove the phase of gamma.
        double phase_angle = std::arg(gamma_pq);
        Cdouble phase_conj = std::exp(Cdouble(0, -phase_angle));
        // Now gamma_real = |gamma_pq|.
        double gamma_real = g;

        // Jacobi rotation for 2x2 real symmetric [[alpha, gamma_r],
        //                                          [gamma_r, beta]].
        double tau = (beta_qq - alpha_pp) / (2.0 * gamma_real);
        double t =
            (tau >= 0 ? -1.0 : 1.0) / (std::abs(tau) + std::sqrt(1 + tau * tau));
        double c = 1.0 / std::sqrt(1 + t * t);
        double s_r = c * t;  // real rotation parameter

        apply_jacobi_rotation(W.data(), n, m, p, q, c, s_r, phase_conj);
        apply_jacobi_rotation(V.data(), n, n, p, q, c, s_r, phase_conj);
      }
    }
    if (off_norm < tol * tol) break;
  }

  // Singular values = column norms of W; U = W / sigma.
  std::vector<int> order(n);
  for (int j = 0; j < n; ++j) {
    double nrm = 0.0;
    for (int i = 0; i < m; ++i) nrm += std::norm(W[i * n + j]);
    S[j] = std::sqrt(nrm);
    order[j] = j;
  }

  // Sort by descending singular value.
  std::sort(order.begin(), order.end(),
            [&S](int a, int b) { return S[a] > S[b]; });

  std::vector<double> S_sorted(n);
  for (int j = 0; j < n; ++j) S_sorted[j] = S[order[j]];
  std::copy(S_sorted.begin(), S_sorted.end(), S);

  // U = W * diag(1/sigma), with columns reordered.
  for (int j = 0; j < n; ++j) {
    int oj = order[j];
    double inv_s = (S[j] > 1e-30) ? 1.0 / S[j] : 0.0;
    for (int i = 0; i < m; ++i) {
      U[i * n + j] = W[i * n + oj] * inv_s;
    }
  }

  // Vt = V^H, with columns reordered.
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      Vt[i * n + j] = std::conj(V[j * n + order[i]]);
    }
  }
}

// ===========================================================================
// Matrix exponential: scaling and squaring with Taylor series
// ===========================================================================

void matrix_expm(const Cdouble* M, int n, Cdouble* out) {
  int n2 = n * n;

  // Compute 1-norm of M.
  double norm1 = 0.0;
  for (int j = 0; j < n; ++j) {
    double col_sum = 0.0;
    for (int i = 0; i < n; ++i) col_sum += std::abs(M[i * n + j]);
    norm1 = std::max(norm1, col_sum);
  }

  // Choose s so that ||M / 2^s|| < 0.5.
  int s = 0;
  double scale = norm1;
  while (scale > 0.5) {
    scale /= 2.0;
    ++s;
  }
  double denom = std::pow(2.0, s);

  // Scaled matrix Ms = M / 2^s.
  std::vector<Cdouble> Ms(n2);
  for (int i = 0; i < n2; ++i) Ms[i] = M[i] / denom;

  // Taylor series: exp(Ms) ≈ I + Ms + Ms^2/2! + ... + Ms^p/p!
  // We accumulate: term = Ms^k / k!, result = sum of terms.
  std::vector<Cdouble> result(n2, 0.0);
  for (int i = 0; i < n; ++i) result[i * n + i] = 1.0;

  std::vector<Cdouble> term(n2, 0.0);
  for (int i = 0; i < n; ++i) term[i * n + i] = 1.0;

  std::vector<Cdouble> tmp(n2);
  const int p = 20;  // enough terms for double precision
  for (int k = 1; k <= p; ++k) {
    // term = term @ Ms / k
    std::fill(tmp.begin(), tmp.end(), 0.0);
    zgemm(term.data(), n, Ms.data(), n, tmp.data(), n, n, n, n);
    for (int i = 0; i < n2; ++i) term[i] = tmp[i] / static_cast<double>(k);
    for (int i = 0; i < n2; ++i) result[i] += term[i];

    // Early exit if term is negligible.
    double term_norm = 0.0;
    for (int i = 0; i < n2; ++i) term_norm += std::norm(term[i]);
    if (term_norm < 1e-32) break;
  }

  // Repeated squaring: result = result^(2^s).
  for (int k = 0; k < s; ++k) {
    std::fill(tmp.begin(), tmp.end(), 0.0);
    zgemm(result.data(), n, result.data(), n, tmp.data(), n, n, n, n);
    std::copy(tmp.begin(), tmp.end(), result.begin());
  }

  std::copy(result.begin(), result.end(), out);
}

// ===========================================================================
// Tensor contraction
// ===========================================================================

Tensor tensor_contract(const Tensor& A, const int axes_a[], const Tensor& B,
                       const int axes_b[], int n_contract) {
  // Build permutation for A: free axes first, then contracted axes.
  int a_free[MAX_NDIM], a_nfree = 0;
  int b_free[MAX_NDIM], b_nfree = 0;

  // Mark contracted axes.
  bool a_contracted[MAX_NDIM] = {};
  bool b_contracted[MAX_NDIM] = {};
  for (int c = 0; c < n_contract; ++c) {
    a_contracted[axes_a[c]] = true;
    b_contracted[axes_b[c]] = true;
  }

  for (int k = 0; k < A.ndim; ++k)
    if (!a_contracted[k]) a_free[a_nfree++] = k;
  for (int k = 0; k < B.ndim; ++k)
    if (!b_contracted[k]) b_free[b_nfree++] = k;

  // Compute contraction dimension.
  int K = 1;
  for (int c = 0; c < n_contract; ++c) K *= A.shape[axes_a[c]];

  // M = product of A's free dims, N = product of B's free dims.
  int M = 1, N = 1;
  for (int i = 0; i < a_nfree; ++i) M *= A.shape[a_free[i]];
  for (int i = 0; i < b_nfree; ++i) N *= B.shape[b_free[i]];

  // Build permutation for A: [free..., contracted...].
  int perm_a[MAX_NDIM];
  for (int i = 0; i < a_nfree; ++i) perm_a[i] = a_free[i];
  for (int i = 0; i < n_contract; ++i) perm_a[a_nfree + i] = axes_a[i];

  // Build permutation for B: [contracted..., free...].
  int perm_b[MAX_NDIM];
  for (int i = 0; i < n_contract; ++i) perm_b[i] = axes_b[i];
  for (int i = 0; i < b_nfree; ++i) perm_b[n_contract + i] = b_free[i];

  // Transpose and reshape to 2D.
  Tensor At = tensor_transpose(A, perm_a);
  Tensor Bt = tensor_transpose(B, perm_b);

  int sh_a2[2] = {M, K};
  int sh_b2[2] = {K, N};
  At.reshape(2, sh_a2);
  Bt.reshape(2, sh_b2);

  // GEMM: C(M, N) = At(M, K) @ Bt(K, N).
  int sh_c[2] = {M, N};
  Tensor C(2, sh_c);
  zgemm(At.data, K, Bt.data, N, C.data, N, M, N, K);

  // Reshape C to the output shape: [A_free_dims..., B_free_dims...].
  int out_ndim = a_nfree + b_nfree;
  if (out_ndim == 0) out_ndim = 1;  // scalar -> rank-1 of size 1
  int out_shape[MAX_NDIM];
  int idx = 0;
  for (int i = 0; i < a_nfree; ++i) out_shape[idx++] = A.shape[a_free[i]];
  for (int i = 0; i < b_nfree; ++i) out_shape[idx++] = B.shape[b_free[i]];
  if (idx == 0) {
    out_shape[0] = 1;
    idx = 1;
  }
  C.reshape(idx, out_shape);
  return C;
}

// ===========================================================================
// Kronecker product
// ===========================================================================

void kron(const Cdouble* A, int ra, int ca, const Cdouble* B, int rb, int cb,
          Cdouble* C) {
  int cc = ca * cb;
  for (int ia = 0; ia < ra; ++ia) {
    for (int ja = 0; ja < ca; ++ja) {
      Cdouble a_val = A[ia * ca + ja];
      for (int ib = 0; ib < rb; ++ib) {
        for (int jb = 0; jb < cb; ++jb) {
          C[(ia * rb + ib) * cc + (ja * cb + jb)] = a_val * B[ib * cb + jb];
        }
      }
    }
  }
}
