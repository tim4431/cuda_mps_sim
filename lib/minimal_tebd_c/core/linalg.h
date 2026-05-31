#ifndef LINALG_H_
#define LINALG_H_

#include "tensor.h"

// ---------------------------------------------------------------------------
// Complex matrix multiply: C = alpha * A @ B + beta * C
// A is (m x k), B is (k x n), C is (m x n). Row-major, leading dim = ld*.
// ---------------------------------------------------------------------------
void zgemm(const Cdouble* A, int lda, const Cdouble* B, int ldb, Cdouble* C,
           int ldc, int m, int n, int k, Cdouble alpha = 1.0,
           Cdouble beta = 0.0);

// ---------------------------------------------------------------------------
// Complex matrix-vector multiply: y = alpha * A @ x + beta * y
// A is (m x n) row-major.
// ---------------------------------------------------------------------------
void zgemv(const Cdouble* A, int lda, const Cdouble* x, Cdouble* y, int m,
           int n, Cdouble alpha = 1.0, Cdouble beta = 0.0);

// ---------------------------------------------------------------------------
// One-sided Jacobi SVD: A(m x n) = U(m x n) * diag(S) * Vt(n x n)
// S is real, length min(m,n). A is not modified. U, S, Vt must be
// pre-allocated. Singular values returned in descending order.
// ---------------------------------------------------------------------------
void svd(const Cdouble* A, int m, int n, int lda, Cdouble* U, double* S,
         Cdouble* Vt);

// ---------------------------------------------------------------------------
// Matrix exponential via scaling and squaring with Taylor series.
// M is n x n row-major. Result written to out (n x n, pre-allocated).
// ---------------------------------------------------------------------------
void matrix_expm(const Cdouble* M, int n, Cdouble* out);

// ---------------------------------------------------------------------------
// Tensor contraction: contracts A's axes_a[0..n_contract-1] with
// B's axes_b[0..n_contract-1]. Returns a new tensor whose rank is
// A.ndim + B.ndim - 2*n_contract.
// ---------------------------------------------------------------------------
Tensor tensor_contract(const Tensor& A, const int axes_a[], const Tensor& B,
                       const int axes_b[], int n_contract);

// ---------------------------------------------------------------------------
// Kronecker product: C = A ⊗ B. C is (ra*rb x ca*cb), pre-allocated.
// ---------------------------------------------------------------------------
void kron(const Cdouble* A, int ra, int ca, const Cdouble* B, int rb, int cb,
          Cdouble* C);

#endif  // LINALG_H_
