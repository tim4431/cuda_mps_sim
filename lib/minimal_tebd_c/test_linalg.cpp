#include "linalg.h"

#include <cmath>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"

static constexpr double TOL = 1e-10;

// ===========================================================================
// GEMM
// ===========================================================================

TEST(Linalg, GemmIdentity) {
  // C = I @ B should equal B.
  Cdouble I[4] = {1, 0, 0, 1};
  Cdouble B[4] = {1, 2, 3, 4};
  Cdouble C[4] = {};
  zgemm(I, 2, B, 2, C, 2, 2, 2, 2);
  for (int i = 0; i < 4; ++i) EXPECT_NEAR(std::abs(C[i] - B[i]), 0.0, TOL);
}

TEST(Linalg, GemmKnownProduct) {
  // [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]]
  Cdouble A[4] = {1, 2, 3, 4};
  Cdouble B[4] = {5, 6, 7, 8};
  Cdouble C[4] = {};
  zgemm(A, 2, B, 2, C, 2, 2, 2, 2);
  EXPECT_NEAR(std::real(C[0]), 19.0, TOL);
  EXPECT_NEAR(std::real(C[1]), 22.0, TOL);
  EXPECT_NEAR(std::real(C[2]), 43.0, TOL);
  EXPECT_NEAR(std::real(C[3]), 50.0, TOL);
}

TEST(Linalg, GemmAlphaBeta) {
  Cdouble A[4] = {1, 0, 0, 1};
  Cdouble B[4] = {2, 0, 0, 2};
  Cdouble C[4] = {1, 1, 1, 1};
  // C = 2*I*B + 3*C = [[4+3, 0+3],[0+3, 4+3]] = [[7,3],[3,7]]
  zgemm(A, 2, B, 2, C, 2, 2, 2, 2, Cdouble(2.0), Cdouble(3.0));
  EXPECT_NEAR(std::real(C[0]), 7.0, TOL);
  EXPECT_NEAR(std::real(C[1]), 3.0, TOL);
  EXPECT_NEAR(std::real(C[2]), 3.0, TOL);
  EXPECT_NEAR(std::real(C[3]), 7.0, TOL);
}

TEST(Linalg, GemmRectangular) {
  // A(2x3) @ B(3x2) = C(2x2)
  Cdouble A[6] = {1, 2, 3, 4, 5, 6};
  Cdouble B[6] = {7, 8, 9, 10, 11, 12};
  Cdouble C[4] = {};
  zgemm(A, 3, B, 2, C, 2, 2, 2, 3);
  // C[0,0] = 1*7 + 2*9 + 3*11 = 58
  // C[0,1] = 1*8 + 2*10 + 3*12 = 64
  // C[1,0] = 4*7 + 5*9 + 6*11 = 139
  // C[1,1] = 4*8 + 5*10 + 6*12 = 154
  EXPECT_NEAR(std::real(C[0]), 58.0, TOL);
  EXPECT_NEAR(std::real(C[1]), 64.0, TOL);
  EXPECT_NEAR(std::real(C[2]), 139.0, TOL);
  EXPECT_NEAR(std::real(C[3]), 154.0, TOL);
}

TEST(Linalg, GemmComplex) {
  // (1+i)*(1-i) = 2
  Cdouble A[1] = {Cdouble(1, 1)};
  Cdouble B[1] = {Cdouble(1, -1)};
  Cdouble C[1] = {};
  zgemm(A, 1, B, 1, C, 1, 1, 1, 1);
  EXPECT_NEAR(std::real(C[0]), 2.0, TOL);
  EXPECT_NEAR(std::imag(C[0]), 0.0, TOL);
}

// ===========================================================================
// GEMV
// ===========================================================================

TEST(Linalg, GemvBasic) {
  Cdouble A[4] = {1, 2, 3, 4};
  Cdouble x[2] = {1, 1};
  Cdouble y[2] = {};
  zgemv(A, 2, x, y, 2, 2);
  EXPECT_NEAR(std::real(y[0]), 3.0, TOL);
  EXPECT_NEAR(std::real(y[1]), 7.0, TOL);
}

// ===========================================================================
// SVD
// ===========================================================================

TEST(Linalg, SvdDiagonal) {
  // SVD of diag(3, 2, 1) should give singular values {3, 2, 1}.
  Cdouble A[9] = {3, 0, 0, 0, 2, 0, 0, 0, 1};
  Cdouble U[9], Vt[9];
  double S[3];
  svd(A, 3, 3, 3, U, S, Vt);
  EXPECT_NEAR(S[0], 3.0, TOL);
  EXPECT_NEAR(S[1], 2.0, TOL);
  EXPECT_NEAR(S[2], 1.0, TOL);
}

TEST(Linalg, SvdReconstruct) {
  // Random-ish 3x3 matrix. Check A ≈ U @ diag(S) @ Vt.
  Cdouble A[9] = {Cdouble(1, 2),  Cdouble(3, -1), Cdouble(0, 1),
                   Cdouble(-2, 0), Cdouble(1, 1),  Cdouble(4, -3),
                   Cdouble(0, 0),  Cdouble(2, 2),  Cdouble(-1, 1)};
  Cdouble U[9], Vt[9];
  double S[3];
  svd(A, 3, 3, 3, U, S, Vt);

  // Reconstruct: R = U @ diag(S) @ Vt.
  Cdouble US[9];
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) US[i * 3 + j] = U[i * 3 + j] * S[j];
  Cdouble R[9] = {};
  zgemm(US, 3, Vt, 3, R, 3, 3, 3, 3);

  for (int i = 0; i < 9; ++i)
    EXPECT_NEAR(std::abs(R[i] - A[i]), 0.0, 1e-8);
}

TEST(Linalg, SvdRectangular) {
  // 4x3 matrix, should get 3 singular values.
  Cdouble A[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  Cdouble U[12], Vt[9];
  double S[3];
  svd(A, 4, 3, 3, U, S, Vt);

  // Reconstruct.
  Cdouble US[12];
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 3; ++j) US[i * 3 + j] = U[i * 3 + j] * S[j];
  Cdouble R[12] = {};
  zgemm(US, 3, Vt, 3, R, 3, 4, 3, 3);

  for (int i = 0; i < 12; ++i)
    EXPECT_NEAR(std::abs(R[i] - A[i]), 0.0, 1e-8);
}

TEST(Linalg, SvdUnitary) {
  // Check U^H @ U = I and Vt @ Vt^H = I.
  Cdouble A[9] = {Cdouble(1, 2),  Cdouble(3, -1), Cdouble(0, 1),
                   Cdouble(-2, 0), Cdouble(1, 1),  Cdouble(4, -3),
                   Cdouble(0, 0),  Cdouble(2, 2),  Cdouble(-1, 1)};
  Cdouble U[9], Vt[9];
  double S[3];
  svd(A, 3, 3, 3, U, S, Vt);

  // U^H @ U
  Cdouble UhU[9] = {};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 3; ++k)
        UhU[i * 3 + j] += std::conj(U[k * 3 + i]) * U[k * 3 + j];

  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      EXPECT_NEAR(std::abs(UhU[i * 3 + j] - (i == j ? 1.0 : 0.0)), 0.0,
                  1e-8);
}

// ===========================================================================
// Matrix exponential
// ===========================================================================

TEST(Linalg, ExpmZero) {
  // exp(0) = I
  Cdouble M[4] = {0, 0, 0, 0};
  Cdouble out[4];
  matrix_expm(M, 2, out);
  EXPECT_NEAR(std::real(out[0]), 1.0, TOL);
  EXPECT_NEAR(std::abs(out[1]), 0.0, TOL);
  EXPECT_NEAR(std::abs(out[2]), 0.0, TOL);
  EXPECT_NEAR(std::real(out[3]), 1.0, TOL);
}

TEST(Linalg, ExpmDiagonal) {
  // exp(diag(1, 2)) = diag(e, e^2)
  Cdouble M[4] = {1, 0, 0, 2};
  Cdouble out[4];
  matrix_expm(M, 2, out);
  EXPECT_NEAR(std::real(out[0]), std::exp(1.0), 1e-8);
  EXPECT_NEAR(std::abs(out[1]), 0.0, TOL);
  EXPECT_NEAR(std::abs(out[2]), 0.0, TOL);
  EXPECT_NEAR(std::real(out[3]), std::exp(2.0), 1e-8);
}

TEST(Linalg, ExpmPauliX) {
  // exp(i*t*X) for Pauli X = [[0,1],[1,0]]
  // = cos(t)*I + i*sin(t)*X
  double t = 0.3;
  Cdouble M[4] = {0, Cdouble(0, t), Cdouble(0, t), 0};
  Cdouble out[4];
  matrix_expm(M, 2, out);
  EXPECT_NEAR(std::real(out[0]), std::cos(t), 1e-8);
  EXPECT_NEAR(std::imag(out[1]), std::sin(t), 1e-8);
  EXPECT_NEAR(std::imag(out[2]), std::sin(t), 1e-8);
  EXPECT_NEAR(std::real(out[3]), std::cos(t), 1e-8);
}

TEST(Linalg, ExpmUnitary) {
  // exp(M) for anti-Hermitian M (M^H = -M) should be unitary.
  // M = [[0.5i, 1+0.3i], [-1+0.3i, 0.8i]]
  Cdouble H[4] = {Cdouble(0, 0.5), Cdouble(1, 0.3), Cdouble(-1, 0.3),
                   Cdouble(0, 0.8)};
  Cdouble out[4];
  matrix_expm(H, 2, out);
  // Check U @ U^H = I
  Cdouble UUh[4] = {};
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j)
      for (int k = 0; k < 2; ++k)
        UUh[i * 2 + j] += out[i * 2 + k] * std::conj(out[j * 2 + k]);
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j)
      EXPECT_NEAR(std::abs(UUh[i * 2 + j] - (i == j ? 1.0 : 0.0)), 0.0,
                  1e-8);
}

// ===========================================================================
// Tensor contraction
// ===========================================================================

TEST(Linalg, TensorContractMatMul) {
  // Matrix multiply as tensor contraction: A(2,3) @ B(3,4) = C(2,4)
  int shA[2] = {2, 3};
  Tensor A(2, shA);
  for (int i = 0; i < 6; ++i) A(i) = Cdouble(i + 1, 0);

  int shB[2] = {3, 4};
  Tensor B(2, shB);
  for (int i = 0; i < 12; ++i) B(i) = Cdouble(i + 1, 0);

  int ax_a[] = {1}, ax_b[] = {0};
  Tensor C = tensor_contract(A, ax_a, B, ax_b, 1);
  EXPECT_EQ(C.ndim, 2);
  EXPECT_EQ(C.shape[0], 2);
  EXPECT_EQ(C.shape[1], 4);

  // C[0,0] = A[0,:] @ B[:,0] = 1*1 + 2*5 + 3*9 = 38
  EXPECT_NEAR(std::real(C(0, 0)), 38.0, TOL);
}

TEST(Linalg, TensorContractInnerProduct) {
  // Inner product: contract all axes of two rank-1 tensors.
  int sh[1] = {3};
  Tensor A(1, sh);
  Tensor B(1, sh);
  for (int i = 0; i < 3; ++i) {
    A(i) = Cdouble(i + 1, 0);
    B(i) = Cdouble(i + 1, 0);
  }
  int ax[] = {0};
  Tensor C = tensor_contract(A, ax, B, ax, 1);
  // 1*1 + 2*2 + 3*3 = 14
  EXPECT_NEAR(std::real(C(0)), 14.0, TOL);
}

// ===========================================================================
// Kronecker product
// ===========================================================================

TEST(Linalg, KronIdentity) {
  // I2 kron I2 = I4
  Cdouble I2[4] = {1, 0, 0, 1};
  Cdouble C[16] = {};
  kron(I2, 2, 2, I2, 2, 2, C);
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j)
      EXPECT_NEAR(std::abs(C[i * 4 + j] - (i == j ? 1.0 : 0.0)), 0.0, TOL);
}

TEST(Linalg, KronPauliX) {
  // X kron X = [[0,0,0,1],[0,0,1,0],[0,1,0,0],[1,0,0,0]]
  Cdouble X[4] = {0, 1, 1, 0};
  Cdouble C[16] = {};
  kron(X, 2, 2, X, 2, 2, C);
  EXPECT_NEAR(std::real(C[0 * 4 + 3]), 1.0, TOL);
  EXPECT_NEAR(std::real(C[1 * 4 + 2]), 1.0, TOL);
  EXPECT_NEAR(std::real(C[2 * 4 + 1]), 1.0, TOL);
  EXPECT_NEAR(std::real(C[3 * 4 + 0]), 1.0, TOL);
  EXPECT_NEAR(std::abs(C[0 * 4 + 0]), 0.0, TOL);
}
