#include "mps.h"

#include <cmath>
#include <vector>

#include "gtest/gtest.h"
#include "linalg.h"

static constexpr double TOL = 1e-10;

TEST(MPS, ProductStateAllUp) {
  // |0,0,...,0> = all spin-up.
  int L = 6;
  std::vector<int> states(L, 0);
  MPS psi = MPS::product_state(L, states.data(), 2, 10);
  EXPECT_EQ(psi.L, L);
  EXPECT_EQ(psi.d, 2);
  EXPECT_EQ(psi.max_bond_dim(), 1);

  // <Sz> should be +1 everywhere.
  std::vector<double> Sz(L);
  psi.expect(PAULI_Z, Sz.data());
  for (int i = 0; i < L; ++i) EXPECT_NEAR(Sz[i], 1.0, TOL);
}

TEST(MPS, ProductStateNeel) {
  // Neel state |0,1,0,1,...>
  int L = 6;
  std::vector<int> states(L);
  for (int i = 0; i < L; ++i) states[i] = i % 2;
  MPS psi = MPS::product_state(L, states.data(), 2, 10);

  std::vector<double> Sz(L);
  psi.expect(PAULI_Z, Sz.data());
  for (int i = 0; i < L; ++i) {
    double expected = (i % 2 == 0) ? 1.0 : -1.0;
    EXPECT_NEAR(Sz[i], expected, TOL);
  }
}

TEST(MPS, ProductStateEntropy) {
  // Product state has zero entanglement entropy.
  int L = 6;
  std::vector<int> states(L, 0);
  MPS psi = MPS::product_state(L, states.data(), 2, 10);

  std::vector<double> S(L - 1);
  psi.entropy(S.data());
  for (int i = 0; i < L - 1; ++i) EXPECT_NEAR(S[i], 0.0, TOL);
}

TEST(MPS, ProductStateSxZero) {
  // All spin-up: <Sx> = 0 everywhere.
  int L = 4;
  std::vector<int> states(L, 0);
  MPS psi = MPS::product_state(L, states.data(), 2, 10);

  std::vector<double> Sx(L);
  psi.expect(PAULI_X, Sx.data());
  for (int i = 0; i < L; ++i) EXPECT_NEAR(Sx[i], 0.0, TOL);
}

TEST(MPS, Theta1Shape) {
  int L = 4;
  std::vector<int> states(L, 0);
  MPS psi = MPS::product_state(L, states.data(), 2, 10);

  Tensor t1 = psi.theta1(0);
  EXPECT_EQ(t1.ndim, 3);
  EXPECT_EQ(t1.shape[0], 1);  // chi_L = 1 for product state
  EXPECT_EQ(t1.shape[1], 2);  // d = 2
  EXPECT_EQ(t1.shape[2], 1);  // chi_R = 1 for product state
}

TEST(MPS, Theta2Shape) {
  int L = 4;
  std::vector<int> states(L, 0);
  MPS psi = MPS::product_state(L, states.data(), 2, 10);

  Tensor t2 = psi.theta2(0);
  EXPECT_EQ(t2.ndim, 4);
  EXPECT_EQ(t2.shape[0], 1);
  EXPECT_EQ(t2.shape[1], 2);
  EXPECT_EQ(t2.shape[2], 2);
  EXPECT_EQ(t2.shape[3], 1);
}

TEST(MPS, CorrDiagProduct) {
  // For product state |0...0>, <XX>_{ii} = <X^2>_i = <I>_i = 1.
  int L = 4;
  std::vector<int> states(L, 0);
  MPS psi = MPS::product_state(L, states.data(), 2, 10);

  std::vector<double> C(L * L);
  psi.corr(PAULI_X, PAULI_X, C.data());
  for (int i = 0; i < L; ++i) EXPECT_NEAR(C[i * L + i], 1.0, TOL);
  // Off-diagonal: <X_i X_j> = <X_i><X_j> = 0 for product state.
  for (int i = 0; i < L; ++i)
    for (int j = 0; j < L; ++j)
      if (i != j) EXPECT_NEAR(C[i * L + j], 0.0, TOL);
}

TEST(MPS, ToStateVectorProduct) {
  // |0,0> = (1, 0, 0, 0) in computational basis.
  int L = 2;
  std::vector<int> states(L, 0);
  MPS psi = MPS::product_state(L, states.data(), 2, 10);

  Tensor sv = psi.to_state_vector();
  EXPECT_EQ(sv.size, 4);
  EXPECT_NEAR(std::abs(sv(0)), 1.0, TOL);
  EXPECT_NEAR(std::abs(sv(1)), 0.0, TOL);
  EXPECT_NEAR(std::abs(sv(2)), 0.0, TOL);
  EXPECT_NEAR(std::abs(sv(3)), 0.0, TOL);
}

TEST(MPS, ToStateVectorNeel) {
  // |0,1> = (0, 1, 0, 0) in computational basis.
  int L = 2;
  int states[2] = {0, 1};
  MPS psi = MPS::product_state(L, states, 2, 10);

  Tensor sv = psi.to_state_vector();
  EXPECT_EQ(sv.size, 4);
  EXPECT_NEAR(std::abs(sv(0)), 0.0, TOL);
  EXPECT_NEAR(std::abs(sv(1)), 1.0, TOL);
  EXPECT_NEAR(std::abs(sv(2)), 0.0, TOL);
  EXPECT_NEAR(std::abs(sv(3)), 0.0, TOL);
}

TEST(MPS, BondDims) {
  int L = 4;
  std::vector<int> states(L, 0);
  MPS psi = MPS::product_state(L, states.data(), 2, 10);

  std::vector<int> bd(L - 1);
  psi.bond_dims(bd.data());
  for (int i = 0; i < L - 1; ++i) EXPECT_EQ(bd[i], 1);
}
