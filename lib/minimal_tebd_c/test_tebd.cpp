#include "tebd.h"

#include <cmath>
#include <vector>

#include "gtest/gtest.h"
#include "mps.h"
#include "model.h"

static constexpr double TOL = 1e-6;

TEST(TEBD, OneStepNormPreserved) {
  // After one TEBD step, the MPS should remain normalized (|<psi|psi>| ~ 1).
  int L = 6;
  std::vector<int> init(L, 0);
  BondModel model = tfi_chain(L, 1.0, 1.0);
  MPS psi = MPS::product_state(L, init.data(), 2, 20);
  TEBDEngine eng(&psi, &model, 0.1, 2, 1, 20, 1e-12);

  eng.run();

  // Check normalization via state vector inner product.
  Tensor sv = psi.to_state_vector();
  Cdouble norm_sq = 0.0;
  for (int i = 0; i < sv.size; ++i)
    norm_sq += std::conj(sv(i)) * sv(i);
  EXPECT_NEAR(std::abs(norm_sq), 1.0, 1e-8);
}

TEST(TEBD, SzConservationShortTime) {
  // For very short times, total <Sz> should be approximately conserved.
  int L = 4;
  std::vector<int> init(L, 0);
  BondModel model = tfi_chain(L, 1.0, 1.0);
  MPS psi = MPS::product_state(L, init.data(), 2, 20);

  // Initial total Sz.
  std::vector<double> Sz0(L);
  psi.expect(PAULI_Z, Sz0.data());
  double total_Sz0 = 0.0;
  for (int i = 0; i < L; ++i) total_Sz0 += Sz0[i];

  // One small step.
  TEBDEngine eng(&psi, &model, 0.01, 4, 1, 20, 1e-12);
  eng.run();

  std::vector<double> Sz1(L);
  psi.expect(PAULI_Z, Sz1.data());
  double total_Sz1 = 0.0;
  for (int i = 0; i < L; ++i) total_Sz1 += Sz1[i];

  // Sz is not conserved in TFI, but for dt=0.01 the change is O(dt).
  EXPECT_NEAR(total_Sz0, total_Sz1, 0.1);
}

TEST(TEBD, EntropyGrows) {
  // Starting from product state, entropy should grow under TFI evolution.
  int L = 6;
  std::vector<int> init(L, 0);
  BondModel model = tfi_chain(L, 1.0, 1.0);
  MPS psi = MPS::product_state(L, init.data(), 2, 40);
  TEBDEngine eng(&psi, &model, 0.1, 2, 10, 40, 1e-12);

  eng.run();  // 10 * 0.1 = 1.0 time units

  std::vector<double> S(L - 1);
  psi.entropy(S.data());
  // Mid-chain entropy should be nonzero after evolution.
  EXPECT_GT(S[L / 2 - 1], 0.01);
}

TEST(TEBD, BondDimGrows) {
  // Bond dimension should increase from 1 (product state) after evolution.
  int L = 6;
  std::vector<int> init(L, 0);
  BondModel model = tfi_chain(L, 1.0, 1.0);
  MPS psi = MPS::product_state(L, init.data(), 2, 40);
  EXPECT_EQ(psi.max_bond_dim(), 1);

  TEBDEngine eng(&psi, &model, 0.1, 2, 5, 40, 1e-12);
  eng.run();
  EXPECT_GT(psi.max_bond_dim(), 1);
}

TEST(TEBD, TrotterOrders) {
  // Higher-order Trotter should give smaller state-vector error.
  // Compare TEBD state vectors to each other: order 2 and 4 should be
  // closer than order 1 and 4.
  int L = 4;
  std::vector<int> init(L, 0);
  double dt = 0.1;
  int steps = 5;

  // Order 1
  BondModel m1 = tfi_chain(L, 1.0, 1.0);
  MPS psi1 = MPS::product_state(L, init.data(), 2, 20);
  TEBDEngine eng1(&psi1, &m1, dt, 1, steps, 20, 1e-12);
  eng1.run();

  // Order 2
  BondModel m2 = tfi_chain(L, 1.0, 1.0);
  MPS psi2 = MPS::product_state(L, init.data(), 2, 20);
  TEBDEngine eng2(&psi2, &m2, dt, 2, steps, 20, 1e-12);
  eng2.run();

  // Order 4
  BondModel m4 = tfi_chain(L, 1.0, 1.0);
  MPS psi4 = MPS::product_state(L, init.data(), 2, 20);
  TEBDEngine eng4(&psi4, &m4, dt, 4, steps, 20, 1e-12);
  eng4.run();

  // Compare full state vectors: |1 - |<psi_i|psi_4>||
  Tensor sv1 = psi1.to_state_vector();
  Tensor sv2 = psi2.to_state_vector();
  Tensor sv4 = psi4.to_state_vector();
  int dim = sv4.size;

  Cdouble o14 = 0, o24 = 0;
  for (int i = 0; i < dim; ++i) {
    o14 += std::conj(sv1(i)) * sv4(i);
    o24 += std::conj(sv2(i)) * sv4(i);
  }
  double infid_14 = 1.0 - std::abs(o14);
  double infid_24 = 1.0 - std::abs(o24);
  // Order 2 should be closer to order 4 than order 1 is.
  EXPECT_LT(infid_24, infid_14);
}

TEST(TEBD, SvdTruncateBasic) {
  // theta shape (1, 2, 2, 1), product of two |0> states.
  int sh[4] = {1, 2, 2, 1};
  Tensor theta(4, sh);
  theta.zero();
  theta(0, 0, 0, 0) = 1.0;  // |00>

  Tensor A, B;
  double S[2];
  double eps;
  svd_truncate(theta, 10, 1e-12, A, S, B, eps);

  EXPECT_EQ(A.shape[0], 1);
  EXPECT_EQ(A.shape[1], 2);
  EXPECT_EQ(A.shape[2], 1);  // only 1 nonzero singular value
  EXPECT_NEAR(S[0], 1.0, 1e-8);
  EXPECT_NEAR(eps, 0.0, 1e-8);
}
