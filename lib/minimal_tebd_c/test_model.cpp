#include "model.h"

#include <cmath>

#include "gtest/gtest.h"
#include "linalg.h"
#include "mps.h"

static constexpr double TOL = 1e-10;

TEST(Model, TFIChainDims) {
  BondModel model = tfi_chain(6, 1.0, 1.0);
  EXPECT_EQ(model.L, 6);
  EXPECT_EQ(model.d, 2);
  // Should have L-1 = 5 bond Hamiltonians.
  for (int b = 0; b < 5; ++b) {
    EXPECT_EQ(model.H_bonds[b].ndim, 4);
    EXPECT_EQ(model.H_bonds[b].shape[0], 2);
    EXPECT_EQ(model.H_bonds[b].shape[1], 2);
    EXPECT_EQ(model.H_bonds[b].shape[2], 2);
    EXPECT_EQ(model.H_bonds[b].shape[3], 2);
  }
}

TEST(Model, TFIChainHermitian) {
  // Each H_bond as 4x4 matrix should be Hermitian.
  BondModel model = tfi_chain(4, 1.0, 1.5);
  for (int b = 0; b < 3; ++b) {
    Cdouble* h = model.H_bonds[b].data;
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 4; ++j)
        EXPECT_NEAR(std::abs(h[i * 4 + j] - std::conj(h[j * 4 + i])), 0.0,
                    TOL);
  }
}

TEST(Model, TFIChainZeroField) {
  // g=0: H_bond = -J * X_i X_{i+1}, no Z field.
  BondModel model = tfi_chain(3, 1.0, 0.0);
  Cdouble* h = model.H_bonds[0].data;  // -1 * kron(X, X)
  // kron(X,X) = [[0,0,0,1],[0,0,1,0],[0,1,0,0],[1,0,0,0]]
  EXPECT_NEAR(std::real(h[0 * 4 + 3]), -1.0, TOL);
  EXPECT_NEAR(std::real(h[1 * 4 + 2]), -1.0, TOL);
  EXPECT_NEAR(std::real(h[2 * 4 + 1]), -1.0, TOL);
  EXPECT_NEAR(std::real(h[3 * 4 + 0]), -1.0, TOL);
  EXPECT_NEAR(std::abs(h[0 * 4 + 0]), 0.0, TOL);
}

TEST(Model, TFIChainFieldSum) {
  // For L=3, the on-site Z terms should sum correctly.
  // Bond 0: site 0 gets full g, site 1 gets 0.5*g (interior).
  // Bond 1: site 1 gets 0.5*g (interior), site 2 gets full g.
  // So summing bond 0 and bond 1 diagonal Z contributions:
  // site 0: g, site 1: 0.5g + 0.5g = g, site 2: g.
  // This is the correct splitting for uniform field.
  BondModel model = tfi_chain(3, 0.0, 1.0);
  // With J=0, H_bond = -gL*kron(Z,I) - gR*kron(I,Z)
  // kron(Z,I) = diag(1,1,-1,-1), kron(I,Z) = diag(1,-1,1,-1)
  Cdouble* h0 = model.H_bonds[0].data;
  Cdouble* h1 = model.H_bonds[1].data;
  // Bond 0: gL=1.0 (site 0, first bond), gR=0.5 (site 1, interior)
  // H0_diag = -1*{1,1,-1,-1} - 0.5*{1,-1,1,-1} = {-1.5, -0.5, 0.5, 1.5}
  EXPECT_NEAR(std::real(h0[0 * 4 + 0]), -1.5, TOL);
  EXPECT_NEAR(std::real(h0[1 * 4 + 1]), -0.5, TOL);
  EXPECT_NEAR(std::real(h0[2 * 4 + 2]), 0.5, TOL);
  EXPECT_NEAR(std::real(h0[3 * 4 + 3]), 1.5, TOL);
}

TEST(Model, TFIChainMoveSemantics) {
  BondModel model = tfi_chain(4, 1.0, 1.0);
  BondModel moved = std::move(model);
  EXPECT_EQ(moved.L, 4);
  EXPECT_EQ(model.H_bonds, nullptr);
}
