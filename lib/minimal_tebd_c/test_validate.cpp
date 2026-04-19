#include "validate.h"

#include <cmath>
#include <vector>

#include "gtest/gtest.h"
#include "model.h"

static constexpr double TOL_OBS = 1e-4;
static constexpr double TOL_FID = 1e-4;

TEST(Validate, SmallTFI_Order2) {
  // L=4 TFI at criticality, short evolution.
  // TEBD with order 2 should match ED to reasonable precision.
  int L = 4;
  BondModel model = tfi_chain(L, 1.0, 1.0);
  ValidationSummary s =
      validate_tebd_vs_ed(model, 0.05, 0.5, 64, 1e-12, 2, nullptr, false);

  // Check all time-step errors are small.
  for (size_t t = 0; t < s.err_Sz.size(); ++t) {
    EXPECT_LT(s.err_Sz[t], TOL_OBS) << "Sz error too large at step " << t;
    EXPECT_LT(s.err_Sx[t], TOL_OBS) << "Sx error too large at step " << t;
  }
  // Fidelity should be very close to 1.
  EXPECT_LT(s.fidelity_drop, TOL_FID);
}

TEST(Validate, SmallTFI_Order4) {
  // Order 4 should be even more accurate.
  int L = 4;
  BondModel model = tfi_chain(L, 1.0, 1.0);
  ValidationSummary s =
      validate_tebd_vs_ed(model, 0.05, 0.5, 64, 1e-12, 4, nullptr, false);

  for (size_t t = 0; t < s.err_Sz.size(); ++t) {
    EXPECT_LT(s.err_Sz[t], 1e-5);
    EXPECT_LT(s.err_Sx[t], 1e-5);
  }
  EXPECT_LT(s.fidelity_drop, 1e-6);
}

TEST(Validate, Order4BetterThanOrder2) {
  int L = 4;
  BondModel m2 = tfi_chain(L, 1.0, 1.0);
  ValidationSummary s2 =
      validate_tebd_vs_ed(m2, 0.1, 1.0, 64, 1e-12, 2, nullptr, false);

  BondModel m4 = tfi_chain(L, 1.0, 1.0);
  ValidationSummary s4 =
      validate_tebd_vs_ed(m4, 0.1, 1.0, 64, 1e-12, 4, nullptr, false);

  // Order 4 should have smaller final fidelity drop.
  EXPECT_LT(s4.fidelity_drop, s2.fidelity_drop);
}

TEST(Validate, NeelInitialState) {
  // Test with Neel initial state instead of all-zeros.
  int L = 4;
  int init[4] = {0, 1, 0, 1};
  BondModel model = tfi_chain(L, 1.0, 1.0);
  ValidationSummary s =
      validate_tebd_vs_ed(model, 0.05, 0.3, 64, 1e-12, 4, init, false);

  for (size_t t = 0; t < s.err_Sz.size(); ++t) {
    EXPECT_LT(s.err_Sz[t], TOL_OBS);
  }
  EXPECT_LT(s.fidelity_drop, TOL_FID);
}

TEST(Validate, EntropyMatchesED) {
  int L = 4;
  BondModel model = tfi_chain(L, 1.0, 1.0);
  ValidationSummary s =
      validate_tebd_vs_ed(model, 0.05, 0.5, 64, 1e-12, 4, nullptr, false);

  for (size_t t = 0; t < s.err_entropy.size(); ++t) {
    EXPECT_LT(s.err_entropy[t], 1e-5)
        << "Entropy error too large at step " << t;
  }
}

TEST(Validate, CorrelationMatchesED) {
  int L = 4;
  BondModel model = tfi_chain(L, 1.0, 1.0);
  ValidationSummary s =
      validate_tebd_vs_ed(model, 0.05, 0.3, 64, 1e-12, 4, nullptr, false);

  for (size_t t = 0; t < s.err_corr_XX.size(); ++t) {
    EXPECT_LT(s.err_corr_XX[t], TOL_OBS)
        << "XX correlation error too large at step " << t;
  }
}
