#include "tensor.h"

#include <cmath>

#include "gtest/gtest.h"

static constexpr double TOL = 1e-12;

TEST(Tensor, AllocAndAccess) {
  int sh[3] = {2, 3, 4};
  Tensor t(3, sh);
  EXPECT_EQ(t.ndim, 3);
  EXPECT_EQ(t.size, 24);
  EXPECT_EQ(t.shape[0], 2);
  EXPECT_EQ(t.shape[1], 3);
  EXPECT_EQ(t.shape[2], 4);

  t(1, 2, 3) = Cdouble(3.14, -2.71);
  EXPECT_NEAR(t(1, 2, 3).real(), 3.14, TOL);
  EXPECT_NEAR(t(1, 2, 3).imag(), -2.71, TOL);
}

TEST(Tensor, ReshapeRoundTrip) {
  int sh3[3] = {2, 3, 4};
  Tensor t(3, sh3);
  for (int i = 0; i < 24; ++i) t(i) = Cdouble(i, -i);

  int sh2[2] = {6, 4};
  t.reshape(2, sh2);
  EXPECT_EQ(t.ndim, 2);
  EXPECT_EQ(t.size, 24);

  // Element (1,2,3) in original = flat index 1*12 + 2*4 + 3 = 23.
  // In reshaped (6,4): row=23/4=5, col=23%4=3.
  EXPECT_NEAR(t(5, 3).real(), 23.0, TOL);

  // Reshape back.
  t.reshape(3, sh3);
  EXPECT_NEAR(t(1, 2, 3).real(), 23.0, TOL);
}

TEST(Tensor, Transpose) {
  int sh[3] = {2, 3, 4};
  Tensor t(3, sh);
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 4; ++k)
        t(i, j, k) = Cdouble(i * 100 + j * 10 + k, 0);

  int perm[3] = {2, 0, 1};  // new shape (4, 2, 3)
  Tensor tp = tensor_transpose(t, perm);
  EXPECT_EQ(tp.shape[0], 4);
  EXPECT_EQ(tp.shape[1], 2);
  EXPECT_EQ(tp.shape[2], 3);

  // tp(k, i, j) should equal t(i, j, k).
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 4; ++k)
        EXPECT_NEAR(tp(k, i, j).real(), t(i, j, k).real(), TOL);
}

TEST(Tensor, DeepCopy) {
  int sh[2] = {3, 3};
  Tensor t(2, sh);
  t(1, 2) = Cdouble(42.0, 0.0);

  Tensor copy = t;
  copy(1, 2) = Cdouble(0.0, 0.0);
  EXPECT_NEAR(t(1, 2).real(), 42.0, TOL);  // original unchanged
}

TEST(Tensor, MoveSemantics) {
  int sh[2] = {3, 3};
  Tensor t(2, sh);
  t(0, 0) = 99.0;

  Tensor moved = std::move(t);
  EXPECT_EQ(t.data, nullptr);
  EXPECT_NEAR(moved(0, 0).real(), 99.0, TOL);
}

TEST(Tensor, Scale) {
  int sh[2] = {2, 2};
  Tensor t(2, sh);
  t(0, 0) = 1.0;
  t(0, 1) = 2.0;
  t(1, 0) = 3.0;
  t(1, 1) = 4.0;
  t.scale(Cdouble(0, 1));  // multiply by i
  EXPECT_NEAR(t(0, 0).imag(), 1.0, TOL);
  EXPECT_NEAR(t(1, 1).imag(), 4.0, TOL);
  EXPECT_NEAR(t(1, 1).real(), 0.0, TOL);
}

TEST(Tensor, Identity) {
  Tensor I = Tensor::identity(3);
  EXPECT_NEAR(I(0, 0).real(), 1.0, TOL);
  EXPECT_NEAR(I(1, 1).real(), 1.0, TOL);
  EXPECT_NEAR(I(2, 2).real(), 1.0, TOL);
  EXPECT_NEAR(I(0, 1).real(), 0.0, TOL);
}

TEST(Tensor, Trace) {
  Tensor I = Tensor::identity(4);
  EXPECT_NEAR(std::real(tensor_trace(I)), 4.0, TOL);
}

TEST(Tensor, PreAllocCapacity) {
  int sh[3] = {1, 2, 1};
  Tensor t(3, sh, 100 * 2 * 100);
  EXPECT_EQ(t.size, 2);
  EXPECT_EQ(t.capacity, 20000);
  // Reshape to larger size within capacity.
  int sh2[3] = {2, 2, 2};
  t.reshape(3, sh2);
  EXPECT_EQ(t.size, 8);
}
