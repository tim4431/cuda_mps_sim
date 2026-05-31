#ifndef TENSOR_H_
#define TENSOR_H_

#include <cassert>
#include <complex>
#include <iostream>

using Cdouble = std::complex<double>;

static constexpr int MAX_NDIM = 4;

class Tensor {
 public:
  Cdouble* data;
  int ndim;
  int shape[MAX_NDIM];
  int stride[MAX_NDIM];
  int size;
  int capacity;

  // Default constructor: empty tensor.
  Tensor();

  // Allocate tensor with given shape. capacity = size.
  Tensor(int ndim, const int shape[]);

  // Allocate tensor with given shape and explicit capacity (for pre-alloc).
  Tensor(int ndim, const int shape[], int capacity);

  ~Tensor();

  // Deep copy.
  Tensor(const Tensor& other);
  Tensor& operator=(const Tensor& other);

  // Move.
  Tensor(Tensor&& other) noexcept;
  Tensor& operator=(Tensor&& other) noexcept;

  // Element access (row-major offset).
  Cdouble& operator()(int i);
  Cdouble& operator()(int i, int j);
  Cdouble& operator()(int i, int j, int k);
  Cdouble& operator()(int i, int j, int k, int l);
  const Cdouble& operator()(int i) const;
  const Cdouble& operator()(int i, int j) const;
  const Cdouble& operator()(int i, int j, int k) const;
  const Cdouble& operator()(int i, int j, int k, int l) const;

  // In-place operations.
  void reshape(int new_ndim, const int new_shape[]);
  void zero();
  void scale(Cdouble alpha);

  // Factories.
  static Tensor zeros(int ndim, const int shape[]);
  static Tensor identity(int n);

 private:
  void compute_strides();
};

// Free functions.
Tensor tensor_transpose(const Tensor& t, const int perm[]);
Cdouble tensor_trace(const Tensor& t);
void tensor_print(const Tensor& t, const char* name = "");

#endif  // TENSOR_H_
