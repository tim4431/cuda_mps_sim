#include "tensor.h"

#include <algorithm>
#include <cstring>

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

void Tensor::compute_strides() {
  if (ndim == 0) return;
  stride[ndim - 1] = 1;
  for (int k = ndim - 2; k >= 0; --k) {
    stride[k] = stride[k + 1] * shape[k + 1];
  }
}

// ---------------------------------------------------------------------------
// Constructors / destructor
// ---------------------------------------------------------------------------

Tensor::Tensor() : data(nullptr), ndim(0), size(0), capacity(0) {
  std::fill(shape, shape + MAX_NDIM, 0);
  std::fill(stride, stride + MAX_NDIM, 0);
}

Tensor::Tensor(int ndim, const int shape[]) : ndim(ndim) {
  assert(ndim >= 1 && ndim <= MAX_NDIM);
  size = 1;
  for (int k = 0; k < ndim; ++k) {
    this->shape[k] = shape[k];
    size *= shape[k];
  }
  for (int k = ndim; k < MAX_NDIM; ++k) this->shape[k] = 0;
  capacity = size;
  data = new Cdouble[capacity]();
  compute_strides();
}

Tensor::Tensor(int ndim, const int shape[], int capacity)
    : ndim(ndim), capacity(capacity) {
  assert(ndim >= 1 && ndim <= MAX_NDIM);
  size = 1;
  for (int k = 0; k < ndim; ++k) {
    this->shape[k] = shape[k];
    size *= shape[k];
  }
  for (int k = ndim; k < MAX_NDIM; ++k) this->shape[k] = 0;
  assert(capacity >= size);
  data = new Cdouble[capacity]();
  compute_strides();
}

Tensor::~Tensor() { delete[] data; }

Tensor::Tensor(const Tensor& other)
    : ndim(other.ndim), size(other.size), capacity(other.capacity) {
  std::copy(other.shape, other.shape + MAX_NDIM, shape);
  std::copy(other.stride, other.stride + MAX_NDIM, stride);
  data = new Cdouble[capacity];
  std::copy(other.data, other.data + size, data);
}

Tensor& Tensor::operator=(const Tensor& other) {
  if (this == &other) return *this;
  delete[] data;
  ndim = other.ndim;
  size = other.size;
  capacity = other.capacity;
  std::copy(other.shape, other.shape + MAX_NDIM, shape);
  std::copy(other.stride, other.stride + MAX_NDIM, stride);
  data = new Cdouble[capacity];
  std::copy(other.data, other.data + size, data);
  return *this;
}

Tensor::Tensor(Tensor&& other) noexcept
    : data(other.data),
      ndim(other.ndim),
      size(other.size),
      capacity(other.capacity) {
  std::copy(other.shape, other.shape + MAX_NDIM, shape);
  std::copy(other.stride, other.stride + MAX_NDIM, stride);
  other.data = nullptr;
  other.ndim = 0;
  other.size = 0;
  other.capacity = 0;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
  if (this == &other) return *this;
  delete[] data;
  data = other.data;
  ndim = other.ndim;
  size = other.size;
  capacity = other.capacity;
  std::copy(other.shape, other.shape + MAX_NDIM, shape);
  std::copy(other.stride, other.stride + MAX_NDIM, stride);
  other.data = nullptr;
  other.ndim = 0;
  other.size = 0;
  other.capacity = 0;
  return *this;
}

// ---------------------------------------------------------------------------
// Element access
// ---------------------------------------------------------------------------

Cdouble& Tensor::operator()(int i) {
  assert(i >= 0 && i < size);
  return data[i];
}

Cdouble& Tensor::operator()(int i, int j) {
  assert(ndim >= 2);
  return data[i * stride[0] + j * stride[1]];
}

Cdouble& Tensor::operator()(int i, int j, int k) {
  assert(ndim >= 3);
  return data[i * stride[0] + j * stride[1] + k * stride[2]];
}

Cdouble& Tensor::operator()(int i, int j, int k, int l) {
  assert(ndim == 4);
  return data[i * stride[0] + j * stride[1] + k * stride[2] + l * stride[3]];
}

const Cdouble& Tensor::operator()(int i) const {
  assert(i >= 0 && i < size);
  return data[i];
}

const Cdouble& Tensor::operator()(int i, int j) const {
  assert(ndim >= 2);
  return data[i * stride[0] + j * stride[1]];
}

const Cdouble& Tensor::operator()(int i, int j, int k) const {
  assert(ndim >= 3);
  return data[i * stride[0] + j * stride[1] + k * stride[2]];
}

const Cdouble& Tensor::operator()(int i, int j, int k, int l) const {
  assert(ndim == 4);
  return data[i * stride[0] + j * stride[1] + k * stride[2] + l * stride[3]];
}

// ---------------------------------------------------------------------------
// In-place operations
// ---------------------------------------------------------------------------

void Tensor::reshape(int new_ndim, const int new_shape[]) {
  assert(new_ndim >= 1 && new_ndim <= MAX_NDIM);
  int new_size = 1;
  for (int k = 0; k < new_ndim; ++k) new_size *= new_shape[k];
  assert(new_size <= capacity);
  ndim = new_ndim;
  size = new_size;
  for (int k = 0; k < new_ndim; ++k) shape[k] = new_shape[k];
  for (int k = new_ndim; k < MAX_NDIM; ++k) shape[k] = 0;
  compute_strides();
}

void Tensor::zero() { std::memset(data, 0, size * sizeof(Cdouble)); }

void Tensor::scale(Cdouble alpha) {
  for (int i = 0; i < size; ++i) data[i] *= alpha;
}

// ---------------------------------------------------------------------------
// Factories
// ---------------------------------------------------------------------------

Tensor Tensor::zeros(int ndim, const int shape[]) {
  return Tensor(ndim, shape);  // constructor already zeros
}

Tensor Tensor::identity(int n) {
  int sh[2] = {n, n};
  Tensor t(2, sh);
  for (int i = 0; i < n; ++i) t(i, i) = 1.0;
  return t;
}

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

Tensor tensor_transpose(const Tensor& t, const int perm[]) {
  int new_shape[MAX_NDIM];
  for (int k = 0; k < t.ndim; ++k) new_shape[k] = t.shape[perm[k]];
  Tensor out(t.ndim, new_shape);

  // Multi-index iteration with permuted write.
  int idx[MAX_NDIM] = {0};
  for (int flat = 0; flat < t.size; ++flat) {
    // Compute source offset.
    int src_off = 0;
    for (int k = 0; k < t.ndim; ++k) src_off += idx[k] * t.stride[k];
    // Compute destination offset with permuted indices.
    int dst_off = 0;
    for (int k = 0; k < t.ndim; ++k) dst_off += idx[perm[k]] * out.stride[k];
    out.data[dst_off] = t.data[src_off];

    // Increment multi-index (innermost dimension first).
    for (int k = t.ndim - 1; k >= 0; --k) {
      if (++idx[k] < t.shape[k]) break;
      idx[k] = 0;
    }
  }
  return out;
}

Cdouble tensor_trace(const Tensor& t) {
  assert(t.ndim == 2 && t.shape[0] == t.shape[1]);
  Cdouble tr = 0.0;
  for (int i = 0; i < t.shape[0]; ++i) tr += t(i, i);
  return tr;
}

void tensor_print(const Tensor& t, const char* name) {
  std::cout << "Tensor";
  if (name && name[0]) std::cout << " " << name;
  std::cout << " shape=(";
  for (int k = 0; k < t.ndim; ++k) {
    if (k) std::cout << ",";
    std::cout << t.shape[k];
  }
  std::cout << ") size=" << t.size << std::endl;
  if (t.ndim == 2) {
    for (int i = 0; i < t.shape[0]; ++i) {
      for (int j = 0; j < t.shape[1]; ++j) {
        auto v = t(i, j);
        std::cout << "  (" << v.real() << "," << v.imag() << ")";
      }
      std::cout << std::endl;
    }
  }
}
