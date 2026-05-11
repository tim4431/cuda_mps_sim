#ifndef GPU_TEBD_H_
#define GPU_TEBD_H_

#include "mps.h"
#include "tebd.h"

// Persistent device workspace for the GPU update_bond path.  Allocated once,
// reused across every TEBD bond update.  Mirrors the pre-allocation strategy
// of the host MPS: zero device allocations during evolution.
struct GpuTebdWorkspace {
  void* handle;  // opaque pointer to cublasHandle_t / cusolverDnHandle_t bundle
  int d;
  int chi_max;
  int m_max;  // = chi_max * d
  int n_max;  // = d * chi_max

  // Device buffers.
  void* d_theta;     // (chivL, d, d, chivR) complex
  void* d_gate;      // (d, d, d, d) complex
  void* d_Utheta;    // gate * theta, complex (also reused as A_col for SVD)
  void* d_U;         // cusolver U: col-major (n_max, K)
  void* d_Vt;        // cusolver Vt: col-major (K, m_max)
  double* d_S;       // singular values, length K
  double* d_Ss_in;   // Schmidt values on the left edge, length chi_max
  void* d_newBi;     // (chivL, d, keep) complex
  void* d_newBj;     // (keep, d, chivR) complex (alias of d_U via reinterpret)
  void* d_work;
  int* d_info;
  int lwork;
  int gesvdj_lwork;
  void* gesvdj_params;

  // Page-locked host staging buffers for fast PCIe transfers.
  void* h_theta_pinned;
  void* h_outA_pinned;
  void* h_outB_pinned;
  double* h_S_pinned;

  GpuTebdWorkspace(int d, int chi_max);
  ~GpuTebdWorkspace();
  GpuTebdWorkspace(const GpuTebdWorkspace&) = delete;
  GpuTebdWorkspace& operator=(const GpuTebdWorkspace&) = delete;
};

// GPU version of update_bond.  Uses a custom CUDA kernel for the two-site
// gate contraction, cuSOLVER (Jacobi SVD) for the SVD, and a custom kernel
// for the Vidal-form restore.  Reads/writes the host MPS in place.
double update_bond_gpu(MPS* psi, int site, const Tensor& gate, int chi_max,
                       double svd_min, GpuTebdWorkspace& ws);

// One TEBD step using update_bond_gpu in place of update_bond.  Mirrors
// TEBDEngine::step() but routes every bond update through the GPU.
void tebd_step_gpu(TEBDEngine& eng, GpuTebdWorkspace& ws);

#endif  // GPU_TEBD_H_
