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
  double* d_Ss_in;   // 1/Schmidt values on the left edge, length chi_max
  void* d_newBi;     // (chivL, d, keep) complex  [non-persistent path only]
  void* d_newBj;     // (keep, d, chivR) complex  [non-persistent path only]
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

  // CUDA stream for async operations (nullptr = default stream).
  // Used by update_bond_gpu_persistent(); update_bond_gpu() always uses null.
  void* stream;  // opaque cudaStream_t

  // stream = nullptr uses the default (null) CUDA stream.
  GpuTebdWorkspace(int d, int chi_max, void* stream = nullptr);
  ~GpuTebdWorkspace();
  GpuTebdWorkspace(const GpuTebdWorkspace&) = delete;
  GpuTebdWorkspace& operator=(const GpuTebdWorkspace&) = delete;
};

// Per-stage timing accumulator (milliseconds).  When passed to
// update_bond_gpu, the function uses CUDA events to record the wall time
// of each sub-stage on the GPU stream and adds it to the corresponding
// field.  Pass nullptr to disable timing.
struct GpuBondTimings {
  double h2d_ms = 0;
  double gate_kernel_ms = 0;
  double row_to_col_ms = 0;
  double svd_ms = 0;
  double restore_ms = 0;
  double d2h_ms = 0;
  int n_calls = 0;
};

// GPU version of update_bond.  Uses a custom CUDA kernel for the two-site
// gate contraction, cuSOLVER (Jacobi SVD) for the SVD, and a custom kernel
// for the Vidal-form restore.  Reads/writes the host MPS in place.
double update_bond_gpu(MPS* psi, int site, const Tensor& gate, int chi_max,
                       double svd_min, GpuTebdWorkspace& ws,
                       GpuBondTimings* timings = nullptr);

// Same as update_bond_gpu but uses the custom one-sided Jacobi SVD kernel
// instead of cuSOLVER.  Produces identical results (to ~1e-13) for correctness
// validation.  Used by bench_svd to compare performance.
double update_bond_gpu_jacobi(MPS* psi, int site, const Tensor& gate,
                               int chi_max, double svd_min,
                               GpuTebdWorkspace& ws,
                               GpuBondTimings* timings = nullptr);

// One TEBD step using update_bond_gpu in place of update_bond.  Mirrors
// TEBDEngine::step() but routes every bond update through the GPU.
void tebd_step_gpu(TEBDEngine& eng, GpuTebdWorkspace& ws);

// ---------------------------------------------------------------------------
// Persistent device MPS
// ---------------------------------------------------------------------------

// Device-resident MPS: keeps all B tensors and Schmidt spectra on the GPU.
// Zero H2D/D2H for tensors during TEBD evolution; sync to host only at
// measurement points.  Each d_Bs[i] is allocated at chi_max*d*chi_max;
// the live extent is chi[i]*d*chi[i+1].
struct GpuMps {
  int L, d, chi_max;
  void** d_Bs;  // [L]   device pointers (each chi_max*d*chi_max complex)
  void** d_Ss;  // [L]   device pointers (each chi_max doubles)
                //         d_Ss[i] = Schmidt spectrum on bond LEFT of site i
  int*   chi;   // [L+1] current bond dimensions (host side)

  explicit GpuMps(const MPS& mps);
  ~GpuMps();
  GpuMps(const GpuMps&) = delete;
  GpuMps& operator=(const GpuMps&) = delete;

  // Copy all device tensors back to host MPS.  Call only at measurement
  // points; avoid calling in the inner TEBD loop.
  void sync_to_host(MPS& mps) const;
};

// Update bond using device-resident MPS: no H2D/D2H for B tensors.
// theta2 is built on-device; vidal restore writes directly into gmps.d_Bs.
// The gate (256 bytes) is still transferred from host each call.
// Uses ws.stream for all async operations.
double update_bond_gpu_persistent(GpuMps& gmps, int site, const Tensor& gate,
                                   int chi_max, double svd_min,
                                   GpuTebdWorkspace& ws,
                                   GpuBondTimings* timings = nullptr);

// One TEBD step using persistent device MPS.  n_ws workspaces are assigned
// round-robin across independent bonds in each sub-step.  With n_ws > 1 and
// distinct ws.stream values the GPU can overlap SVD work across streams.
// A cudaDeviceSynchronize() is issued after each even/odd sub-step before
// proceeding to the next.
// ws_arr[0..n_ws-1] are pointers to independent workspaces (allows storing
// non-copyable GpuTebdWorkspace in a std::vector<unique_ptr<...>>).
void tebd_step_gpu_persistent(TEBDEngine& eng, GpuMps& gmps,
                               GpuTebdWorkspace* const* ws_arr, int n_ws);

// Convenience single-workspace overload (no stream overlap).
inline void tebd_step_gpu_persistent(TEBDEngine& eng, GpuMps& gmps,
                                     GpuTebdWorkspace& ws) {
  GpuTebdWorkspace* p = &ws;
  tebd_step_gpu_persistent(eng, gmps, &p, 1);
}

// ---------------------------------------------------------------------------
// Stream management helpers (thin wrappers so app .cpp files need not include
// cuda_runtime.h directly; the real cudaStream_t lives only in gpu_tebd.cu).
// ---------------------------------------------------------------------------

// Create a new CUDA stream.  Returns opaque void* (cast to cudaStream_t inside
// gpu_tebd.cu).  Pass the returned pointer to GpuTebdWorkspace constructor.
void* gpu_stream_create();

// Destroy a stream previously created by gpu_stream_create().
void  gpu_stream_destroy(void* stream);

// Host-side barrier: wait until all work on all streams has completed.
void  gpu_device_sync();

// Query number of available CUDA devices.
int   gpu_device_count();

// Set the CUDA device for the calling thread.
void  gpu_set_device(int device);

#endif  // GPU_TEBD_H_
