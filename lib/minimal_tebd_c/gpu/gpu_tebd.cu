// CUDA backend for TEBD update_bond.
//
// Pipeline for a single bond:
//   1. H2D: copy theta (chivL, d, d, chivR), gate (d, d, d, d), Ss[site].
//   2. Custom kernel gate_contract: utheta_row = gate * theta on inner d^2.
//   3. Custom kernel row_to_col: transpose utheta to column-major for SVD.
//   4. cusolverDnZgesvdj: A_col(m,n) = U_col Sigma V_col^H, with m=chivL*d,
//      n=d*chivR.  Singular values come back sorted descending.
//   5. Host: compute keep, eps, renormalise S.
//   6. Custom kernel vidal_restore_left: new_Bi[a,p,b] = (1/Ss[a]) *
//      U_col[a*d+p, b] * S[b], producing row-major (chivL, d, keep).
//   7. Custom kernel vidal_restore_right: new_Bj[c,q,e] = conj(V_col[q*chivR+e, c]),
//      producing row-major (keep, d, chivR).  The conjugate is because the
//      SVD identity is A = U Sigma V^H, and we want V^H's row c, column j.
//   8. D2H: copy new_Bi, new_Bj.

#include "gpu_tebd.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstring>
#include <vector>

#include <cublas_v2.h>
#include <cuComplex.h>
#include <cusolverDn.h>
#include <cuda_runtime.h>

#include "linalg.h"

#define CUDA_CHECK(stmt)                                                       \
  do {                                                                         \
    cudaError_t err__ = (stmt);                                                \
    if (err__ != cudaSuccess) {                                                \
      std::fprintf(stderr, "CUDA error %s at %s:%d\n",                         \
                   cudaGetErrorString(err__), __FILE__, __LINE__);             \
      std::abort();                                                            \
    }                                                                          \
  } while (0)

#define CUSOLVER_CHECK(stmt)                                                   \
  do {                                                                         \
    cusolverStatus_t s__ = (stmt);                                             \
    if (s__ != CUSOLVER_STATUS_SUCCESS) {                                      \
      std::fprintf(stderr, "cuSOLVER error %d at %s:%d\n", int(s__), __FILE__, \
                   __LINE__);                                                  \
      std::abort();                                                            \
    }                                                                          \
  } while (0)

#define CUBLAS_CHECK(stmt)                                                     \
  do {                                                                         \
    cublasStatus_t s__ = (stmt);                                               \
    if (s__ != CUBLAS_STATUS_SUCCESS) {                                        \
      std::fprintf(stderr, "cuBLAS error %d at %s:%d\n", int(s__), __FILE__,   \
                   __LINE__);                                                  \
      std::abort();                                                            \
    }                                                                          \
  } while (0)

__device__ __forceinline__ cuDoubleComplex cd_mul(cuDoubleComplex a,
                                                  cuDoubleComplex b) {
  return make_cuDoubleComplex(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// ---------------------------------------------------------------------------
// gate_contract_kernel
//
//   gate : (D2, D2) row-major     -- (ip*D + jp, i*D + j)
//   theta: (chivL, D2, chivR) row-major  -- (vL, i*D + j, vR)
//   out  : (chivL, D2, chivR) row-major  -- (vL, ip*D + jp, vR)
//
// Each thread computes one output element.  The full gate (D2*D2 = 16
// complex numbers for d=2) lives in shared memory and is reused by every
// thread in the block.  Grid is (vR-tiles, ipjp-tiles, vL).
// ---------------------------------------------------------------------------

template <int D>
__global__ void gate_contract_kernel(const cuDoubleComplex* __restrict__ gate,
                                     const cuDoubleComplex* __restrict__ theta,
                                     cuDoubleComplex* __restrict__ out,
                                     int chivL, int chivR) {
  constexpr int D2 = D * D;
  __shared__ cuDoubleComplex sgate[D2 * D2];

  int tid = threadIdx.y * blockDim.x + threadIdx.x;
  int nthreads = blockDim.x * blockDim.y;
  for (int idx = tid; idx < D2 * D2; idx += nthreads) sgate[idx] = gate[idx];
  __syncthreads();

  int vR = blockIdx.x * blockDim.x + threadIdx.x;
  int ipjp = blockIdx.y * blockDim.y + threadIdx.y;
  int vL = blockIdx.z;
  if (vR >= chivR || ipjp >= D2 || vL >= chivL) return;

  cuDoubleComplex acc = make_cuDoubleComplex(0.0, 0.0);
  int row_base = vL * (D2 * chivR);
  int g_base = ipjp * D2;
#pragma unroll
  for (int ij = 0; ij < D2; ++ij) {
    cuDoubleComplex g = sgate[g_base + ij];
    cuDoubleComplex t = theta[row_base + ij * chivR + vR];
    cuDoubleComplex p = cd_mul(g, t);
    acc.x += p.x;
    acc.y += p.y;
  }
  out[row_base + ipjp * chivR + vR] = acc;
}

// ---------------------------------------------------------------------------
// row_to_col_kernel
//
// in_row(M, N) row-major  -->  out_col(M, N) column-major (ld = M).
// Coalesced read along N for each thread; write stride is M (not optimal,
// but the matrix is small and this only runs once per bond).
// ---------------------------------------------------------------------------

__global__ void row_to_col_kernel(const cuDoubleComplex* __restrict__ in_row,
                                  cuDoubleComplex* __restrict__ out_col, int M,
                                  int N) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= M || j >= N) return;
  out_col[j * M + i] = in_row[i * N + j];
}

// Build B = A^H for the m < n SVD case.  Both A and B are column-major:
//   A is (m x n), B is (n x m), B[j, i] = conj(A[i, j]).
__global__ void col_major_adjoint_kernel(
    const cuDoubleComplex* __restrict__ A_col,
    cuDoubleComplex* __restrict__ B_col, int m, int n) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;  // A column / B row
  int i = blockIdx.y * blockDim.y + threadIdx.y;  // A row / B column
  if (i >= m || j >= n) return;
  cuDoubleComplex a = A_col[j * m + i];
  B_col[i * n + j] = make_cuDoubleComplex(a.x, -a.y);
}

// ---------------------------------------------------------------------------
// vidal_restore_left_kernel
//
// new_Bi[a, p, b] = (1/Ss[a]) * U_col[a*D + p, b] * S[b]
//
// U_col is column-major (m, K) with leading dim m, where m = chivL*D.  Entry
// (i, k) lives at U_col_buf[k*m + i].  new_Bi is row-major (chivL, D, keep).
// ---------------------------------------------------------------------------

template <int D>
__global__ void vidal_restore_left_kernel(
    const cuDoubleComplex* __restrict__ U_col,
    const double* __restrict__ Ss_inv, const double* __restrict__ S_keep,
    cuDoubleComplex* __restrict__ newBi, int chivL, int keep, int m) {
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  int p = threadIdx.y;
  int a = blockIdx.y;
  if (b >= keep || p >= D || a >= chivL) return;
  int i = a * D + p;
  cuDoubleComplex u = U_col[b * m + i];
  double scl = Ss_inv[a] * S_keep[b];
  newBi[a * D * keep + p * keep + b] = make_cuDoubleComplex(u.x * scl, u.y * scl);
}

// ---------------------------------------------------------------------------
// vidal_restore_right_kernel
//
// new_Bj[c, q, e] = conj(V_col[q*chivR + e, c])
//
// cuSOLVER returns V (not V^H), so we conjugate here to produce V^H.  V_col
// is column-major (n, K) with leading dim n, where n = D*chivR.  new_Bj is
// row-major (keep, D, chivR).
// ---------------------------------------------------------------------------

template <int D>
__global__ void vidal_restore_right_kernel(
    const cuDoubleComplex* __restrict__ V_col,
    cuDoubleComplex* __restrict__ newBj, int keep, int chivR, int n) {
  int e = blockIdx.x * blockDim.x + threadIdx.x;
  int q = threadIdx.y;
  int c = blockIdx.y;
  if (e >= chivR || q >= D || c >= keep) return;
  int j = q * chivR + e;
  cuDoubleComplex v = V_col[c * n + j];
  newBj[c * D * chivR + q * chivR + e] = make_cuDoubleComplex(v.x, -v.y);
}

// ---------------------------------------------------------------------------
// Custom one-sided Jacobi SVD kernel.
//
// Replaces cuSOLVER gesvdj for small complex matrices (m x n, m >= n, n <= 256).
//
// Inputs (device pointers):
//   W      - (m x n col-major) modified in place; on entry holds A.
//            On exit, col j = sigma_j * u_j (unnormalized left sv).
//   V      - (n x n col-major) modified in place; initialized to I_n inside.
//            On exit, V = right singular vectors (same sign convention as cuSOLVER:
//            the vidal_restore_right kernel applies the conjugate).
//
// Outputs (device pointers written by jacobi_finish_kernel):
//   d_U    - (m x K col-major)  left singular vectors (normalized W columns)
//   d_Vout - (n x K col-major)  right singular vectors (= V, not V^H)
//   d_S    - (K)               singular values sorted descending
//
// Both kernels run on a single thread block of JACOBI_BLOCK threads.
// ---------------------------------------------------------------------------

static const int JACOBI_BLOCK  = 128;
static const double JACOBI_TOL = 1e-15;
static const int JACOBI_SWEEPS = 200;

// Shared-memory layout for jacobi_sweep_kernel (doubles):
//   [0 .. JACOBI_BLOCK-1]     : reduction scratch
//   [JACOBI_BLOCK + 0]        : gamma_re
//   [JACOBI_BLOCK + 1]        : gamma_im
//   [JACOBI_BLOCK + 2]        : c  (Jacobi cosine)
//   [JACOBI_BLOCK + 3]        : s_r (Jacobi sine)
//   [JACOBI_BLOCK + 4]        : do_rotate flag (1.0 / 0.0)
//   [JACOBI_BLOCK + 5]        : phase_conj_re
//   [JACOBI_BLOCK + 6]        : phase_conj_im
//   [JACOBI_BLOCK + 7]        : alpha_pp
//   [JACOBI_BLOCK + 8]        : beta_qq
//   [JACOBI_BLOCK + 9]        : any_rotation (sweep-level flag)
// Total: (JACOBI_BLOCK + 10) * 8 bytes  = 1104 bytes for JACOBI_BLOCK=128.

#define JS_SMEM_SZ  (JACOBI_BLOCK + 10)

// Shared-memory layout for jacobi_finish_kernel (doubles):
//   [0 .. 255]                : S_tmp  (column norms, n <= 256)
//   [256 .. 383]              : order  (256 ints packed as 128 doubles, n <= 256)
// + separate reduction: JACOBI_BLOCK doubles (passed as second segment)
// => We use a single extern array and index as shown below.

#define JF_SMEM_SZ  (256 + 128 + JACOBI_BLOCK)   // double units

// ---------------------------------------------------------------------------
// jacobi_sweep_kernel: one thread block, m*n <= m_max * n_max.
// ---------------------------------------------------------------------------
__global__ void jacobi_sweep_kernel(cuDoubleComplex* __restrict__ W,
                                    cuDoubleComplex* __restrict__ V,
                                    int m, int n) {
  extern __shared__ double smem[];
  double* sred      = smem;
  double* gamma_re  = smem + JACOBI_BLOCK;
  double* gamma_im  = smem + JACOBI_BLOCK + 1;
  double* rot_c     = smem + JACOBI_BLOCK + 2;
  double* rot_s     = smem + JACOBI_BLOCK + 3;
  double* do_rot    = smem + JACOBI_BLOCK + 4;
  double* pcon_re   = smem + JACOBI_BLOCK + 5;
  double* pcon_im   = smem + JACOBI_BLOCK + 6;
  double* alpha_sh  = smem + JACOBI_BLOCK + 7;
  double* beta_sh   = smem + JACOBI_BLOCK + 8;
  double* any_rot   = smem + JACOBI_BLOCK + 9;

  int tid = threadIdx.x;

  // Initialise V = I_n.
  {
    int total = n * n;
    for (int k = tid; k < total; k += blockDim.x) {
      int col = k / n, row = k % n;
      V[k] = make_cuDoubleComplex((col == row) ? 1.0 : 0.0, 0.0);
    }
  }
  __syncthreads();

  for (int sweep = 0; sweep < JACOBI_SWEEPS; ++sweep) {
    if (tid == 0) *any_rot = 0.0;
    __syncthreads();

    for (int p = 0; p < n; ++p) {
      for (int q = p + 1; q < n; ++q) {
        const cuDoubleComplex* Wp = W + (long)p * m;
        const cuDoubleComplex* Wq = W + (long)q * m;

        // --- alpha_pp = ||Wp||^2 ---
        {
          double loc = 0.0;
          for (int i = tid; i < m; i += blockDim.x)
            loc += Wp[i].x * Wp[i].x + Wp[i].y * Wp[i].y;
          sred[tid] = loc; __syncthreads();
          for (int off = blockDim.x >> 1; off > 0; off >>= 1) {
            if (tid < off) sred[tid] += sred[tid + off];
            __syncthreads();
          }
          if (tid == 0) *alpha_sh = sred[0];
          __syncthreads();
        }

        // --- beta_qq = ||Wq||^2 ---
        {
          double loc = 0.0;
          for (int i = tid; i < m; i += blockDim.x)
            loc += Wq[i].x * Wq[i].x + Wq[i].y * Wq[i].y;
          sred[tid] = loc; __syncthreads();
          for (int off = blockDim.x >> 1; off > 0; off >>= 1) {
            if (tid < off) sred[tid] += sred[tid + off];
            __syncthreads();
          }
          if (tid == 0) *beta_sh = sred[0];
          __syncthreads();
        }

        // --- gamma_pq = Wp^H * Wq (complex) ---
        {
          double gre = 0.0, gim = 0.0;
          for (int i = tid; i < m; i += blockDim.x) {
            // conj(Wp[i]) * Wq[i]
            gre += Wp[i].x * Wq[i].x + Wp[i].y * Wq[i].y;
            gim += Wp[i].x * Wq[i].y - Wp[i].y * Wq[i].x;
          }
          sred[tid] = gre; __syncthreads();
          for (int off = blockDim.x >> 1; off > 0; off >>= 1) {
            if (tid < off) sred[tid] += sred[tid + off];
            __syncthreads();
          }
          if (tid == 0) *gamma_re = sred[0];
          __syncthreads();
          sred[tid] = gim; __syncthreads();
          for (int off = blockDim.x >> 1; off > 0; off >>= 1) {
            if (tid < off) sred[tid] += sred[tid + off];
            __syncthreads();
          }
          if (tid == 0) *gamma_im = sred[0];
          __syncthreads();
        }

        // --- Rotation params (thread 0 only) ---
        if (tid == 0) {
          double gx = *gamma_re, gy = *gamma_im;
          double g = sqrt(gx * gx + gy * gy);
          double alpha_pp = *alpha_sh, beta_qq = *beta_sh;
          if (g < JACOBI_TOL * sqrt(alpha_pp * beta_qq + 1e-300)) {
            *do_rot = 0.0;
          } else {
            *do_rot    = 1.0;
            *any_rot   = 1.0;
            *pcon_re   =  gx / g;
            *pcon_im   = -gy / g;
            double tau = (beta_qq - alpha_pp) / (2.0 * g);
            double sign = (tau >= 0.0) ? -1.0 : 1.0;
            double at   = (tau >= 0.0) ?  tau : -tau;
            double t    = sign / (at + sqrt(1.0 + tau * tau));
            double c    = 1.0 / sqrt(1.0 + t * t);
            *rot_c = c;
            *rot_s = c * t;
          }
        }
        __syncthreads();

        // --- Apply rotation (all threads) ---
        if (*do_rot > 0.5) {
          double c   = *rot_c,  sr  = *rot_s;
          double pcr = *pcon_re, pci = *pcon_im;

          // Rotate W columns p, q.
          cuDoubleComplex* Wpw = W + (long)p * m;
          cuDoubleComplex* Wqw = W + (long)q * m;
          for (int i = tid; i < m; i += blockDim.x) {
            double mpr = Wpw[i].x, mpi = Wpw[i].y;
            double mqr = Wqw[i].x, mqi = Wqw[i].y;
            double pcmqr = pcr * mqr - pci * mqi;
            double pcmqi = pcr * mqi + pci * mqr;
            Wpw[i].x =  c * mpr + sr * pcmqr;
            Wpw[i].y =  c * mpi + sr * pcmqi;
            Wqw[i].x = -sr * mpr + c * pcmqr;
            Wqw[i].y = -sr * mpi + c * pcmqi;
          }
          __syncthreads();

          // Rotate V columns p, q.
          cuDoubleComplex* Vpw = V + (long)p * n;
          cuDoubleComplex* Vqw = V + (long)q * n;
          for (int i = tid; i < n; i += blockDim.x) {
            double mpr = Vpw[i].x, mpi = Vpw[i].y;
            double mqr = Vqw[i].x, mqi = Vqw[i].y;
            double pcmqr = pcr * mqr - pci * mqi;
            double pcmqi = pcr * mqi + pci * mqr;
            Vpw[i].x =  c * mpr + sr * pcmqr;
            Vpw[i].y =  c * mpi + sr * pcmqi;
            Vqw[i].x = -sr * mpr + c * pcmqr;
            Vqw[i].y = -sr * mpi + c * pcmqi;
          }
          __syncthreads();
        }
      }  // q
    }  // p

    // Early exit if no rotation occurred this sweep.
    __syncthreads();
    if (*any_rot < 0.5) break;
  }  // sweep
}

// ---------------------------------------------------------------------------
// jacobi_finish_kernel
//
// Post-processes W (=U*diag(S), col-major m x n) and V (col-major n x n)
// after jacobi_sweep_kernel, computing sorted singular values and writing
// normalised U and V columns to the output buffers.
//
// Shared memory:
//   [0..255]         : S_tmp  (double, n <= 256)
//   [256..319]       : order  (int, stored as 2-per-double slot, n <= 256)
//   [320..447]       : reduction scratch (JACOBI_BLOCK doubles)
// ---------------------------------------------------------------------------
__global__ void jacobi_finish_kernel(
    const cuDoubleComplex* __restrict__ W,  // m x n col-major (W = U * diag(S))
    const cuDoubleComplex* __restrict__ V,  // n x n col-major
    cuDoubleComplex* __restrict__ d_U,      // output m x K col-major
    cuDoubleComplex* __restrict__ d_Vout,   // output n x K col-major
    double*          __restrict__ d_S,      // output K
    int m, int n, int K) {

  extern __shared__ double smem[];
  double* S_tmp  = smem;            // [0..255]
  int*    order  = (int*)(smem + 256);  // [256..511 bytes = 64 doubles]
  double* sred   = smem + 256 + 64; // [320..447]

  int tid = threadIdx.x;

  // Step 1: compute column norms of W in parallel.
  for (int j = 0; j < n; ++j) {
    const cuDoubleComplex* Wj = W + (long)j * m;
    double loc = 0.0;
    for (int i = tid; i < m; i += blockDim.x)
      loc += Wj[i].x * Wj[i].x + Wj[i].y * Wj[i].y;
    sred[tid] = loc; __syncthreads();
    for (int off = blockDim.x >> 1; off > 0; off >>= 1) {
      if (tid < off) sred[tid] += sred[tid + off];
      __syncthreads();
    }
    if (tid == 0) S_tmp[j] = sqrt(sred[0]);
    __syncthreads();
  }

  // Step 2: sort by descending singular value (thread 0, insertion sort, n<=256).
  if (tid == 0) {
    for (int j = 0; j < n; ++j) order[j] = j;
    // Insertion sort descending.
    for (int i = 1; i < n; ++i) {
      int key_ord = order[i];
      double key_s = S_tmp[key_ord];
      int j = i - 1;
      while (j >= 0 && S_tmp[order[j]] < key_s) {
        order[j + 1] = order[j];
        --j;
      }
      order[j + 1] = key_ord;
    }
    // Write sorted S to d_S.
    for (int k = 0; k < K; ++k) d_S[k] = S_tmp[order[k]];
  }
  __syncthreads();

  // Step 3: write normalised U and V to output (first K columns, sorted order).
  for (int k = 0; k < K; ++k) {
    int j     = order[k];
    double sv = S_tmp[j];
    double inv_s = (sv > 1e-300) ? 1.0 / sv : 0.0;

    // d_U[:,k] = W[:,j] / sv
    const cuDoubleComplex* Wj = W + (long)j * m;
    cuDoubleComplex*        Uk = d_U + (long)k * m;
    for (int i = tid; i < m; i += blockDim.x) {
      Uk[i].x = Wj[i].x * inv_s;
      Uk[i].y = Wj[i].y * inv_s;
    }
    __syncthreads();

    // d_Vout[:,k] = V[:,j]
    const cuDoubleComplex* Vj  = V + (long)j * n;
    cuDoubleComplex*        Vok = d_Vout + (long)k * n;
    for (int i = tid; i < n; i += blockDim.x) Vok[i] = Vj[i];
    __syncthreads();
  }
}

// ---------------------------------------------------------------------------
// Handle bundle.
// ---------------------------------------------------------------------------

struct GpuHandles {
  cublasHandle_t cublas;
  cusolverDnHandle_t cusolver;
};

GpuTebdWorkspace::GpuTebdWorkspace(int d, int chi_max, void* stream)
    : d(d),
      chi_max(chi_max),
      m_max(chi_max * d),
      n_max(d * chi_max),
      d_theta(nullptr),
      d_gate(nullptr),
      d_Utheta(nullptr),
      d_U(nullptr),
      d_Vt(nullptr),
      d_S(nullptr),
      d_Ss_in(nullptr),
      d_newBi(nullptr),
      d_newBj(nullptr),
      d_work(nullptr),
      d_info(nullptr),
      lwork(0),
      gesvdj_lwork(0),
      gesvdj_params(nullptr),
      h_theta_pinned(nullptr),
      h_outA_pinned(nullptr),
      h_outB_pinned(nullptr),
      h_S_pinned(nullptr),
      stream(stream) {
  auto* h = new GpuHandles();
  CUBLAS_CHECK(cublasCreate(&h->cublas));
  CUSOLVER_CHECK(cusolverDnCreate(&h->cusolver));
  handle = h;

  const int K = std::min(m_max, n_max);
  size_t Acol_bytes =
      sizeof(cuDoubleComplex) * size_t(m_max) * size_t(n_max);
  CUDA_CHECK(cudaMalloc(&d_theta, Acol_bytes));   // dual-use: input + col-major
  CUDA_CHECK(cudaMalloc(&d_Utheta, Acol_bytes));  // gate kernel output (row-major)
  CUDA_CHECK(cudaMalloc(&d_gate, sizeof(cuDoubleComplex) * d * d * d * d));
  // U_col: (m, K) col-major, ld = m.
  CUDA_CHECK(
      cudaMalloc(&d_U, sizeof(cuDoubleComplex) * size_t(m_max) * size_t(K)));
  // V_col: (n, K) col-major, ld = n.
  CUDA_CHECK(
      cudaMalloc(&d_Vt, sizeof(cuDoubleComplex) * size_t(n_max) * size_t(K)));
  CUDA_CHECK(cudaMalloc(&d_S, sizeof(double) * size_t(K)));
  CUDA_CHECK(cudaMalloc(&d_Ss_in, sizeof(double) * chi_max));
  CUDA_CHECK(cudaMalloc(&d_newBi, sizeof(cuDoubleComplex) * size_t(chi_max) *
                                      size_t(d) * size_t(chi_max)));
  CUDA_CHECK(cudaMalloc(&d_newBj, sizeof(cuDoubleComplex) * size_t(chi_max) *
                                      size_t(d) * size_t(chi_max)));
  CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));

  gesvdjInfo_t params = nullptr;
  CUSOLVER_CHECK(cusolverDnCreateGesvdjInfo(&params));
  CUSOLVER_CHECK(cusolverDnXgesvdjSetTolerance(params, 1e-14));
  CUSOLVER_CHECK(cusolverDnXgesvdjSetMaxSweeps(params, 100));
  gesvdj_params = params;

  int lw = 0;
  CUSOLVER_CHECK(cusolverDnZgesvdj_bufferSize(
      h->cusolver, CUSOLVER_EIG_MODE_VECTOR, /*econ=*/1, /*m=*/m_max,
      /*n=*/n_max, reinterpret_cast<cuDoubleComplex*>(d_theta), /*lda=*/m_max,
      d_S, reinterpret_cast<cuDoubleComplex*>(d_U), /*ldu=*/m_max,
      reinterpret_cast<cuDoubleComplex*>(d_Vt), /*ldv=*/n_max, &lw, params));
  lwork = lw;
  gesvdj_lwork = lw;
  CUDA_CHECK(cudaMalloc(&d_work, sizeof(cuDoubleComplex) * size_t(lwork)));

  CUDA_CHECK(cudaMallocHost(&h_theta_pinned, Acol_bytes));
  CUDA_CHECK(cudaMallocHost(&h_outA_pinned,
                            sizeof(cuDoubleComplex) * size_t(chi_max) *
                                size_t(d) * size_t(chi_max)));
  CUDA_CHECK(cudaMallocHost(&h_outB_pinned,
                            sizeof(cuDoubleComplex) * size_t(chi_max) *
                                size_t(d) * size_t(chi_max)));
  CUDA_CHECK(cudaMallocHost(&h_S_pinned, sizeof(double) * size_t(K)));
}

GpuTebdWorkspace::~GpuTebdWorkspace() {
  auto* h = reinterpret_cast<GpuHandles*>(handle);
  if (gesvdj_params)
    cusolverDnDestroyGesvdjInfo(reinterpret_cast<gesvdjInfo_t>(gesvdj_params));
  if (h) {
    cublasDestroy(h->cublas);
    cusolverDnDestroy(h->cusolver);
    delete h;
  }
  cudaFree(d_theta);
  cudaFree(d_Utheta);
  cudaFree(d_gate);
  cudaFree(d_U);
  cudaFree(d_Vt);
  cudaFree(d_S);
  cudaFree(d_Ss_in);
  cudaFree(d_newBi);
  cudaFree(d_newBj);
  cudaFree(d_work);
  cudaFree(d_info);
  if (h_theta_pinned) cudaFreeHost(h_theta_pinned);
  if (h_outA_pinned) cudaFreeHost(h_outA_pinned);
  if (h_outB_pinned) cudaFreeHost(h_outB_pinned);
  if (h_S_pinned) cudaFreeHost(h_S_pinned);
}

// Tiny RAII wrapper that records start/stop events bracketing a region and
// adds the elapsed-time delta into the given accumulator field on destruct.
namespace {
struct EventTimer {
  cudaEvent_t start{}, stop{};
  double* dst;
  EventTimer(double* dst_) : dst(dst_) {
    if (!dst) return;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
  }
  ~EventTimer() {
    if (!dst) return;
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, start, stop);
    *dst += double(ms);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }
};
}  // namespace

double update_bond_gpu(MPS* psi, int site, const Tensor& gate, int chi_max,
                       double svd_min, GpuTebdWorkspace& ws,
                       GpuBondTimings* timings) {
  assert(psi->d == ws.d);
  auto* h = reinterpret_cast<GpuHandles*>(ws.handle);
  const int d = ws.d;
  const int D2 = d * d;
  if (timings) timings->n_calls += 1;

  Tensor theta = psi->theta2(site);
  int chivL = theta.shape[0];
  int chivR = theta.shape[3];
  int m = chivL * d;
  int n = d * chivR;
  int K = std::min(m, n);
  size_t theta_elems = size_t(chivL) * D2 * chivR;

  if (m > ws.m_max || n > ws.n_max) {
    std::fprintf(stderr,
                 "update_bond_gpu: bond too large (m=%d > %d or n=%d > %d) "
                 "at site=%d\n",
                 m, ws.m_max, n, ws.n_max, site);
    std::abort();
  }

  std::memcpy(ws.h_theta_pinned, theta.data,
              theta_elems * sizeof(cuDoubleComplex));

  {
    EventTimer t(timings ? &timings->h2d_ms : nullptr);
    CUDA_CHECK(cudaMemcpyAsync(ws.d_theta, ws.h_theta_pinned,
                               theta_elems * sizeof(cuDoubleComplex),
                               cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyAsync(ws.d_gate, gate.data,
                               D2 * D2 * sizeof(cuDoubleComplex),
                               cudaMemcpyHostToDevice));
  }

  // Step 2: gate contraction.
  {
    EventTimer t(timings ? &timings->gate_kernel_ms : nullptr);
    int blkx = std::min(32, chivR > 0 ? chivR : 1);
    int blky = D2;
    dim3 block(blkx, blky);
    dim3 grid((chivR + block.x - 1) / block.x,
              (D2 + block.y - 1) / block.y, chivL);
    if (d == 2) {
      gate_contract_kernel<2><<<grid, block>>>(
          reinterpret_cast<cuDoubleComplex*>(ws.d_gate),
          reinterpret_cast<cuDoubleComplex*>(ws.d_theta),
          reinterpret_cast<cuDoubleComplex*>(ws.d_Utheta), chivL, chivR);
    } else {
      std::fprintf(stderr,
                   "gate_contract_kernel: only d=2 supported in this build\n");
      std::abort();
    }
    CUDA_CHECK(cudaGetLastError());
  }

  // Step 3: row-major utheta -> col-major A.  Reuse d_theta as A_col.
  {
    EventTimer t(timings ? &timings->row_to_col_ms : nullptr);
    dim3 block(32, 8);
    dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);
    row_to_col_kernel<<<grid, block>>>(
        reinterpret_cast<cuDoubleComplex*>(ws.d_Utheta),
        reinterpret_cast<cuDoubleComplex*>(ws.d_theta), m, n);
    CUDA_CHECK(cudaGetLastError());
  }

  // Step 4: SVD via Jacobi.  A_col(m, n) = U_col(m, K) Sigma V_col(n, K)^H.
  // Re-query bufferSize for the actual (m, n).  cuSOLVER's lwork is not
  // monotone in (m, n): the QR pre-pass inside gesvdj has different
  // workspace requirements depending on the shape.
  {
    int need_lwork = 0;
    CUSOLVER_CHECK(cusolverDnZgesvdj_bufferSize(
        h->cusolver, CUSOLVER_EIG_MODE_VECTOR, /*econ=*/1, m, n,
        reinterpret_cast<cuDoubleComplex*>(ws.d_theta), /*lda=*/m, ws.d_S,
        reinterpret_cast<cuDoubleComplex*>(ws.d_U), /*ldu=*/m,
        reinterpret_cast<cuDoubleComplex*>(ws.d_Vt), /*ldv=*/n, &need_lwork,
        reinterpret_cast<gesvdjInfo_t>(ws.gesvdj_params)));
    if (need_lwork > ws.lwork) {
      cudaFree(ws.d_work);
      ws.d_work = nullptr;
      CUDA_CHECK(cudaMalloc(&ws.d_work,
                            sizeof(cuDoubleComplex) * size_t(need_lwork)));
      ws.lwork = need_lwork;
    }
  }
  {
    EventTimer t(timings ? &timings->svd_ms : nullptr);
    CUSOLVER_CHECK(cusolverDnZgesvdj(
        h->cusolver, CUSOLVER_EIG_MODE_VECTOR, /*econ=*/1,
        /*m=*/m, /*n=*/n, reinterpret_cast<cuDoubleComplex*>(ws.d_theta),
        /*lda=*/m, ws.d_S, reinterpret_cast<cuDoubleComplex*>(ws.d_U),
        /*ldu=*/m, reinterpret_cast<cuDoubleComplex*>(ws.d_Vt), /*ldv=*/n,
        reinterpret_cast<cuDoubleComplex*>(ws.d_work), ws.lwork, ws.d_info,
        reinterpret_cast<gesvdjInfo_t>(ws.gesvdj_params)));
  }

  int info = 0;
  CUDA_CHECK(cudaMemcpy(&info, ws.d_info, sizeof(int), cudaMemcpyDeviceToHost));
  if (info != 0) {
    std::fprintf(stderr,
                 "cusolverDnZgesvdj info=%d (m=%d n=%d K=%d chivL=%d chivR=%d)\n",
                 info, m, n, K, chivL, chivR);
    std::abort();
  }

  CUDA_CHECK(cudaMemcpy(ws.h_S_pinned, ws.d_S, size_t(K) * sizeof(double),
                        cudaMemcpyDeviceToHost));
  int keep = 0;
  for (int i = 0; i < K; ++i)
    if (ws.h_S_pinned[i] > svd_min) ++keep;
  if (keep > chi_max) keep = chi_max;
  if (keep < 1) keep = 1;

  double norm_sq = 0.0;
  for (int i = 0; i < K; ++i) norm_sq += ws.h_S_pinned[i] * ws.h_S_pinned[i];
  double kept_sq = 0.0;
  for (int i = 0; i < keep; ++i)
    kept_sq += ws.h_S_pinned[i] * ws.h_S_pinned[i];
  double eps = 1.0 - kept_sq;
  double norm = std::sqrt(kept_sq);

  std::vector<double> S_keep(keep);
  for (int i = 0; i < keep; ++i) S_keep[i] = ws.h_S_pinned[i] / norm;

  CUDA_CHECK(cudaMemcpy(ws.d_S, S_keep.data(),
                        size_t(keep) * sizeof(double),
                        cudaMemcpyHostToDevice));

  std::vector<double> Ss_inv(chivL);
  for (int a = 0; a < chivL; ++a) {
    double s = psi->Ss[site][a];
    Ss_inv[a] = (s > 1e-14) ? 1.0 / s : 0.0;
  }
  CUDA_CHECK(cudaMemcpy(ws.d_Ss_in, Ss_inv.data(), chivL * sizeof(double),
                        cudaMemcpyHostToDevice));

  // Step 6 and 7: Vidal restore.
  {
    EventTimer t(timings ? &timings->restore_ms : nullptr);
    {
      int blkx = std::min(64, keep > 0 ? keep : 1);
      dim3 block(blkx, d, 1);
      dim3 grid((keep + block.x - 1) / block.x, chivL, 1);
      vidal_restore_left_kernel<2><<<grid, block>>>(
          reinterpret_cast<cuDoubleComplex*>(ws.d_U), ws.d_Ss_in, ws.d_S,
          reinterpret_cast<cuDoubleComplex*>(ws.d_newBi), chivL, keep, m);
      CUDA_CHECK(cudaGetLastError());
    }
    {
      int blkx = std::min(64, chivR > 0 ? chivR : 1);
      dim3 block(blkx, d, 1);
      dim3 grid((chivR + block.x - 1) / block.x, keep, 1);
      vidal_restore_right_kernel<2><<<grid, block>>>(
          reinterpret_cast<cuDoubleComplex*>(ws.d_Vt),
          reinterpret_cast<cuDoubleComplex*>(ws.d_newBj), keep, chivR, n);
      CUDA_CHECK(cudaGetLastError());
    }
  }

  size_t Bi_bytes =
      size_t(chivL) * d * size_t(keep) * sizeof(cuDoubleComplex);
  size_t Bj_bytes =
      size_t(keep) * d * size_t(chivR) * sizeof(cuDoubleComplex);
  {
    EventTimer t(timings ? &timings->d2h_ms : nullptr);
    CUDA_CHECK(cudaMemcpyAsync(ws.h_outA_pinned, ws.d_newBi, Bi_bytes,
                               cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpyAsync(ws.h_outB_pinned, ws.d_newBj, Bj_bytes,
                               cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  int sh_Bi[3] = {chivL, d, keep};
  Tensor new_Bi(3, sh_Bi, psi->Bs[site].capacity);
  std::memcpy(new_Bi.data, ws.h_outA_pinned,
              size_t(new_Bi.size) * sizeof(cuDoubleComplex));
  psi->Bs[site] = std::move(new_Bi);

  delete[] psi->Ss[site + 1];
  psi->Ss[site + 1] = new double[psi->chi_max];
  std::fill(psi->Ss[site + 1], psi->Ss[site + 1] + psi->chi_max, 0.0);
  for (int i = 0; i < keep; ++i) psi->Ss[site + 1][i] = S_keep[i];
  psi->chi[site + 1] = keep;

  int sh_Bj[3] = {keep, d, chivR};
  Tensor new_Bj(3, sh_Bj, psi->Bs[site + 1].capacity);
  std::memcpy(new_Bj.data, ws.h_outB_pinned,
              size_t(new_Bj.size) * sizeof(cuDoubleComplex));
  psi->Bs[site + 1] = std::move(new_Bj);

  return eps;
}

// ---------------------------------------------------------------------------
// update_bond_gpu_jacobi
//
// Identical to update_bond_gpu but replaces cuSOLVER gesvdj with the custom
// one-sided Jacobi SVD kernel (jacobi_sweep_kernel + jacobi_finish_kernel).
//
// Buffer reuse (no extra allocations):
//   W = d_theta   (col-major A, written by row_to_col_kernel)
//   V = d_Utheta  (free after gate-contract stage)
// ---------------------------------------------------------------------------
double update_bond_gpu_jacobi(MPS* psi, int site, const Tensor& gate,
                               int chi_max, double svd_min,
                               GpuTebdWorkspace& ws,
                               GpuBondTimings* timings) {
  assert(psi->d == ws.d);
  const int d = ws.d;
  const int D2 = d * d;
  if (timings) timings->n_calls += 1;

  Tensor theta = psi->theta2(site);
  int chivL = theta.shape[0];
  int chivR = theta.shape[3];
  int m = chivL * d;
  int n = d * chivR;
  int K = std::min(m, n);
  size_t theta_elems = size_t(chivL) * D2 * chivR;

  if (m > ws.m_max || n > ws.n_max) {
    std::fprintf(stderr,
                 "update_bond_gpu_jacobi: bond too large (m=%d, n=%d)\n", m, n);
    std::abort();
  }
  if (K > 256) {
    std::fprintf(stderr,
                 "update_bond_gpu_jacobi: K=%d > 256 (custom Jacobi limit)\n",
                 K);
    std::abort();
  }

  // Step 1: H2D theta and gate.
  std::memcpy(ws.h_theta_pinned, theta.data,
              theta_elems * sizeof(cuDoubleComplex));
  {
    EventTimer t(timings ? &timings->h2d_ms : nullptr);
    CUDA_CHECK(cudaMemcpyAsync(ws.d_theta, ws.h_theta_pinned,
                               theta_elems * sizeof(cuDoubleComplex),
                               cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyAsync(ws.d_gate, gate.data,
                               D2 * D2 * sizeof(cuDoubleComplex),
                               cudaMemcpyHostToDevice));
  }

  // Step 2: gate contraction.
  {
    EventTimer t(timings ? &timings->gate_kernel_ms : nullptr);
    int blkx = std::min(32, chivR > 0 ? chivR : 1);
    dim3 block(blkx, D2);
    dim3 grid((chivR + block.x - 1) / block.x,
              (D2 + block.y - 1) / block.y, chivL);
    if (d == 2) {
      gate_contract_kernel<2><<<grid, block>>>(
          reinterpret_cast<cuDoubleComplex*>(ws.d_gate),
          reinterpret_cast<cuDoubleComplex*>(ws.d_theta),
          reinterpret_cast<cuDoubleComplex*>(ws.d_Utheta), chivL, chivR);
    } else {
      std::fprintf(stderr, "gate_contract_kernel: only d=2 supported\n");
      std::abort();
    }
    CUDA_CHECK(cudaGetLastError());
  }

  // Step 3: row-major utheta -> col-major in d_theta (W).
  {
    EventTimer t(timings ? &timings->row_to_col_ms : nullptr);
    dim3 block(32, 8);
    dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);
    row_to_col_kernel<<<grid, block>>>(
        reinterpret_cast<cuDoubleComplex*>(ws.d_Utheta),
        reinterpret_cast<cuDoubleComplex*>(ws.d_theta), m, n);
    CUDA_CHECK(cudaGetLastError());
  }

  // Step 4: custom one-sided Jacobi SVD.
  // One-sided Jacobi orthogonalizes columns, so it expects rows >= cols.
  // For wide matrices, decompose A^H (n x m) and swap the singular vector
  // outputs back into A's U/V slots:
  //   A^H = U_b S V_b^H  =>  A = V_b S U_b^H.
  {
    EventTimer t(timings ? &timings->svd_ms : nullptr);
    size_t smem_sweep = size_t(JACOBI_BLOCK + 10) * sizeof(double);
    size_t smem_finish = JF_SMEM_SZ * sizeof(double);

    if (m >= n) {
      // W = A in d_theta (m x n), V in d_Utheta (n x n).
      jacobi_sweep_kernel<<<1, JACOBI_BLOCK, smem_sweep>>>(
          reinterpret_cast<cuDoubleComplex*>(ws.d_theta),
          reinterpret_cast<cuDoubleComplex*>(ws.d_Utheta), m, n);
      CUDA_CHECK(cudaGetLastError());

      jacobi_finish_kernel<<<1, JACOBI_BLOCK, smem_finish>>>(
          reinterpret_cast<cuDoubleComplex*>(ws.d_theta),
          reinterpret_cast<cuDoubleComplex*>(ws.d_Utheta),
          reinterpret_cast<cuDoubleComplex*>(ws.d_U),
          reinterpret_cast<cuDoubleComplex*>(ws.d_Vt), ws.d_S, m, n, K);
      CUDA_CHECK(cudaGetLastError());
    } else {
      // Build W = A^H in d_Utheta (n x m).  Reuse d_theta for V (m x m).
      dim3 block(32, 8);
      dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);
      col_major_adjoint_kernel<<<grid, block>>>(
          reinterpret_cast<const cuDoubleComplex*>(ws.d_theta),
          reinterpret_cast<cuDoubleComplex*>(ws.d_Utheta), m, n);
      CUDA_CHECK(cudaGetLastError());

      jacobi_sweep_kernel<<<1, JACOBI_BLOCK, smem_sweep>>>(
          reinterpret_cast<cuDoubleComplex*>(ws.d_Utheta),
          reinterpret_cast<cuDoubleComplex*>(ws.d_theta), n, m);
      CUDA_CHECK(cudaGetLastError());

      // For A^H, finish writes U_b (n x K) and V_b (m x K).  Store U_b in
      // ws.d_Vt (A's right singular vectors) and V_b in ws.d_U (A's left).
      jacobi_finish_kernel<<<1, JACOBI_BLOCK, smem_finish>>>(
          reinterpret_cast<cuDoubleComplex*>(ws.d_Utheta),
          reinterpret_cast<cuDoubleComplex*>(ws.d_theta),
          reinterpret_cast<cuDoubleComplex*>(ws.d_Vt),
          reinterpret_cast<cuDoubleComplex*>(ws.d_U), ws.d_S, n, m, K);
      CUDA_CHECK(cudaGetLastError());
    }
  }

  // Sync and read back S.
  CUDA_CHECK(cudaMemcpy(ws.h_S_pinned, ws.d_S, size_t(K) * sizeof(double),
                        cudaMemcpyDeviceToHost));
  int keep = 0;
  for (int i = 0; i < K; ++i)
    if (ws.h_S_pinned[i] > svd_min) ++keep;
  if (keep > chi_max) keep = chi_max;
  if (keep < 1) keep = 1;

  double kept_sq = 0.0;
  for (int i = 0; i < keep; ++i)
    kept_sq += ws.h_S_pinned[i] * ws.h_S_pinned[i];
  double eps  = 1.0 - kept_sq;
  double norm = std::sqrt(kept_sq < 1e-300 ? 1.0 : kept_sq);

  // Normalize S in-place in pinned buffer, then H2D.
  double* h_S = reinterpret_cast<double*>(ws.h_S_pinned);
  for (int i = 0; i < keep; ++i) h_S[i] = ws.h_S_pinned[i] / norm;
  CUDA_CHECK(cudaMemcpy(ws.d_S, ws.h_S_pinned,
                        size_t(keep) * sizeof(double), cudaMemcpyHostToDevice));

  std::vector<double> Ss_inv(chivL);
  for (int a = 0; a < chivL; ++a) {
    double s = psi->Ss[site][a];
    Ss_inv[a] = (s > 1e-14) ? 1.0 / s : 0.0;
  }
  CUDA_CHECK(cudaMemcpy(ws.d_Ss_in, Ss_inv.data(),
                        chivL * sizeof(double), cudaMemcpyHostToDevice));

  // Step 6 & 7: Vidal restore (same kernels as cuSOLVER path).
  {
    {
      int blkx = std::min(64, keep > 0 ? keep : 1);
      dim3 block(blkx, d, 1);
      dim3 grid((keep + block.x - 1) / block.x, chivL, 1);
      vidal_restore_left_kernel<2><<<grid, block>>>(
          reinterpret_cast<cuDoubleComplex*>(ws.d_U), ws.d_Ss_in, ws.d_S,
          reinterpret_cast<cuDoubleComplex*>(ws.d_newBi), chivL, keep, m);
      CUDA_CHECK(cudaGetLastError());
    }
    {
      int blkx = std::min(64, chivR > 0 ? chivR : 1);
      dim3 block(blkx, d, 1);
      dim3 grid((chivR + block.x - 1) / block.x, keep, 1);
      vidal_restore_right_kernel<2><<<grid, block>>>(
          reinterpret_cast<cuDoubleComplex*>(ws.d_Vt),
          reinterpret_cast<cuDoubleComplex*>(ws.d_newBj), keep, chivR, n);
      CUDA_CHECK(cudaGetLastError());
    }
  }

  // Step 8: D2H.
  size_t Bi_bytes = size_t(chivL) * d * size_t(keep) * sizeof(cuDoubleComplex);
  size_t Bj_bytes = size_t(keep) * d * size_t(chivR) * sizeof(cuDoubleComplex);
  {
    EventTimer t(timings ? &timings->d2h_ms : nullptr);
    CUDA_CHECK(cudaMemcpyAsync(ws.h_outA_pinned, ws.d_newBi, Bi_bytes,
                               cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpyAsync(ws.h_outB_pinned, ws.d_newBj, Bj_bytes,
                               cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  // Write back to host MPS.
  int sh_Bi[3] = {chivL, d, keep};
  Tensor new_Bi(3, sh_Bi, psi->Bs[site].capacity);
  std::memcpy(new_Bi.data, ws.h_outA_pinned,
              size_t(new_Bi.size) * sizeof(cuDoubleComplex));
  psi->Bs[site] = std::move(new_Bi);

  delete[] psi->Ss[site + 1];
  psi->Ss[site + 1] = new double[psi->chi_max];
  std::fill(psi->Ss[site + 1], psi->Ss[site + 1] + psi->chi_max, 0.0);
  for (int i = 0; i < keep; ++i) psi->Ss[site + 1][i] = h_S[i];
  psi->chi[site + 1] = keep;

  int sh_Bj[3] = {keep, d, chivR};
  Tensor new_Bj(3, sh_Bj, psi->Bs[site + 1].capacity);
  std::memcpy(new_Bj.data, ws.h_outB_pinned,
              size_t(new_Bj.size) * sizeof(cuDoubleComplex));
  psi->Bs[site + 1] = std::move(new_Bj);

  return eps;
}

void tebd_step_gpu(TEBDEngine& eng, GpuTebdWorkspace& ws) {
  int L = eng.model->L;
  for (int s = 0; s < eng.n_schedule; ++s) {
    int start = eng.schedule[s].parity;
    Tensor* gates = eng.gate_sets[eng.schedule[s].frac_idx];
    for (int b = start; b < L - 1; b += 2) {
      double eps = update_bond_gpu(eng.psi, b, gates[b], eng.chi_max,
                                   eng.svd_min, ws);
      eng.trunc_err_eps = eng.trunc_err_eps + eps - eng.trunc_err_eps * eps;
    }
  }
}

// ===========================================================================
// Persistent device MPS
// ===========================================================================

// ---------------------------------------------------------------------------
// theta2_gpu_kernel
//
// Builds the two-site tensor theta entirely on device, reading from
// device-resident B tensors and Schmidt values.
//
//   theta[vL, i, j, vR] = Ss[vL] * sum_k Bi[vL, i, k] * Bj[k, j, vR]
//
// Shapes (all row-major):
//   Bi    : (chivL, D, chi_mid)
//   Bj    : (chi_mid, D, chivR)
//   Ss    : chi_mid >= chivL values (only first chivL used)
//   theta : (chivL, D, D, chivR)
//
// Grid/block mirror gate_contract_kernel: grid(vR-tiles, D2, chivL),
// block(blkx, D2).  Each thread computes one (vL, i*D+j, vR) element.
// ---------------------------------------------------------------------------
template <int D>
__global__ void theta2_gpu_kernel(
    const cuDoubleComplex* __restrict__ Bi,
    const cuDoubleComplex* __restrict__ Bj,
    const double* __restrict__ Ss,
    cuDoubleComplex* __restrict__ theta,
    int chivL, int chi_mid, int chivR) {
  constexpr int D2 = D * D;
  int vR  = blockIdx.x * blockDim.x + threadIdx.x;
  int ij  = blockIdx.y * blockDim.y + threadIdx.y;  // i*D + j
  int vL  = blockIdx.z;
  if (vR >= chivR || ij >= D2 || vL >= chivL) return;
  int i = ij / D, j = ij % D;

  double scale = Ss[vL];
  cuDoubleComplex acc = make_cuDoubleComplex(0.0, 0.0);
  for (int k = 0; k < chi_mid; ++k) {
    // Bi[vL, i, k] = Bi[(vL*D + i)*chi_mid + k]
    cuDoubleComplex bi = Bi[(vL * D + i) * chi_mid + k];
    // Bj[k, j, vR] = Bj[(k*D + j)*chivR + vR]
    cuDoubleComplex bj = Bj[(k * D + j) * chivR + vR];
    cuDoubleComplex p = cd_mul(bi, bj);
    acc.x += p.x;
    acc.y += p.y;
  }
  // theta[vL, i, j, vR] = theta[((vL*D+i)*D+j)*chivR + vR]
  theta[((vL * D + i) * D + j) * chivR + vR] =
      make_cuDoubleComplex(scale * acc.x, scale * acc.y);
}

// ---------------------------------------------------------------------------
// compute_ss_inv_kernel
//
// Computes element-wise 1/Ss for the vidal_restore_left_kernel.
// Writes zeros for near-zero entries (threshold 1e-14).
// ---------------------------------------------------------------------------
__global__ void compute_ss_inv_kernel(const double* __restrict__ Ss,
                                      double* __restrict__ Ss_inv, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  double s = Ss[i];
  Ss_inv[i] = (s > 1e-14) ? 1.0 / s : 0.0;
}

// ---------------------------------------------------------------------------
// GpuMps
// ---------------------------------------------------------------------------

GpuMps::GpuMps(const MPS& mps)
    : L(mps.L), d(mps.d), chi_max(mps.chi_max),
      d_Bs(nullptr), d_Ss(nullptr), chi(nullptr) {
  d_Bs = new void*[L];
  d_Ss = new void*[L];
  chi  = new int[L + 1];
  for (int i = 0; i <= L; ++i) chi[i] = mps.chi[i];

  size_t Bsize = sizeof(cuDoubleComplex) * size_t(chi_max) * d * chi_max;
  size_t Ssize = sizeof(double) * size_t(chi_max);

  for (int i = 0; i < L; ++i) {
    CUDA_CHECK(cudaMalloc(&d_Bs[i], Bsize));
    size_t elems = size_t(chi[i]) * d * chi[i + 1];
    CUDA_CHECK(cudaMemcpy(d_Bs[i], mps.Bs[i].data,
                          elems * sizeof(cuDoubleComplex),
                          cudaMemcpyHostToDevice));
  }
  for (int i = 0; i < L; ++i) {
    CUDA_CHECK(cudaMalloc(&d_Ss[i], Ssize));
    CUDA_CHECK(cudaMemcpy(d_Ss[i], mps.Ss[i],
                          size_t(chi[i]) * sizeof(double),
                          cudaMemcpyHostToDevice));
  }
}

GpuMps::~GpuMps() {
  if (d_Bs) {
    for (int i = 0; i < L; ++i) cudaFree(d_Bs[i]);
    delete[] d_Bs;
  }
  if (d_Ss) {
    for (int i = 0; i < L; ++i) cudaFree(d_Ss[i]);
    delete[] d_Ss;
  }
  delete[] chi;
}

void GpuMps::sync_to_host(MPS& mps) const {
  assert(mps.L == L && mps.d == d && mps.chi_max == chi_max);
  for (int i = 0; i <= L; ++i) mps.chi[i] = chi[i];
  for (int i = 0; i < L; ++i) {
    size_t elems = size_t(chi[i]) * d * chi[i + 1];
    // Rebuild Tensor with correct shape (reuse existing capacity).
    int sh[3] = {chi[i], d, chi[i + 1]};
    Tensor new_B(3, sh, mps.Bs[i].capacity);
    CUDA_CHECK(cudaMemcpy(new_B.data, d_Bs[i],
                          elems * sizeof(cuDoubleComplex),
                          cudaMemcpyDeviceToHost));
    mps.Bs[i] = std::move(new_B);
    // Sync Schmidt values.
    CUDA_CHECK(cudaMemcpy(mps.Ss[i], d_Ss[i],
                          size_t(chi[i]) * sizeof(double),
                          cudaMemcpyDeviceToHost));
  }
}

// ---------------------------------------------------------------------------
// update_bond_gpu_persistent
//
// Like update_bond_gpu but reads/writes device-resident GpuMps instead of
// host MPS.  Eliminates H2D/D2H for the B tensors; only small transfers
// remain (gate H2D, singular-value D2H for truncation, S_keep H2D).
// All GPU operations are queued on ws.stream.
// ---------------------------------------------------------------------------
double update_bond_gpu_persistent(GpuMps& gmps, int site, const Tensor& gate,
                                   int chi_max, double svd_min,
                                   GpuTebdWorkspace& ws,
                                   GpuBondTimings* timings) {
  assert(gmps.d == ws.d);
  auto* h = reinterpret_cast<GpuHandles*>(ws.handle);
  cudaStream_t st = reinterpret_cast<cudaStream_t>(ws.stream);
  const int d = ws.d;
  const int D2 = d * d;
  if (timings) timings->n_calls += 1;

  // Bond geometry.
  int chivL   = gmps.chi[site];
  int chi_mid = gmps.chi[site + 1];
  int chivR   = gmps.chi[site + 2];
  int m = chivL * d, n = d * chivR;
  int K = std::min(m, n);

  if (m > ws.m_max || n > ws.n_max) {
    std::fprintf(stderr,
                 "update_bond_gpu_persistent: bond too large (m=%d>%d or "
                 "n=%d>%d) at site=%d\n",
                 m, ws.m_max, n, ws.n_max, site);
    std::abort();
  }

  // Set cuSOLVER stream.
  CUSOLVER_CHECK(cusolverDnSetStream(h->cusolver, st));

  // Step 1: Build theta2 on device from d_Bs[site] and d_Bs[site+1].
  {
    EventTimer t(timings ? &timings->gate_kernel_ms : nullptr);
    int blkx = std::min(32, chivR > 0 ? chivR : 1);
    dim3 block(blkx, D2);
    dim3 grid((chivR + blkx - 1) / blkx, 1, chivL);
    if (d == 2) {
      theta2_gpu_kernel<2><<<grid, block, 0, st>>>(
          reinterpret_cast<const cuDoubleComplex*>(gmps.d_Bs[site]),
          reinterpret_cast<const cuDoubleComplex*>(gmps.d_Bs[site + 1]),
          reinterpret_cast<const double*>(gmps.d_Ss[site]),
          reinterpret_cast<cuDoubleComplex*>(ws.d_theta),
          chivL, chi_mid, chivR);
    } else {
      std::fprintf(stderr,
                   "theta2_gpu_kernel: only d=2 supported in this build\n");
      std::abort();
    }
    CUDA_CHECK(cudaGetLastError());

    // Step 2: H2D gate (256 bytes; tiny transfer).
    CUDA_CHECK(cudaMemcpyAsync(ws.d_gate, gate.data,
                               D2 * D2 * sizeof(cuDoubleComplex),
                               cudaMemcpyHostToDevice, st));

    // Step 3: Gate contraction.
    int blkx2 = std::min(32, chivR > 0 ? chivR : 1);
    dim3 blk2(blkx2, D2);
    dim3 grd2((chivR + blkx2 - 1) / blkx2, 1, chivL);
    gate_contract_kernel<2><<<grd2, blk2, 0, st>>>(
        reinterpret_cast<cuDoubleComplex*>(ws.d_gate),
        reinterpret_cast<cuDoubleComplex*>(ws.d_theta),
        reinterpret_cast<cuDoubleComplex*>(ws.d_Utheta), chivL, chivR);
    CUDA_CHECK(cudaGetLastError());
  }

  // Step 4: row-major utheta -> col-major A.
  {
    EventTimer t(timings ? &timings->row_to_col_ms : nullptr);
    dim3 block(32, 8);
    dim3 grid((n + 31) / 32, (m + 7) / 8);
    row_to_col_kernel<<<grid, block, 0, st>>>(
        reinterpret_cast<cuDoubleComplex*>(ws.d_Utheta),
        reinterpret_cast<cuDoubleComplex*>(ws.d_theta), m, n);
    CUDA_CHECK(cudaGetLastError());
  }

  // Step 5: Compute 1/Ss_in on device (replaces host-side H2D).
  {
    int blkx = std::min(128, chivL > 0 ? chivL : 1);
    compute_ss_inv_kernel<<<(chivL + blkx - 1) / blkx, blkx, 0, st>>>(
        reinterpret_cast<const double*>(gmps.d_Ss[site]),
        ws.d_Ss_in, chivL);
    CUDA_CHECK(cudaGetLastError());
  }

  // Step 6: SVD via cuSOLVER Jacobi.
  {
    int need_lwork = 0;
    CUSOLVER_CHECK(cusolverDnZgesvdj_bufferSize(
        h->cusolver, CUSOLVER_EIG_MODE_VECTOR, /*econ=*/1, m, n,
        reinterpret_cast<cuDoubleComplex*>(ws.d_theta), /*lda=*/m, ws.d_S,
        reinterpret_cast<cuDoubleComplex*>(ws.d_U), /*ldu=*/m,
        reinterpret_cast<cuDoubleComplex*>(ws.d_Vt), /*ldv=*/n, &need_lwork,
        reinterpret_cast<gesvdjInfo_t>(ws.gesvdj_params)));
    if (need_lwork > ws.lwork) {
      cudaFree(ws.d_work);
      ws.d_work = nullptr;
      CUDA_CHECK(cudaMalloc(&ws.d_work,
                            sizeof(cuDoubleComplex) * size_t(need_lwork)));
      ws.lwork = need_lwork;
    }
  }
  {
    EventTimer t(timings ? &timings->svd_ms : nullptr);
    CUSOLVER_CHECK(cusolverDnZgesvdj(
        h->cusolver, CUSOLVER_EIG_MODE_VECTOR, /*econ=*/1,
        m, n, reinterpret_cast<cuDoubleComplex*>(ws.d_theta), m,
        ws.d_S, reinterpret_cast<cuDoubleComplex*>(ws.d_U), m,
        reinterpret_cast<cuDoubleComplex*>(ws.d_Vt), n,
        reinterpret_cast<cuDoubleComplex*>(ws.d_work), ws.lwork, ws.d_info,
        reinterpret_cast<gesvdjInfo_t>(ws.gesvdj_params)));
  }

  // Sync stream to get singular values on host (small D2H: K doubles).
  CUDA_CHECK(cudaMemcpyAsync(ws.h_S_pinned, ws.d_S,
                             size_t(K) * sizeof(double),
                             cudaMemcpyDeviceToHost, st));
  CUDA_CHECK(cudaStreamSynchronize(st));

  int info = 0;
  CUDA_CHECK(cudaMemcpy(&info, ws.d_info, sizeof(int), cudaMemcpyDeviceToHost));
  if (info != 0) {
    std::fprintf(stderr,
                 "cusolverDnZgesvdj (persistent) info=%d "
                 "(m=%d n=%d K=%d site=%d)\n",
                 info, m, n, K, site);
    std::abort();
  }

  // Host-side truncation and normalization.
  int keep = 0;
  for (int i = 0; i < K; ++i)
    if (ws.h_S_pinned[i] > svd_min) ++keep;
  if (keep > chi_max) keep = chi_max;
  if (keep < 1) keep = 1;

  double kept_sq = 0.0;
  for (int i = 0; i < keep; ++i)
    kept_sq += ws.h_S_pinned[i] * ws.h_S_pinned[i];
  double eps = 1.0 - kept_sq;
  double norm = std::sqrt(kept_sq);

  // Write normalized S_keep into the pinned staging buffer (safe: we are done
  // reading the raw singular values from it after the stream sync above).
  double* h_S = reinterpret_cast<double*>(ws.h_S_pinned);
  for (int i = 0; i < keep; ++i) h_S[i] = ws.h_S_pinned[i] / norm;

  // H2D normalized S_keep -> d_S (for vidal restore kernels).
  // Using the pinned buffer avoids dangling-pointer issues with async copies.
  CUDA_CHECK(cudaMemcpyAsync(ws.d_S, ws.h_S_pinned,
                             size_t(keep) * sizeof(double),
                             cudaMemcpyHostToDevice, st));

  // H2D normalized S_keep -> d_Ss[site+1] (new Schmidt spectrum on device).
  // Both copies reference ws.h_S_pinned which lives until workspace destructor.
  CUDA_CHECK(cudaMemcpyAsync(gmps.d_Ss[site + 1], ws.h_S_pinned,
                             size_t(keep) * sizeof(double),
                             cudaMemcpyHostToDevice, st));

  // Step 7: Vidal restore — write directly into d_Bs on device.
  {
    EventTimer t(timings ? &timings->restore_ms : nullptr);
    {
      int blkx = std::min(64, keep > 0 ? keep : 1);
      dim3 block(blkx, d, 1);
      dim3 grid((keep + blkx - 1) / blkx, chivL, 1);
      vidal_restore_left_kernel<2><<<grid, block, 0, st>>>(
          reinterpret_cast<cuDoubleComplex*>(ws.d_U), ws.d_Ss_in, ws.d_S,
          reinterpret_cast<cuDoubleComplex*>(gmps.d_Bs[site]),
          chivL, keep, m);
      CUDA_CHECK(cudaGetLastError());
    }
    {
      int blkx = std::min(64, chivR > 0 ? chivR : 1);
      dim3 block(blkx, d, 1);
      dim3 grid((chivR + blkx - 1) / blkx, keep, 1);
      vidal_restore_right_kernel<2><<<grid, block, 0, st>>>(
          reinterpret_cast<cuDoubleComplex*>(ws.d_Vt),
          reinterpret_cast<cuDoubleComplex*>(gmps.d_Bs[site + 1]),
          keep, chivR, n);
      CUDA_CHECK(cudaGetLastError());
    }
  }

  // Update host-side bond dimension.
  gmps.chi[site + 1] = keep;

  return eps;
}

// ---------------------------------------------------------------------------
// tebd_step_gpu_persistent
//
// One Trotter step using device-resident MPS.  n_ws workspaces are assigned
// round-robin across independent bonds in each even/odd sub-step.  With
// distinct ws.stream values, the GPU can overlap SVD work across streams.
// ---------------------------------------------------------------------------
void tebd_step_gpu_persistent(TEBDEngine& eng, GpuMps& gmps,
                               GpuTebdWorkspace* const* ws_arr, int n_ws) {
  int L = eng.model->L;
  for (int s = 0; s < eng.n_schedule; ++s) {
    int start = eng.schedule[s].parity;
    Tensor* gates = eng.gate_sets[eng.schedule[s].frac_idx];
    int bond_idx = 0;
    for (int b = start; b < L - 1; b += 2, ++bond_idx) {
      GpuTebdWorkspace& ws = *ws_arr[bond_idx % n_ws];
      double eps = update_bond_gpu_persistent(gmps, b, gates[b], eng.chi_max,
                                              eng.svd_min, ws);
      eng.trunc_err_eps = eng.trunc_err_eps + eps - eng.trunc_err_eps * eps;
    }
    // Synchronize ALL streams after this sub-step.  The vidal_restore and
    // S_keep H2D ops are still in-flight on their streams; the next
    // sub-step's theta2_kernel will read those updated d_Bs / d_Ss values.
    CUDA_CHECK(cudaDeviceSynchronize());
  }
}

// ---------------------------------------------------------------------------
// Stream management helpers exposed to host .cpp files.
// ---------------------------------------------------------------------------

void* gpu_stream_create() {
  cudaStream_t st;
  CUDA_CHECK(cudaStreamCreate(&st));
  return reinterpret_cast<void*>(st);
}

void gpu_stream_destroy(void* stream) {
  if (stream) {
    cudaStream_t st = reinterpret_cast<cudaStream_t>(stream);
    CUDA_CHECK(cudaStreamDestroy(st));
  }
}

void gpu_device_sync() {
  CUDA_CHECK(cudaDeviceSynchronize());
}

int gpu_device_count() {
  int n = 0;
  cudaGetDeviceCount(&n);
  return n;
}

void gpu_set_device(int device) {
  CUDA_CHECK(cudaSetDevice(device));
}
