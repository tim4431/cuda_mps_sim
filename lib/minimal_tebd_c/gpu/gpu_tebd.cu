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
// Handle bundle.
// ---------------------------------------------------------------------------

struct GpuHandles {
  cublasHandle_t cublas;
  cusolverDnHandle_t cusolver;
};

GpuTebdWorkspace::GpuTebdWorkspace(int d, int chi_max)
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
      h_S_pinned(nullptr) {
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

double update_bond_gpu(MPS* psi, int site, const Tensor& gate, int chi_max,
                       double svd_min, GpuTebdWorkspace& ws) {
  assert(psi->d == ws.d);
  auto* h = reinterpret_cast<GpuHandles*>(ws.handle);
  const int d = ws.d;
  const int D2 = d * d;
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
    CUDA_CHECK(cudaMemcpyAsync(ws.d_theta, ws.h_theta_pinned,
                               theta_elems * sizeof(cuDoubleComplex),
                               cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyAsync(ws.d_gate, gate.data,
                               D2 * D2 * sizeof(cuDoubleComplex),
                               cudaMemcpyHostToDevice));
  }

  // Step 2: gate contraction.
  {
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
