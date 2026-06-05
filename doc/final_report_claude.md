# GPU-Accelerated TEBD for the 1D Transverse-Field Ising Model

**Final Report — CME 213: Parallel Computing with CUDA, MPI and OpenMP**

**Team members:** Hercy Shen, Chiling Han, Xin Wei
**Date:** June 8, 2026

**Hardware/software.** Multi-GPU benchmarks run on the course cluster partition
`gpu-turing` (NVIDIA Quadro RTX 6000, Turing `sm_75`, 24 GB, NVHPC 24.1, CUDA
12.3, OpenMPI), one GPU per MPI rank. Single-GPU kernel development and the
crossover study additionally used a local NVIDIA RTX 5090 (`sm_120`, 32 GB,
CUDA 12.9). The CPU baseline is built with `g++ -O2 -std=c++14` on one host
core. The code is a from-scratch C++14/CUDA/MPI implementation: every linear
algebra routine (GEMM, GEMV, SVD, `expm`) is hand-written, with cuBLAS/cuSOLVER
used only for the production SVD, so every kernel is visible to the profiler.

---

## 1. Project description

We simulate the **real-time quench dynamics** of the one-dimensional
transverse-field Ising model (TFIM), a canonical quantum many-body system,

$$H = -J\sum_i \sigma^x_i\sigma^x_{i+1} \;-\; g\sum_i \sigma^z_i .$$

A chain of $L$ spin-$\tfrac12$ sites lives in a Hilbert space of dimension
$2^L$, so storing the wavefunction exactly is impossible beyond $L\approx 30$.
The key physical fact that makes simulation tractable is that low-energy and
short-time states of 1D local Hamiltonians have **bounded entanglement**, and
can be compressed into a **Matrix Product State (MPS)**: the $2^L$ amplitudes
are factored into $L$ small rank-3 tensors connected by *bond* indices of
dimension $\chi$. The MPS stores only $O(L\,\chi^2 d)$ numbers ($d=2$), where
$\chi$ grows with the entanglement across each cut.

**Computation.** We evolve the state with **Time-Evolving Block Decimation
(TEBD)**. The propagator $e^{-iHt}$ is Suzuki–Trotter split into a product of
nearest-neighbour two-site gates $U_b=e^{-i\,\Delta t\,H_b}$. Each Trotter step
applies $L-1$ gates; **applying one gate is one small complex SVD** that
re-compresses the two-site tensor back to bond dimension $\chi$. This SVD is the
hot spot — it consumes $>99.6\%$ of GPU kernel time — and is the target of all
parallel work in this project.

- **Inputs:** chain length $L$; couplings $(J,g)$; an initial product state;
  Trotter step $dt$; number of steps $N$; max bond dimension $\chi_{\max}$;
  Trotter order (1, 2, or 4).
- **Outputs:** time series of site magnetizations $\langle\sigma^z_i\rangle$,
  $\langle\sigma^x_i\rangle$, two-point correlations $\langle\sigma^x_i\sigma^x_j\rangle$,
  the bipartite von-Neumann **entanglement entropy** on every cut, the
  bond-dimension profile $\chi(t)$, and the accumulated SVD truncation error.
- **Assumptions/simplifications:** open boundary conditions, local dimension
  $d=2$, nearest-neighbour bonds only.

**Why this matters, and why parallelism.** Entanglement grows under a quench, so
the bond dimension $\chi(t)$ climbs over time (Fig. 1) until it saturates the
budget $\chi_{\max}$. Because each bond update costs $O(\chi^3)$, doubling the
*effective* bond dimension is an $8\times$ hit, and the cost of mapping out the
physics — sweeping the $(J,g)$ phase diagram, each point a full evolution — grows
quickly. A single GPU accelerates each evolution once $\chi$ is large; **multiple
GPUs are needed** to turn a phase-diagram sweep that takes minutes-per-point into
a study that finishes in wall-clock minutes. This is exactly the regime where a
CUDA+MPI implementation pays off.

![Reference TEBD dynamics: mid-chain entropy grows roughly linearly after the
quench, driving the active bond dimension up to the $\chi_{\max}$ cap, after
which truncation error begins to accumulate.](figs/summary.png)

*Figure 1. Reference $L=20$ TFIM quench ($J=g=1$, order 4, $\chi_{\max}=64$).
Entanglement growth drives bond-dimension growth, which sets the SVD sizes that
dominate runtime.*

---

## 2. Algorithms and state of the art

**Core algorithm.** We implement TEBD on a **right-canonical MPS in Vidal
$\Gamma$–$\Lambda$ (B-) form**, the standard for real-time evolution because it
keeps the Schmidt spectrum $\Lambda$ on every bond explicit and makes truncation
a local, optimal (in the 2-norm) operation. The two-site update is:
(1) contract the gate with the two-site tensor $\Theta$; (2) SVD the reshaped
$(\chi_L d)\times(d\chi_R)$ matrix; (3) truncate to the largest $\chi_{\max}$
singular values and renormalize; (4) restore Vidal form. The Trotter layer
supports orders 1, 2 (symmetric), and 4 (Forest–Ruth), so the user can trade
gate count for $O(dt^p)$ accuracy.

**State of the art.** The MPS/TEBD foundations are Vidal's original
time-evolution papers and Schollwöck's DMRG/MPS review. Mature library
implementations — **TeNPy**, **ITensor**, **quimb**, **TensorNetwork** — are
CPU/Python-centric and highly general (arbitrary lattices, symmetries, MPO
evolution). On GPUs, NVIDIA's **cuTensorNet** (cuQuantum) provides
library-level batched contraction and SVD. The main algorithmic variants
applicable to our problem are: (i) the SVD engine — one-sided **Jacobi**
(Brent–Luk pattern), QR-based (`gesvd`), or **randomized** SVD; (ii) the time
integrator — Trotter–Suzuki (used here) versus global Krylov/TDVP; and
(iii) the parallel decomposition — single-chain *spatial* parallelism versus
*task* parallelism over independent runs.

**Our choice and how it compares.** We deliberately do **not** try to beat
cuTensorNet or TeNPy on raw throughput. Instead we build a clean, fully
instrumented TEBD in which *every* kernel is visible, so we can attribute time
to gate contraction, SVD, truncation, and PCIe traffic without a library hiding
the cost. For the SVD we chose **one-sided Jacobi**: at our matrix sizes
($2\chi \lesssim 256$) it converges in 5–10 sweeps and avoids the QR
preprocessing that `gesvd` only amortizes on large matrices, and it matches our
hand-written CPU reference so the CPU↔GPU correctness comparison is
apples-to-apples. The production path uses cuSOLVER's `gesvdj`; we additionally
wrote a **custom one-sided Jacobi CUDA kernel** as the from-scratch deliverable
and to quantify the gap to the vendor library (§7).

---

## 3. Parallelization strategy (CUDA + MPI)

### 3.1 The central difficulty and the decomposition we chose

A single TEBD chain has **limited spatial parallelism**. Within one Trotter
sub-step the even bonds $\{0,2,4,\dots\}$ and odd bonds $\{1,3,5,\dots\}$ touch
disjoint tensors and are independent ($\sim(L-1)/2$-way), but the even and odd
sub-steps are *sequential*, and a *spatial* decomposition that splits the chain
across ranks would require **one MPI round-trip per sub-step** to exchange the
boundary Schmidt spectrum. At our target sizes ($L\le 80$) the per-step compute
is far too small to hide that latency — communication would dominate.

We therefore use **task parallelism**: each MPI rank owns one GPU and runs
*complete, independent* TEBD evolutions for a set of $(J,g)$ points in the phase
diagram, with **zero in-loop communication**. This is the genuinely useful
multi-GPU mode for this problem, and it mirrors how production studies actually
use libraries like ITensor (independent parameter runs). We rejected two
alternatives explicitly: (a) *spatial chain decomposition* — infeasible at small
$L$ due to per-sub-step synchronization; (b) *replicated-parameter + distributed
chain* — needs per-bond MPI barriers, same problem.

### 3.2 Single-GPU bond-update pipeline (CUDA kernels)

The bond update is implemented as a five-stage device pipeline. With $m=\chi_L d$,
$n=d\chi_R$, $K=\min(m,n)$ and block shapes for $d=2$:

| # | Stage | Kernel / call | Thread/block organization & GPU features |
|---|---|---|---|
| 1 | gate contraction | `gate_contract_kernel<2>` | grid $(\lceil\chi_R/32\rceil, d^2, \chi_L)\times(32,d^2)$; the 16-element gate staged in **shared memory** via a cooperative strided load, `#pragma unroll` over the inner $d^2$ |
| 2 | row→col layout | `row_to_col_kernel` | $(\lceil n/32\rceil,\lceil m/8\rceil)\times(32,8)$; **coalesced** reads to feed the column-major SVD |
| 3 | SVD | `cusolverDnZgesvdj` (`econ=1`) | Jacobi, $K=\min(m,n)$; `lwork` re-queried per call (see §9) |
| 4 | Vidal restore (left) | `vidal_restore_left_kernel<2>` | elementwise $\text{new}B_i[a,p,b]=U[a d{+}p,b]\,S[b]/\Lambda_a$ |
| 5 | Vidal restore (right) | `vidal_restore_right_kernel<2>` | elementwise, in-place conjugation of $V$ |

Every stage is bracketed by CUDA events (gated by an optional `GpuBondTimings*`)
so `bench_breakdown` can report per-stage time. The custom kernels exploit shared
memory for the gate, coalesced layout transforms, and template specialization on
$d=2$; they are individually tiny ($<0.4\%$ of kernel time, §6), which is the
*point* — the SVD is where the work is.

### 3.3 Persistent device-resident MPS and streams

Two single-GPU optimizations matter for the design:

- **Persistent device MPS (`GpuMps`).** Milestone 3 staged the two-site tensor
  through host memory on *every* bond. The final implementation keeps all $L$
  B-tensors ($\mathbb{C}^{\chi_{\max}\times d\times\chi_{\max}}$) and Schmidt
  spectra device-resident for the whole evolution. `theta2_gpu_kernel` builds the
  two-site tensor on device and `compute_ss_inv_kernel` inverts the left Schmidt
  spectrum on device, so the **bulk tensor transfers are eliminated** — profiled
  H2D/D2H drops from tens of MB to $\sim0.2$ MB. All device buffers are
  pre-allocated once at engine construction (`GpuTebdWorkspace`), so there is
  **zero `cudaMalloc` during evolution**. Host↔device sync happens only at
  measurement points.

- **Streams within a sub-step.** `tebd_step_gpu_persistent` accepts $n$
  workspaces, each with its own `cudaStream_t` and stream-bound cuSOLVER handle,
  and assigns independent bonds round-robin, closing each sub-step with one
  `cudaDeviceSynchronize`. *However*, truncation (deciding how many singular
  values to keep) is done host-side, forcing a `cudaStreamSynchronize` mid-bond
  to read the singular values back — so bonds run essentially serially and the
  measured stream speedup is negligible. This is an honest negative result that
  the performance model (§4) predicts.

### 3.4 MPI decomposition and communication pattern

Each rank, after `MPI_Init`, calls `cudaSetDevice(rank % n_gpus)` (one GPU per
rank on a 4-GPU node; by local rank on multi-node jobs), builds its own $(J,g)$
points *locally* (no scatter of parameters), and runs independent evolutions on
its own `GpuMps`, gate cache, and cuSOLVER handle. **The TEBD loop contains no
MPI calls.** Only at the very end does rank 0 collect results, using **MPI
collectives** as encouraged:

1. `MPI_Gather` of one `int` per rank (each rank's record count), then
2. `MPI_Gatherv` of the variable-length `(J,g,t_\text{wall},\chi_{\max},S_\text{mid})`
   records — `Gatherv` handles the uneven per-rank counts of strong mode.

MPI never touches device buffers: `GpuMps::sync_to_host` copies at most
$L\cdot\chi_{\max}$ doubles ($\approx10$ KB) into a host vector first. Because
communication occurs strictly *after* all GPU work completes, there is no
computation–communication overlap to manage — and, as §4–§6 show, none is needed.

`sweep_mpi` exposes two modes: **strong** ($4\times4=16$ fixed $(J,g)$ pairs
split round-robin across ranks) and **weak** (4 new $(J,g)$ pairs per rank, total
work grows with $P$).

---

## 4. Performance model

We build a quantitative, falsifiable model from three pieces: a per-bond compute
model, a communication model, and a parallel-efficiency model.

**(a) Per-bond compute and the latency↔compute crossover.** The dominant cost is
the SVD of a $2\chi_a\times2\chi_a$ complex matrix, where $\chi_a(t)$ is the
*active* (realized) bond dimension. One-sided Jacobi does $O(\chi_a^3)$ flops per
sweep over $\sim5$–$10$ sweeps, so per-bond compute is $t_\text{svd}\approx c\,\chi_a^3$.
Each kernel launch and the cuSOLVER dispatch carry a fixed overhead
$t_0$. Modeling one full step as $L-1$ bonds:

$$T_\text{step}(L,\chi_a)\;\approx\;(L-1)\,\big(t_0 + c\,\chi_a^3\big).$$

This makes two **falsifiable predictions**: (i) when $c\chi_a^3\ll t_0$ the code
is **latency-bound**, $T$ is *linear in $L$* and *flat in $\chi_{\max}$* (since
$\chi_a$, not $\chi_{\max}$, sets the matrix size); (ii) when $c\chi_a^3\gg t_0$
the code is **compute-bound**, $T\propto\chi_a^3$. The crossover is at the $\chi_a$
where $c\chi_a^3\approx t_0$.

**(b) Roofline / arithmetic intensity.** Each Jacobi sweep streams the
$O(\chi_a^2)$-word matrix (16 B/complex) and does $O(\chi_a^3)$ flops, so
arithmetic intensity $I=O(\chi_a)$ flops/byte *grows with $\chi_a$*. Small
matrices sit at low $I$ and below the launch-overhead floor (latency-bound);
large matrices climb past the roofline ridge into the compute-bound region. The
custom gate kernel does $O(\chi^2)$ work and is far below the FP64 peak — it is
launch-bound, not compute-bound, consistent with the model.

**(c) Communication model ($\alpha+\beta n$).** The only message is the final
`Gatherv` of $R$ records of $\sim$40 B each, $n\lesssim10$ KB total. With latency
$\alpha\sim$ a few $\mu$s and inverse-bandwidth $\beta n$ negligible at this size,
$T_\text{comm}\approx\alpha\ll T_\text{compute}$. **Prediction:** the
communication fraction is $<10^{-4}$, i.e. invisible in a profiler.

**(d) Parallel efficiency (task-parallel Amdahl/load-balance).** With $P$ ranks
and per-pair costs $t_i$, total wall time is the *slowest rank*:
$T(P)=\max_r\sum_{i\in r} t_i$, against an ideal $\big(\sum_i t_i\big)/P$. Strong
efficiency is therefore $E=\dfrac{\sum_i t_i}{P\,\max_r\sum_{i\in r}t_i}$, set
**entirely by load imbalance**, not communication. Because pair cost rises
steeply with $J$ (more entanglement → larger $\chi_a$ → $O(\chi_a^3)$), the
hardest pairs cost $\sim2.2\times$ the cheapest. Round-robin gives one rank all
the hardest pairs, so the model predicts $E\approx80\%$ at $P=4$. For **weak**
scaling, each added rank is handed *higher-$J$* (more expensive) pairs, so
per-rank work grows with $P$ and naive weak efficiency must fall — this is a
property of non-uniform work, not of communication.

---

## 5. Benchmarking and instrumentation

**Instrumentation.** Per-stage GPU time is measured with **CUDA events** around
each pipeline stage (`bench_breakdown`); full-step wall time via host timers
(`bench_gpu`); SVD-engine comparison via `bench_svd`; and end-to-end timelines
with **Nsight Systems** (`nsys profile --stats=true`), run both single-GPU and
under `mpirun -n 4` with a per-rank report. The MPI driver reports
`t_rank_max_s` $=\max_r\sum_{i\in r}t_i$ directly.

**Single-GPU full-step timing (`bench_gpu`, $L=20$, 20 steps, `gpu-turing`):**

| $\chi_{\max}$ | CPU (s) | M3 GPU (s) | Persistent (s) | Streamed (s) | CPU/Persistent |
|---:|---:|---:|---:|---:|---:|
| 16  | 2.049  | 5.372  | 5.180  | 5.183  | 0.40 |
| 32  | 9.601  | 13.196 | 12.992 | 13.064 | 0.74 |
| 48  | 20.073 | 17.428 | 17.075 | 17.079 | 1.18 |
| 64  | 24.099 | 18.250 | 17.709 | 17.745 | 1.36 |
| 96  | 24.484 | 18.269 | 17.722 | 17.854 | 1.38 |
| 128 | 24.533 | 18.334 | 17.848 | 17.853 | 1.37 |

*Table 1. The GPU overtakes the CPU at $\chi_{\max}\ge48$; persistent MPS is
consistently 1–3% faster than M3 host-staging; four streams add nothing.*

![M4 single-GPU timing: persistent MPS consistently beats the M3 host-staging
path; four streams yield negligible improvement.](figs/m4_gpu_timing.png)

*Figure 2. Single-GPU full-step timing vs. $\chi_{\max}$.*

**Per-stage kernel-time breakdown (`bench_breakdown` / Nsight `cuda_gpu_kern_sum`):**

| GPU kernel group | Share of kernel time | Note |
|---|---:|---|
| cuSOLVER `gesvdbj` + `svd_col/row_rotate` | 96.4% | top-3 Jacobi SVD kernels |
| all remaining cuSOLVER helpers | 3.2% | sort, scale, QR, etc. |
| `gate_contract_kernel` (ours) | 0.09% | |
| `vidal_restore` left+right (ours) | 0.13% | |
| `theta2` + `ss_inv` kernels (ours) | 0.11% | persistent path |
| `row_to_col_kernel` (ours) | 0.06% | |
| **MPI communication** | **< 0.01%** | `Gatherv` of $\sim$10 KB at end |

*Table 2. cuSOLVER Jacobi is $>99.6\%$ of GPU kernel time; all four custom
kernels together are $<0.4\%$.*

**Model vs. measurement.** The crossover study (`bench_update`, single
`update_bond` vs. $\chi$ on a synthetic $L=16$ state, Fig. 3) confirms model
prediction (a): at the TFIM-realized $\chi_a\approx41$ the GPU is latency-bound
(flat across $\chi_{\max}=64$–128 in Table 1, exactly as predicted), and as the
matrix is forced larger the GPU crosses the CPU near $\chi\approx96$ and reaches
$\sim5\times$ at $\chi=128$, recovering the $O(\chi^3)$ asymptote. The Nsight
distributed profile confirms model (c): **no MPI call appears in the timeline**;
every rank spends 96.2–96.5% of kernel time in the three Jacobi kernels, matching
the single-GPU profile. The main *discrepancy* is the negligible stream speedup,
which the model explains: the host-side truncation sync serializes bonds.

![M3 benchmark: wall time vs. L (latency-bound, linear) and per-call
update_bond vs. chi (CPU/GPU crossover near chi≈96).](figs/bench_M3.png)

*Figure 3. (a) Wall time for 15 steps vs. $L$ at $\chi_{\max}=64$: linear,
$\sim1.3$ ms/bond — latency-bound, as predicted. (b) Per-call `update_bond` vs.
$\chi$: CPU/GPU crossover near $\chi\approx96$–128.*

---

## 6. Bottleneck analysis

The implementation has three regimes, each tied to hardware:

- **Single GPU, large $\chi$ — compute-bound on the SVD.** cuSOLVER's Jacobi
  kernels are $>99.6\%$ of GPU kernel time (Table 2). The iterative
  Gram-orthogonalization is FP64 ALU work; on Turing's limited FP64 throughput
  this dominates, and no amount of stream overlap or transfer elimination can
  move the ceiling. This is the binding bottleneck for the physics we care about.
- **Single GPU, small $\chi$ — latency-bound.** At the TFIM-realized
  $\chi_a\approx41$, every SVD matrix is small, so per-bond time is set by kernel
  launch + cuSOLVER dispatch, not flops (model (a)). This is why the GPU only
  *wins* once $\chi_{\max}\ge48$ (Table 1): below that, the GPU pays launch
  overhead the CPU does not.
- **Multi-GPU — load imbalance, not communication.** Computation is $>99\%$ of
  wall time and the final `Gatherv` is $<0.01\%$ (Table 2), so the distributed
  bottleneck is purely **load imbalance**: round-robin gives one rank the four
  hardest ($J=2.0$) pairs, which run $2.2\times$ longer than the cheapest,
  explaining the $\sim19\%$ efficiency gap at $P=4$ (§8). This ties back to the
  warp/SM execution model only indirectly — the imbalance is at the *task* level
  (entanglement → $\chi_a$ → $O(\chi_a^3)$), above the kernel.

A secondary bottleneck is the **mid-pipeline host sync** for truncation, which
prevents stream overlap; removing it would require a device-side top-$k$
selection of singular values.

---

## 7. Algorithmic variants and their effect on performance

We explored four variants; each shifts the bottleneck differently.

1. **Host-staging (M3) → persistent device MPS.** Eliminating per-bond bulk
   H2D/D2H drops profiled PCIe traffic from tens of MB to $\sim0.2$ MB and gives
   a steady 1–3% full-step speedup (Table 1). The gain is *small precisely
   because* the SVD already dominates — i.e. the variant confirms the bottleneck
   is compute, not transfers.

2. **Single stream → 4 streams.** Negligible (Table 1, Fig. 2). The variant is
   instructive: the model predicted it, and profiling showed why — the host-side
   truncation `cudaStreamSynchronize` serializes bonds. A variant that *looks*
   like it should help, but the data and model agree it cannot until truncation
   moves on-device.

3. **cuSOLVER `gesvdj` vs. custom one-sided Jacobi SVD.** Our hand-written Jacobi
   kernel (one 128-thread block, cyclic sweeps, tol $10^{-15}$, $\le200$ sweeps,
   with an $A^*$-adjoint path for wide $m<n$ matrices) is **1.2–2.2× slower** than
   cuSOLVER and agrees to $\lesssim10^{-12}$ (Table 3, Fig. 4). cuSOLVER remains
   the production backend; the custom kernel is the from-scratch deliverable and
   quantifies the vendor gap. The bottleneck (SVD) is unchanged — this variant
   tells us *how much* headroom a better single-block Jacobi would need to find.

4. **Trotter order (1 / 2 / 4).** Higher order trades more gates per step
   (more SVDs) for $O(dt^p)$ accuracy; order 4 (Forest–Ruth) lets us reach the
   ED-validation floor ($\sim10^{-6}$, §9) at larger $dt$, i.e. fewer steps for a
   given physics target.

| $\chi_{\max}$ | cuSOLVER (s) | Custom Jacobi (s) | Ratio | max $|\Delta\sigma|$ |
|---:|---:|---:|---:|---:|
| 16  | 5.318  | 6.125  | 0.868 | $4.74\times10^{-14}$ |
| 32  | 13.028 | 20.441 | 0.637 | $5.84\times10^{-13}$ |
| 64  | 18.272 | 39.982 | 0.457 | $1.00\times10^{-12}$ |
| 128 | 18.458 | 40.343 | 0.458 | $1.00\times10^{-12}$ |

*Table 3. `bench_svd`: cuSOLVER vs. custom Jacobi, $L=20$, 20 steps. The
wide-matrix adjoint fix reduced error from $O(10^{-1})$ to $\lesssim10^{-12}$.*

![Custom Jacobi is 1.2–2.2x slower than cuSOLVER; accuracy is below 1e-12 after
the wide-matrix adjoint fix.](figs/svd_compare.png)

*Figure 4. Custom Jacobi SVD vs. cuSOLVER `gesvdj`: time ratio and Schmidt-spectrum error.*

---

## 8. Scalability analysis

We study scaling of the task-parallel sweep on `gpu-turing`, using total rank
wall time $T(P)=\max_r\sum_{i\in r}t_i$ (the model metric from §4d).

| Mode | Ranks | Pairs/rank | Time (s) | Speedup | Efficiency |
|---|---:|---:|---:|---:|---:|
| strong | 1 | 16 | 321.6 | 1.00 | 100% |
| strong | 2 |  8 | 184.3 | 1.75 |  87% |
| strong | 4 |  4 |  99.1 | 3.24 |  81% |
| weak   | 1 |  4 |  17.1 | —    | 100% |
| weak   | 2 |  4 |  59.1 | —    |  29% |
| weak   | 4 |  4 | 103.5 | —    |  17% |

*Table 4. MPI strong and weak scaling. Strong speedup relative to the 1-rank
total (321.6 s).*

![Left: strong-scaling speedup with efficiency annotated. Right: weak-scaling
total rank wall time with efficiency.](figs/scaling_m4.png)

*Figure 5. Strong-scaling speedup (left) and weak-scaling wall time (right).*

**Strong scaling.** Speedup reaches $3.24\times$ at $P=4$ (81% efficiency). The
19% loss is **exactly the load imbalance the model predicted**: with zero in-loop
communication, the only inefficiency is that round-robin hands rank 3 the four
$J=2.0$ pairs (highest entanglement, largest $\chi_a$), which run $\sim2.2\times$
longer than rank 0's cheap $J=0.25$ pairs. Confirmed against communication being
$<0.01\%$. This is the regime our design is built for, and it behaves as designed.

**Weak scaling.** Efficiency falls sharply (29% at $P=2$, 17% at $P=4$) — but
this is *not* a communication or synchronization failure. As the model warned, the
weak mode hands each added rank *higher-$J$* pairs, which are intrinsically more
expensive ($\sim2$–$6\times$ the cost of the $P=1$ pairs). The per-rank workload
is not held constant in *compute*, only in *pair count*, so the apparent weak
efficiency reflects rising per-pair cost, not parallel overhead. True
constant-work weak scaling would require balancing by *estimated entanglement*
rather than pair count.

**Scaling limits.** Strong mode exhausts the 16-pair grid at 16 ranks; beyond
that, ranks idle. Because communication is negligible, the design would otherwise
scale to as many ranks as there are independent $(J,g)$ points — the limit is the
size of the physics sweep, not the network.

---

## 9. Correctness and verification

We verify correctness at four independent levels:

1. **Unit tests.** 54 Google Test cases across 6 binaries cover tensor ops, the
   hand-written GEMM/GEMV/SVD/`expm`/contraction/`kron`, MPS observables, the TFI
   model (Hermiticity, sum rules), and TEBD invariants (norm preservation,
   $S^z$ conservation, entropy growth, Trotter-order convergence). All pass at
   $10^{-12}$ (exact ops) / $10^{-10}$ (SVD-dependent).

2. **Analytical / exact-diagonalization oracle.** A full $2^L\times2^L$
   state-vector evolution ($U=\text{expm}(-i\,dt\,H)$ via GEMV) validates TEBD on
   small systems. At $L=8$, $dt=0.05$, $t=2$, order 4, $\chi_{\max}=64$: max error
   $5.1\times10^{-6}$ in $\langle\sigma^z\rangle$, $4.7\times10^{-6}$ in
   $\langle\sigma^x\sigma^x\rangle$, $2.2\times10^{-6}$ in entropy, and final
   state-vector fidelity loss $4.3\times10^{-11}$. The residual is **dominated by
   the $O(dt^4)$ Trotter error**, not the implementation — the error sits at the
   Trotter floor from the first step onward (Fig. 6).

3. **CPU↔GPU agreement.** `validate_gpu` (per-bond GPU path) and
   `validate_persistent` (the persistent device-MPS path the sweep uses) run the
   GPU engine side-by-side with the CPU reference for 30 steps and compare
   $\langle\sigma^z\rangle$, $\langle\sigma^x\rangle$, and all $L-1$ entropies at
   every step. Max discrepancies are $3.53\times10^{-13}$ and $3.60\times10^{-13}$
   — at the floor where the CPU's one-sided Jacobi and cuSOLVER's `gesvdj` agree
   on singular values. The custom Jacobi path agrees to $\lesssim10^{-12}$
   (Table 3).

4. **Multi-rank consistency.** 1-, 2-, and 4-rank sweeps produce **identical**
   $(J,g)\!\to\!(\chi_{\max},S_\text{mid})$ records, and the 1-stream and 4-stream
   persistent configurations agree to every printed digit.

![Per-step max error of TEBD against exact diagonalization: after the first step
the error sits at the Trotter floor (~1e-6) for the entire run.](figs/validate_errors.png)

*Figure 6. TEBD vs. exact diagonalization, per-step max error. Final fidelity
loss $4.3\times10^{-11}$.*

**Debugging notes.** Three subtle bugs were caught and fixed with
`compute-sanitizer` and careful auditing: (i) a silent gauge bug from cuSOLVER
returning $V$ (not $V^H$) in column-major, fixed by an explicit row→col transpose
and a conjugation in the right-restore kernel; (ii) a gate shared-memory load
that zero-padded when a block had $<16$ threads, fixed with a cooperative strided
load; (iii) an out-of-bounds write from `gesvdj_bufferSize` being non-monotone in
$(m,n)$, fixed by re-querying `lwork` per call.

---

## 10. Discussion, limitations, and future work

**What worked, and the largest payoffs.** The single biggest design decision —
**task-parallel MPI with zero in-loop communication** — is the right one at this
problem scale and delivered 81% strong-scaling efficiency limited only by load
imbalance. On the single GPU, the **persistent device MPS** removed the M3 PCIe
bottleneck cleanly. Both were high-payoff for low complexity.

**What was not worth the complexity.** **CUDA streams** within a sub-step gave
negligible speedup because host-side truncation serializes bonds — the complexity
of multiple workspaces/handles bought nothing measurable. The **custom Jacobi
SVD** was valuable as a learning and from-scratch deliverable but stays 1.2–2.2×
slower than cuSOLVER, so it is not the production path.

**Limitations and what we would do differently with more time/hardware.**

- **Load-balanced assignment.** Round-robin ignores the $\sim2$–$6\times$ runtime
  spread across $(J,g)$. Sorting pairs by estimated $\chi_a$ and bin-packing to
  ranks should lift strong-scaling efficiency toward $\gtrsim95\%$ and fix the
  apparent weak-scaling collapse (which is really non-uniform work).
- **Device-side truncation.** A GPU top-$k$ singular-value selection would remove
  the mid-pipeline host sync and finally let streams overlap bonds — the clearest
  path to single-GPU speedup beyond the SVD ceiling.
- **GPU-side observables.** The sweep currently reports $\chi_{\max}$ and mid-bond
  entropy; full $\langle\sigma^z_i\rangle/\langle\sigma^x_i\rangle$/correlation
  heatmaps over the phase diagram need GPU-side measurement code (planned, not
  finished).
- **Spatial decomposition for $L\gg100$.** When the chain is long enough that a
  single evolution exceeds one GPU's memory, or when there are too few $(J,g)$
  points to fill the ranks, a bond-cut spatial decomposition becomes necessary —
  at the cost of one `MPI_Allgather` of the boundary Schmidt spectrum
  ($\chi_{\max}$ doubles) per Trotter sub-step. Our cost model says this only pays
  off once per-sub-step compute exceeds that round-trip, i.e. at large $\chi$.
- **CUDA-aware MPI** would let the final gather read device buffers directly, but
  the $<10$ KB transfer makes it immaterial here.

**Connection to lecture.** The project touches the GPU memory hierarchy (shared
memory for the gate, coalesced layout transforms, pre-allocation to avoid
`cudaMalloc`), the roofline model (arithmetic intensity $O(\chi)$ separating
latency-bound small SVDs from compute-bound large ones), MPI collectives
(`Gather`/`Gatherv` over point-to-point), and Amdahl/load-balance analysis (strong
efficiency set by the slowest rank). The central lesson: for this workload the
useful parallelism is *task-level*, the bottleneck is a single dense kernel (the
SVD), and a good performance model predicted — ahead of the measurements — both
where the GPU would win and why streams and weak scaling would disappoint.

---

## References

1. G. Vidal, *Efficient classical simulation of slightly entangled quantum
   computations*, PRL 91, 147902 (2003); *Efficient simulation of one-dimensional
   quantum many-body systems*, PRL 93, 040502 (2004).
2. U. Schollwöck, *The density-matrix renormalization group in the age of matrix
   product states*, Ann. Phys. 326, 96 (2011).
3. R. P. Brent and F. T. Luk, *The solution of singular-value and symmetric
   eigenvalue problems on multiprocessor arrays*, SIAM J. Sci. Stat. Comput. (1985).
4. N. Hatano and M. Suzuki, *Finding exponential product formulas of higher
   orders* (Suzuki–Trotter decomposition).
5. NVIDIA cuSOLVER / cuQuantum (cuTensorNet) documentation.
6. TeNPy, ITensor, quimb — open-source tensor-network libraries.
</content>
</invoke>
