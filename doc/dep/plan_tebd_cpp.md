# Plan: C++ implementation of minimal_tebd (CUDA-ready)

## Context

CME 213 Milestone 2: C++ CPU reference of `lib/minimal_tebd`, designed from the start for CUDA
porting. All linear algebra (GEMM, SVD, matrix exp) is hand-written — no LAPACK/OpenBLAS.
Compiler: `nvc++` (NVIDIA HPC SDK), C++14.

**Goal:** `lib/minimal_tebd_c/` that reproduces `example_notebook.py`, outputting data that
the Python `visualize.py` can plot.

**Style reference:** `tim4431/paracomp_hw` HW2 — the `DeviceMatrix` / `MatrixAccessor` split,
nvc++ Makefile conventions, `CHECK_LAUNCH` pattern, Google Test validation.

---

## 1. Complex number type

```cpp
#include <complex>
using Cdouble = std::complex<double>;
```

`std::complex<double>` is layout-compatible with `cuDoubleComplex` (two consecutive doubles).
nvc++ supports it on host; for device code, `reinterpret_cast` to `cuDoubleComplex*`.
Arithmetic operators (`+`, `*`, `std::conj()`, `std::abs()`) work out of the box.

---

## 2. Tensor storage — the core design question

### 2.1 The `Tensor` class

Follows the HW2 pattern: a **host-side class** that owns memory and provides element access.
For the CUDA port, we'll add a lightweight `TensorAccessor<NDIM>` (analogous to `MatrixAccessor`)
that wraps a raw pointer + shape and gets passed into kernels.

```cpp
static constexpr int MAX_NDIM = 4;

class Tensor {
public:
    Cdouble *data;              // flat row-major storage
    int      ndim;              // actual rank (1–4)
    int      shape[MAX_NDIM];   // shape[k] for k < ndim
    int      stride[MAX_NDIM];  // stride[k] = product of shape[k+1..ndim-1]
    int      size;              // logical elements = product of shape
    int      capacity;          // allocated elements (>= size)

    // constructors / destructor
    Tensor();
    Tensor(int ndim, const int shape[]);
    Tensor(int ndim, const int shape[], int capacity);  // pre-alloc to capacity
    ~Tensor();
    Tensor(const Tensor& other);                        // deep copy
    Tensor& operator=(const Tensor& other);
    Tensor(Tensor&& other) noexcept;                    // move
    Tensor& operator=(Tensor&& other) noexcept;

    // element access (bounds-checked in debug)
    Cdouble& operator()(int i);
    Cdouble& operator()(int i, int j);
    Cdouble& operator()(int i, int j, int k);
    Cdouble& operator()(int i, int j, int k, int l);
    // const versions
    const Cdouble& operator()(int i) const;
    const Cdouble& operator()(int i, int j) const;
    const Cdouble& operator()(int i, int j, int k) const;
    const Cdouble& operator()(int i, int j, int k, int l) const;

    // in-place ops
    void reshape(int new_ndim, const int new_shape[]);  // no copy, reinterpret
    void zero();                                        // memset to 0
    void scale(Cdouble alpha);                          // data[i] *= alpha

    // factory
    static Tensor zeros(int ndim, const int shape[]);
    static Tensor identity(int n);                      // n×n identity matrix
};

// free functions
Tensor tensor_transpose(const Tensor& t, const int perm[]);   // returns new tensor
Cdouble tensor_trace(const Tensor& t);                        // rank-2 trace
void tensor_print(const Tensor& t, const char* name = "");    // debug print
```

**Row-major with explicit strides.** Element `(i,j,k,l)` lives at offset
`i*stride[0] + j*stride[1] + k*stride[2] + l*stride[3]`. Strides are recomputed on reshape.

**Why `capacity` separate from `size`:** When we pre-allocate to `chi_max`, the buffer
is larger than the current tensor. `size` tracks the logical elements; `capacity` tracks
the allocation. This avoids reallocation as bond dimensions grow — crucial for GPU where
`cudaMalloc` is expensive.

### 2.2 MPS tensor storage strategy

MPS tensors `Bs[i]` have shape `(chi_L, d, chi_R)` where `chi_L`, `chi_R` change during
evolution (growing from 1 up to `chi_max`).

**Strategy: Pre-allocate to max, track actual dimensions.**

Each `Bs[i]` is allocated once with `capacity = chi_max * d * chi_max`. The actual dimensions
are stored in `shape[]`. When the SVD changes bond dimension, we update `shape` and `stride`
but never reallocate.

Memory cost: L × chi_max × d × chi_max × 16 bytes. For L=1000, d=2, chi_max=100:
= 1000 × 100 × 2 × 100 × 16 = 320 MB — acceptable.

**Why this is CUDA-friendly:**
- Zero `cudaMalloc` during time evolution — all memory allocated once at init
- All Bs[i] have the same buffer layout → batch into pointer arrays for
  `cublasZgemmBatched` / `cusolverDnZgesvdjBatched`
- Predictable memory footprint

### 2.3 CUDA transition: TensorAccessor (future)

Following the HW2 `MatrixAccessor<OrderPolicy>` pattern, the CUDA port will add a lightweight
device-side accessor:

```cpp
// Future: passed into CUDA kernels (like MatrixAccessor in HW2)
template<int NDIM>
struct TensorAccessor {
    Cdouble *data;          // device pointer (or reinterpret_cast<cuDoubleComplex*>)
    int shape[NDIM];
    int stride[NDIM];

    __host__ __device__ Cdouble& operator()(int i, int j);         // rank-2
    __host__ __device__ Cdouble& operator()(int i, int j, int k);  // rank-3
};
```

The host `Tensor` would provide `getAccessor<NDIM>()` just like `DeviceMatrix::getAccessor<OrderPolicy>()`.

---

## 3. Hand-written linear algebra (`linalg.h/cpp`)

### 3.1 Complex matrix multiply (GEMM)

C = α·A·B + β·C where A is (m×k), B is (k×n), C is (m×n). All complex.

```
for i in 0..m:
  for p in 0..k:       // i,p,j loop order: better cache behavior (row-wise B access)
    for j in 0..n:
      C(i,j) += A(i,p) * B(p,j)
```

For CUDA, this becomes a thread-per-output-element kernel with shared memory tiling
(same pattern as HW2 Q2 matrix ops with `dim3 block(16,16)`).

Signature uses explicit leading dimensions for pre-allocated tensors:
```cpp
void zgemm(const Cdouble* A, int lda,
           const Cdouble* B, int ldb,
           Cdouble* C, int ldc,
           int m, int n, int k,
           Cdouble alpha = 1.0, Cdouble beta = 0.0);
```

### 3.2 SVD — one-sided Jacobi

For a matrix A (m×n, m ≥ n), compute A = U·Σ·V†.

**One-sided Jacobi SVD:**
1. Copy A → W (working copy, m×n)
2. Initialize V = I (n×n)
3. Sweep: for each column pair (p, q) with p < q:
   - Compute: α = wₚ†·wₚ,  β = wq†·wq,  γ = wₚ†·wq
   - If |γ| < tol·√(α·β), skip (already orthogonal)
   - Compute 2×2 Jacobi rotation (c, s) that zeros γ
   - Apply rotation to columns p, q of W and V
4. Repeat sweeps until convergence (all off-diagonal < tol)
5. σᵢ = ‖wᵢ‖,  U = W · diag(1/σᵢ),  V† from accumulated rotations
6. Sort by descending singular value

**Why Jacobi for this project:**
- Simple to implement correctly (~100 lines for complex case)
- Numerically stable
- **Naturally parallel**: non-overlapping column pairs processed simultaneously → CUDA threads
  (same grid-stride pattern as HW2 Q1 `recurrence` kernel)
- Good for our sizes (up to ~200×200)
- 5–10 sweeps typical for convergence

### 3.3 Matrix exponential (for gate computation only)

Only needed for small matrices: `d²×d² = 4×4` for spin-1/2.

**Scaling and squaring with Taylor series:**
1. Choose integer s such that ‖M/2ˢ‖ < 1
2. Compute exp(M/2ˢ) via truncated Taylor series (~15 terms)
3. Square the result s times: exp(M) = (exp(M/2ˢ))^(2ˢ)

For 4×4 matrices, this is ~15 matrix multiplies of 4×4. Trivial cost, called once at init.

### 3.4 Tensor contraction (replaces `np.tensordot`)

All tensor contractions in TEBD decompose as:
1. Transpose so contracted indices are innermost/outermost
2. Reshape both tensors to 2D
3. Call GEMM
4. Reshape result to target rank

```cpp
Tensor tensor_contract(const Tensor& A, const int axes_a[],
                       const Tensor& B, const int axes_b[],
                       int n_contract);
```

### 3.5 Kronecker product

For model construction only (building H_bonds from Pauli matrices).
Simple nested loop. 4×4 output for d=2.

```cpp
void kron(const Cdouble* A, int ra, int ca,
          const Cdouble* B, int rb, int cb,
          Cdouble* C);
```

---

## 4. Module breakdown

### 4.1 `tensor.h/cpp`

Tensor class as described in Section 2.1. Core operations: alloc/free, element access,
reshape, transpose, scale, zero, copy/move, debug print.

### 4.2 `linalg.h/cpp`

```cpp
// Matrix multiply: C = alpha * A @ B + beta * C
void zgemm(const Cdouble* A, int lda, const Cdouble* B, int ldb,
           Cdouble* C, int ldc, int m, int n, int k,
           Cdouble alpha = 1.0, Cdouble beta = 0.0);

// Jacobi SVD: A(m×n) = U(m×n) · diag(S) · Vt(n×n)
void svd(const Cdouble* A, int m, int n, int lda,
         Cdouble* U, double* S, Cdouble* Vt);

// Matrix exponential: result = expm(M), M is n×n
void matrix_expm(const Cdouble* M, int n, Cdouble* result);

// Tensor contraction
Tensor tensor_contract(const Tensor& A, const int axes_a[],
                       const Tensor& B, const int axes_b[],
                       int n_contract);

// Kronecker product
void kron(const Cdouble* A, int ra, int ca,
          const Cdouble* B, int rb, int cb,
          Cdouble* C);
```

### 4.3 `mps.h/cpp` — port of `mps.py`

```cpp
struct MPS {
    int      L, d;
    Tensor  *Bs;        // array of L tensors, each rank-3 (chi_L, d, chi_R)
    double **Ss;        // array of L double* vectors (Schmidt values)
    int     *chi;       // chi[i] = left bond dim at site i, length L+1
    int      chi_max;   // pre-allocation size

    // constructors
    MPS(int L, int d, int chi_max);
    ~MPS();
    static MPS product_state(int L, const int states[], int d, int chi_max);

    // tensor accessors
    Tensor theta1(int i) const;     // diag(Ss[i]) @ Bs[i]
    Tensor theta2(int i) const;     // theta1(i) @ Bs[i+1]
    void bond_dims(int out[]) const;

    // observables
    void entropy(double out[]) const;               // L-1 entanglement entropies
    void expect(const Cdouble op[4], double out[]) const;   // <op_i> for i=0..L-1
    void corr(const Cdouble opA[4], const Cdouble opB[4],
              double out[]) const;                  // <A_i B_j>, out is L×L
};

// Pauli matrices (file-scope constants)
extern const Cdouble PAULI_I[4];
extern const Cdouble PAULI_X[4];
extern const Cdouble PAULI_Y[4];
extern const Cdouble PAULI_Z[4];
```

### 4.4 `model.h/cpp` — port of `model.py`

```cpp
struct BondModel {
    int     L, d;
    Tensor *H_bonds;    // L-1 rank-4 tensors (d,d,d,d)

    ~BondModel();
};

BondModel tfi_chain(int L, double J, double g);
```

### 4.5 `tebd.h/cpp` — port of `tebd.py`

```cpp
struct TrotterStep {
    int parity;     // 0 = even, 1 = odd
    int frac_idx;   // index into unique_fracs
};

struct TEBDEngine {
    MPS        *psi;
    BondModel  *model;
    double      dt, evolved_time, trunc_err_eps;
    int         order, N_steps, chi_max;
    double      svd_min;
    int         type_evo;   // 0=real, 1=imag

    // gate cache
    int            n_schedule;
    TrotterStep   *schedule;
    int            n_unique_fracs;
    double        *unique_fracs;
    Tensor       **gate_sets;     // gate_sets[frac_idx][bond]

    TEBDEngine(MPS* psi, BondModel* model,
               double dt, int order, int N_steps,
               int chi_max, double svd_min, int type_evo = 0);
    ~TEBDEngine();

    void run();     // advance by N_steps * dt
    void step();    // one Trotter step
};

// Core update (the hot path for CUDA porting)
double update_bond(MPS* psi, int site, const Tensor& gate,
                   int chi_max, double svd_min);

// SVD + truncation
void svd_truncate(const Tensor& theta,
                  int chi_max, double svd_min,
                  Tensor& A, double* S, Tensor& B, double& eps);
```

### 4.6 `exact_diag.h/cpp` — port of `exact_diag.py`

Full state-vector reference for validation. Only feasible for small L (≤ 14).

```cpp
// Build the full d^L × d^L Hamiltonian from bond Hamiltonians
// H = Σ_b  I^{⊗b} ⊗ H_b ⊗ I^{⊗(L-b-2)}
void bond_model_to_full_hamiltonian(const BondModel& model, Cdouble* H_full);

// Embed single-site operator at site i into the full d^L space
void single_site_op_full(const Cdouble* op, int i, int L, int d, Cdouble* out);

// Embed two-site operator A_i B_j into the full d^L space
void two_site_op_full(const Cdouble* opA, const Cdouble* opB,
                      int i, int j, int L, int d, Cdouble* out);

struct ExactEvolution {
    int     L, d;
    int     dim;            // d^L
    Cdouble *psi;           // state vector, length dim
    Cdouble *H;             // full Hamiltonian, dim × dim
    Cdouble *U_dt;          // expm(-i·dt·H), dim × dim
    double   dt;
    double   evolved_time;
    int      type_evo;      // 0=real, 1=imag

    ExactEvolution(const BondModel& model, const Cdouble* state_vector,
                   double dt, int type_evo = 0);
    ~ExactEvolution();

    // Construct from product state (like Python classmethod)
    static ExactEvolution from_product_state(const BondModel& model,
                                             const int states[], double dt,
                                             int type_evo = 0);

    void step();    // advance by one dt

    // Observables — same interface as MPS for easy comparison
    void expectation_value(const Cdouble* op, double out[]) const;   // <op_i>, length L
    void correlation_function(const Cdouble* opA, const Cdouble* opB,
                              double out[]) const;                   // L×L
    void entanglement_entropy(double out[]) const;                   // L-1 bonds
};
```

**Key operations:**
- `bond_model_to_full_hamiltonian`: assembles H via Kronecker products (uses `kron` from linalg)
- `ExactEvolution::step()`: `psi = U_dt @ psi` (matrix-vector multiply via `zgemm`)
- `entanglement_entropy`: reshape psi to `(d^b, d^(L-b))`, SVD, compute -Σ s² log(s²)

### 4.7 `validate.h/cpp` — port of `validate.py`

Side-by-side TEBD vs ED comparison. Runs both from the same initial state, measures
observables at each step, reports per-observable max error.

```cpp
struct ValidationSummary {
    std::vector<double> t;
    std::vector<double> err_Sz, err_Sx, err_corr_XX, err_entropy;
    double trunc_err;
    double fidelity_drop;   // 1 - |⟨ψ_TEBD | ψ_ED⟩|
};

ValidationSummary validate_tebd_vs_ed(
    const BondModel& model,
    double dt = 0.05,
    double t_max = 1.0,
    int chi_max = 64,
    double svd_min = 1e-12,
    int order = 4,
    const int* initial_states = nullptr,  // default: all zeros
    bool verbose = true
);
```

This function:
1. Creates both `MPS` + `TEBDEngine` and `ExactEvolution` from the same product state
2. Steps both forward together
3. At each step, computes `⟨σz⟩`, `⟨σx⟩`, `⟨XiXj⟩`, entropy from both
4. Records max |TEBD - ED| per observable per time step
5. At the end, computes fidelity via `mps.to_state_vector()` dotted with `ed.psi`
6. If `verbose`, prints a per-step table (matching Python output format)

### 4.8 `main.cpp` — driver

Mirrors `example_notebook.py`. Runs L=30 TFI quench, prints CSV to stdout:
```
# t,S_mid,max_chi,trunc_err
0.00,0.0000,1,0.00e+00
0.10,0.0552,6,8.22e-15
...
```

Python visualization reads this CSV.

---

## 5. File structure

```
lib/minimal_tebd_c/
├── Makefile                  # nvc++ -std=c++14
├── tensor.h / tensor.cpp
├── linalg.h / linalg.cpp
├── mps.h    / mps.cpp
├── model.h  / model.cpp
├── tebd.h   / tebd.cpp
├── exact_diag.h / exact_diag.cpp
├── validate.h   / validate.cpp
├── main.cpp                  # driver: runs quench, outputs CSV
├── test_tensor.cpp           # Google Test: tensor operations
├── test_linalg.cpp           # Google Test: GEMM, SVD, expm
├── test_mps.cpp              # Google Test: MPS observables
├── test_model.cpp            # Google Test: TFI H_bonds
├── test_tebd.cpp             # Google Test: TEBD single/multi-step vs Python
├── test_validate.cpp         # Google Test: TEBD vs ED validation
└── googletest/               # vendored Google Test (same as HW2)
```

### Makefile (following HW2 conventions)

```makefile
CXX = nvc++
CXXFLAGS = -O2 -std=c++14

# Google Test (vendored, same layout as HW2)
GTEST_DIR = ./googletest/googletest
GTEST_INC = $(GTEST_DIR)/include
GTEST_FLAGS = -isystem $(GTEST_INC) -O2

# --- production target ---
SRCS = tensor.cpp linalg.cpp mps.cpp model.cpp tebd.cpp main.cpp
OBJS = $(SRCS:.cpp=.o)
TARGET = tebd_sim

default: $(TARGET)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@

# --- Google Test library ---
gtest-all.o:
	$(CXX) $(GTEST_FLAGS) -I$(GTEST_DIR) -c $(GTEST_DIR)/src/gtest-all.cc
gtest_main.o:
	$(CXX) $(GTEST_FLAGS) -I$(GTEST_DIR) -c $(GTEST_DIR)/src/gtest_main.cc
gtest_main.a: gtest-all.o gtest_main.o
	$(AR) $(ARFLAGS) $@ $^

# --- test targets ---
LIB_OBJS = tensor.o linalg.o mps.o model.o tebd.o exact_diag.o validate.o
TEST_CXXFLAGS = $(CXXFLAGS) -isystem $(GTEST_INC)

test_tensor: test_tensor.o tensor.o gtest_main.a
	$(CXX) $^ -o $@
test_linalg: test_linalg.o tensor.o linalg.o gtest_main.a
	$(CXX) $^ -o $@
test_mps: test_mps.o $(LIB_OBJS) gtest_main.a
	$(CXX) $^ -o $@
test_model: test_model.o $(LIB_OBJS) gtest_main.a
	$(CXX) $^ -o $@
test_tebd: test_tebd.o $(LIB_OBJS) gtest_main.a
	$(CXX) $^ -o $@
test_validate: test_validate.o $(LIB_OBJS) gtest_main.a
	$(CXX) $^ -o $@

test_%_o: test_%.cpp
	$(CXX) $(TEST_CXXFLAGS) -c $< -o $@

tests: test_tensor test_linalg test_mps test_model test_tebd test_validate

run_tests: tests
	./test_tensor && ./test_linalg && ./test_mps && ./test_model && ./test_tebd && ./test_validate

clean:
	rm -f *.o $(TARGET) test_tensor test_linalg test_mps test_model test_tebd test_validate gtest_main.a
```

---

## 6. Implementation order

| Step | Files | Deliverable |
|---|---|---|
| 1 | `tensor.h/cpp`, `test_tensor.cpp` | Tensor class + tests |
| 2 | `linalg.h/cpp`, `test_linalg.cpp` | GEMM, SVD, expm, tensor_contract, kron + tests |
| 3 | `mps.h/cpp`, `test_mps.cpp` | MPS struct + observables + tests |
| 4 | `model.h/cpp`, `test_model.cpp` | TFI chain + tests |
| 5 | `tebd.h/cpp`, `test_tebd.cpp` | TEBD engine + single/multi-step tests |
| 6 | `exact_diag.h/cpp` | Full-Hamiltonian builder + state-vector evolution |
| 7 | `validate.h/cpp`, `test_validate.cpp` | TEBD vs ED comparison + Google Test validation |
| 8 | `main.cpp` | Full driver, CSV output, compare with Python |

---

## 7. Validation test suite (Google Test)

All tests use the same vendored `googletest` as HW2. Tolerances: `1e-12` for exact
operations, `1e-10` for SVD-dependent results.

### 7.1 `test_tensor.cpp`

```cpp
TEST(Tensor, AllocAndAccess) {
    // Create rank-3 tensor (2,3,4), set elements, read back
}
TEST(Tensor, ReshapeRoundTrip) {
    // Reshape (2,3,4) → (6,4) → (2,3,4), verify data unchanged
}
TEST(Tensor, Transpose) {
    // Transpose rank-3 (a,b,c) with perm={2,0,1}, verify against manual
}
TEST(Tensor, CopyAndMove) {
    // Deep copy: modify original, verify copy unchanged
    // Move: verify source is emptied
}
TEST(Tensor, Scale) {
    // Scale by complex scalar, verify all elements
}
```

### 7.2 `test_linalg.cpp`

```cpp
TEST(GEMM, IdentityMultiply) {
    // A @ I = A for a small complex matrix
}
TEST(GEMM, KnownProduct) {
    // 3×2 @ 2×4 with known values, compare element-wise
}
TEST(SVD, Diagonal) {
    // SVD of diag(3,2,1): singular values should be [3,2,1]
}
TEST(SVD, Rank2Complex) {
    // SVD of a 4×3 complex matrix, verify U·S·V† ≈ A
}
TEST(SVD, Truncation) {
    // SVD of a matrix with known rank, verify discarded weight
}
TEST(Expm, IdentityMatrix) {
    // expm(0) = I
}
TEST(Expm, DiagonalMatrix) {
    // expm(diag(a,b,c,d)) = diag(exp(a),exp(b),exp(c),exp(d))
}
TEST(Expm, PauliX) {
    // expm(-i*t*σx) has known closed form, compare at t=0.1
}
TEST(TensorContract, MatMul) {
    // Contract rank-2 tensors on axis 1,0 = matrix multiply
}
TEST(TensorContract, Rank3Rank2) {
    // Contract (chi,d,chi) with (d,d) on physical leg
}
TEST(Kron, PauliXX) {
    // kron(σx, σx) should be the known 4×4 matrix
}
```

### 7.3 `test_mps.cpp`

```cpp
TEST(MPS, ProductStateNorm) {
    // |↑↑↑↑⟩: norm² = 1
}
TEST(MPS, ProductStateSz) {
    // |↑↑↑↑⟩: ⟨σz_i⟩ = 1 for all i
}
TEST(MPS, ProductStateSx) {
    // |↑↑↑↑⟩: ⟨σx_i⟩ = 0 for all i
}
TEST(MPS, ProductStateEntropy) {
    // Product state: all entanglement entropies = 0
}
TEST(MPS, Theta1Shape) {
    // theta1(i) has shape (chi_L, d, chi_R)
}
TEST(MPS, Theta2Shape) {
    // theta2(i) has shape (chi_L, d, d, chi_R)
}
TEST(MPS, CorrelationDiagonal) {
    // Product state: ⟨σz_i σz_i⟩ = 1 (since σz² = I)
}
```

### 7.4 `test_model.cpp`

```cpp
TEST(TFIChain, BondCount) {
    // L=10 chain has L-1 = 9 bond Hamiltonians
}
TEST(TFIChain, HermitianBonds) {
    // Each H_bond reshaped to 4×4 should be Hermitian
}
TEST(TFIChain, KnownElements) {
    // H_bonds[0] for L=4, J=1, g=1: compare element-by-element with Python output
    //   Python: TFIChain(L=4,J=1,g=1).H_bonds[0].reshape(4,4)
}
```

### 7.5 `test_tebd.cpp` — validates against Python reference values

These tests compare C++ output against hard-coded values from running the Python code.
Generate reference values once with:
```python
from lib.minimal_tebd import TFIChain, SimpleMPS, TEBDEngine
model = TFIChain(L=8, J=1.0, g=1.0)
psi = SimpleMPS.from_product_state(8, [0]*8, d=2)
eng = TEBDEngine(psi, model, {'dt':0.05, 'order':4, 'N_steps':1,
                              'trunc_params':{'chi_max':64, 'svd_min':1e-12}})
eng.run()
print(psi.entanglement_entropy())
print(psi.expectation_value('Sigmaz'))
# ... etc
```

```cpp
TEST(TEBD, TrotterScheduleOrder2) {
    // Order-2: [(even,0.5), (odd,1.0), (even,0.5)]
}
TEST(TEBD, TrotterScheduleOrder4) {
    // Order-4: 7 sub-steps with Forest-Ruth coefficients
}
TEST(TEBD, OneStepL8_Entropy) {
    // After 1 step (dt=0.05, order=4) on L=8 TFI:
    // mid-chain entropy should match Python reference value
}
TEST(TEBD, OneStepL8_Sz) {
    // ⟨σz_i⟩ after 1 step, compare with Python reference
}
TEST(TEBD, OneStepL8_Sx) {
    // ⟨σx_i⟩ after 1 step, compare with Python reference
}
TEST(TEBD, MultiStepL8_Entropy) {
    // Run 20 steps (t=1.0), compare mid-chain entropy with Python
}
TEST(TEBD, MultiStepL8_TruncErr) {
    // Accumulated truncation error after t=1.0 on L=8
}
TEST(TEBD, FullQuenchL8_Fidelity) {
    // Full t=2.0 evolution on L=8, compare state vector via
    // MPS::to_state_vector() against Python exact diag reference
    // (fidelity drop < 1e-8 for chi_max=64)
}
```

### 7.6 `test_validate.cpp` — TEBD vs exact diag (the real validation)

This is the C++ port of `validate.py`. It runs both TEBD and exact diag in the same
test, so no hard-coded reference values — the ED result *is* the reference.

```cpp
TEST(ExactDiag, FullHamiltonianHermitian) {
    // Build H for L=6 TFI, verify H = H†
}
TEST(ExactDiag, EnergyConservation) {
    // Evolve for 20 steps, verify ⟨ψ|H|ψ⟩ stays constant (real-time evo)
}
TEST(ExactDiag, ProductStateSz) {
    // |↑↑↑↑⟩: ⟨σz_i⟩ = 1 for all i (no evolution)
}
TEST(ExactDiag, NormPreservation) {
    // After 20 real-time steps, |⟨ψ|ψ⟩| should remain 1
}
TEST(ExactDiag, EntropyVsSVD) {
    // Entanglement entropy from ED matches direct SVD of state vector
}

TEST(Validate, L8_dt005_order4_Sz) {
    // Run validate_tebd_vs_ed(L=8, dt=0.05, t_max=1.0, chi_max=64, order=4)
    // Assert max|Sz_TEBD - Sz_ED| < 1e-6 at all times
}
TEST(Validate, L8_dt005_order4_Sx) {
    // Assert max|Sx_TEBD - Sx_ED| < 1e-6
}
TEST(Validate, L8_dt005_order4_CorrXX) {
    // Assert max|corr_XX_TEBD - corr_XX_ED| < 1e-5
}
TEST(Validate, L8_dt005_order4_Entropy) {
    // Assert max|entropy_TEBD - entropy_ED| < 1e-6
}
TEST(Validate, L8_dt005_order4_Fidelity) {
    // Fidelity drop < 1e-8 at t=1.0 with chi_max=64
}
TEST(Validate, TrotterConvergence) {
    // Run at dt=0.1 and dt=0.05 with order=2:
    // errors should scale as dt^2 (ratio ~ 4x)
    // Verifies the Trotter splitting is implemented correctly
}
```
