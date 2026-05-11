#include "mps.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <vector>

#include "linalg.h"

// ---------------------------------------------------------------------------
// Pauli matrices (row-major, 2x2)
// ---------------------------------------------------------------------------

const Cdouble PAULI_I[4] = {1, 0, 0, 1};
const Cdouble PAULI_X[4] = {0, 1, 1, 0};
const Cdouble PAULI_Y[4] = {0, Cdouble(0, -1), Cdouble(0, 1), 0};
const Cdouble PAULI_Z[4] = {1, 0, 0, -1};

// ---------------------------------------------------------------------------
// Constructor / destructor
// ---------------------------------------------------------------------------

MPS::MPS(int L, int d, int chi_max) : L(L), d(d), chi_max(chi_max) {
  Bs = new Tensor[L];
  Ss = new double*[L];
  chi = new int[L + 1];
  for (int i = 0; i < L; ++i) {
    Ss[i] = nullptr;
    chi[i] = 0;
  }
  chi[L] = 0;
}

MPS::~MPS() {
  for (int i = 0; i < L; ++i) delete[] Ss[i];
  delete[] Ss;
  delete[] Bs;
  delete[] chi;
}

MPS::MPS(MPS&& other) noexcept
    : L(other.L),
      d(other.d),
      Bs(other.Bs),
      Ss(other.Ss),
      chi(other.chi),
      chi_max(other.chi_max) {
  other.Bs = nullptr;
  other.Ss = nullptr;
  other.chi = nullptr;
  other.L = 0;
}

MPS& MPS::operator=(MPS&& other) noexcept {
  if (this == &other) return *this;
  for (int i = 0; i < L; ++i) delete[] Ss[i];
  delete[] Ss;
  delete[] Bs;
  delete[] chi;
  L = other.L;
  d = other.d;
  chi_max = other.chi_max;
  Bs = other.Bs;
  Ss = other.Ss;
  chi = other.chi;
  other.Bs = nullptr;
  other.Ss = nullptr;
  other.chi = nullptr;
  other.L = 0;
  return *this;
}

// ---------------------------------------------------------------------------
// Product state
// ---------------------------------------------------------------------------

MPS MPS::product_state(int L, const int states[], int d, int chi_max) {
  MPS mps(L, d, chi_max);
  int cap = chi_max * d * chi_max;
  for (int i = 0; i < L; ++i) {
    int sh[3] = {1, d, 1};
    mps.Bs[i] = Tensor(3, sh, cap);
    mps.Bs[i].zero();
    mps.Bs[i](0, states[i], 0) = 1.0;
    mps.Ss[i] = new double[chi_max];
    std::fill(mps.Ss[i], mps.Ss[i] + chi_max, 0.0);
    mps.Ss[i][0] = 1.0;
    mps.chi[i] = 1;
  }
  mps.chi[L] = 1;
  return mps;
}

// ---------------------------------------------------------------------------
// Tensor accessors
// ---------------------------------------------------------------------------

Tensor MPS::theta1(int i) const {
  // theta1 = diag(Ss[i]) @ Bs[i]
  // Bs[i] has shape (chi_L, d, chi_R).
  int chi_L = Bs[i].shape[0];
  int chi_R = Bs[i].shape[2];
  int sh[3] = {chi_L, d, chi_R};
  Tensor out(3, sh);
  for (int a = 0; a < chi_L; ++a) {
    double s = Ss[i][a];
    for (int p = 0; p < d; ++p) {
      for (int b = 0; b < chi_R; ++b) {
        out(a, p, b) = s * Bs[i](a, p, b);
      }
    }
  }
  return out;
}

Tensor MPS::theta2(int i) const {
  // theta2 = theta1(i) @ Bs[i+1]
  // theta1 shape (chi_L, d, chi_R), Bs[i+1] shape (chi_R, d, chi_R2)
  // Result shape (chi_L, d, d, chi_R2)
  Tensor t1 = theta1(i);
  int chi_L = t1.shape[0];
  int chi_mid = t1.shape[2];
  int chi_R = Bs[i + 1].shape[2];

  int sh[4] = {chi_L, d, d, chi_R};
  Tensor out(4, sh);
  // Contract t1(a, p, m) * Bs[i+1](m, q, b) -> out(a, p, q, b)
  for (int a = 0; a < chi_L; ++a) {
    for (int p = 0; p < d; ++p) {
      for (int q = 0; q < d; ++q) {
        for (int b = 0; b < chi_R; ++b) {
          Cdouble s = 0.0;
          for (int m = 0; m < chi_mid; ++m) {
            s += t1(a, p, m) * Bs[i + 1](m, q, b);
          }
          out(a, p, q, b) = s;
        }
      }
    }
  }
  return out;
}

void MPS::bond_dims(int out[]) const {
  for (int i = 0; i < L - 1; ++i) out[i] = Bs[i].shape[2];
}

int MPS::max_bond_dim() const {
  int mx = 0;
  for (int i = 0; i < L - 1; ++i) mx = std::max(mx, Bs[i].shape[2]);
  return mx;
}

// ---------------------------------------------------------------------------
// Observables
// ---------------------------------------------------------------------------

void MPS::entropy(double out[]) const {
  for (int b = 0; b < L - 1; ++b) {
    double ent = 0.0;
    int chi_b = chi[b + 1];
    for (int k = 0; k < chi_b; ++k) {
      double s = Ss[b + 1][k];
      if (s > 1e-20) {
        double s2 = s * s;
        ent -= s2 * std::log(s2);
      }
    }
    out[b] = ent;
  }
}

void MPS::expect(const Cdouble* op, double out[]) const {
  // <psi| op_i |psi> for each site i.
  // theta = theta1(i), shape (chi_L, d, chi_R)
  // op_theta(p', a, b) = sum_p op(p', p) * theta(a, p, b)
  // result = sum_{a,p,b} conj(theta(a, p, b)) * op_theta(p, a, b)
  for (int i = 0; i < L; ++i) {
    Tensor t = theta1(i);
    int chi_L = t.shape[0];
    int chi_R = t.shape[2];
    Cdouble val = 0.0;
    for (int a = 0; a < chi_L; ++a) {
      for (int p = 0; p < d; ++p) {
        for (int b = 0; b < chi_R; ++b) {
          // Compute (op @ theta)(a, p, b) = sum_q op(p,q) * theta(a,q,b)
          Cdouble ot = 0.0;
          for (int q = 0; q < d; ++q) {
            ot += op[p * d + q] * t(a, q, b);
          }
          val += std::conj(t(a, p, b)) * ot;
        }
      }
    }
    out[i] = std::real(val);
  }
}

void MPS::corr(const Cdouble* op_A, const Cdouble* op_B, double out[]) const {
  // C[i,j] = <psi| A_i B_j |psi>, stored in out[i*L + j].
  std::fill(out, out + L * L, 0.0);

  // Diagonal: C[i,i] = <(A@B)_i>
  Cdouble AB[4];
  // AB = A @ B (2x2 matrix multiply).
  for (int p = 0; p < d; ++p)
    for (int q = 0; q < d; ++q) {
      AB[p * d + q] = 0.0;
      for (int r = 0; r < d; ++r) AB[p * d + q] += op_A[p * d + r] * op_B[r * d + q];
    }

  // Diagonal elements.
  for (int i = 0; i < L; ++i) {
    Tensor t = theta1(i);
    int cL = t.shape[0], cR = t.shape[2];
    Cdouble val = 0.0;
    for (int a = 0; a < cL; ++a)
      for (int p = 0; p < d; ++p)
        for (int b = 0; b < cR; ++b) {
          Cdouble ot = 0.0;
          for (int q = 0; q < d; ++q) ot += AB[p * d + q] * t(a, q, b);
          val += std::conj(t(a, p, b)) * ot;
        }
    out[i * L + i] = std::real(val);
  }

  // Upper triangle: i < j, propagate transfer matrix to the right.
  for (int i = 0; i < L; ++i) {
    Tensor ti = theta1(i);
    int cL = ti.shape[0], cR_i = ti.shape[2];

    // left(a*, b) = sum_{p, a} conj(theta_i(a, p, a*)) * (A @ theta_i)(a, p', b)
    // Actually: left = sum_p conj(theta_i)^T_{a*,p} @ (op_A @ theta_i)_{p, b}
    // Dimensions: left is (cR_i x cR_i)
    std::vector<Cdouble> left(cR_i * cR_i, 0.0);
    for (int a = 0; a < cL; ++a) {
      for (int p = 0; p < d; ++p) {
        // A_theta(b) = sum_q op_A(p, q) * ti(a, q, b)
        std::vector<Cdouble> A_theta(cR_i, 0.0);
        for (int b = 0; b < cR_i; ++b)
          for (int q = 0; q < d; ++q)
            A_theta[b] += op_A[p * d + q] * ti(a, q, b);
        for (int b1 = 0; b1 < cR_i; ++b1)
          for (int b2 = 0; b2 < cR_i; ++b2)
            left[b1 * cR_i + b2] +=
                std::conj(ti(a, p, b1)) * A_theta[b2];
      }
    }

    for (int j = i + 1; j < L; ++j) {
      int cL_j = Bs[j].shape[0], cR_j = Bs[j].shape[2];
      assert(cL_j == cR_i || (j == i + 1 && cL_j == cR_i));
      int cl = cL_j;  // should equal current left dim

      // Compute C[i,j] = sum_{a,b} left(a, b) * <B_j>(a, b)
      // <B_j>(a, b) = sum_p conj(Bj(a, p, b')) * (op_B @ Bj)(a, p', b')
      // Wait, let's be more careful.
      // C[i,j] = Tr[left @ T_j^B]
      // T_j^B(b1, b2) = sum_p conj(Bj(b1, p, b2')) * sum_q op_B(p,q) * Bj(b1_in, q, b2)
      // Hmm, this needs left to contract on the left indices of Bj.

      // C[i,j] = sum_{a, b, p, q, c} left(a, b) * conj(Bj(a, p, c)) * op_B(p,q) * Bj(b, q, c)
      Cdouble val = 0.0;
      for (int a = 0; a < cl; ++a) {
        for (int b = 0; b < cl; ++b) {
          Cdouble lft = left[a * cl + b];
          if (std::abs(lft) < 1e-30) continue;
          for (int p = 0; p < d; ++p) {
            for (int c = 0; c < cR_j; ++c) {
              Cdouble bj_conj = std::conj(Bs[j](a, p, c));
              Cdouble ob = 0.0;
              for (int q = 0; q < d; ++q)
                ob += op_B[p * d + q] * Bs[j](b, q, c);
              val += lft * bj_conj * ob;
            }
          }
        }
      }
      out[i * L + j] = std::real(val);

      // Advance left through site j with identity operator.
      // new_left(c1, c2) = sum_{a, b, p} conj(Bj(a, p, c1)) * left(a, b) * Bj(b, p, c2)
      std::vector<Cdouble> new_left(cR_j * cR_j, 0.0);
      for (int a = 0; a < cl; ++a) {
        for (int b = 0; b < cl; ++b) {
          Cdouble lft = left[a * cl + b];
          if (std::abs(lft) < 1e-30) continue;
          for (int p = 0; p < d; ++p) {
            for (int c1 = 0; c1 < cR_j; ++c1) {
              Cdouble bc1 = std::conj(Bs[j](a, p, c1));
              for (int c2 = 0; c2 < cR_j; ++c2) {
                new_left[c1 * cR_j + c2] +=
                    lft * bc1 * Bs[j](b, p, c2);
              }
            }
          }
        }
      }
      left = std::move(new_left);
      cR_i = cR_j;
    }
  }

  // Lower triangle: C[i,j] for i > j, propagate from site j to the right.
  for (int j = 0; j < L; ++j) {
    Tensor tj = theta1(j);
    int cL = tj.shape[0], cR_j = tj.shape[2];

    // right(a*, b) from site j with op_B.
    std::vector<Cdouble> right(cR_j * cR_j, 0.0);
    for (int a = 0; a < cL; ++a) {
      for (int p = 0; p < d; ++p) {
        std::vector<Cdouble> B_theta(cR_j, 0.0);
        for (int b = 0; b < cR_j; ++b)
          for (int q = 0; q < d; ++q)
            B_theta[b] += op_B[p * d + q] * tj(a, q, b);
        for (int b1 = 0; b1 < cR_j; ++b1)
          for (int b2 = 0; b2 < cR_j; ++b2)
            right[b1 * cR_j + b2] +=
                std::conj(tj(a, p, b1)) * B_theta[b2];
      }
    }

    int cur_dim = cR_j;
    for (int i = j + 1; i < L; ++i) {
      int cL_i = Bs[i].shape[0], cR_i = Bs[i].shape[2];
      int cl = cL_i;

      // C[i,j] with op_A at site i.
      Cdouble val = 0.0;
      for (int a = 0; a < cl; ++a) {
        for (int b = 0; b < cl; ++b) {
          Cdouble rht = right[a * cl + b];
          if (std::abs(rht) < 1e-30) continue;
          for (int p = 0; p < d; ++p) {
            for (int c = 0; c < cR_i; ++c) {
              Cdouble bi_conj = std::conj(Bs[i](a, p, c));
              Cdouble oa = 0.0;
              for (int q = 0; q < d; ++q)
                oa += op_A[p * d + q] * Bs[i](b, q, c);
              val += rht * bi_conj * oa;
            }
          }
        }
      }
      out[i * L + j] = std::real(val);

      // Advance right through site i with identity.
      std::vector<Cdouble> new_right(cR_i * cR_i, 0.0);
      for (int a = 0; a < cl; ++a) {
        for (int b = 0; b < cl; ++b) {
          Cdouble rht = right[a * cl + b];
          if (std::abs(rht) < 1e-30) continue;
          for (int p = 0; p < d; ++p) {
            for (int c1 = 0; c1 < cR_i; ++c1) {
              Cdouble bc1 = std::conj(Bs[i](a, p, c1));
              for (int c2 = 0; c2 < cR_i; ++c2) {
                new_right[c1 * cR_i + c2] +=
                    rht * bc1 * Bs[i](b, p, c2);
              }
            }
          }
        }
      }
      right = std::move(new_right);
      cur_dim = cR_i;
    }
  }
}

Tensor MPS::to_state_vector() const {
  // Build diag(Ss[0]) @ Bs[0] @ Bs[1] @ ... @ Bs[L-1] and flatten.
  // psi is stored as 2D: (front, chi) where front = d^(sites contracted so far).
  Tensor t1 = theta1(0);  // (chi_L=1, d, chi_R)
  int chi_R0 = t1.shape[2];
  // Reshape to (d, chi_R).
  int sh_init[2] = {d, chi_R0};
  t1.reshape(2, sh_init);
  Tensor psi = std::move(t1);

  for (int i = 1; i < L; ++i) {
    int chi_mid = psi.shape[1];  // right bond dim = chi_R of previous site
    int chi_R = Bs[i].shape[2];
    int front = psi.shape[0];    // d^i (accumulated physical dimensions)

    // Flatten Bs[i] to 2D: (chi_mid, d * chi_R)
    int sh2_b[2] = {chi_mid, d * chi_R};
    Tensor Bi_flat(2, sh2_b);
    for (int a = 0; a < chi_mid; ++a)
      for (int p = 0; p < d; ++p)
        for (int b = 0; b < chi_R; ++b)
          Bi_flat(a, p * chi_R + b) = Bs[i](a, p, b);

    // GEMM: (front, chi_mid) @ (chi_mid, d*chi_R) = (front, d*chi_R)
    int sh_out[2] = {front * d, chi_R};
    Tensor new_psi(2, sh_out);
    // The GEMM output is (front, d*chi_R), which we interpret as (front*d, chi_R).
    zgemm(psi.data, chi_mid, Bi_flat.data, d * chi_R, new_psi.data,
          d * chi_R, front, d * chi_R, chi_mid);
    // Reshape the flat buffer: (front, d*chi_R) = (front*d, chi_R).
    // Data layout is the same since it's row-major. new_psi already has the right shape.
    psi = std::move(new_psi);
  }

  // At this point, psi has shape (d^L, 1) since the rightmost chi_R = 1.
  // Reshape to 1D.
  int total = psi.size;
  int sh1[1] = {total};
  psi.reshape(1, sh1);
  return psi;
}
