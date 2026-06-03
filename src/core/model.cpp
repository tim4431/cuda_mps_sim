#include "model.h"

#include <cassert>

#include "linalg.h"
#include "mps.h"  // for PAULI constants

// ---------------------------------------------------------------------------
// BondModel
// ---------------------------------------------------------------------------

BondModel::BondModel() : L(0), d(0), H_bonds(nullptr) {}

BondModel::~BondModel() { delete[] H_bonds; }

BondModel::BondModel(BondModel&& other) noexcept
    : L(other.L), d(other.d), H_bonds(other.H_bonds) {
  other.H_bonds = nullptr;
  other.L = 0;
}

BondModel& BondModel::operator=(BondModel&& other) noexcept {
  if (this == &other) return *this;
  delete[] H_bonds;
  L = other.L;
  d = other.d;
  H_bonds = other.H_bonds;
  other.H_bonds = nullptr;
  other.L = 0;
  return *this;
}

// ---------------------------------------------------------------------------
// TFI chain
// ---------------------------------------------------------------------------

BondModel tfi_chain(int L, double J, double g) {
  BondModel model;
  model.L = L;
  model.d = 2;
  int nbonds = L - 1;
  model.H_bonds = new Tensor[nbonds];

  // Pauli matrices (real parts only, since X and Z are real).
  double X[4] = {0, 1, 1, 0};
  double Z[4] = {1, 0, 0, -1};
  double Id[4] = {1, 0, 0, 1};

  for (int b = 0; b < nbonds; ++b) {
    // On-site Z field split across bonds.
    double gL = (b != 0) ? 0.5 * g : g;
    double gR = (b + 1 != L - 1) ? 0.5 * g : g;

    // H_bond = -J * kron(X, X) - gL * kron(Z, I) - gR * kron(I, Z)
    // All 4x4 matrices, then reshape to (2,2,2,2).
    Cdouble XX[16], ZI[16], IZ[16];
    Cdouble cX[4], cZ[4], cId[4];
    for (int i = 0; i < 4; ++i) {
      cX[i] = X[i];
      cZ[i] = Z[i];
      cId[i] = Id[i];
    }
    kron(cX, 2, 2, cX, 2, 2, XX);
    kron(cZ, 2, 2, cId, 2, 2, ZI);
    kron(cId, 2, 2, cZ, 2, 2, IZ);

    int sh[4] = {2, 2, 2, 2};
    model.H_bonds[b] = Tensor(4, sh);
    for (int i = 0; i < 16; ++i) {
      model.H_bonds[b].data[i] =
          -J * XX[i] - gL * ZI[i] - gR * IZ[i];
    }
  }
  return model;
}
