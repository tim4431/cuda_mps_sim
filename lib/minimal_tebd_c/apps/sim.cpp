#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <vector>

#include "model.h"
#include "mps.h"
#include "tebd.h"

// Quench driver. Initializes the chain in |0...0> (ferromagnetic in sigma_z)
// and evolves under the transverse-field Ising Hamiltonian, dumping observables
// to CSV files in an output directory.

static void mkdirp(const std::string& dir) {
  ::mkdir(dir.c_str(), 0755);
}

int main(int argc, char** argv) {
  // Defaults: large enough to be interesting, fast enough for a milestone run.
  int L = 20;
  double dt = 0.05;
  int order = 4;
  int chi_max = 64;
  double svd_min = 1e-12;
  double t_max = 4.0;
  std::string outdir = "output";

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto need = [&](int k) {
      if (i + k >= argc) {
        std::cerr << "missing value for " << a << std::endl;
        std::exit(1);
      }
    };
    if (a == "-L") { need(1); L = std::stoi(argv[++i]); }
    else if (a == "--dt") { need(1); dt = std::stod(argv[++i]); }
    else if (a == "--order") { need(1); order = std::stoi(argv[++i]); }
    else if (a == "--chi") { need(1); chi_max = std::stoi(argv[++i]); }
    else if (a == "--tmax") { need(1); t_max = std::stod(argv[++i]); }
    else if (a == "--out") { need(1); outdir = argv[++i]; }
    else {
      std::cerr << "unknown arg: " << a << std::endl;
      return 1;
    }
  }

  mkdirp(outdir);

  BondModel model = tfi_chain(L, 1.0, 1.0);
  std::vector<int> init(L, 0);
  MPS psi = MPS::product_state(L, init.data(), model.d, chi_max);
  TEBDEngine eng(&psi, &model, dt, order, 1, chi_max, svd_min);

  std::vector<double> entropy(L - 1);
  std::vector<double> Sx(L), Sz(L);
  std::vector<int> bonds(L - 1);

  // Open CSV files.
  std::ofstream f_summary(outdir + "/summary.csv");
  std::ofstream f_entropy(outdir + "/entropy_grid.csv");
  std::ofstream f_sz(outdir + "/sz_grid.csv");
  std::ofstream f_sx(outdir + "/sx_grid.csv");
  std::ofstream f_chi(outdir + "/chi_grid.csv");

  f_summary << "t,S_mid,max_chi,trunc_err\n";
  // For grid files, header = "t,col0,col1,...".
  auto write_header = [&](std::ofstream& f, int n, const char* prefix) {
    f << "t";
    for (int k = 0; k < n; ++k) f << "," << prefix << k;
    f << "\n";
  };
  write_header(f_entropy, L - 1, "S_b");
  write_header(f_sz, L, "Sz_");
  write_header(f_sx, L, "Sx_");
  write_header(f_chi, L - 1, "chi_b");

  auto measure = [&]() {
    psi.entropy(entropy.data());
    psi.expect(PAULI_X, Sx.data());
    psi.expect(PAULI_Z, Sz.data());
    psi.bond_dims(bonds.data());
  };

  auto write_grid_row = [](std::ofstream& f, double t,
                           const double* v, int n, int prec) {
    f << std::fixed << std::setprecision(3) << t;
    f << std::scientific << std::setprecision(prec);
    for (int k = 0; k < n; ++k) f << "," << v[k];
    f << "\n";
  };
  auto write_grid_row_int = [](std::ofstream& f, double t,
                               const int* v, int n) {
    f << std::fixed << std::setprecision(3) << t;
    for (int k = 0; k < n; ++k) f << "," << v[k];
    f << "\n";
  };

  auto write_summary_row = [&]() {
    int mid = L / 2 - 1;
    f_summary << std::fixed << std::setprecision(3) << eng.evolved_time
              << "," << std::setprecision(6) << entropy[mid] << ","
              << psi.max_bond_dim() << "," << std::scientific
              << std::setprecision(3) << eng.trunc_err_eps << "\n";
  };

  std::cout << "# minimal_tebd_c quench driver\n"
            << "# L=" << L << "  dt=" << dt << "  order=" << order
            << "  chi_max=" << chi_max << "  t_max=" << t_max
            << "  outdir=" << outdir << "\n";
  std::cout << "# t,S_mid,max_chi,trunc_err" << std::endl;

  measure();
  write_summary_row();
  write_grid_row(f_entropy, eng.evolved_time, entropy.data(), L - 1, 6);
  write_grid_row(f_sz, eng.evolved_time, Sz.data(), L, 6);
  write_grid_row(f_sx, eng.evolved_time, Sx.data(), L, 6);
  write_grid_row_int(f_chi, eng.evolved_time, bonds.data(), L - 1);

  while (eng.evolved_time < t_max - 1e-9) {
    eng.run();
    measure();
    write_summary_row();
    write_grid_row(f_entropy, eng.evolved_time, entropy.data(), L - 1, 6);
    write_grid_row(f_sz, eng.evolved_time, Sz.data(), L, 6);
    write_grid_row(f_sx, eng.evolved_time, Sx.data(), L, 6);
    write_grid_row_int(f_chi, eng.evolved_time, bonds.data(), L - 1);

    int mid = L / 2 - 1;
    std::printf("t=%.2f  max_chi=%3d  S_mid=%.4f  trunc=%.2e\n",
                eng.evolved_time, psi.max_bond_dim(), entropy[mid],
                eng.trunc_err_eps);
    std::fflush(stdout);
  }

  std::cout << "# wrote: " << outdir
            << "/{summary,entropy_grid,sz_grid,sx_grid,chi_grid}.csv\n";
  return 0;
}
