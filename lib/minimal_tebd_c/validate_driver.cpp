#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <vector>

#include "model.h"
#include "validate.h"

// Standalone driver: runs TEBD vs exact diagonalization on a small chain
// and dumps per-step error series to CSV for plotting.

static void mkdirp(const std::string& dir) { ::mkdir(dir.c_str(), 0755); }

int main(int argc, char** argv) {
  int L = 8;
  double dt = 0.05;
  double t_max = 2.0;
  int chi_max = 64;
  int order = 4;
  std::string outdir = "output";

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto need = [&](int k) {
      if (i + k >= argc) { std::cerr << "missing value for " << a << "\n"; std::exit(1); }
    };
    if (a == "-L") { need(1); L = std::stoi(argv[++i]); }
    else if (a == "--dt") { need(1); dt = std::stod(argv[++i]); }
    else if (a == "--tmax") { need(1); t_max = std::stod(argv[++i]); }
    else if (a == "--chi") { need(1); chi_max = std::stoi(argv[++i]); }
    else if (a == "--order") { need(1); order = std::stoi(argv[++i]); }
    else if (a == "--out") { need(1); outdir = argv[++i]; }
    else { std::cerr << "unknown arg: " << a << "\n"; return 1; }
  }
  mkdirp(outdir);

  BondModel model = tfi_chain(L, 1.0, 1.0);
  std::cout << "# validate_driver L=" << L << " dt=" << dt << " tmax=" << t_max
            << " chi=" << chi_max << " order=" << order << "\n";

  ValidationSummary s = validate_tebd_vs_ed(
      model, dt, t_max, chi_max, /*svd_min=*/1e-12, order, nullptr,
      /*verbose=*/true);

  std::ofstream f(outdir + "/validate_errors.csv");
  f << "t,err_Sz,err_Sx,err_corr_XX,err_entropy\n";
  for (size_t i = 0; i < s.t.size(); ++i) {
    f << s.t[i] << "," << s.err_Sz[i] << "," << s.err_Sx[i] << ","
      << s.err_corr_XX[i] << "," << s.err_entropy[i] << "\n";
  }
  std::cout << "# wrote " << outdir << "/validate_errors.csv  (fidelity_drop="
            << s.fidelity_drop << ")\n";
  return 0;
}
