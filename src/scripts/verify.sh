#!/bin/bash
#SBATCH --job-name=cme_verify
#SBATCH --partition=gpu-turing
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --time=00:15:00
#SBATCH --output=slurm_%j.out

# End-to-end correctness verification for the M4 (distributed/persistent) work.
# Submit from src/ :  sbatch scripts/verify.sh
# Then send back the resulting slurm_<jobid>.out.

set -uo pipefail

# Resolve the build dir robustly.  sbatch copies this script to a spool dir, so
# $0 is unreliable; the job starts in SLURM_SUBMIT_DIR (where sbatch was run).
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
if [ ! -f Makefile.local ] && [ -f src/Makefile.local ]; then
  cd src
fi
if [ ! -f Makefile.local ]; then
  echo "ERROR: cannot find Makefile.local (cwd=$(pwd)). Submit from src/."
  exit 1
fi

echo "=== ENV ==="; hostname; date; echo "build dir: $(pwd)"; nvidia-smi -L || true; echo

echo "=== BUILD ==="
make -f Makefile.local clean
make -f Makefile.local       || { echo "BUILD FAILED (apps)";  exit 1; }
make -f Makefile.local tests || { echo "BUILD FAILED (tests)"; exit 1; }
echo "build OK"; echo

echo "=== UNIT TESTS (CPU) ==="
make -f Makefile.local run_tests; echo

echo "=== validate_gpu (M3 path) ==="
./apps/validate_gpu 12 32 30; echo

echo "=== validate_persistent (M4 GpuMps path) ==="
./apps/validate_persistent 12 32 30 1     # 1 stream
./apps/validate_persistent 12 32 30 4     # 4 streams (exercises stream path)
echo

echo "=== bench_svd (custom Jacobi vs cuSOLVER) ==="
./apps/bench_svd 64 10; echo

echo "=== sweep_mpi consistency: strong, chi=32, steps=10, ranks 1/2/4 ==="
for P in 1 2 4; do
  echo "--- ranks=$P ---"
  mpirun -n $P ./apps/sweep_mpi --strong --chi 32 --steps 10 \
      > sweep_p${P}.csv 2> sweep_p${P}.err
  cat sweep_p${P}.err                      # the "# scaling:" summary line
done
norm () { tail -n +2 "$1" | awk -F, '{printf "%.2f,%.2f,%d,%.6f\n",$2,$3,$8,$9}' | sort; }
norm sweep_p1.csv > .s1; norm sweep_p2.csv > .s2; norm sweep_p4.csv > .s4
if diff -q .s1 .s2 >/dev/null && diff -q .s1 .s4 >/dev/null; then
  echo "CONSISTENCY: MATCH (1 == 2 == 4 ranks)"
else
  echo "CONSISTENCY: MISMATCH"
  echo "-- 1 vs 2 --"; diff .s1 .s2 | head
  echo "-- 1 vs 4 --"; diff .s1 .s4 | head
fi
rm -f .s1 .s2 .s4; echo
echo "=== DONE ==="; date
