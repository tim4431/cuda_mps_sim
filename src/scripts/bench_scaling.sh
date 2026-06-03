#!/bin/bash
#SBATCH --job-name=cme_scaling
#SBATCH --partition=gpu-turing
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --time=00:15:00
#SBATCH --output=slurm_%j.out

# Strong + weak MPI scaling measurement for the milestone_4 section 6 table.
# Reports t_rank_max_s ( = max_r sum_i t_i ) per (mode, ranks) and a computed
# speedup / parallel-efficiency table.
#
# Submit from lib/minimal_tebd_c/ :
#   sbatch scripts/bench_scaling.sh
# Override problem size (e.g. if the job risks the 15-min cap, lower STEPS):
#   sbatch --export=ALL,CHI=64,STEPS=6 scripts/bench_scaling.sh

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
if [ ! -f Makefile.local ] && [ -f lib/minimal_tebd_c/Makefile.local ]; then
  cd lib/minimal_tebd_c
fi
if [ ! -f Makefile.local ]; then
  echo "ERROR: cannot find Makefile.local (cwd=$(pwd)). Submit from lib/minimal_tebd_c/."
  exit 1
fi

CHI=${CHI:-64}
STEPS=${STEPS:-10}

echo "=== ENV ==="; hostname; date
echo "build dir: $(pwd)   CHI=$CHI  STEPS=$STEPS"
nvidia-smi -L || true; echo

echo "=== BUILD (incremental) ==="
make -f Makefile.local apps/sweep_mpi || { echo "BUILD FAILED"; exit 1; }
echo "build OK"; echo

SUM=$(mktemp)

run () {                       # $1 = --strong|--weak   $2 = ranks
  local flag="$1" P="$2" tag="${1#--}"
  echo "--- mode=$tag ranks=$P (chi=$CHI steps=$STEPS) ---"
  mpirun -n "$P" ./apps/sweep_mpi "$flag" --chi "$CHI" --steps "$STEPS" \
      > "sweep_${tag}_p${P}.csv" 2> "sweep_${tag}_p${P}.err"
  grep '^# scaling:' "sweep_${tag}_p${P}.err" | tee -a "$SUM"
}

# Strong first (the primary table) so a timeout still leaves it complete.
echo "=== STRONG SCALING ==="
for P in 1 2 4; do run --strong "$P"; done
echo
echo "=== WEAK SCALING ==="
for P in 1 2 4; do run --weak "$P"; done
echo

echo "=== SUMMARY (speedup vs that mode's 1-rank t_rank_max_s) ==="
awk '
{
  for (i=1;i<=NF;i++) if ($i ~ /=/) { split($i,kv,"="); v[kv[1]]=kv[2] }
  m=v["mode"]; P=v["n_ranks"]+0;
  T[m,P]=v["t_rank_max_s"]+0;
}
END{
  printf "%-7s %6s %14s %9s %8s\n","mode","ranks","t_rank_max_s","speedup","eff(%)";
  split("strong weak", MM, " ");
  split("1 2 4", RR, " ");
  for (mi=1; mi<=2; mi++){
    m=MM[mi]; if (!((m SUBSEP 1) in T)) continue;
    base=T[m,1];
    for (ri=1; ri<=3; ri++){
      P=RR[ri]+0; if (!((m SUBSEP P) in T)) continue;
      sp=base/T[m,P]; ef=100*sp/P;
      printf "%-7s %6d %14.4f %9.2f %8.1f\n", m, P, T[m,P], sp, ef;
    }
  }
}
' "$SUM"

rm -f "$SUM"
echo
echo "=== DONE ==="; date
