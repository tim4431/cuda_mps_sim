#!/bin/bash
# Submit all M4 benchmark jobs to the gpu-turing partition.
# Run this script from anywhere — it uses its own absolute path.

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Project dir: $DIR"
echo

# validate_gpu
JID=$(sbatch --partition=gpu-turing --gres=gpu:1 \
  --job-name=validate_gpu --output="$DIR/validate_gpu.out" \
  --wrap="cd '$DIR' && ./apps/validate_gpu" \
  | awk '{print $NF}')
echo "validate_gpu   -> job $JID  (output: validate_gpu.out)"

# bench_gpu  (CPU / M3 / M4-persistent / M4-streamed)
JID=$(sbatch --partition=gpu-turing --gres=gpu:1 \
  --job-name=bench_gpu --output="$DIR/bench_gpu.out" \
  --wrap="cd '$DIR' && ./apps/bench_gpu > '$DIR/bench_gpu.csv'" \
  | awk '{print $NF}')
echo "bench_gpu      -> job $JID  (data: bench_gpu.csv)"

# Nsight Systems profile — single GPU (bench_gpu, compact run: L=20, chi=64, 3 steps)
# Captures GPU kernel breakdown; binary can be opened in Nsight Systems GUI.
JID=$(sbatch --partition=gpu-turing --gres=gpu:1 \
  --job-name=nsys_m4 --output="$DIR/nsys_m4.out" \
  --wrap="cd '$DIR' && nsys profile --stats=true -o '$DIR/nsys_m4' \
          ./apps/bench_gpu 20 64 3" \
  | awk '{print $NF}')
echo "nsys single-GPU -> job $JID  (binary: nsys_m4.nsys-rep, stats: nsys_m4.out)"

# Nsight Systems profile — multi-GPU distributed run (sweep_mpi, 4 ranks, 3 steps)
# Profiles each MPI rank separately; confirms zero in-loop MPI communication.
# Generates nsys_sweep_{0,1,2,3}.nsys-rep (one per rank).
JID=$(sbatch --partition=gpu-turing --gres=gpu:4 --ntasks=4 \
  --job-name=nsys_sweep --output="$DIR/nsys_sweep.out" \
  --wrap="cd '$DIR' && mpirun -n 4 nsys profile --stats=true \
          -o '$DIR/nsys_sweep_%q{OMPI_COMM_WORLD_RANK}' \
          ./apps/sweep_mpi --chi 64 --steps 3 --strong" \
  | awk '{print $NF}')
echo "nsys multi-GPU  -> job $JID  (binaries: nsys_sweep_{0..3}.nsys-rep, stats: nsys_sweep.out)"

# bench_svd  (cuSOLVER vs custom Jacobi)
JID=$(sbatch --partition=gpu-turing --gres=gpu:1 \
  --job-name=bench_svd --output="$DIR/bench_svd.out" \
  --wrap="cd '$DIR' && ./apps/bench_svd > '$DIR/bench_svd.csv'" \
  | awk '{print $NF}')
echo "bench_svd      -> job $JID  (data: bench_svd.csv)"

# MPI strong scaling at 1, 2, 4 ranks
# (gpu-turing nodes have at most 4 GPUs; requesting more causes sbatch to fail
#  with "Requested node configuration is not available".)
for NP in 1 2 4 8; do
  JID=$(sbatch --partition=gpu-turing \
    --gres=gpu:$NP --ntasks=$NP \
    --job-name="sweep_p${NP}" --output="$DIR/sweep_p${NP}.out" \
    --wrap="cd '$DIR' && mpirun -n $NP ./apps/sweep_mpi \
            --chi 64 --steps 20 --strong \
            > '$DIR/strong_p${NP}.csv'" \
    | awk '{print $NF}')
  echo "sweep strong n=$NP -> job $JID  (data: strong_p${NP}.csv)"
done

# MPI weak scaling at 1, 2, 4 ranks
for NP in 1 2 4 8; do
  JID=$(sbatch --partition=gpu-turing \
    --gres=gpu:$NP --ntasks=$NP \
    --job-name="sweakp${NP}" --output="$DIR/weak_p${NP}.out" \
    --wrap="cd '$DIR' && mpirun -n $NP ./apps/sweep_mpi \
            --chi 64 --steps 20 --weak \
            > '$DIR/weak_p${NP}.csv'" \
    | awk '{print $NF}')
  echo "sweep weak   n=$NP -> job $JID  (data: weak_p${NP}.csv)"
done

echo
echo "Check status with:  squeue -u \$USER"
echo
echo "After all jobs complete, regenerate report figures with:"
echo "  python3 scripts/plot_m4_gpu.py bench_gpu.csv \\"
echo "      --out ../../doc/figs/m4_gpu_timing.png"
echo "  python3 scripts/plot_svd_compare.py bench_svd.csv \\"
echo "      --out ../../doc/figs/svd_compare.png"
echo "  python3 scripts/plot_scaling.py \\"
echo "      --strong strong_p1.csv strong_p2.csv strong_p4.csv \\"
echo "      --weak   weak_p1.csv   weak_p2.csv   weak_p4.csv \\"
echo "      --out    ../../doc/figs/scaling_m4.png"
