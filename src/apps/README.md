# `apps/` — driver and benchmark binaries

Each `.cpp` is a standalone `main()`. The build picks them up via `$(wildcard)`,
so adding a file here just works. CPU-only apps build with `g++`; the rest
require `nvcc` (and MPI for `sweep_mpi`).

## Production / physics

### `sim` — CPU quench driver
Initialises the chain in `|0…0⟩` and evolves under the TFIM Hamiltonian,
dumping observable time series to CSV in an output directory (`summary`,
`sz_grid`, `sx_grid`, `entropy_grid`, `chi_grid`, …). The reference run behind
the physics figures in the reports. CPU-only; was the M2 `main.cpp`.

### `sweep_mpi` — multi-GPU MPI parameter sweep
The CUDA + MPI deliverable. Each rank calls `cudaSetDevice(rank % n_gpus)` and
runs **independent** GPU TEBD evolutions over `(J, g)` points it builds locally
(no params are scattered), with *zero in-loop communication*. At the end rank 0
gathers per-rank record counts (`MPI_Gather`) and then the
`(J, g, t_wall, max_χ, S_mid)` records themselves (`MPI_Gatherv`, which handles
the uneven per-rank counts of strong mode). CSV to stdout:
`rank,J,g,L,chi_max,N_steps,t_wall_s,max_chi,final_entropy`.

```
mpirun -n <P> ./sweep_mpi [options]
  --L <int>         chain length        (20)
  --chi <int>       max bond dimension  (64)
  --dt <float>      Trotter step        (0.1)
  --steps <int>     TEBD steps          (20)
  --order <int>     Trotter order 2|4   (2)
  --n-streams <int> streams per rank    (1)
  --strong          16-pair fixed grid split across ranks   (default)
  --weak            4 pairs/rank, total work grows with P
```

`--strong` measures strong scaling (fixed 16-pair `4×4` grid, round-robin to
ranks); `--weak` measures weak scaling (constant 4 pairs/rank).

## Correctness validators

### `validate_gpu`
Runs the M3 per-bond GPU path (`tebd_step_gpu`) side-by-side with the CPU
`update_bond` from the same product state and compares `⟨σᶻ⟩`, `⟨σˣ⟩`, and the
`L-1` entropies at every step. Tolerances are loose (the two SVDs differ by
column phase) but observables and singular values must match (`~10⁻¹³`).

### `validate_persistent`
Same idea for the persistent device-MPS path (`tebd_step_gpu_persistent`, the
one the sweep uses), syncing `GpuMps` back to host each step. Confirms the
1-stream and 4-stream configurations agree to every printed digit.

## Benchmarks

| Binary | Measures | CSV columns |
|---|---|---|
| `bench_gpu` | full-step wall time: CPU / M3 GPU / persistent / streamed | `L,chi_max,N_steps,t_cpu_s,t_gpu_s,speedup,t_persistent_s,t_streamed_s,max_chi` |
| `bench_update` | single `update_bond` at fixed `χ`, CPU vs GPU (isolates kernel cost, no variable-`χ` noise) | — |
| `bench_breakdown` | per-stage GPU time of `update_bond_gpu` vs `χ`, via CUDA events | `chi,h2d_ms,gate_ms,r2c_ms,svd_ms,restore_ms,d2h_ms,total_ms` |
| `bench_svd` | cuSOLVER `gesvdj` vs custom Jacobi per `χ_max` (time + Schmidt-spectrum error) | `chi_max,N_steps,t_cusolver_s,t_jacobi_s,ratio,max_err` |

`bench_gpu`, `bench_update`, and `bench_svd` accept either a default sweep or a
single `L chi_max N_steps` (resp. `chi_max N_steps`) config on the command line.

## Running on the cluster

GPU apps are submitted to the `gpu-turing` partition (1 GPU for the validators
and single-GPU benchmarks, 4 GPUs for the sweep):

```bash
sbatch --partition=gpu-turing --gres=gpu:1 --wrap="./apps/bench_gpu"
sbatch --partition=gpu-turing --gres=gpu:4 --ntasks=4 \
       --wrap="mpirun -n 4 ./apps/sweep_mpi --chi 64 --steps 20"
```

The wrapper scripts in [`../scripts/`](../scripts/) (`submit_all.sh`,
`verify.sh`, `bench_scaling.sh`) batch these up and feed the plotters.
