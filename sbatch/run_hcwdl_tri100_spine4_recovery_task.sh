#!/usr/bin/env bash
set -euo pipefail
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
: "${PROJECT_DIR:?PROJECT_DIR is required}"
: "${HCWDL_SPINE4_RECOVERY:?HCWDL_SPINE4_RECOVERY is required}"
: "${HCWDL_SPINE4_TASK:?HCWDL_SPINE4_TASK is required}"
: "${HCWDL_SPINE4_DDP_WORLD_SIZE:?HCWDL_SPINE4_DDP_WORLD_SIZE is required}"
source "${PROJECT_DIR}/sbatch/common.sh"
hlt_activate
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export NUMEXPR_MAX_THREADS=64
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
if [ "${HCWDL_SPINE4_DDP_WORLD_SIZE}" -eq 4 ]; then
  test "${SLURM_NNODES}" -eq 4
  mapfile -t HCWDL_SPINE4_HOSTS < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
  test "${#HCWDL_SPINE4_HOSTS[@]}" -eq 4
  export MASTER_ADDR="${HCWDL_SPINE4_HOSTS[0]}"
  export MASTER_PORT="$((20000 + SLURM_JOB_ID % 20000))"
  exec srun \
    --nodes=4 --ntasks=4 --ntasks-per-node=1 \
    --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
    --gpus-per-task=1 --gpu-bind=single:1 --kill-on-bad-exit=1 \
    python -s "${PROJECT_DIR}/scripts/run_hcwdl_tri100_spine4_recovery_task.py" \
      --recovery "${HCWDL_SPINE4_RECOVERY}" \
      --task "${HCWDL_SPINE4_TASK}" --device cuda \
      --distributed-world-size 4
fi
exec python -s "${PROJECT_DIR}/scripts/run_hcwdl_tri100_spine4_recovery_task.py" \
  --recovery "${HCWDL_SPINE4_RECOVERY}" \
  --task "${HCWDL_SPINE4_TASK}" --device cuda --distributed-world-size 1
