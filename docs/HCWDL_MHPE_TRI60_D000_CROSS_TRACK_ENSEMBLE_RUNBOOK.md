# TRI60 four-D000 cross-track ensemble diagnostic

## Purpose and scope

This is a separate exploratory validation-only calculation over four already
completed exact-HLT D000 models:

1. `LOGIT_D000_from_U000`;
2. `LOGIT_D000_from_U050E`;
3. `RSET_D000_from_U000`; and
4. `RREL_D000_from_U000`.

The component list is frozen in source. Path discovery cannot add a later
D000 fit. The primary ensemble applies each model's FP32 softmax at temperature
one and averages the four probabilities with exact uniform weight 1/4 using
FP64 accumulation. Four predeclared leave-one-out averages are reported only
to diagnose a harmful member; they do not select or redefine the primary
result.

The worker streams the authenticated validation HLT view once and forwards all
four models over each batch. It never reads final test and writes only one small
content-hashed JSON report. In-memory logits, probabilities, and particle views
are released on exit. No active TRI60 dependency, fit, reducer, lock, or
checkpoint is changed.

## Preconditions

- The source-pinned TRI60 campaign spec remains at
  `/home/ryreu/atlas/HLT_Classification/checkpoints/hcwdl_mhpe_tri60_full_d218961c_r1/campaign_spec.json`.
- Each frozen component has a complete authenticated `training_report.json`
  and selected checkpoint under that campaign root.
- The implementation commit has been pushed and is checked out in a clean
  detached worktree.
- The output path does not already exist.

Campaign completion is intentionally not a precondition.

## Source-pinned Tigris submission

After substituting the pushed 40-character commit, run this block. It submits
one independent job and does not cancel, hold, release, or depend on any active
TRI60 job.

```bash
set -euo pipefail

source /home/ryreu/miniforge3-aarch64/etc/profile.d/conda.sh
conda activate atlas_kd_tigris

export MAIN_REPO=/home/ryreu/atlas/HLT_Classification
export DIAG_COMMIT=REPLACE_WITH_PUSHED_40_CHARACTER_COMMIT
export DIAG_SHORT="${DIAG_COMMIT:0:8}"
export PROJECT_DIR="/home/ryreu/atlas/HLT_Classification_tri60_d000x4_${DIAG_SHORT}"
export TRI60_SPEC="${MAIN_REPO}/checkpoints/hcwdl_mhpe_tri60_full_d218961c_r1/campaign_spec.json"
export DIAG_ROOT="${MAIN_REPO}/checkpoints/hcwdl_mhpe_tri60_d000x4_${DIAG_SHORT}_r1"
export DIAG_OUTPUT="${DIAG_ROOT}/validation_report.json"

cd "${MAIN_REPO}"
git fetch origin main
test "$(git rev-parse origin/main)" = "${DIAG_COMMIT}"
test -f "${TRI60_SPEC}"
test ! -e "${DIAG_OUTPUT}"

for node in \
  LOGIT_D000_from_U000 \
  LOGIT_D000_from_U050E \
  RSET_D000_from_U000 \
  RREL_D000_from_U000
do
  test -f "$(dirname "${TRI60_SPEC}")/training/${node}/training_report.json"
done

if [ -e "${PROJECT_DIR}" ]; then
  test "$(git -C "${PROJECT_DIR}" rev-parse HEAD)" = "${DIAG_COMMIT}"
  test -z "$(git -C "${PROJECT_DIR}" status --porcelain)"
else
  git worktree add --detach "${PROJECT_DIR}" "${DIAG_COMMIT}"
fi

JOB_ID="$(
  sbatch --parsable \
    --account=reu-aisocial \
    --partition=tigris \
    --cpus-per-task=8 \
    --mem=96G \
    --time=06:00:00 \
    --gres=gpu:gh200:1 \
    --job-name=hcwtri60_d000x4 \
    --chdir="${PROJECT_DIR}" \
    --output="${MAIN_REPO}/slurm-%j.out" \
    --export=ALL,PROJECT_DIR="${PROJECT_DIR}",HCWDL_TRI60_SPEC="${TRI60_SPEC}",HCWDL_TRI60_D000_ENSEMBLE_OUTPUT="${DIAG_OUTPUT}",HCWDL_TRI60_D000_ENSEMBLE_COMMIT="${DIAG_COMMIT}" \
    "${PROJECT_DIR}/sbatch/run_hcwdl_mhpe_tri60_d000_ensemble.sh"
)"

echo "D000 x4 diagnostic job: ${JOB_ID}"
squeue -j "${JOB_ID}" -o "%.18i %.35j %.2t %.10M %R"
```

## Result check

```bash
python -m json.tool "${DIAG_OUTPUT}" >/dev/null
python -s - "${DIAG_OUTPUT}" <<'PY'
import json
import math
import sys

report = json.load(open(sys.argv[1]))
row = report["primary_ensemble"]
metrics = row["metrics"]
log_r50 = metrics["macro_mean_log_qcd_rejection_at_50pct_signal"]
print("contract:", report["contract"])
print("components:", ", ".join(row["component_order"]))
print("accuracy:", f"{metrics['accuracy']:.6f}")
print("AUC:", f"{metrics['macro_ovr_auc']:.6f}")
print("R50:", f"{math.exp(log_r50):.1f}")
print(
    "dAUC vs best component:",
    f"{report['primary_delta']['auc_minus_best_component']:+.6f}",
)
print("final test accessed:", report["final_test_accessed"])
PY
```
