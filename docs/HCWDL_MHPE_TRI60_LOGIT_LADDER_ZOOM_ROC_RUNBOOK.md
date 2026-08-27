# TRI60 LOGIT ladder zoomed Hbb/Hcc rejection curves

## Exact display mapping

This validation-only diagnostic intentionally uses presentation aliases while
retaining the exact artifact identity in its content-hashed report:

| displayed label | exact source artifact |
|---|---|
| `Offline` | `U000` |
| `D100` | `LOGIT_U100E` |
| `D080` | `DX_LOGIT_D083E` |
| `D060` | `LOGIT_U100_from_U050E` |
| `D040` | `LOGIT_D033E` |
| `D020` | `DX_LOGIT_D083_from_LOGIT_U100E` |
| `D000` | `LOGIT_D000E` |

Every legend places `Offline` first as the external reference, followed by
`D100`, `D080`, `D060`, `D040`, `D020`, and `D000` in exactly that order.
The signal-efficiency axis is restricted to `[0.30, 0.50]`. Separate Hbb and
Hcc figures and a combined two-panel figure are written in both PDF and PNG.

The score is `p_signal/(p_signal+p_QCD)`, the rejection is
`N_QCD/max(1,N_QCD_passing)`, and zero observed QCD passes use the finite
empirical ceiling `N_QCD`. The five ensemble/reference curves consume their
authenticated durable validation probability banks. The two specialist
curves receive one selected-checkpoint validation inference pass on their
native authenticated rung views; those predictions remain process-local.

This creates no fit, checkpoint, prediction bank, scheduler dependency,
deployable model, campaign mutation, or final-test capability. Durable output
is limited to six figures, one compact curve NPZ, the Slurm log, and a small
lineage report.

## Queue independently on Tigris

The source and DX artifacts listed above must already be complete. After
pushing the implementation, replace the commit placeholder and run:

```bash
set -euo pipefail

source /home/ryreu/miniforge3-aarch64/etc/profile.d/conda.sh
conda activate atlas_kd_tigris
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1

export MAIN_REPO=/home/ryreu/atlas/HLT_Classification
export CHECKPOINTS="${MAIN_REPO}/checkpoints"
export ZOOM_COMMIT=REPLACE_WITH_PUSHED_40_CHARACTER_COMMIT
export ZOOM_SHORT="${ZOOM_COMMIT:0:8}"
export PROJECT_DIR="/home/ryreu/atlas/HLT_Classification_tri60_logit_zoom_${ZOOM_SHORT}"
export SOURCE_SPEC="${CHECKPOINTS}/hcwdl_mhpe_tri60_full_d218961c_r1/campaign_spec.json"
export DENSE_SPEC="${CHECKPOINTS}/hcwdl_mhpe_tri60_dense_e2e107ee_r1/campaign_spec.json"
export ZOOM_ROOT="${CHECKPOINTS}/hcwdl_mhpe_tri60_logit_zoom_${ZOOM_SHORT}_r1"

cd "${MAIN_REPO}"
git fetch origin main
test "$(git rev-parse origin/main)" = "${ZOOM_COMMIT}"
test -f "${SOURCE_SPEC}"
test -f "${DENSE_SPEC}"
test ! -e "${ZOOM_ROOT}"

if [ -e "${PROJECT_DIR}" ]; then
  test "$(git -C "${PROJECT_DIR}" rev-parse HEAD)" = "${ZOOM_COMMIT}"
  test -z "$(git -C "${PROJECT_DIR}" status --porcelain)"
else
  git worktree add --detach "${PROJECT_DIR}" "${ZOOM_COMMIT}"
fi

SOURCE_ROOT="$(dirname "${SOURCE_SPEC}")"
DENSE_ROOT="$(dirname "${DENSE_SPEC}")"

for distribution in U000 LOGIT_U100E LOGIT_D033E LOGIT_D000E; do
  test -f "${SOURCE_ROOT}/probabilities/${distribution}/lock.json"
  test -f "${SOURCE_ROOT}/probabilities/${distribution}/validation_manifest.json"
  test -f "${SOURCE_ROOT}/reports/stages/${distribution}.json"
done
test -f "${SOURCE_ROOT}/training/LOGIT_U100_from_U050E/training_report.json"

for distribution in DX_LOGIT_D083E; do
  test -f "${DENSE_ROOT}/probabilities/${distribution}/lock.json"
  test -f "${DENSE_ROOT}/probabilities/${distribution}/validation_manifest.json"
  test -f "${DENSE_ROOT}/reports/stages/${distribution}.json"
done
test -f "${DENSE_ROOT}/training/DX_LOGIT_D083_from_LOGIT_U100E/training_report.json"

mkdir -p "${ZOOM_ROOT}"

ZOOM_JOB="$(
  sbatch --parsable \
    --account=reu-aisocial \
    --partition=tigris \
    --gres=gpu:gh200:1 \
    --cpus-per-task=72 \
    --mem=192G \
    --time=12:00:00 \
    --nice=10000 \
    --job-name=hcwtri60_logitzoom \
    --chdir="${PROJECT_DIR}" \
    --output="${ZOOM_ROOT}/slurm-%j.out" \
    --export=ALL,PROJECT_DIR="${PROJECT_DIR}",HCWDL_TRI60_SOURCE_SPEC="${SOURCE_SPEC}",HCWDL_TRI60_DENSE_SPEC="${DENSE_SPEC}",HCWDL_TRI60_LOGIT_ZOOM_OUTPUT_DIR="${ZOOM_ROOT}",HCWDL_TRI60_LOGIT_ZOOM_COMMIT="${ZOOM_COMMIT}" \
    "${PROJECT_DIR}/sbatch/run_hcwdl_mhpe_tri60_logit_ladder_zoom_roc.sh"
)"

echo "Independent LOGIT zoom plot job: ${ZOOM_JOB}"
squeue -j "${ZOOM_JOB}" -o "%.18i %.35j %.2t %.10M %R"
```

There is deliberately no `--dependency`. The positive nice value places this
presentation diagnostic below ordinary same-account campaign jobs.

## Inspect the outputs

```bash
export REPORT="${ZOOM_ROOT}/logit_ladder_zoom_hbb_hcc_report.json"
python -m json.tool "${REPORT}" >/dev/null

python -s - "${REPORT}" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1]))
print("Display order:", " -> ".join(report["display_order"]))
print("Signal-efficiency range:", report["signal_efficiency_range"])
print("Final test accessed:", report["final_test_accessed"])
print()
for name, row in report["figures"].items():
    print(f"{name:<14} {row['path']}")
PY
```

The most convenient files for sharing are:

- `logit_ladder_zoom_hbb_rejection.png`
- `logit_ladder_zoom_hcc_rejection.png`
- `logit_ladder_zoom_hbb_hcc_rejection.png`
