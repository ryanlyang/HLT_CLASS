# TRI60 LOGIT/RSET D000 50/50 validation blend

## Scope

This is one isolated post-hoc validation diagnostic over the already-complete
`LOGIT_D000E` and `RSET_D000E` probability ensembles. It does not train a
model. It authenticates and reads the two durable validation probability
banks, aligns them by canonical 32-byte jet identity, and computes

```text
P(blend) = 1/2 P(LOGIT_D000E) + 1/2 P(RSET_D000E)
```

in lexical order with FP64 accumulation and one FP32 cast. It streams the
authenticated validation population only to recover row labels. `M0CE60` is
the fixed zero-recovery reference and `U000` is the fixed one-recovery
reference. R50 recovery is calculated in linear rejection space.

The job is CPU-only, uses a distinct worktree, job name, output root, and
report contract, and has deliberately lowered Slurm priority. It has no
dependency on, or write path into, the running source campaign. It does not
hold, cancel, release, or submit any ladder job. No final-test access, fit,
checkpoint, target bank, persistent prediction array, or deployable artifact
is possible.

## Source-pinned submission

After pushing the implementation, replace the commit placeholder with the
exact 40-character pushed commit and run this block on Tigris. It submits
exactly one independent diagnostic job.

```bash
set -euo pipefail

source /home/ryreu/miniforge3-aarch64/etc/profile.d/conda.sh
conda activate atlas_kd_tigris
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1

export MAIN_REPO=/home/ryreu/atlas/HLT_Classification
export DIAG_COMMIT=REPLACE_WITH_PUSHED_40_CHARACTER_COMMIT
export DIAG_SHORT="${DIAG_COMMIT:0:8}"
export PROJECT_DIR="/home/ryreu/atlas/HLT_Classification_tri60_lr50_${DIAG_SHORT}"
export TRI60_SPEC="${MAIN_REPO}/checkpoints/hcwdl_mhpe_tri60_full_d218961c_r1/campaign_spec.json"
export DIAG_ROOT="${MAIN_REPO}/checkpoints/hcwdl_mhpe_tri60_lr50_${DIAG_SHORT}_r1"
export DIAG_OUTPUT="${DIAG_ROOT}/validation_report.json"

cd "${MAIN_REPO}"
git fetch origin main
test "$(git rev-parse origin/main)" = "${DIAG_COMMIT}"
test -f "${TRI60_SPEC}"
test ! -e "${DIAG_ROOT}"

if [ -e "${PROJECT_DIR}" ]; then
  test "$(git -C "${PROJECT_DIR}" rev-parse HEAD)" = "${DIAG_COMMIT}"
  test -z "$(git -C "${PROJECT_DIR}" status --porcelain)"
else
  git worktree add --detach "${PROJECT_DIR}" "${DIAG_COMMIT}"
fi

export PYTHONPATH="${PROJECT_DIR}/src"

export CE_CONTROL_SPEC="$(
  python -s - "${TRI60_SPEC}" "${MAIN_REPO}/checkpoints" <<'PY'
import json
import sys
from pathlib import Path

source = json.load(open(sys.argv[1]))
checkpoints = Path(sys.argv[2])
candidates = []
for path in checkpoints.glob("*/control_spec.json"):
    try:
        spec = json.load(open(path))
        report_path = Path(spec["campaign_root"]) / "training/M0CE60/training_report.json"
        report = json.load(open(report_path))
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        continue
    if (
        spec.get("contract") == "HCWDL_MHPE_TRI60_CE60_CONTROL_SPEC/v1"
        and spec.get("parents", {}).get("source_campaign") == source["content_hash"]
        and report.get("contract")
        == "HCWDL_MHPE_TRI60_CE60_CONTROL_TRAINING_REPORT/v1"
        and report.get("complete") is True
        and report.get("final_test_accessed") is False
    ):
        candidates.append(path.resolve())
if len(candidates) != 1:
    raise SystemExit(
        "Expected exactly one completed source-matched M0CE60 control; found "
        f"{len(candidates)}:\n" + "\n".join(map(str, candidates))
    )
print(candidates[0])
PY
)"

for distribution in LOGIT_D000E RSET_D000E U000; do
  test -f "$(dirname "${TRI60_SPEC}")/probabilities/${distribution}/lock.json"
  test -f "$(dirname "${TRI60_SPEC}")/probabilities/${distribution}/validation_manifest.json"
  test -f "$(dirname "${TRI60_SPEC}")/reports/stages/${distribution}.json"
done
test -f "${CE_CONTROL_SPEC}"

mkdir -p "${DIAG_ROOT}"

JOB_ID="$(
  sbatch --parsable \
    --account=reu-aisocial \
    --partition=tigris \
    --cpus-per-task=8 \
    --mem=64G \
    --time=02:00:00 \
    --nice=10000 \
    --job-name=hcwtri60_lr50 \
    --chdir="${PROJECT_DIR}" \
    --output="${DIAG_ROOT}/slurm-%j.out" \
    --export=ALL,PROJECT_DIR="${PROJECT_DIR}",HCWDL_TRI60_SPEC="${TRI60_SPEC}",HCWDL_TRI60_CE_CONTROL_SPEC="${CE_CONTROL_SPEC}",HCWDL_TRI60_LOGIT_RSET_OUTPUT="${DIAG_OUTPUT}",HCWDL_TRI60_LOGIT_RSET_COMMIT="${DIAG_COMMIT}" \
    "${PROJECT_DIR}/sbatch/run_hcwdl_mhpe_tri60_d000_logit_rset_blend.sh"
)"

echo "Independent LOGIT/RSET 50/50 diagnostic job: ${JOB_ID}"
squeue -j "${JOB_ID}" -o "%.18i %.35j %.2t %.10M %R"
```

The submission contains no `--dependency` and requests no GPU. The positive
nice value ensures this exploratory diagnostic cannot outrank ordinary jobs
from the same account, although—as with any Slurm job—it still consumes its
declared CPU and memory while running.

## Result table

After the job completes:

```bash
test -f "${DIAG_OUTPUT}"
python -m json.tool "${DIAG_OUTPUT}" >/dev/null

python -s - "${DIAG_OUTPUT}" <<'PY'
import json
import math
import sys

report = json.load(open(sys.argv[1]))
rows = report["reference_rows"] + report["component_rows"] + [{
    "row_id": report["primary_ensemble"]["ensemble_id"],
    "kind": "50/50 blend",
    "metrics": report["primary_ensemble"]["metrics"],
    "recovery_m0ce60_to_u000": report["primary_ensemble"][
        "recovery_m0ce60_to_u000"
    ],
}]

def r50(metrics):
    return math.exp(metrics["macro_mean_log_qcd_rejection_at_50pct_signal"])

def pct(value):
    return "n/a" if value is None else f"{100 * value:+.1f}%"

print("Report:", sys.argv[1])
print("Final test accessed:", report["final_test_accessed"])
print()
print(
    f"{'model':<31} {'kind':<20} {'accuracy':>10} {'AUC':>10} "
    f"{'R50':>10} {'AUC rec.':>10} {'R50 rec.':>10}"
)
for row in rows:
    metrics = row["metrics"]
    recovery = row["recovery_m0ce60_to_u000"]
    print(
        f"{row['row_id']:<31} {row['kind']:<20} "
        f"{metrics['accuracy']:>10.6f} {metrics['macro_ovr_auc']:>10.6f} "
        f"{r50(metrics):>10.1f} {pct(recovery['macro_ovr_auc']):>10} "
        f"{pct(recovery['macro_r50_linear']):>10}"
    )

print()
print("Selected per-class QCD rejection at 50% signal efficiency")
print(
    f"{'model':<31} {'Hbb R50':>10} {'Hbb rec.':>10} "
    f"{'Hcc R50':>10} {'Hcc rec.':>10} {'Hqq R50':>10} {'Hqq rec.':>10}"
)
for row in rows:
    metrics = row["metrics"]
    recovery = row["recovery_m0ce60_to_u000"]["per_class_r50_linear"]
    values = []
    for name in ("Xbb", "Xcc", "Xqq"):
        values.append(metrics["per_class"][name]["qcd_rejection"]["50pct"]["rejection"])
        values.append(pct(recovery[name]))
    print(
        f"{row['row_id']:<31} {values[0]:>10.1f} {values[1]:>10} "
        f"{values[2]:>10.1f} {values[3]:>10} {values[4]:>10.1f} {values[5]:>10}"
    )
PY
```
