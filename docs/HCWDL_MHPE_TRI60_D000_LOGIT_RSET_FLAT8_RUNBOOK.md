# TRI60 flat-eight LOGIT/RSET D000 validation ensemble

## Scope

This is one isolated, post-hoc validation diagnostic over the completed
`LOGIT_D000E` and `RSET_D000E` probability banks. The former is a uniform
ensemble of five specialists and the latter is a uniform ensemble of three.
The diagnostic computes

```text
P(flat8) = 5/8 P(LOGIT_D000E) + 3/8 P(RSET_D000E)
```

so every one of the eight underlying specialists has nominal effective weight
`1/8`. It also recomputes the equal-family `1/2 + 1/2` result in the same
authenticated evaluation as a direct comparator.

The calculation uses the durable FP32 family banks, FP64 cross-family
accumulation, and one FP32 publication cast. It is mathematically the flat
eight-member average before each family bank's earlier FP32 publication; it
can differ by negligible roundoff from fresh inference and direct accumulation
of all eight raw model outputs. The report records that boundary explicitly.

This job is validation-only and CPU-only. It has a separate source-pinned
worktree, output root, report contract, and low-priority job name. It creates
no fit, checkpoint, target, deployable model, persistent prediction bank,
scheduler dependency, source-campaign write, or final-test access.

## Source-pinned submission

After pushing the implementation, replace the commit placeholder with the
exact pushed 40-character commit and run this block on Tigris. It submits
exactly one independent diagnostic job and does not modify any running ladder.

```bash
set -euo pipefail

source /home/ryreu/miniforge3-aarch64/etc/profile.d/conda.sh
conda activate atlas_kd_tigris
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1

export MAIN_REPO=/home/ryreu/atlas/HLT_Classification
export FLAT8_COMMIT=REPLACE_WITH_PUSHED_40_CHARACTER_COMMIT
export FLAT8_SHORT="${FLAT8_COMMIT:0:8}"
export PROJECT_DIR="/home/ryreu/atlas/HLT_Classification_tri60_flat8_${FLAT8_SHORT}"
export TRI60_SPEC="${MAIN_REPO}/checkpoints/hcwdl_mhpe_tri60_full_d218961c_r1/campaign_spec.json"
export FLAT8_ROOT="${MAIN_REPO}/checkpoints/hcwdl_mhpe_tri60_flat8_${FLAT8_SHORT}_r1"
export FLAT8_OUTPUT="${FLAT8_ROOT}/validation_report.json"

cd "${MAIN_REPO}"
git fetch origin main
test "$(git rev-parse origin/main)" = "${FLAT8_COMMIT}"
test -f "${TRI60_SPEC}"
test ! -e "${FLAT8_ROOT}"

if [ -e "${PROJECT_DIR}" ]; then
  test "$(git -C "${PROJECT_DIR}" rev-parse HEAD)" = "${FLAT8_COMMIT}"
  test -z "$(git -C "${PROJECT_DIR}" status --porcelain)"
else
  git worktree add --detach "${PROJECT_DIR}" "${FLAT8_COMMIT}"
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

mkdir -p "${FLAT8_ROOT}"

JOB_ID="$(
  sbatch --parsable \
    --account=reu-aisocial \
    --partition=tigris \
    --cpus-per-task=8 \
    --mem=64G \
    --time=02:00:00 \
    --nice=10000 \
    --job-name=hcwtri60_flat8 \
    --chdir="${PROJECT_DIR}" \
    --output="${FLAT8_ROOT}/slurm-%j.out" \
    --export=ALL,PROJECT_DIR="${PROJECT_DIR}",HCWDL_TRI60_SPEC="${TRI60_SPEC}",HCWDL_TRI60_CE_CONTROL_SPEC="${CE_CONTROL_SPEC}",HCWDL_TRI60_FLAT8_OUTPUT="${FLAT8_OUTPUT}",HCWDL_TRI60_FLAT8_COMMIT="${FLAT8_COMMIT}" \
    "${PROJECT_DIR}/sbatch/run_hcwdl_mhpe_tri60_d000_logit_rset_flat8.sh"
)"

echo "Independent flat-eight diagnostic job: ${JOB_ID}"
squeue -j "${JOB_ID}" -o "%.18i %.35j %.2t %.10M %R"
```

## Result table

After the job completes, use the variables from the submission shell or set
`FLAT8_OUTPUT` again, then run:

```bash
test -f "${FLAT8_OUTPUT}"
python -m json.tool "${FLAT8_OUTPUT}" >/dev/null

python -s - "${FLAT8_OUTPUT}" <<'PY'
import json
import math
import sys

report = json.load(open(sys.argv[1]))
equal = report["equal_family_comparator"]
rows = equal["reference_rows"] + equal["component_rows"] + [
    {
        "row_id": equal["primary_ensemble"]["ensemble_id"],
        "kind": "equal-family 50/50",
        "metrics": equal["primary_ensemble"]["metrics"],
        "recovery_m0ce60_to_u000": equal["primary_ensemble"][
            "recovery_m0ce60_to_u000"
        ],
    },
    {
        "row_id": report["primary_ensemble"]["ensemble_id"],
        "kind": "flat eight-member",
        "metrics": report["primary_ensemble"]["metrics"],
        "recovery_m0ce60_to_u000": report["primary_ensemble"][
            "recovery_m0ce60_to_u000"
        ],
    },
]

def r50(metrics):
    return math.exp(metrics["macro_mean_log_qcd_rejection_at_50pct_signal"])

def pct(value):
    return "n/a" if value is None else f"{100 * value:+.1f}%"

print("Report:", sys.argv[1])
print("Contract:", report["contract"])
print("Final test accessed:", report["final_test_accessed"])
print("Members:")
for family, members in report["family_member_registry"].items():
    print(f"  {family} ({len(members)}): " + ", ".join(members))
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

print()
print("FLAT8 MINUS COMPARATORS")
for name, delta in report["primary_delta"].items():
    print(
        f"{name:<24} dAUC={delta['macro_ovr_auc']:+.6f}  "
        f"dR50={delta['macro_r50_linear']:+.1f}"
    )
PY
```
