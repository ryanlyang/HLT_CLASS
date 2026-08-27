# Original TRI60 LOGIT D000E Hbb/Hcc rejection curves

## Scope

This is an isolated validation-only diagnostic for the original, non-DX
`LOGIT_D000E` probability ensemble. It compares three fixed models on the
same authenticated all-mapped validation population:

- exact-HLT CE-only `M0CE60`;
- original exact-HLT `LOGIT_D000E`; and
- projected-offline-input `U000`.

For Hbb and Hcc separately, the score is
`p_signal / (p_signal + p_QCD)`. The evaluator computes the exact
tied-threshold signal-efficiency versus QCD background-rejection scan. The
figure and compact NPZ use a deterministic 4,096-point signal-efficiency-grid
projection of its upper envelope, while the registered 30%, 50%, and 80%
working points come from the unthinned scan. Zero observed QCD passes use the
finite empirical ceiling `N_QCD`, not infinity. Markers identify the 50%
signal-efficiency working point.

The durable `LOGIT_D000E` and `U000` validation probability banks are consumed
read-only. Because the CE control has no durable prediction bank, the selected
`M0CE60` checkpoint receives one validation inference pass. Those probabilities
remain process-local. The only durable numerical payload is the compact curve
NPZ plus a small lineage report; no prediction bank, particle view, fit,
checkpoint, target, deployable model, campaign dependency, or final-test
capability is created.

## Queue independently on Tigris

After pushing the implementation, replace the commit placeholder and run:

```bash
set -euo pipefail

source /home/ryreu/miniforge3-aarch64/etc/profile.d/conda.sh
conda activate atlas_kd_tigris
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1

export MAIN_REPO=/home/ryreu/atlas/HLT_Classification
export CHECKPOINTS="${MAIN_REPO}/checkpoints"
export ROC_COMMIT=REPLACE_WITH_PUSHED_40_CHARACTER_COMMIT
export ROC_SHORT="${ROC_COMMIT:0:8}"
export PROJECT_DIR="/home/ryreu/atlas/HLT_Classification_tri60_original_logit_roc_${ROC_SHORT}"
export TRI60_SPEC="${CHECKPOINTS}/hcwdl_mhpe_tri60_full_d218961c_r1/campaign_spec.json"
export ROC_ROOT="${CHECKPOINTS}/hcwdl_mhpe_tri60_original_logit_roc_${ROC_SHORT}_r1"

cd "${MAIN_REPO}"
git fetch origin main
test "$(git rev-parse origin/main)" = "${ROC_COMMIT}"
test -f "${TRI60_SPEC}"
test ! -e "${ROC_ROOT}"

if [ -e "${PROJECT_DIR}" ]; then
  test "$(git -C "${PROJECT_DIR}" rev-parse HEAD)" = "${ROC_COMMIT}"
  test -z "$(git -C "${PROJECT_DIR}" status --porcelain)"
else
  git worktree add --detach "${PROJECT_DIR}" "${ROC_COMMIT}"
fi

export PYTHONPATH="${PROJECT_DIR}/src"

export CE_CONTROL_SPEC="$(
  python -s - "${TRI60_SPEC}" "${CHECKPOINTS}" <<'PY'
import json
import sys
from pathlib import Path

source = json.load(open(sys.argv[1]))
checkpoints = Path(sys.argv[2])
candidates = []
for path in checkpoints.glob("*/control_spec.json"):
    try:
        spec = json.load(open(path))
        report_path = (
            Path(spec["campaign_root"])
            / "training/M0CE60/training_report.json"
        )
        report = json.load(open(report_path))
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        continue
    if (
        spec.get("contract")
        == "HCWDL_MHPE_TRI60_CE60_CONTROL_SPEC/v1"
        and spec.get("parents", {}).get("source_campaign")
        == source["content_hash"]
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

for distribution in LOGIT_D000E U000; do
  test -f "$(dirname "${TRI60_SPEC}")/probabilities/${distribution}/lock.json"
  test -f "$(dirname "${TRI60_SPEC}")/probabilities/${distribution}/validation_manifest.json"
  test -f "$(dirname "${TRI60_SPEC}")/reports/stages/${distribution}.json"
done
test -f "${CE_CONTROL_SPEC}"

mkdir -p "${ROC_ROOT}"

ROC_JOB="$(
  sbatch --parsable \
    --account=reu-aisocial \
    --partition=tigris \
    --gres=gpu:gh200:1 \
    --cpus-per-task=16 \
    --mem=96G \
    --time=06:00:00 \
    --nice=10000 \
    --job-name=hcwtri60_logitroc \
    --chdir="${PROJECT_DIR}" \
    --output="${ROC_ROOT}/slurm-%j.out" \
    --export=ALL,PROJECT_DIR="${PROJECT_DIR}",HCWDL_TRI60_SPEC="${TRI60_SPEC}",HCWDL_TRI60_CE_CONTROL_SPEC="${CE_CONTROL_SPEC}",HCWDL_TRI60_ORIGINAL_LOGIT_ROC_OUTPUT_DIR="${ROC_ROOT}",HCWDL_TRI60_ORIGINAL_LOGIT_ROC_COMMIT="${ROC_COMMIT}" \
    "${PROJECT_DIR}/sbatch/run_hcwdl_mhpe_tri60_original_logit_d000e_roc.sh"
)"

echo "Independent original LOGIT D000E ROC job: ${ROC_JOB}"
squeue -j "${ROC_JOB}" -o "%.18i %.35j %.2t %.10M %R"
```

There is no `--dependency`: this job neither waits for nor modifies any
original or DX ladder job. The positive nice value keeps this exploratory
plot below ordinary same-account work.

## Inspect and copy the result

```bash
export REPORT="${ROC_ROOT}/original_logit_d000e_hbb_hcc_report.json"
python -m json.tool "${REPORT}" >/dev/null

python -s - "${REPORT}" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1]))
print("PDF:", report["figures"]["pdf"]["path"])
print("PNG:", report["figures"]["png"]["path"])
print("Curves:", report["curves_path"])
print("Final test accessed:", report["final_test_accessed"])
print()
print(f"{'signal':<8} {'eff.':>6} {'M0CE60':>12} {'LOGIT_D000E':>14} {'U000':>12} {'recovery':>11}")
for signal in report["signals"]:
    for key, rows in report["working_points"][signal].items():
        recovery = rows["LOGIT_D000E"][
            "linear_rejection_recovery_m0ce60_to_u000"
        ]
        recovery_text = "n/a" if recovery is None else f"{100*recovery:+.1f}%"
        print(
            f"{signal:<8} {key:>6} "
            f"{rows['M0CE60']['qcd_rejection']:>12.1f} "
            f"{rows['LOGIT_D000E']['qcd_rejection']:>14.1f} "
            f"{rows['U000']['qcd_rejection']:>12.1f} "
            f"{recovery_text:>11}"
        )
PY
```

From a local terminal, copy the figures with `scp` using the printed paths.
