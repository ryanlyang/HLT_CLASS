# Evaluate the completed Luka campaign on its sealed test split

This queues **two jobs**, not more training: all twelve frozen checkpoints on
149553 test jets, then the comparison report. Use SPORC debug and preserve every
old worktree/artifact. The registered [plan](plans/LUKA_FULLSIM_FINAL_TEST_PLAN.md)
and [contracts](contracts/LUKA_FULLSIM_FINAL_TEST.md) govern access.

## 1. Windows: explicit commit and push

Run in the repository. Do not stage unrelated work with `git add .`.

```powershell
$Files = @(
  'src/hlt_classification/luka_fullsim/final_test_campaign.py',
  'src/hlt_classification/luka_fullsim/final_test_data.py',
  'src/hlt_classification/luka_fullsim/final_test_worker.py',
  'src/hlt_classification/luka_fullsim/final_test_submission.py',
  'scripts/luka_fullsim_final_test.py',
  'scripts/queue_luka_fullsim_final_test.sh',
  'sbatch/run_luka_fullsim_final_test.sh',
  'tests/test_luka_fullsim_final_test.py',
  'docs/plans/LUKA_FULLSIM_FINAL_TEST_PLAN.md',
  'docs/contracts/LUKA_FULLSIM_FINAL_TEST.md',
  'docs/LUKA_FULLSIM_FINAL_TEST_RUNBOOK.md',
  'docs/LEGACY_SOURCE_MAP.md'
)
git add -- $Files
if ($LASTEXITCODE -ne 0) { throw 'Staging failed' }
git diff --cached --stat
git diff --cached --check
if ($LASTEXITCODE -ne 0) { throw 'Inspect staged changes' }
git commit -m "Add locked all-model Luka final-test evaluation"
if ($LASTEXITCODE -ne 0) { throw 'Commit failed' }
git push origin HEAD:main
if ($LASTEXITCODE -ne 0) { throw 'Push failed' }
git rev-parse HEAD
```

HANDOFF has pre-existing unrelated edits. Stage only the new test-stage section
with `git add -p -- docs/HANDOFF.md` before committing if desired; preserve the rest.

## 2. sporcsubmit: create locks and dry-review

Copy only the Bash contents, not Markdown fences. Keep the full new commit.
No test particles are opened or jobs submitted by this step.

```bash
(
set -euo pipefail
trap 'echo "Stopped at line ${LINENO}; preserve everything and paste the error." >&2' ERR
read -r -p 'New final-test commit: ' COMMIT </dev/tty
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
MAIN=/home/ryreu/atlas/HLT_Classification
PROJECT="/home/ryreu/atlas/HLT_Classification_luka_test_${COMMIT:0:8}"
ROOT="${MAIN}/checkpoints/luka_test_${COMMIT:0:8}_r1"
git -C "${MAIN}" fetch origin main
git -C "${MAIN}" merge-base --is-ancestor "${COMMIT}" origin/main
if [[ -e "${PROJECT}" ]]; then
    test "$(git -C "${PROJECT}" rev-parse HEAD)" = "${COMMIT}"
    test -z "$(git -C "${PROJECT}" status --porcelain --untracked-files=all)"
else
    git -C "${MAIN}" worktree add --detach "${PROJECT}" "${COMMIT}"
fi
LOG="$(mktemp "${MAIN}/checkpoints/luka-test-review.${COMMIT:0:8}.XXXXXX.log")"
nohup bash "${PROJECT}/scripts/queue_luka_fullsim_final_test.sh" "${COMMIT}" "${ROOT}" \
    >"${LOG}" 2>&1 </dev/null &
printf 'Review PID: %s\nStudy: %s\nLog: %s\n' "$!" "${ROOT}" "${LOG}"
printf 'Watch with:\ntail -f "%s"\n' "${LOG}"
)
```

Wait for `Dry review complete: 2 jobs`. Parent authentication checks all old
checkpoint/report/bank bytes and can take time. It uses the original pinned
science worktree, not a modified copy. Confirm:

- `luka_test_evaluate`: 6 CPUs, 256000 MiB, one A100, debug, 23:06:00 limit;
- `luka_test_report`: 1 CPU, 32768 MiB, debug, 04:00:00, after evaluation;
- all twelve checkpoints, the same split hash, 149553 rows and 337 files.

The GPU limit reuses conservative measured training headroom; it is not an
evaluation ETA. Test preprocessing is done once for all models. No extra fit.

## 3. Execute the exact reviewed plan

```bash
(
set -euo pipefail
read -r -p 'Same final-test commit: ' COMMIT </dev/tty
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
MAIN=/home/ryreu/atlas/HLT_Classification
PROJECT="/home/ryreu/atlas/HLT_Classification_luka_test_${COMMIT:0:8}"
ROOT="${MAIN}/checkpoints/luka_test_${COMMIT:0:8}_r1"
test -f "${ROOT}/submission/dry_run_submission_ledger.json"
PLAN_HASH="$(/home/ryreu/miniconda3/envs/atlas_kd_sporc/bin/python -s -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["content_hash"])' \
    "${ROOT}/submission/command_plan.json")"
[[ "${PLAN_HASH}" =~ ^[0-9a-f]{64}$ ]]
printf 'Reviewed plan hash: %s\n' "${PLAN_HASH}"
read -r -p 'Evaluate all 12 frozen models and consume the test holdout? Type YES: ' CONFIRM </dev/tty
[[ "${CONFIRM}" == YES ]] || exit 0
LOG="$(mktemp "${MAIN}/checkpoints/luka-test-submit.${COMMIT:0:8}.XXXXXX.log")"
nohup bash "${PROJECT}/scripts/queue_luka_fullsim_final_test.sh" "${COMMIT}" "${ROOT}" \
    --execute "${PLAN_HASH}" >"${LOG}" 2>&1 </dev/null &
printf 'Submission PID: %s\nLog: %s\n' "$!" "${LOG}"
printf 'Watch with:\ntail -f "%s"\n' "${LOG}"
)
```

Confirm `QUEUE_HELPER_EXIT=0` and the two numerical IDs. The helper survives SSH
disconnect; Ctrl-C stops tail only. Never launch duplicate helpers, delete a
claim, overwrite a partial evaluation or invent a new test split to retry. An
interrupted test execution requires explicit recovery inspection; preserve all
evidence. Both locks and the training-only smoke must pass before any test read.

## 4. Print the completed test table

Enter the same commit again; each earlier block uses a private subshell, so
its variables do not remain set. Results reading authenticates artifacts; it does
not run another test inference. Save output if your SSH connection is unstable.

```bash
(
set -euo pipefail
read -r -p 'Same final-test commit: ' COMMIT </dev/tty
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
PROJECT="/home/ryreu/atlas/HLT_Classification_luka_test_${COMMIT:0:8}"
ROOT="/home/ryreu/atlas/HLT_Classification/checkpoints/luka_test_${COMMIT:0:8}_r1"
source /home/ryreu/miniconda3/etc/profile.d/conda.sh
conda activate /home/ryreu/miniconda3/envs/atlas_kd_sporc
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${PROJECT}/src"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -u -s "${PROJECT}/scripts/luka_fullsim_final_test.py" results \
    --lock "${ROOT}/execution_lock.json"
)
```

Full report: `ROOT/report/results.json`. The final table retains oracle/capacity
labels. Rejection@50 uses the ROC definition, not a validation-fixed threshold.
Paired intervals are fixed-fit/source-file uncertainty, not multi-seed evidence.
