# Queue the Luka 100k/50k fusion science campaign on SPORC

Authority: [plan](plans/LUKA_FULLSIM_SCIENCE_PLAN.md) and
[contracts](contracts/LUKA_FULLSIM_SCIENCE.md). This is **21 science jobs**, not
another foundation, particle audit, preparation, or technical preflight.
The worker reuses accepted job **21835630** and its unchanged kernels.
No existing jobs/artifacts are cancelled or overwritten. Final test stays sealed.

## 1. Commit and push locally (PowerShell)

Review these explicit paths. Do not `git add .`: this checkout has unrelated
documentation edits and large untracked historical test directories.

```powershell
$Files = @(
  'src/hlt_classification/luka_fullsim/science_admission.py',
  'src/hlt_classification/luka_fullsim/science_campaign.py',
  'src/hlt_classification/luka_fullsim/science_cache.py',
  'src/hlt_classification/luka_fullsim/science_worker.py',
  'src/hlt_classification/luka_fullsim/science_submission.py',
  'scripts/luka_fullsim_science.py',
  'scripts/queue_luka_fullsim_science.sh',
  'sbatch/run_luka_fullsim_science.sh',
  'tests/test_luka_fullsim_science.py',
  'docs/plans/LUKA_FULLSIM_SCIENCE_PLAN.md',
  'docs/contracts/LUKA_FULLSIM_SCIENCE.md',
  'docs/LUKA_FULLSIM_SCIENCE_RUNBOOK.md',
  'docs/LEGACY_SOURCE_MAP.md'
)
git add -- $Files
git diff --cached --stat
git diff --cached --check
```

Review `git diff -- docs/HANDOFF.md` separately; it already contains unrelated
work. Stage only the new stage-3 section with `git add -p -- docs/HANDOFF.md` if
desired. Do not discard the other edits. Check staged files before committing:

```powershell
git diff --cached --name-only
git commit -m "Add source-bound Luka FullSim fusion science campaign"
if ($LASTEXITCODE -ne 0) { throw 'Commit failed' }
git push origin HEAD:main
if ($LASTEXITCODE -ne 0) { throw 'Push failed' }
git rev-parse HEAD
```

## 2. Prepare and dry-review on sporcsubmit

Copy only the contents of the Bash block, not the Markdown fences. Enter the
new 40-character commit when prompted; do not reuse the preflight commit.

```bash
(
set -euo pipefail
trap 'echo "Stopped at line ${LINENO}; preserve files and paste the error." >&2' ERR
read -r -p 'New science commit: ' COMMIT </dev/tty
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
MAIN=/home/ryreu/atlas/HLT_Classification
PROJECT="/home/ryreu/atlas/HLT_Classification_luka_science_${COMMIT:0:8}"
ROOT="${MAIN}/checkpoints/luka_science_${COMMIT:0:8}_r1"
git -C "${MAIN}" fetch origin main
git -C "${MAIN}" merge-base --is-ancestor "${COMMIT}" origin/main
if [[ -e "${PROJECT}" ]]; then
    test "$(git -C "${PROJECT}" rev-parse HEAD)" = "${COMMIT}"
    test -z "$(git -C "${PROJECT}" status --porcelain --untracked-files=all)"
else
    git -C "${MAIN}" worktree add --detach "${PROJECT}" "${COMMIT}"
fi
LOG="$(mktemp "${MAIN}/checkpoints/luka-science-review.${COMMIT:0:8}.XXXXXX.log")"
nohup bash "${PROJECT}/scripts/queue_luka_fullsim_science.sh" "${COMMIT}" "${ROOT}" \
    >"${LOG}" 2>&1 </dev/null &
printf 'Review PID: %s\nStudy: %s\nLog: %s\n' "$!" "${ROOT}" "${LOG}"
printf 'Watch with:\ntail -f "%s"\n' "${LOG}"
)
```

Wait for `Dry review complete. No jobs submitted.` Source/payload authentication
can take time. The saved plan is `ROOT/submission/command_plan.json`.
Review **12 train, 7 reduce, 2 CPU report jobs**, all on debug. GPU jobs use
6 CPUs, 256000 MiB, one A100. Measured limits are 23:06:00 for training and
02:11:00 for reducers, not expected completion times. The long training request
can affect queue waiting. Do not reduce the measured memory or time silently.

The helper's historical paths are explicit:

```text
prepared: /home/ryreu/atlas/HLT_Classification/checkpoints/luka_fullsim_1256c756_r2/prepared_mm_r1
preparation source: /home/ryreu/atlas/HLT_Classification_luka_1256c756
preflight: /home/ryreu/atlas/HLT_Classification/checkpoints/luka_preflight_v2_bcccd498_r1/preflight.json
preflight source: /home/ryreu/atlas/HLT_Classification_luka_preflight_v2_bcccd498
```

Preserve these old clean worktrees and data. No copied/edit-in-place source lock
or replacement path based only on existence is allowed. Missing evidence,
accounting, source drift, or changed environment stops admission.

## 3. Submit the full reviewed DAG once

After reviewing the dry plan, run this on sporcsubmit. It reads and prints the
saved plan hash to avoid a placeholder-copy error, then asks for explicit YES.

```bash
(
set -euo pipefail
read -r -p 'Same new science commit: ' COMMIT </dev/tty
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
MAIN=/home/ryreu/atlas/HLT_Classification
PROJECT="/home/ryreu/atlas/HLT_Classification_luka_science_${COMMIT:0:8}"
ROOT="${MAIN}/checkpoints/luka_science_${COMMIT:0:8}_r1"
PLAN="${ROOT}/submission/command_plan.json"
test -f "${ROOT}/submission/dry_run_submission_ledger.json"
PLAN_HASH="$(/home/ryreu/miniconda3/envs/atlas_kd_sporc/bin/python -s -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["content_hash"])' "${PLAN}")"
[[ "${PLAN_HASH}" =~ ^[0-9a-f]{64}$ ]]
printf 'Reviewed plan: %s\n' "${PLAN_HASH}"
read -r -p 'Submit all 21 reviewed Luka science jobs? Type YES: ' CONFIRM </dev/tty
[[ "${CONFIRM}" == YES ]] || exit 0
LOG="$(mktemp "${MAIN}/checkpoints/luka-science-submit.${COMMIT:0:8}.XXXXXX.log")"
nohup bash "${PROJECT}/scripts/queue_luka_fullsim_science.sh" "${COMMIT}" "${ROOT}" \
    --execute "${PLAN_HASH}" >"${LOG}" 2>&1 </dev/null &
printf 'Submission PID: %s\nLog: %s\n' "$!" "${LOG}"
printf 'Watch with:\ntail -f "%s"\n' "${LOG}"
)
```

The detached helper survives SSH disconnects. Confirm `QUEUE_HELPER_EXIT=0` and
the complete 21-ID ledger before assuming all jobs queued. Ctrl-C on `tail`
stops watching only. Slurm `afterok` dependencies release descendants; no
long-running controller is needed after submission. Never launch another helper
while one is active. On partial failure, retain the live claim and journal and
inspect them; do not delete the claim to force a retry.

```bash
squeue --me -o "%.18i %.12P %.36j %.2t %.10M %R"
```

## 4. Read partial or complete results

Set `COMMIT` to the science commit from above. This reads saved results only.

```bash
(
set -euo pipefail
read -r -p 'Science commit: ' COMMIT </dev/tty
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
PROJECT="/home/ryreu/atlas/HLT_Classification_luka_science_${COMMIT:0:8}"
ROOT="/home/ryreu/atlas/HLT_Classification/checkpoints/luka_science_${COMMIT:0:8}_r1"
source /home/ryreu/miniconda3/etc/profile.d/conda.sh
conda activate /home/ryreu/miniconda3/envs/atlas_kd_sporc
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PROJECT}/src"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -u -s "${PROJECT}/scripts/luka_fullsim_science.py" results --spec "${ROOT}/campaign_spec.json"
)
```

`UNFINISHED` means no valid completion receipt, not necessarily pending in Slurm.
Logs and `TASK/phases/*` separate view construction from scientific training.
Recovery is relative to this run's M0HLT/pure OFFLINE; validation is exploratory,
one seed, and fusion oracles/two-encoder HLT models have unequal input/capacity.
