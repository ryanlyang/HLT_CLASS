# Automatically continue CORR_MID on SPORC tier3

The existing debug gate stays unchanged:

- authenticate_release: 21820095, user reported COMPLETED/0:0;
- build_foundation: 21820096, user reported RUNNING;
- preflight: 21820097, user reported PENDING/Dependency.

Actual accounting, job identity, source and artifacts are rechecked before use.
This is the frozen 100k/50k DIRECT+COARSE comparison, not new tuning. Dataset
production finished as Tigris 229068; user verified 1M/250k/1M metadata and
manifest `ef015d3c86f8188784169f67723a0f9772af96c4dab0a06387b208d9d33b4610`.
No final-test evaluation.

## 1. Commit and push only this amendment on Windows

Run from the local repository. Review the staged diff; preserve unrelated files.

```powershell
$paths = @(
  'src/hlt_classification/cms_proxy_ladder/production.py',
  'src/hlt_classification/cms_proxy_ladder/correlated_tier3.py',
  'src/hlt_classification/correlated_ladder_followup.py',
  'scripts/jetclass2_correlated_tier3.py',
  'scripts/start_jetclass2_correlated_tier3_followup.sh',
  'tests/test_correlated_tier3.py',
  'tests/test_correlated_ladder_followup.py',
  'docs/plans/JETCLASS2_CORRELATED_TIER3_FOLLOWUP_PLAN.md',
  'docs/contracts/JETCLASS2_CORRELATED_TIER3_FOLLOWUP.md',
  'docs/JETCLASS2_CORRELATED_TRACKING_PRODUCTION_RUNBOOK.md',
  'docs/JETCLASS2_CORRELATED_TIER3_FOLLOWUP_RUNBOOK.md',
  'docs/HANDOFF.md',
  'docs/LEGACY_SOURCE_MAP.md'
)
git add -- $paths
if ($LASTEXITCODE -ne 0) { throw 'Staging failed' }
git diff --cached --stat
# Stop if unrelated changes were already staged.
git commit -m "Add automatic CORR_MID tier3 follow-up after existing debug gate" -- $paths
if ($LASTEXITCODE -ne 0) { throw 'Commit failed' }
git push origin HEAD:main
if ($LASTEXITCODE -ne 0) { throw 'Push failed; do not force-push' }
git rev-parse HEAD
```

## 2. Arm on sporcsubmit, using a new detached worktree

Paste the **new** 40-character commit when prompted. Do not reuse the old
062713ea worktree for modified source. Leave its running jobs alone.

```bash
bash <<'BASH'
set -euo pipefail
trap 'echo "Stopped at line ${LINENO}; preserve everything and paste the error." >&2' ERR
read -r -p 'New pushed 40-character follow-up commit: ' COMMIT </dev/tty
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
test "${COMMIT}" != 062713ead3e00721d9f3cb7073e88aef83ebd325
MAIN=/home/ryreu/atlas/HLT_Classification
PROJECT="/home/ryreu/atlas/HLT_Classification_corrtier3_${COMMIT:0:8}"
STUDY="${MAIN}/checkpoints/jc2_corrmid_100k50k_062713ea_r1"
SPEC="${STUDY}/gate/gate_spec.json"
PYTHON=/home/ryreu/miniconda3/envs/atlas_kd_sporc/bin/python

test -x "${PYTHON}"
test -f "${SPEC}"
git -C "${MAIN}" fetch origin main
git -C "${MAIN}" merge-base --is-ancestor "${COMMIT}" origin/main
if [ -e "${PROJECT}" ]; then
    test "$(git -C "${PROJECT}" rev-parse HEAD)" = "${COMMIT}"
    test -z "$(git -C "${PROJECT}" status --porcelain --untracked-files=all)"
else
    git -C "${MAIN}" worktree add --detach "${PROJECT}" "${COMMIT}"
fi

GATE_HASH="$("${PYTHON}" -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["content_hash"])' "${SPEC}")"
HELPER="${PROJECT}/scripts/start_jetclass2_correlated_tier3_followup.sh"
bash "${HELPER}" "${COMMIT}" "${SPEC}" "${GATE_HASH}" 21820097

read -r -p 'Arm automatic 11-job tier3 submission after this gate passes? Type YES: ' CONFIRM </dev/tty
if [[ "${CONFIRM}" != YES ]]; then
    echo 'Read-only review finished. No follow-up armed.'
    exit 0
fi
bash "${HELPER}" "${COMMIT}" "${SPEC}" "${GATE_HASH}" 21820097 --execute
BASH
```

Use the printed tail command and wait for **ARMED**. Authentication may take
several minutes; a background PID alone is not proof. Ctrl-C stops tail only.
The nohup controller survives SSH disconnects; it is not a Slurm training job.

The original three debug jobs remain in their current queue. Once all succeed,
the controller verifies gate outputs, creates an unsubmitted original science
spec, derives the separate tier3 spec with explicit source/profile transfer,
saves and prints the complete dry plan, enforces the preauthorized policy,
runs test-only site checks, and submits all eleven tier3 jobs with dependencies.
There is no further manual step on success and no extra preflight rerun.

Watch for `ALL 11 TIER3 SCIENCE JOBS QUEUED`. Job names start with `jc2crt_`.
The controller deliberately stops on failed gates, changed source, invalid
artifacts, existing debug-science submissions, or ambiguous partial submission.
Preserve its log and journals; do not delete a claim or blindly restart it.

```bash
squeue --me -o '%.18i %.12P %.50j %.2t %.10M %R' |
    grep -E 'JOBID|jc2crg_|jc2crt_'
```

## 3. Saved results

Use the new executor worktree and SPORC environment. No final-test inference:

```bash
export PROJECT_DIR="${PROJECT}" JC2_SITE=sporc_a100
source "${PROJECT_DIR}/sbatch/jetclass2_delphes_common.sh"
python -u -s "${PROJECT_DIR}/scripts/jetclass2_correlated_tier3.py" results \
    --spec /home/ryreu/atlas/HLT_Classification/checkpoints/jc2_corrmid_100k50k_062713ea_r1/science_tier3/campaign_spec.json
```

Reassign PROJECT to the new executor path if using a fresh shell; variables
inside the setup heredoc above do not persist in the calling shell.
Results include accuracy, AUC, per-class QCD rejection @50% and recovery.
