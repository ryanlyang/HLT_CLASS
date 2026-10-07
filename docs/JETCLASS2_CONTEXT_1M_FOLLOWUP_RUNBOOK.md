# Arm automatic Oscar CONTEXT_V1 1M science

This replaces the manual MODE=science step for the already submitted gate.
Do not rerun gate submission, change its worktree, or launch a second manual
science submitter. Wait for **ARMED** in the controller log before disconnecting.
The controller is not a Slurm job; the sixteen science jobs appear only after
the full gate and saved-artifact validation pass.

## Publish only the followup changes (Windows)

The local `context1m-followup-only.patch` is an index-only publication snapshot.
It contains the new operational files and their handoff/donor sections, not
the concurrent V2 work. It is not itself part of the commit. Do not git add .
or git commit -a. If HEAD or the staging area changed, stop rather than reset.

```powershell
& {
    $ErrorActionPreference = 'Stop'
    Set-Location 'C:\Users\22rya\ComputerScience\CERN\HLT_Classification'
    if ((git rev-parse HEAD) -ne '9cebdbe133da20f00cc14ec4155bd627accb7e85') {
        throw 'HEAD changed; stop and inspect before publishing.'
    }
    if (@(git diff --cached --name-only).Count -ne 0) {
        throw 'Other changes are staged; preserve them and stop.'
    }
    git apply --cached --check context1m-followup-only.patch
    if ($LASTEXITCODE -ne 0) { throw 'Patch check failed.' }
    git apply --cached context1m-followup-only.patch
    if ($LASTEXITCODE -ne 0) { throw 'Staging failed.' }
    git diff --cached --check
    if ($LASTEXITCODE -ne 0) { throw 'Staged whitespace check failed.' }
    git diff --cached --stat
    git commit -m 'Add automatic Oscar CONTEXT_V1 1M science followup'
    if ($LASTEXITCODE -ne 0) { throw 'Commit failed.' }
    git push origin HEAD:main
    if ($LASTEXITCODE -ne 0) { throw 'Push failed; do not force-push.' }
    git rev-parse HEAD
}
```

## Review and arm on Oscar

Paste the new executor commit when prompted. The scientific source remains
9cebdbe1. This is a separate worktree and does not invalidate the active gate.
The first helper invocation is read-only. YES authorizes only the bounded
conditional science submission, retaining full measured-profile admission.

```bash
bash <<'BASH'
set -euo pipefail
trap 'echo "Stopped at line ${LINENO}; preserve everything and paste the error." >&2' ERR
read -r -p 'New pushed followup commit: ' COMMIT </dev/tty
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
BASE=/oscar/home/rlyang/hlt_classification
MAIN="${BASE}/source/HLT_Classification"
PROJECT="${BASE}/source/HLT_Classification_context1m_followup_${COMMIT:0:8}"
ROOT="${BASE}/checkpoints/jc2_context_1m250k_direct_coarse_9cebdbe1_r1"
SPEC="${ROOT}/gate/gate_spec.json"
GATE_HASH=f4ad62f5a85f83c4df3e5c43b1fac81380249c175ee3a90e9ab2d4826b68624e
SCIENCE_COMMIT=9cebdbe133da20f00cc14ec4155bd627accb7e85
PREFLIGHT=7083141

test -f "${SPEC}"
git -C "${MAIN}" fetch origin main
git -C "${MAIN}" cat-file -e "${COMMIT}:scripts/start_jetclass2_context_full_followup.sh"
git -C "${MAIN}" merge-base --is-ancestor "${COMMIT}" origin/main
if [ -e "${PROJECT}" ]; then
    test "$(git -C "${PROJECT}" rev-parse HEAD)" = "${COMMIT}"
    test -z "$(git -C "${PROJECT}" status --porcelain --untracked-files=all)"
else
    git -C "${MAIN}" worktree add --detach "${PROJECT}" "${COMMIT}"
fi
HELPER="${PROJECT}/scripts/start_jetclass2_context_full_followup.sh"
ARGS=("${COMMIT}" "${SPEC}" "${GATE_HASH}" "${SCIENCE_COMMIT}" "${PREFLIGHT}")
bash "${HELPER}" "${ARGS[@]}"
read -r -p 'Arm automatic 16-job science after the full gate passes? Type YES: ' CONFIRM </dev/tty
if [[ "${CONFIRM}" != YES ]]; then
    echo 'Stopped without arming.'
    exit 0
fi
bash "${HELPER}" "${ARGS[@]}" --execute
BASH
```

Follow the printed log with tail. Authentication may take several minutes.
ARMED means it is waiting for all three gate jobs; ALL 16 SCIENCE JOBS QUEUED
means the complete journaled science graph was submitted. FOLLOWUP_EXIT=0
means the controller finished successfully, not that training finished.

The process survives ordinary SSH disconnects. Host failure or administrative
cleanup can still stop it. If it exits with an error, inspect that log and
retained `context1m_followup/` evidence; do not delete claims or blindly rerun.
Existing live science ledgers/claims/nonempty journals deliberately stop a new
controller. No automatic failed-job retries or cancellations are performed.

Read-only gate status:

```bash
sacct -j 7083139,7083140,7083141 -X -P \
    -o JobID,JobName%45,State,ExitCode,Elapsed
```

This does not change population, recipe, model, KD, seeds, original 100k jobs
or the sealed final test. No assumption that scientific performance will improve.
