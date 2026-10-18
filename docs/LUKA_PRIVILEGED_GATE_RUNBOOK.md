# Privileged attention-gate ladders on SPORC debug

Authority: [plan](plans/LUKA_PRIVILEGED_GATE_PLAN.md) and
[v1 contract](contracts/LUKA_PRIVILEGED_GATE.md).

This is a NEW campaign: three warm-started D ladders plus three HLT-only
replacement variants and one compute-depth control (13 fits,23 science jobs).
Existing attention jobs, their checkpoints and Fresh_check remain untouched.
No independent test exists for this exploratory299234/74895 split.

Recommended overnight path: **review `auto`, launch `auto`, wait for ARMED**.
One preflight is queued now; all23 science jobs are submitted automatically
only after verified successful preflight and resource admission. No second
confirmation is needed after the gate. Do NOT also submit science manually.

## 1. Commit and push the implementation on Windows

Review/stage ONLY these new paths, not `git add .` (there is unrelated work):

```powershell
git add -- src/hlt_classification/luka_gating scripts/luka_gating.py scripts/queue_luka_gating.sh sbatch/run_luka_gating.sh tests/test_luka_gating.py tests/test_luka_gating_followup.py docs/plans/LUKA_PRIVILEGED_GATE_PLAN.md docs/contracts/LUKA_PRIVILEGED_GATE.md docs/LUKA_PRIVILEGED_GATE_RUNBOOK.md
git diff --cached --stat
```

Also review the new sections in HANDOFF and LEGACY_SOURCE_MAP with `git add -p`
if desired; do not stage unrelated changes wholesale. Commit the reviewed
files, push without force, and record the full40-character commit. If a push
is non-fast-forward, stop and reconcile remote work before continuing. The
new helper uses a clean detached worktree, not the dirty main checkout.

```powershell
$confirm = Read-Host 'Commit and push exactly the staged files you reviewed? Type YES'
if ($confirm -eq 'YES') {
    git commit -m "Add privileged attention gates and matched HLT replacement ladders"
    if ($LASTEXITCODE -ne 0) { throw 'Commit failed or nothing staged; inspect before continuing.' }
    git push origin HEAD:main
    if ($LASTEXITCODE -ne 0) { throw 'Push failed. Do not force-push; reconcile remote work first.' }
    git rev-parse HEAD
}
```

## 2. SPORC: create a clean worktree and dry-review gate plus automatic follow-up

This submits nothing. Set QUOTA to YOUR available GiB after `df -h`
and any site quota inspection. It is an explicit upper bound, not an assertion
that shared free storage belongs to you. Keep at least the declared reserve.

```bash
bash <<'BASH'
set -euo pipefail
read -r -p 'Paste the new pushed 40-character commit: ' COMMIT </dev/tty
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
MAIN=/home/ryreu/atlas/HLT_Classification
PROJECT="/home/ryreu/atlas/HLT_Classification_lukagate_${COMMIT:0:8}"
ROOT="${MAIN}/checkpoints/luka_gate_${COMMIT:0:8}_r1"
PARENT="${MAIN}/checkpoints/luka_fixed30_74aa58c5_r1/science/campaign_spec.json"
read -r -p 'Personally available quota for this campaign (integer GiB): ' QUOTA </dev/tty
[[ "${QUOTA}" =~ ^[1-9][0-9]*$ ]]
git -C "${MAIN}" fetch origin main
git -C "${MAIN}" merge-base --is-ancestor "${COMMIT}" origin/main
if [[ -e "${PROJECT}" ]]; then
    test "$(git -C "${PROJECT}" rev-parse HEAD)" = "${COMMIT}"
    test -z "$(git -C "${PROJECT}" status --porcelain --untracked-files=all)"
else
    git -C "${MAIN}" worktree add --detach "${PROJECT}" "${COMMIT}"
fi
bash "${PROJECT}/scripts/queue_luka_gating.sh" "${COMMIT}" "${ROOT}" "${PARENT}" "${QUOTA}" auto
printf '\nROOT=%s\nPROJECT=%s\nQUOTA=%s\n' "${ROOT}" "${PROJECT}" "${QUOTA}"
BASH
```

Review GPU A100,debug,reu-aisocial,qos_tier3,6CPU,128000MiB,8h preflight.
It authenticates the fixed30 U100 and controls, reads ONLY retained D-side
inputs, tests real installed Weaver/gate parity and the full technical graph,
then measures memory/disk/runtime. Poor accuracy never blocks admission.
Also review the23-job science ceiling template: GPU tasks use the same profile
and at most24h EACH; aggregate uses1CPU/16384MiB/noGPU/1h. Actual train/export
walltimes come from the accepted measurement, with existing safety margins;
24h is a ceiling, not the automatic request for every task. No rows or epochs
are silently dropped to fit. Copy the printed `FOLLOWUP_PLAN_HASH=...` value
(the outermost FOLLOWUP_PLAN content_hash), not either nested gate/science
template hash.

## 3. Queue the preflight and arm automatic science

Use the same values printed above. This block uses shell `read` for prompts;
no Python stdin/TTY mixing, no nested command substitutions around input.

```bash
bash <<'BASH'
set -euo pipefail
read -r -p 'Commit: ' COMMIT </dev/tty
read -r -p 'Quota GiB from dry review: ' QUOTA </dev/tty
read -r -p 'Reviewed FOLLOWUP_PLAN content_hash (outermost): ' PLAN_HASH </dev/tty
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ && "${PLAN_HASH}" =~ ^[0-9a-f]{64}$ && "${QUOTA}" =~ ^[1-9][0-9]*$ ]]
MAIN=/home/ryreu/atlas/HLT_Classification
PROJECT="/home/ryreu/atlas/HLT_Classification_lukagate_${COMMIT:0:8}"
ROOT="${MAIN}/checkpoints/luka_gate_${COMMIT:0:8}_r1"
PARENT="${MAIN}/checkpoints/luka_fixed30_74aa58c5_r1/science/campaign_spec.json"
read -r -p 'Queue the gate and authorize all23 science jobs after verified success? Type YES: ' CONFIRM </dev/tty
[[ "${CONFIRM}" == YES ]] || exit 0
LOG="$(mktemp "${ROOT}/auto-followup.XXXXXX.log")"
nohup bash "${PROJECT}/scripts/queue_luka_gating.sh" "${COMMIT}" "${ROOT}" "${PARENT}" "${QUOTA}" auto --execute "${PLAN_HASH}" >"${LOG}" 2>&1 </dev/null &
printf 'Submission PID: %s\nLog: %s\n' "$!" "${LOG}"
printf 'Watch with: tail -f "%s"\n' "${LOG}"
BASH
```

A helper PID is not proof of arming. Follow the printed `tail -f` command and
wait for `ARMED: preflight <ID>`. Ctrl-C stops tail only; nohup keeps the
controller running after SSH disconnect. While the gate is pending/running,
only the preflight appears in squeue, not the23 science jobs. After admission
the log prints `ALL SCIENCE JOBS QUEUED`, then `QUEUE_HELPER_EXIT=0`.

If a gate was already submitted with the SAME pinned source/root, dry-review
`auto` and authorize its hash to bind that accepted exact journal; it will not
submit another gate. Existing science or ambiguous submissions are rejected.
The source and graph must remain unchanged. A login-host reboot/process kill
can stop nohup: inspect the log and `followup/{armed,complete,failure}.json`
and exact journals if activity stops. Preserve failures and partial submissions;
do not delete claims, launch a duplicate controller or retry blindly.

## 4. Manual alternative only: submit science after successful preflight

SKIP this section when using `auto`. For a deliberately manual workflow use
`gate` instead of `auto` in both review and launch, and use the gate PLAN hash
instead of the FOLLOWUP_PLAN hash. That authorization queues only the preflight,
with no controller. Only that manual workflow needs the following second step.

Run this self-contained block, including after a new SSH login:

```bash
bash <<'BASH'
set -euo pipefail
read -r -p 'Same pushed commit used by the successful gate: ' COMMIT </dev/tty
read -r -p 'Same quota GiB declared at gate review: ' QUOTA </dev/tty
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ && "${QUOTA}" =~ ^[1-9][0-9]*$ ]]
MAIN=/home/ryreu/atlas/HLT_Classification
PROJECT="/home/ryreu/atlas/HLT_Classification_lukagate_${COMMIT:0:8}"
ROOT="${MAIN}/checkpoints/luka_gate_${COMMIT:0:8}_r1"
PARENT="${MAIN}/checkpoints/luka_fixed30_74aa58c5_r1/science/campaign_spec.json"
bash "${PROJECT}/scripts/queue_luka_gating.sh" "${COMMIT}" "${ROOT}" "${PARENT}" "${QUOTA}" science
read -r -p 'Reviewed SCIENCE plan content_hash: ' PLAN_HASH </dev/tty
read -r -p 'Submit the reviewed 23 science jobs? Type YES: ' CONFIRM </dev/tty
if [[ "${CONFIRM}" == YES && "${PLAN_HASH}" =~ ^[0-9a-f]{64}$ ]]; then
    LOG="$(mktemp "${ROOT}/science-submit.XXXXXX.log")"
    nohup bash "${PROJECT}/scripts/queue_luka_gating.sh" "${COMMIT}" "${ROOT}" "${PARENT}" "${QUOTA}" science --execute "${PLAN_HASH}" >"${LOG}" 2>&1 </dev/null &
    printf 'Submission PID: %s\nLog: %s\n' "$!" "${LOG}"
fi
BASH
```

This validates accounting AND the accepted receipt, payloads, source, every
probe and measured requests. Review the 23-job DAG before answering YES;
use its new SCIENCE hash, not the gate hash. Answer anything else to leave
the plan in dry-review state without submitting jobs.

Measurements over debug24h reject science; there is no silent reduction of
epochs, batches, constituents, or scientific rows. Existing campaigns do not
need cancellation. Gates use128000MiB; overlapping jobs may wait on QOS limits.

## Results and progress

```bash
bash <<'BASH'
set -euo pipefail
read -r -p 'Campaign commit: ' COMMIT </dev/tty
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
PROJECT="/home/ryreu/atlas/HLT_Classification_lukagate_${COMMIT:0:8}"
ROOT="/home/ryreu/atlas/HLT_Classification/checkpoints/luka_gate_${COMMIT:0:8}_r1"
squeue --me -o '%.18i %.12P %.45j %.2t %.10M %R'
sacct -u "$USER" -S now-7days -X -n -P -o JobID,JobName%60,State,ExitCode,Elapsed,End
source /home/ryreu/miniconda3/etc/profile.d/conda.sh
conda activate /home/ryreu/miniconda3/envs/atlas_kd_sporc
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${PROJECT}/src"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -u -B -s "${PROJECT}/scripts/luka_gating.py" results --spec "${ROOT}/science/campaign_spec.json"
BASH
```

Partial rows print PENDING. Full validation selects checkpoints; original50k
never selects. Compare PRIV_D000 with SELF_D000 as an oracle diagnostic, then
REPLACE_LOGIT/GATE/STAGED with SELF_CONT for HLT-only endpoint comparisons.
Do not equate a high privileged recovery with deployable HLT performance.

## Local verification

The focused suites are `tests/test_luka_gating.py`, `tests/test_luka_gating_followup.py`, `tests/test_luka_attention.py`,
`tests/test_luka_alljets_30epoch.py`, and `tests/test_luka_fullsim_regular.py`.
Run them in the existing scientific Python environment with installed Weaver,
`PYTHONDONTWRITEBYTECODE=1`, and a fresh pytest temporary directory. Standard
installed-Weaver FP32/BF16 CUDA parity and complete synthetic production-worker
rehearsals are local evidence only; they cannot substitute for the real debug
gate on this dataset and A100 allocation.
