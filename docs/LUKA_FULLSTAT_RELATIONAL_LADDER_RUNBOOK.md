# Queue the fresh FullStat K2.5/E10 relational ladder on Oscar

Four fresh20-epoch fits: UE100 -> DE066 -> DE033 -> DE000, each using
`k2p5_e10_rel`:0.5 CE + total KD2.5 + total embedding10 + unchanged relation.
No standalone
preflight, new controls, polling controller or changes to current jobs.
Use the COMPLETED finalization/retry campaign containing the new DE000, not
luka_splithead_recovery_0c4a71a5_r1. The launcher verifies its receipt AND
successful accounting. An unfinished parent stops before submitting any jobs.

Authority: [plan](plans/LUKA_FULLSTAT_RELATIONAL_LADDER_PLAN.md),
[contracts](contracts/LUKA_FULLSTAT_RELATIONAL_LADDER.md).
Recipe is version2 and execution registration is version3; parent binding
remains version2. Old K5 plans/tokens are rejected. Use a fresh campaign root,
not an edited existing spec or pinned worktree. All new workers
use the verified runtime_repair_v1 environment, not the damaged legacy prefix.

## One-time local commit/push

The finalizer retry is separate work. Commit/push that recovery separately
first if still dirty; preserve its exact source for its jobs. Never fold
unrelated files into this commit or modify a pinned remote worktree.
All parent runtime files must match the completed parent's pinned bytes.
Do not use git add . in this worktree.

```powershell
git add -- src/hlt_classification/luka_fullstat_relational_ladder
git add -- scripts/luka_fullstat_relational_ladder.py scripts/queue_luka_fullstat_relational_ladder.sh scripts/launch_luka_fullstat_relational_ladder.sh sbatch/run_luka_fullstat_relational_ladder.sh
git add -- tests/test_luka_fullstat_relational_ladder.py tests/test_luka_fullstat_relational_launch.py docs/plans/LUKA_FULLSTAT_RELATIONAL_LADDER_PLAN.md docs/contracts/LUKA_FULLSTAT_RELATIONAL_LADDER.md docs/LUKA_FULLSTAT_RELATIONAL_LADDER_RUNBOOK.md
git diff --cached --stat
git commit -m "Add K2.5 E10 relational ladder with detached submission"
if ($LASTEXITCODE -ne 0) { throw "Inspect commit status before continuing" }
git push origin HEAD:main
if ($LASTEXITCODE -ne 0) { throw "Push failed" }
git rev-parse HEAD
```

HANDOFF and LEGACY_SOURCE_MAP contain unrelated changes: use git add -p if
including only this task's new sections. Do not stage whole dirty documents.

## One-time Oscar source setup

This transfers source; it submits no jobs. Run in the same shell as the launch.
Enter the exact completed finalization/retry spec path, not a guessed latest
directory. Personal available home quota must be at least16GiB; filesystem
free space is not quota.

```bash
BASE=/oscar/home/rlyang/hlt_classification
REPO="${BASE}/source/HLT_Classification"
read -r -p 'Pushed relational commit (40 characters): ' COMMIT </dev/tty
read -r -p 'Completed finalization/retry campaign_spec.json (absolute path): ' PARENT </dev/tty
checkquota
read -r -p 'Personally available durable result GiB (minimum16): ' QUOTA </dev/tty
(
set -euo pipefail
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ && "${PARENT}" == /* && "${QUOTA}" =~ ^[1-9][0-9]*$ ]]
test -r "${PARENT}"
git -C "${REPO}" fetch origin main
git -C "${REPO}" cat-file -e "${COMMIT}^{commit}"
git -C "${REPO}" merge-base --is-ancestor "${COMMIT}" origin/main
PROJECT="${BASE}/source/HLT_Classification_relational_${COMMIT:0:8}"
if [[ ! -e "${PROJECT}" ]]; then
    git -C "${REPO}" worktree add --detach "${PROJECT}" "${COMMIT}"
fi
test "$(git -C "${PROJECT}" rev-parse HEAD)" = "${COMMIT}"
test -z "$(git -C "${PROJECT}" status --porcelain --untracked-files=all)"
)
PROJECT="${BASE}/source/HLT_Classification_relational_${COMMIT:0:8}"
ROOT="${BASE}/checkpoints/luka_fullstat_relational_k2p5_e10_${COMMIT:0:8}_r1"
```

If setup errors, stop; do not launch. No jobs are submitted by source setup.

## Launch once: one command, immediate log view

The token below authorizes exactly the fixed four-fit campaign with your
source, parent and quota. It approves automated full dry review followed by
submission of that exact plan, without a second interactive prompt.

```bash
bash "${PROJECT}/scripts/launch_luka_fullstat_relational_ladder.sh" "${COMMIT}" "${ROOT}" "${PARENT}" "${QUOTA}" --authorize AUTHORIZE_FULLSTAT_RELATIONAL_FOUR_FITS_AUTOSUBMIT_V2
```

It detaches a nohup/setsid background launcher BEFORE slow authentication and
becomes tail immediately. Ctrl-C or SSH disconnect stops only the log view.
The launcher continues on that login node, then exits after handing four jobs
to Slurm. This protects against terminal disconnect, not login-node reboot,
site process policy, runtime corruption or failed integrity/resource checks.

Look for DRY_REVIEW_PASSED, then SUBMITTED with four IDs, LAUNCH_COMPLETE and
LAUNCHER_EXIT=0. Until those IDs appear, submission is still in progress.
No separate preflight job or polling controller is created.

Reconnect using the exact path printed at launch:

```bash
tail -n 60 -f "${ROOT}.launch/launcher.log"
```

ROOT.launch contains intent.json, process_claim, approval.json and complete.json
or failure.json. The study contains submission/command_plan.json,
dry_run_submission_ledger.json, live_claim and submission journals. A shell or
Python startup failure may appear only in launcher.log with nonzero LAUNCHER_EXIT.
Never rerun/delete claims after disconnect or ambiguous sbatch output. Inspect
records/accounting first. No implicit r2, retry, cancellation or old-file edits.
An incomplete parent stops this launch; it does not wait or repeat training.

## Optional manual dry-review/hash path

Instead of automatic launch:

```bash
bash "${PROJECT}/scripts/queue_luka_fullstat_relational_ladder.sh" "${COMMIT}" "${ROOT}" "${PARENT}" "${QUOTA}"
```

Review PLAN_HASH and four commands, then explicitly execute:

```bash
read -r -p 'Reviewed PLAN_HASH: ' PLAN_HASH </dev/tty
bash "${PROJECT}/scripts/queue_luka_fullstat_relational_ladder.sh" "${COMMIT}" "${ROOT}" "${PARENT}" "${QUOTA}" --execute "${PLAN_HASH}"
```

Do not combine manual submission with an active/failed automatic launch.

## Progress/results

```bash
squeue --me -o '%.18i %.12P %.40j %.2t %.10M %.12l %R'
```

Worker logs are train_UE100-JOBID.out and similarly for the three D stages.
Each worker retains disposable integrated parity/memory/timing checks, then
starts science from scratch. Estimates are not measured full epochs or
guarantees. On technical failure preserve artifacts and diagnose.

To read saved authenticated validation results:

```bash
(
set -euo pipefail
export PROJECT_DIR="${PROJECT}"
source "${PROJECT_DIR}/sbatch/luka_splithead_finalize_common.sh"
python -u -B -s "${PROJECT_DIR}/scripts/luka_fullstat_relational_ladder.py" results --spec "${ROOT}/campaign_spec.json"
)
```

No new inference/test access. Only DE000 is HLT-deployable. R30/R50 and recovery
use the same AUC-selected checkpoint; historical references are not new controls.
