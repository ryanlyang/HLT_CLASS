# Queue FullStat K1/E40/plain DE022 -> DE011 -> DE000 on Oscar

Three fresh20-epoch fits, exact offline fractions2/9,1/9,0, CE0.5/KD1/E40,
separate auxiliary heads, no relational loss. Teachers are the completed
non-relational ladder plus M0HLT and cumulative new rungs. No separate preflight.
Authority: [plan](plans/LUKA_FULLSTAT_FINE_TAIL_PLAN.md),
[v1 contracts](contracts/LUKA_FULLSTAT_FINE_TAIL.md).

This does not cancel or modify current jobs. The parent must already have
successfully completed; this launcher does not wait for an incomplete parent.
It does not depend on the ongoing relational ladder. No test jobs are queued.

## Local commit/push

Do not use git add .; preserve unrelated changes. All old parent runtime bytes
must remain unchanged. Run in local PowerShell:

```powershell
git add -- src/hlt_classification/luka_fullstat_fine_tail
git add -- scripts/luka_fullstat_fine_tail.py scripts/queue_luka_fullstat_fine_tail.sh scripts/launch_luka_fullstat_fine_tail.sh sbatch/run_luka_fullstat_fine_tail.sh
git add -- tests/test_luka_fullstat_fine_tail.py tests/test_luka_fullstat_fine_tail_launch.py docs/plans/LUKA_FULLSTAT_FINE_TAIL_PLAN.md docs/contracts/LUKA_FULLSTAT_FINE_TAIL.md docs/LUKA_FULLSTAT_FINE_TAIL_RUNBOOK.md
git diff --cached --stat
git commit -m "Add K1 E40 plain exact-ninths FullStat fine tail"
if ($LASTEXITCODE -ne 0) { throw "Inspect commit status before continuing" }
git push origin HEAD:main
if ($LASTEXITCODE -ne 0) { throw "Push failed" }
git rev-parse HEAD
```

HANDOFF and LEGACY_SOURCE_MAP already contain unrelated changes: use git add -p
to include only this task's sections if desired, not entire dirty documents.

## Oscar: source setup and immediate detached launch

Check current quota first. Reserve at least8GiB of personally available durable
space for this campaign, after accounting for other running jobs. Existing
teacher banks are reused in place; D022/D011 views are built in RAM, not saved
as another particle cache. Original ROOT files and assignment cache must remain
available and unchanged. Do not infer available quota from filesystem free space.

Paste the following block on Oscar after pushing. It prompts only for the commit
and available quota before launching. The supplied parent is the existing
completed finalization/retry; change it only if that exact execution moved.

```bash
(
set -euo pipefail
BASE=/oscar/home/rlyang/hlt_classification
REPO="${BASE}/source/HLT_Classification"
PARENT="${BASE}/checkpoints/luka_splithead_finalize_retry_5e704ef4_r1/campaign_spec.json"
read -r -p 'Pushed fine-tail commit (40 characters): ' COMMIT </dev/tty
checkquota
read -r -p 'Personally available result GiB (minimum 8): ' QUOTA </dev/tty
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ && "${QUOTA}" =~ ^[1-9][0-9]*$ ]]
(( QUOTA >= 8 ))
test -r "${PARENT}"
git -C "${REPO}" fetch origin main
git -C "${REPO}" cat-file -e "${COMMIT}^{commit}"
git -C "${REPO}" merge-base --is-ancestor "${COMMIT}" origin/main
PROJECT="${BASE}/source/HLT_Classification_fine_tail_${COMMIT:0:8}"
ROOT="${BASE}/checkpoints/luka_fullstat_fine_tail_k1_e40_${COMMIT:0:8}_r1"
if [[ ! -e "${PROJECT}" ]]; then
    git -C "${REPO}" worktree add --detach "${PROJECT}" "${COMMIT}"
fi
test "$(git -C "${PROJECT}" rev-parse HEAD)" = "${COMMIT}"
bash "${PROJECT}/scripts/launch_luka_fullstat_fine_tail.sh" \
    "${COMMIT}" "${ROOT}" "${PARENT}" "${QUOTA}" \
    --authorize AUTHORIZE_FULLSTAT_FINE_TAIL_THREE_FITS_AUTOSUBMIT_V1
)
```

Once it prints **Detached launcher PID** and the log path, you can disconnect:
slow authentication, full dry review and authorized submission continue without
another prompt. Ctrl-C stops only tail. This protects against SSH disconnect,
not a login-node reboot or site policy. It is not confirmation that submission
already succeeded. Look later for SUBMITTED, LAUNCH_COMPLETE and LAUNCHER_EXIT=0.
An integrity/resource failure stops safely and leaves logs; never rerun blindly.

The token authorizes exactly the3-job plan, not arbitrary future submissions.
ROOT.launch stores intent/approval/complete or failure records. ROOT/submission
stores the dry plan, exact job ledger and journals. Never delete claims to retry.

## Progress and results

Use the exact ROOT and PROJECT printed above, not an ambiguous wildcard.

```bash
squeue --me -o '%.18i %.12P %.40j %.2t %.10M %.12l %R'
tail -n 60 -f "${ROOT}.launch/launcher.log"
```

Worker logs are train_DE022-JOBID.out, train_DE011-JOBID.out and
train_DE000-JOBID.out. New view preparation precedes training in the first two
jobs; expect `FINE_TAIL views` progress before epoch1. Each logs epoch/batch
progress and validation AUC. Startup checks use fresh disposable weights;
scientific initialization then resets. All20 epochs run even if AUC is poor.

To print authenticated completed results, including R30/R50:

```bash
(
set -euo pipefail
export PROJECT_DIR="${PROJECT}"
source "${PROJECT_DIR}/sbatch/luka_splithead_finalize_common.sh"
python -u -B -s "${PROJECT_DIR}/scripts/luka_fullstat_fine_tail.py" \
    results --spec "${ROOT}/campaign_spec.json"
)
```

Only saved validation results are read; no inference or test access. Only
DE000 is deployable. Prior DE000 is an unchanged reference, not a new control.

## Optional manual dry-review path

Instead of detached launch, use the same variables to review:

```bash
bash "${PROJECT}/scripts/queue_luka_fullstat_fine_tail.sh" "${COMMIT}" "${ROOT}" "${PARENT}" "${QUOTA}"
read -r -p 'Reviewed PLAN_HASH: ' PLAN_HASH </dev/tty
bash "${PROJECT}/scripts/queue_luka_fullstat_fine_tail.sh" "${COMMIT}" "${ROOT}" "${PARENT}" "${QUOTA}" --execute "${PLAN_HASH}"
```

Do not combine manual submission with an active or failed detached launch.
Local synthetic CUDA/ROOT tests are not a fresh full-scale Oscar measurement;
integrated allocation/parity/memory/time checks remain mandatory in each job.

## Local verification

2026-10-27 release checks:179passed,2skipped across new fine-tail/launcher tests
and existing split-head/relational/launcher regression suites. New tests account
for65passes and one Linux-only HUP skip. The other skip is the corresponding
historical HUP test; Windows Bash smoke tests passed with synthetic launch tools.
Three full20-epoch native-architecture CUDA miniature fits, five original bank
imports, new target/checkpoint roundtrips, original-finalizer replay, real
synthetic ROOT role guards, exact D-kernel parity and spawn/serial equality pass.
Scheduler and remote allocation checks are mocked locally. No genuine Oscar or
Tigris run, push, submission or cancellation was performed by implementation.
