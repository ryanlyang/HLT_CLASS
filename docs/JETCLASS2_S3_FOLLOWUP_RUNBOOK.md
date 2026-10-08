# Arm S3 science after the existing preflight

[Plan](plans/JETCLASS2_S3_FOLLOWUP_PLAN.md) and
[contract](contracts/JETCLASS2_S3_FOLLOWUP.md).

Existing SPORC preflight: **21826026**. Do not cancel, replace, or update its
worktree. This new controller runs from a separate source checkout and invokes
all scientific commands in the original S3 checkout. No new dataset or baseline.

The controller waits for successful accounting and authenticated genuine
preflight evidence. It then produces the complete measured dry plan, checks the
authorized seven-job scope and submits once. The seven science IDs do **not**
appear in `squeue` until that succeeds. This is not seven pre-submitted jobs with
`afterok:21826026`: their time limits cannot be fixed until the gate measures them.
Once submitted, their usual exact-ID internal dependencies control execution.

## Windows: scoped publish

From the repository, review only these changes. Other dirty work is preserved.
The shared HANDOFF may also contain unrelated documentation; inspect its diff
before including it, and use `git add -p` for it if other edits are not ready.

```powershell
$followupFiles = @(
    "src/hlt_classification/s3_ladder_followup.py"
    "scripts/jetclass2_s3_ladder_followup.py"
    "scripts/start_jetclass2_s3_ladder_followup.sh"
    "tests/test_s3_ladder_followup.py"
    "docs/plans/JETCLASS2_S3_FOLLOWUP_PLAN.md"
    "docs/contracts/JETCLASS2_S3_FOLLOWUP.md"
    "docs/JETCLASS2_S3_FOLLOWUP_RUNBOOK.md"
    "docs/LEGACY_SOURCE_MAP.md"
)
git add -- $followupFiles
if ($LASTEXITCODE -ne 0) { throw "Staging failed" }
git diff --cached --stat -- $followupFiles
git commit --only -m "Add one-time S3 science follow-up after preflight" -- $followupFiles
if ($LASTEXITCODE -ne 0) { throw "Commit failed" }
git push origin HEAD:main
if ($LASTEXITCODE -ne 0) { throw "Push failed; do not force-push" }
git rev-parse HEAD
```

HANDOFF is intentionally excluded from this automatic list because it already
had unrelated uncommitted documentation when this change began. Its new S3
evidence can be staged by hunk in a separate documentation commit. It is not
part of the controller's source lock.

## SPORC: arm the bounded continuation

Paste the new commit when prompted. This command authorizes exactly the original
four S3 KD fits, two reducers and summary in debug, after the existing gate.
The automatic dry review checks measured walltimes against the 24-hour ceiling.

```bash
read -r -p 'Paste the new 40-character pushed controller commit: ' FOLLOWUP_COMMIT
(
set -euo pipefail
trap 'echo "Stopped at line ${LINENO}; preserve files and paste the error." >&2' ERR
[[ "$FOLLOWUP_COMMIT" =~ ^[0-9a-f]{40}$ ]]
MAIN=/home/ryreu/atlas/HLT_Classification
CONTROLLER="/home/ryreu/atlas/HLT_Classification_s3_followup_${FOLLOWUP_COMMIT:0:8}"

git -C "$MAIN" fetch origin main
git -C "$MAIN" merge-base --is-ancestor "$FOLLOWUP_COMMIT" origin/main
if [ -e "$CONTROLLER" ]; then
    test "$(git -C "$CONTROLLER" rev-parse HEAD)" = "$FOLLOWUP_COMMIT"
    test -z "$(git -C "$CONTROLLER" status --porcelain --untracked-files=all)"
else
    git -C "$MAIN" worktree add --detach "$CONTROLLER" "$FOLLOWUP_COMMIT"
fi

bash "$CONTROLLER/scripts/start_jetclass2_s3_ladder_followup.sh" \
    "$FOLLOWUP_COMMIT" 21826026 --execute
)
```

For a read-only controller review, omit `--execute`; it validates source,
binding and scope without waiting or writing controller evidence.

Run the printed `tail -f` command. Initial authentication may take several
minutes. Confirm:

```text
ARMED: after successful preflight and artifact validation, all seven S3 science jobs will submit automatically.
Preflight 21826026: PENDING
```

Ctrl-C stops **tail only**. The `nohup` controller survives SSH disconnects,
but not login-host reboot or process termination. It is not a Slurm job. Keep
the printed log path; errors are explicit. A PID alone does not prove arming.
Only one controller can hold the study's `science_followup/controller.lock`.

After the gate succeeds, the log contains the measured dry plan and eventually
`ALL SEVEN SCIENCE JOBS QUEUED` with exact IDs. The immutable completion receipt
records submission, not training completion. A completed controller rerun checks
its ledger/journal and does not duplicate jobs. Failed gate/artifact validation
or ambiguous submission stops the controller; **do not delete claims/journals or
blindly relaunch**. Inspect the log before any recovery. No automatic campaign
extension or final-test evaluation is authorized.
