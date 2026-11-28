# Repair a failed full D20 input build while other jobs run

Parent: `/cs/home/rlyang/hlt_u100_pairs/runs/d20_full_5af66449_r1`.
Original source: `5af66449d2b56ae37b01a642c7246027db94b097`.
The user's qualification239597 and release239598 completed. M0HLT_s22_p01 job
243576 then timed out before any scientific epoch: two workers repeatedly
waited in fsspec file-context exit while loading ROOT baskets on gpu1906.
The stacks locate the I/O stall but do not establish its server/client cause.
Other M0HLT seeds were already in segment2; active OFFLINE jobs remain untouched.

This separate execution adapter uses a persistent native local reader plus
bounded fresh-worker retries. It authenticates/reuses the original scientific
source and qualification, adds a small actual CPU reader-equality probe, then
replaces only the requested failed chain and final report. No full17h GPU
qualification replay is requested. See [the contract](contracts/LUKA_D20_INPUT_RECOVERY.md).

The implementation lives in `src/hlt_classification/luka_d20_input_recovery/`:
`reader.py` changes only the physical source, `decode.py` bounds retries,
`runtime.py` declares the process-local overrides, `probe.py` checks actual
authorized data, and `controller.py` plans/journals selective recovery.
Original science files and immutable worktrees are not edited. New execution
receipts and accepted job records bind the I/O policy to replacement epochs.

In local PowerShell:

```powershell
Set-Location 'C:\Users\22rya\ComputerScience\CERN\HLT_Classification'
$Files = @(
  'src/hlt_classification/luka_d20_input_recovery'
  'scripts/recover_luka_d20_inputs.py'
  'scripts/launch_luka_d20_input_recovery.sh'
  'sbatch/run_luka_d20_input_recovery.sh'
  'tests/test_luka_d20_input_recovery.py'
  'docs/LUKA_D20_INPUT_RECOVERY.md'
  'docs/contracts/LUKA_D20_INPUT_RECOVERY.md'
  'docs/plans/LUKA_D20_FULL_HYDRA_PLAN.md'
)
git add -- $Files
if ($LASTEXITCODE -ne 0) { throw 'Staging failed' }
git diff --cached --stat -- $Files
git commit --only -m "Add targeted D20 input recovery with persistent reads and bounded retries" -- $Files
if ($LASTEXITCODE -ne 0) { throw 'Commit failed' }
git push origin HEAD:main
if ($LASTEXITCODE -ne 0) { throw 'Push failed' }
git rev-parse HEAD
```

Shared HANDOFF/LEGACY_SOURCE_MAP are updated separately;
their unrelated existing edits are deliberately excluded from whole-file staging.

In the Hydra terminal, paste the new full commit:

```bash
bash <<'BASH'
set -euo pipefail
BASE=/cs/home/rlyang/hlt_u100_pairs
read -r -p 'Pushed input-recovery commit: ' COMMIT </dev/tty
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]]
git -C "$BASE/code_65952ed3" fetch origin main
git -C "$BASE/code_65952ed3" merge-base --is-ancestor "$COMMIT" origin/main
PROJECT="$BASE/code_d20_input_${COMMIT:0:8}"
if [[ ! -e "$PROJECT" && ! -L "$PROJECT" ]]; then
    git -C "$BASE/code_65952ed3" worktree add --detach "$PROJECT" "$COMMIT"
fi
ROOT="$BASE/runs/d20_input_recovery_${COMMIT:0:8}_r1"
bash "$PROJECT/scripts/launch_luka_d20_input_recovery.sh" \
    "$COMMIT" "$ROOT" M0HLT_s22_p01
BASH
```

Run once. The detached launcher prints/dry-reviews the actual targeted plan,
then queues CPU probe -> CPU release. Existing jobs are not cancelled before
the probe succeeds. Release recomputes current status, preserves running fits,
and creates replacement M0HLT_s22 segments1--4 and a report. It can cancel only
exact pending descendants, normally the old report243624; cancelled seed22
followers are replaced without touching other seeds/teachers.

If a requested job has acquired a newer attempt, an active scientific descendant
exists, or submission is uncertain, recovery stops with evidence. Do not delete
claims or journals to rerun. Other failures require a separately registered
explicit task request after inspecting their cause; this is not a broad restart.

After launching, inspect the printed recovery log or:

```bash
squeue --me -o '%.18i %.40j %.2t %.10M %R'
tail -n 60 -f "$ROOT.launch/launcher.log"
```

After the CPU release, replacements write their normal logs and scientific
epochs under the original parent campaign. The original results/progress CLI
continues to work, because the repair appends authenticated original-format
accepted bindings. Additional I/O provenance is under the recovery root's
`reader_parity.json`, `executions/`, `retries/` and `released.json`.

Local validation on2026-11-28:34 repair tests passed in48.26s, including native
ROOT/full-input-builder byte equality in all four view/target configurations,
role isolation, real spawned workers, timeout cleanup, bounded retry/replay,
checksum failure, complete probe coverage, deployment/source drift, exact
pending-only cancellation and selective live-job preservation. Eleven existing
D20 checkpoint/restore, disk/export failure, startup and qualification guards
passed in99.23s. Bash syntax and the isolated CLI parser also passed.
Remote CPU parity and real replacement allocation acceptance remain pending;
no remote job or final-test inference was performed during implementation.
