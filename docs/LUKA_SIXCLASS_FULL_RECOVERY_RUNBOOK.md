# Full-validation DE000 epoch comparison

Print full-validation macro AUC, then binary Hbb AUC, with fixed old-M0HLT and
new-OFFLINE recovery anchors. One validation-only GPU replay is necessary:
full macro AUC was not saved per epoch. Running training stays unchanged.

## Local commit and push

From repository PowerShell, stage these eight paths only, then inspect the
staged diff, commit them and push main. Preserve unrelated work and any
unrelated staged files (use the same explicit paths with `git commit --`).

```text
src/hlt_classification/operations/sixclass_full_recovery.py
scripts/compare_luka_sixclass_full_validation.py
scripts/queue_luka_sixclass_full_validation.sh
sbatch/run_luka_sixclass_full_validation.sh
tests/test_sixclass_full_recovery.py
docs/plans/LUKA_SIXCLASS_FULL_RECOVERY_PLAN.md
docs/contracts/LUKA_SIXCLASS_FULL_RECOVERY.md
docs/LUKA_SIXCLASS_FULL_RECOVERY_RUNBOOK.md
```

The dated HANDOFF and LEGACY_SOURCE_MAP additions can be staged separately
using `git add -p`; these files already contain unrelated edits.

## Queue on Oscar

Use the pushed40-character commit. The helper locates the campaign via the
unique OFFLINE/job9442678 log; new DE000 jobs authenticate from saved journals.

```bash
bash <<'BASH'
set -euo pipefail
BASE=/oscar/home/rlyang/hlt_classification
REPO="$BASE/source/HLT_Classification_sixclass_balanced_oscar_8a25f126"
read -r -p 'Pushed 40-character comparison commit: ' COMMIT </dev/tty
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]]
git -C "$REPO" fetch origin main
git -C "$REPO" merge-base --is-ancestor "$COMMIT" origin/main
PROJECT="$BASE/source/HLT_Classification_full_recovery_${COMMIT:0:8}"
[[ ! -e "$PROJECT" && ! -L "$PROJECT" ]]
git -C "$REPO" worktree add --detach "$PROJECT" "$COMMIT"
bash "$PROJECT/scripts/queue_luka_sixclass_full_validation.sh" "$COMMIT"
BASH
```

The helper prints exact log/report paths. Watch with `tail -n 80 -f LOG_PATH`.
Once complete, `sed -n '/^Full validation: /,$p' LOG_PATH` shows both tables.
Ctrl-C stops the view, not the analysis. Snapshot availability is determined
when the GPU analysis begins; later training epochs do not enter that report.
If training is complete by then it includes all20. Missing future epochs are
dashes. Anchors retain the same primary-selected epochs for both metrics.

After inspecting an interruption, explicit resume uses the same source and
commit: `bash "$PROJECT/scripts/queue_luka_sixclass_full_validation.sh" "$COMMIT" resume`.
Do not delete launch records or resume a completed report. Resume retains the
same snapshot; additional epochs require a separate analysis output.

## Verification boundary

Existing Hbb/lower-LR regression:57passed in89.88s. New tests:14passed in9.67s;
final combined analysis regression:36passed in21.61s. Coverage includes
exact CPU/CUDA installed-Weaver checkpoint replay, pooled metrics,
fixed anchor selection, population joins, snapshots, resume/corruption guards,
and real Bash queue/dry-review/resume execution with simulated Slurm. Bash
syntax, CLI help and `git diff --check` passed.
No remote evaluation or submission performed. Exact Oscar replay is checked
inside the analysis allocation. Scientific source and training are unchanged.
