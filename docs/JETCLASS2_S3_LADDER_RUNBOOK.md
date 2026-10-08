# Frozen S3 direct/coarse follow-up

[Plan](plans/JETCLASS2_S3_LADDER_PLAN.md) and
[contract](contracts/JETCLASS2_S3_LADDER.md) define the experiment.

Same 100k train / 50k development validation. Reuse OFFLINE and S3 controls and
the original OFFLINE temperature-2 training bank. Four new fresh KD fits, two
train-only reducers, one summary; no ascending U000/dense path, regeneration,
old-job changes, or test inference. S3=0%, OFFLINE=100% recovery. The final
comparison is coarse D000 versus direct D000, not an oracle intermediate.

## Publish scoped source

From Windows, review and commit only:

```powershell
$s3Files = @(
    "src/hlt_classification/s3_ladder"
    "scripts/jetclass2_s3_ladder.py"
    "scripts/queue_jetclass2_s3_ladder.sh"
    "tests/test_s3_ladder.py"
    "docs/plans/JETCLASS2_S3_LADDER_PLAN.md"
    "docs/contracts/JETCLASS2_S3_LADDER.md"
    "docs/JETCLASS2_S3_LADDER_RUNBOOK.md"
    "docs/HANDOFF.md"
    "docs/LEGACY_SOURCE_MAP.md"
)
git add -- $s3Files
if ($LASTEXITCODE -ne 0) { throw "Staging failed" }
git diff --cached --stat -- $s3Files
git commit --only -m "Add frozen S3 direct versus coarse comparison" -- $s3Files
if ($LASTEXITCODE -ne 0) { throw "Commit failed" }
git push origin HEAD:main
if ($LASTEXITCODE -ne 0) { throw "Push failed; do not force-push" }
git rev-parse HEAD
```

Old screen and CORR_HIGH_TOPO scientific files must remain byte-identical. The
new checkout cannot reuse old gates if those sources drift. Keep both old
worktrees and campaign/dataset artifacts available.

## SPORC: prepare the new gate

The source screen is the completed `jc2_gap_100k50k_8abdc943_r1`, summary job
21825507. This is not a new strength sweep. No fresh baseline fit is queued.

```bash
read -r -p 'Paste the new 40-character pushed commit: ' COMMIT
MAIN=/home/ryreu/atlas/HLT_Classification
PROJECT="/home/ryreu/atlas/HLT_Classification_s3_${COMMIT:0:8}"
ROOT="${MAIN}/checkpoints/jc2_s3_100k50k_${COMMIT:0:8}_r1"
SWEEP="${MAIN}/checkpoints/jc2_gap_100k50k_8abdc943_r1/study_spec.json"
(
set -euo pipefail
trap 'echo "Stopped at line ${LINENO}; preserve files and paste the error." >&2' ERR
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]]
test -f "$SWEEP"
git -C "$MAIN" fetch origin main
git -C "$MAIN" merge-base --is-ancestor "$COMMIT" origin/main
if [ -e "$PROJECT" ]; then
    test "$(git -C "$PROJECT" rev-parse HEAD)" = "$COMMIT"
    test -z "$(git -C "$PROJECT" status --porcelain --untracked-files=all)"
else
    git -C "$MAIN" worktree add --detach "$PROJECT" "$COMMIT"
fi
LOG="$(mktemp "${MAIN}/checkpoints/s3-prepare.XXXXXX.log")"
nohup bash "$PROJECT/scripts/queue_jetclass2_s3_ladder.sh" \
    "$COMMIT" "$SWEEP" "$ROOT" gate >"$LOG" 2>&1 </dev/null &
printf 'Preparation PID: %s\nLog: %s\n' "$!" "$LOG"
printf 'Watch with:\ntail -f "%s"\n' "$LOG"
)
```

Wait for `Dry review only. No jobs submitted.` Review one debug A100 job,
6 CPUs/90000 MiB/2h ceiling. Real installed-Weaver parity, unchanged S3 input
replay, full-size KD pass with the actual teacher, and reducer timing run here.
Local CPU tests are not substitutes. Ctrl-C stops tail only.

In the same SSH session, submit the exact gate:

```bash
(
set -euo pipefail
PLAN="${ROOT}/submission_gate/command_plan.json"
test -f "${ROOT}/submission_gate/dry_run_submission_ledger.json"
cat "$PLAN"
HASH="$(/home/ryreu/miniconda3/envs/atlas_kd_sporc/bin/python -s -c \
    'import json,sys; p=json.load(open(sys.argv[1])); assert p["mode"]=="gate"; assert len(p["commands"])==1; print(p["content_hash"])' "$PLAN")"
read -r -p 'Submit this exact S3 preflight? Type YES: ' CONFIRM </dev/tty
if [ "$CONFIRM" != YES ]; then exit 0; fi
LOG="$(mktemp "${ROOT}/gate-submit.XXXXXX.log")"
nohup bash "$PROJECT/scripts/queue_jetclass2_s3_ladder.sh" \
    "$COMMIT" "$SWEEP" "$ROOT" gate --execute "$HASH" >"$LOG" 2>&1 </dev/null &
printf 'Submission helper PID: %s\nLog: %s\n' "$!" "$LOG"
printf 'Watch with:\ntail -f "%s"\n' "$LOG"
)
```

A helper PID is not a Slurm ID. Confirm the actual gate job in the ledger/log.
Do not blindly retry an interrupted live submission or delete a claim.

## After successful gate: seven science jobs

Run the helper with `science` **without** `--execute` first (use `nohup`/a log
as above if needed). It authenticates the new measured preflight and emits
the separately reviewed science plan under `submission_science/`:

```bash
bash "$PROJECT/scripts/queue_jetclass2_s3_ladder.sh" \
    "$COMMIT" "$SWEEP" "$ROOT" science
```

Expected tasks: direct fit and coarse D066 fit have no mutual dependency;
D066 reducer -> D033 fit -> D033 reducer -> coarse D000 fit;
CPU summary after all four fits. All GPU jobs use debug/A100/6 CPUs/90000 MiB,
with measured walltimes. Previous baseline ~38min runtimes are context, not a
KD runtime or queue guarantee. No dependency on historical scheduler IDs.

After reviewing the plan, read its exact hash and authorize all seven jobs:

```bash
(
set -euo pipefail
PLAN="${ROOT}/submission_science/command_plan.json"
test -f "${ROOT}/submission_science/dry_run_submission_ledger.json"
cat "$PLAN"
HASH="$(/home/ryreu/miniconda3/envs/atlas_kd_sporc/bin/python -s -c \
    'import json,sys; p=json.load(open(sys.argv[1])); assert p["mode"]=="science"; assert len(p["commands"])==7; print(p["content_hash"])' "$PLAN")"
read -r -p 'Submit the reviewed seven S3 science jobs? Type YES: ' CONFIRM </dev/tty
if [ "$CONFIRM" != YES ]; then exit 0; fi
LOG="$(mktemp "${ROOT}/science-submit.XXXXXX.log")"
nohup bash "$PROJECT/scripts/queue_jetclass2_s3_ladder.sh" \
    "$COMMIT" "$SWEEP" "$ROOT" science --execute "$HASH" >"$LOG" 2>&1 </dev/null &
printf 'Submission helper PID: %s\nLog: %s\n' "$!" "$LOG"
printf 'Watch with:\ntail -f "%s"\n' "$LOG"
)
```

## Results

```bash
(
set -euo pipefail
export PROJECT_DIR="$PROJECT" JC2_SITE=sporc_a100_debug
source "$PROJECT/sbatch/jetclass2_delphes_common.sh"
LOG="$(mktemp "${ROOT}/results.XXXXXX.log")"
nohup python -u -s "$PROJECT/scripts/jetclass2_s3_ladder.py" results \
    --spec "$ROOT/study_spec.json" >"$LOG" 2>&1 </dev/null &
printf 'Results log: %s\nWatch with:\ntail -f "%s"\n' "$LOG" "$LOG"
)
```

Results can be partial; `NOT COMMITTED` is not a scheduler state. Completed
reports must authenticate their selected weights, teacher and input lineage.
All four results are required for the final summary, regardless of quality.
Keep one-seed/reused-validation/unequal-total-training-budget caveats. No
final-test evaluation or dataset generation is included.
