# Baseline gap sweep: fast reuse, not another ladder

[Scientific plan](plans/JETCLASS2_GAP_SWEEP_PLAN.md) and
[contract](contracts/JETCLASS2_GAP_SWEEP.md) are authoritative.

This adds **three full CE fits** to the existing 100k/50k CORR_HIGH_TOPO
population. It reuses the completed OFFLINE and M0HLT controls, avoids a new
dataset/foundation/matching job, and allows the three fits to run independently.
One new genuine A100 technical preflight precedes a separate science dry/live
review. Expect roughly baseline-scale fit times, not guaranteed queue times;
the supplied parent baselines took about 40–43 minutes each. Do not compare
short acceptance metrics against the fully trained controls.

No new KD jobs, final-test access, parent mutation, cancellation, deletion,
or 2.25M export. A future KD campaign must authenticate the frozen chosen
endpoint and declare its intermediate views separately. This screen does
not implement or authorize that campaign. Parent artifacts must be preserved.

## Source and preparation

Commit only the new `src/hlt_classification/gap_sweep/`,
`scripts/jetclass2_gap_sweep.py`, `scripts/queue_jetclass2_gap_sweep.sh`,
`tests/test_gap_sweep.py`, this runbook, new plan/contract, and the relevant
HANDOFF/LEGACY_SOURCE_MAP updates. Preserve unrelated dirty files. Push the
commit normally; do not force-push. Parent scientific files must remain
byte-identical to `8c5b47029ed15f88aa852b6f38d06e98bbbfcd9b`.

On **sporcsubmit**, set COMMIT to the new pushed 40-character commit, then:

```bash
MAIN=/home/ryreu/atlas/HLT_Classification
PROJECT="/home/ryreu/atlas/HLT_Classification_gap_${COMMIT:0:8}"
ROOT="${MAIN}/checkpoints/jc2_gap_100k50k_${COMMIT:0:8}_r1"
PARENT="${MAIN}/checkpoints/jc2_corrht_100k50k_8c5b4702_r1/science/campaign_spec.json"
(
set -euo pipefail
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]]
test -f "$PARENT"
git -C "$MAIN" fetch origin main
git -C "$MAIN" merge-base --is-ancestor "$COMMIT" origin/main
if [[ -e "$PROJECT" ]]; then
    test "$(git -C "$PROJECT" rev-parse HEAD)" = "$COMMIT"
    test -z "$(git -C "$PROJECT" status --porcelain --untracked-files=all)"
else
    git -C "$MAIN" worktree add --detach "$PROJECT" "$COMMIT"
fi
bash "$PROJECT/scripts/queue_jetclass2_gap_sweep.sh" "$COMMIT" "$PARENT" "$ROOT" gate
)
```

Review one `jc2gap_preflight` job: debug, A100, 6 CPUs, 90,000 MiB, 2h
limit (not a runtime prediction). It replays all candidates on training
rows, performs installed-Weaver parity and one full S1 CE acceptance pass.
Save the printed plan content_hash as PLAN_HASH, not a placeholder.

```bash
LOG="$(mktemp "${MAIN}/checkpoints/gap-gate-submit.XXXXXX.log")"
nohup bash "$PROJECT/scripts/queue_jetclass2_gap_sweep.sh" \
    "$COMMIT" "$PARENT" "$ROOT" gate --execute "$PLAN_HASH" \
    >"$LOG" 2>&1 </dev/null &
printf 'Helper PID: %s\nLog: %s\n' "$!" "$LOG"
tail -f "$LOG"
```

Ctrl-C stops tail, not the helper. A helper PID is not submission proof;
look for the saved Slurm ledger under `submission_gate/submission_ledger.json`.
Never relaunch an interrupted live submission with an unresolved claim/journal.

## Three independent baseline fits, then summary

After the gate completes successfully, dry-review science:

```bash
bash "$PROJECT/scripts/queue_jetclass2_gap_sweep.sh" \
    "$COMMIT" "$PARENT" "$ROOT" science
```

Expect fit_S1/S2/S3 (each A100, 6 CPUs, 90,000 MiB, measured walltime)
with no mutual dependencies; summary is CPU-only afterok all three fits.
They may start together if GPUs are available. No early pruning on quality.
Set SCIENCE_HASH to this separately printed plan hash and submit:

```bash
LOG="$(mktemp "${MAIN}/checkpoints/gap-science-submit.XXXXXX.log")"
nohup bash "$PROJECT/scripts/queue_jetclass2_gap_sweep.sh" \
    "$COMMIT" "$PARENT" "$ROOT" science --execute "$SCIENCE_HASH" \
    >"$LOG" 2>&1 </dev/null &
printf 'Helper PID: %s\nLog: %s\n' "$!" "$LOG"
```

Read partial/final results in the registered SPORC environment:

```bash
export PROJECT_DIR="$PROJECT" JC2_SITE=sporc_a100_debug
source "$PROJECT/sbatch/jetclass2_delphes_common.sh"
python -u -s "$PROJECT/scripts/jetclass2_gap_sweep.py" results --spec "$ROOT/study_spec.json"
```

Reports contain raw accuracy, AUC, QCD rejection, and OFFLINE gap in
**percentage points**. Select the mildest candidate with **gap >1 and <=2 pp**
only when all three fits are committed. No qualifying candidate is a successful
scientific result with no winner. Reused validation and one seed do not establish
independent confirmation or guarantee a population-level gap. Full baseline
reports/selected weights and train-only descriptive statistics are under
`fit_S*/`; `summary.json` is the authenticated final screen result.
