# Queue CONTEXT_V1 100k/50k fusion on Oscar

This is not the dzfix/SPORC fusion launcher. Existing dataset and 100k campaign
stay untouched. Eight new fits + six reducers + two summaries; no duplicate
CE/direct/coarse fits. Final-test inference remains sealed.

## Publish

Review only `src/hlt_classification/context_fusion/`,
`scripts/jetclass2_context_fusion.py`, `scripts/queue_jetclass2_context_fusion.sh`,
`tests/test_context_fusion.py`, this runbook, the new plan/contract, and the
specific HANDOFF/LEGACY_SOURCE_MAP additions. Do not stage unrelated work.
Commit/push these together. No historical scientific file should change.

## Gate: dry review first

On Oscar, set COMMIT to the new full pushed commit, not the parent commit:

```bash
read -r -p 'New 40-character fusion implementation commit: ' COMMIT
MAIN=/oscar/home/rlyang/hlt_classification/source/HLT_Classification
BASE=/oscar/home/rlyang/hlt_classification
PROJECT="${BASE}/source/HLT_Classification_contextfusion_${COMMIT:0:8}"
ROOT="${BASE}/checkpoints/jc2_context_fusion_100k50k_${COMMIT:0:8}_r1"
PARENT="${BASE}/checkpoints/jc2_context_100k50k_direct_coarse_f889f359_r1/science/campaign_spec.json"
(
set -euo pipefail
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]]
git -C "$MAIN" fetch origin main
git -C "$MAIN" merge-base --is-ancestor "$COMMIT" origin/main
if [ -e "$PROJECT" ]; then
  test "$(git -C "$PROJECT" rev-parse HEAD)" = "$COMMIT"
  test -z "$(git -C "$PROJECT" status --porcelain --untracked-files=all)"
else
  git -C "$MAIN" worktree add --detach "$PROJECT" "$COMMIT"
fi
bash "$PROJECT/scripts/queue_jetclass2_context_fusion.sh" "$COMMIT" "$PARENT" "$ROOT" gate
)
```

Those variables remain available in this shell; restore them using the same
assignments after reconnecting. After
reviewing the single gate plan, set HASH to its printed 64-character content
hash. Submit with a disconnect-safe log:

```bash
(
set -euo pipefail
[[ "$HASH" =~ ^[0-9a-f]{64}$ ]]
LOG="$(mktemp "${ROOT}/gate-submit.XXXXXX.log")"
nohup bash "$PROJECT/scripts/queue_jetclass2_context_fusion.sh" \
  "$COMMIT" "$PARENT" "$ROOT" gate --execute "$HASH" >"$LOG" 2>&1 </dev/null &
printf 'PID: %s\nLog: %s\n' "$!" "$LOG"
)
```

Wait for `preflight/result.json` and a successful Slurm exit. A PID alone does
not prove submission. Preserve claims/journals on error; never relaunch blindly.
180000 MiB may hit user memory QoS; do not suspend/cancel other work here.

## Science

After the new fusion gate succeeds:

```bash
bash "$PROJECT/scripts/queue_jetclass2_context_fusion.sh" "$COMMIT" "$PARENT" "$ROOT" science
```

Review all 16 commands and measured limits. Use the NEW science plan hash in
the same nohup pattern, replacing `gate` by `science`. All new dependencies
queue together. No historical Slurm IDs are required and no automatic follow-up
is armed. Never submit both a manual and separate automatic controller.

## Results

```bash
(
export PROJECT_DIR="$PROJECT" JC2_SITE=oscar_l40s
source "$PROJECT_DIR/sbatch/jetclass2_delphes_common.sh"
python -u -s "$PROJECT_DIR/scripts/jetclass2_context_fusion.py" results --spec "$ROOT/study_spec.json"
)
```

Saved validation results only, including incomplete rows. The decisive endpoints
are FINAL_DIRECT_D000 and FINAL_BRIDGE_D000 versus the original direct/coarse
single-HLT models. U/D offline-containing fusion rows are oracle diagnostics.
