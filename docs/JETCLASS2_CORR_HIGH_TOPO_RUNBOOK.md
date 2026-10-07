# CORR_HIGH_TOPO: queue and inspect

This is a new **100k train / 50k validation** SPORC debug experiment, not a
replacement for CORR_MID or a new 2.25M export. Keep the original CORR_MID
release, its physical blocks, raw offline files and NOISE_V3 pilot: the new
paired reader authenticates those parents. No deletion or test evaluation.

[Scientific plan](plans/JETCLASS2_CORR_HIGH_TOPO_PLAN.md) and
[contract](contracts/JETCLASS2_CORR_HIGH_TOPO.md) define the frozen semantics.
The accuracy gap is an observation, not a tuning target or job acceptance rule.

## Prepare a clean pushed worktree on SPORC

Review and commit the implementation locally first; record its exact pushed
40-character commit. Existing dirty/unrelated work must not be discarded.
Run the following in the **sporcsubmit Bash shell**, not Tigris/Oscar/PowerShell.

```bash
read -r -p 'Exact pushed CORR_HIGH_TOPO implementation commit: ' COMMIT
MAIN=/home/ryreu/atlas/HLT_Classification
PROJECT="/home/ryreu/atlas/HLT_Classification_corrht_${COMMIT:0:8}"
ROOT="${MAIN}/checkpoints/jc2_corrht_100k50k_${COMMIT:0:8}_r1"
OLD="${MAIN}/checkpoints/jc2_corrmid_100k50k_062713ea_r1/science_tier3/campaign_spec.json"
NOISE="${MAIN}/checkpoints/jc2_literature_noise_d505e807_r1/study_spec.json"
```

Use the completed campaign's authenticated release path, not a guessed
directory. These commands only discover metadata and prepare a fresh worktree:

```bash
(
set -euo pipefail
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]]
git -C "$MAIN" fetch origin main
git -C "$MAIN" cat-file -e "${COMMIT}^{commit}"
git -C "$MAIN" merge-base --is-ancestor "$COMMIT" origin/main
test -f "$OLD"
test -f "$NOISE"
if [ -e "$PROJECT" ]; then
    test "$(git -C "$PROJECT" rev-parse HEAD)" = "$COMMIT"
    test -z "$(git -C "$PROJECT" status --porcelain --untracked-files=all)"
else
    git -C "$MAIN" worktree add --detach "$PROJECT" "$COMMIT"
fi
)
```

Stop on any error. In the same shell, resolve the original release and inspect
quota. `df` is filesystem headroom, not personal available quota:

```bash
ORIGINAL="$(python -s -c 'import json,sys; from pathlib import Path; s=json.load(open(sys.argv[1])); print(Path(s["foundation"]["release_root"])/"release.json")' "$OLD")"
printf 'Original release: %s\nNew study: %s\n' "$ORIGINAL" "$ROOT"
df -h "$MAIN"
quota -s
```

Choose a persistent checkpoint location with at least **10 GiB personal
headroom**. Set `QUOTA_GIB` to the actual available amount; do not copy a
filesystem total. The new particle payload also has a 5 GiB hard budget.
Interrupted output is preserved; do not remove it to force a retry.

## Gate: generation, foundation, full GPU resource check

```bash
read -r -p 'Available quota GiB at the persistent study location: ' QUOTA_GIB
HELPER="${PROJECT}/scripts/queue_jetclass2_correlated_topology.sh"
bash "$HELPER" "$COMMIT" "$ROOT" "$ORIGINAL" "$NOISE" "$QUOTA_GIB" gate
```

This is a dry review. Expect three dependent jobs: CPU materialization,
CPU foundation, then genuine installed-Weaver/A100 CE+KD/resource preflight.
Each requests 6 CPUs/90000 MiB/8h; only preflight requests one A100. This is
technical acceptance, not another strength-selection pilot. Poor classifier
performance will not block science. Review the exact printed plan hash.

```bash
read -r -p 'Reviewed gate plan content_hash: ' PLAN_HASH
LOG="$(mktemp "${ROOT}/gate-submit.XXXXXX.log")"
nohup bash "$HELPER" "$COMMIT" "$ROOT" "$ORIGINAL" "$NOISE" "$QUOTA_GIB" \
    gate --execute "$PLAN_HASH" >"$LOG" 2>&1 </dev/null &
printf 'Helper PID: %s\nLog: %s\n' "$!" "$LOG"
```

Inspect the log and gate submission ledger for **accepted numeric job IDs**;
a background PID alone does not mean jobs were queued. Do not repeat a live
submission after an ambiguous interruption; preserve its claim/journal.

## Science after the gate completes

```bash
bash "$HELPER" "$COMMIT" "$ROOT" "$ORIGINAL" "$NOISE" "$QUOTA_GIB" science
```

Expect **11 jobs**: six fresh fits, three teacher reducers, aggregate,
completion. Models: M0HLT, pure OFFLINE, direct OFFLINE->D000, coarse
OFFLINE->D066->D033->D000. No dense/ascent branch. All deployable endpoints
consume the same saved proxy particles. GPU times come from measured
preflight and must fit debug's 24h limit.

After reviewing this second exact plan:

```bash
read -r -p 'Reviewed science plan content_hash: ' PLAN_HASH
LOG="$(mktemp "${ROOT}/science-submit.XXXXXX.log")"
nohup bash "$HELPER" "$COMMIT" "$ROOT" "$ORIGINAL" "$NOISE" "$QUOTA_GIB" \
    science --execute "$PLAN_HASH" >"$LOG" 2>&1 </dev/null &
printf 'Helper PID: %s\nLog: %s\n' "$!" "$LOG"
```

No automatic followup is silently armed by this helper.

## Diagnostics and validation results

Activate `/home/ryreu/miniconda3/envs/atlas_kd_sporc`, set `PYTHONNOUSERSITE=1`,
`PYTHONDONTWRITEBYTECODE=1`, `PYTHONPATH="$PROJECT/src"`, prepend
`${CONDA_PREFIX}/lib` to `LD_LIBRARY_PATH`, and set numerical thread counts to 1.

```bash
python -u -s "$PROJECT/scripts/jetclass2_correlated_topology.py" diagnostics \
    --spec "$ROOT/gate/gate_spec.json"
python -u -s "$PROJECT/scripts/jetclass2_correlated_topology.py" results \
    --spec "$ROOT/science/campaign_spec.json"
```

Diagnostics show ordinary-role particle budgets plus a **train-only**
OFFLINE/proxy mean-and-SD chart (jet counts/PIDs/p4, particle kinematics,
tracking/error/significance/masks). Saved fixed histograms, tails and all PID
categories are in `gate/release/diagnostics.json`. SD denotes width, not
uncertainty. The chart also prints the bounded historical ARM/x86 tracking
replay audit; new serial/process output must be bitwise identical.
Validation results include raw accuracy/AUC, recoveries and
per-class QCD rejection at 50%; zero-background entries remain censored.

Before a larger dataset, inspect the measured distributions and the matched
direct/coarse D000 comparison. A single seed is exploratory evidence, not a
significance claim or a CMS detector-response validation.
