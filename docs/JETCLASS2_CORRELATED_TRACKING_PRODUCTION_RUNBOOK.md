# Frozen CORR_MID: full dataset, then SPORC direct/coarse

**Current user choice:** preserve the running debug gate, automatically launch
science on tier3. Use the [new follow-up runbook](JETCLASS2_CORRELATED_TIER3_FOLLOWUP_RUNBOOK.md)
instead of manually submitting debug science in section 3 below. The original
dataset/gate paths and pins remain valid.

Authority: [plan](plans/JETCLASS2_CORRELATED_TRACKING_PRODUCTION_PLAN.md),
[production contract](contracts/JETCLASS2_CORRELATED_TRACKING_PRODUCTION.md),
[classifier contract](contracts/JETCLASS2_CORRELATED_TRACKING_LADDER.md).

Freeze MID from Tigris pilot 228972, commit
`0667355d8c4d72e48e32c03f2d1540595fe654b7`. No strength screen or new recipe fit.
This dataset is synthetic, not measured CMS HLT. A small CE accuracy gap or
ladder advantage has not been established.

## 1. Push and pin the implementation

Commit only the intended implementation and documentation, preserving unrelated
work, then push normally. Use the **new full 40-character implementation commit**
below, not the historical pilot commit. Never force-push or edit old worktrees.

On Tigris, use a fresh detached worktree. This step does not submit jobs:

```bash
read -r -p 'New pushed 40-character commit: ' COMMIT
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]] || exit 1
MAIN=/home/ryreu/atlas/HLT_Classification
PROJECT="/home/ryreu/atlas/HLT_Classification_corrprod_${COMMIT:0:8}"
ROOT="/home/ryreu/atlas/datasets/jetclass2_correlated_mid_2250k_${COMMIT:0:8}_r1"
git -C "${MAIN}" fetch origin main
git -C "${MAIN}" merge-base --is-ancestor "${COMMIT}" origin/main || exit 1
if [ -e "${PROJECT}" ]; then
    test "$(git -C "${PROJECT}" rev-parse HEAD)" = "${COMMIT}" || exit 1
    test -z "$(git -C "${PROJECT}" status --porcelain --untracked-files=all)" || exit 1
else
    git -C "${MAIN}" worktree add --detach "${PROJECT}" "${COMMIT}" || exit 1
fi
```

The original pilot/source, dzfix raw offline, inventory and TRAIN_1M profile
must remain available at their existing authenticated RIT locations. No dataset
transfer is necessary between Tigris and SPORC.

## 2. Create and review the whole production plan on Tigris

Check both filesystem space and your **own available quota**, not shared free
space alone. The default budget is 20 GiB plus 2 GiB free headroom. Attest only
after confirming sufficient quota and that the destination is persistent:

```bash
df -h /home/ryreu/atlas/datasets
quota -s
read -r -p 'Currently available personal quota, GiB (at least 22): ' QUOTA
read -r -p 'Confirmed persistent destination and quota? Type YES: ' STORAGE
test "${STORAGE}" = YES || exit 1
export CORRELATED_PERSISTENT_STORAGE=YES
bash "${PROJECT}/scripts/queue_jetclass2_correlated_tracking_dataset.sh" \
    "${COMMIT}" "${ROOT}" "${QUOTA}"
```

No jobs are submitted yet. Review the printed plan: CPU-only Tigris,
reu-aisocial, 36 CPUs/64 GiB/2h preflight and generation, `%16` generation
concurrency, 1 CPU/16 GiB/2h finalizer. Counts must be 1M train, 250k validation,
1M sealed test. Preserve a partially created root if any check fails; do not
delete it or invent a replacement artifact.

After review, enter the printed plan content hash. This **one execution queues
the entire generation DAG**, not just the preflight:

```bash
read -r -p 'Reviewed 64-character production plan hash: ' PLAN_HASH
[[ "${PLAN_HASH}" =~ ^[0-9a-f]{64}$ ]] || exit 1
LOG="$(mktemp "${MAIN}/checkpoints/corrmid-production.XXXXXX.log")"
nohup bash "${PROJECT}/scripts/queue_jetclass2_correlated_tracking_dataset.sh" \
    "${COMMIT}" "${ROOT}" "${QUOTA}" --execute "${PLAN_HASH}" \
    >"${LOG}" 2>&1 </dev/null &
printf 'Submission helper PID: %s\nLog: %s\n' "$!" "${LOG}"
tail -f "${LOG}"
```

Wait for the ledger with all three scheduler IDs. A background PID alone does
not establish successful submission. Ctrl-C stops `tail`, not the helper.
Preflight reproduces 64 saved pilot training rows using the real production
reader/processes/writer, then admits all shards automatically on success.
The finalizer runs after the array terminates and fails closed if incomplete.
Resource projections are estimates, not an ETA or physics quality test.

```bash
squeue --me -o '%.18i %.12P %.30j %.2t %.10M %R'
sacct -u "$USER" -S now-2days -X -n -P \
    -o JobID,JobName%40,State,ExitCode,Elapsed,End | grep 'jc2crp_'
```

Authoritative completion is `${ROOT}/dataset_manifest.json` plus authenticated
shard receipts, not just COMPLETED in Slurm. Ordinary manifests live under
`releases/train.json` and `releases/validation.json`. Physical blocks live under
`attempts/<attempt>/shards/<shard>/block_*.npz`, joined by canonical jet identity
and original file/tree/entry membership. Never pair by glob order. `valid` masks
and mm/GeV units are mandatory; source keys are metadata only.

If execution fails, preserve logs, reservations, completed blocks and journals.
The CLI `recover --spec ROOT/study_spec.json --name recovery1` requires all exact
prior job IDs to be terminal, validates retained shards and prepares a new
missing-only attempt. It does not submit it; review its plan and use the recovery
authorization phrase printed there. Never simply rerun an ambiguous live submit.

## 3. SPORC 100k/50k gate, then the complete comparison

On **sporcsubmit**, set the same COMMIT/PROJECT/ROOT values from above. The
worktree and dataset are on shared storage. Wait for complete dataset publication.
Keep the production ROOT unchanged and define a separate classifier study:

```bash
STUDY="/home/ryreu/atlas/HLT_Classification/checkpoints/jc2_corrmid_100k50k_${COMMIT:0:8}_r1"
HELPER="${PROJECT}/scripts/queue_jetclass2_correlated_tracking_ladder.sh"
bash "${HELPER}" "${COMMIT}" "${ROOT}" "${STUDY}" gate
```

Review the three debug gate jobs (release authentication, pairing/cache foundation,
full-population A100 preflight), then execute the printed exact plan:

```bash
read -r -p 'Reviewed gate plan hash: ' GATE_HASH
[[ "${GATE_HASH}" =~ ^[0-9a-f]{64}$ ]] || exit 1
LOG="$(mktemp "${STUDY}/gate-submit.XXXXXX.log")"
nohup bash "${HELPER}" "${COMMIT}" "${ROOT}" "${STUDY}" gate \
    --execute "${GATE_HASH}" >"${LOG}" 2>&1 </dev/null &
printf 'Gate submission PID: %s\nLog: %s\n' "$!" "${LOG}"
```

This is infrastructure validation, not a CE strength-selection study. It includes
installed-Weaver forward/gradient parity, one CE/KD acceptance pass and measured
memory/runtime on SPORC. After `gate/gate_complete.json` is authenticated:

```bash
bash "${HELPER}" "${COMMIT}" "${ROOT}" "${STUDY}" science
read -r -p 'Reviewed science plan hash: ' SCIENCE_HASH
[[ "${SCIENCE_HASH}" =~ ^[0-9a-f]{64}$ ]] || exit 1
LOG="$(mktemp "${STUDY}/science-submit.XXXXXX.log")"
nohup bash "${HELPER}" "${COMMIT}" "${ROOT}" "${STUDY}" science \
    --execute "${SCIENCE_HASH}" >"${LOG}" 2>&1 </dev/null &
printf 'Science submission PID: %s\nLog: %s\n' "$!" "${LOG}"
```

All 11 science jobs submit together with dependencies: six fresh fits, three
probability reducers, aggregate and completion. OFFLINE and HLT CE controls;
direct OFFLINE→D000 and coarse OFFLINE→D066→D033→D000. No dense/ascent/screening
fits; ascent would repeat identical OFFLINE views for this tracking-only recipe.
Every train/reducer requests one A100, 6 CPUs, 90000 MiB, debug, measured time
within 24h. Never use an old NOISE_V3/CONTEXT/CMS-proxy profile or campaign spec.

For saved validation results, activate the existing SPORC environment via the
absolute common helper before invoking the CLI:

```bash
export PROJECT_DIR="${PROJECT}" JC2_SITE=sporc_a100_debug
source "${PROJECT_DIR}/sbatch/jetclass2_delphes_common.sh"
python -u -s "${PROJECT}/scripts/jetclass2_correlated_tracking_ladder.py" results \
    --spec "${STUDY}/science/campaign_spec.json"
```

Reports include accuracy, AUC, QCD rejection at 50% signal efficiency and recovery
against fresh HLT/OFFLINE controls. All final-test inference remains sealed.
Poor recovery is a result; it never cancels remaining registered experiments.
