# HCWDL Unified-Balanced Full-Data Three-Arm Runbook

This runbook starts the all-mapped 20-pass campaign. It does not submit a new
smoke. One live command submits the shared foundation and a source-pinned
`afterok` autolaunch job. The autolaunch creates and submits the three arm
ledgers only after `locks/foundation.json` exists.

## 1. Source-pinned worktree and roots

Run on Tigris after the implementation commit is pushed:

```bash
set -euo pipefail
source /home/ryreu/miniforge3-aarch64/etc/profile.d/conda.sh
conda activate atlas_kd_tigris
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

export MAIN_REPO=/home/ryreu/atlas/HLT_Classification
git -C "${MAIN_REPO}" fetch origin main
export UB_FULL_COMMIT="$(git -C "${MAIN_REPO}" rev-parse origin/main)"
export UB_FULL_SHORT="${UB_FULL_COMMIT:0:8}"
export UB_FULL_WORKTREE="/home/ryreu/atlas/HLT_Classification_hcwdl_ub_full3_${UB_FULL_SHORT}"
export UB_FULL_FOUNDATION_ROOT="${MAIN_REPO}/checkpoints/hcwdl_ub_full3_foundation_${UB_FULL_SHORT}_r1"
export UB_FULL_ARMS_ROOT="${MAIN_REPO}/checkpoints/hcwdl_ub_full3_arms_${UB_FULL_SHORT}_r1"
export UB_FULL_LEDGER_ROOT="${MAIN_REPO}/checkpoints/hcwdl_ub_full3_ledgers_${UB_FULL_SHORT}_r1"

test ! -e "${UB_FULL_WORKTREE}"
git -C "${MAIN_REPO}" worktree add --detach \
  "${UB_FULL_WORKTREE}" "${UB_FULL_COMMIT}"
test -z "$(git -C "${UB_FULL_WORKTREE}" status --porcelain)"
export PYTHONPATH="${UB_FULL_WORKTREE}/src"
```

Bind the exact prepared 300k U/J campaign used as the immutable preprocessing
template. Its coupling, endpoint-equality, and graph/recipe locks must exist;
its training results are not teachers for this campaign.

```bash
export PARENT_UJ_ROOT=/absolute/path/to/the/authenticated/300k/hcwdl_uj_campaign
export PARENT_UJ_SPEC="${PARENT_UJ_ROOT}/campaign_spec.json"
test -f "${PARENT_UJ_SPEC}"
test -f "${PARENT_UJ_ROOT}/locks/coupling_lock.json"
test -f "${PARENT_UJ_ROOT}/locks/endpoint_equality_lock.json"
test -f "${PARENT_UJ_ROOT}/locks/graph_recipe_lock.json"
```

## 2. Create the immutable all-mapped foundation specification

```bash
python -s "${UB_FULL_WORKTREE}/scripts/create_hcwdl_unified_balanced_full_foundation.py" \
  --parent-homotopy-spec "${PARENT_UJ_SPEC}" \
  --campaign-root "${UB_FULL_FOUNDATION_ROOT}" \
  --project-dir "${UB_FULL_WORKTREE}" \
  --source-commit "${UB_FULL_COMMIT}" \
  --authorize-live-submission \
  --authorization-phrase \
    "AUTHORIZE HCWDL UB FULL3 ALL MAPPED FOUNDATION EXACT SPEC"
```

The emitted role counts are the exact mapped split inventory. Approximate
`2.6M/1M/1M` labels are not used as identities.

## 3. Nonmutating dry run

```bash
python -s "${UB_FULL_WORKTREE}/scripts/submit_hcwdl_unified_balanced_full_campaign.py" \
  --foundation-spec "${UB_FULL_FOUNDATION_ROOT}/foundation_spec.json" \
  --foundation-ledger "${UB_FULL_LEDGER_ROOT}/foundation_dry/submission_ledger.json" \
  --arms-root "${UB_FULL_ARMS_ROOT}" \
  --arm-ledger-root "${UB_FULL_LEDGER_ROOT}/arms_live" \
  --autolaunch-receipt "${UB_FULL_LEDGER_ROOT}/autolaunch_receipt.json" \
  --output "${UB_FULL_LEDGER_ROOT}/campaign_dry.json"
```

This authenticates the complete foundation command plan without changing
Slurm. Arm publication remains deferred because its immutable parent is the
future foundation lock. The same arm creator/submitter has a bounded synthetic
three-arm dry-run test in the repository.

## 4. Submit foundation plus automatic three-arm launch

This is the only full-campaign scheduler mutation command:

```bash
python -s "${UB_FULL_WORKTREE}/scripts/submit_hcwdl_unified_balanced_full_campaign.py" \
  --foundation-spec "${UB_FULL_FOUNDATION_ROOT}/foundation_spec.json" \
  --foundation-ledger "${UB_FULL_LEDGER_ROOT}/foundation_live/submission_ledger.json" \
  --arms-root "${UB_FULL_ARMS_ROOT}" \
  --arm-ledger-root "${UB_FULL_LEDGER_ROOT}/arms_live" \
  --autolaunch-receipt "${UB_FULL_LEDGER_ROOT}/autolaunch_receipt.json" \
  --output "${UB_FULL_LEDGER_ROOT}/campaign_live.json" \
  --execute \
  --authorization-phrase \
    "SUBMIT HCWDL UB FULL3 FOUNDATION THEN THREE ARMS EXACT IDS"
```

The foundation rebuilds full train/validation assignments, couplings,
balanced sidecars, `U000`, `M0paired`, and compact U000 logits. Its measured
resource/endpoint gate runs before training. Once the foundation lock
completes, the autolaunch submits three independent sequential factorized
arms: `C25P75`, `C10P90`, and `C10P75G15`. No final-test job exists.

Show complete names and reasons:

```bash
squeue --me -o "%.18i %.65j %.2t %.10M %R"
```

## 5. Monitoring, cancellation, and recovery

The foundation ledger is under `foundation_live`; after autolaunch, each arm
has its own `arms_live/<ARM>/submission_ledger.json`. Monitor either canonical
spec and ledger with `monitor_hcwdl_unified_balanced_full.py`. Preview exact
IDs with `cancel_hcwdl_unified_balanced_full.py` before using its explicit
execution phrase. Recovery is separately source-pinned or resource-only and
schedules only the authenticated failed/downstream closure.

The initial all-mapped campaign exposed one execution bug: scale calibration
treated the compact `all_rows: true` selection as every raw ROOT entry instead
of every assignment-authenticated mapped identity. If and only if the
assignment lock completed and scale calibration is the first failed task, use
the classified mapped-identity recovery. It starts at scale calibration and
reuses the immutable assignment prefix; it does not rebuild assignments or
change the selected population. The recovery CLI requires:

```text
--execution-repair all_mapped_assignment_identity_filter_v1
--authorization-phrase "AUTHORIZE HCWDL UB FULL3 ALL MAPPED IDENTITY EXECUTION REPAIR"
```

The mapped-identity recovery may expose a separate integration-only miss if
it completes `coupling_lock` and `balanced_config`, then both balanced-sidecar
arrays fail at their first call into the corrected common iterator. Monitor
the *canonical first recovery spec* and its live ledger, then create a
repeated recovery with the original foundation spec as `--scope-spec` and the
first recovery as `--parent-recovery-spec`. The exact authorization is:

```text
--execution-repair balanced_assignment_store_wiring_v1
--authorization-phrase "AUTHORIZE HCWDL UB FULL3 BALANCED ASSIGNMENT WIRING EXECUTION REPAIR"
```

`monitor_hcwdl_unified_balanced_full.py` accepts both campaign specs and
recovery specs; it reads task attestations from `campaign_root` or
`recovery_root`, respectively. This second recovery must start with
`train_balanced` and `validation_balanced`. It reuses the completed coupling
lock and balanced configuration and does not repeat calibration, residual
coupling construction, or the exhaustive coupling audit.

After the recovery foundation lock completes, launch the arms from the
original scientific worktree/commit named by `foundation_spec.json`. The arms
do not execute the repaired preprocessing path, so retaining their original
source binding preserves their registered training semantics. The new
autolaunch must depend on the recovered `foundation_lock` job ID; the old
autolaunch dependency can never be satisfied and should be cancelled by its
exact campaign-bound ID.

Final-test access remains outside this campaign and requires a later,
separately implemented finalist and execution lock.
