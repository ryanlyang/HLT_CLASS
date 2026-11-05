# Group A: native HLT compatibility and membership on Hydra

This implements candidates3/5/6 from the
[40-row plan](plans/LUKA_SEVEN_CANDIDATE_IMPLEMENTATION_PLAN.md): learned jet
compatibility J, particle soup P, and intact pairs/triplets G. There are22
selected method/control configurations, plus reused C0/C1. J5 is automatically
reused when its original completed result in the contrastive transfer authenticates,
leaving21 new science fits; otherwise22. No existing
campaign is modified, canceled or reprioritized. Groups B/C remain future work;
this helper cannot release the full seven-candidate campaign.

All fits keep fresh native HLT CE+U100-w5, batch128/full30. The new prediction
heads are training-only. Read the [v1 contract](contracts/LUKA_NEXT_SUPERVISION_HYDRA.md)
for features, sampling and exact losses. Code lives in
[luka_next_supervision_hydra](../src/hlt_classification/luka_next_supervision_hydra/__init__.py).

## Files and data needed

Brown already has the original `bundle_65952ed3`, pinned Python3.10 environment,
contrastive OFFLINE-key delta, original qualified A6000 cross-message spec,
and completed particle-teacher C0/C1 fits. Existing artifacts retain their own
contracts/absolute lineage; new consumers follow allowlisted relocated payloads.
They never follow old SPORC absolute paths on Brown.

The only new physical transfer is a TRAIN-only measured-particle directory plus
the original M0HLT checkpoint. It contains raw physical p4/charge/category,
tracking in declared mm and validity for HLT/OFFLINE. It does not copy ROOT data,
validation OFFLINE sets, contextual particle targets or another environment.
Its exact bytes/size/hash are printed by export; reserve space based on that
measurement. Do not remove existing shared inputs or source worktrees.

## Commit only the reviewed change set

New runtime paths are `src/hlt_classification/luka_next_supervision_hydra/`,
`scripts/luka_next_supervision_hydra.py`, `scripts/queue_luka_next_supervision_hydra.sh`,
`sbatch/run_luka_next_supervision_hydra.sh` and
`tests/test_luka_next_supervision_hydra.py`. Include this runbook,
`docs/LUKA_NEXT_SUPERVISION_GROUP_A_VALIDATION.md`,
`docs/contracts/LUKA_NEXT_SUPERVISION_HYDRA.md`, both planning documents, and the
README/HANDOFF/plan-index/source-map additions after reviewing their existing
unrelated changes. Use explicit paths, inspect the staged diff, run
`git diff --cached --check`, commit and push. Keep the full40-character commit.
Remote registration requires a clean detached worktree at that pushed source.

## Export additional TRAIN measurements on SPORC

No GPU is needed. This reads the original paired TRAIN source only, with
before/after source checksums and the original prepared conventions. It writes
one new directory; an incomplete directory is preserved and cannot be reused.

```bash
bash <<'BASH'
set -euo pipefail
read -r -p 'Pushed Group A commit (40 characters): ' COMMIT </dev/tty
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
MAIN=/home/ryreu/atlas/HLT_Classification
PROJECT="${MAIN}_next_supervision_${COMMIT:0:8}"
OUT="/home/ryreu/atlas/luka_next_raw_${COMMIT:0:8}"
git -C "${MAIN}" fetch origin main
git -C "${MAIN}" merge-base --is-ancestor "${COMMIT}" origin/main
if [[ -e "${PROJECT}" ]]; then
    test "$(git -C "${PROJECT}" rev-parse HEAD)" = "${COMMIT}"
    test -z "$(git -C "${PROJECT}" status --porcelain --untracked-files=all)"
else
    git -C "${MAIN}" worktree add --detach "${PROJECT}" "${COMMIT}"
fi
ENV_DIR=/home/ryreu/miniconda3/envs/atlas_kd_sporc
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="${ENV_DIR}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
"${ENV_DIR}/bin/python" -u -B -s "${PROJECT}/scripts/luka_next_supervision_hydra.py" export \
    --prepared "${MAIN}/checkpoints/luka_alljets_9e591c7a_r1/prepared" \
    --origin-spec "${MAIN}/checkpoints/luka_u100_pair_correction_c08f6df6_debug_r1/campaign_spec.json" \
    --output "${OUT}" --expected-commit "${COMMIT}"
BASH
```

Retain the printed64-character `manifest_hash`. Copy this new directory to Brown
using rsync/scp with your normal RIT MFA. For example, replace COMMIT8 below with
the actual eight-character prefix on the Brown shell:

```bash
rsync -av --partial --progress \
  ryreu@sporcsubmit.rc.rit.edu:/home/ryreu/atlas/luka_next_raw_COMMIT8/ \
  "$HOME/hlt_u100_pairs/raw_next_COMMIT8/"
```

Individual transferred members authenticate against that manifest before use;
path existence and successful rsync alone do not admit a payload.

## Register a fresh Group A root on Brown

Fetch the pushed commit and create a clean detached worktree using the existing
`$HOME/hlt_u100_pairs/code_65952ed3` Git repository, as in the earlier campaign
runbooks. Call the new worktree `code_next_COMMIT8`. Use the **same pinned Brown
environment** from the completed A6000 parent; Python3.12 is not that environment.

Create a small JSON reuse map with paths to completed original campaign specs:

```json
{
  "C0": "/cs/home/rlyang/hlt_u100_pairs/runs/particle_teacher_ddcdd5fa_a6000_r1/campaign_spec.json",
  "C1": "/cs/home/rlyang/hlt_u100_pairs/runs/particle_teacher_ddcdd5fa_a6000_r1/campaign_spec.json"
}
```

Leave J5 absent to reuse the completed SPORC `mix_l0p3_s0` carried in the existing
contrastive delta. Registration validates its original full30 recipe, seeds,
U100/OFFLINE parents, exported completion proof and saved validation metrics.
It remains labeled **SPORC reused**, not a new Hydra fit; its checkpoint is not
transferred or re-inferred. If no completed J5 exists in the delta, it becomes
a new fit. An optional `"J5": "/absolute/path/to/completed/Hydra/spec.json"`
instead requests a locally completed original-worktree proof. C0/C1 are
mandatory reused references; missing or incompatible proofs stop registration.

Below, set PROJECT/ENV_DIR/RAW_ROOT/RAW_HASH/DELTA_ROOT/DELTA_HASH to the actual
worktree, environment and transferred manifests. The contrastive delta path/hash
are the `delta_root`/`delta_hash` fields of the existing Brown contrastive spec.
The parent path below is the original completed A6000 execution, not a failed
L40 setup. Use a fresh output root; registration never overwrites old claims.

```bash
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="${ENV_DIR}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
BASE="$(cd "$HOME" && pwd -P)/hlt_u100_pairs"
COMMIT="$(git -C "${PROJECT}" rev-parse HEAD)"
"${ENV_DIR}/bin/python" -B -s "${PROJECT}/scripts/luka_next_supervision_hydra.py" create \
  --root "${BASE}/runs/next_A_${COMMIT:0:8}_a6000_r1" \
  --parent-spec "${BASE}/runs/cross_message_199928da_3693bcda_a6000_r1/campaign_spec.json" \
  --raw-root "${RAW_ROOT}" --raw-hash "${RAW_HASH}" \
  --delta-root "${DELTA_ROOT}" --delta-hash "${DELTA_HASH}" \
  --reuse-map "${BASE}/group_A_reuse.json" --expected-commit "${COMMIT}" --quota-gib 4
SPEC="${BASE}/runs/next_A_${COMMIT:0:8}_a6000_r1/campaign_spec.json"
```

Reserve four GiB for outputs **after** environment and raw transfer plus one GiB
filesystem headroom. This is a checked allowance, not a claim that all four GiB
will be used. Source, views, banks, reuse proofs and environment authenticate
before output registration. The old source worktrees must remain available.

## Queue setup, then independently queue science

```bash
bash "${PROJECT}/scripts/queue_luka_next_supervision_hydra.sh" "${SPEC}" setup
```

Review the exact dry run and type only `YES` at the prompt. `lukanx_setup` requests
one RTX A6000,6CPUs,32000MiB, up to12h; no node pin. It verifies installed-Weaver
parity, builds frozen HLT keys, fits TRAIN-only descriptor statistics, checks
full-universe capacity and runs warmed full/partial production miniatures.
Separate sampler/source/interaction paths are measured; scalar weights and
replicate variants share only identical compute/support paths. Quality is not
an admission gate. Resource failures preserve measured diagnostics.

Within each six-CPU allocation, four CPU-only sampling workers share immutable
TRAIN arrays. CUDA stays in the parent, with P9 mining scored in query batches.
These are processes inside one job, not scheduler lanes or additional GPU jobs.
The setup measures their IPC/runtime and includes child memory in admission.

After setup is COMPLETED and its acceptance receipt validates:

```bash
bash "${PROJECT}/scripts/queue_luka_next_supervision_hydra.sh" "${SPEC}" science
```

This is an explicit Group A-only release. All new rows submit in one pass,
filtered from the global40-row order:

```text
P1 G1 J1 P2 G2 J2 P11 G8 J6 P3 G5 [J5 if new]
P5 G3 J3 P6 G4 J4 P7 P8 P9 P4
```

C0/C1 appear as reused comparison rows and have no job. Each fit requests one
A6000/6CPUs/32000MiB with measured walltime up to48h. There are **no science
dependencies, holds, family lanes, arrays/concurrency caps or priority/nice
changes**. Submission order encourages mixed coverage; Slurm decides starts.
Once the helper returns the complete exact-ID ledger, no login watcher remains
responsible for later dispatch. There is no dependent report job.

The combined all-seven release remains unavailable until B/C are implemented
and qualified. It must use the original full interleaving, not this filtered
order while claiming all-candidate coverage.

## Saved results, progress and recovery

```bash
"${ENV_DIR}/bin/python" -B -s "${PROJECT}/scripts/luka_next_supervision_hydra.py" results --spec "${SPEC}"
"${ENV_DIR}/bin/python" -B -s "${PROJECT}/scripts/luka_next_supervision_hydra.py" progress --spec "${SPEC}"
"${ENV_DIR}/bin/python" -B -s "${PROJECT}/scripts/luka_next_supervision_hydra.py" inspect-submission --spec "${SPEC}"
```

These read saved files/accounting without inference or writes. Completed results
authenticate receipts/probabilities/metrics and same-seed control deltas; progress
labels running epochs provisional and reports the last emitted batch marker.
Failed, pending, incomplete and unconfirmed rows are retained. Historical recovery
is `(AUC-M0HLT)/(OFFLINE-M0HLT)*100`; original50k overlaps validation, not a test.

A complete stage can be called again idempotently. An exclusive live claim or
partial journal without a completed ledger stops automatic submission. Inspect
exact accepted IDs and accounting, especially a potentially lost sbatch response;
do not delete claims or blindly rerun. No automatic partial recovery/cancellation
is implemented. Preserve failed diagnostics and register reviewed recovery
separately. Never cancel by an ambiguous job-name prefix.
