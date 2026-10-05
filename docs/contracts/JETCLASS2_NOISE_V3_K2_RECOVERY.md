# NOISE_V3 K2 targeted recovery v1

Authority: the operational recovery amendment in the
[active plan](../plans/JETCLASS2_NOISE_V3_K2_OSCAR_100K_PLAN.md).
New artifacts use `NOISE_V3_K2_RECOVERY_*/v1`, schema 1, canonical content hashes
and `final_test_accessed=false`. Original scientific contracts remain v1.

`SPEC` binds the original spec/ledgers by file and content hash, original exact
jobs, terminal accounting snapshot, completed receipt hashes, incomplete
science task list, accepted preflight hash/environment, new pushed checkout,
source compatibility, sibling output root, and excluded nodes (gpu3001 required).
Every original tracked Python file under src plus the environment shell helpers
must be byte-identical in the original and recovery checkout. New recovery
modules are additive. Original matching/model/data/training source is unchanged.

Creation requires complete authenticated gate/science submission journals and
ledgers. All original jobs must be terminal and absent from the live queue.
Completed scheduler jobs require valid completed receipts; malformed/corrupt
receipts never become silent retries. Every non-science task must be completed.
Failed/cancelled science jobs without receipts are retryable; active/unknown
jobs, completed descendants of missing parents, and an all-complete campaign
are rejected. No metric threshold chooses reuse or retry.

The source campaign remains read-only. Completed artifacts are resolved at
their original paths, not copied, relabelled or symlinked. Replacement artifacts
are relative to the recovery root and carry separate task/source lineage.
Parent receipts and all recorded file hashes are verified before use.
Scientific reports preserve the original campaign identity and add recovery
provenance. Compact banks bind the exact selected teacher report and identities.
Results combine original and replacement report-validation rows with original
HLT_X1_CE and OFFLINE_CE recovery references. Final test stays sealed.

The full replacement graph has only new-job `afterok` dependencies. No old job
ID is used as a scheduler dependency. CPU metadata remains batch; fits/reducers
remain gpu/norm-gpu/L40S at original resources. GPU jobs include
`--exclude=gpu3001` (additional explicit excluded nodes allowed). Worker checks
the exact journal ID, source, allocation, environment, accepted GPU identity,
and node exclusion; a CUDA forward/backward smoke precedes expensive caching.

Live submission requires its immutable dry run, current quiescence of original
jobs, native acceptance, matching installed environment, Slurm test-only checks,
and phrase `AUTHORIZE OSCAR NOISE K2 TARGETED RECOVERY`.
A sibling `.noise_k2_recovery_claims/<original-spec-hash>.json` reserves one
recovery for this source campaign without writing into the original root.
An exclusive recovery submission claim protects ambiguous sbatch outcomes.
Complete ledgers are idempotent; partial/ambiguous submissions fail closed for
exact-ID review. Never delete claims to retry. This v1 does not recursively
recover another recovery or change previously submitted job resources.

`TASK` is published last and binds original/recovery specs, replacement job ID,
original task ID, source commit, parents, result pointers and all new output
bytes. Failed output directories remain for diagnosis; they are not overwritten.
No training resume is claimed: a failed fit restarts with the original cold seed.

CLI: `scripts/noise_k2_recovery.py create|submit|run|results|monitor`.
Queue helper: `scripts/queue_noise_k2_recovery.sh`, dry by default. It requires
PROJECT_DIR, NOISE_K2_RECOVERY_COMMIT, NOISE_K2_SOURCE_SPEC, RECOVERY_ROOT;
optional NOISE_K2_EXCLUDE_NODES defaults to gpu3001. `--execute` requires an
already created and reviewed recovery spec/dry run, never creates one silently.

## Operator workflow

Commit/push the additive recovery files and use a fresh detached checkout on
OSCAR. The main repository is
`/oscar/home/rlyang/hlt_classification/source/HLT_Classification`, not its
`/oscar/home/rlyang/hlt_classification` parent. Keep the original checkout at
`/oscar/home/rlyang/hlt_classification_noise_k2_0ffb4ba5` untouched.

Set these exported variables in a new shell (replace the recovery commit and
checkout with the actual pushed source; the helper validates them):

```bash
export NOISE_K2_SOURCE_SPEC=/oscar/scratch/rlyang/hlt_classification/checkpoints/noise_v3_k2_100k_0ffb4ba5_r1/campaign_spec.json
export NOISE_K2_RECOVERY_COMMIT=FULL_PUSHED_RECOVERY_COMMIT
export PROJECT_DIR=/oscar/home/rlyang/hlt_classification_noise_k2_recovery_COMMIT
export RECOVERY_ROOT=/oscar/scratch/rlyang/hlt_classification/checkpoints/noise_v3_k2_recovery_COMMIT_r1
export NOISE_K2_EXCLUDE_NODES=gpu3001
```

First, dry run only:

```bash
bash "${PROJECT_DIR}/scripts/queue_noise_k2_recovery.sh" --dry-run
```

Creation runs read-only scheduler queries and authenticates file hashes. This
can take several minutes; it does not rerun assignment, matching, cache builds,
training or the native preflight. Inspect `recovery_spec.json`,
`command_plan.json` and `dry_run.json` under RECOVERY_ROOT. For the reported
state, expect these **10** replacement jobs:

1. train_HLT_X3_CE
2. train_OFFLINE_CE
3. reduce_CONCAT_K2_D050
4. train_CONCAT_K2_D025
5. reduce_CONCAT_K2_D025
6. train_CONCAT_K2_D000
7. reduce_CONCAT_K2_D000
8. train_HLT_X1_COMPRESSED
9. aggregate
10. complete

No old training job or old Slurm dependency should appear in this plan. The
first three jobs can run independently. Both controls are required for the
final aggregate, not for proceeding down the ladder.

After reviewing the dry run, explicit live submission is:

```bash
bash "${PROJECT_DIR}/scripts/queue_noise_k2_recovery.sh" --execute
```

This runs Slurm test-only validation for each resource shape, then journals
new exact IDs under RECOVERY_ROOT. It never cancels an old job. If any original
job is still live, creation/submission stops; inspect its exact ID before
deciding what to do. If an sbatch acknowledgement is lost, preserve the claim
and journal and inspect Slurm; do not delete the claim or blindly retry with
another root.

For monitoring or combined saved results, activate the same environment helper
and run the new CLI (the old CLI sees original reports only):

```bash
export JC2_SITE=oscar_l40s
source "${PROJECT_DIR}/sbatch/jetclass2_delphes_common.sh"
python -s "${PROJECT_DIR}/scripts/noise_k2_recovery.py" monitor --spec "${RECOVERY_ROOT}/recovery_spec.json"
python -s "${PROJECT_DIR}/scripts/noise_k2_recovery.py" results --spec "${RECOVERY_ROOT}/recovery_spec.json"
```

Results read saved validation REPORT metrics, not new inference. Offline-gap
recovery becomes available once the OFFLINE_CE replacement completes. The
restored environment must continue to match the original acceptance fingerprint;
this helper neither repairs nor upgrades packages. A new CUDA failure should
be diagnosed separately; the small startup smoke is not a guarantee of GPU
health throughout a full fit.
