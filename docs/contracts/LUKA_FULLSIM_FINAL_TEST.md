# Luka FullSim locked final-test contracts v1

Authority: [final-test plan](../plans/LUKA_FULLSIM_FINAL_TEST_PLAN.md).
This is an additive namespace, `LUKA_FULLSIM_FINAL_TEST_*/v1`, not a weakening
or relabelling of ordinary science-v1 or prepared-v1 artifacts.

- SOURCE_REUSE binds distinct science/execution snapshots. Every old tracked
  src file stays byte-identical; exactly four final_test modules are added.
- FINALIST_LOCK binds all twelve validation-selected checkpoints, scientific
  reports, validation summaries, completed campaign, prepared data and original
  file-disjoint split. No test particles are read to create locks.
- EXECUTION_LOCK binds that finalist lock, full protocol, exact clean pushed
  execution source, resource requests and original artifact locators. All
  controls/oracles remain labelled. No implicit single-finalist claim is made.
- PLAN and SUBMISSION_CLAIM bind two exact Slurm jobs and their dependency. The
  shared exact-DAG helper retains its own versioned submission ledger/journal.
  Dry runs only write metadata/check scheduler feasibility; they cannot unseal.
- TRAIN_SMOKE verifies the new reconstruction against existing TRAIN assignments
  for all seven views, installed/native parity and two identical inference
  replays for each of twelve selected checkpoints. It does not fit a model.
- ACCESS_CLAIM is atomically created under the old science root's new
  `final_test_access/` directory, before test reads. Its job/execution/finalist
  binding and authenticated passing TRAIN_SMOKE reference permit exactly one
  process-bound reader/cache capability. Missing,
  stale, cross-job or interrupted claims fail closed. Ordinary readers retain
  their old test prohibition. `final_test_accessed=true` on this claim means
  authorization consumed conservatively before reading, not completed inference.
- TASK receipts publish last, hash every output and bind both locks and claim.
  Evaluation publishes a common ordered population NPZ and twelve T=1 FP32
  probability NPZs, not TRAIN teacher banks or model-input caches. Whole test
  files are rehashed before/after bounded selected-entry reads. CPU reporting
  rechecks membership, labels, probabilities, metrics and all receipt hashes.
- RESULTS includes every frozen row, validation/test metrics, test-control
  recoveries, and paired file-bootstrap intervals for the registered bridge
  contrasts. A failed/partial model cannot be silently omitted. Poor results
  do not fail publication. No test-selected winner or retraining operation exists.

Test feature arrays are RAM-only, identically shared per coordinate, preserve
native D000 bytes, and use original physical/matching/view/input kernels. The
new final-test caches reuse ordinary tensor batching only through a distinct
capability-checked class; they never claim to be training/validation data.
D000/D000 aliases one native cache. Model forward receives only features,
vectors and mask, never identity/label/file/assignment columns.

Resources reuse accepted SPORC 6 CPU / 256000 MiB / A100 identity/environment;
1386 minutes is a conservative whole-evaluation limit, not measured test timing.
Cache growth including copies/temporaries is limited to 15% of host allocation;
runtime peak host/GPU checks retain 85%/90% ceilings. No support truncation or
adaptive batch/model/precision change is made after inspecting test particles.

Bootstrap uses paired source-file resampling, 200 draws with seed 20261012,
weighted tie-correct OVR AUC and accuracy differences. Missing-class draws are
reported invalid for AUC, not silently redrawn. Intervals do not include training
seed variability, model-selection correction or proven event-level independence.
The original test set is consumed by this fixed comparison; later test-driven
changes require a new independent holdout for confirmatory claims.
Historical science receipts retain `final_test_accessed=false` as a statement
about their original creation, not the current holdout state. After this stage,
the new access claim/results record consumption; do not interpret an old science
summary's "test untouched" footer as permission to unseal again.
