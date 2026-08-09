# HCWDL-RKD implementation traceability

This ledger maps the requirements in
[`HCWDL_MATCHING_FREE_REPRESENTATION_KD_ASCENTS.md`](HCWDL_MATCHING_FREE_REPRESENTATION_KD_ASCENTS.md)
to repository code and local evidence. The active plan remains authoritative.
A checked local item means that the implementation and focused local evidence
exist; it does not mean that installed-Weaver or genuine Tigris acceptance has
run.

## Plan-section ownership

| Plan sections | Requirement | Implementation surface | Focused evidence |
|---|---|---|---|
| 1--6, 12, 14--15, 18--19, 22 | four ascents, 24 nodes, exact teacher graph, cold/warm initialization, schedules and immutable overlay | `src/hlt_classification/scouting/hcwdl_representation_graph.py`, `hcwdl_representation_recipe.py` | `tests/test_hcwdl_representation_graph.py`, `test_hcwdl_representation_recipe.py` |
| 3.1--3.3 | actual parent import and class-weighted CE/unweighted-KD attestation | `hcwdl_parent_loss.py`, `hcwdl_representation_locks.py`, corrected opt-in path in `engine.py` | `tests/test_hcwdl_parent_loss.py`, `test_hcwdl_representation_locks.py` |
| 7, 17, 28.2 | single-forward ordinary and TOFF surfaces and architecture attestation | `src/hlt_classification/models/scouting_particle_transformer.py`, `models/hcwdl_surfaces.py` | `tests/test_hcwdl_representation_model.py`, `test_hcwdl_representation_locks.py` |
| 8--11, 13--14, 17--18 | jet/set/relation/orthogonality math, fixed kernels, family policy, calibration and ramps | `hcwdl_representation_kernels.py`, `hcwdl_representation_losses.py`, `hcwdl_representation_calibration.py` | `tests/test_hcwdl_representation_math.py`, `test_hcwdl_representation_calibration.py` |
| 16, 20, 21, 29 | logical target banks, signature-bound generations, RAM joins, cleanup and reconstruction | `hcwdl_representation_targets.py`, `hcwdl_representation_target_runtime.py`, `hcwdl_representation_target_recovery.py` | `tests/test_hcwdl_representation_targets.py` |
| 21, 28.3, 31 | versioned JSON contracts, binary envelopes and one canonical route template per contract | `hcwdl_representation_contracts.py`, `hcwdl_representation_artifacts.py`, `hcwdl_representation_layout.py` | `tests/test_hcwdl_representation_contracts.py`, `test_hcwdl_representation_artifacts.py`, `test_hcwdl_representation_layout.py` |
| 23 | zero coefficient, jet-only, no-relation and two within-class shuffled controls | `hcwdl_representation_graph.py`, `hcwdl_representation_controls.py` | `tests/test_hcwdl_representation_controls.py`, `test_hcwdl_representation_training.py` |
| 24 | node diagnostics, gap recovery, screen/confirmation aggregates and paired bootstrap | `hcwdl_representation_reporting.py`, `hcwdl_paired_bootstrap.py` | `tests/test_hcwdl_representation_reporting.py`, `test_hcwdl_paired_bootstrap.py` |
| 25 | global population registration, disposition/reservation/claim, label-free prediction and locked join | `hcwdl_shared_final.py`, `hcwdl_final_stream.py`, `hcwdl_representation_final.py` | `tests/test_hcwdl_shared_final.py`, `test_hcwdl_final_stream.py`, `test_hcwdl_representation_final.py` |
| 26 | fail-closed lineage/numerics and finite-poor-result continuation | strict validators plus workflow/runtime dispatch | negative tests throughout the focused suites |
| 27, 30 | resources, measured fixed-size inventory/storage estimate, fixed CLIs, two campaign workers, bounded nonfinal bootstrap workers, runtime binding and preemption | `hcwdl_representation_resources.py`, `hcwdl_representation_task_runtime.py`, `hcwdl_representation_bootstrap.py`, `scripts/build_hcwdl_representation_fixed_size_inventory.py`, `scripts/build_hcwdl_representation_storage_estimate.py`, `scripts/*hcwdl_representation*`, `scripts/*hcwdl_shared_final*`, `sbatch/run_hcwdl_representation_*` | `tests/test_hcwdl_representation_cli.py`, `test_hcwdl_representation_bootstrap.py`, `test_hcwdl_representation_preemption.py` |
| 28.5--28.6 | dedicated trainer, sequence resume, AUC-first selection and HLT-only extraction | `hcwdl_representation_training.py`, `hcwdl_representation_resume.py` | `tests/test_hcwdl_representation_training.py`, `test_hcwdl_representation_resume.py` |
| 30.3 | symbolic dry run, ledger materialization, append-only monitoring, recovery chain and exact cancellation | `hcwdl_representation_campaign.py`, `hcwdl_representation_recovery.py` | `tests/test_hcwdl_representation_campaign.py`, `test_hcwdl_representation_recovery.py` |
| 32--33 | Blocks A--G and locally testable Block-H paths | campaign adapters, production adapters, synthetic final fixtures and local smoke | focused campaign/task-runtime/smoke tests; final integrated audit recorded in `docs/HANDOFF.md` |
| 34--36 | frozen values and scoped scientific claims | recipe/graph validators, runbook and this ledger | recipe/hash tests and documentation audit |

The nonexecuting resource/evidence producer inventory is explicit:

```text
scripts/build_hcwdl_representation_fixed_size_inventory.py
scripts/build_hcwdl_representation_storage_estimate.py
scripts/build_hcwdl_representation_scheduler_evidence.py
scripts/build_hcwdl_representation_miniature_evidence.py
scripts/build_hcwdl_representation_resource_profile.py
scripts/build_hcwdl_representation_usr1_exact_resume_proof.py
scripts/build_hcwdl_representation_validation_proxy_proof.py
scripts/build_hcwdl_representation_production_worker_smoke_proof.py
scripts/build_hcwdl_representation_tigris_action_proof.py
scripts/build_hcwdl_representation_tigris_evidence_bundle.py
scripts/build_hcwdl_representation_tigris_acceptance.py
```

These commands publish authenticated evidence from already observed files and
values. They do not launch scheduler work, grant bootstrap/final-role access,
or authorize a pilot.

## Exact node and control registry

Every name below is an immutable registry identity, not a parsed convention.

| Strategy | Cold nodes | Warm nodes | Representation source by rung |
|---|---|---|---|
| RSET | `RSET_M1c`, `RSET_M2c`, `RSET_M3c`, `RSET_M4c`, `RSET_M5c`, `RSET_M6c` | `RSET_M1w`, `RSET_M2w`, `RSET_M3w`, `RSET_M4w`, `RSET_M5w`, `RSET_M6w` | D0, D25, D50, D75, D100, TOFF |
| RREL | `RREL_M1c`, `RREL_M2c`, `RREL_M3c`, `RREL_M4c`, `RREL_M5c`, `RREL_M6c` | `RREL_M1w`, `RREL_M2w`, `RREL_M3w`, `RREL_M4w`, `RREL_M5w`, `RREL_M6w` | D0, D25, D50, D75, D100, TOFF |

The distinct four-row control registry contains exactly:

1. `RSET_M5c_JET_ONLY_REP`;
2. `RREL_M5c_NO_REL_REP`;
3. `RSET_M5c_WITHIN_CLASS_SHUFFLED_REP`;
4. `RREL_M5c_WITHIN_CLASS_SHUFFLED_REP`.

## Contract checklist

The full strings and common envelope validators live in
`src/hlt_classification/scouting/hcwdl_representation_contracts.py`. The
canonical producer/path/validation-surface mapping lives in
`hcwdl_representation_layout.py` and is tested as an exact set equality with
the contract registry. A `validation_surface` value is an auditable ownership
label, not a promise that every label names a dynamically imported callable;
consumers invoke their typed validators directly.

- [x] parent and architecture: `PARENT_IMPORT`, `PARENT_LOSS_ATTESTATION`,
  `ARCHITECTURE_ATTESTATION`, `TAP`, `SURFACE_PARITY`
- [x] graph and recipe: `ASCENT_GRAPH`, `REPRESENTATION_RECIPE`,
  `KERNEL_RESOURCES`, `CONTROL_REGISTRY`, `ZERO_COEFFICIENT_ACCEPTANCE`,
  `SHUFFLE_MAP`
- [x] target lifecycle: `TARGET_FORWARD_SPEC`,
  `TARGET_EXECUTION_ATTESTATION`, `TARGET_LOGICAL_BANK`,
  `TARGET_CONSUMER_REGISTRY`, `TARGET_BUILD_INTENT`, `TARGET_GENERATION`,
  `TARGET_SHARD`, `TARGET_MANIFEST`, `TARGET_CLEANUP_AUTHORIZATION`,
  `TARGET_CLEANUP_COMPLETION`, `TARGET_RECOVERY_PLAN`
- [x] training: `GRADIENT_CALIBRATION`, `CALIBRATION_SELECTION`,
  `GRADIENT_CALIBRATION_MANIFEST`, `DIAGNOSTIC_BATCH`,
  `NUMERICAL_ACCEPTANCE`, `SMOKE_PROBE`, `RESUME_STATE`,
  `TRAINING_REPORT`, `SELECTED_TRAINING_CHECKPOINT`,
  `FINAL_TRAINING_CHECKPOINT`, `CHECKPOINT_SELECTION`,
  `DEPLOYABLE_EXTRACTION`
- [x] reporting: `PAIRED_BOOTSTRAP`, `SCREEN_AGGREGATE`,
  `CONFIRMATION_REGISTRY`, `CONFIRMATION_RUN`, `CONFIRMATION_AGGREGATE`,
  `VALIDATION_ONLY_AGGREGATE`, `FINAL_AGGREGATE`
- [x] shared final boundary: `FINAL_DISPOSITION`, `PARENT_FINAL_STATE`,
  `SHARED_IMMUTABLE_BINARY_ENVELOPE`, `SHARED_FINAL_POPULATION`,
  `SHARED_FINAL_POPULATION_DISJOINTNESS`, `SHARED_FINAL_EXPOSURE_LEDGER`,
  `SHARED_LEGACY_FINAL_EXPOSURE`,
  `SHARED_FINAL_POPULATION_REGISTRATION`, `SHARED_FINAL_RESERVATION`,
  `SHARED_FINAL_LEGACY_CANCELLATION`, `FINAL_ASSIGNMENT_SPEC`,
  `PRETRAINING_FINALIST_POLICY`, `FINALIST_LOCK`,
  `SHARED_FINAL_TASK_REGISTRY`, `SHARED_FINAL_EXECUTION_CLAIM`,
  `SHARED_FINAL_ROLE_CAPABILITY`, `SHARED_FINAL_RECOVERY_PLAN`,
  `SHARED_FINAL_ROW_SELECTION`, `SHARED_FINAL_LABEL_ESCROW`,
  `SHARED_FINAL_ASSIGNMENT_SHARD`, `SHARED_FINAL_BRANCH_ACCESS`,
  `SHARED_FINAL_ASSIGNMENT_AUDIT`, `SHARED_FINAL_DATA_ATTESTATION`,
  `EXECUTION_LOCK`, `FINAL_PREDICTION_SPEC`, `FINAL_EVALUATION`,
  `PREDICTION_SHARD`, `PREDICTION_MANIFEST`, `METRIC_JOIN`
- [x] orchestration and acceptance: `CAMPAIGN_SPEC`, `COMMAND_PLAN`,
  `RUNTIME_BINDING`, `RUNTIME_PREREQUISITES`, `RUNTIME_DRY_RUN_AUDIT`,
  `WORKER_RUNTIME_MEASUREMENT`, `EXECUTABLE_CANDIDATE_AUDIT`,
  `SUBMISSION_EVENT`,
  `SUBMISSION_LEDGER`, `RECOVERY_SUBMISSION_LEDGER`, `MONITOR_REPORT`,
  `RESOURCE_PROFILE`, `STORAGE_ESTIMATE`, `FIXED_SIZE_INVENTORY`,
  `SCHEDULER_EVIDENCE`, `MINIATURE_EVIDENCE`, `TIGRIS_EVIDENCE_BUNDLE`,
  `TIGRIS_ACTION_PROOF`, `USR1_EXACT_RESUME_PROOF`,
  `VALIDATION_PROXY_PROOF`, `PRODUCTION_WORKER_SMOKE_PROOF`,
  `LOCAL_SMOKE_REPORT`, `CACHE_MINIATURE`, `CACHE_MINIATURE_BANK`,
  `TIGRIS_ACCEPTANCE`, `SUBMISSION_AUTHORIZATION`,
  `ACCEPTANCE_BOOTSTRAP`

The selected/final training-envelope, zero-coefficient, confirmation-run, and
validation-only aggregate identities make plan-required publication boundaries
explicit; they do not broaden an old HCWDL artifact.

## Local evidence checklist

- [x] contract registry, canonical routes, graph serialization and negative tests
- [x] legacy PMARD default-loss compatibility and corrected parent-loss tests
- [x] ordinary and TOFF single-forward surface tests with synthetic installed-interface doubles
- [x] strict HLT-only extraction and fresh-head warm-start tests
- [x] kernel, family, relation, analytic-gradient, invariance and FP32 tests
- [x] calibration support, one-forward, nonmutation and barrier tests
- [x] bounded ordinary/TOFF target build/load/cleanup/rebuild tests
- [x] sequence-resume crash tests and exact training continuation fixtures
- [x] all 24 primary nodes and four controls represented in immutable registries
- [x] every registered HCWDL-RKD command script has an explicit working-help test
- [x] all four campaign/bootstrap worker scripts have environment, dispatch,
  final-`exec`, role-isolation, and deterministic-worker regressions
- [x] bounded bootstrap spec/worker prefix binds source/worker bytes, exact
  runtime rows, conservative resources, no arrays and no final-role authority
- [ ] execute dedicated nonfinal two-update RSET/RREL and USR1 exact-resume
  actions; the action-proof contracts do not substitute for unrun evidence and
  the current full-loss bootstrap probe explicitly performs no optimizer step
- [x] parent-final-state, fail-closed disposition, and terminal exact-ID cancellation producer fixtures
- [x] shared-final registrar race/recovery, capability and label-isolation tests
- [x] complete bounded semantic smoke and full non-submitting command plan:
  86 tasks / 152 expanded rows, 24 primary nodes / four controls, no final-role
  access, no scheduler mutation, and no production authorization
- [x] complete repository suite and `git diff --check`: 656 passed, two
  Windows-only POSIX-shell checks skipped, and only 14 pre-existing
  Matplotlib/PyParsing warnings
- [x] prohibited-pattern and accidental privileged-input audit: no runtime
  `Fresh_check`, `TODO`, `FIXME`, `NotImplementedError`, executable `pass`, or
  deployable privileged-input surface; all 389 Python files parse and the
  untracked-file whitespace scan is clean
- [x] documentation, runbook, traceability, CLI inventory and donor-source-map update
- [ ] authoritative installed-Weaver CPU parity — unavailable in the local environment unless Weaver is installed
- [ ] genuine Tigris worker/cache/USR1 miniature, measured resources, clean pushed source and explicit pilot authorization — intentionally pending and prohibited by the local mandate

Local synthetic surface doubles and planning adapters are evidence for local
failure topology only. They cannot be cited as installed-Weaver parity, real
ROOT execution, Tigris acceptance, resource measurement, pilot authorization,
or final-test use.

The retained nonauthorizing closure artifacts are under the ignored
`checkpoints/hcwdl_rkd_local_verification/closure_20260809/` directory. The
local-smoke/scientific-probe content hashes are respectively
`0ed46019c7c62d9e997ea843051cfb85e1d6cb824b80e0e072445698dac69f67`
and `966491893d4b7456dd39d72a29e5bcdf03cde71fdd60a424826a5ce3990ecbba`.
The command-plan/runtime-binding/dry-run-audit hashes are respectively
`86728de85232c914de92eac751146908025077628c3cee7a3406a67482ed05ea`,
`1696ca5e2ea38c5929061b1757011c427a3218486ac27ad762cf71fcae9df7ec`,
and `38d5f1851a887c6d9354f2580931d4f7440e0cc18c74e9c1b865804638fe608c`.
Every artifact records that it is local/nonmutating and cannot authorize a
Tigris or pilot action.
