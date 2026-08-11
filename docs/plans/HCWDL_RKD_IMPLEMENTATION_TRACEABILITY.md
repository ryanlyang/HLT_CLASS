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
| Controlling dense-descent amendment; 1--6, 12, 14--15, 18--19, 22 | four descents, 86 nodes, exact immediate-teacher graph, cold/warm initialization, schedules and immutable v3 overlay | `src/hlt_classification/scouting/hcwdl_representation_graph.py`, `hcwdl_representation_recipe.py` | `tests/test_hcwdl_representation_graph.py`, `test_hcwdl_representation_recipe.py` |
| 3.1--3.3 | actual v8-prefix/v4 parent import and exact-all-ones CE/KD attestation | `hcwdl_parent_loss.py`, `hcwdl_representation_locks.py`, corrected opt-in path in `engine.py` | `tests/test_hcwdl_parent_loss.py`, `test_hcwdl_representation_locks.py`, `test_hcwdl_parent_authority.py` |
| 7, 17, 28.2 | single-forward ordinary and TOFF surfaces and architecture attestation | `src/hlt_classification/models/scouting_particle_transformer.py`, `models/hcwdl_surfaces.py` | `tests/test_hcwdl_representation_model.py`, `test_hcwdl_representation_locks.py` |
| 8--11, 13--14, 17--18 | jet/set/relation/orthogonality math, fixed kernels, family policy, calibration and ramps | `hcwdl_representation_kernels.py`, `hcwdl_representation_losses.py`, `hcwdl_representation_calibration.py` | `tests/test_hcwdl_representation_math.py`, `test_hcwdl_representation_calibration.py` |
| 16, 20, 21, 29 | logical target banks, signature-bound generations, RAM joins, cleanup and reconstruction | `hcwdl_representation_targets.py`, `hcwdl_representation_target_runtime.py`, `hcwdl_representation_target_recovery.py` | `tests/test_hcwdl_representation_targets.py` |
| 21, 28.3, 31 | versioned JSON contracts, binary envelopes and one canonical route template per contract | `hcwdl_representation_contracts.py`, `hcwdl_representation_artifacts.py`, `hcwdl_representation_layout.py` | `tests/test_hcwdl_representation_contracts.py`, `test_hcwdl_representation_artifacts.py`, `test_hcwdl_representation_layout.py` |
| Controlling dense-descent amendment; 23 | zero-coefficient acceptance retained; retired M5 controls excluded from the active v2 empty registry | `hcwdl_representation_graph.py`, `hcwdl_representation_controls.py` | `tests/test_hcwdl_representation_graph.py`, `test_hcwdl_representation_training.py` |
| 24 | node diagnostics, gap recovery, screen/confirmation aggregates and paired bootstrap | `hcwdl_representation_reporting.py`, `hcwdl_paired_bootstrap.py` | `tests/test_hcwdl_representation_reporting.py`, `test_hcwdl_paired_bootstrap.py` |
| 25 | global population registration, disposition/reservation/claim, label-free prediction and locked join | `hcwdl_shared_final.py`, `hcwdl_final_stream.py`, `hcwdl_representation_final.py` | `tests/test_hcwdl_shared_final.py`, `test_hcwdl_final_stream.py`, `test_hcwdl_representation_final.py` |
| 26 | fail-closed lineage/numerics and finite-poor-result continuation | strict validators plus workflow/runtime dispatch | negative tests throughout the focused suites |
| 27, 30 | resources, measured fixed-size inventory/storage estimate, fixed CLIs, two campaign workers, bounded nonfinal bootstrap workers, runtime binding and preemption | `hcwdl_representation_resources.py`, `hcwdl_representation_task_runtime.py`, `hcwdl_representation_bootstrap.py`, `scripts/build_hcwdl_representation_fixed_size_inventory.py`, `scripts/build_hcwdl_representation_storage_estimate.py`, `scripts/*hcwdl_representation*`, `scripts/*hcwdl_shared_final*`, `sbatch/run_hcwdl_representation_*` | `tests/test_hcwdl_representation_cli.py`, `test_hcwdl_representation_bootstrap.py`, `test_hcwdl_representation_preemption.py` |
| Historical 21, 30--32 | retired recipe-v2 ten-action authority, retained for identity/readability tests but forbidden from authorizing the dense recipe-v3 graph | `hcwdl_representation_nonfinal_acceptance.py`, `hcwdl_representation_validation_proxy.py` | `tests/test_hcwdl_representation_nonfinal_acceptance.py`, `test_hcwdl_representation_validation_proxy.py` |
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
scripts/build_hcwdl_representation_nonfinal_acceptance_action_result.py
scripts/build_hcwdl_representation_usr1_exact_resume_proof.py
scripts/build_hcwdl_representation_validation_proxy_proof.py
scripts/build_hcwdl_representation_production_worker_smoke_proof.py
scripts/build_hcwdl_representation_tigris_action_proof.py
scripts/build_hcwdl_representation_tigris_evidence_bundle.py
scripts/build_hcwdl_representation_tigris_acceptance.py
scripts/build_hcwdl_representation_two_update_acceptance_proof.py
```

These commands publish authenticated evidence from already observed files and
values. They do not launch scheduler work, grant bootstrap/final-role access,
or authorize a pilot.

## Exact node and control registry

Every name below is an immutable registry identity, not a parsed convention.

| Strategy | Root | Cold nodes | Warm nodes |
|---|---|---|---|
| RSET | `RSET_D100` from TOFF | `RSET_D95c`, `RSET_D90c`, ..., `RSET_D0c`, `RSET_M1c` | `RSET_D95w`, `RSET_D90w`, ..., `RSET_D0w`, `RSET_M1w` |
| RREL | `RREL_D100` from TOFF | `RREL_D95c`, `RREL_D90c`, ..., `RREL_D0c`, `RREL_M1c` | `RREL_D95w`, `RREL_D90w`, ..., `RREL_D0w`, `RREL_M1w` |

The active control registry is exactly empty. D100--D5 are nondeployable
teacher intermediates; D0 and terminal M1 are HLT-only deployable nodes.

## Contract checklist

The full strings and common envelope validators live in
`src/hlt_classification/scouting/hcwdl_representation_contracts.py`. The
canonical producer/path/validation-surface mapping lives in
`hcwdl_representation_layout.py` and is tested as an exact set equality with
the contract registry. A `validation_surface` value is an auditable ownership
label, not a promise that every label names a dynamically imported callable;
consumers invoke their typed validators directly.

The migrated authority tuples are exact and intentionally nonparallel:
`HCWDL_REPRESENTATION_PARENT_IMPORT/v3` uses schema 2,
`HCWDL_REPRESENTATION_PARENT_LOSS_ATTESTATION/v3` uses schema 3, and
`HCWDL_REPRESENTATION_RECIPE/v3` uses schema 1. The strengthened
`USR1_EXACT_RESUME_PROOF/v2`, `VALIDATION_PROXY_PROOF/v2`,
`TIGRIS_EVIDENCE_BUNDLE/v2`, and `TIGRIS_ACCEPTANCE/v2` also use common-
envelope schema 1. The latter two freeze the new bounded non-final check
registry rather than expanding their v1 semantics. The parent authority itself
is exactly executable `HCWDL_CAMPAIGN_SPEC/v8` with
`parent_prefix_through_finalist_lock` scope plus primary
`HCWDL_RECIPE/v4`.

- [x] parent and architecture: `PARENT_IMPORT`, `PARENT_LOSS_ATTESTATION`,
  `ARCHITECTURE_ATTESTATION`, `TAP`, `SURFACE_PARITY`
- [x] graph and recipe: `DENSE_DESCENT_GRAPH`, `REPRESENTATION_RECIPE`,
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
  `MODEL_EXTRACTION`
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
  `TIGRIS_ACTION_PROOF`, `TWO_UPDATE_ACCEPTANCE_PROOF`,
  `USR1_DELIVERY_RECEIPT`, `USR1_EXACT_RESUME_PROOF/v2`,
  `VALIDATION_PROXY_BRANCH_ACCESS`, `VALIDATION_PROXY_PROOF/v2`,
  `NONFINAL_ACCEPTANCE_SCHEDULER_EVIDENCE`,
  `PRODUCTION_WORKER_SMOKE_PROOF`,
  `LOCAL_SMOKE_REPORT`, `CACHE_MINIATURE`, `CACHE_MINIATURE_BANK`,
  `TIGRIS_ACCEPTANCE`, `SUBMISSION_AUTHORIZATION`,
  `ACCEPTANCE_BOOTSTRAP`, `NONFINAL_ACCEPTANCE_ACTION_INPUTS`,
  `NONFINAL_ACCEPTANCE_ACTION_ASSEMBLY`,
  `NONFINAL_ACCEPTANCE_EXECUTION_RECEIPT`,
  `ACCEPTANCE_REAL_BATCH_FULL_LOSS`, `NONFINAL_ACCEPTANCE_ACTION_RESULT`,
  `NONFINAL_ACCEPTANCE_AUTHORITY`

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
- [x] all 86 primary dense-descent nodes and the empty v2 control registry represented immutably
- [x] every registered HCWDL-RKD command script has an explicit working-help test
- [x] all four campaign/bootstrap worker scripts have environment, dispatch,
  final-`exec`, role-isolation, and deterministic-worker regressions
- [x] bounded bootstrap spec/worker prefix binds source/worker bytes, exact
  runtime rows, conservative resources, no arrays and no final-role authority
- [x] retired recipe-v2 non-final authority remains covered by identity and
  boundary tests; it is explicitly ineligible for dense recipe-v3 execution
- [ ] implement, version, authorize, and run a dense-descent production-worker
  miniature that exercises the repaired-view D100--D5 path; the old ten-action
  authority and local fixtures cannot substitute for this evidence
- [x] parent-final-state, fail-closed disposition, and terminal exact-ID cancellation producer fixtures
- [x] shared-final registrar race/recovery, capability and label-isolation tests
- [x] complete bounded semantic smoke and full non-submitting command plan:
  295 tasks / 361 expanded rows, 86 primary nodes / zero controls, no final-role
  access, no scheduler mutation, and no production authorization
- [x] complete repository suite after the dense-descent migration: 789 passed,
  6 expected Windows platform skips, 0 failures
- [x] prohibited-pattern and accidental privileged-input audit: no runtime
  `Fresh_check`, `TODO`, `FIXME`, `NotImplementedError`, executable `pass`, or
  deployable privileged-input surface; all 405 Python files parse and the
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
