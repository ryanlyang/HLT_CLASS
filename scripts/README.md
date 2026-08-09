# Command-Line Entry Points

Scripts in this directory will remain thin wrappers around
`src/hlt_classification`.

Implemented in Transfer Block 2:

- `preflight_data.py`: strict recursive schema and per-class capacity audit;
- `build_splits.py`: deterministic balanced construction and atomic immutable
  publication of the authenticated split manifest.

The preflight's diagnostic skip mode always reports itself as
non-production-eligible. The split builder has no unreadable-file skip path.

Implemented in Transfer Block 4:

- `build_offline_cache.py`: build or resume one bounded immutable offline cache;
- `build_hlt_cache.py`: deterministically degrade an authenticated offline
  parent into bounded HLT-v3 shards;
- `audit_hlt_cache.py`: authenticate all HLT shards and optional exact profile,
  replica, and degradation lineage.

Both builders publish only complete deterministic shards, require an explicit
source-snapshot SHA-256, and provide `--max-new-shards` for focused
interruption/resume validation. An intentionally incomplete build exits with
status 3 and has no final manifest. The HLT CLI emits progress as JSON lines.

Implemented out of order for Transfer Block 5:

- `validate_weaver_parity.py`: standalone installed-Weaver FP32 comparison of
  logits, feature-input gradients, parameter gradients, masks, and state
  dictionaries. The auxiliary Lorentz-input derivative is compared by exact
  finite/NaN/+Inf/-Inf topology because installed Weaver produces nonfinite
  values there even for all-valid inputs; production training does not
  differentiate that input.

It requires Weaver and PyTorch. On Tigris, activate `atlas_kd_tigris`, disable
the user site, and run:

```bash
PYTHONNOUSERSITE=1 python -s scripts/validate_weaver_parity.py --device cpu
```

The optional `--bf16-smoke --device cuda` path checks only runtime finiteness;
it is never a substitute for the default FP32 equivalence attestation.

Implemented in Transfer Block 6:

- `train_part.py`: fixed-budget canonical ParT training with exact resume;
- `evaluate_part.py`: ordered label-free inference followed by an
  identity-bound label join and frozen metrics.

Training configuration is a versioned JSON object. Poor performance never
changes the exit status or remaining update budget; invalid or nonfinite
execution fails closed.

Implemented in Transfer Block 7:

- `create_campaign.py`: capture a clean Git source snapshot and create one
  immutable smoke or explicitly authorized production specification;
- `validate_campaign.py`: revalidate the campaign and active source;
- `measure_storage.py`: bind storage headroom and task requests to a campaign;
- `capture_slurm_resources.py`: derive immutable elapsed-time, MaxRSS, CPU,
  and campaign-byte evidence from exact successful smoke job IDs;
- `capture_runtime_environment.py`: publish source-bound Python, package,
  CUDA, and GPU identity from the real training worker;
- `submit_campaign.py`: dry-run, failure-simulate, smoke-submit, or gated
  full-production submission using exact numeric Slurm job IDs;
- `assemble_submission_journal.py`: recover the exact already-submitted IDs
  if `sbatch` fails partway through constructing an initial or resume graph;
- `monitor_campaign.py`: query exact campaign IDs and authenticate every
  attested file before declaring a task reusable;
- `resume_campaign.py`: create a new failed-node-and-descendant recovery graph;
- `run_campaign_task.py`: shared production task executor used by the thin
  Slurm workers.

Every submit, monitor, resume, and worker entry path validates the active
clean source before checking reusable artifacts.

Full production requires both the storage measurement and its successful
smoke resource-evidence parent. The measured request table is passed into the
actual `sbatch` commands rather than merely validated and ignored.
Every successful `sbatch --parsable` return is durably journaled before the
next job is submitted, so a submit-side outage cannot lose already-created
job IDs or cause an ambiguous duplicate submission.

PRAD campaign foundations:

- `build_prad_splits.py`: exact three-role 500k/150k/500k split, seed 1337;
- `audit_prad_data.py`: bounded real paired-view and Hungarian-match audit;
- `build_prad_paired_cache.py`: immutable paired offline/HLT view shards;
- `build_prad_targets.py`: compact match indices and exclusive-C/A targets.
- `validate_prad_runtime.py`: installed-Weaver PRAD preflight for zero-gate
  parity, HLT-only inference, relation/gate gradients, teacher freezing, and
  the relation-only Stage-A and mixed-freeze Stage-B CUDA backward surfaces;
- `fit_prad_statistics.py`: train-only input moments and semantic positive weights;
- `train_prad_reference.py`: E0 baseline and E4 logit-KD reference graphs;
- `train_prad_teacher.py`: from-scratch offline relation teacher;
- `train_prad_student.py`: E2/E3/E5--E10 and configuration variants;
- `cache_prad_teacher_outputs.py`: immutable frozen-teacher jet outputs, with
  optional float16 dense relation/bias tensors and test access requiring the
  execution lock;
- `evaluate_prad.py`: HLT-only label-free inference, metrics, and validation
  resource benchmark;
- `report_prad.py`: locked five-seed CSV, bootstrap, plot, and report assembly;
- `create_prad_campaign.py`, `submit_prad_campaign.py`, and
  `run_prad_task.py`: immutable PRAD campaign construction and execution;
- `monitor_prad_campaign.py` and `resume_prad_campaign.py`: exact-ID state,
  attestation validation, and failed-descendant recovery;
- `cancel_prad_campaign.py`: exact ledger-ID cancellation only;
- `write_prad_resource_requests.py` and `capture_prad_resources.py`: reviewed
  production requests bound to completed smoke `sacct` evidence;
- `measure_prad_storage.py`: conservative production peak and free-space gate
  bound to the same smoke/source evidence.

The campaign specification uses `prad_minimum_durable_storage_v1`: workers
rebuild paired views, structural targets, and teacher outputs under
`$SLURM_TMPDIR` (or `/dev/shm`) and remove them after each task. Completed runs
retain optimizer-free `selected_model.pt` and `final_model.pt`; `last.pt` is a
rolling exact-resume file and is removed after successful completion.

The audit and minimum-storage smoke must run against the real Tigris data
before any full training submission. Standalone partial cache builds still
exit 3 and never publish a complete manifest; they are diagnostic surfaces,
not durable nodes in the minimum-storage campaign.

See [`docs/PRAD_RUNBOOK.md`](../docs/PRAD_RUNBOOK.md) for the ordered Tigris
commands and production authorization boundary.

PMARD/Scouting commands are configuration-driven wrappers over
`hlt_classification.scouting`: `validate_scouting_data.py`,
`build_scouting_splits.py`, `audit_scouting_features.py`, matcher and Weaver
validation, `validate_fitted_strict_matcher.py` for the immutable selective
matcher bundle, `build_pmard_row_selection.py` for deterministic proportional
campaign membership, `build_pmard_assignment_cache.py` for resumable compact
per-source matching and manifest finalization, the Scouting teacher/student/evaluation
commands, and the PMARD
create/submit/monitor/resume/cancel workers. `dry_run_pmard_campaign.py`
renders the complete production graph without submission;
`capture_pmard_evidence.py` binds a completed smoke ledger to exact resource,
RAM, I/O, GPU, and durable-storage measurements. See
[`docs/PMARD_RUNBOOK.md`](../docs/PMARD_RUNBOOK.md). The smoke DAG never reads
the scientific final-test role. The source audit streams scalar selection and
label branches to authenticate per-file class counts; the split builder uses
an exact 60/20/20 multiclass minimax allocation of whole files and fails if
any realized class fraction misses its target by more than 0.02. All ordinary
smoke training commands consume the same authenticated 4,096-row-per-role
selection; the one registered
`ram_cache_miniature` additionally uses `--cache-teacher-targets` to validate
bounded teacher-logit caching and HLT-only identity joins. Pilot mode registers
exact 300k/100k/100k membership and sealed final-role assignments. A selected
`alpha=0` remains executable through representation and generation stages via
an in-memory HLT-to-privileged identity alias; it never triggers offline reads.
`import_pmard_pilot_prefix.py` is a narrowly scoped, hard-link-only recovery
worker for the authenticated argv-string-normalization, canonical
assignment-root, selective matched-token identity-scope, and endpoint-eligible
cache corrections. It proves source compatibility, reuses the independent HLT
budget and temperature training, and emits a monitor that rebuilds assignments
and dependent locks before submitting teachers. It cannot import failed
teachers or later outputs.

The supplemental T100 transfer study uses
`create_pmard_t100_kd_sweep.py`, `build_pmard_t100_kd_targets.py`,
`train_pmard_t100_kd_sweep.py`, `aggregate_pmard_t100_kd_sweep.py`, and
`submit_pmard_t100_kd_sweep.py`. It binds a completed pilot prefix, computes
T0/T100 logits once, runs an uncapped 36-row HLT-only
weight/temperature/exposure array, and aggregates validation without
final-test access. The exact
separate-worktree launch procedure is in `docs/PMARD_RUNBOOK.md`.

The paired optimizer/cadence follow-up uses
`create_pmard_kd_followup.py`, `train_pmard_kd_followup.py`,
`aggregate_pmard_kd_followup.py`, and `submit_pmard_kd_followup.py`. It is
created only from a complete authenticated T100-sweep aggregate, reuses that
sweep's compact T0/T100 logit cache, and reruns matched CE-only, T0 self-KD,
and selected T100 KD rows with validation after every train pass. It runs both
the square-root exposure-scaled and original fixed learning rates at 20/40/60
passes; no matcher, repaired view, or test-role job is registered.

The explicitly authorized all-model exploratory test comparison uses
`create_pmard_exploratory_test.py`, `run_pmard_exploratory_test_task.py`, and
`submit_pmard_exploratory_test.py`. It freezes all 36 T100-sweep and all 27
distinct completed schedule-follow-up checkpoints before access, converts the
pilot's 100,000 `final_test` jets into an exploratory comparison set,
evaluates an uncapped `0-62` HLT-only array, and publishes metrics JSON only. It never writes
per-jet predictions, and its reports permanently forbid confirmatory claims
from this consumed holdout.

HCWDL commands package the high-coverage completion-shell matcher and run the
fixed Shell Exact cold/warm ladder. The matcher/assignment surface is
`validate_highcov_resources.py`, `build_highcov_assignment_shard.py`,
`finalize_highcov_assignments.py`, and `audit_highcov_assignments.py`.
Recipe, resource, submission, and human diagnostic gates are built by
`build_hcwdl_recipe.py`, `build_hcwdl_resource_profile.py`,
`build_hcwdl_submission_authorization.py`, and
`build_hcwdl_endpoint_ack.py`. Campaign creation/submission/dry-run,
monitor/resume/cancel, node/qualifier/control training, cache miniature,
selection, final evaluation, and aggregation each have a thin `*hcwdl*` CLI.
`run_hcwdl_local_smoke.py` executes the complete 23-node graph on bounded
synthetic tensors without final-test access. Live submission is two-phase:
`submit_hcwdl_campaign.py` stops after the six fixed qualifier jobs; after
their exact reports are inspected and the acknowledgement is published,
`continue_hcwdl_campaign.py` validates the entire prefix and submits the lock
plus downstream ladder. Slurm holds are not used. Executable specs default to
and enforce the exact checkout invoking campaign creation, so concurrent
campaigns use dedicated Git worktrees. No executable pilot/production spec can be made from the bundled
test-only recipe or unmeasured planning resources. After a real miniature,
create a locked non-live candidate spec with the measured resource profile,
inspect its dry run, and pass that candidate to
`build_hcwdl_submission_authorization.py`; authorization v3 binds the exact
resource requests and `HCWDL_COMMAND_PLAN/v2` hash. The first bounded smoke
may use its explicitly authorized conservative bootstrap requests without
claiming measurement; pilot/production candidates require measured profiles.
Recreate the live spec with the same root,
recipe, profile, and authorization. Any command-path, resource, task, or
lineage difference fails closed.
An authorized `HCWDL_RECIPE/v3` is built with `--train-row-selection`; the
builder derives the exact square-root inverse-frequency weights from that
artifact and binds both its hash and its 15 train counts. The campaign's
deterministic row-selection task must reproduce those exact immutable bytes.

HCWDL-RKD is an additive representation-distillation campaign; it does not
broaden the logit-only HCWDL graph or recipe. Its thin command surface is
grouped as follows:

- pre-campaign audit and freezing:
  `audit_hcwdl_representation_parent_final_state.py`,
  `build_hcwdl_representation_final_disposition.py`,
  `build_hcwdl_representation_fixed_size_inventory.py`,
  `build_hcwdl_representation_storage_estimate.py`,
  `measure_hcwdl_representation_worker_runtime.py`,
  `build_hcwdl_representation_runtime_prerequisites.py`,
  `build_hcwdl_representation_runtime_rows.py`,
  `build_hcwdl_representation_runtime_binding.py`,
  `create_hcwdl_representation_campaign.py`, and
  `dry_run_hcwdl_representation_campaign.py`;
- execution review and the separate human authority boundary:
  `audit_hcwdl_representation_executable_candidate.py` and
  `build_hcwdl_representation_submission_authorization.py`;
- bounded first-evidence bootstrap (nonfinal and nonauthorizing):
  `build_hcwdl_representation_acceptance_bootstrap.py` and
  `run_hcwdl_representation_acceptance_bootstrap.py`;
- nonexecuting resource/evidence capture and action-proof assembly:
  `build_hcwdl_representation_scheduler_evidence.py`,
  `build_hcwdl_representation_miniature_evidence.py`,
  `build_hcwdl_representation_resource_profile.py`,
  `build_hcwdl_representation_usr1_exact_resume_proof.py`,
  `build_hcwdl_representation_validation_proxy_proof.py`,
  `build_hcwdl_representation_production_worker_smoke_proof.py`,
  `build_hcwdl_representation_tigris_action_proof.py`,
  `build_hcwdl_representation_tigris_evidence_bundle.py`, and
  `build_hcwdl_representation_tigris_acceptance.py`;
- parent/tap/recipe/target/training work:
  `build_hcwdl_representation_parent_import.py`,
  `build_hcwdl_representation_recipe.py`,
  `build_hcwdl_representation_targets.py`,
  `train_hcwdl_representation_node.py`,
  `select_hcwdl_representation_checkpoint.py`,
  `extract_hcwdl_representation_model.py`,
  `authorize_hcwdl_representation_cleanup.py`, and
  `complete_hcwdl_representation_cleanup.py`;
- validation reporting:
  `aggregate_hcwdl_representation_screen.py` and
  `build_hcwdl_representation_confirmation_registry.py`;
- population-scoped final protection and label-free inference:
  `register_hcwdl_shared_final_population.py`,
  `build_hcwdl_shared_final_legacy_cancellation.py`,
  `claim_hcwdl_shared_final_execution.py`, the selection/assignment/data-
  attestation commands, the prediction shard/finalizer commands,
  `join_hcwdl_shared_final_metrics.py`, and
  `recover_hcwdl_shared_final.py`;
- operations:
  `submit_hcwdl_representation_campaign.py`,
  `monitor_hcwdl_representation_campaign.py`,
  `resume_hcwdl_representation_campaign.py`,
  `cancel_hcwdl_representation_campaign.py`,
  `run_hcwdl_representation_task.py`, and
  `run_hcwdl_representation_local_smoke.py`.

The parent-final audit accepts each immutable legacy candidate through a
repeated `--candidate-artifact`; an empty list is an explicit audit of no
candidate artifacts. The final-disposition builder automatically degrades a
requested combined execution to validation-only when it finds prior exposure
or a live legacy worker. The legacy-cancellation command only publishes proof
after its exact reserved job inventory is terminal; it never performs broad
job-name cancellation.

All task wrappers select only a registered task and optional array index from
the immutable campaign specification. They expose no coefficient, layer,
kernel, teacher, row, or seed override. Deterministic target and prediction
rows use `sbatch/run_hcwdl_representation_deterministic_task.sh`; all other
rows use `sbatch/run_hcwdl_representation_task.sh`. Both workers use the
absolute project path and finish with `exec python -s`.

`measure_hcwdl_representation_worker_runtime.py` runs inside one already
allocated resource class and publishes the clean-checkout, active Conda,
`python -s`, installed-Weaver byte inventory, device/backend, and reusable row
runtime signature. Its CPU/memory/walltime/GPU arguments must exactly equal
the planning spec's request for that class. The immutable result always records
`scheduler_mutated=false`, `scientific_authorization=false`,
`authorizes_tigris_or_pilot=false`, and `final_role_accessed=false`; it cannot
submit a job or stand in for real-worker acceptance.

The isolated acceptance-bootstrap workers are
`sbatch/run_hcwdl_representation_acceptance_bootstrap.sh` and
`sbatch/run_hcwdl_representation_acceptance_bootstrap_deterministic.sh`.
Their exact bytes are bound in the bootstrap artifact. They can run only the
authorized scalar prefix through zero-coefficient acceptance, never ladder
training, reservation, array, or final-role work. The full-loss probe performs
no optimizer step, so these workers do not yet provide two-update/USR1
evidence and cannot complete or authorize Tigris acceptance.

The complete non-submitting procedure and acceptance boundary are in
[`docs/HCWDL_RKD_RUNBOOK.md`](../docs/HCWDL_RKD_RUNBOOK.md). Local dry runs,
fixtures, and smoke reports do not authorize Tigris or final-test access.
