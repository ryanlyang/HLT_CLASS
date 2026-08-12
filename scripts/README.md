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
fixed Shell Exact cold/warm ladder in smoke, pilot, `midscale500k`,
`midscale1m`, `midscale2m`, or production mode. The matcher/assignment surface is
`validate_highcov_resources.py`, `build_highcov_assignment_shard.py`,
`finalize_highcov_assignments.py`, and `audit_highcov_assignments.py`.
Recipe, resource, submission, and human diagnostic gates are built by
`build_hcwdl_recipe.py`, `build_hcwdl_resource_profile.py`,
`build_hcwdl_submission_authorization.py`, and
`build_hcwdl_endpoint_ack.py`. Campaign creation/submission/dry-run,
monitor/resume/cancel, node/qualifier/control training, cache miniature,
selection, final evaluation, and aggregation each have a thin `*hcwdl*` CLI.
`run_hcwdl_local_smoke.py` executes the complete 23-node graph on bounded
synthetic tensors without final-test access. Live submission is either manual
two-phase or explicitly `preauthorized_automatic`. Automatic mode queues the
complete `afterok` DAG; its endpoint gate derives a lineage-bound review
waiver only after all six reports and endpoint invariants validate. Manual
mode retains `continue_hcwdl_campaign.py`. Slurm holds are not used.
Executable specs default to
and enforce the exact checkout invoking campaign creation, so concurrent
campaigns use dedicated Git worktrees. `midscale500k`, `midscale1m`, and
`midscale2m` are separately named 500,000/250,000/250,000,
1,000,000/400,000/400,000, and 2,000,000/500,000/500,000 modes, so later
midscale populations cannot silently reuse an existing identity. No
executable pilot/named-midscale/production spec can be made from the bundled
test-only recipe or unmeasured planning resources. After a real miniature,
create a locked non-live candidate spec with the measured resource profile,
inspect its dry run, and pass that candidate to
`build_hcwdl_submission_authorization.py`; authorization v7 binds the exact
resource requests, `HCWDL_COMMAND_PLAN/v3` hash, and continuation mode. The first bounded smoke
may use its explicitly authorized conservative bootstrap requests without
claiming measurement; pilot/named-midscale/production candidates require measured profiles.
Recreate the live spec with the same root,
recipe, profile, and authorization. Any command-path, resource, task, or
lineage difference fails closed.
An authorized `HCWDL_RECIPE/v4` is built with `--train-row-selection`; the
builder publishes fifteen exact ones for unweighted per-jet loss and binds
both the selection hash and its 15 train counts. The campaign's
deterministic row-selection task must reproduce those exact immutable bytes.

The validation-only dense cold supplement uses
`create_hcwdl_dense_pilot.py`, `run_hcwdl_dense_task.py`, and
`submit_hcwdl_dense_pilot.py`. It imports the completed unweighted 300k
pilot's authenticated selection, compact assignments, locks, M0, label-only
D100, and TOFF without copying or rematching them. It then runs the sequential
fresh-start `TOFF -> D100offkd -> D90 -> ... -> D0 -> M1` graph with one
shared nested repair coordinate system and publishes validation recovery only.
It has no final-test task. See `docs/HCWDL_DENSE_COLD_RUNBOOK.md`.
The same entry points accept `--rung-step 5` at creation time to register the
separate 22-model `TOFF -> D100offkd -> D95 -> ... -> D5 -> D0 -> M1`
screen. That graph uses distinct `HCWDL_DENSE5_*` contracts, authorization
phrases, `hcddp5_*` job names, output root, and aggregate. See
`docs/HCWDL_DENSE5_COLD_RUNBOOK.md`.

Dense ladders pinned to an older broken source commit are continued with
`create_hcwdl_dense_recovery.py`, `create_hcwdl_dense_reschedule.py`,
`submit_hcwdl_dense_recovery.py`, and `run_hcwdl_dense_recovery_task.py`.
The recovery requires a complete original
ledger plus an authenticated monitor, preserves the valid completed prefix,
and submits only the failed/downstream closure from a clean repair checkout.
It does not rematch, retrain the valid prefix, or access final test.

Ordinary pilot, named-midscale, and production failed closures pinned to an
older broken source commit use `create_hcwdl_campaign_recovery.py`,
`submit_hcwdl_campaign_recovery.py`, and
`run_hcwdl_campaign_recovery_task.py`. The recovery authenticates every
external completed parent, resumes the failed task or tasks from compatible
rolling checkpoints, and rebuilds only their unioned downstream closure under
the corrected source checkout. See `docs/HCWDL_CAMPAIGN_RECOVERY_RUNBOOK.md`.

An interrupted sealed final evaluation is recovered only through
`create_hcwdl_final_recovery.py`, `submit_hcwdl_final_recovery.py`, and
`run_hcwdl_final_recovery_task.py`. That two-job supplement reuses the exact
existing execution claim and frozen finalist registry, validates and preserves
any already-published finalist reports, evaluates only missing reports, and
then reruns aggregation. It never performs a new finalist selection. See
`docs/HCWDL_FINAL_RECOVERY_RUNBOOK.md`.

## HCWDL structural-feature homotopy

The validation-only structural-feature study uses the thin
`create_hcwdl_homotopy_pilot.py`, `submit_hcwdl_homotopy_pilot.py`,
`run_hcwdl_homotopy_task.py`, `monitor_hcwdl_homotopy.py`,
`resume_hcwdl_homotopy.py`, and `cancel_hcwdl_homotopy.py` surfaces.
Preparation helpers are `build_hcwdl_upper_calibration.py`,
`build_hcwdl_upper_coupling_shard.py`,
`finalize_hcwdl_upper_coupling.py`, `build_hcwdl_toff_targets.py`,
`build_hcwdl_homotopy_resume_evidence.py`, and
`build_hcwdl_homotopy_resource_profile.py`.
For the human-authorized thinned-v2 launch, the alternative
`build_hcwdl_homotopy_operational_waiver.py` records completed v1 worker
evidence plus completed v2 infrastructure locks without claiming the partial
v2 smoke completed.
It imports an authenticated HCWDL smoke or unweighted 300k pilot, registers no
final-test task, stores only compact coupling/target artifacts, and rebuilds
particle views in process-local RAM. The exact artifact and recovery rules are
in `docs/contracts/HCWDL_STRUCTURAL_FEATURE_HOMOTOPY.md`.
`run_hcwdl_homotopy_local_smoke.py` is a bounded synthetic behavioral check
that executes all 45 v2 fits without ROOT or Slurm; it is not a substitute for
the real production-worker miniature. Before the miniature,
`validate_hcwdl_homotopy_weaver.py` records direct-versus-wrapper FP32 parity
for both the unified student and two-stream native TOFF factories from the
clean pinned checkout. After that miniature,
`build_hcwdl_homotopy_resource_profile.py` authenticates the exact completed
ledger, monitor, worker GPU measurements, Slurm RAM/wall/I/O counters, and
durable artifact bytes before any 300k specification can be executable.
Source-only and resource-only failed closures share
`resume_hcwdl_homotopy.py` but require distinct contracts and authorization
phrases. See `docs/HCWDL_STRUCTURAL_FEATURE_HOMOTOPY_RUNBOOK.md`.

## HCWDL architecture–input factorial

The validation-only four-fit common-schema ablation uses
`create_hcwdl_architecture_factorial.py`,
`submit_hcwdl_architecture_factorial.py`, and
`run_hcwdl_architecture_factorial_task.py`. It compares exact HLT and P0
inputs under unified and full-depth charged/noncharged architectures, imports
native TOFF as a reference rather than a factorial cell, and registers no
final-test task. Smoke mode runs two updates per cell; the 300k pilot retains
the parent recipe's 60-pass/every-pass-validation schedule. The 300k exact-HLT
cells request six hours, while the projected-offline P0 cells request sixteen
hours based on the first pilot's measured pre-training ROOT/cache cost. Interrupted
training resumes through the generic rolling checkpoint when its exact Slurm
job is requeued; partial DAG submission is journaled before retry. See
`docs/plans/HCWDL_ARCHITECTURE_INPUT_FACTORIAL_PLAN.md`.
