# HCWDL-RKD Local and Research-Compute Runbook

This runbook covers the additive matching-free representation-KD campaign in
[`HCWDL_MATCHING_FREE_REPRESENTATION_KD_ASCENTS.md`](plans/HCWDL_MATCHING_FREE_REPRESENTATION_KD_ASCENTS.md).
The parent HCWDL campaign remains governed by
[`HIGH_COVERAGE_COLD_WARM_DISTILLATION_LADDER.md`](plans/HIGH_COVERAGE_COLD_WARM_DISTILLATION_LADDER.md).

Current boundary: the repository may be implemented, tested, and rendered as
a non-submitting command plan locally. A live Tigris miniature, installed-
Weaver acceptance, measured resources, a clean pushed source identity, and
explicit user authorization are separate gates. Local fixtures never satisfy
those gates and never authorize final-test access.

## Scientific identity

The registered ablation contains four complete six-rung ascents:

```text
RSET-cold  RSET_M1c ... RSET_M6c
RSET-warm  RSET_M1w ... RSET_M6w
RREL-cold  RREL_M1c ... RREL_M6c
RREL-warm  RREL_M1w ... RREL_M6w
```

The four validation-only M5 controls are
`RSET_M5c_JET_ONLY_REP`, `RREL_M5c_NO_REL_REP`,
`RSET_M5c_WITHIN_CLASS_SHUFFLED_REP`, and
`RREL_M5c_WITHIN_CLASS_SHUFFLED_REP`. Students always receive HLT inputs.
Privileged D/TOFF views are opened only by authenticated target builders;
students join detached compact summaries by canonical identity. Representation
heads are training-only and are absent from extracted deployable checkpoints.

The parent base objective is class-weighted CE plus unweighted FP32 forward-KL
row means. Parent artifacts created under class-weighted KD are incompatible;
the parent-loss attestation must reject them rather than relabel them.
Its executable policy hash is derived from the versioned code constant and
authenticated runtime-source bytes; workers never read this plan or another
concept document to decide loss semantics.

## Local environment

Use the established Windows scientific environment with bytecode and threaded
BLAS variability disabled:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONPATH='src'
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$python = if ($env:HCWDL_PYTHON) { $env:HCWDL_PYTHON } else { 'python' }
```

Activate the intended environment first, or set `HCWDL_PYTHON` to its Python
executable. Never freeze a developer-specific absolute path into campaign
documentation or an artifact identity.

Run focused RKD tests before the repository-wide suite:

```powershell
& $python -m pytest -q -p no:cacheprovider --assert=plain `
  tests/test_hcwdl_representation_contracts.py `
  tests/test_hcwdl_representation_graph.py `
  tests/test_hcwdl_representation_math.py `
  tests/test_hcwdl_representation_targets.py `
  tests/test_hcwdl_representation_training.py `
  tests/test_hcwdl_representation_campaign.py `
  tests/test_hcwdl_representation_cli.py
```

Then run:

```powershell
& $python -m pytest -q -p no:cacheprovider --assert=plain
git diff --check
```

If Weaver is not installed locally, record installed-Weaver CPU parity as
unavailable; do not replace it with a synthetic-double result.

## Immutable prerequisite order

The operator-visible order is fixed:

1. freeze the tap schema;
2. run the public-forward/surface-forward parity boundary;
3. build the architecture attestation by reopening the parity report, tap,
   source files, parent reports, and checkpoint bytes;
4. build the corrected parent-loss attestation from actual parent reports;
5. publish the complete parent import;
6. publish kernel resources, numerical acceptance, the 24-node graph, the
   four-control registry, and the representation overlay recipe;
7. audit the parent final state and freeze the final disposition;
8. build the exhaustive path/runtime binding and immutable planning campaign;
9. render and inspect the complete symbolic command plan;
10. publish the nonauthorizing executable-candidate audit, which reopens the
    exact planning spec, plan, runtime binding, evidence/source lineage, and
    every task/array row;
11. only after explicit user approval, bind that exact candidate hash in the
    submission authorization and recreate the same campaign identity as the
    authorized live spec.

The three pre-campaign final-state producers are intentionally separate:

```powershell
& $python scripts/audit_hcwdl_representation_parent_final_state.py `
  --candidate-artifact <legacy-artifact.json> `
  --parent-campaign-sha256 <sha256> `
  --output <representation-root>/import/parent_final_state.json

& $python scripts/build_hcwdl_representation_final_disposition.py `
  --parent-final-state <representation-root>/import/parent_final_state.json `
  --requested combined_confirmatory `
  --output <representation-root>/import/final_disposition.json
```

Repeat `--candidate-artifact` for every legacy claim, output, or scheduler-row
artifact. Supplying none explicitly records that no candidate artifact was
found. A live/pending worker or prior model-derived exposure forces the
validation-only disposition.

When a combined reservation names legacy jobs, cancellation itself must use
the parent campaign's exact authenticated IDs. Only after an independent
post-cancellation state/output audit may its proof be published:

```powershell
& $python scripts/build_hcwdl_shared_final_legacy_cancellation.py `
  --reservation <population-namespace>/reservation.json `
  --jobs <exact-terminal-job-states.json> `
  --output-audit-sha256 <sha256> `
  --output <population-namespace>/legacy_cancellation.json
```

`--jobs` is an array of exact `job_id` and `scheduler_state_after` objects.
Any missing, extra, duplicate, or nonterminal job fails closed. This command
publishes evidence; it does not issue `scancel` and cannot cancel by name.

## Local planning, smoke, and dry run

Campaign creation consumes authenticated paths and hashes. Use
`create_hcwdl_representation_campaign.py --help` for the fixed input list;
scientific values are read from the graph/recipe and cannot be overridden.
Planning specifications remain non-executable. The fixed-size inventory is an
exact path/SHA-256 reference supplied through `--fixed-size-inventory`; its
measured files are reopened during every strict validation.

Build the fixed-size inventory and conservative storage estimate from real
files before campaign creation. Repeat each inventory flag for every file in
that measured class; a convenient-looking nominal size is not evidence.

```powershell
& $python scripts/build_hcwdl_representation_fixed_size_inventory.py `
  --parent-import-sha256 <parent-import-content-hash> `
  --retained-resume <exact-resume-state-file> `
  --selected-checkpoint <exact-selected-checkpoint-file> `
  --final-assignment <exact-final-assignment-file> `
  --fixed-artifact <exact-other-fixed-artifact-file> `
  --output <representation-root>/resources/fixed_size_inventory.json

& $python scripts/build_hcwdl_representation_storage_estimate.py `
  --train-rows <train-row-count> `
  --validation-rows <validation-row-count> `
  --final-rows <sealed-final-row-count> `
  --prediction-finalists <locked-finalist-count> `
  --parent-import-sha256 <parent-import-content-hash> `
  --fixed-size-inventory <representation-root>/resources/fixed_size_inventory.json `
  --output <representation-root>/resources/storage_estimate.json
```

The campaign-spec contract has one parameterized publication route,
`${campaign_spec_parent}/campaign_spec.json`: `planning` identifies the
reviewed nonauthorizing spec, while `.` identifies the live spec or a bounded
bootstrap spec in its own never-promoted root. The temporary
`planning/runtime_draft.json` is a nonpublished construction input, not a third
canonical campaign-spec route. First create that draft after the measured
resource profile is fixed, build the exhaustive runtime binding against it,
then recreate the planning spec at
`<representation-root>/planning/campaign_spec.json` with `--runtime-binding`.
The reviewed planning file must not later be overwritten because the candidate
audit binds its exact bytes. The final live spec is written only to
`<representation-root>/campaign_spec.json`. The runtime binding has its own
contract and canonical route at `runtime/runtime_binding.json`; it is not a
command-plan subtype.

Do not hand-assemble any task row. Assemble the closed prerequisite document
from its seven explicit operator registries, generate every scalar and array
row, and then freeze the runtime binding:

First measure each of the five resource classes from a clean checkout inside
an already allocated job of that exact class. The measurement command never
calls `sbatch`, opens a final role, or grants scientific authority. For example,
the deterministic target-class measurement is:

```bash
export PYTHONNOUSERSITE=1
cd /home/ryreu/atlas/HLT_Classification
python -s scripts/measure_hcwdl_representation_worker_runtime.py \
  --planning-campaign-spec <representation-root>/planning/runtime_draft.json \
  --data-root /home/ryreu/atlas/PracticeTagging/data \
  --conda-environment atlas_kd_tigris \
  --resource-class gpu_target \
  --cpus 8 --memory 320G --walltime 24:00:00 \
  --gpu gpu:gh200:1 --device cuda --deterministic-worker \
  --output <representation-root>/runtime/measurements/gpu_target.json
```

Repeat with the exact request, device, and worker route frozen in the planning
spec for `cpu_small`, `cpu_io`, `gpu_representation`, and
`gpu_final_prediction`. Authenticate all five measurement artifacts. Their
source snapshot, project, Conda, package, and Weaver identities must agree;
the GPU rows must also agree on the registered public device signature. Build
`runtime_signatures.json` by mapping each resource-class name to that
measurement's `row_runtime_signature_sha256`. Build `runtime_facts.json` from
the shared measured facts, the fixed data root, and `device=cuda`; do not type
invented hashes or version labels. The physical GPU UUID is audit-only: it is
excluded from the reusable row signature and is freshly measured into every
target execution attestation.

Every production worker repeats the same live measurement before it opens any
registered scientific input. A dirty or byte-mutated checkout, wrong commit,
wrong Conda/Python, enabled user site, changed Weaver source, incompatible
device/backend, or row-signature mismatch fails before target or training I/O.
The explicit `--local-planning-fixture` path remains nonauthorizing and never
manufactures this live evidence.

```powershell
& $python scripts/build_hcwdl_representation_runtime_prerequisites.py `
  --planning-campaign-spec <representation-root>/planning/runtime_draft.json `
  --runtime-facts <representation-root>/operator/runtime_facts.json `
  --runtime-signatures <representation-root>/operator/runtime_signatures.json `
  --static-inputs <representation-root>/operator/static_inputs.json `
  --artifact-content-hashes <representation-root>/operator/artifact_content_hashes.json `
  --target-generations <representation-root>/operator/target_generations.json `
  --bundle-members <representation-root>/operator/bundle_members.json `
  --settings <representation-root>/operator/settings.json `
  --output <representation-root>/runtime/runtime_prerequisites.json

& $python scripts/build_hcwdl_representation_runtime_rows.py `
  --planning-campaign-spec <representation-root>/planning/runtime_draft.json `
  --runtime-prerequisites <representation-root>/runtime/runtime_prerequisites.json `
  --output <representation-root>/runtime/task_rows.json

& $python scripts/build_hcwdl_representation_runtime_binding.py `
  --planning-campaign-spec <representation-root>/planning/runtime_draft.json `
  --runtime-facts <representation-root>/operator/runtime_facts.json `
  --task-rows <representation-root>/runtime/task_rows.json `
  --output <representation-root>/runtime/runtime_binding.json
```

The prerequisite builder rejects missing or extra exogenous routes and the row
builder validates every fixed production-adapter schema. Pilot and production
rows always bind scientific mode and exactly 60 passes; operator settings
cannot downgrade them to synthetic execution.

After the planning spec and runtime binding exist, the safe local surfaces are:

```powershell
& $python scripts/run_hcwdl_representation_local_smoke.py `
  --campaign-spec <representation-root>/planning/campaign_spec.json `
  --output <representation-root>/acceptance/local_smoke_report.json

& $python scripts/dry_run_hcwdl_representation_campaign.py `
  --campaign-spec <representation-root>/planning/campaign_spec.json `
  --runtime-binding <representation-root>/runtime/runtime_binding.json `
  --output <representation-root>/command_plan.json `
  --audit-output <representation-root>/runtime/dry_run_audit.json

& $python scripts/audit_hcwdl_representation_executable_candidate.py `
  --planning-spec <representation-root>/planning/campaign_spec.json `
  --command-plan <representation-root>/command_plan.json `
  --runtime-binding <representation-root>/runtime/runtime_binding.json `
  --output <representation-root>/acceptance/executable_candidate_audit.json
```

The final strict candidate must be built from the exact clean Tigris checkout
and durable Tigris artifact paths. Its three references are absolute and are
reopened by every later executable validation; a Windows-only reference cannot
authorize a Tigris worker. The PowerShell form above is useful for local
failure-path testing only.

The dry-run audit validates every bound scalar/array row against its exact
production-adapter schema, binds the runtime-binding and command-plan hashes,
and records `scheduler_mutated=false`. The smoke must enumerate every task and array row, exercise the full RSET/RREL
math and all four controls, keep `final_role_accessed=false`, and remain
`authorizes_tigris_or_pilot=false`. The command plan contains typed dependency
tokens, exact resources/paths, and no scheduler IDs. Do not invoke the submit
CLI as part of local acceptance.

Target generations are built just in time in the frozen order. Cold and warm
bank generations do not coexist across the cleanup barrier. A cleanup task may
delete only the exact shard set named by a durable authorization and only after
every registered consumer report validates. Confirmation/recovery creates a
new physical generation and must reproduce the original logical target hash.

The candidate audit always records
`human_submission_authorization_present=false`, `scheduler_mutated=false`, and
`authorizes_submission=false`. It cannot substitute for user authorization.
After review, the exact phrase gate creates the separate lock:

```powershell
& $python scripts/build_hcwdl_representation_submission_authorization.py `
  --planning-spec <representation-root>/planning/campaign_spec.json `
  --executable-candidate-audit <representation-root>/acceptance/executable_candidate_audit.json `
  --authorization-phrase "AUTHORIZE EXACT HCWDL-RKD SPEC FOR TIGRIS" `
  --output <representation-root>/locks/00_submission_authorization.json
```

For real authorization, run this command in the same Tigris checkout so the
candidate's durable references are reopened there before the lock is written.

Re-run `create_hcwdl_representation_campaign.py` with identical scientific,
resource, evidence, path, and runtime arguments, plus
`--authorized-executable`, `--executable-candidate-audit`, and
`--submission-authorization`, writing the result to the canonical campaign
path. Executable validation reopens the candidate's three byte-authenticated
references and requires the authorization to name that candidate's content
hash. A planning spec cannot embed either artifact.

## Workers

`sbatch/run_hcwdl_representation_task.sh` owns ordinary training, reporting,
locks, and cleanup. `sbatch/run_hcwdl_representation_deterministic_task.sh`
owns only target-build and final-prediction rows and sets
`CUBLAS_WORKSPACE_CONFIG=:4096:8` before Python starts. Both workers:

- require absolute `PROJECT_DIR`, campaign-spec, task, and runtime-binding
  environment values;
- source `${PROJECT_DIR}/sbatch/common.sh`;
- activate `atlas_kd_tigris`;
- set `PYTHONNOUSERSITE=1` and prepend `${CONDA_PREFIX}/lib`;
- forward an exact array index; and
- replace the batch shell with `exec python -s` so USR1 reaches the Python
  preemption monitor.

The Python dispatcher revalidates the executable campaign, task row, worker
class, array bounds, runtime binding, immutable inputs, and registered outputs
before executing an adapter.

## Genuine Tigris acceptance still required

Under separate authorization, the exact clean pushed source must complete:

1. installed-Weaver CPU surface/logit/input-gradient/parameter-gradient parity;
2. a genuine production-worker smoke with bounded real rows;
3. ordinary-D100 and TOFF target build, load, and authorized cleanup;
4. RSET/RREL cold and warm two-update/full-loss probes;
5. USR1 delivery and exact resume through the real worker;
6. validation-proxy selection, assignment, label-free prediction, escrow join,
   and partial-wave recovery without final-test access;
7. measured walltime, RAM, GPU, I/O, and durable-storage evidence;
8. a strict nonauthorizing executable-candidate audit and complete
   non-submitting dry run bound to those measurements.

Only then may an explicit pilot authorization be requested. The pilot remains
300,000 train, 100,000 validation, and 100,000 sealed final-test jets. A
validation-only disposition never registers final selection, assignment,
prediction, metric-join, bootstrap, or final-aggregate tasks.

The complete nonexecuting evidence-producer inventory is:

```text
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

These builders reopen and authenticate observed outputs; none submits a job or
turns an unrun action into evidence. Assemble the seven action proofs only after
their registered actions have actually completed, then build the evidence
bundle and acceptance artifact in that order. Every one of the seven frozen
Tigris check rows must contain its mutually bound `scheduler_evidence`,
`miniature_evidence`, and `action_proof`; either two generic records without the
action proof are insufficient.

The scheduler-evidence CLI accepts no caller-entered completion flag, exit
code, RSS, or elapsed value. It consumes the byte-authenticated raw output of
the frozen `sacct --parsable2 --units=K` field set and derives those values from
the allocation and step rows. The evidence job must have been submitted with
the exact `hcwdl-rkd-evidence-v1:<binding-hash>` comment, job name, and worker
path; the binding hash covers source commit, representation recipe (when
applicable), task key, resource class/request, worker role, and worker bytes.
The capture builder itself fails unless it is running as a Slurm job in the
declared `atlas_kd_tigris` environment with `PYTHONNOUSERSITE=1` and the Conda
library prefix. The miniature-evidence CLI additionally requires the immutable
semantic result artifact and binds its byte reference, contract/content hash,
and an execution identity covering the same job/source/recipe/task/worker
lineage. A locally constructed scheduler fixture is permanently marked
`local_fixture/v1`; its action proof may test the schema but cannot enter an
evidence bundle or satisfy `HCWDL_REPRESENTATION_TIGRIS_ACCEPTANCE/v1`.

### First-miniature bootstrap boundary

The full production workers deliberately require an executable campaign, while
an executable campaign deliberately requires genuine Tigris miniature and
measured-resource evidence. The additive
`HCWDL_REPRESENTATION_ACCEPTANCE_BOOTSTRAP/v1` contract permits only a frozen,
dependency-closed prefix through `zero_coefficient_acceptance`. It binds the
exact planning-spec bytes, exhaustive runtime binding, source commit, both
bootstrap-worker byte hashes, bounded smoke populations, and the unchanged
conservative smoke resource table. It explicitly records
`final_role_access_authorized=false`, `pilot_submission_authorized=false`, and
`scheduler_mutated=false`.

Build that authority only after the planning spec and runtime binding validate:

```powershell
& $python scripts/build_hcwdl_representation_acceptance_bootstrap.py `
  --planning-spec <bootstrap-smoke-root>/campaign_spec.json `
  --runtime-binding <bootstrap-smoke-root>/runtime/runtime_binding.json `
  --ordinary-worker <clean-project>/sbatch/run_hcwdl_representation_acceptance_bootstrap.sh `
  --deterministic-worker <clean-project>/sbatch/run_hcwdl_representation_acceptance_bootstrap_deterministic.sh `
  --through-task zero_coefficient_acceptance `
  --authorization-phrase "AUTHORIZE BOUNDED NONFINAL HCWDL-RKD ACCEPTANCE BOOTSTRAP" `
  --output <bootstrap-smoke-root>/acceptance/bootstrap/spec.json
```

The bootstrap uses a dedicated smoke root whose planning-only spec occupies
that root's canonical `campaign_spec.json` path, exactly as the bound runtime
rows require. Do not overwrite or upgrade this immutable smoke root into a
live campaign. The later reviewed planning candidate and authorized pilot use
a distinct campaign root and retain their separate planning/live spec files.

The two bootstrap workers accept only one task from that authorized prefix and
reopen the bootstrap artifact, planning spec, runtime binding, and worker bytes
before dispatch. They are not the campaign submission workers and do not
create a submission ledger. Scheduler submission, dependency capture, and
post-job `sacct` evidence remain explicit operator/evidence-builder work; this
builder itself never invokes `sbatch`.

This narrow authority does **not** solve the entire first-miniature cycle. The
current `smoke_probe` performs the full RSET/RREL backward probes but its own
artifact freezes `optimizer_or_scheduler_step_performed=false`. No bootstrap
action currently authorizes a two-update training run or a USR1
interrupt/resume workspace, and no final-role or final-validation-proxy task is
available. Consequently this prefix cannot by itself produce the complete
Tigris-acceptance bundle and must never be used to authorize a pilot. Dedicated
action-proof contracts and validators authenticate eventual results; they do
not expand this bootstrap authority. Adding those actions requires a separately
reviewed nonfinal acceptance authority that must not repurpose ladder rows,
bypass `pretraining_reservation`, or weaken executable campaign validation.

`HCWDL_REPRESENTATION_TIGRIS_ACCEPTANCE/v1` must fail closed unless it reopens
and validates every required action-specific proof. A completed job plus a
self-authored `passed=true` field is not proof of USR1 trajectory equality,
final-role isolation, or production-worker semantics. Do not hand-author an
acceptance bundle to cross this gate, and do not cite the existence of proof
builders as evidence that their actions ran.
