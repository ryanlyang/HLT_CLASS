# Brown CS Hydra execution contracts v1 and v2

Authority: [Hydra execution plan](../plans/LUKA_U100_PAIR_HYDRA_PLAN.md), with the
unchanged [correction science](LUKA_U100_PAIR_CORRECTION.md). New execution
artifacts use `LUKA_FULLSIM_ALLJETS_PAIR_HYDRA_<KIND>/v1`. Extraction uses
`LUKA_U100_PAIR_HYDRA_EXTRACTION/v1`. Original scientific reports, particle banks,
miniatures, deployments and resource evidence retain their original contracts
and bind the new SPEC hash. Historical artifacts are never rewritten.

Only the plan's pinned archive and manifest are admitted. Extraction verifies
the archive before copying, validates every regular member while copying,
rehashes the archive afterward and publishes a completion receipt atomically.
Paths must be relative, explicit and free of symlinks/traversal. Existing or
incomplete destinations are not overwritten. Each subsequent input use validates
its original byte count and checksum. Original absolute source paths are evidence
only; no legacy validation routine may follow them on Brown. Missing input files
fail without reconstruction. TRAIN U100 and assignments never become model inputs.

The portable loader validates the original foundation and split identities,
block membership, row order, labels, shapes and full view fingerprints. Particle
target assembly and TRAIN statistics import the original implementation. U100
embedding bank validation imports the original Bank without teacher inference.
Only one native control checkpoint is loaded, for validation portability, never
to initialize any of the four fresh students.

The isolated environment must equal the recorded INSTALLED_ENVIRONMENT/v2,
including Python patch version, package versions, CUDA/cuDNN and every Weaver
Python source digest. The source guard allows only this additive Hydra runtime
relative to `65952ed3805f812c06d1ea54117c35b630cc684e`; original scientific runtime
must be unchanged. Source must be clean and present in fetched origin/main.

Hydra allocation: cluster hydra, partition gpus, account default, job QoS normal,
one NVIDIA L40, six CPUs, 120000 MiB, x86_64, registered venv and user-site disabled.
Workers prepend its lib directory to LD_LIBRARY_PATH and set deterministic cuBLAS
configuration. No environment or GPU substitution is automatic.

The controller must identify the exact running job, account, partition, QoS,
single node and six CPUs. Equal endpoint count ranges (`1-1`, `6-6`) are accepted
as exact counts. For task count and CPUs/task, concrete controller and exported
batch values must agree; at least one source must provide each. Missing, `N/A`
or `(null)` controller task fields require the explicit batch value. A missing
cluster export, including the literal `(null)` observed in job 131385, is
checked through `scontrol show config`. Its response must explicitly identify
`hydra`; a failed query, unknown controller name or different cluster fails.
The raw export remains logged and is never overwritten. Concrete conflicting
exports still fail. Missing memory export
requires exact `AllocTRES.mem` in MiB. When both sources provide a value, any
conflict fails. No memory rounding tolerance is introduced. The existing exact
accepted-job journal check still precedes task execution.

The log records only allowlisted scheduler fields before validation and names
every mismatch on rejection. This representation repair does not version the
scientific or artifact contracts: resources, inputs, source binding and study
semantics are unchanged. Slurm documents the
[batch variables](https://slurm.schedmd.com/sbatch.html#SECTION_OUTPUT-ENVIRONMENT-VARIABLES);
its [23.11.4 controller formatter](https://github.com/SchedMD/slurm/blob/slurm-23-11-4-1/src/api/job_info.c)
can emit `NumTasks=N/A` when the value is unspecified.

Original v1 shared setup first checks saved control BF16 probabilities on all validation
rows using the plan's numerical tolerances. Diagnostics are saved before numerical
admission. Then the unchanged production resource suite exercises all four real
miniatures, initial native parity, reload, full-support backward and timing.
Its L40 evidence is required before scientific submission. No accuracy, recovery,
or target-quality threshold decides whether a finite scientific run completes.

Full dry review covers setup and the four-fit/report envelope. The detached
dispatcher submits setup, polls its exact job ID, authenticates completion and
measured resources, then queues C/A/D/B in two afterany lanes plus a report.
Only measured walltime reductions are allowed. Setup failure releases no fits;
an individual student failure does not cancel independent later students.
Exclusive claims and durable exact-job journals prevent blind duplicate retries.

Reserve eight GiB for outputs plus one GiB filesystem headroom, separately from
the archive, extracted inputs and environment. No automatic deletion, cancellation,
new teacher, reference fit, final-test access or unrelated campaign mutation.

## Separate precision diagnostic contracts v1

The explicitly requested inference-only diagnostic uses distinct kinds under
the same prefix: `PRECISION_SPEC`, `PRECISION_PLAN`, `PRECISION_SUBMIT_CLAIM`,
`PRECISION_RUN_CLAIM`, `PRECISION_FULL`, `PRECISION_REPORT`, `PRECISION_RECEIPT`.
These new v1 kinds do not redefine SPEC, CONTROL_PARITY, TASK or ACCEPTANCE.
Diagnostic completion alone is not accepted as scientific resource admission.
The explicit v2 qualification below imports it as numerical evidence only.

PRECISION_SPEC binds the pinned transfer/origin/source hashes, exact recorded
environment, validation D000 population, fixed batch selection, four precision
modes and unchanged original probability limits. It declares zero scientific,
reference and teacher fits, no automatic retry or science release, and
`admission=false`. Source validation admits only the additive Hydra modules and
CLI/queue helper; original scientific runtime is unchanged.

The one-hour L40 job retains the same site/allocation authentication and exact
submission journal ownership as production. An exclusive submission claim and
fresh task directory prevent duplicate submission and overwrite. The worker
opens only validation D000 payloads and the native checkpoint/saved predictions;
shared provenance metadata may describe TRAIN but its payloads are not read.

PRECISION_FULL stores full-validation differences, metrics and report-only
old-limit comparisons. PRECISION_REPORT includes the fixed-batch arrays' row
identity digest, backend settings, timing and comparisons, immutable output
references, checkpoint/reference hashes, environment and GPU. A completion
receipt binds the spec, report and exact accepted job. Results verify lineage
and output bytes without performing new inference. Evidence is atomically
published; a partial job has no completion receipt. Numerical disagreement is
recorded, not converted to scientific admission. Any later policy change needs
an explicit plan/contract version change and separate validation.

## Execution v2: qualified Hydra diagnostic reuse

Authority is the 2026-11-03 v2 section of the active execution plan, following
the user's request to queue after reviewing successful diagnostic job 131484.
New science specs and result tables use SPEC/v2 and RESULTS/v2 under the
existing Hydra prefix. New PORTABILITY_POLICY/v2, PORTABILITY_EVIDENCE/v2 and
PORTABILITY/v2 artifacts bind the admission decision. Other command/resource
and scientific artifacts keep their original contracts and bind the new spec
hash. V1 remains readable with its original strict probability gate; old and
new numerical admission are not interchangeable.

The only imported diagnostic is job 131484 from clean pushed source
`ad55d3cd32d7826d6e16a475228fece3c6099da4`. Read-only validation executes in that
original source, authenticating its spec, exact accepted-job journal, completed
receipt and output hashes. Import all evidence paths/hashes into the new spec;
workers recheck immutable bytes and recompute metrics from saved arrays. Bind
the same transferred checkpoint, software, GPU, validation identities/labels,
original batch selection and backend settings. Missing/mutated evidence fails.

Numerical qualification now requires local CPU/GPU FP32 maximum <=1e-5 and RMS
<=1e-6; exact repeated BF16 original batches; and the unchanged full-validation
cross-site AUC-drift bound <=5e-5. All required quantities must be finite.
Historical cross-site BF16 max/RMS limits are preserved as report-only values
under v2. Accuracy and rejection are reported without quality gates. This
explicit change applies only to new v2 executions; it does not bless prior
failed setup receipts or establish A100 FP32 equivalence.

Recomputed saved floating-point metrics allow only 1e-12 absolute/relative
scalar reduction roundoff; discrete values, structure, identities and file
hashes must match exactly. Qualification uses the recomputed arrays. After that
check, reusable summary hashes retain the authenticated original scalar values,
so login/compute CPU reduction paths cannot change a completion record.

Shared setup writes PORTABILITY/v2 only after its current L40/backend and loaded
validation population match the imported diagnostic. No native-control inference
is repeated. It still exports particle targets and completes every original
production miniature, support extreme, deployment reload and resource profile.
PORTABILITY has `resource_acceptance=false`; it cannot replace ACCEPTANCE or
any miniature receipt. Submission requires `AUTHORIZE_LUKA_U100_PAIR_HYDRA_V2`
and the complete reviewed setup/four-fit/report envelope. No extra preflight.

RESULTS/v2 includes distinct U100_W5_HYDRA and saved SPORC reference rows. The
delta-matched fields now explicitly bind `comparison=U100_W5_HYDRA`. Recovery
uses labelled historical SPORC M0HLT/OFFLINE anchors. All reference probabilities
remain immutable; no reference model is retrained. Particle targets, losses,
BF16 training/inference, seeds and selection remain unchanged.
