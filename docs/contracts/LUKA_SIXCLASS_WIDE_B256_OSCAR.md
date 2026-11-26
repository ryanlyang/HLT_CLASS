# Six-class wide-range batch-256 Oscar contract v1

Execution extensions SPEC/v2 through SPEC/v5 below retain the v1 scientific recipe.

Scientific authority: [active plan](../plans/LUKA_SIXCLASS_WIDE_B256_OSCAR_PLAN.md).
Owned artifacts use `LUKA_SIXCLASS_WIDE_B256_OSCAR_<KIND>/v1`, schema_version 1,
canonical content hashes, explicit parents and development-validation/sealed
test flags. Other campaign namespaces remain distinct and are rejected.

## Frozen inputs

- Foundation: `2f05adcca889288e705a69fc1e54670f1713d29ec16ab75c4f694ab7b5a23d02`
- Transfer manifest: `7034f2153e585068000185ce4c1e7fb3e61fe96e5bbfeb5e650608a150071ae4`
- Extended role parent: `7f2413152ecb588764c862b4b83bd6f900320a1b915a3b77f95540dd6e283939`
- Owned population: `57eb7febd6ee6ae2c9729bdcc5f58a4f34dea232d7b1eaf12b7901ca405025d1`
- Owned recipe: `69df385334411bd372031c3ad8b7e42714eae5c52695270f23d5452135b47b41`

Population reconstruction replays the original unused-only role extension with
its original identity domain before selecting labels 0–5 and [200,2600). It
never draws new split memberships. Exact ordered IDs, labels, scalar pT, cell
counts, support counts/maxima and report bytes must replay. No final-test row
arrays or particle reads are exported. Scalar role counts alone are audited.

## Recipe and execution

The spec binds all six fresh nodes, same-campaign teacher receipts, own population,
native six-output model, CE-only main policies, auxiliary/contrastive recipe,
batch256 / 2488148 draws / 9720 updates / last84 / 20 epochs, LR1e-4 to5e-6,
export batch128, natural in-range selection and every-epoch pooled diagnostics.
Changing any of these fields and recomputing the JSON hash still fails semantic
validation. Constants and scope are not user-adjustable CLI knobs.

`SPEC` also binds exact clean pushed commit, donor file hashes, installed runtime
and backend fingerprint, paths, Oscar GPU/resource policy and output budget.
Source must be clean and tracked on the pinned worktree; `donors.json` pins
reused and adapted donor semantics at the documented commit.

`PLAN`, `SUBMISSION_INTENT` and `SUBMITTED` bind each exact sbatch command and
job ID. Execution requires `AUTHORIZE_SIXCLASS_WIDE_B256_OSCAR_V1`, a full dry
scheduler review, exclusive submission lock and no ambiguous prior intents.
No historical campaign is submitted. Terminal accounting and explicit resume
are required for continuation; pending/running predecessors are not cancelled.

`EPOCH_COMMIT` binds complete epoch state, row, job, stage, spec, population and
teacher hashes. Loss coefficients, schedule, full/primary validation counts,
finite metrics and deterministic sampled exposure replay before reuse.
`RESULT` is published last and references selected native weights, report,
in/out predictions, metrics, views, startup evidence and outgoing bank.

`BANK` is unique eligible TRAIN only, embedding width128, probability width6,
T2, fixed normalization/export dtypes and selected-model/report/recipe/spec
parents. Native inference has no auxiliary heads and no offline inputs for
M0HLT/DE000. Conditional views and source-blind CONCAT are explicitly privileged.

Reports retain primary selection and calculate pooled full metrics on disjoint
validation identities. Read-only inspection authenticates report/epoch hashes
and job/teacher bindings without loading binary payloads. It must disclose that
binary payloads, full source lineage and current scheduler were not reverified.
No evaluation command can unseal final test.

## Explicit memory recovery: SPEC/v2

Only the recovery SPEC uses schema_version2 and
`LUKA_SIXCLASS_WIDE_B256_OSCAR_SPEC/v2`; recipe, population, report and payload
contracts retain v1 because their scientific meanings are unchanged. A v1 SPEC
cannot enable the new execution policy. New fields are never written into the
old spec. `EXECUTION_RECOVERY/v1` binds the original spec file, failure record,
two imported baseline result files, exact pair-storage policy and four fresh
tail stages. The permitted parent spec hash is
`38c65fcb5d6850d4a74f79470337b314d73cb5c1d31c1dd3f7d6c035fb448f33`,
commit `d534fcfea4f11a4bc845127d03c31b779e391a8c`.

Result/epoch/target hashes retain their original spec parents when imported.
`artifacts.completed` and metadata inspection resolve OFFLINE/M0HLT back to that
authenticated parent; new fits bind those exact receipt hashes. A reused BANK
is validated against its originating spec and the identical current TRAIN
identities and labels. Completed models, source worktrees and old predictions
are not copied, rewritten, resubmitted or reselected.

`EARLY_MEMORY/v1` records the three authorized real witness identities/supports,
parity evidence, memory and scope (native graph only), and binds the new spec.
The completion-last STARTUP includes its checksummed file reference and actual
CONCAT storage counters; full-objective acceptance remains mandatory. Probe
models never initialize the scientific model. A checksum-bound
`RECOVERY_CLAIM/v1` under the parent, protected by its submission lock, permits
only one recovery spec to submit. Ambiguous scheduler receipts fail closed.

## Single long CONCAT allocation: SPEC/v5 (current)

`LUKA_SIXCLASS_WIDE_B256_OSCAR_SPEC/v5` adds `long_recovery`, a
`LONG_RECOVERY/v1` record binding the exact failed v3 spec/source/job and
startup evidence identified in the v4 section below. It fixes a single
CONCAT epoch window1--20, one GPU, six CPUs,4,320 minutes, RAM-only input
preparation and unchanged conservative timing margins. The full20-epoch
estimate is recomputed from the bound measurements and must fit72h.

Tasks are the original six-node DAG; only four unfinished fits are submitted.
The original two baseline results remain authenticated under their v1 parent.
All non-CONCAT resource budgets must match the original parent, including
48-hour student limits. V1--v4 cannot silently accept the longer request.
The `--single-concat` registration option requires the exact failed v3 parent;
`long-detach` selects this option. New output/source roots are mandatory.

V5 STARTUP binds task, full epoch window, prior committed count and measured
remaining epochs. Timing is checked against the exact registered allocation.
No SEGMENT receipt is produced or needed. All20 epoch commits, selected
evaluation and TRAIN-bank publication complete inside one normal allocation;
explicit same-source recovery after an interruption restores committed state.

V4 and v5 use the same `concat_time_recovery_claim_v1.json` under the v3 parent
and the same submission lock. Either alternative excludes the other. The old
failure/spec/evidence remain unchanged. Recipe/population/result/bank meanings
and final-test seals are unchanged.

## Checkpointed CONCAT execution: SPEC/v4 (superseded option)

`LUKA_SIXCLASS_WIDE_B256_OSCAR_SPEC/v4` adds `time_recovery`, a
`TIME_RECOVERY/v1` record binding exact previous SPEC/v3 hash
`2fea86c9d8109a20d1738b096095d3287630babdf03407f3b4c265809f5cb8fc`,
commit `2bf345f96ffd45f46f0cd9e6934798af62349401`, job9620918 failure,
early acceptance and five startup phase files. The measurements hash is
`1b8cb8d35b6d554b93964e23a6671ff0ebe64b3cabaec15c612756a53b5183dc`.
The policy fixes windows1--10 and11--20, with complete epoch-boundary state
continuation, twenty-epoch selection, final export only and unchanged timing
margins. Recipe/population/epoch/result/bank contracts retain their meanings.

The DAG adds `train_CONCAT_part1`; final `train_CONCAT` depends on it. Both
allocations write attempt evidence under `train_CONCAT/runs/JOBID`, shared
completion-last epoch markers under `train_CONCAT/epochs`, and separate logs
named for their actual tasks. Epoch receipts must belong to a submitted job
whose registered window contains that epoch. Final result publication still
requires the final task, all20 epochs and complete teacher export.

`SEGMENT/v1` at `train_CONCAT/segments/part1.json` binds the ten epoch receipts,
fit parents, first-task job, startup/view checksums and `teacher_published=false`.
All checkpoint bytes and acceptance evidence are checked before continuation.
V4 STARTUP records its task, window, prior committed count and measured remaining
epochs. A resumed half may finish from its last complete epoch; if a crash occurs
after epoch10 but before the segment marker, the first task can finalize it
without repeating optimizer updates. The final task also supports export-only
continuation after20 commits. No partial TRAINING_REPORT or BANK is published.

`concat_time_recovery_claim_v1.json` under the failed v3 root allows only one
new execution, serialized using the original baseline parent's submission lock.
The earlier memory/gradient claims remain immutable. New code refuses to resume
a v3 execution already superseded by this claim. Registration requires terminal
accounting for the prior four submissions and zero committed tail epochs.

## Pointwise input-gradient recovery: SPEC/v3

`LUKA_SIXCLASS_WIDE_B256_OSCAR_SPEC/v3` adds `gradient_recovery`, a hashed
`GRADIENT_RECOVERY/v1` record. Its exact previous SPEC/v2 is
`8f9ed0893cbe6a0c7f99beae9bab76d8870a75b3460ffc6d1b2894b6550b9d6f`,
commit `44aa1522d4720fd8045e55d16f977689bd7df0aa`. The record binds the previous
spec, job9615391 failure, job9618095 diagnostic log, original parent claim and
`sixclass_pair_pointwise_dgrad_v1` execution policy. The existing
`execution_recovery` record is identical to v2: completed baseline payloads retain
their original parents. All four prior recovery jobs and the diagnostic must be
terminal; the two observed failures must match, with no committed tail progress.

A new `concat_gradient_recovery_claim_v1.json` belongs to the failed v2 root.
The original v1 memory-recovery claim is preserved. New submission remains
serialized by the original parent's lock and refuses a different second fork.
The new code refuses continuation of a superseded v2 execution. Do not invoke
the old 44aa1522 launcher after submitting the new execution.

Only CONCAT enables the policy: native sparse layout `(1,channels,pairs)`, four
Conv1d modules with kernel1/stride1/padding0/dilation1/groups1, unchanged parameter
keys. Above `2**31-1` elements per input/output sparse sample, native forward and
native parameter backward are retained; input backward is `W.T @ dY`. Autocast
dtype is preserved, with no chunked parameter reductions or split BatchNorm.
The forced small-array path is solely a parity probe and is never enabled in a
scientific model. No second derivatives are supported or used by this campaign.

Both EARLY_MEMORY and STARTUP must carry the exact pointwise policy and nonzero
replacement input-gradient counters. EARLY_MEMORY must include four passing
parity records (native storage and forced fallback, FP32/BF16). These fields are
required by SPEC/v3 when reusing CONCAT results; old v1/v2 artifacts remain valid
under their own original specs. Missing evidence or changed policy fails closed.
