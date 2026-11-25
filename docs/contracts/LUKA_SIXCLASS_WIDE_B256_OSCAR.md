# Six-class wide-range batch-256 Oscar contract v1

Execution extensions SPEC/v2 and SPEC/v3 below retain the v1 scientific recipe.

Scientific authority: [active plan](../plans/LUKA_SIXCLASS_WIDE_B256_OSCAR_PLAN.md).
Owned artifacts use `LUKA_SIXCLASS_WIDE_B256_OSCAR_<KIND>/v1`, schema_version 1,
canonical content hashes, explicit parents and development-validation/sealed
test flags. Historical schema namespaces remain unchanged and are rejected.

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
