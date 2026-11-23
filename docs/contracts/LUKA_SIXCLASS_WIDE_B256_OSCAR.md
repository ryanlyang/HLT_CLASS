# Six-class wide-range batch-256 Oscar contract v1

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
