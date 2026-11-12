# R8 reconstruction follow-up v2: eighteen Hydra fits

## Scientific question and authority

Test whether R8's best reported seed-0 validation recovery (46.2%, versus
37.3% for C0) repeats, and whether weight or strategy adjustments improve it.
Those are exploratory selected checkpoints, not a replicated result or an
independent test. This user-approved v2 design supersedes the earlier twelve-fit
follow-up draft. Existing registered campaigns remain owned by their pinned
source/specs; never reinterpret their task IDs using this registry.

Every fit uses the authenticated smaller dataset (299234 TRAIN, 74895 validation),
30 epochs, global batch128, original optimizer/LR, native HLT ParT and CE +
U100 embedding SmoothL1 weight5. No new teacher, export, or held-job change is
needed. All reconstruction targets are the complete paired OFFLINE TRAIN
collection. HLT particle inputs and deployed native classifier remain unchanged.

## Exact matrix

| Task | Seed | Change from R8 |
|---|---:|---|
| C2 | 2 | CE + U100-w5 only; no reconstruction decoder |
| R13 | 2 | Exact R8 replication: weight0.3, three-epoch ramp |
| C3 | 3 | CE + U100-w5 only; no reconstruction decoder |
| R14 | 3 | Exact R8 replication: weight0.3, three-epoch ramp |
| R15 | 0 | Reconstruction weight0.15 |
| R16 | 0 | Reconstruction weight0.2 |
| R17 | 0 | Reconstruction weight0.5 |
| R18 | 0 | Reconstruction weight0.7 |
| R19 | 0 | Reconstruction weight1 |
| R20 | 0 | Reconstruction weight2 |
| R21 | 0 | Reconstruction weight5 |
| R22 | 0 | Reconstruction weight10 |
| R23 | 0 | Ramp reconstruction over six epochs |
| R24 | 0 | Wait three epochs, then ramp over three epochs |
| R25 | 0 | Standard ramp; hold0.3 through epoch15; fade to zero at25 |
| R26 | 0 | Add whole-jet consistency at internal weight0.1 |
| R27 | 0 | Normalized kinematic/composition/track field weights1:1:2 |
| R28 | 0 | Access live block4 and block8 particle tokens with shared projection |

C0/C1 are authenticated historical references, not new fits. Compare R13-C2,
R14-C3, and seed-0 variants-C0. Historical R8 is the seed-0 replication anchor;
its reported AUC is context, not an unauthenticated imported result row.

## Frozen objective and variants

R8 uses a four-block width256/eight-head transformer decoder with fixed learned
queries, global-h FiLM, and global plus contextual block8 token memory. Predict
all TRAIN-supported output slots, particle fields, presence and count. Exact
one-to-one assignment compares predictions to OFFLINE targets; no true
HLT/OFFLINE particle matching is required. The original three-epoch coefficient
ramp, equal field weighting and decoder initialization stay identical in replicas.

R25 multiplies the usual ramp by clip((25-position)/10,0,1). Decoder forward,
loss diagnostics and optimizer membership remain present after fading; only the
auxiliary gradient coefficient becomes zero. There is no optimizer reset or
unannounced checkpoint restart. R24 likewise leaves native RNG and initialization
unchanged during its initial zero-reconstruction phase.

R26 uses Lreco + 0.1*Lconsistency before the external0.3 coefficient. The existing
consistency loss equally averages total four-momentum, signed mass-squared and
charged-energy-fraction penalties. It does not change assignment.

R27 replaces (kinematic+composition+track)/3 by
(kinematic+composition+2*track)/4 in BOTH assignment cost and differentiable
matched loss. Presence/count terms and all field normalization/validity masks
stay unchanged. Keep unweighted field components in the diagnostic logs.

R28 concatenates block4 and block8 live contextual tokens and their respective
native masks. Both use the existing shared128-to-256 projection, without new
learned depth/position codes. Global embedding conditioning remains. Gradients
flow through both taps; all hooks are removed after every forward. No deployed
parameters change, and decoder parameter count is unchanged; extra memory and
compute are measured explicitly.

Log raw/weighted losses and once-per-epoch TRAIN diagnostics. Compare CE,
U100-w5 and weighted reconstruction gradient norms/cosines at the common global
representation; record direct token-memory VJP norms separately. These are not
full-backbone gradient norms. No accuracy threshold gates a registered fit.

## Setup, queue order and automatic release

Use the existing A6000 BF16-native/FP32-decoder numerical path, six CPUs and32GB
host RAM. A hardware/precision change requires a separate qualified execution;
this implementation does not assume the A6000 is currently free or undrained.

Queue order:

`C2 R13 C3 R14 R15 R23 R16 R24 R17 R25 R18 R26 R19 R27 R20 R28 R21 R22`

The first four fits are the new matched pairs. Thereafter alternate weights and
strategies; scheduler order encourages coverage but cannot guarantee concurrency.
Every science fit is independent, with no fit-to-fit dependencies or lanes.

One bootstrap confirmation submits GPU setup and a durable CPU release with
`afterok` on that exact setup job. Successful setup authenticates targets/native
parity and measures thirteen paths: one control, nine reconstruction weights
(including0.3), consistency, track weighting, and two-depth memory. Each path
runs fresh warm and measured production miniatures, including full/partial
batches and real support tails. All weights, including5/10, must be finite.
Schedules/replicate seeds share measured execution paths. Storage budgets cover
18 science plus26 technical native exports, banks, predictions and contingency.

Release validates exact source, original authorization, successful setup job and
all receipts/resources, dry-reviews requests, then submits18 fits automatically.
An `afterany` CPU report waits for all18 fits, so failed fits remain visible.
Total jobs: setup + release +18 fits + report =21. Logout is safe once bootstrap
records the two real job IDs. A failed setup releases no science. An ambiguous
partial submission stops for exact-ID recovery instead of duplicating jobs.

## Evaluation and readiness

Select native validation AUC, then CE, then earliest. Report selected-control and
same-epoch control deltas; poor finite performance still completes all30 epochs.
The original50k slice overlaps validation and is report-only. The fourteen seed-0
screens require later replication before claiming stable improvements.

Local tests establish readiness to queue measured setup, not real Brown
qualification or scientific benefit. See the follow-up runbook and validation
record. No remote training, submission, commit or push is part of implementation.

## 2026-11-12 execution-only C3/R14 recovery

Source58d0c0ed setup168256 and release168257 completed on Brown. C3 job168726
and R14 job168727 failed at the initial Slurm controller query before claims or
training output; their controller records were subsequently purged. The
[recovery contract](../contracts/LUKA_R8_RECOVERY.md) permits two replacement
jobs using the exact original scientific runtime and setup, a separate root,
explicit replacement lineage and bounded read-only controller retries. No
scientific configuration or other campaign job is changed. See the
[recovery runbook](../LUKA_R8_RECOVERY_RUNBOOK.md).
