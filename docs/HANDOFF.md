# Current Handoff

## HCWDL dense cold 300k supplement (2026-08-09)

The new isolated `HCWDL_DENSE_COLD_GRAPH/v1` implements the requested
validation-only 300k descent: imported TOFF teaches fresh D100offkd, followed
by fresh D90/D80/.../D10/D0 and one fresh born-again M1. Every D node uses the
locked unweighted 0.25 CE plus 0.75 KD recipe at temperature 2; M1 uses the
same weights at HLT temperature 1. All nodes train for 60 passes, validate
every pass, and select macro-AUC first. M2--M6 and warm nodes are absent.

The supplement authenticates and directly reuses the completed pilot's split,
row selection, recipe, train/validation assignment stores, assignment and
Shell Exact locks, and M0/D100/TOFF reports. It does not rematch, copy repaired
datasets, or register final-test access. Twelve sequential checkpointable GPU
jobs feed one CPU aggregate.

The implementation audit found that the original coarse runner used a
domain-derived discrete repair seed. Individual coarse views remain valid,
but their discrete switches were not strictly nested across alpha. The dense
runner explicitly uses one shared seed across every D domain, making the new
10-point descent a monotone discrete repair trajectory. The old campaign is
retained as the coarse empirical control and is not mutated or relabeled.

Implementation lives in `hcwdl_dense.py`, `hcwdl_dense_runner.py`, and
`hcwdl_dense_workflow.py`, with three thin CLIs and
`sbatch/run_hcwdl_dense_task.sh`. The exact scientific contract and Tigris
commands are in `docs/contracts/HCWDL_DENSE_COLD_PILOT.md` and
`docs/HCWDL_DENSE_COLD_RUNBOOK.md`. Focused HCWDL/dense/repair validation
passes 95/95; the complete repository suite passes 329/329 with the same 14
Matplotlib/Pyparsing warnings; Python compilation and `git diff --check` pass.
No remote job was submitted by this change. The remaining runtime boundary is
an exact pushed clean commit and execution of the creation/dry-run/submission
sequence in the dense runbook against the completed unweighted pilot root.

## HCWDL unweighted primary and automatic continuation (2026-08-09)

The primary HCWDL recipe is now `HCWDL_RECIPE/v4`: qualifiers, privileged
teachers, cold/warm ladder nodes, controls, and confirmation runs use
unweighted natural-population per-jet loss through an authenticated vector of
fifteen exact ones. The deterministic train selection and all class counts
remain bound. Earlier square-root-weighted campaigns are ablations and cannot
be resumed as primary parents.

All six endpoint qualifiers share one stochastic trajectory seed, pairing
initialization, sampler order, repair randomness, dropout, and training RNG
across views. Campaign/submission contracts advance to v7 and the command plan
to v3. In `preauthorized_automatic` mode the full `afterok` DAG is queued once;
the endpoint gate creates a lineage-bound nonselecting waiver only after all
six finite reports and endpoint invariants validate. Poor finite science
continues, while job failures, nonfinite metrics, corrupt artifacts, and
lineage mismatches stop descendants. Manual acknowledgement remains supported.

The weighted 300k/500k/1M/2M campaign identities are stale for this primary
recipe and require fresh recipes, specs, worktrees, and exact job IDs.

Local verification after this change: focused HCWDL/high-coverage tests pass
78/78 and the complete repository suite passes 322/322 with the same 14
Matplotlib/Pyparsing warnings. A fresh unweighted Tigris smoke and the four
new campaign identities remain to be launched from an exact clean pushed
commit; no remote jobs were mutated from this local implementation.

## HCWDL named midscale campaign modes (2026-08-08)

HCWDL now registers three distinct immutable midscale populations:
`midscale500k` uses 500,000 train, 250,000 validation, and 250,000 final-test
jets; `midscale1m` uses 1,000,000 train, 400,000 validation, and 400,000
final-test jets; and `midscale2m` uses 2,000,000 train, 500,000 validation, and
500,000 final-test jets. Each reuses the
same count-agnostic matching, persistent assignments, two-phase endpoint gate,
23-node ladder, confirmation, and sealed-test machinery, but receives its own
campaign identity and requires a recipe whose class weights are bound to its
exact deterministic train selection. Future midscale sizes must
receive different mode names rather than editing these counts.

Campaign specs and submission authorizations advance to v6; v3 through v5
artifacts remain readable. Validation requires the stored
role counts to exactly equal the registered mode, closing a previously
implicit integrity check. The creation CLI exposes the new mode and focused
tests cover its exact counts, uncapped dry-run DAG, authorization, tamper
rejection, and v3 compatibility. This change does not alter or cancel the
currently running or prepared v3-v5 campaign.

The named modes require independent deterministic preselections, recipes,
resource/storage evidence, authorization, worktrees, assignments, locks, and
two-phase submissions. None may reuse another mode's recipe lineage.

Verification after registering all three named midscale modes: the focused
high-coverage/HCWDL suite passes 72/72 and the complete repository suite passes
316/316 with the same 14 Matplotlib/Pyparsing warnings. `midscale2m` has not
yet been executed on Tigris. Its exact next step is a clean commit and push,
then a dedicated worktree, deterministic 2m/500k/500k preselection, a recipe
bound to that 2m train-selection hash, scaled storage/profile evidence, exact
v6 candidate dry run and authorization, and an independent two-phase launch.

## PMARD all-model exploratory test comparison (2026-08-08)

The user explicitly authorized evaluation of all completed T100-sweep and
paired-schedule models and reclassified the pilot's 100,000-jet `final_test`
role as an exploratory comparison set. The new isolated contract freezes the
36 plus 27 distinct report/checkpoint identities before access, publishes dedicated
authorization and execution locks, builds one deterministic test selection,
runs an uncapped HLT-only `0-62` GPU array, and aggregates only metrics. Every
evaluation must attest the same ordered identity hash; per-jet predictions are
never written. The aggregate permanently records that post-test rankings are
descriptive and confirmatory claims from this holdout are forbidden.

Implementation lives in
`src/hlt_classification/scouting/exploratory_test.py`, the three
`scripts/*pmard_exploratory_test*.py` entry points, and
`sbatch/run_pmard_exploratory_test.sh`. Scientific and operational semantics
are documented in
[`docs/contracts/PMARD_EXPLORATORY_TEST_COMPARISON.md`](contracts/PMARD_EXPLORATORY_TEST_COMPARISON.md)
and [`docs/PMARD_RUNBOOK.md`](PMARD_RUNBOOK.md). The existing confirmatory
single-finalist code path was not weakened. Current focused verification is
16/16 passing; the complete local suite is 310/310 passing with 15 existing
Matplotlib/pytest-cache warnings. Tigris execution remains to be performed
from exact pushed clean source.

The first Tigris creation attempt at commit `f9926ab` exposed a historical
lineage compatibility bug before any test access: the supplemental sweep's
immutable parent campaign was being recomputed against today's expanded PMARD
campaign registry. The exploratory validator now authenticates every archived
embedded artifact by its own versioned contract and recorded content hash,
while still validating the sweep/follow-up relationship, every training
report, and every checkpoint hash. It does not reinterpret the old
campaign with current constants.

The second pre-access creation attempt established that the completed follow-up
contains 27 rather than 28 models: its registered construction correctly
deduplicated one recipe that won both CE and utility roles. The fixed-64
assumption was therefore invalid. Contract v2 freezes the exact registered
inventory (36 + 27 = 63) and derives the
array bound from the immutable specification. No test-role access occurred in
either failed creation attempt.

The third pre-access creation attempt found that some historical training
reports predate `balanced_accuracy` and `always_qcd_accuracy`. These derived
fields are now optional only in copied validation summaries; the five common
selection/performance metrics remain mandatory, and every exploratory test
evaluation computes the full current metric set. This attempt also stopped
before publishing a specification or reading the test role.

## HCWDL matching-free representation-KD plan (2026-08-08)

The new implementation-grade registered-ablation plan is
[`docs/plans/HCWDL_MATCHING_FREE_REPRESENTATION_KD_ASCENTS.md`](plans/HCWDL_MATCHING_FREE_REPRESENTATION_KD_ASCENTS.md).
It adds four planned full M1--M6 ascents without changing the implemented
logit-only `HCWDL_GRAPH/v1` or `HCWDL_RECIPE/v3`: paired jet plus unordered
token-set KD (`RSET`) under cold and warm initialization, and the same package
plus differentiable latent-relation KD (`RREL`) under cold and warm
initialization. The existing logit cold/warm ascents remain the primary
controls.

The plan freezes 24 new node identities, exact privileged teachers, separate
native-offline charged/neutral representation spaces, fixed RFF set/relation
sketches, train-only gradient calibration, a `rho_repr=0.10` auxiliary budget,
pass-based ramps, 60 passes with validation every pass, macro-AUC-first
selection, paired stochastic streams, one-time compact target construction,
generation-aware just-in-time target cleanup/recovery, zero-coefficient,
jet-only, no-relation, and shuffled-pair controls, and a population-scoped
combined final-test reservation honored by both evaluator families. It includes the
proposed source/module map, CLI and Slurm boundaries, artifact layout,
contracts, failure semantics, complete test matrix, implementation blocks, and
definition of done.

The planning audit found a pre-existing parent-loss mismatch that is now an
explicit implementation blocker: the active HCWDL plan requires class-weighted
CE and unweighted KD row means, while the current shared `pmard_loss` runtime
class-weights CE and both KD terms. HCWDL-RKD requires a versioned
parent-loss attestation, runtime correction, and consistent parent artifacts;
affected reports cannot be relabeled as paired. The plan also requires a new
architecture attestation because current parent artifacts do not publish the
needed model/tap architecture hash.

This is documentation only. No representation-ascent contract, model tap,
target bank, training node, worker, smoke, or submission was implemented or
authorized in this step. Before editing, the focused existing HCWDL/PMARD
training suite passed 40/40 under the local `tagging-hlt` environment with
`PYTHONPATH=src`. After the final plan audit, the four focused HCWDL suites
passed 58/58 in the same environment; Markdown targets and `git diff --check`
also passed aside from line-ending notices. The separately authorized
primary-HCWDL Tigris-validation work remains independent; this new plan does
not submit, cancel, or alter it.

## Preliminary PMARD pilot evidence (2026-08-07)

The recovered 300k/100k/100k PMARD pilot has now completed its teacher ladder,
authenticated validation oracles, K2 alpha sweep, alpha selection, and K0--K6
logit-KD controls. Alpha `1.0` was selected. On validation, K2 improved over
the ordinary K1 self-KD control from CE `0.675880` to `0.675060`, macro AUC
`0.937334` to `0.937468`, and mean log QCD rejection `7.022452` to `7.034357`;
overall accuracy was effectively unchanged. The selective T100 endpoint
closed only about 16--21% of the native-offline oracle gap, and direct TOFF KD
in K6 improved CE/AUC slightly beyond K2 without improving rejection. These
are single-screening-seed validation observations, not confirmed or final-test
claims. Mechanism controls remain in progress; representation, generation,
confirmation, and sealed final-test stages remain pending.

The complete numerical tables, exact endpoint semantics, current limitations,
artifact locations, and code-symbol map are recorded in
[`docs/PMARD_PILOT_PRELIMINARY_RESULTS.md`](PMARD_PILOT_PRELIMINARY_RESULTS.md).
This section supersedes older statements below that no real PMARD scientific
output exists; it does not supersede their implementation or recovery history.

An authenticated supplemental T100 transfer sweep is now implemented. It
holds CE at 25%, scans T100 KD weights `0.15/0.25/0.35/0.50`, T100
temperatures `1/2/4`, and training exposures `10/20/40` complete passes,
assigns the remaining 75% teacher mass to T0 at temperature 1, and trains all
36 rows on HLT inputs only. A single prerequisite
pass caches T0/T100 train logits, so repaired alpha-one inputs are not rebuilt
for every student. The sweep is bound to the completed pilot prefix, compares
against both K1 and the original alpha-one K2 row, and cannot read final test.
Its specification and commands are documented in
`docs/contracts/PMARD_T100_KD_SWEEP.md` and `docs/PMARD_RUNBOOK.md`.

The first live supplemental target-cache job, `48034`, failed after six
minutes before publishing a cache because the selective authorization stores
the versioned family `SELECTIVE_FULL_PARTICLE_ENDPOINT/v1`, while the tensor
builder historically accepted only the runtime selector without `/v1`.
`runtime_repair_family()` now performs an explicit allow-listed translation
for both full-endpoint contracts, and a regression proves the versioned
selective family produces byte-identical repaired tensors. The failed
source-bound sweep specification must not be resumed under corrected source;
recovery requires a new clean commit, worktree, sweep root, and specification.
The next clean attempt, target job `48142`, then showed that normalization at
the tensor builder was too late: the PMARD streamer had already omitted native
offline branches because its projection dispatch still compared the versioned
name literally. The streamer now normalizes at entry, before category gates,
branch projection, confidence dispatch, or tensor construction. Its
fitted-strict regression requires the versioned selective family to project
and forward native offline arrays under the unversioned runtime selector. The
supplemental target worker also evaluates the constrained T100 repaired path
before the ordinary T0 path so future endpoint incompatibilities fail early.

A validation-only paired schedule follow-up is now implemented for the
observed 20-pass CE-control instability. It consumes the completed 36-row
T100 sweep aggregate, freezes both its lowest-CE and best-utility recipe at
each of 10/20/40 passes, carries the 40-pass winners into a predeclared
60-pass extrapolation, and reruns those recipes beside fresh K0 CE-only and
K1 T0-self-KD controls. Every row validates once per complete train pass.
Ten-pass rows use the parent `3e-4` peak LR; 20/40/60-pass groups run both that
fixed LR and the predeclared `L*sqrt(10/passes)` schedule. The maximum grid is
28 uncapped jobs (smaller only when the two parent selection rules choose the
same recipe). It reuses the parent T0/T100 float32 target cache and opens only
the 300k train and 100k validation HLT views; it does not rematch, construct
repaired views, or access test. Contracts and launch procedure are in
`docs/contracts/PMARD_KD_SCHEDULE_FOLLOWUP.md` and `docs/PMARD_RUNBOOK.md`.
Local source compilation passed; the dependency-bearing pytest suite was not
available under the current Windows Python environment and still requires
execution in `atlas_kd_tigris` before push/Tigris submission.

## Active PMARD implementation (2026-08-05)

The user explicitly activated the PMARD mandate. The active scientific plan is
`docs/plans/SCOUTING_ALPHA_REPAIR_DISTILLATION_OPTIMAL_CAMPAIGN.md`; PRAD is a
separate prior campaign and cannot redefine PMARD.

Implemented locally in the current worktree:

- version-4 Scouting schema with the fitted-strict donor's native
  constituent-count boundary (`n_cpfcands` plus `n_lts`) as the authoritative
  charged/lost-track layout; the auxiliary `cpfcandlt_isLostTrack` model
  feature is deliberately not a matching-projection or boundary requirement;
  plus versioned source, label, row-identity, split, artifact, and
  role-capability contracts;
- version-2 60/20/20 source-file split construction: the source audit streams
  per-file 15-class counts, an exact MILP minimizes the worst class-by-role
  fraction deviation under whole-file disjointness, and validation fails
  closed above a predeclared 0.02 absolute tolerance;
- exact 15-class mapping, 21-channel CMSSW transform, 200-token HLT view, and
  90/60 native-offline no-SV view;
- projected bounded ROOT streaming with rank/worker file partition and sealed
  pre-lock final-test branch access, plus eight-file round-robin train reads
  and deterministic natural-population 32,768-row RAM shuffle buffers;
- v2 sparse candidate graphs with audited node/edge/track-validity features,
  three-round bipartite message passing, positive-unlabeled bootstrap,
  two ultra-conservative pseudo-label rounds, held-out edge and post-assignment
  calibration, deterministic score quantization, exact alternative margins,
  Hungarian/OT controls, and `M0`--`M5`;
- adversarial synthetic loss/fake/split/merge/category-confusion tests,
  genuinely different event-mixed jets, independent HLT/offline perturbation
  stability, confidence Brier and matching-only category gates;
- the canonical `fitted_strict` selective matcher as a standalone production
  module: immutable semantic hashes for the 7,500-jet edge/confidence/audit
  bundle, exact empirical LLR scoring, strict PID/charge and broad kinematic
  gates, rectangular Hungarian private dummies, calibrated ten-diagnostic
  confidence at threshold `0.9828147479721088`, native-index restoration after
  lost-track exclusion, batch inference, and persistent-cache construction;
- deterministic class-proportional row-selection manifests and immutable
  fitted-strict assignment caches computed once per selected jet: compact
  per-source CSR/DEFLATE shards store only entry, uint8 HLT index, uint16
  offline index, and uint16 confidence; consumers use a bounded lazy LRU and
  verify split/selection/matcher/data hashes before reuse;
- exact `P4_ONLY/v1` alpha endpoints and byte-identical `alpha=0`, plus charged
  eligibility, shuffled/corrupted matches, direction/response/wrong/random
  controls, log-angular interpolation, and confidence weighting;
- versioned `SELECTIVE_FULL_PARTICLE_ENDPOINT/v1` mixed-type repair: unique
  accepted assignments, exact all-21-channel plus p4 offline replacement for
  accepted tokens at `alpha=1`, byte-identical unmatched HLT tokens, raw
  continuous interpolation, SHA-256 identity-bound nested discrete/validity
  switches, charged/neutral track-applicability coherence, and unchanged HLT
  count/order/mask without durable repaired datasets;
- a hash-chained selective-assignment authorization boundary binding the
  canonical fitted-strict artifact and threshold, split, deterministic row
  selection, compact train/validation assignment manifest, all five input
  categories, and the exact unmatched-as-HLT policy before any privileged
  teacher or student;
- 21-input/15-output canonical Weaver adapter, native-offline diagnostic model,
  K0--K6 loss semantics including a valid alpha-zero K2 collapse, class
  weights, separate `R4_PAIR`/`R4_GRAM` representation arms, corrected
  per-signal QCD discriminants, and a fixed three-generation registry whose
  representation and anchored-companion paths remain executable when the
  selector legitimately chooses alpha zero via an exact HLT identity alias;
- cross-file full-batch packing, Weaver no-decay optimizer exclusions,
  named RNG domains, FP32 CE/KL/recursive representation and pair/Gram math
  plus FP32 RAM logits under BF16 ParT autocast, eight-check validation cadence,
  durable 50-update interval-mean loss history, validation/preemption-only
  rolling checkpoints, batch-shell-to-Python `exec` signal delivery, exact
  batch-offset/RNG/optimizer/best-state resume, a shared deterministic
  4,096-row smoke selection, one bounded RAM-cache miniature, and HLT-only post-cache streams
  with unused GPU teachers/matchers released for R0 and zero-coefficient
  controls, plus the HLT-only inference signature,
  fixed observer-only stratification and final nested paired statistics;
- campaign-spec v10 process-local repaired-view caching: nonzero-alpha
  teachers retain only FP32 privileged train/validation model views, positive
  representation KD retains aligned HLT/privileged train views, and
  identity-indexed replay reproduces the exact file/chunk/32,768-row-buffer
  epoch sampler and batch tails; the 320-GiB/75%-of-Slurm preallocation gate,
  192-GiB smoke/pilot and 384-GiB production requests, zero durable output,
  v9 validation support, and exact-source enforcement keep the active
  recovered pilot scientifically unchanged;
- hash-chained freeze locks and a minimum-storage Tigris DAG whose smoke graph
  cannot access final test, with exact-ID monitor/resume/cancel, authenticated
  resource/storage/miniature evidence, nonmutating production dry run, and
  explicit production authorization gates; a pilot graph uses exact
  300k/100k/100k row selections, constructs final-test selection/assignments
  only after the execution lock, and reports HLT, native-offline, and
  matched-offline endpoint oracles alongside deployable finalists.

Local verification passes `228 tests` with 14 Matplotlib warnings. The new
RAM-view-cache coverage includes three-epoch online-versus-cached
identity/batch/tensor equivalence, memory fail-before-allocation, alpha-zero
and native-offline bypass, campaign v9 compatibility, and campaign v10
resource wiring. Existing coverage includes selective
all-field endpoint, persistent sparse join, pilot-DAG, matcher, KD, resume, and
legacy campaign regressions. CLI help and `git diff --check` also pass; the
remaining notices are repository line-ending notices.

The first genuine pilot DAG, campaign `pmard_pilot_62dd448731cdd932` at pushed
commit `00f03a6b1818ad16f9538e8e9e155082293677cd`, reached the persistent
assignment-cache array (`43465_0`--`43465_42`) after its source, split,
feature-audit, row-selection, Weaver, and matcher-artifact prerequisites
completed. Every assignment shard failed before publication because the port
required `cpfcandlt_isLostTrack` to equal a positional collection boundary.
That check was stricter than the fitted-strict donor, which slices regular
charged candidates from the authenticated scalar counts. Schema v4 corrects
the projection and decoder contract: `n_cpfcands + n_lts` defines the charged
family and the appended lost-track suffix, while the auxiliary model feature
cannot redefine that boundary. No matcher-quality or PMARD result can be
inferred from this infrastructure failure, and the immutable v3 pilot must not
be resumed under v4 source.

The schema-v4 replacement pilot, campaign `pmard_pilot_3f379cb66398fcc8` at
commit `775bed70`, completed its persistent assignment cache, baseline budget
grid, budget selection, and temperature grid. Privileged-teacher array
elements `43733_1`--`43733_5` then failed before launching the trainer because
the workflow appended the authenticated matcher threshold to `argv` as a
Python float after `_script` had stringified the initial arguments;
`subprocess` accepts only text, bytes, or path-like arguments. The same latent
fault existed in the nonzero-alpha student builder. Both builders now
normalize their complete final command vectors to strings, with regressions
covering the exact threshold in teacher and student commands. This is an
execution-only correction and does not change matcher, repair, loss, model, or
selection semantics. The replacement pilot remains immutable and cannot be
continued in place under the corrected source snapshot. A version-4 prefix
import now provides the narrow recovery path: it proves Git/AST compatibility,
reuses independent source/Weaver/HLT-budget/temperature artifacts, rebuilds
the assignment cache plus its dependent authorization and training locks, and
then submits from `teachers` onward. Failed teacher/descendant artifacts are
categorically excluded.
The first prefix-resume submission exposed a Slurm-controller lifetime edge:
the resume helper attached the newly submitted `teachers` job to historical
completed job `43732`, although the imported `training_lock` artifact already
authenticated that predecessor. Completed IDs may age out of the controller
and are not valid scheduling dependencies. Resume commands now include only
dependencies newly submitted by the same resume invocation; imported reusable
predecessors remain artifact dependencies, and scheduler failures retain the
actual `sbatch` diagnostic. Focused resume/recovery/campaign tests pass 10/10,
and the complete suite passes 218/218 (plus one local pytest-cache warning).
The next recovered teacher launch exposed a second execution-only defect:
assignment manifests intentionally store shard paths relative to
`matcher/assignments` (or `matcher/final_assignments`), while the persistent
consumer resolved them relative to the manifest's `matcher` parent. The
recovery had correctly imported every shard, but the consumer looked one
directory too high and failed closed before the first training update. The
store now resolves both canonical initial and final assignment roots while
retaining colocated custom-manifest behavior. Prefix-import contract v4 proves
the narrowly authorized execution corrections by AST comparison against pilot
commit `775bed70`; no trained model or matcher output is reinterpreted.
The subsequent teacher launch reached selective full-field repair and exposed
an over-broad HLT identity check. Selective repair validated all visible HLT
tokens, including unmatched ambiguous-PID tokens that the contract requires it
to preserve.
Repair now validates chargedness only on `matched_tokens`; complete repair
still validates every token because its matched set is complete. A regression
proves that an invalid unmatched identity remains byte-identical while the
same identity fails when matched, and the v4 recovery AST proof authorizes
exactly this scope correction in addition to the two earlier execution fixes.
The following real-data launch then established that fitted-strict's exact PID
gate could accept unknown-to-unknown (`-1`) pairs. Those pairs are outside the
authorized five-category endpoint and can carry invalid offline identities.
Assignment-cache construction now requires exact categories 0--4, exact charge
in -1/0/+1, and charged/neutral charge coherence on both endpoints. Because
the original cache violates that scientific eligibility rule, recovery v4
does not reuse it. It resubmits only the inexpensive 43-way compact cache,
manifest, full-endpoint lock, and training lock while retaining the expensive
HLT-only budget and temperature checkpoints across their scheduling-only gate.

Still required externally before production: commit and push the exact clean
source; authenticate the native 53-file dataset and split; run installed-Weaver
FP32 parity; complete the genuine streamed Tigris smoke graph; capture measured
RAM/GPU/I/O/wall/storage evidence; and inspect the full production dry run.
The smoke DAG ends at `miniature_summary` and structurally contains no
`final_test` task. No PMARD scientific result, real matcher lock, checkpoint,
resource measurement, or final-test output is claimed in this handoff.

Track-only and p4-plus-track repair remain deliberately unavailable until the
real Stage-A units/definition audit creates a compatibility lock; attempting
either fails closed. The old complete full-particle implementation remains
locally verified but is not the selected scientific endpoint. The selected
fitted-strict path is intentionally partial and uses the new selective
all-field contract; it never claims that the unmatched fraction became offline.

## Repository state

Transfer Blocks 1--4, 6, and 7 are complete locally. Transfer Block 5 is
code-complete and its authoritative installed-Weaver FP32 equivalence check
passed on Tigris. A complete real PRAD miniature is still required.

Implemented:

- frozen JetClass schema, canonical identities, bounded ROOT reads, and
  deterministic balanced splits;
- registered `fixed_hlt_v3_track_dominant_proxy/v1` degradation and replica
  contracts with donor-v1 deterministic parity;
- immutable authenticated offline and HLT cache shards;
- atomic no-overwrite publication and exact completed-shard reuse;
- streaming cache range/batch readers that do not materialize the campaign;
- source, split, schema, profile, replica, generator, offline-parent, and
  source-array lineage;
- aggregate HLT diagnostics with construction indices excluded from deployable
  artifacts;
- exact label, identity, role, validity-state, padding, and parent validation;
- canonical 17-feature Particle Transformer input construction;
- frozen from-scratch standard-four Weaver baseline adapter and standalone
  FP32 logits/training-gradient/mask/state-dictionary attestation, including
  exact topology comparison for Weaver's nonfinite auxiliary Lorentz-input
  derivative.
- deterministic shard-local epoch sampling and fixed-update AdamW training;
- exact model, optimizer, schedule, scaler, sampler, replica-cycle, RNG, and
  history checkpoint resume;
- deterministic validation checkpoint selection without early termination;
- ordered label-free prediction shards and authenticated label joins;
- frozen accuracy, CE, efficiency, AUC, Brier, 15-bin top-label ECE,
  QCD-rejection, and paired-bootstrap definitions.
- clean-source snapshots, immutable campaign specifications, storage gates,
  task attestations, per-job durable submission journals, exact-ID Slurm
  ledgers, monitoring, and resume plans;
- thin absolute-path Tigris workers and two-lock final-test authorization;
- a smoke-only deliberate update-one training interruption followed by
  dependent exact-checkpoint resume through the ordinary training worker.
- the active PRAD implementation: proportional 500k/150k/500k split contract,
  paired-view audit/cache, replica-aligned charged/neutral Hungarian matching,
  exclusive C/A K=2/3/4 targets, and train-only moments/positive weights;
- a parity-oriented offline-teacher/HLT-student Weaver extension with symmetric
  contextual relations, bounded centered head bias, zero-initialized gates,
  HLT-only deployment signature, and explicit nondeployable oracle path;
- optional validated float16 dense teacher relation/bias caching and a
  mandatory installed-Weaver CUDA PRAD mechanics attestation before E0;
- configuration-driven E0--E10 and all registered V1--V10 switches, fixed
  staged budgets, frozen teacher, exact resume, deterministic identity-bound
  in-batch shuffled control, and five confirmation seeds;
- PRAD validation selection, bounded relation-fidelity diagnostics, label-free
  inference, validation benchmarks, pre-test five-seed final selection,
  restart-safe finalist/execution claims, paired bootstrap reporting for every
  final five-seed graph, and complete required plot assembly;
- a source-bound PRAD Tigris DAG using `reu-aisocial`/`tigris`, absolute worker
  paths, exact numeric dependencies, durable partial-submit recovery, exact-ID
  monitoring/resume/cancellation, and production requests bound to completed
  smoke `sacct` evidence plus conservative source-bound storage headroom.
- the versioned `prad_minimum_durable_storage_v1` profile: the 15-node DAG
  rebuilds paired views, structural targets, and teacher outputs in Slurm
  job-local temporary storage and deletes them on worker exit; completed runs
  retain optimizer-free best/final checkpoints, incomplete runs retain only
  one rolling exact-resume checkpoint, and prediction shards store logits with
  split-row-bound implicit identities.

Still not implemented or accepted:

- a completed real Tigris minimum-storage miniature.
  The first authentic PRAD smoke submission, source commit
  `8bb20be8cb351f0a0fd71f50dabcc03467790161` and jobs `41126`--`41147`,
  failed closed before data/model execution: sorted JSON role keys exposed an
  order-sensitive split-builder check, and the independent Weaver worker read
  the absent split manifest before dispatch. Both defects now have local
  regressions and fixes. A direct installed-Weaver diagnostic then established
  that every auxiliary Lorentz-input derivative is NaN for all-valid, singly
  padded, and multiply padded fixtures alike, while logits, feature gradients,
  and parameter gradients remain finite. The v3 parity contract compares that
  diagnostic topology exactly and keeps all training-required quantities
  strictly finite. The v3 authoritative rerun passed on Tigris at source
  commit `2b3e34c961067f1d458561d67b51b2ffd780704a`. A disk-backed smoke from
  that source was submitted, but complete monitor/resource evidence has not
  been supplied and it cannot validate the new storage profile;
- PRAD's real `reports/data_audit.md`, exact full split and ephemeral-data
  attestations,
  trained checkpoints, experiment results, sealed 500k test metrics, plots,
  and final scientific recommendation. Those are intentionally not fabricated
  locally and require the clean committed source plus the real miniature first;
- all twelve HOSD implementation steps. The migrated plan is authoritative,
  but `src/hlt_classification/hosd/`, its target extractors/caches/teachers,
  taps/heads/probes, auxiliary and feedback training, combinations, metrics,
  confirmation/scale, final selection/seal, and HOSD production DAG do not
  yet exist.

No file in this repository imports `Fresh_check` at runtime. The repository
has an independent Git history. No remote is configured; direct Tigris access
from the implementation session reached the cluster but was rejected at the
required interactive Duo MFA step.

The exact PRAD execution sequence, including clean-source transfer, smoke dry
run, monitored miniature, reviewed resource requests, and separately reviewed
production dry run, is in `docs/PRAD_RUNBOOK.md`.

## Contracts added

```text
scientific HLT profile:
fixed_hlt_v3_track_dominant_proxy/v1

serialized HLT profile:
hlt_classification_hlt_v3_profile_v1

replica policy:
hlt_classification_hlt_replica_manifest_v1

cache shard:
hlt_classification_cache_shard_v1

offline cache manifest:
hlt_classification_offline_cache_manifest_v1

HLT cache manifest:
hlt_classification_hlt_cache_manifest_v1

canonical ParT inputs:
hlt_classification_part_inputs_v1

Weaver adapter/parity report:
hlt_classification_weaver_part_v1
hlt_classification_weaver_part_v2
hlt_classification_weaver_part_v3

training checkpoint:
hlt_classification_training_checkpoint_v2

training report:
hlt_classification_part_training_report_v2

prediction manifest:
hlt_classification_prediction_manifest_v1

evaluation report:
hlt_classification_evaluation_report_v1

metrics:
hlt_classification_metrics_v1

source snapshot and compact campaign:
hlt_classification_source_snapshot_v1
hlt_classification_baseline_campaign_spec_v1

execution evidence:
hlt_classification_storage_measurement_v1
hlt_classification_slurm_resource_evidence_v1
hlt_classification_runtime_environment_v1
hlt_classification_task_attestation_v1
hlt_classification_submission_job_record_v1
hlt_classification_submission_ledger_v1
hlt_classification_monitor_report_v1
hlt_classification_resume_plan_v1

smoke interruption evidence:
hlt_classification_training_interruption_evidence_v1

final-test locks:
hlt_classification_finalist_lock_v1
hlt_classification_final_test_execution_lock_v1
hlt_classification_final_test_execution_claim_v1

PRAD campaign surfaces:
hlt_classification_prad_split_manifest_v1
hlt_classification_prad_paired_view_v1
hlt_classification_prad_structural_targets_v1
hlt_classification_prad_teacher_outputs_v1
hlt_classification_prad_relation_v1
hlt_classification_prad_checkpoint_v1
hlt_classification_prad_model_checkpoint_v1
hlt_classification_prad_ephemeral_dataset_v1
hlt_classification_prad_training_report_v1
hlt_classification_prad_training_v3
hlt_classification_prad_prediction_manifest_v2
hlt_classification_prad_teacher_prediction_manifest_v1
hlt_classification_prad_evaluation_report_v1
hlt_classification_prad_campaign_spec_v2
hlt_classification_prad_submission_ledger_v1
hlt_classification_prad_task_attestation_v1
hlt_classification_prad_resource_evidence_v1
hlt_classification_prad_final_selection_v1
hlt_classification_prad_runtime_validation_v3
hlt_classification_prad_statistical_report_v2

PMARD/Scouting reviewed surfaces:
hlt_classification_scouting_schema_v4
hlt_classification_fitted_strict_matcher_v1
hlt_classification_scouting_feature_audit_v2
hlt_classification_pmard_row_selection_v1
hlt_classification_pmard_selective_assignment_shard_v1
hlt_classification_pmard_selective_assignment_manifest_v1
hlt_classification_pmard_ephemeral_assignment_v2
hlt_classification_pmard_matcher_report_v2
hlt_classification_pmard_matcher_validation_v2
hlt_classification_pmard_full_role_coverage_v2
hlt_classification_pmard_training_report_v4 (legacy readable parent)
hlt_classification_pmard_resume_checkpoint_v4 (legacy readable parent)
hlt_classification_pmard_training_report_v5
hlt_classification_pmard_resume_checkpoint_v5
hlt_classification_pmard_lock_v4
hlt_classification_pmard_campaign_spec_v9
hlt_classification_pmard_t100_kd_sweep_spec_v2
hlt_classification_pmard_t100_kd_targets_v1
hlt_classification_pmard_t100_kd_sweep_report_v2
hlt_classification_pmard_t100_kd_sweep_ledger_v2
hlt_classification_pmard_kd_followup_spec_v1
hlt_classification_pmard_kd_followup_report_v1
hlt_classification_pmard_kd_followup_ledger_v1
```

Changing registered-v1 event, substream, replica-cycle, degradation, or cache
semantics requires a new applicable scientific or serialized contract version.

## Readiness matrix

| Capability | Status | Evidence |
|---|---|---|
| Package and documentation | Passed locally | `tests/test_scaffold.py` |
| JetClass data foundation | Passed locally | `tests/test_data_foundation.py` |
| HLT-v3 mechanism contracts | Passed locally | `tests/test_hlt_v3.py` |
| Donor-v1 deterministic parity | Passed locally | frozen array/diagnostic hashes |
| Authenticated cache construction/resume | Passed locally | `tests/test_cache_pipeline.py` |
| Cache corruption/cross-profile rejection | Passed locally | `tests/test_cache_pipeline.py` |
| HLT shard/batch layout invariance | Passed locally | bitwise concatenated arrays |
| Canonical ParT transforms | Passed locally | `tests/test_part_inputs.py` |
| Wrapper/parity logic | Passed with interface-faithful test double | `tests/test_particle_transformer.py` |
| Installed-Weaver FP32 parity | Passed on Tigris | v3 report at `2b3e34c` matched logits, masks, model state, training-required gradients, and auxiliary Lorentz-gradient topology |
| Real Tigris minimum-storage miniature | Attempt 6 completed through E4 | core-screen array `42278` reached Stage B in E5--E10 and exposed an optimized-SDPA mixed-freeze backward incompatibility |
| Baseline training/resume | Passed locally | `tests/test_training_engine.py` |
| Inference and metrics | Passed locally | `tests/test_inference.py`, `tests/test_metrics.py` |
| Production DAG dry run | Passed locally | `tests/test_campaign.py` |
| Source-drift/all-reused validation | Passed locally | `tests/test_campaign.py` |
| Failure injection and exact-ID recovery | Passed locally | `tests/test_campaign.py` |
| Worker static contracts | Passed locally | `tests/test_campaign.py` |
| Full production authorization | Not authorized | Transfer Block 8 |

## Donor provenance

Inspected donor commit:

```text
bfcda5bd6fd48037ab198bcca1eadf6f4d87e131
```

Block-4 cache references were inspected, not copied unchanged:

```text
jetclass_fresh/hlt_cache.py
5631ac115a4f352739492e0db06c5e9c3111fa52d522d273243504e0731823c4

teacher_logit_reco/relation_expert_token_bridge/hlt_cache.py
57d0dde4e4aad5aa622fc62c2020954ae69a0108cf89178d7304dabd2f6345e4
```

Exact migration details for all blocks are in `docs/LEGACY_SOURCE_MAP.md`.

## Verification evidence

Validated with bytecode and pytest caches disabled:

```text
python:
C:\Users\22rya\miniconda3\envs\tagging-hlt\python.exe

focused Block 4:
8 passed

focused Block 7:
8 passed

complete discovered suite:
205 passed, 14 warnings

focused PRAD suite:
60 passed

focused PMARD/Scouting suite:
54 passed
```

The Block-4 suite proves byte-identical clean versus interrupted/resumed
offline and HLT artifact trees, fail-closed corruption and lineage drift,
cross-profile rejection, exact parent-role validation, deployable-artifact
hygiene, and bitwise HLT array equivalence across different source shard,
output shard, and processing-batch layouts.

No real Tigris data, GPU, or installed Weaver path was exercised in this
block. The local environment has PyTorch 2.5.1 but no Weaver installation.

The current PMARD/Scouting focused evidence is 54 passing tests. It includes exact
uninterrupted-versus-resumed state equivalence, a deliberately poor model that
still completes, nonfinite failure injection, bitwise batch-invariant
prediction shards, all frozen metric edge cases, full-endpoint authorization
rejection with full-role count-only lineage, recursive FP32 representation KD
under BF16 inputs, durable interval-mean history, batch-shell signal delivery,
bounded cross-file reads, sparse validation/checkpoint cadence, and an
end-to-end zero-alpha selection through representation KD, anchored companion
training, and the next dual-teacher generation student.
The fitted-strict additions authenticate the canonical model bundle, reproduce
reference scores/margins/confidences numerically, exercise both rectangular
assignment directions, private dummies, post-assignment rejection, strict
PID/charge gates, wrapped phi, invalid PID, native lost-track boundaries,
one-to-one integrity, unmatched p4 preservation, artifact tampering, and the
online PMARD-stream dispatch.

The campaign portion includes source drift after an
all-reused graph, cross-campaign lineage rejection, storage insufficiency,
exact dependency IDs, task-byte corruption, dry-run nonmutation,
failed-descendant recovery, stale exact-job cancellation lineage, and
final-test lock authorization. Python compilation, all 21 PRAD script `--help`
paths, all 11 shell syntax checks, `git diff --check`, and the whitespace audit
pass.

The PRAD-focused suite covers exact split constants and proportional/disjoint
construction, Hungarian thresholds and pair masks, C/A targets, symmetric
relations, masked centering/bounds, exact zero-gate parity, class-gradient
flow, replica-aligned target selection, frozen teacher behavior, HLT-only
inference, test locks, exact checkpoint state, frozen primary metrics,
train-only statistics, source-bound campaign guards, numeric dependencies,
partial-submit reuse, smoke-bound production resources, dense teacher-output
shape/storage validation, accumulated training time, semantic-only causal
isolation, teacher semantic validation, exact final-claim recovery, exact
array-element attestations, and synthetic installed-interface runtime
attestation. Minimum-storage coverage additionally proves in-memory
reconstruction does not publish arrays, task-local cache paths cannot fall
inside the campaign/project tree, completed training removes full optimizer
state while retaining loadable best/final model-only checkpoints, and compact
predictions bind identities through authenticated split row ranges. The two
Tigris-derived regressions prove split construction is
invariant to canonical JSON key sorting and independent Weaver dispatch does
not touch a split artifact. A direct installed-Weaver run at corrected commit
`22c15fc2051f103e50361beb2759731cb956dc8a` then reported zero maximum error
for logits and feature/parameter gradients but a nonfinite Lorentz-input
gradient difference. This is consistent with the fixture's multiple identical
padded zero four-vectors. The v3 contract instead authenticates exact nonfinite
topology for that unused derivative while requiring every training surface
finite; its Tigris run at `2b3e34c` passed with exit code zero.

The first minimum-storage miniature at `288aafc` authenticated split, audit,
installed-Weaver parity, CUDA PRAD runtime, and task-local train statistics in
jobs `41820`--`41824`. E0 job `41825` completed training and published compact
model checkpoints, then failed closed when evaluation restored Weaver's
registered tensor `_counter` buffer by assigning a Python integer. The restore
helper now mutates tensor counters in place while preserving legacy integer
counters, with a regression that proves registered-buffer identity and value.

The replacement miniature at `6bcda2d` then completed split, audit, parity,
runtime, statistics, and the entire E0 train/evaluate task in jobs
`41967`--`41972`. E1 job `41973` failed before training because its CLI passed
the intended `model_factory` keyword to an engine signature that still named
the parameter `model`, even though the engine body already invoked
`model_factory()`. The signature is aligned, the CLI now binds its complete
keyword set against the real engine signature in a regression, and a tiny
full-budget teacher test exercises factory construction through compact
best/final checkpoint publication and rolling-checkpoint removal.

The `58ee660b` miniature passed the repaired teacher path, oracle, and E3/E4.
Core-screen array `42015` then failed E5--E10 during their relation-only Stage
A with PyTorch `RuntimeError: LSE is not correctly aligned (strideH)`. Those
are exactly the pair-supervised graphs; E3 begins directly in all-trainable
Stage C and completed. The Stage-A loss had anchored zero-valued hard/KD
placeholders to final logits, unnecessarily asking installed CUDA attention to
backpropagate a zero gradient through frozen later blocks while only its
additive bias path remained differentiable. The zero anchor now uses the
pre-attention relation tensor. A regression requires final logits to receive
no Stage-A gradient, and runtime-validation v2 reproduces the registered
freeze schedule on installed CUDA before E0.

The next `d2237422` miniature completed split, installed-Weaver parity, data
audit, and train statistics. Its runtime-validation subprocess returned exit
code zero in job `42084`, which establishes that the new Stage-A CUDA
backward attestation passed. The task wrapper then rejected that valid report:
it requested the v2 contract name while the reusable content-hash validator
still defaulted to schema version 1. The validator now accepts an explicit
expected schema version while preserving version 1 as its default, the PRAD
runtime worker explicitly requests version 2, and a regression exercises the
real dispatch path with a content-authenticated v2 report.

The `f90cf5fe` miniature then completed its runtime worker, E0 baseline, E1
teacher, E2 oracle, and E3/E4. Core-screen E5--E10 all failed after the same
roughly four-minute interval with
`RuntimeError: LSE is not correctly aligned (strideH)`. Their shared timing
and exclusive pair-supervised stage schedule identify the transition from the
repaired relation-only Stage A to partially unfrozen Stage B as the failure
surface.
Stage B intentionally freezes early ParT blocks while training later blocks
and the differentiable relation bias; this differs from both Stage A and E3's
fully trainable Stage C. Training contract v3 records
`math_for_stage_b_mixed_freeze_v1`: CUDA Stage B selects PyTorch's reliable
math SDPA backend, while Stage A, Stage C, CPU, and inference retain automatic
selection. Runtime-validation v3 now performs a full mixed-freeze Stage-B
forward/backward with the production loss and requires finite nonzero
gradients before E0.

## Historical PRAD next task (superseded as the active conversation task)

Commit the Stage-B backend policy and runtime-validation v3, cancel only the
pending descendants of `smoke_min_f90cf5fe` through its v2 ledger, transfer
the exact fix commit to Tigris, and execute a fresh PRAD real miniature exactly
as `docs/PRAD_RUNBOOK.md` specifies. Runtime-validation v3 must authenticate
the mixed-freeze CUDA backward before E0; the miniature must still complete
training, selection, locked-test, and aggregation.

```bash
export PYTHONNOUSERSITE=1
python -s scripts/validate_weaver_parity.py --device cpu
```

Do not mark PRAD production readiness accepted until every new smoke task
attestation authenticates and measured resource/storage evidence is captured.
A full PRAD campaign remains unauthorized in this handoff.

## Exact active next task: corrected PMARD pilot

Commit and push the prefix-import recovery utility, update the clean Tigris
checkout to that exact commit, create a fresh pilot spec, import the compatible
completed prefix from `pmard_pilot_3f379cb66398fcc8`, inspect the recovery
dry run, and submit from `teachers` onward. Do not reuse failed teacher or
descendant outputs. Cancel only the old campaign's exact dependency-doomed
jobs through its own submission ledger.

## HCWDL local implementation closure (2026-08-08)

The first real smoke reached and completed all six endpoint qualifiers, but
Tigris represented the submitted `--hold` gate as `JobHeldAdmin`; the ordinary
submitting user could not release it. The same execution also demonstrated
that a campaign bound to the shared main checkout fails if another experiment
pulls that checkout mid-DAG. Campaign/command-plan contracts advance to v3/v2.
Creation and submission now bind the exact invoking worktree, the initial
submission stops after endpoint qualification, and
`continue_hcwdl_campaign.py` validates the complete attested prefix plus the
human acknowledgement before submitting the gate and downstream ladder with
no Slurm hold. This correction passes the 58 focused orchestration tests and
the complete 311-test suite locally; it still requires a fresh exact-commit
Tigris smoke.

The active conversation task is now the local implementation of
`docs/plans/HIGH_COVERAGE_COLD_WARM_DISTILLATION_LADDER.md`. The previous PRAD
and corrected-PMARD operational next tasks above are historical context and are
not authorization to contact Tigris in this task.

All six locally implementable HCWDL blocks now exist:

1. The exact high-coverage donor runtime and three fitted resources are
   packaged under `src/hlt_classification/scouting`; every copied donor file is
   recorded in `docs/LEGACY_SOURCE_MAP.md` at exact clean donor commit
   `64be1a82f11f42949fdffa639a869ccea2528bfa`.
2. Cross-fitted empirical matching, native-index restoration, lost-track
   exclusion, lexicographic one-to-one assignment, 18 confidence diagnostics,
   compact int16/uint16 shards, complete-role manifests, deterministic sampled
   recomputation, and strict sub-10% dustbin authorization are implemented.
3. Shell Exact, Shell Soft, and HC Exact repair all 21 fields. Shell Exact has
   confidence-warped intermediate alphas, byte-exact D0, exact assigned D100,
   exact HLT dustbins, deterministic identity-bound discrete switches, and a
   fixed HLT skeleton.
4. The immutable 23-node cold/warm graph, HLT/D25/D50/D75/D100/TOFF domains,
   fresh/warm initialization, one/two-teacher FP32 KD, recipe-bound AdamW
   epsilon and microbatch accumulation, 60 passes, every-pass validation,
   AUC/CE/logR/earliest selection, final and selected checkpoints, and exact
   resume are executable.
5. Smoke/pilot/midscale500k/midscale1m/midscale2m/production DAGs, source-byte reauthentication, split validation,
   locks, fixed endpoint qualification, exact human acknowledgement, 55-row
   confirmation registry, reporting, one-claim final evaluation, exact-ID
   recovery/cancellation, resource/storage evidence, and Slurm command
   generation are implemented. Submission stops after the six diagnostics;
   after inspection and lineage-bound acknowledgement, a separately invoked
   continuation validates the complete prefix and submits the remaining DAG.
   A measured non-executable candidate
   spec now hashes the exact future Slurm commands independently of the
   enclosing spec; submission authorization v3 binds that command plan and
   exact resource-request hash, and the executable spec must reproduce both.
   The first explicitly authorized smoke miniature uses conservative bootstrap
   requests without claiming they were measured; pilot/production still
   require a genuine measured resource profile.
6. Donor parity, tamper/failure guards, assignment/repair/graph/training tests,
   all CLI help surfaces, a complete bounded local smoke, and a full planning
   pilot dry run are implemented locally.

The primary two-teacher ladder decision was locked on 2026-08-08 from the
completed `pmard_kd_followup_b8a493547de8bd7e` results: 60-pass maximum,
validation every pass, macro-AUC-first checkpoint selection, dual-teacher peak
LR `3e-4`, and CE/predecessor/privileged weights `0.25/0.40/0.35` with
privileged temperature `2`. `HCWDL_RECIPE/v3` now locks the complete primary
profile: single-teacher CE/KD `0.25/0.75`; privileged/HLT temperatures `2/1`;
all peak LRs `3e-4`; effective/microbatch 256 with accumulation one; fixed
AdamW and five-percent warmup/cosine/minimum schedule semantics; authenticated
sqrt-inverse train-count class weights; constant coefficients; and the warm
label-only control. D0 teaching M1 is routed to the HLT temperature one while
sole privileged D teachers use two. A changed value requires a separately
identified `registered_ablation` and cannot silently alter the primary.

New contract families are `HIGHCOV_MATCHER_RESOURCES/v1`,
`HIGHCOV_DENSE_ASSIGNMENT_{SHARD,MANIFEST}/v2`,
`HIGHCOV_DENSE_ASSIGNMENT_LOCK/v1`,
`HIGHCOV_ASSIGNMENT_RECOMPUTATION_AUDIT/v1`, `HCWDL_ARTIFACT/v1`,
`HCWDL_{NODE_SPEC,GRAPH,TRAINING_REPORT,CHECKPOINT_SELECTION}/v1`,
`HCWDL_RECIPE/v3`,
`HCWDL_EPHEMERAL_{VIEW,TARGET}_BANK/v1`, `HCWDL_LOCK/v1`,
`HCWDL_EXECUTION_CLAIM/v1`, `HCWDL_ENDPOINT_{QUALIFICATION,DIAGNOSTIC_ACK}/v1`,
`HCWDL_{SCREEN,CONFIRMATION,FINAL}_AGGREGATE/v1`,
`HCWDL_FINAL_{EVALUATION,EVALUATION_MANIFEST}/v1`,
`HCWDL_CAMPAIGN_SPEC/v6` (v3-v5 readable), `HCWDL_COMMAND_PLAN/v2`,
`HCWDL_SUBMISSION_LEDGER/v2`, `HCWDL_MONITOR_REPORT/v1`,
`HCWDL_{RESOURCE_PROFILE,STORAGE_ESTIMATE}/v1`,
`HCWDL_SUBMISSION_AUTHORIZATION/v6` (v3-v5 readable),
`HCWDL_CACHE_MINIATURE/v1`, and `HCWDL_LOCAL_SMOKE_REPORT/v1`.
PMARD training/resume contracts advance to v6 for explicit microbatch,
accumulation, and Adam-epsilon semantics; validators retain v4/v5 report
compatibility.

Local verification evidence at the final post-handoff audit:

- repository-wide suite: 299 passed with the same 14 Matplotlib/Pyparsing
  deprecation warnings;
- all 51 HCWDL/high-coverage Python surfaces compile, all 25 thin CLIs have
  tested help surfaces, and Slurm shell invariants pass;
- standalone bounded smoke:
  `%TEMP%/hcwdl_local_smoke_final_500de1f8e45f44ba8fb84b0eebca898a/smoke_report.json`,
  content hash
  `8f3157b434d6b371f3513de83ffa175d13cbb9e5a9d6e9a196849013b13dae0e`;
  all 23 nodes and five alphas completed, endpoint exactness passed, and the
  final-test role was not accessed;
- the pilot dry-run regression renders exact 300,000/100,000/100,000 roles,
  every dependency and resource class, uncapped source arrays, the held human
  gate, and no submission.

The placeholder/donor-import/development-path static scans were clean and
`git diff --check` passed with line-ending notices only. During that local
implementation audit, no SSH, Slurm, remote push, ROOT write, or Tigris action
occurred.

The first separately authorized Tigris smoke attempt reached source audit but
stopped at split validation (`source_audit` job 56135 completed; `splits` job
56136 failed before matching or training). The cause was an order-sensitive
role-key check: immutable JSON publication canonically sorts object keys while
the validator required the pre-serialization insertion order. Split role maps
are now validated by their exact key set and consumed in the canonical
`SPLIT_ROLES` order. A regression publishes, reloads, and validates a real
split manifest through the immutable JSON path. Focused HCWDL/split tests pass
69/69 and the repository-wide suite remains 299 passed with 14 warnings.

The fresh smoke at corrected commit `68406fd9c9077ead912555fe85ff7fb0e6208d00`
then exposed a second adapter-only split failure (`source_audit` job 56179
completed; `splits` job 56180 failed). The authenticated source-manifest file
rows contain audit metadata such as `baseline_selected_entries`, branch count,
and branch-schema hash in addition to the six fields that define
`SourceFileRecord`; HCWDL had incorrectly expanded the richer row directly
into that narrower dataclass. A reusable explicit projection now preserves
the six split-identity fields while the source manifest and all of its richer
metadata remain independently content-hash authenticated. Both HCWDL and the
canonical split-building CLI use that projection, with a rich-row regression.
Focused HCWDL/split tests pass 70/70 and the repository-wide suite passes 300
with 14 warnings. That fix was committed as `659ebb6294038497d5da874f92beba05927021e8`
and exercised by the next fresh smoke identity.

The next smoke at commit `659ebb6294038497d5da874f92beba05927021e8`
passed both source/split boundaries and produced every train/validation
assignment shard, then stopped while aggregating job 56236. Real HLT data can
contain category flag rows that are not exactly one-hot; the established
decoder represents these tokens as category `-1`, and the broad matcher gate
correctly forces them to the HLT-preserving dustbin. Assignment v1 counted
these tokens in the dense HLT skeleton but had only five category counters,
making its own conservation equation impossible. Assignment shard/manifest
contracts advance to v2 and now record `unclassified_hlt_tokens` explicitly:
five-category visible totals plus unclassified equal all visible tokens,
five-category assigned totals equal all assigned tokens, and an unclassified
token is forbidden from carrying an offline assignment. Dustbin fraction
continues to cover the complete HLT skeleton, including these tokens, so the
strict `<10%` authorization criterion is unchanged.
Focused matcher/HCWDL tests pass 64/64 and the repository-wide suite passes
301 with the same 14 warnings. The v2 fix requires an exact clean commit and a
fresh Tigris smoke identity; v1 assignment shards must not be relabeled or
reused.

The v2 smoke then completed assignment manifest publication and the strict
assignment lock (`56351` and `56352`) before cache miniature job 56353 exposed
a separate bounded-stream bug. The miniature supplied the authenticated
4,096-row validation selection and also asked the PMARD stream for a
class-balanced `max_rows=4096`; that second uniform per-class quota is not the
same as the selection's authenticated natural class proportions, so it
discarded selected rows and could not cover the promised bound. PMARD streams
now expose an explicit `stream_prefix` bound policy. The cache miniature uses
that policy for D0 and D100, taking exactly the requested deterministic prefix
after row-selection filtering without rebalancing or changing training
semantics. The original `class_balanced` policy remains the default for all
existing callers. A skewed-class regression proves the prefix reaches its
exact bound where the former second quota could not.
Focused cache/stream/HCWDL tests pass 47/47 and the repository-wide suite
passes 302 with the same 14 warnings. The fix still requires a clean pushed
commit and a fresh smoke identity.

## Exact active next task: separately authorized HCWDL Tigris validation

Resolve and sign the immutable optimization recipe from independent evidence;
push an exact clean commit only under a separate authorization; run installed
Weaver parity and a genuine production-worker Tigris miniature; capture
measured RAM, walltime, I/O, GPU, and storage evidence; build a measured
resource profile and a fresh complete dry run; then request explicit pilot
submission authorization. Do not claim runtime acceptance or release the held
endpoint gate until those steps and the six real endpoint diagnostics exist.
