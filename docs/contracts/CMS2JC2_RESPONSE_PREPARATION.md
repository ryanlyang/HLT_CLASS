# CMS2JC2 response: staged calibration, confirmation and transfer v1

Authority: [three-family implementation plan](../plans/CMS_CALIBRATED_JETCLASS2_HLT_RESPONSE_THREE_FAMILY_IMPLEMENTATION_PLAN.md).

Status, 2026-09-17: **local implementation and staged queue tooling complete**.
Full-science submission still requires genuine SPORC acceptance and measured
resources, exact pushed source and a reviewed authorized plan. Local tests do
not establish those facts or authorize a reduced campaign.

## Implemented surface

The new `src/hlt_classification/cms2jc2_response/` namespace contains:

| Modules | Implemented responsibility |
| --- | --- |
| `contracts`, `provenance` | Content-hashed v1 artifacts, source-pinned execution, candidate registry, explicit physical-review gate |
| `audit`, `splits` | Latest-cycle CMS header audit, historical-training-only population authentication, file-disjoint roles, exact nested budgets, JC2 source authentication |
| `bridge`, `readers` | Raw shared-field adapters and role-bounded reads; separate JC2 offline-only branch capability |
| `rng`, `features` | Label-independent keyed draws, basic/neighbourhood predictors, fit-only preprocessing |
| `association` | Partial/grouped exact component set-cover, dustbins, deterministic structural/search limits and explicit unresolved status |
| `families`, `residuals` | Portable table/spline/tree conditional models, separate residual calibration, quantile/copula primitives |
| `assumptions` | Exact user-authorized provisional native convention contract, distinct from verified review |
| `records`, `parallel`, `topology`, `response`, `worker` | RAM-only weighted calibration records, deterministic file quotas, topology exposures, complete particle generation, shared bounded/full-fit engine |
| `metrics`, `conditioning`, `evaluation`, `selection` | Six-block summaries, fit-only scales, offline-defined conditional populations, paired file bootstrap and one-SE/qualification primitives |
| `support`, `transfer` | Fitted aggregate support cells and locked offline-only JetClass2 transfer diagnostics |
| `storage`, `graph` | Atomic quota guards and registered 71-task scientific graph |
| `diagnostics`, `plots` | Nonselecting tails/joint tails, occupancy, file-disjoint two-sample test and bounded physical example SVG/JSON |
| `campaign`, `submission`, `orchestration`, `tasks` | Staged creation/dry/live submission, immutable claims/receipts, exact-ID monitoring/reconciliation and terminal-only recovery |
| `measurement`, `synthetic_acceptance` | Genuine Linux CPU process-tree measurements and explicitly synthetic mechanism checks within real acceptance |
| `preparation` | One isolated source-bound, read-only CPU metadata job; no scientific fits or automatic follow-on submissions |

The set-response generator is now integrated and has bounded real fit-role
development checks. These are not evidence of detector closure or a genuine
production-worker miniature. Association limits remain development defaults;
a measured fit-role profile freezes production limits. The staged coordinator
and diagnostics are implemented; genuine CPU acceptance remains unperformed.

`scripts/cms2jc2_response.py` currently provides `headers`,
`create-preparation`, `submit-preparation`, `run-preparation`, `assumptions`,
`preview`, `graph`, `readiness`, `create-acceptance`, `create-science`,
`create-confirmation`, `dry-run`, `submit`, `run-task`, `monitor`, `results`,
`reconcile`, and `recover`. `readiness --spec` validates an actual source-pinned
stage; without a spec it reports tooling readiness but cannot claim science
readiness. All live submission commands default to dry.

## Artifact and access rules

Artifacts use `CMS2JC2_RESPONSE_<KIND>/v1`, schema version 1, canonical content
hashes, named parent hashes, and `final_test_accessed: false`. Immutable JSON
publication is reused from `data/cache_contracts.py`. Source records bind exact
module, donor, plan, CLI, worker and package-declaration bytes. Executable records
require a clean dedicated checkout whose commit is present in a recorded remote
branch. Local dirty-checkout evidence is explicitly non-executable.

The preparation spec binds source, explicit raw roots, and exact input-manifest
bytes. Its output receipt binds all emitted artifact checksums. A complete
receipt must validate before reuse; file existence alone is insufficient.
Changing source, inputs, or scientific meaning requires new artifacts/root.

CMS calibration derives solely from original historical training files and
replays their exact baseline/mapped selection. Original validation/final-test
particle branches are not read. Response-specific roles are whole-file disjoint.
Small-budget masks must cover the exact fit-file registry, have the registered
counts, nest within the next budget, and remain inside authenticated eligible
rows. A missing membership cannot default to the full file population.

JC2 transfer uses the registered `TRAIN_500K` profile. Its reader requests only
`jet_nparticles` and the shared `part_*` fields. It does not request labels,
`hlt_matched`, native HLT constituents, or HLT jet coordinates. Existing masks
retain their historical HLT-match conditioning; this is disclosed, not undone.
No JC2 final-test particle access is provided.

## Physical review and explicitly authorized provisional study

The compatibility template starts unresolved. Synthetic tests supply explicitly
synthetic conventions only; they cannot certify production units. A reusable
review requires a named reviewer, source/locator/content-digest evidence for
each decision, documented positive conversions, and explicit displacement-sign
conventions. The core reads refuse an unresolved review.

Still required from the CMS and JC2 producers:

- Stored momentum/energy units and weighted versus unweighted PF four-vectors.
- Displacement units, transverse sign/reference, longitudinal reference, and
  zero/nonfinite/sentinel meanings and tracking applicability.
- Whether CMS displacement significance permits `abs(value/significance)` as
  an uncertainty conversion, and what JC2 uncertainty fields mean.
- Compatible PID/charge categories and the CMS regular-PF/lost-track boundary.

The user explicitly authorized proceeding under likely assumptions on 2026-09-17.
`CMS2JC2_RESPONSE_PROVISIONAL_COMPATIBILITY/v1` registers those decisions with
their own identity. It does not turn an unresolved review into a verified one.
The authorized policy uses GeV four-vectors; CMS cm-to-mm conversion; stored JC2
mm; a JC2 transverse-sign flip into the CMS convention; validity-aware CMS
value/significance uncertainty conversion; regular PF without lost tracks;
and explicit unknown PID states. Zero/missing measurements retain validity masks.

Native four-vectors and vertex references are retained. Known CMS raw-PF versus
JC2 PUPPI-processing and reference-point differences are explicitly disclosed,
not claimed to disappear through unit conversion. Every fitted response and
transfer report carries `physical_status`: provisional, producer-unconfirmed,
and not eligible for a physically qualified transfer claim. A contradictory
producer answer requires new convention/source-bound artifacts, not edits to
existing fitted responses. Provisional semantics do not waive source, resource,
separate-execution authorization, final-test, or storage gates.

## Integrated response and evaluation details

- Calibration and generation use the same ordered merge proposals. Consumed
  particles cannot also be emitted by singleton modules. Singleton multiplicity
  is 0/1/2; a separate jet-conditioned module describes unexplained components.
- Categorical output identity/validity states precede joint continuous response.
  Mass below 1e-6 GeV is the explicit numerical zero-mass state. Continuous
  coordinates include log-pT response, eta/phi shifts, mass, values and log errors.
- Family A's optional linear input-tracking term affects only output tracking
  coordinates, not momentum. Its predictors are bounded by observed fit support.
  Fitted scale predictions use calibrated log-scale limits. Final coordinates
  clamp to a declared location/residual-fit envelope with counted clamp flags;
  the held-out quantile backend separately reports its outer-tail clamps.
  This bounded-support approximation is not a claim of correct extreme tails.
- Calibration records use bottom-hash category/pT/crowding sampling with N/k
  inclusion weights. Parallel work adds equal authenticated-file quotas so
  worker completion order/count does not determine retained records. Default
  cap is 2M per role/module, with 20M as a hard ceiling. Nothing writes those
  raw calibration records to shared storage.
- Source component search has a feasible greedy upper bound, exact cost
  dominance and an admissible per-vertex lower bound. Exhausted or oversized
  components remain unresolved; a greedy solution is never called an optimum.
- Histogram W1 and binned rank correlations expose numerical approximation
  bounds. Fit ranges within 64 float64 epsilons of their scale are explicitly
  treated as numerically degenerate. Qualification cannot call a threshold
  certain when its uncertainty interval plus summary bound crosses that limit.
- Particle-condition cells use offline source/group category, log-pT, abs-eta
  and crowding. Jet observables use offline category presence and median
  particle coordinates. Axes are separate, not a Cartesian grid. Cohort
  populations depend only on offline particles and permitted source groups,
  never observed matches or proxy outcomes. Unmatched tracking remains in
  overall closure without an invented offline conditioning partner. A cell
  with fewer than 1000 real jets, or an observable with fewer than 1000 covered
  real jets, explicitly backs off to the overall comparison.
- All replicas share the same real population. The 200 bootstrap draws resample
  source files jointly, average replica metrics, and warn about few source
  groups. Class labels are permitted only through an explicit held-out diagnostic
  reader switch; neither response fitting nor generation receives them.
- One-SE selection requires all 27 primary rows and six family-best gate
  sensitivities. If the simpler one-SE winner differs from its family best,
  the six tests do not establish its own robustness: this is marked unresolved,
  not silently promoted to a robust calibration.
- The graph allows two 16-CPU fit lanes and four up-to-eight-CPU evaluation lanes,
  at most 64 CPUs combined. Current serial metric/evaluation/visual engines
  actually request one CPU; no idle eight-CPU request is claimed as parallelism.
  Measured resource locks replace proposed envelopes before full submission.
- Publication guards count all files, including failed attempts, toward 12 GiB;
  reports/figures/examples count toward 2 GiB. They check free-space headroom,
  reject path/symlink escapes, and never overwrite a different immutable output.

## Preparation execution boundary

`create-preparation` requires a fresh output root disjoint from both raw roots.
It writes a canonical command plan and dry ledger without invoking Slurm.
`submit-preparation` defaults to dry and, for live execution, requires exactly:

`AUTHORIZE CMS2JC2 RESPONSE READ ONLY PREPARATION`

The request is one CPU-only `debug` job, account `reu-aisocial`, QoS
`qos_tier3`, 4 CPUs, 16 GiB, 2 hours. Partition state is checked before submission;
the scheduler must still accept the account/QoS. This is not a certified site or
full-size resource lock. Dataset visibility from SPORC remains to be tested.

Per the 2026-09-17 user routing change, all subsequent acceptance, science,
confirmation, transfer, reporting, and recovery jobs also request `debug`.
`qos_tier3` is retained; it is a QoS name, not the requested partition.
Only routing changes: scientific contracts and resource envelopes are unchanged.
Old immutable tier3 specifications are not rewritten or silently rerouted;
create fresh specifications and dry plans pinned to the updated source.

Submission uses an exclusive durable intent and `sbatch --parsable` receipt.
An ambiguous acknowledgement stops for reconciliation; it is never blindly
retried. Worker claims reject duplicate writers. No existing job is canceled,
held, reprioritized, or reconfigured. No active environment is installed/upgraded.
The absolute-path worker activates `atlas_kd_sporc`, disables user packages and
bytecode, prepends the environment library directory, and limits thread pools.

Preparation reads CMS scalar selection fields and source metadata/checksums,
not particle arrays. It publishes inventories, memberships, an unresolved
review template, environment versions, and a preparation report explicitly
marked `science_queue_ready: false`. It does not generate proxy data or launch
fits, confirmation, classifiers, or KD runs.

## Staged execution and remaining remote acceptance

Acceptance has 15 tasks and remains CPU-only. It exercises 20k real fit-role jets
in all three low-complexity families, serial/process replay and the largest
probe jet; synthetic merge/split/loss/additional tests supplement, but never
substitute for, those real rows. High-complexity family fits, common metrics,
three-replica evaluation and three association gates use separate 100k probes.
An execution lock derives 1.5x RAM and 2x projected-runtime requests within the
plan envelopes. Invalid or excessive estimates block `create-science` without
discarding data, reducing models, or changing poor-result completion semantics.

Science has 71 tasks: shared metric lock, 27 primary fits/evaluations, family
finalists, six sensitivity fits/evaluations, selection lock, bounded examples,
comparison completion. Confirmation/transfer is a separately authorized nine-task
stage, fixed to one model per family and the selected winner. No stage queues
another automatically. The active plan section 15.1 freezes diagnostic sampling,
resource projections, plot strata and tie-order sensitivity details.

The exact live phrases are:

- `AUTHORIZE CMS2JC2 RESPONSE ACCEPTANCE EXACT PLAN`
- `AUTHORIZE CMS2JC2 RESPONSE SCIENCE EXACT PLAN`
- `AUTHORIZE CMS2JC2 RESPONSE CONFIRMATION EXACT PLAN`

`submit --execute` also requires `--reviewed-plan-hash` matching the canonical
dry plan. Durable intent precedes `sbatch`; exact job receipts bind source/spec,
task and resolved dependencies. Uncertain acknowledgements require explicit
`reconcile --task ... --job-id ...`, which checks scheduler ownership/comment/
source path. Workers verify source and installed numerical-library bytes, claim
their task exclusively and publish output receipts last. Failed attempts remain
on disk and count against the shared cap. `recover --attempt r2
--approved-job-ids ...` refuses all active/pending/unknown jobs, reuses authenticated
completed products and restarts only approved terminal tasks in a fresh attempt.
It does not call `scancel`. Source changes require a new acceptance campaign.

New artifact kinds, all under the same isolated v1 prefix: `CAMPAIGN_SPEC`,
`COMMAND_PLAN`, `SUBMISSION_INTENT`, `SUBMISSION_RECEIPT`, `SUBMISSION_LEDGER`,
`MONITOR`, `TASK_CLAIM`, `TASK_OUTPUTS`, `ALLOCATION`, `TASK_MEASUREMENT`,
`CPU_MINIATURE`, `ASSOCIATION_PROFILE`, `EXECUTION_LOCK`, `NONSELECTING_DIAGNOSTICS`,
`EXAMPLES`, `FAMILY_FINALISTS`, `COMPARISON_COMPLETE`, `CONFIRMATION_CLAIM`,
`CONFIRMATION_REPORT`, `TRANSFER_CLAIM`, and `CAMPAIGN_COMPLETE`. No existing
classifier, matching or ladder artifact is relabelled or modified by this study.

Remaining execution gates:

1. Commit/push only this implementation and use a clean dedicated source checkout.
2. Run/verify the isolated SPORC preparation metadata job and explicit provisional
   compatibility artifact (or verified review if producer answers have arrived).
3. Review and authorize the 15-task real acceptance plan; inspect measured limits
   and association coverage. Remote data visibility/site permissions are not
   certified locally. Low closure qualifies science separately from job success.
4. Review/authorize the full 71-task science plan only after its measured lock
   passes; later separately authorize nine-task confirmation/transfer.
5. If a producer answer contradicts a provisional convention, publish new
   compatibility/source-bound artifacts and rerun affected stages. Never mutate
   old fitted responses or silently claim verified detector transfer.

No live remote jobs have been submitted during implementation. These execution
gates are not discharged by local synthetic tests or tiny real-data previews.
