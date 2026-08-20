# Current Handoff

## HCWDL-U-RKD matched-rung probability ensembles (2026-08-20)

A validation-only post-hoc evaluator now registers fixed equal-weight
`LOGIT + RSET + RREL` probability ensembles at D40, D20, and D0. It performs
no training or weight tuning, reconstructs each exact shared homotopy view
once, authenticates all selected reports/checkpoints, requires standalone
metric parity before combining predictions, binds validation identity order,
and has no final-test capability. The source campaign is not mutated. The
new v1 report contract and thin Tigris worker are documented in
[the ensemble contract](contracts/HCWDL_HOMOTOPY_REPRESENTATION_ENSEMBLE.md).
Focused ensemble/homotopy/training/reporting tests passed 47/47. The complete
repository suite passed 995 tests with nine platform skips and 14 existing
Matplotlib deprecation warnings in 415.98 seconds. CLI help, Python compilation,
and `git diff --check` passed. Real results remain pending execution on Tigris.

The first live ensemble attempt at source `f893ace5` failed before inference:
the evaluator incorrectly required the optional launch-time `report_sha256`
field on future parent logit references. Parent campaigns created before those
reports completed intentionally bind only their registered path and expected
node; the established aggregate authenticates the report when it appears. The
evaluator now follows that exact behavior, still binds the observed report and
checkpoint hashes into its own immutable result, and retains strict equality
when a launch-time hash was present. The repaired focused suite passed 48/48
and `git diff --check` passed.

## HCWDL-U-RKD chained resource recovery (2026-08-17)

The running pilot's `F_RREL_U100` source-recovery job reached pass 58 of 60,
committed authenticated state after update 68,448, and then exhausted its
six-hour allocation. The source-recovery ledger already binds corrected
commit `8de32baf`; the earlier resource-recovery builder incorrectly required
the immutable campaign's original source and therefore could not safely
increase walltime without reverting the runtime repairs.

Resource recovery now accepts the exact prior recovery artifact when and only
when its hash is the recovery parent of the monitored ledger. It preserves
that recovery's effective clean project, commit, and semantic-source hashes,
adds the effective recovery as an immutable parent, and permits only the
resource table to change. Missing, unrelated, or cross-campaign prior
recoveries fail closed. This is an operational recovery-composition repair;
scientific graph, losses, views, target banks, resume cursor, and checkpoint
selection are unchanged.

Failed-closure construction now rejects live jobs only when they belong to the
closure being replaced. A concurrent live task on the independent RKD track
no longer blocks recovery; live failed descendants remain fail-closed and
must still be cancelled by exact ledger IDs before replacement.
Any recovered join retains an authenticated `afterok` dependency on the exact
independent live ledger job, so aggregate/completion cannot overtake the other
track.

Local evidence in `tagging-hlt` with bytecode disabled: 24 focused homotopy,
recovery, and storage tests passed; the complete repository suite passed with
989 tests, 9 expected platform skips, and 14 existing Matplotlib/Pyparsing
warnings in 452.20 seconds. CLI help and `git diff --check` passed.

## HCWDL-U-RKD authenticated storage retirement (2026-08-17)

The running 300k HCWDL-U-RKD pilot at
`hcwdl_u_rkd_pilot_189486bf_r1` reached 41 GiB. A read-only allocation audit
identified 22.218 GiB of rolling resume generations and 15.690 GiB of compact
target payloads; selected and final checkpoint envelopes together occupied
less than 1 GiB. The existing target-cleanup contract authenticated and
removed exactly 160 payload shards from `TOFF`, `F_RSET_U020`,
`F_RSET_U040`, `F_RREL_U020`, and `F_RREL_U040`. Every bank retained its
manifest plus cleanup authorization/completion evidence, both U080 training
jobs remained live, and the campaign root fell from 41 GiB to 29 GiB.

The repository now implements the separately user-authorized operational
retirement of rolling resumes for completed nodes. The new v1 authorization
and completion contracts do not change the v2 campaign, graph, recipe, loss,
view, checkpoint-selection, or final-test semantics. Before deletion the tool
authenticates the immutable campaign, complete 60-pass engine and wrapper
reports, selected and final binary envelopes and all their members, and every
direct downstream consumer's completed wrapper. It freezes every exact path,
serialized byte hash, and byte count; execution revalidates path confinement
and bytes, deletes commits before sidecars before state payloads, is
idempotently restartable, and publishes completion evidence. Reports,
selected/final envelopes, active-node resumes, and still-needed target banks
are never retirement members.

The immediate eligible set is `F_RSET_U020`, `F_RSET_U040`,
`F_RREL_U020`, and `F_RREL_U040`. U060 remains ineligible until its direct
U080 consumer completes. No local command deleted a Tigris resume and no
scheduler mutation was issued by implementation.

Local evidence under the established `tagging-hlt` environment with
`PYTHONDONTWRITEBYTECODE=1`, repository `src` on `PYTHONPATH`, and pytest
caches disabled:

- pre-change focused resume/homotopy suite: 32 passed;
- final focused storage/resume/homotopy suite: 37 passed in 5.02 seconds;
- complete repository suite: 985 passed, 9 expected platform skips, and 14
  existing Matplotlib/Pyparsing warnings in 427.96 seconds;
- cleanup CLI help and `git diff --check`: passed.

The remaining action is to commit and push this operational amendment, create
one clean detached Tigris worktree, run the authorization-only pass for the
four eligible nodes, inspect its byte total, and then execute the exact
phrase-bound retirement. U060 retirement and U060 target cleanup should be
run only after both U080 consumers finish.

## HCWDL-U-RKD resume-publication performance repair (2026-08-17)

The detailed U040 phase logs identified resume publication as the remaining
dominant runtime cost.  At RREL pass 53 the optimizer required 209.241 seconds,
validation 5.773 seconds, calibration 0.001 seconds, and the representation
diagnostic 0.353 seconds, while `resume_commit` alone required 288.235 seconds.
The job therefore spent more time repeatedly authenticating checkpoint state
than optimizing the model.  RSET reached a valid completed-pass-60 generation
before timeout; RREL retained a valid completed-pass-53 generation.

The serialized resume artifact and every-pass boundary are unchanged.  Each
generation is still fully inventoried, serialized with the complete model,
representation heads, Adam state, scheduler, sampler, RNG, validation history,
selection, calibration, and lineage; its state, sidecar, and commit are still
hashed, atomically linked, fsynced, and byte-verified before continuation.  Two
independently authenticated generations remain retained.

The execution repair removes redundant validation work only inside one live
training process.  The process now carries forward handles for generations
that were already exhaustively authenticated at startup or immediately after
publication.  It uses those handles for old-generation and selected-candidate
pruning instead of torch-loading and logically reinventoring the same retained
states multiple additional times per pass.  A freshly started or recovered
process continues to perform the full independent disk scan, serialized-byte
hash verification, torch load, namespace inventory, tensor logical hashing,
lineage validation, and corruption fallback before restoring any state.

Focused resume/training/recovery coverage passes 50/50.  New regressions prove
that the in-process path performs no redundant generation reload or scan, that
candidate pruning consumes only authenticated handles, and that a subsequent
cold scan independently validates the two durable generations.  Existing
fault injection—including the optimized pruning path—interrupted publication,
two-generation retention, corrupt-newest fallback, and exact resume tests
remain enabled.  No scientific,
serialized-artifact, or contract version changed.  The complete repository
suite passes 980 tests with nine expected platform skips and 14 dependency
warnings in 441.81 seconds.  A source-pinned Tigris recovery remains the
required performance measurement.

## HCWDL-U-RKD diagnostic-buffer recovery correction (2026-08-16)

The first source-pinned runtime recovery confirmed that the preceding cache
and boundary repair is effective.  RSET U020's 300k/100k cache phases fell
from 2,136.530/735.479 seconds to 1,167.599/389.572 seconds (47.9 to 26.0
minutes total), its resumed pass-50 optimizer phase took 159.134 seconds, and
validation took 7.393 seconds.  It then failed closed before publishing the
pass-50 boundary because the new synchronization-free diagnostic mutation
guard treated an expected registered-buffer update as forbidden model-state
mutation.

Production Weaver forwards may legitimately mutate registered runtime buffers;
the historical full-state round trip restored those buffers implicitly.  The
corrected guard now clones only registered buffers on their existing device,
rejects buffer topology/object/metadata replacement, and restores their exact
values after the diagnostic.  Parameters, gradients, Adam state, RNG, trimmer
runtime state, external sampler state, and module modes retain the cheaper
fail-closed identity/version or exact-state guards.  The repair therefore
preserves the original diagnostic's nonmutation guarantee without restoring
the complete parameter and optimizer payload every pass.

Focused training/resume/recovery coverage passes 46/46, including a regression
whose diagnostic forward deliberately mutates a registered buffer and proves
that the exact pre-diagnostic state is restored.  The complete repository
suite passes 976 tests with nine expected platform skips and 14 dependency
warnings in 433.30 seconds.  The failed job did not publish a new resume
generation, so its preceding valid pass-49 generation remains the recovery
source.  No scientific or artifact contract changed.

## HCWDL-U-RKD pilot cache and epoch-boundary repair (2026-08-16)

The first optimized 300k recovery jobs reached valid authenticated resume
generations but exhausted their six-hour allocations: RSET U020 committed
pass 49/update 57,428 and RREL U020 committed pass 52 plus batch 624 of pass
53/update 61,568.  Neither resume store contained an invalid generation or an
orphan file, so the jobs remain exactly recoverable and do not need to restart
from pass zero.

The production phase logs separated the remaining cost.  RSET materialized
the 300k train cache in 2,136.530 seconds and the 100k validation cache in
735.479 seconds; RREL required 1,279.387 and 426.920 seconds respectively.
Slurm reported only 5:49:53 and 5:41:05 total CPU for the two six-hour,
eight-CPU jobs, confirming that the ragged cache builder was still effectively
single-core.  Cache construction was therefore material, but it could not by
itself explain the remaining wall time after optimizer training began.

The execution-only repair removes both sources of avoidable work while
preserving the existing scientific and artifact contracts:

- offline endpoint projection is computed once per source row and reused by
  P0, coupling partitions, and Shell Exact repair;
- Shell Exact copies/converts only the HLT model-projection branches it can
  mutate, rather than duplicating the native-offline, label, and observer
  payload carried alongside them;
- label bounds are checked before device transfer, while validation retains
  its small ordered logit/label result on the GPU and performs one finite check
  and host transfer per role instead of synchronizing once per batch;
- the per-pass identity audit accumulates immutable chunks and concatenates
  once at the pass boundary instead of reallocating a growing array for every
  batch;
- the representation diagnostic uses tensor identity/version guards instead
  of cloning and restoring the complete model, gradient, and Adam state each
  pass; exact checkpoint and resume publication still snapshots full bytes;
- when a new best checkpoint is selected, its already materialized boundary
  snapshot is reused for that pass's resume generation; and
- phase timings now expose optimizer, validation, calibration, representation
  diagnostic, selected-checkpoint, resume-commit, and total boundary costs for
  the next Tigris measurement.

No U/D coordinate, particle value/order, token metadata, target, RSET/RREL
loss, coefficient, stochastic stream, 60-pass schedule, validation metric,
checkpoint rule, or HLT-only deployment behavior changed.  No durable repaired
view was introduced and no contract version was broadened.  Focused
homotopy/repair/training/resume/recovery coverage passes 131/131.  Source
compilation and the complete repository suite pass 976 tests with nine
expected platform skips and 14 dependency warnings in 406.01 seconds.  A
source-pinned Tigris recovery from the two existing valid resume generations
is the remaining runtime measurement; no job was submitted locally.

## HCWDL-U-RKD U/J v2 parent compatibility (2026-08-15)

The first 300k autolaunch preparation correctly located the active U/J pilot
at `hcwdl_uj_v2_pilot_80096d75_auto_r1`, then failed closed before campaign
creation because HCWDL-U-RKD still dispatched every parent through the frozen
U/J v1 campaign validator.  The active twenty-point U/J parent is an exact
`HCWDL_STRUCTURAL_FEATURE_PILOT_SPEC/v2` artifact; changing its contract label
or weakening content-hash validation would be invalid.

HCWDL-U-RKD now dispatches v1 and v2 parents explicitly.  The existing v1
validator remains unchanged.  The v2 consumer validator authenticates the
exact campaign envelope, sealed role boundary, split, selection, base recipe,
all assignment manifests, coupling configuration, coordinate, overlay, graph,
command plan, and every frozen semantic-source hash before the existing
runtime lock/control checks run.  It validates v2 as a consumed parent and
never relabels it as v1 or treats it as an executable child campaign.  This is
lineage compatibility only: no view, target, representation loss, schedule,
seed, checkpoint, resource request, or HLT-only deployment semantics changed.

Focused HCWDL-U-RKD campaign coverage passes 11/11, including exact v1/v2
dispatch and rejection of unknown future parent versions.  The corrected
source still requires a clean commit/push and a Tigris authentication check
against the live v2 parent before creating the immutable 300k candidate.

## HCWDL-U-RKD ragged preprocessing repair (2026-08-14)

The factorized homotopy representation-KD runtime inherited the original
row-wise U/J endpoint projection path.  For every selected row that path
reconverted every ragged offline and HLT branch for the whole source chunk,
making U/D student-view materialization and every non-TOFF predecessor target
bank quadratic in the chunk population.  The same defect described by
`docs/HCWDL_RAGGED_PREPROCESSING_PERFORMANCE_GUIDE.md` therefore applied to
both representation ladders even though each completed RAM cache was already
replayed efficiently during the 60 optimizer passes.

The runtime now converts each required ragged branch exactly once per source
chunk into immutable prepared offline/HLT endpoint records and reuses those
records for every selected row.  Source chunks are built through a bounded
ordered thread pool whose default worker count is the authenticated
`SLURM_CPUS_PER_TASK` allocation (eight for the locked campaign request).  At
most one chunk per worker is in flight, results are emitted in canonical
source/entry order, source-index target partitioning is unchanged, and the
existing identity-authenticated RAM caches remain the only repaired-view
storage.  The TOFF-native target path is intentionally unchanged.  Explicit
phase logs now separate teacher-target preparation, train/validation student
cache construction, and optimizer training so future stalls are attributable.

No endpoint, coupling, U/D coordinate, model input, target, loss, seed,
schedule, checkpoint-selection, graph, or HLT-only deployment semantics
changed.  A 1,024-row repeated-input benchmark measured 4.848491 seconds for
the legacy row-wise projection and 0.229431 seconds for the prepared path, a
21.13x local speedup.  Focused homotopy/RKD/recovery/target/runtime coverage
passes 141/141; the complete repository suite passes 965 tests with nine
expected platform skips and 14 dependency warnings in 474.00 seconds.  All 14
HCWDL-U-RKD CLI help surfaces pass, all 29 contract identities are unique (27
v2 and two v1 under schema v2), source compilation passes, and
`git diff --check` passes with Windows line-ending notices only.  No local or
Tigris job was submitted by this repair; the corrected source still requires
a source-pinned Tigris smoke or exact failed-closure recovery before a 300k
launch.

## HCWDL-U-RKD staged-parent launch correction (2026-08-13)

The implementation-authoritative plan already permits the RSET and RREL
tracks to start once the U/J coupling, coordinate, endpoint-equality, and
graph-recipe locks exist; it requires the U/J logit-only reports only for the
final comparative aggregate. The campaign implementation had incorrectly
strengthened that into a full-parent-completion launch gate.

The parent importer now authenticates the immutable training inputs and M0/
TOFF controls at campaign creation, freezes the canonical expected U/J report
paths and node identities, and allows both representation ladders to train in
parallel with the U/J logit ladder. The aggregate still fails closed until
every expected U/J report exists and authenticates its node identity; any
report already present at creation is additionally hash-frozen. No view,
target, loss, seed, schedule, checkpoint, or HLT-only deployment semantics
changed.

When the parent is still running, its authenticated submission ledger supplies
the exact `campaign_complete` Slurm job ID. Only the child aggregate receives
that external `afterok` dependency; TOFF target construction and both complete
RSET/RREL training paths start immediately. Focused staged-parent, graph,
prerequisite, and recovery tests pass 15/15. The final complete suite passes
959 tests with 9 expected Windows platform skips and 14 dependency warnings.
Python compilation, campaign CLI help, and `git diff --check` pass.

## HCWDL direct offline-to-HLT KD ablation (2026-08-12)

The requested five-fit direct ablation is implemented as a separate
validation-only `HCWDL_DIRECT_OFFLINE_KD_*/v1` campaign. It trains fresh
paired `HLT_CE`, `HLT_LOGIT`, `HLT_RSET`, and `HLT_RREL` models on exact HLT
inputs plus a fresh native-offline `TOFF_CE` teacher. The three distilled
students share one authenticated TOFF train-target bank; no ladder view,
validation target, repaired dataset, or final-test row is used. HLT stochastic
streams are paired, training is unweighted for 60 passes with every-pass
validation and macro-AUC checkpoint selection, and only the four HLT-input
models are deployable.

The immutable five-node graph, combined recipe, pilot campaign/command plan,
target contracts and two-phase resumable cleanup, runtime,
aggregate/completion reporting, thin creation/worker/submission CLIs,
exact-ID monitor/cancel operations, and repeatable same-source failed-closure
recovery are present. All GPU tasks are frozen at 8 CPUs, 96 GiB, six hours,
and one GH200. Source compilation, all six CLI help surfaces, the 17 unique
direct-contract identities, Markdown/link checks through the full suite, and
`git diff --check` pass. Focused direct/representation/homotopy tests pass
39/39. The complete repository suite passes 955 tests with 9 expected
platform skips and 14 pre-existing Matplotlib/Pyparsing warnings in 688.12
seconds. No job has been submitted. A source-pinned Tigris production-worker
miniature and dry-run remain the research-compute acceptance gate before the
300k live submission, as required by `AGENTS.md`.

The first real pilot creation attempts closed two portable-artifact boundary
bugs before submission: the direct campaign initially looked for v5
`scientific_values` outside its authenticated `payload`, then its dry-run
validator compared tuple-valued in-memory teachers with their JSON list form.
The v5 lookup is now contract-correct and the graph producer emits canonical
JSON-native teacher lists before hashing. A create-to-disk-reload validation
regression now exercises the same boundary as the submission CLI; the direct
campaign tests pass 8/8 and `git diff --check` passes. Neither failed attempt
submitted a Slurm job.

The first live direct pilot at `19012e7b` completed authentication, both fresh
roots, and the single shared TOFF target bank. All three distilled students
then failed before their first optimizer update. `HLT_LOGIT` exposed a parent
key mismatch: its authenticated RAM target header carried the correct split
hash, but the runner registered that hash under `split_manifest` instead of
the PMARD engine's canonical `split_manifest_sha256` key. `HLT_RSET` and
`HLT_RREL` exposed an independent cache adapter omission: exact HLT views were
cached without the HCWDL token metadata required by the representation
calibration/training boundary. The runner now uses the canonical split parent
key and requests paired metadata while constructing representation-only HLT
caches; ordinary HLT inference features and all scientific losses remain
unchanged. Regressions cover both boundaries. The direct and shared
representation target/training/production suites pass 88/88. Because the v1
recovery contract is intentionally same-source only, the corrected source
must create a new immutable campaign rather than relabel or requeue the
`19012e7b` artifacts.

The corrected `800574fa` pilot confirmed the logit repair end to end:
`HLT_LOGIT` completed all training and selection. Its RSET/RREL jobs exposed a
second, narrower metadata-stream issue before optimization. Metadata
attachment had also selected the dataset reader's `canonical_order` mode,
which is intentionally restricted to a one-source manifest, while the 300k
train role spans multiple authenticated sources. Direct representation caches
now read only the HLT branches, attach HCWDL metadata, disable the unrelated
one-source canonical mode, and rely on the existing identity-authenticated RAM
cache for deterministic epoch replay. A regression freezes all five reader
arguments (`input_mode=hlt`, metadata on, canonical mode off, no within-chunk
shuffle, one-file interleave). The direct plus shared representation suites
pass 89/89 and `git diff --check` passes. Scientific views, targets, losses,
seeds, and selection are unchanged. As above, corrected execution requires a
fresh immutable source identity under the v1 contract.

The subsequent `d2dfa8f5` pilot confirmed that repair by carrying both
representation jobs through multi-source cache construction and into the
shared representation trainer. `HLT_RSET` and `HLT_RREL` then failed before
their first optimizer update because the trainer's early resume-lineage gate
admitted only the original ascent and homotopy-RKD graph hashes, even though
its execution resolver and terminal report validator already recognized the
direct graph. One shared graph resolver now selects exactly one registered
graph for every execution and is used consistently by resume admission,
training, and report validation; arbitrary graph hashes remain rejected. A
direct regression admits the exact direct hash and rejects a different valid
SHA-256. Focused direct/shared representation coverage passes 99/99, the
complete suite passes 961 tests with 9 expected Windows platform skips and 14
dependency warnings in 429.16 seconds, and `git diff --check` passes. The
failed `d2dfa8f5` jobs published no representation training report and require
a fresh immutable source identity under the v1 recovery contract.

The `e0e1181f` pilot advanced both representation students through one full
natural-population pass, validation, and checkpoint-candidate creation. They
failed before publishing their first durable resume generation because the
generic resume module retained its own older two-graph allowlist. Graph
identity is now centralized in one closed representation-runtime registry
containing exactly the original representation graph, the U/D homotopy-RKD
graph, and the direct offline-KD graph. Training admission, report validation,
resume publication, resume scanning, and resume restoration all consume that
registry. A new regression executes the exact failed boundary end to end for
the direct graph (publish, scan, highest-valid restore, tensor equality) and
rejects an unregistered well-formed SHA-256. Focused resume/direct/training
coverage passes 53/53; the broader direct and representation closure passes
117/117; the complete suite passes 967 tests with 9 expected Windows platform
skips and 14 dependency warnings in 1236.50 seconds. `git diff --check`
passes. No committed resume generation or representation report was published
by the failed jobs, so corrected execution again requires a fresh immutable
source identity under the v1 recovery contract.

The `e8df4690` retry confirmed durable direct-graph resume publication and
advanced both representation students beyond the earlier boundary. RSET
continued training; RREL failed during its first jet/set-only ramp interval.
The frozen schedule activates jet/set representation KD after pass 2 and
relation KD after pass 4. The component builder correctly omitted relation
while its coefficient was exactly zero, but the lower-level RREL scheduler
incorrectly required the relation tensor unconditionally. It now substitutes
an exact zero only while the frozen relation ramp is zero and continues to
fail closed if relation is absent once that ramp is active. Schedule-level and
real `compute_node_loss` regressions cover the pre-relation interval and the
active-relation rejection. Focused schedule/calibration/training/resume/direct
coverage passes 70/70; the complete suite passes 968 tests with 9 expected
Windows platform skips and 14 dependency warnings in 425.81 seconds.
Compilation and `git diff --check` pass. The change affects neither the
scientific coefficients nor RSET; it implements the already-declared staggered
RREL ramp.

## HCWDL-U-RKD prerequisite bootstrap repair (2026-08-12)

The first real Tigris preflight for commit `a407f056` found a valid completed
HCWDL-UJ smoke parent but no reusable representation recipe, architecture
attestation, numerical acceptance, or kernel envelope. The campaign itself
correctly refused to create from absent paths, but the implementation lacked
the promised non-training publication route for those four prerequisites.

`prepare_hcwdl_homotopy_representation_prerequisites.py` and its reusable
module now publish the complete source-pinned prerequisite bundle from the
completed U/J parent plus the frozen historical unweighted HCWDL campaign.
The original exact-hash proposal was rejected by real evidence: the U/J and
historical recipes have different hashes because their evidence, natural
class counts, and row-selection lineage differ. The authorized repair binds
both exact recipes and requires equality of their complete execution-policy
projection while permitting only those declared lineage fields to differ. It
reopens the complete historical checkpoint architecture registry, runs real
installed-Weaver surface parity, numerical and zero-coefficient acceptance,
publishes the v5 recipe, and commits deterministic spectral resources through
the shared binary envelope. It runs zero fits and has no final-test input.
Recipe mismatch fails before creating the output root; later failures leave
immutable partial evidence and require a new root.

Source compilation and `git diff --check` pass. The focused pytest invocation
could not collect in the currently exposed Windows Python because that
interpreter has no NumPy; it failed before running tests. The new prerequisite
tests and the complete suite must run in `atlas_kd_tigris` after this repair is
pushed. No local or Tigris job was submitted by this repair.

## HCWDL common-schema architecture–input factorial (2026-08-11)

The validation-only four-way architecture/input ablation requested after the
HCWDL-UJ smoke is implemented. It registers four fresh, paired, unweighted
CE-only cells: exact HLT and projected-native P0 inputs under either the
canonical unified 21-channel ParT or a new common-schema charged/noncharged
two-stream ParT. The split control retains every visible token exactly once,
routes unknown/unclassified tokens to the noncharged stream, uses two full
eight-block 21-channel encoders plus the canonical TOFF fusion topology, and
handles an absent stream without inventing a particle. Native 19/7 `TOFF`
remains a fifth imported reference and is never relabeled as a factorial cell.

The separate `HCWDL_ARCHITECTURE_INPUT_FACTORIAL_*/v1` family implements the
exact four-cell graph, content-hashed parent/source/spec/command lineage,
process-local one-time HLT/P0 caches, shared sampler and within-architecture
initialization seeds, 60-pass/every-pass validation, macro-AUC checkpoint
selection, all five prespecified factorial effects, runtime parameter-count
disclosure, zero final-test rows, journaled submission, immutable task
attestations, and generic rolling-checkpoint requeue recovery. The Slurm
training request is exactly 8 CPUs, 96 GiB, six hours, and one GH200. Campaign
creation and execution are thin CLIs; no job was submitted locally.

Focused architecture/graph/campaign tests pass 5/5. The complete repository
suite passes 423/423 with the 14 existing Matplotlib/Pyparsing warnings. All
three new CLI help surfaces pass, the eight contract identities are asserted,
Markdown links resolve through the complete suite, and `git diff --check`
passes with Windows line-ending notices only. The next acceptance layer is a
real source-pinned Tigris smoke using the completed authenticated unweighted
HCWDL smoke parent. No donor code or donor commit was used.

## HCWDL-UJ first Tigris dry-run repair (2026-08-11)

The authenticated unweighted parent smoke completed through aggregate job
`80916`, and installed-Weaver CUDA FP32 parity passed for source
`c70a34e4edb515a9636110ad920dd21ff84d2e07`.  The first HCWDL-UJ campaign
creation then completed, but its immediate dry-run validation failed closed
before any of the 101 campaign jobs were submitted.  The immutable graph JSON
reloaded each node's `teachers` tuple as a JSON list, while
`HomotopyNodeSpec.payload()` regenerated a Python tuple; the validator
therefore rejected its own byte-valid artifact as
`HCWDL-UJ immutable local campaign artifacts differ`.

`HomotopyNodeSpec.payload()` now emits a JSON-native teacher list before
hashing or publication.  This does not change the canonical graph hash or any
scientific graph semantics because tuples and lists had the same canonical
JSON encoding; it makes the in-memory contract representation agree with its
immutable reloaded representation.  A regression checks JSON round-trip
identity for all 80 node payloads.  A dependency-isolated execution of the
actual graph module confirms all 80 payloads round-trip exactly, source/test
compilation passes, and `git diff --check` passes.  The local Windows Python
lacks NumPy, so the focused pytest module cannot be collected locally; the
focused and complete suites must be rerun in `atlas_kd_tigris` on the pushed
repair before the smoke is relaunched.  Failed launcher jobs `80998`, `81219`,
`81220`, and `81221` submitted no HCWDL-UJ campaign tasks and do not alter the
completed parent.

## HCWDL dense measured-resource reschedule (2026-08-11)

The pending corrected dense10 and dense5 recovery roots inherited the pilot's
conservative `gpu_single` request of `320G` for 72 hours. Tigris jobs `77534`
and `77546` provide direct completed-rung evidence: D90c/D95c finished in
8,983/9,187 seconds with peak RSS 50,285,376/50,287,104 KiB. The new
`HCWDL_DENSE_RESCHEDULE_SPEC/v1` therefore freezes a right-sized `96G`,
six-hour, eight-CPU, one-GH200 request while leaving the original CPU aggregate
request unchanged.

The reschedule is operational only. It binds the exact prior recovery spec and
live submission ledger, retains the original dense scientific spec, failed
closure, graph, recipe, assignments, teachers, seeds, output root, and rolling
checkpoints, and records the prior ledger's exact IDs as superseded. Arbitrary
resource edits and recursive resizing fail closed. Operators create and
dry-run both replacements before cancelling either old ledger, cancel only
through `cancel_hcwdl_campaign.py`, confirm removal, and then submit the
right-sized closures. No local or remote Slurm job was mutated during this
implementation.

Implementation is in `hcwdl_dense_recovery.py` with the thin
`create_hcwdl_dense_reschedule.py` surface and the existing authenticated
submitter/worker. Both dense runbooks and the recovery contract document the
new path. Focused dense/recovery/CLI tests pass 71/71; the complete repository
suite passes 415/415 with the 14 existing Matplotlib/Pyparsing warnings; both
CLI help surfaces and `git diff --check` pass (Windows line-ending notices
only). No donor code or donor commit was used.

## HCWDL structural-feature homotopy implementation (2026-08-11)

The validation-only structural/feature homotopy in
[`docs/plans/HCWDL_STRUCTURAL_FEATURE_HOMOTOPY_IMPLEMENTATION_PLAN.md`](plans/HCWDL_STRUCTURAL_FEATURE_HOMOTOPY_IMPLEMENTATION_PLAN.md)
is now implemented locally and ready for its required real Tigris smoke. This
supersedes the earlier documentation-only status below; it does not mutate or
relabel any completed primary HCWDL, dense10, dense5, or recovery artifact.

Reusable implementation was added under `src/hlt_classification/scouting/`:
the exact 80-fit graph and recipe overlay; typed P0/D100 endpoint partition;
label-free maximum-cardinality minimum-cost residual coupling; immutable base
shards, switch sidecars, manifests, audits, and locks; the generalized
`V(s,f)` carrier; dedicated homotopy streaming; durable authenticated TOFF
targets; process-local student/teacher caches; per-node training; campaign,
workflow, reporting, source recovery, resource-only recovery, and bounded
synthetic-smoke modules. The existing all-21-field projector is exposed from
`repair.py` without changing Shell Exact v1. Generic PMARD checkpoint
publication now accepts only semantically identical interrupted retries, and
HCWDL training accepts an explicitly validated per-node loss resolver while
retaining the old default behavior.

Thin command surfaces were added for coupling calibration/build/finalization,
TOFF targets, campaign creation/submission/workers, monitoring, exact-ID
cancellation, recovery, resource profiling, installed-Weaver parity, and the
local smoke. Three Slurm workers use the pinned absolute `PROJECT_DIR`, Tigris
environment rules, `exec python -s`, and `USR1@120` on checkpointable GPU
jobs. The smoke/pilot command plan has 101 tasks: 21 infrastructure/reporting
tasks and all 80 fits. Train/validation roles are 4,096/4,096 for smoke and
300,000/100,000 for pilot; both register zero final-test rows and no final-test
task.

The reusable contract index is
[`docs/contracts/HCWDL_STRUCTURAL_FEATURE_HOMOTOPY.md`](contracts/HCWDL_STRUCTURAL_FEATURE_HOMOTOPY.md).
It records 36 unique v1 identities covering coupling/config/calibration/cache,
TOFF targets, coordinates/graph/recipe/locks, node/runtime/aggregate/resource
reports, installed-Weaver parity, completion, and both recovery families.
The operator procedure is
[`docs/HCWDL_STRUCTURAL_FEATURE_HOMOTOPY_RUNBOOK.md`](HCWDL_STRUCTURAL_FEATURE_HOMOTOPY_RUNBOOK.md).
No donor or exploratory file was copied, so there is no donor commit or new
`LEGACY_SOURCE_MAP` entry. The pre-existing unrelated modification at
`.worktrees/hcwdl-rkd-local` was left untouched.

Local evidence from the final tree:

- focused homotopy plus CLI tests: 101/101 passed;
- cross-module HCWDL/PMARD/Shell Exact tests: 144/144 passed before the final
  parity-only addition;
- complete repository suite: 412/412 passed in 142.47 seconds, with one
  unrelated existing PyTorch warning;
- bounded synthetic graph: all 80 cold-start fits completed two updates,
  every loss was finite, teacher-own-domain routing was exercised, and
  `final_test_accessed` remained false;
- 14/14 new CLI help paths, three/three Slurm shell syntax checks, Python AST
  parsing, all documentation links, 36/36 exact contract identities, and
  `git diff --check` passed (Windows line-ending notices only).

Local installed-Weaver runtime parity cannot be claimed because the Windows
development environment does not contain Weaver. The exact next acceptance
step is therefore the runbook's pinned Tigris GPU parity command followed by
the genuine 4,096/4,096 production-worker smoke. No Slurm job was submitted.
After the smoke completes, its exact ledger/monitor/runtime reports must be
converted into the measured resource profile; only then may the runbook emit
the nonmutating 300k dry run and request separate live-pilot authorization.

## HCWDL completed-node recovery idempotency (2026-08-10)

The source-pinned RNG recoveries retrained already-completed dense prefix nodes
(`D90c` in dense10 and `D95c` in dense5). Training finished successfully, but
publication then correctly rejected the newly serialized `final_model.pt`
bytes because an immutable final checkpoint already existed. This was a
recovery orchestration defect, not a model-training or scientific failure.

Primary and dense HCWDL workflows now authenticate and reuse a completed node
before constructing views or launching training. Reuse requires both the PMARD
engine report and HCWDL node report, valid content hashes, exact campaign,
graph, recipe, node, teacher, lock, split, source, and warm-parent lineage, and
byte-identical selected and final checkpoints. A partial, stale, corrupt, or
lineage-mismatched node fails closed. No checkpoint is overwritten, and no
matching, repair, model, loss, seed, schedule, or graph semantics changed.

Focused dense/primary/recovery/resume tests pass 65/65. The complete repository
suite passes 355/355 with the same 14 Matplotlib/Pyparsing warnings, and
`git diff --check` passes apart from Windows line-ending notices. Real Tigris
validation remains: submit a new source-pinned dense recovery from the pushed
fix; it should reuse D90c/D95c immediately and begin D80c/D90c respectively.

## HCWDL structural-feature homotopy implementation plan (2026-08-10)

The new implementation-authoritative supplemental plan is
[`docs/plans/HCWDL_STRUCTURAL_FEATURE_HOMOTOPY_IMPLEMENTATION_PLAN.md`](plans/HCWDL_STRUCTURAL_FEATURE_HOMOTOPY_IMPLEMENTATION_PLAN.md).
It specifies the missing upper bridge from native-offline support to exact
D100 support and an edge/update-depth-matched comparison between a factorized
`U then D` path and a joint `U+D` path. It is additive: completed primary,
dense10, dense5, and recovery artifacts retain their original contracts.

The planning audit found and closed a critical representation ambiguity.
Canonical TOFF is a two-stream 19/7-feature, 90/60-particle oracle, while all
U/J/D students use the unified 21-channel ParT. The plan therefore defines P0
as the exact TOFF-visible particle multiset under the existing 21-field
projection without claiming tensor/adapter identity. It then freezes typed
A/B/K/C_A/C_B/O/R endpoint records, a label-free maximum-cardinality
minimum-cost residual coupling, atomic substitution/removal/insertion edits,
an exact integer train-calibrated switch coordinate, a maximum-200 carrier,
and one `V(s,f)` builder. U100 must be byte-identical to current D100;
factorized D0 and joint J100 must be byte-identical to HLT, including canonical
raw-length metadata.

The factorized and joint paths each contain 20 cold predecessor-KD transitions
plus one born-again M1, use the locked unweighted 60-pass recipe, and are paired
by transition seed. The exact 80-fit registry includes P0 adapter diagnostics,
paired/direct endpoints, a ten-generation D100 chain, and a 21-generation HLT
stationary-depth chain so path gains are not confused with repeated
distillation depth. The initial 300k study is validation-only and structurally
has no final-test task. This step is documentation only: no coupling, graph,
worker, artifact, Tigris submission, or remote mutation was implemented or
authorized. After the independent science/implementation/editorial audit, the
focused high-coverage/HCWDL suite passed 60/60 in the local `tagging-hlt`
environment. The complete suite passed 355/355 with the 14 existing
Matplotlib/Pyparsing warnings plus one local pytest-cache permission warning;
`git diff --check` passed apart from line-ending notices.

## Primary HCWDL source-pinned closure recovery (2026-08-10)

The unweighted 1M campaign completed both D50 branches, then `D25c` job
`60508` and `D25w` job `60518` failed while restoring CUDA RNG byte states
through the old source-pinned checkout. The shared RNG implementation is
already repaired, but the ordinary HCWDL resumer intentionally cannot execute
an immutable parent spec through a different commit.

`HCWDL_FAILED_CLOSURE_RECOVERY_SPEC/v1` now provides the missing proper path.
It authenticates the complete original ledger and immutable Slurm monitor,
forms the exact union of failed descendants, requires every external parent to
be completed and artifact-valid, preserves the original scientific spec and
output root, and executes through a clean pushed repair worktree. For the 1M
failure this reuses D50c/D50w, resumes D25c/D25w independently from their
rolling checkpoints, and continues only their cold/warm descendants and
shared downstream stages. Separate recovery attestations and exact-ID
supersession lineage are retained.

The implementation is in `hcwdl_campaign_recovery.py` with thin create,
submit, worker, and Slurm entry points. The operational contract and commands
are documented in `docs/contracts/HCWDL_CAMPAIGN_RECOVERY.md` and
`docs/HCWDL_CAMPAIGN_RECOVERY_RUNBOOK.md`. Focused recovery/contract/dense
tests pass 33/33, all HCWDL CLI tests pass 44/44, and the complete repository
suite passes 352/352 with the same 14 Matplotlib/Pyparsing warnings. The repair
still requires an exact pushed commit and a real Tigris recovery run.

## GPU-mapped exact-resume RNG repair (2026-08-10)

The live ten-point dense recovery completed D90, then D80 job `77535` exposed
a shared exact-resume defect after a rolling checkpoint was loaded with
`map_location=cuda`. That mapping correctly moved model and optimizer tensors,
but it also moved the serialized CUDA-generator byte states onto CUDA;
`torch.cuda.set_rng_state_all()` requires CPU `ByteTensor` state even for CUDA
generators. This was an execution-only resume failure and did not alter dense
matching, repair, loss, model, sampler, or graph semantics.

Shared RNG restoration now validates every CPU and CUDA state as a
one-dimensional `uint8` tensor, normalizes it to contiguous CPU storage before
any generator is mutated, validates the CUDA device count, and only then
restores all generators. Existing rolling checkpoints and checkpoint contracts
remain compatible, so D80 can resume rather than restart. Regressions simulate
a device-mapped tensor without requiring a local GPU and reject malformed
dtype/shape before generator mutation.

Focused training/resume, PMARD, HCWDL ladder, and dense tests pass 48/48. The
complete repository suite passes 347/347 with the same 14
Matplotlib/Pyparsing warnings. The remaining runtime action is an exact pushed
commit followed by a new source-pinned dense recovery; the completed D90 report
and D80 rolling checkpoint must be preserved.

## HCWDL continuous Shell Exact and dense-closure repair (2026-08-10)

The live ten-point D90 and five-point D95 workers exposed a shared runtime
guard that still restricted every repair family to PMARD's seven-value alpha
grid. `HIGHCOV_SHELL_EXACT/v1` now accepts its declared continuous finite
coordinate on `[0,1]`; other repair families retain the legacy grid. Direct
regressions construct D90 and D95 confidence-warped views and preserve endpoint
and range checks.

Because the failed Slurm jobs are immutably pinned to the broken commit,
ordinary requeue is not valid. `HCWDL_DENSE_RECOVERY_SPEC/v1` binds each
complete original ledger and monitor report, authenticates the completed
D100offkd prefix, and submits only the failed/downstream closure from a clean
repair checkout. It writes into the original scientific campaign root under
the original graph while keeping separate recovery attestations and never
accessing final test. The exact contract is
`docs/contracts/HCWDL_DENSE_RECOVERY.md`.

Focused dense, matching, repair, and CLI validation passes 76/76. The complete
repository suite passes 344/344 with the same 14 Matplotlib/Pyparsing warnings;
Python compilation and `git diff --check` pass. The real ten-point and
five-point Tigris closures remain to be submitted from the exact pushed repair
commit.

## HCWDL interrupted sealed-final repair (2026-08-10)

The 300k unweighted pilot's sealed evaluation failed after its one-time claim
because `load_pmard_model()` restored checkpoint storage onto CUDA but left the
newly constructed ParT module on CPU. The loader now explicitly moves the
restored module to the requested device.

The repair does not reset or bypass the final-test seal. Exact-claim validation
and recovery were added alongside `HCWDL_FINAL_EVALUATION_MANIFEST/v2`, which
validates and reuses completed per-finalist reports and evaluates only missing
ones after an interruption. A separate source-pinned
`HCWDL_FINAL_RECOVERY_SPEC/v1` registers only sealed evaluation and aggregation,
binds the original failed job through an immutable monitor report, and forbids
new selection. The contract and operator procedure are in
`docs/contracts/HCWDL_FINAL_EVALUATION_RECOVERY.md` and
`docs/HCWDL_FINAL_RECOVERY_RUNBOOK.md`.

Focused final-recovery, HCWDL-contract, and CLI validation passes 65/65. The
complete repository suite passes 337/337 with the same 14
Matplotlib/Pyparsing warnings; Python compilation and `git diff --check` pass.
The real Tigris recovery remains to be run from the exact pushed repair commit.

## HCWDL five-point dense cold 300k supplement (2026-08-09)

The dense cold implementation now also registers an isolated five-point
screen: imported TOFF teaches D100offkd, then fresh D95/D90/.../D5/D0 and one
fresh M1. This is 22 new sequential GPU training nodes plus one CPU aggregate.
It retains the locked unweighted 0.25 CE plus 0.75 sole-teacher KD recipe,
temperature 2 for privileged rungs and temperature 1 for M1, 60 passes,
per-pass validation, macro-AUC-first checkpoint selection, and the shared
nested Shell Exact repair coordinate. It reuses the completed unweighted 300k
pilot read-only and has no final-test task.

The new scientific identities are the `HCWDL_DENSE5_* /v1` graph, node,
specification, command-plan, training-report, and aggregate contracts. Its
Slurm jobs use `hcddp5_*`, so it can run beside the primary campaign and the
ten-point supplement without artifact or scheduler-name collision. The same
thin CLIs dispatch by immutable spec contract; creation requires
`--rung-step 5` and separate exact authorization phrases.

This extension audit also fixed a latent generic-training bug before the
ten-point ladder reached D90: custom dense domain registries are now passed
through training configuration rather than looking up only the original
25-point HCWDL domains. D90 and D95 input selection are regression tested.
The exact contract and launch procedure are in
`docs/contracts/HCWDL_DENSE5_COLD_PILOT.md` and
`docs/HCWDL_DENSE5_COLD_RUNBOOK.md`. Focused HCWDL/dense/high-coverage/repair
validation passes 118/118; the complete repository suite passes 330/330 with
the same 14 Matplotlib/Pyparsing warnings. CLI help, Python compilation, and
`git diff --check` pass. No Tigris jobs were submitted by this implementation.

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
## HCWDL-RKD four-track dense descent (2026-08-10)

The active representation experiment has been versioned away from the retired
M1--M6 ascents.  It now registers four downward ladders: RSET cold/warm and
RREL cold/warm.  Each strategy owns a separately trained TOFF-to-D100 root;
each branch then descends D95, D90, ..., D5, D0 and terminates at M1.  The
immediate predecessor supplies both logits and representation targets.  Cold
rungs are fresh; warm rungs load only that immediate predecessor's selected
model state and reset optimizer/scheduler state.

The graph contains exactly 86 training nodes.  D100--D5 are authenticated,
nondeployable repaired-view training intermediates; D0 and M1 are HLT-only
deployable models, and the four M1 nodes are the primary endpoints.  The old
M5 controls and M2--M6 representation nodes are absent.  The registered
512/256/0 dense smoke contains 261 tasks, while the 300k/100k/0 pilot contains
275 tasks.  The additional rows are target-build/cleanup and four-M1
confirmation lifecycle work, not M2--M6 training rungs. Representation Slurm
job names begin `hcwdlr_`.

The semantic-version migration includes dense graph v1, recipe v4, target
family v3, training/checkpoint/model-extraction v2, screen aggregate v3,
confirmation v2, and campaign/command-plan v3. Old ascent artifacts cannot be
relabeled into these identities. The 089b472 base-HCWDL smoke remains
historical base-worker evidence; no new base ladder or corrected-parent prefix
is required by the dense route. A genuine dense smoke at the final pushed
representation commit remains the execution gate. No job was submitted and
no final role was accessed during implementation.

The complete representation test surface is **427 passed, 9 expected platform
skips, 0 failures**, executed in three definitive shards after the single
serial command exceeded its five-minute shell timeout at 66% with no failure.
Eight skips are directly executable POSIX-shell checks and one is the
Windows-unavailable real-`SIGUSR1` check. The earlier ten-action non-final
authority is bound to the retired direct-D0/M1 graph and cannot authorize
recipe v4.

The launch-preparation implementation adds four phrase-bound Tigris resource
probes plus one dependent accounting collector, a genuine dense resource
profile, a maximum-topology `RREL_D100` storage template, conservative
86-node/83-generation storage estimates for smoke and pilot, and a fresh
free-space check in both candidate review and immediately before scheduler
mutation. The probes authorize no training or final access. The dense smoke
acceptance reopens the 261-task ledger, monitor, filesystem output audit, and
four-M1 terminal aggregate before it can gate the pilot. No probe, dense
smoke, or pilot has been authorized or submitted.
The dense route also has a standalone tap/installed-Weaver parity CLI, so it
does not require the canceled full-parent evidence preparer merely to
bootstrap its historical TOFF training teacher.
The first genuine dense-teacher attempt exposed and fixed a validator mismatch:
JSON wrapper/engine parents use canonical content hashes, while the selected
checkpoint parent uses its raw file-byte hash. The failed attempt published no
teacher import and its immutable tap/parity files remain reusable.
The same operational pass also fixed dense target planning to use the frozen
tap schema's canonical `tap_schema_sha256()` identity; `tap.json` intentionally
has no self-referential top-level `content_hash`.
The next Tigris preflight exposed one more stale call to the assignment-cache
validator before any graph or recipe artifact was published.  The dense
recipe, target-planning, and registered recipe-task paths now share one
path-based train-assignment authority check.  It derives the expected train
row count independently from the authenticated v4 recipe or row selection,
binds the historical split and row-selection hashes plus the canonical matcher
resource digest, and reopens every referenced assignment shard.  The failed
preflight root therefore remains safely reusable for its already authenticated
tap, parity, and dense-teacher artifacts; graph/control/recipe/disposition were
absent at failure.
The post-fix focused dense-teacher, target, registered-artifact, and CLI suite
is **130 passed, 8 expected POSIX-shell skips** on Windows.
The first exact dense resource probes all completed, but their dependent
collector exposed two site-interface mismatches before publishing accounting
evidence: current Tigris removed `ReqGRES` in favor of `ReqTRES`, and CPU
collector hosts use the `gg-*` family in addition to GPU `gh-*` hosts.  The
parser now authenticates typed GPU requests from `ReqTRES`, recognizes both
Tigris host families, and distinguishes the dense resource collector from the
non-final acceptance collector.  Collector recovery is limited to requeuing
the same failed collector ID from one clean direct compatibility successor of
the measured commit; that successor may change only the exact parser,
collector validation, focused tests, and this handoff.  Job `80194` was then
found to be purged from Slurm, so requeue is impossible. A separate v1
collector-recovery authorization/ledger now binds the original four completed
probe IDs, failed log, original ledger, exact compatibility source, and one
replacement collector. It records zero measurement jobs and has no probe,
dense-training, pilot, array, or final-role submission path.
The final parser/recovery-focused suite is **26 passed**; the broader
scheduler/resource/contracts/CLI suite is **116 passed, 8 expected
POSIX-shell skips**.
The subsequent purged-collector recovery contracts, canonical routes, CLI
inventory, and exact one-/two-commit compatibility boundary pass **30 focused
tests** (29 contract/resource tests plus the explicit CLI inventory check).
Replacement collector `80400` then confirmed that current Tigris returns an
empty `sacct Comment` column even though the exact `--comment=<binding>` token
is preserved in `SubmitLine`. The parser now permits only that exact pair:
blank raw column plus the required bound submit token; altered or missing
tokens still fail. A requeued collector may reuse the first attempt's partial
`cpu_small/sacct.psv` only when a fresh capture is byte-identical, and every
other raw/evidence path must remain fresh.
All four measurement jobs (`80190`--`80193`) completed, and replacement
collector `80400` completed after exact requeue under the blank-comment parser
compatibility commit. The resulting genuine dense profile is
`a1069ee9efd9f98798495ee4e663346431175926502b4a59b09d4743df337b92`;
the 512/256/0 storage estimate is
`88ba2e77916192d1e7a3dd0816985684ae74e7b3680791e402068422e1bb4f07`,
requiring 25,718,067,627 bytes against 51,678,019,584 observed free bytes.
Because the probes measured commit `290f9f8` and the later commits changed
only the collector/accounting compatibility surface, the new
`HCWDL_REPRESENTATION_DENSE_COMPATIBLE_RESOURCE_PROFILE/v1` wrapper freezes
that exact Git diff and reuses the completed measurements without relabeling
or rerunning them. Any target, model, training, data, graph, or worker change
fails the wrapper. No dense training or final-role job has been submitted.
The compatibility projection also reopens the old `gpu_target` result through
that profile and refreshes only its producer source identity from the clean
campaign checkout. The production target preflight now consumes the typed
dense storage/profile contracts directly, including their authenticated
scheduler peak, instead of routing them through the legacy generic resource
schema. Thus jobs `80190`--`80193` are reused and are not rerun. The complete
post-change representation suite is **473 passed, 9 expected Windows platform
skips, 0 failures**; the exact changed-path registry matches the full
measured-commit-to-working-tree Git diff and `git diff --check` is clean.
The post-probe path is now one non-submitting command,
`prepare_hcwdl_representation_dense_smoke_candidate.py`. It projects only the
source/project fields in the four authenticated runtime signatures, publishes
all 83 target authorities and all 261 bound runtime rows, and emits the dry-run
and strict candidate hashes. Candidate review validates the storage template
against the commit that genuinely measured it, while the compatible profile
still binds the clean campaign source. No probe or scheduler call occurs in
this preparer.

## HCWDL non-final corrected-parent prefix (2026-08-09)

The base HCWDL execution layer now has a dedicated non-final 60-pass parent
scope. `HCWDL_CAMPAIGN_SPEC/v8` with execution scope exactly
`parent_prefix_through_finalist_lock` retains the selected non-smoke
train/validation populations, v4 recipe, all 23 ladder nodes, canonical
confirmation registry, 60 complete natural-role passes, and every-pass
validation. Its task registry ends at `finalist_lock`; it contains no
execution lock, final-test selection/assignment/evaluation, or aggregate task.
Stored final-role counts are sealed metadata, while execution-lock and
final-role authority are explicitly false.

The matching command plan and submission authorization are
`HCWDL_COMMAND_PLAN/v4` and `HCWDL_SUBMISSION_AUTHORIZATION/v8`. Candidate
authorization, initial submission, manual continuation, and exact recovery
use prefix-specific phrases. Dry-run, monitoring, ledgers, and recovery reuse
the existing exact-ID machinery but reconstruct only the v8 prefix graph.
Existing v3--v7 campaign and authorization artifacts retain their original
meaning; they are not upgraded or relabeled.

Final local verification in the `tagging-hlt` environment is **790 passed,
6 platform skips, 0 failures** across all 796 collected tests. The skips are
five directly executable POSIX-shell checks and one real-`SIGUSR1` check that
are unavailable on Windows; the 14 warnings are the existing
Matplotlib/PyParsing deprecations. The integrated parent-authority, loss,
locks, central-contract, runtime, non-final-acceptance, parity, and prefix
shard is **124 passed, 1 expected Windows skip**. The dedicated prefix suite
is **9 passed** and covers v8 graph closure, scope tampering, smoke and
open-ended-production rejection, v8 authorization/executable validation,
worker-route denial, dry-run/submit planning, monitor/recovery closure, and
create/authorize CLIs. No scheduler command was executed, no campaign was
authorized or submitted, and no final role was accessed. Operator boundaries
and exact future commands are documented in
[`HCWDL_PARENT_PREFIX_RUNBOOK.md`](HCWDL_PARENT_PREFIX_RUNBOOK.md).

The downstream HCWDL-RKD boundary now accepts only this exact v8 prefix scope.
Parent import v3/schema 2 and parent-loss attestation v3/schema 3 reopen all
six qualifier, 23 screen, and every registered confirmation engine report;
they require the exact reconstructed configuration, corrected loss semantics,
60 validation records at exact pass boundaries, completed update budget,
independently selected metrics, and authenticated selected/final checkpoint
bytes. A forged imported artifact cannot bypass the independent v8 campaign
check in non-final acceptance. No parent run or representation execution is
authorized by this implementation.

The completed d56 parent smoke (`hcwdl_parent_smoke_d56c622_r1`, 42 tasks) is
valid smoke evidence only. It is categorically ineligible for parent import or
non-final action inputs. The failed representation root
`hcwdl_rkd_nonfinal_acceptance_d56c622_r1` contains review registries and the
early tap only; it stopped during installed-Weaver parity construction and
must not be resumed, treated as acceptance evidence, or reused. A new clean
root backed by an exact completed non-smoke v8 60-pass prefix is required.

## HCWDL-RKD installed-Weaver surface-parity v2 repair (2026-08-09)

The first parent-evidence attempt against installed Weaver on Tigris reached
the real CPU forward/backward boundary and failed before publishing
`architecture/surface_parity.json`: the v1 fixture drew the energy component
of each standard-four vector independently, so most fixture vectors were
unphysical, and the v1 report also attempted to serialize Weaver's known
nonfinite auxiliary Lorentz-input derivatives as a numeric maximum. Canonical
JSON correctly rejected that NaN/Infinity. The environment, corrected parent,
and registries were not the cause. The failed representation root contains
only its three review registries and immutable `architecture/tap.json`; no
Slurm work or final-role access occurred.

The contract is now `HCWDL_REPRESENTATION_SURFACE_PARITY/v2` with schema
version 2. Its fixture constructs deterministic timelike unit-mass vectors.
Logits, feature-input gradients, and parameter gradients remain strictly
finite and parity-gated. Lorentz-vector derivatives are explicitly
non-training-required and are compared by exact finite/NaN/+Inf/-Inf topology
plus an absolute tolerance over jointly finite entries, with only JSON-safe
counts and finite-or-null maxima stored. Ordinary, TOFF charged, and TOFF
neutral paths are audited separately, closing the v1 native aggregate that
could hide a NaN behind an earlier finite feature-gradient maximum.
Because that parity artifact is embedded in the architecture attestation, the
outer identity is also versioned honestly as
`HCWDL_REPRESENTATION_ARCHITECTURE_ATTESTATION/v2`, schema version 2; the v1
identity is not silently reinterpreted to require a v2 nested artifact.

Focused local verification passed: 151 tests passed and 6 platform-specific
POSIX/SIGUSR1 tests skipped across the model, contract, acceptance-evidence,
runtime-row, CLI, campaign, provenance, and task-runtime suites. The
installed-Weaver v2 rerun remains pending and must use the exact pushed commit
in a fresh representation root. The
corrected parent smoke remains immutable at its existing source; this repair
neither authorizes nor submits the bounded actions, a pilot, or any final-role
work.

## HCWDL-RKD bounded non-final acceptance authority (2026-08-09)

The separately authorized implementation is complete locally. It adds a
closed ten-action authority outside the campaign DAG: bounded D0c/D0w target
preparation; exact two-update `RSET_M1c`, `RSET_M1w`, `RREL_M1c`, and
`RREL_M1w` probes; a real-`SIGUSR1` reference/interrupt/fresh-resume sequence;
and a validation-only D0c/D100/TOFF proxy. Every action is scalar. Training is
frozen to 512 train rows, 256 validation rows, seed 1337, effective batch 256,
and exactly two optimizer updates. Final rows are zero and campaign, pilot,
reservation, shared-final, final-role, and scheduler-submission powers are all
explicitly false.

Production execution reuses the reviewed target and training adapters through
an authority-private bridge, with live runtime validation before scientific
I/O. The validation proxy performs one label-bearing validation selection;
assignment and all three prediction streams are label-free, and durable
artifacts omit labels. Its metrics and paired-bootstrap sidecars are
validation-only and nonauthorizing. Workers emit immutable execution receipts
binding the exact action assembly, dependencies, source, recipe, worker bytes,
Slurm job, and semantic outputs. A separate Tigris collector must capture raw
terminal `sacct` records before a post-job action result or composite proof can
be genuine; caller-authored completion fields cannot satisfy the gate.

The new authority/proof family includes action inputs, action assemblies,
execution receipts, real-batch full-loss evidence, action results, two-update
evidence, USR1 delivery and exact-resume evidence, validation-proxy branch
access/result, and non-final scheduler evidence. Tigris evidence and
acceptance advance to v2. The exact phrase `AUTHORIZE BOUNDED NONFINAL
HCWDL-RKD ACCEPTANCE ACTIONS` is deliberately separate from the implementation
authorization and has not been issued or used. No job, pilot, campaign
submission, shared-final capability, or final role was created or accessed.

Local verification in the `tagging-hlt` environment is **767 passed, 6
platform skips, 0 failures**, with the same 14 Matplotlib/PyParsing warnings.
The skips are five directly executable POSIX-shell checks and one real
`SIGUSR1` check unavailable on Windows. The final focused USR1, execution-
receipt, scheduler, dependency, and composite-proof lineage gate is **66
passed**. All 405 Python files under `src`, `scripts`, and `tests` parse, and
the final whitespace/diff audit is recorded with this handoff.

This implementation does not remove the parent blocker. The existing
`b3154d67` parent predates explicit loss semantics and cannot be relabeled as
the required v8/v4 parent-loss v3 authority. The next research-compute step is
therefore still a fresh corrected HCWDL v8/v4 parent-prefix run from the exact pushed
source. Only after that parent exists may an operator separately issue the
non-final execution phrase and run the bounded actions. A pilot and any
final-test access remain unauthorized.

## Superseded pre-dense-descent v8-prefix/v4 compatibility closure (2026-08-09)

The matching-free representation-KD branch is now reconciled with the current
parent authority: executable `HCWDL_CAMPAIGN_SPEC/v8` with exact
`parent_prefix_through_finalist_lock` scope, primary
`HCWDL_RECIPE/v4`, and an exact fifteen-one loss vector.  The representation
overlay was `HCWDL_REPRESENTATION_RECIPE/v2` with schema version 1; the strict
parent import is `HCWDL_REPRESENTATION_PARENT_IMPORT/v3` with schema version 2;
and the corrected parent-loss attestation is
`HCWDL_REPRESENTATION_PARENT_LOSS_ATTESTATION/v3` with schema version 3.
Weighted v3 parents and old v1 representation imports cannot be relabeled as
these artifacts.

The pre-campaign builders now reopen the real parent v8 prefix, v4 recipe,
full assignment/qualification/confirmation/finalist authority chain, reports,
checkpoint bytes, installed-Weaver parity, model sources, and one clean parent
source checkout at the exact parent campaign commit.  The runtime binding
supports the two canonical prepublished files in place, including the single
closed read-only recipe consumer needed before the recipe gate; no duplicate
input copy or generic output-path exception is permitted.  The complete
86-task/152-row binding builds at the documented canonical paths.

Current local verification used the `tagging-hlt` environment with bytecode
and threaded BLAS disabled.  The complete test collection was run as two
disjoint selections: HCWDL tests passed **454**, with two Windows-inapplicable
POSIX-shell checks skipped, and the remaining **252** tests passed.  Therefore
all **708 collected tests** are accounted for: **706 passed, 2 skipped, 0
failed**.  All 392 Python files under `src`, `scripts`, and `tests` AST-parse;
`git diff --check`, the untracked-file whitespace scan, and the representation
`Fresh_check`/TODO/FIXME/`NotImplementedError` scan are clean.  The 14 warnings
are the existing Matplotlib/PyParsing deprecations.

This is not yet campaign completion.  The existing unweighted parent candidate
at commit `b3154d67` predates the explicit loss-semantics fields and does not
contain `hcwdl_parent_loss.py`; its reports and checkpoints therefore cannot
satisfy the v3 attestation.  The plan forbids relabeling those outputs, so a
fresh, separately identified HCWDL v8/v4 parent-prefix run from the new exact clean
commit is required before HCWDL-RKD parent evidence can be built. The bounded
bootstrap itself still cannot authorize two-update/USR1 or validation-proxy
actions; the separate non-final implementation described above now closes
that code path, but its phrase-bound execution and genuine Tigris evidence are
still pending. Neither limitation authorizes a pilot or final-role access.

No remote job, scheduler state, parent output, or final-role artifact was
mutated during this compatibility work.

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

## Superseded pre-v8-prefix/v4 HCWDL-RKD local evidence (2026-08-09)

The hashes and counts in this section describe the earlier `b820327` snapshot,
whose retained runtime binding used parent-import, parent-loss, and overlay
recipe v1 lineage. The parent campaign has since advanced to executable
`HCWDL_CAMPAIGN_SPEC/v8` parent prefix plus primary `HCWDL_RECIPE/v4`; therefore none of the
following smoke/binding hashes is current authorization or closure evidence.
They are retained only as historical local-test provenance and must not be
reused for a v3 import.

The implementation-grade plan remains
[`docs/plans/HCWDL_MATCHING_FREE_REPRESENTATION_KD_ASCENTS.md`](plans/HCWDL_MATCHING_FREE_REPRESENTATION_KD_ASCENTS.md).
That isolated local worktree contained the then-current versioned RKD
implementation for the four
registered M1--M6 ascents: paired jet plus unordered token-set KD (`RSET`) and
the same package plus differentiable latent-relation KD (`RREL`), each under
cold and warm initialization. The immutable graph contains exactly 24 primary
nodes plus the four jet-only/no-relation/within-class-shuffle controls and a
separate zero-coefficient acceptance barrier; the existing logit cold/warm
ascents remain the primary controls.

Reusable code now covers the corrected opt-in parent loss and attestation,
single-forward HLT/TOFF model surfaces and architecture taps, fixed
set/relation kernels, train-only gradient calibration, compact target
generations and cleanup/recovery, sequence-exact training resume,
macro-AUC-first checkpoint selection, reports, shared-final registration, and
fail-closed final disposition/reservation/claim handling. Campaign/runtime
surfaces include fixed artifact routes, non-submitting dry-run/local-smoke
paths, ordinary and deterministic Slurm workers, recovery, monitoring, and
exact campaign-bound cancellation. Thin pre-campaign CLIs audit immutable
parent-final evidence, freeze the fail-closed disposition, and publish proof
that every reserved legacy job is terminal; they never perform broad scheduler
cancellation.

The exact requirement-to-code map and pending acceptance boundaries are in
[`docs/plans/HCWDL_RKD_IMPLEMENTATION_TRACEABILITY.md`](plans/HCWDL_RKD_IMPLEMENTATION_TRACEABILITY.md),
and safe local/Tigris operating steps are in
[`docs/HCWDL_RKD_RUNBOOK.md`](HCWDL_RKD_RUNBOOK.md). The superseded evidence
was:

- the repository suite passes **656 tests**, with two Windows-only POSIX-shell
  checks skipped and 14 pre-existing Matplotlib/PyParsing deprecation warnings;
- the bounded local smoke covers all **86 tasks / 152 array-expanded rows**,
  all **24 primary nodes plus four controls**, reports finite losses and active
  gradients, performs no optimizer or scheduler step, and records
  `final_role_accessed=false` and `authorizes_tigris_or_pilot=false`;
- the smoke report content hash is
  `0ed46019c7c62d9e997ea843051cfb85e1d6cb824b80e0e072445698dac69f67`
  and its scientific probe hash is
  `966491893d4b7456dd39d72a29e5bcdf03cde71fdd60a424826a5ce3990ecbba`;
- the exact adapter-bound, non-submitting command plan covers the same
  **86 tasks / 152 rows**. Its command-plan hash is
  `86728de85232c914de92eac751146908025077628c3cee7a3406a67482ed05ea`,
  runtime-binding hash is
  `1696ca5e2ea38c5929061b1757011c427a3218486ac27ad762cf71fcae9df7ec`,
  and dry-run audit hash is
  `38d5f1851a887c6d9354f2580931d4f7440e0cc18c74e9c1b865804638fe608c`;
- the dry-run audit records every registered adapter schema as validated and
  `scheduler_mutated=false`; the retained local artifacts are under the ignored
  `checkpoints/hcwdl_rkd_local_verification/closure_20260809/` directory;
- all 389 Python files in `src`, `scripts`, and `tests` parse successfully;
  `git diff --check` and the separate untracked-file trailing-whitespace scan
  are clean; and runtime scans found no `Fresh_check`, `TODO`, `FIXME`,
  `NotImplementedError`, or executable `pass` path.

The local smoke uses bounded authenticated fixtures and a nonauthorizing
synthetic sealed-final pipeline; it is local behavior and failure-topology
evidence, not a substitute for installed-Weaver, ROOT, GPU, Slurm, or Tigris
evidence. The exact production adapters and every registered input/output row
are separately instantiated and validated by the complete dry run.

No remote mutation, Tigris submission, pilot launch, cancellation, or final
population access was performed by this implementation step. Authoritative
installed-Weaver parity and a genuine Tigris worker/cache/USR1 miniature with
measured resources, exact pushed clean source, and explicit pilot
authorization remain required. Local synthetic doubles and adapters are not
evidence for those production gates.

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

## HCWDL-RKD contract/layout normalization (2026-08-09)

The matching-free representation-KD implementation now has one central
97-contract registry that exactly matches active-plan Section 21 and 97 unique
canonical route templates. Gradient-calibration selection, runtime binding,
submission events, legacy final exposure, and the four action-specific Tigris
proof families no longer borrow an unrelated artifact identity. Section 31 now
uses the implemented kernel, shuffle, bootstrap, checkpoint, and assignment
member names, and documents execution-scoped calibration routes. The runbook
uses a portable Python selector and includes the measured fixed-size inventory
and storage-estimate producers. Local focused evidence is 5 contract/layout,
5 calibration, 13 campaign, 4 campaign-artifact, 8 runtime-provenance,
18 shared-final, 8 acceptance-evidence, 2 action-proof, and 57 CLI tests passed;
two POSIX-only CLI checks
were skipped on Windows. This is local structural evidence only: installed-
Weaver and genuine Tigris acceptance, measured resources, clean pushed source,
and submission authorization remain pending.

## Exact active next task: corrected parent, then bounded Tigris acceptance

The v8-prefix/v4 compatibility migration and bounded non-final authority
implementation are locally complete. Publish the exact clean branch commit and
refresh the isolated Tigris worktree. First create a new HCWDL v8/v4 parent prefix
campaign identity at that commit and run its corrected reports/checkpoints;
the `b3154d67` outputs cannot be reused as v3 parent-loss evidence. Only after
that parent validates may an operator separately issue the exact non-final
execution phrase and run the bounded two-update, USR1 exact-resume, and
validation-proxy actions. The installed-Weaver boundary and genuine worker
evidence must then be measured and audited. Do not reuse superseded hashes,
submit an RKD pilot, or open the final role before every gate passes.

## Dense-smoke runtime source identity correction (2026-08-11)

The first authorized 261-task dense smoke at source `526a620` submitted its
complete ledger but stopped at the root `tap_schema` job before registered
input access. Dense candidate preparation had placed the outer serialized
source-snapshot artifact hash in `runtime_facts.source_snapshot_sha256`; the
worker correctly measures and compares the snapshot's inner scientific
checkout identity, so the two hashes could never agree. Preparation now
validates the source-snapshot artifact and separately binds its exact
`source_snapshot_sha256` field. The regression deliberately makes the outer
artifact hash differ from that inner identity. Focused dense-preparation,
worker-runtime, and campaign tests pass 28/28. This changes the executable
candidate identity but not the graph, recipe, targets, measured resources, or
storage estimate. The impossible `526a620` smoke must be cancelled by the
exact IDs in its immutable submission ledger; its scientific tasks cannot be
resumed or relabeled. Rebuild one fresh candidate from the existing reusable
inputs and compatible measured profile at the correction commit.

## HCWDL U/D representation-KD twenty-point v2 revision (2026-08-12)

The user authorized replacing the unexecuted ten-point representation-KD
candidate with a smaller twenty-point screen. The active implementation plan,
reusable contract summary, runbook, implementation, recovery, reporting, and
tests now agree on this exact graph for each of `F_RSET` and `F_RREL`:

```text
TOFF -> U020 -> U040 -> U060 -> U080 -> U100
     -> D80 -> D60 -> D40 -> D20 -> D0 -> M1
```

This is a new `HCWDL_HOMOTOPY_REPRESENTATION_*/v2` family. The historical v1
candidate below was never executed and is not accepted or relabeled as v2.
The retained rungs preserve their imported U/J seed identities
`transition_02,04,06,08,10,12,14,16,18,20,21`; the scientific spacing change
therefore does not silently renumber the paired stochastic trajectories.
Every track remains cold-started and immediate-predecessor-taught. U/D use
unweighted `0.25 CE + 0.75 KD` at temperature 2, M1 uses temperature 1, and
the v5 RSET/RREL objective remains unchanged.

The immutable v2 graph contains 22 fits and 21 logical train-only target
banks: the two U020 students share the TOFF bank and every nonterminal student
produces exactly one bank for its child. Its command plan contains 47 tasks:
two setup tasks, 21 target generations, 22 fits, aggregate, and completion.
The pilot remains 300,000/100,000/0 rows with 60 validations over 60 passes;
the genuine smoke remains 4,096/4,096/0 with two optimizer updates per fit.
No final-test task or capability was added.

Implementation review also fixed a fail-closed validation defect found during
the revision: `validate_graph(custom_registry)` previously checked teacher
routing and target banks through the global registry. It now validates the
supplied registry throughout, and a tampered predecessor edge has an explicit
regression test.

Local v2 evidence at the final working state:

- focused homotopy graph/recovery suite: 53 passed;
- complete repository suite: 940 passed, 9 expected Windows platform skips,
  and 14 dependency deprecation warnings;
- bounded all-node synthetic execution:
  `HCWDL_HOMOTOPY_REPRESENTATION_LOCAL_SMOKE/v2`, 22/22 fits, 21 target banks,
  graph SHA-256
  `b24a5ae2460ff872b0adf217ba9c1f4b7da606b9d3bbe6c28a7f6f065113e272`,
  content SHA-256
  `b47aa9ea75e3c233e0ed68209f8368b2e21a16a8c4ffddb0b2b33ae9d7cdad97`;
- all 13 HCWDL-U-RKD CLI `--help` surfaces passed;
- all 27 documented v2 contract identities exactly match the module;
- repository-relative Markdown links and Git-Bash worker syntax passed;
- `git diff --check` passed.

No Slurm job was submitted and no final-test data were accessed. The remaining
real closure is to commit and push this exact source, create a new clean
detached Tigris worktree, regenerate integration/numerical evidence at that
commit, and run the separately authorized genuine v2 smoke. A measured v2
smoke profile and a new nonmutating 300k dry run are required before pilot
submission.

## Historical HCWDL U/D representation-KD v1 integration (2026-08-11)

The implementation-authoritative
`docs/plans/HCWDL_HOMOTOPY_REPRESENTATION_KD_IMPLEMENTATION_PLAN.md` is now
locally implemented in the isolated `hcwdl-u-rkd-local` integration worktree.
It combines the repository-local homotopy source line at
`c2a5510d81a1b1edd8f519d271c9dc398a92364c` with the corrected RKD source
line at `63139ca`, without modifying or deleting the pre-existing dirty RKD
worktree. No external donor artifact was copied.

The new supplemental v1 family implements the exact 42-fit graph: 21 cold
`F_RSET` nodes and 21 cold `F_RREL` nodes over
`U010..U100 -> D90..D0 -> M1`. It uses 41 train-only target banks, including
one shared TOFF bank, and retains the fixed v4 base loss plus v5 RSET/RREL
mathematics. U/D use temperature 2, M1 uses temperature 1, all fits use
unweighted `0.25 CE + 0.75 KD`, and the representation coefficient is 0.10.
Pilot execution is fixed at 300,000/100,000/0 rows, 60 passes, validation
every pass, and macro-OVR-AUC-first checkpoint selection. D0/M1 extraction is
HLT-only and the campaign has no final-test task or capability.

Reusable modules now cover exact graph/recipe contracts, teacher-own-domain
target generation, one-time RAM views and target loading, cold node training,
RSET/RREL calibration and exact resume, deployable extraction, reporting,
campaign/workflow dispatch, exact-ID monitoring/cancellation, source/resource
failed-closure recovery, resource measurement, and authenticated target
cleanup. The immutable command plan contains exactly 87 tasks: two setup
tasks, 41 target generations, 42 fits, aggregate, and completion. GPU target
and training jobs request 8 CPUs, 96 GiB, six hours, one GH200, and
`B:USR1@120`. Submission is separately phrase-gated and can resume an exact
validated generic HCWDL submission-event prefix. Source recovery binds and
reopens the corrected checkout's complete scientific source-hash table;
resource recovery can replace CPU, RAM, walltime, and GPU requests without
changing scientific identity.

Local evidence at the final working state:

- combined focused homotopy/representation suites: 170 passed;
- complete repository suite: 940 passed, 9 skipped for unavailable POSIX
  bash/SIGUSR1 behavior on Windows, and 14 dependency deprecation warnings;
- bounded all-node synthetic smoke:
  `HCWDL_HOMOTOPY_REPRESENTATION_LOCAL_SMOKE/v1`, 42/42 fits, 41 target banks,
  graph SHA-256
  `24168ff987eb7f41fc024c4e06384e26b8360f1e100ed8c48aeeede8d6280633`,
  content SHA-256
  `e59501fb5c2654d9d3a296e5d3d57265949686ed9caa149731366dd805240f3c`;
- all 13 new CLI `--help` surfaces passed;
- all 27 plan contract identities exactly match the contract module;
- repository-relative Markdown link test passed;
- the new Slurm worker passed Git-Bash `bash -n`;
- `git diff --check` passed.

This is local implementation evidence, not live-run authorization. Remaining
real closure is: create a clean pushed commit, run installed-Weaver parity and
full numerical acceptance at that exact source, run the separately authorized
genuine Tigris two-update smoke (including real USR1 fresh-process resume),
publish measured memory/CUDA/wall/I/O/storage evidence, and render the complete
nonmutating 300k dry run. A live smoke or pilot still requires its own exact
phrase-bound user authorization. No Slurm job was submitted and no final-test
data were accessed during this implementation.
## HCWDL-U-RKD Tigris prerequisite architecture boundary (2026-08-12)

The exact `c055ea84` research-compute checkout passed the complete repository
suite: 952 tests passed with one pre-existing PyTorch warning. The subsequent
non-training prerequisite bootstrap failed before campaign creation because
the architecture-only checkpoint audit called the corrected parent-loss
validator on `b3154d67` wrappers. Those wrappers are content-authenticated but
predate the explicit loss-semantics fields; the living handoff already records
that they cannot be relabeled as corrected parent-loss evidence.

The architecture audit now validates their original wrapper contract/content
hash, engine and selected-checkpoint byte lineage, and strict canonical model
state load without inferring loss authority. The existing corrected
`validate_hcwdl_training_report` remains unchanged and still rejects the same
legacy wrapper. The bootstrap's separately hashed recipe-compatibility
artifact continues to require complete execution-policy equality and fifteen
exact-one class weights. A regression proves this asymmetric boundary. The
failed partially published prerequisite root is immutable and must not be
overwritten; the next genuine Tigris attempt uses a new pushed commit and new
prerequisite/campaign roots.

The next attempt at `31b78378` passed the 19-test focused closure and fully
published the prerequisite bundle, including installed-Weaver architecture,
numerical, compatibility, recipe, and committed-kernel evidence. Campaign
creation then exposed a planning/runtime adapter mismatch: the portable JSON
stored `committed_directory` as an absolute string, while the shared
production kernel loader requires its internal `RegisteredInputPath` marker.
The closed U-RKD kernel adapter now promotes only the exact one-field absolute
committed-directory form before delegating to the unchanged production
commit/parent/member validators. Arbitrary files and generic raw production
paths remain rejected. A focused regression freezes this boundary. The
complete `31b78378` prerequisite root remains valid evidence but is not reused
by the corrected source identity; the next attempt uses new roots.

The `2b7947b9` attempt then passed the nine-test kernel-focused closure and
published the complete prerequisite bundle. Campaign creation exposed one
remaining artifact-envelope mismatch: the combined U-RKD recipe builder read
`scientific_values` from the top level of an authenticated
`HCWDL_REPRESENTATION_RECIPE/v5`, although the reusable v5 contract stores all
scientific fields under `payload`. The builder now reads the validated v5
payload, and a focused regression joins a genuine v5 example artifact through
the combined-recipe builder so this producer/consumer boundary is exercised
before Tigris campaign creation. Local `py_compile` and `git diff --check`
passed; the local Windows Python installation lacks NumPy, so the focused
pytest closure must run in `atlas_kd_tigris` before the corrected attempt is
submitted. The immutable `2b7947b9` prerequisite root is retained as failed
attempt evidence and is not reused under the corrected source identity.

The `38ac499b` smoke successfully created and submitted its exact 47-task
campaign, then failed in the shared TOFF target job before publishing a target
bank. The strict representation worker backend correctly rejected the process
because its Slurm wrapper had not exported the required
`CUBLAS_WORKSPACE_CONFIG=:4096:8`. Existing deterministic representation
workers already set this value; the combined homotopy-representation wrapper
now does the same before Python can import Torch. A regression verifies the
export exists and precedes the worker `exec`. Since the original specification
is source-pinned, continuation must use the authenticated failed-closure source
recovery rather than requeueing the failed job under `38ac499b`.

The first corrected source-recovery TOFF target job at `d9ac4fe9` then reached
real target-array validation and exposed a distinct count-bound bug. The
validator had used `128`, the contextual latent width, as a universal particle
limit. That is not a Scouting collection bound. Validation now imports the
canonical schema limits and enforces 200 tokens for unified/companion HLT, 90
for native-offline charged, and 60 for native-offline neutral. The target
format, target values, family classifier, model forwards, graph, recipes, and
losses are unchanged. Boundary regressions accept exact 200/90/60 populations
and reject 201/91/61. The failed recovery published no shared TOFF target bank;
continuation again requires repeated authenticated source recovery from its
replacement ledger rather than an ordinary requeue.

Under `2f6658c6`, the shared TOFF target bank completed and both first U020
students completed optimization and checkpoint selection. Both failed only at
terminal deployable extraction: the freshly strict-loaded public model was
CPU-resident while parity inputs captured from validation were on CUDA. The
extractor now requires all parity inputs on one device and moves only the
temporary restored public model to that device before comparing logits. This
does not change selected checkpoints, deployable state bytes, training,
metrics, or inference. A regression records that the restored model is moved
to the parity-input device. Source recovery is expected to reuse the exact
rolling/selected training state and repeat terminal extraction/publication.

## HCWDL-U-RKD bounded performance diagnostic (2026-08-15)

Live Tigris evidence from pilot jobs `87174` (`F_RSET_U020`) and `87195`
(`F_RREL_U020`) showed that the slowdown is host-side rather than GPU-memory
pressure: each process consumed approximately one CPU core despite many idle
threads, GPU utilization was intermittent and low, and the completed pass
times were roughly 35--39 minutes. Their one-time train/validation cache builds
also consumed about 21 and 7 minutes respectively. Live Python stack sampling
could not be performed because neither `gdb` nor `py-spy` is installed on the
compute nodes.

A diagnostic-only bounded profiler now exercises the exact optimized U020
stream, authenticated compact TOFF target bank, student forward, scheduled
RSET/RREL loss, backward pass, production gradient-finiteness scan, and
optimizer update over a configurable number of real batches. It separates
input/target transfer, forward, base logit KD, jet/set/relation representation
terms, backward, gradient checks, optimizer work, and residual step time using
explicit CUDA synchronization only in the profiling path. The ordinary
training path supplies no callback and is unchanged. Profile artifacts use
`HCWDL_U_RKD_PERFORMANCE_PROFILE/v1`, must be published outside the immutable
campaign root, and cannot publish checkpoints, metrics, or campaign lineage.

Local evidence: 38 focused tests passed across the profiler, representation
training, and homotopy-representation campaign modules; Python compilation and
CLI help passed; `git diff --check` passed (line-ending warnings only). The next
step is two independent 50-batch Tigris diagnostic jobs for `F_RSET_U020` and
`F_RREL_U020`. Their measured phase tables will determine whether the primary
repair targets the representation kernel, repeated synchronization in gradient
validation, host/device joining, or another step component. No live job was
submitted by this change and no final-test data were accessed.

The exact pushed profiler at `3406f2d8` subsequently completed on two GH200s.
Job `87648` measured 50 `F_RSET_U020` batches at pass 7: mean step time was
1.723164 s, of which set representation consumed 60.32%, backward 34.46%, and
model forward 3.29%. Job `87649` measured 50 `F_RREL_U020` batches at pass 5:
mean step time was 4.862944 s, of which relation representation consumed
44.64%, set representation 19.68%, backward 33.73%, and model forward 1.15%.
Transfers, target joins, optimizer updates, and gradient-finiteness scans were
each below 1.3%. This localizes the optimizer slowdown to the fragmented
finite-spectral student graph; additional CPUs cannot repair it.

The execution-only repair keeps each immutable spectral omega/phase block on
its consuming device for the worker lifetime, batches padded ordinary/native
set means into a small number of tensor operations, and constructs the exact
top-32 relation topology with the historical FP64 p4, NumPy lexicographic tie,
stratum, and ESS rules before evaluating all live latent cosines and spectral
means in one batched differentiable graph. Historical rowwise implementations
remain independent acceptance oracles. Teacher target generation explicitly
retains the historical rowwise evaluation order, so recovery cannot change
previously frozen target bytes. The scientific kernel resources, targets,
eligibility, reductions, schedules, graph, and contracts are unchanged.

Local pre-Tigris evidence for the repair: 67 combined representation,
homotopy-representation, profiler, and production tests passed, followed by
the complete repository suite at 975 passed, 9 platform skips, and 14
dependency warnings in 462.64 seconds. New parity
fixtures cover ordinary and native set forward/gradient equality, relation
eligibility/count/FP64-ESS equality, relation forward/backbone-gradient
equality, topology reuse and stale-input rejection, padding/ineligible
populations, and device-resource reuse. The scalar-vs-batched maximum
differences in the explicit fixtures are within the existing
numerical-acceptance tolerances. A repeated 50-batch GH200 profile at the exact
corrected commit remains required before recovery of the live pilot. No live
job was submitted by this repair and no final-test data were accessed.

The exact corrected commit `2a1de332` then passed the bounded GH200 acceptance
profile. Job `87699` measured `F_RSET_U020` at 0.123782 s/batch versus the
1.723164 s/batch baseline (13.9x overall); its set component fell from
1.039440 s to 0.009079 s (114.5x). Job `87700` measured `F_RREL_U020` at
0.202051 s/batch versus 4.862944 s/batch (24.1x overall); relation fell from
2.170717 s to 0.079659 s (27.2x) and set from 0.957087 s to 0.007650 s
(125.1x). Both 50-batch jobs completed in under four minutes including
one-time preparation. This validates the execution repair on the required
production GPU and supports six-hour recovery requests for the unfinished
pilot closure.

The pilot submit had separately journaled 45 of 47 reviewed Slurm commands
before its final ledger was published. Its remaining aggregate command could
not later be submitted because imported parent-completion job `83488` was
cancelled and had left the controller; `sbatch --test-only` failed with
`Job dependency problem`. The v2 operational recovery now authenticates an
exact contiguous submission-event prefix as `complete_submission=false`,
represents the absent suffix as `NOT_SUBMITTED`, and includes it in the
ordinary failed/downstream source-recovery closure. It never fabricates job
IDs, weakens the original command, or contacts Slurm while assembling the
partial ledger. No final-test data were accessed.

Local evidence for the partial-ledger repair: the focused homotopy-RKD,
recovery, and CLI suite passed 103 tests with 8 expected POSIX-shell skips;
the known slow runtime/provenance boundary passed independently at 35/35.
Two monolithic Windows runs accumulated unrelated process-global test state
and reached their 15/25-minute ceilings without an assertion failure, so the
complete 985-test collection was rerun with fresh interpreter boundaries.
All 96 modules passed: 976 tests passed and 9 expected Windows platform tests
skipped. The assembler CLI help, Python compilation, contract identities, and
`git diff --check` also passed.
