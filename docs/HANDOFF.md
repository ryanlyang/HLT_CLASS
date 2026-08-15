# Current Handoff

## HCWDL full-data mapped-identity preprocessing repair (2026-08-15)

The all-mapped FULL3 foundation at source commit
`8403b1a1c3c0e3bdaef9d63a858163a1922d85fb` completed its complete
train/validation assignment prefix and assignment lock (job `87461`), then
scale-calibration job `87462` failed on raw ROOT entry
`H0HpHm_mixed_new/dnnTuples_nanov15_0000.root::46`. That entry is absent
from the authenticated assignment shard because it is not in the mapped
population. The compact `all_rows: true` row selection had been interpreted
by coupling preprocessing as every raw ROOT entry instead of every
authenticated mapped entry. Bounded 300k selections store explicit entry
lists and therefore did not expose this full-data-only execution bug.

`hcwdl_upper_builder._selected_source_chunks` now uses each validated dense
assignment shard's sorted entry array as the selected-population carrier,
checks its exact per-source count against the split/selection inventory, and
independently proves that every entry is authorized by `RowSelection` and
present in the streamed ROOT source. Scale calibration, coupling construction,
full audit, and sampled audit all use this one path. Other training/data
streams were reviewed and already apply `baseline_mask & labels >= 0` before
the row-selection mask, so no second all-mapped leak was found. The fix is
label-free and does not change assignment construction, row identities,
coupling science, graph, losses, seeds, endpoints, or final-test access.

A classified execution-only recovery was added because ordinary recovery
correctly rejects changed semantic source. Its exact identities are
`HCWDL_UNIFIED_BALANCED_FULL_MAPPED_IDENTITY_REPAIR_EVIDENCE/v1` and
`HCWDL_UNIFIED_BALANCED_FULL_MAPPED_IDENTITY_RECOVERY_SPEC/v1`, with repair
classification `all_mapped_assignment_identity_filter_v1`. It is authorized
only for a foundation closure beginning at `scale_calibration`, requires the
completed all-mapped assignment lock, preserves that prefix, and fails closed
unless the only changed semantic files are `hcwdl_upper_builder.py`, the
FULL3 recovery module, and the FULL3 contract constants. The recovered
foundation starts at scale calibration. After its foundation lock completes,
the three unchanged scientific arms must be launched from the original
source-bound worktree; they do not execute the repaired preprocessing path.

Files changed for this repair:

- `src/hlt_classification/scouting/hcwdl_upper_builder.py`;
- `src/hlt_classification/scouting/hcwdl_unified_balanced_full_recovery.py`;
- `src/hlt_classification/scouting/hcwdl_unified_balanced_full_contracts.py`;
- `scripts/create_hcwdl_unified_balanced_full_recovery.py`;
- `tests/test_hcwdl_homotopy.py` and
  `tests/test_hcwdl_unified_balanced_full.py`;
- the FULL3 reusable contract, runbook, and this handoff.

Local evidence after the final review:

- direct all-mapped regression proves raw entries absent from assignments are
  excluded; a bounded-selection regression rejects assignment drift;
- recovery closure/evidence regression starts at scale calibration, reuses the
  exact assignment lock, and rejects any fourth semantic-file change;
- post-review homotopy/unified-balanced/full-data focused suite: 91 passed in
  76.53 seconds;
- complete repository suite: 470 passed in 290.96 seconds, with the 14
  existing Matplotlib/Pyparsing deprecation warnings;
- recovery create/submit CLI help, focused Python compilation, the two new
  contract identities, and `git diff --check` passed.

No external donor file or donor commit is involved. No local SSH, Slurm
submission/cancellation, ROOT write, final-test access, or Tigris execution of
the repaired code occurred. Exact next task: commit and push the repair, make
a clean detached Tigris recovery worktree, authenticate the failed foundation
ledger/monitor, submit the classified closure beginning at scale calibration,
and bind a replacement arm autolaunch to the recovered foundation-lock job.

## HCWDL unified-balanced all-mapped three-arm scale-up (2026-08-15)

The implementation-authoritative
[full-data three-arm plan](plans/HCWDL_UNIFIED_BALANCED_FULL_DATA_THREE_ARM_PLAN.md)
is implemented locally as the additive `HCWDL-UB-FULL3/v1` campaign family.
Its reusable semantics are frozen in the
[contract](contracts/HCWDL_UNIFIED_BALANCED_FULL_DATA.md), and the exact
foundation-to-autolaunch procedure is documented in the
[runbook](HCWDL_UNIFIED_BALANCED_FULL_DATA_RUNBOOK.md). Existing 300k U/J and
six-arm artifacts are immutable inputs or contextual results; this work does
not edit or relabel them.

The campaign derives exact train, validation, and sealed-final-test counts
from the authenticated split and selects every mapped train/validation row.
It rebuilds full-population assignments, residual couplings, corrected
balanced switch sidecars, endpoint/resource evidence, fresh unified `U000`
and `M0paired` roots, and one compact identity-ordered FP32 U000 target bank.
Durable reconstructed particle views remain forbidden, and no ordinary task
selects, assigns, couples, caches, or evaluates a final-test row.

The scientific registry contains exactly 38 fresh fits:

- shared CE roots `U000` and `M0paired`;
- factorized-only arms `C25P75`, `C10P90`, and `C10P75G15`;
- in each arm, `U020 -> U040 -> U060 -> U080 -> U100 -> D80F -> D60F ->
  D40F -> D20F -> D0F -> M1F`, plus `D100direct`.

Every fit uses 20 natural-population passes, validation every pass,
macro-AUC-first checkpoint selection, and unweighted per-jet CE. The three
declared homotopy losses are respectively `0.25/0.75/0`, `0.10/0.90/0`, and
`0.10/0.75/0.15` for CE/parent/grandparent KD; unavailable first-edge
grandparent weight transfers to the parent. Homotopy KD temperature is two.
`M1F` remains fixed at `0.25 CE + 0.75 parent KD` with temperature one.
Twenty full-data passes are intentional: at roughly eight to nine times the
300k train population, they process about three times as many jet examples as
a 60-pass 300k fit while avoiding a prohibitive 60-pass full-data bill.

The initial full-data envelopes are 8 CPUs, 256 GiB, 24 hours, and one GH200
for GPU training/target jobs; 16 CPUs, 192 GiB, 24 hours for assignment and
coupling arrays; and 4 CPUs, 64 GiB, 4 hours for reducers and reports. A
measured all-row endpoint/resource gate must stay below 75% of the locked GPU
RAM request before either shared root trains. Resource recovery may only
increase CPU, RAM, or walltime while preserving the GPU class and scientific
identity.

Operationally, one source-pinned campaign submitter registers the foundation
and one `afterok` autolaunch job. Only after the immutable foundation lock
exists does the autolaunch create and submit the three independent arm
ledgers. Foundation, arm, autolaunch, and recovery submissions resume from
immutable journals/artifacts after interruption; no accepted job is silently
resubmitted. Exact-ID cancellation, monitor/task-attestation validation, and
failed/downstream source-pinned or resource-only recovery are present.

Implementation and review evidence:

- focused HCWDL-UB/full-data/ladder suite: 46 passed;
- complete repository suite: 468 passed in 280.19 seconds, with the 14
  existing Matplotlib/Pyparsing deprecation warnings;
- the bounded synthetic all-mapped closure created a 2.6M/1M/1M inventory,
  exact 38-fit graph, foundation plan, three arm specs, three independent
  14-task dry-run ledgers, idempotent arm reuse, and an exact resource
  recovery closure;
- all 12 full-data CLI help surfaces and Python compilation passed;
- all 22 full-data/recovery contract constants are explicitly versioned;
- all three Slurm workers pass Git-Bash syntax checks and use absolute
  `${PROJECT_DIR}` activation plus `exec python -s`;
- the five-document Markdown-link audit and `git diff --check` passed.

There are no external donor files or donor commits for this block, so
`docs/LEGACY_SOURCE_MAP.md` is unchanged. No SSH, push, Slurm submission,
cancellation, final-test access, or new smoke occurred locally. The existing
production-worker evidence is reused exactly as allowed by the plan; the new
all-row endpoint/resource gate is the production preflight and cannot be
bypassed.

Exact next task: commit and push this implementation, create a clean detached
Tigris worktree at that full commit, bind the authenticated prepared 300k U/J
template, create the all-mapped foundation spec, run the nonmutating campaign
dry run, and then use the single explicit full-campaign submission command in
the runbook. That live command submits the foundation and automatically
releases all three arms after its lock; it does not submit a smoke or touch
final test.

## HCWDL unified-root balanced six-arm campaign (2026-08-15)

The implementation-authoritative
[HCWDL unified-root balanced homotopy plan](plans/HCWDL_UNIFIED_BALANCED_HOMOTOPY_IMPLEMENTATION_PLAN.md)
is now implemented locally as a new additive campaign family. It does not
mutate or relabel HCWDL-UJ. The reusable v1 semantics are frozen in the
[contract](contracts/HCWDL_UNIFIED_BALANCED_HOMOTOPY.md), and the exact
source-pinned foundation/six-arm procedure is in the
[runbook](HCWDL_UNIFIED_BALANCED_HOMOTOPY_RUNBOOK.md).

The shared 300k foundation owns exactly `U000` and `M0paired`, balanced
train/validation structural-switch sidecars, endpoint/resource gates, and one
identity-ordered FP32 U000 target cache. `U000` is the freshly initialized
unified 21-channel CE root on the exact P0 particle multiset; native TOFF is a
contextual control, not the primary root. Uniform Shell Exact moves every
matched continuous field with one exact rational offline strength and switches
discrete/validity groups with deterministic confidence-independent nested hash
variates. The original cost-CDF U schedule and confidence-warped D coordinate
remain paired controls only in `C25P75`.

Six separately authenticated 300k-train/100k-validation arms are registered:
`C25P75`, `C10P90`, `C05P95`, `C10P75G15`, `C05P80G15`, and `C00P100`.
Each has its own root, spec, command plan, ledger, monitor, cancellation
surface, recovery closure, aggregate, and completion artifact. There are no
cross-arm teachers or scheduler dependencies. The exact registry is 151 fits:
two shared fits, 34 reference-arm fits, and 23 fits in each of the other five
arms. Training GPU jobs are locked to 8 CPUs, 96 GiB RAM, six hours, and one
GH200. Every arm uses 60 natural-population passes, validation every pass,
macro-AUC-first checkpoint selection, and its declared CE/parent/grandparent
loss. M1 remains fixed at 0.25 CE plus 0.75 parent KD at temperature one.

The final audit fixed several launch-relevant issues rather than accepting
scaffolding:

- all new Slurm workers now use the absolute `${PROJECT_DIR}/sbatch/common.sh`
  activation contract and `exec python -s`;
- the foundation lock binds both shared report hashes, both selected
  checkpoint hashes, the compact U000 target manifest, and a semantically
  validated target lock;
- each arm validator reauthenticates its exact foundation, recipe arm, graph,
  waiver, project, source, and frozen semantic-source map;
- the operational waiver freezes the full model/input/training/cache/runner
  and worker surface, not only the headline U/D modules;
- partial foundation, arm, and recovery submissions resume from immutable
  journals instead of resubmitting already accepted jobs;
- finalist, execution, sealed-evaluation, and campaign-completion artifacts
  now receive semantic validation in addition to content-hash validation;
- recovery is restricted to the exact failed/downstream closure. Source and
  resource recovery remain separate, and resource recovery may only increase
  CPU, RAM, or walltime while retaining the GPU class.

Local readiness evidence at this checkpoint:

- focused HCWDL-UB/homotopy/training/worker suite: 110 passed;
- complete repository suite: 462 passed in 586.68 seconds, with the 14 existing
  Matplotlib/Pyparsing deprecation warnings plus one local pytest-cache
  permission warning;
- bounded synthetic end to end: foundation plus all six arm specs/plans, the
  real six-arm dry-run wrapper, six validated dry-run ledgers, and an exact
  resource-recovery closure/command plan all passed;
- all 17 new CLI help surfaces passed;
- all 30 new contract constants are versioned identities and appear in the
  implementation plan or reusable contract; the launch-boundary correction
  versions foundation spec and operational waiver to `/v2`;
- Python compilation, Git-Bash syntax checks for all three new Slurm workers,
  six-document Markdown-link validation, and `git diff --check` passed.

There are no external donor files or donor commits for this block, so
`docs/LEGACY_SOURCE_MAP.md` is unchanged. Existing repository HCWDL and PMARD
primitives were extended in place; the new campaign modules, scripts, workers,
contracts, runbook, and focused tests are additive. No SSH, Slurm submission,
cancellation, remote push, or final-test access occurred during this local
implementation.

Exact next task: commit and push the target-digest execution repair described
below, create a clean detached Tigris worktree at that full commit, publish one
immutable monitor and execution-repair recovery spec per old arm ledger,
preview then cancel only those six ledgers' exact IDs, dry-run all six recovery
closures, and submit the six replacement closures. The completed shared
foundation must be reused; it must not be rebuilt or edited. No additional
HCWDL-UB smoke is required or claimed.

The first Tigris launch attempt at `c05e3b7` correctly submitted no jobs but
exposed two launch-boundary defects: parent discovery required the unrelated
U/J campaign-completion report, and it named the real coupling lock as
`locks/coupling.json` instead of `locks/coupling_lock.json`. The intended
parallel experiment only requires the already authorized fixed-preprocessing
prefix. Foundation-spec v2 and operational-waiver v2 therefore authenticate
the exact coupling, endpoint-equality, and graph/recipe locks plus their
manifests without waiting for old U/J training descendants. This changes no
particle, graph, loss, seed, or resource semantics and still fails closed if
any required preparation lineage is absent or mismatched. The corrected
HCWDL-UJ/UB/recovery focus passes 86/86; the complete repository suite passes
463/463 in 204.10 seconds with one existing PRAD tensor warning and the local
pytest-cache permission warning. Python compilation, the revised waiver CLI,
and `git diff --check` also pass.

### HCWDL-UB U000 target-manifest digest execution repair (2026-08-15)

The first six-arm launch proved that all shared preparation and training
completed successfully through foundation lock `86841` and autolaunch `86842`.
Every U000-supervised arm root then failed before optimizer training with
`HCWDL-UB shared target manifest is not foundation-locked`, after unnecessarily
building its 300k/100k RAM views. The shared foundation artifacts authenticate
independently. Tigris reported:

- target manifest content hash:
  `5de8bde03d6813a686e9e0042d589537e6f0e1be5095d11193f86f73ed93aa68`;
- U000 training-report hash:
  `e4a6296e551d369be6d06ecb1bdf270251d4b5eb6545898841322186c883ddc3`;
- the foundation and target lock both recorded the latter report hash as the
  manifest hash;
- U000 report/checkpoint and target-lock/foundation-lock parent hashes matched.

Root cause was a local-variable shadow in `validate_target_manifest()`: after
authenticating the manifest, its parent-validation loop overwrote `digest` and
returned the final sorted parent digest, which was the teacher-report hash.
The manifest, FP32 logits, U000 checkpoint, U000 report, M0paired result,
balanced couplings, and endpoints are not corrupt and must be reused.

The validator now returns a dedicated `manifest_digest`. The arm runner
preflights shared-U000 lineage before constructing any RAM view. A distinct
`HCWDL_UNIFIED_BALANCED_EXECUTION_REPAIR_RECOVERY_SPEC/v1` path reconstructs
and binds the exact legacy defect, requires an explicit repair phrase, and
accepts a changed semantic-source map only when the changed files are exactly
the target validator and arm runner. It never rewrites the old locks. Ordinary
execution/source/resource recovery still rejects the mismatch. Recovery
reports bind the recovery spec, actual target-manifest hash, and repair-evidence
hash.

Local verification for this repair: focused HCWDL-UB tests pass 28/28; the
complete repository suite passes 464/464 in 238.65 seconds with the 14 existing
Matplotlib/Pyparsing deprecation warnings; five affected CLI help surfaces,
Python compilation, versioned-contract documentation, and `git diff --check`
pass. No SSH, Slurm cancellation/submission, lock mutation, or final-test access
occurred locally.

## HCWDL-UJ combined execution/resource recovery (2026-08-14)

The linear endpoint-preparation correction changes semantic-source bytes while
the operator also requested that replacement training jobs use 16 rather than
8 CPUs. The pre-existing source-recovery v2 contract preserves resources, and
the resource-only v2 contract preserves source, so neither may be repurposed
or bypassed. A distinct
`HCWDL_STRUCTURAL_FEATURE_EXECUTION_RESOURCE_RECOVERY_{SPEC,COMMAND_PLAN}/v1`
family now binds both reviewed changes without changing the original campaign's
scientific identity.

The combined recovery authenticates the exact cancelled/failed downstream
closure, original ledger and monitor, complete old/new semantic-source maps,
execution-only human classification, and complete old/replacement resource
maps. Resources may only increase; the GPU type cannot change; at least one
resource class used by the closure must increase. The intended 300k recovery
changes only `gpu_training.cpus` from 8 to 16 and retains 96G, six hours, and
one GH200. A dedicated worker environment and distinct authorization and
submission phrases prevent old v2 recovery artifacts from being interpreted
under the combined semantics.

Local evidence at this checkpoint: the focused recovery/resource/worker suite
passes 7/7; the broader HCWDL/stream/repair/cache suite passes 185/185; and the
complete repository suite passes 436/436 with the 14 existing
Matplotlib/Pyparsing warnings. CLI help and `git diff --check` pass. Tigris
accepted corrected commit `f698889`, cancelled only the exact old-ledger IDs,
authenticated a 41-task failed closure, verified 39 training commands at 16
CPUs/96G/six hours/one GH200, and published the new live recovery ledger under
`recovery/linear_16cpu_f6988892_r1/live`. Corrected phase timing and completed
training evidence remain pending.

The reusable diagnosis and implementation guidance is in
[HCWDL_RAGGED_PREPROCESSING_PERFORMANCE_GUIDE.md](HCWDL_RAGGED_PREPROCESSING_PERFORMANCE_GUIDE.md).

## HCWDL-UJ linear endpoint preparation and ordered CPU overlap (2026-08-14)

Live 300k recovery jobs `84176`, `84177`, `84179`, `84182`, `84193`, and
`84204` each remained before their first validation checkpoint after roughly
four hours, while HLT-only job `84211` reached update 38,676/70,320 in about
one hour.  Slurm accounting showed approximately one CPU-hour per elapsed
hour for every blocked job.  This excludes ordinary optimizer training and
the already locked matcher/coupling production as the dominant cost.

The concrete defect was repeated whole-chunk ragged conversion.  The public
one-row offline projector converts every required Awkward branch into all
rows.  P0, D100, U, J, calibration, and audit callers invoked that projector
inside a row loop over a 4,096-row chunk; U/J partition construction also
reconverted every raw HLT branch per row.  The resulting work was quadratic
in chunk population.  Cache allocation itself was already preallocated and
linear.

The execution repair leaves `repair.py`, Shell Exact v1, coupling artifacts,
coordinates, graph, loss, inputs, endpoints, and checkpoint selection
unchanged.  `hcwdl_homotopy.py` now prepares every offline and HLT branch once
per chunk, validates the same count/length rules, and reuses exact raw feature,
validity, and float32-to-float64 p4 endpoints across P0, Shell, partition, and
carrier construction.  Coupling calibration/audit reuse the same prepared
chunk.  The homotopy stream bounds one in-flight chunk per worker, overlaps
chunk construction with a deterministic thread pool sized from
`SLURM_CPUS_PER_TASK`, and always consumes futures in canonical submission
order.  `HCWDL_UJ_VIEW_BUILD_WORKERS` may request fewer workers but is rejected
if it exceeds the Slurm allocation.  Workers now print explicit train-cache,
validation-cache, teacher-target, and optimizer-training phase boundaries.

Local evidence:

- pre-change focused baseline: 77/77 passed;
- post-change endpoint/homotopy/cache focus: 80/80 passed;
- broader HCWDL/CLI/contracts/smoke/repair/cache focus: 168/168 passed;
- complete repository suite: 435/435 passed with the 14 existing
  Matplotlib/Pyparsing warnings;
- exact prepared-versus-legacy raw feature, validity, and p4 arrays are locked
  across a seven-row fixture, and every required branch is converted exactly
  once rather than once per row;
- ordered one-worker versus four-worker emission is locked by regression;
- synthetic P0 preparation at 1,024 rows improved from 4.837 s to 0.364 s
  (13.30x) with exact p4 equality; because the removed term was quadratic,
  the production 4,096-row chunks should benefit more;
- Python compilation and `git diff --check` pass at this checkpoint.

This is local execution evidence, not Tigris acceptance.  The running jobs
remain pinned to the old source and cannot acquire this fix.  After exact-ID
closure handling, the corrected commit requires the existing human-authorized
source-recovery path; completed reports and compatible rolling checkpoints
remain reusable. The eight requested CPUs are now available to bounded
concurrent chunk construction; raising the request above eight should wait for
the corrected Tigris timing rather than masking the eliminated algorithmic
defect.

## HCWDL-UJ exhaustive-audit source parallelism repair (2026-08-14)

The 300k v2 coupling audit timed out twice: original job `83434` exhausted its
four-hour request, and resource-recovery job `83884` exhausted twelve hours.
This is an operational implementation failure, not a failed coupling
invariant or a scientific result. The audit task requested eight CPUs but
`audit_full_roles()` executed the entire 300,000-train plus
100,000-validation source population in one Python process. For every jet it
also accumulated all twenty structural thresholds, reconstructed and compared
the P0/D100/HLT endpoints, rebuilt the residual cost matrix, reran the exact
Hungarian optimum check, and later performed the independent ROOT reread. The
extra walltime therefore left seven requested CPUs unused by the dominant
loop.

The exhaustive audit is now partitioned into one process task per
authenticated train or validation source unit, using up to
`SLURM_CPUS_PER_TASK` workers. Every source still performs the complete row,
endpoint, conservation, transition, and solver-optimum checks. Results are
reduced with exact integer addition in canonical train-then-validation source
order; endpoint and solver proofs use domain-separated, length-framed
per-source hashes. Independent sample rereads use the same source partition.
Completion order and worker count are explicitly excluded from scientific
identity, while the reduction scheme is recorded in the audit and validated
fail-closed. The frozen single-process implementation remains private as a
reference for future Tigris parity audits. No coupling, switch, endpoint,
graph, loss, or dataset semantics changed, and no contract version was
bumped.

Self-review also found that the pre-existing source-recovery worker would have
rejected any corrected file listed in the campaign's semantic-source map,
including this execution-only repair. Source recovery now preserves that
original map inside the unchanged scientific identity and separately binds
the complete corrected execution-source map, the exact old/new hash pair for
every changed semantic file, and an explicit human-authorized execution-only
classification. Both the recovery validator and worker checkout validate
that second map; resource-only recovery continues to require byte-identical
campaign source. This closes the recovery path without disguising changed
source bytes as the original campaign source.

Local evidence for this repair:

- focused HCWDL homotopy and CLI suite: 114/114 passed before the final
  execution-lineage regression, followed by 4/4 targeted reducer,
  process-pool, and validator tests;
- complete repository suite: 432/432 passed with the 14 existing
  Matplotlib/Pyparsing warnings plus one non-scientific pytest-cache warning;
- changed-source AST parsing, four recovery/task CLI help surfaces, v2 graph,
  pilot, and recovery contract identity checks, and `git diff --check` passed;
- no donor code or donor commit was used; unrelated user worktrees, figures,
  and plotting files remain untouched.

The current resource-recovery ledger must be monitored immutably and cancelled
by its exact IDs before a source-pinned failed-closure recovery is submitted.
That recovery reuses the completed calibrations, shards, manifests, switches,
and TOFF targets, retains the twelve-hour CPU-coupling envelope, and starts at
`coupling_audit`; no earlier data work or completed artifact is recomputed.
Real Tigris validation of the parallel audit remains the next required runtime
evidence.

## HCWDL architecture-factorial measured P0 walltime (2026-08-12)

The first 300k architecture-factorial pilot measured a preprocessing-bound
failure: `O_U` job `83191` received Slurm `SIGUSR1` at `05:57:39` under its
six-hour request, with exit `0:10`, empty Python stderr, peak RSS about 24 GiB,
and no `rolling_resume.pt`. It had not reached the first optimizer update; it
was still projecting the 300k train plus 100k validation native-offline P0
population and building the process-local RAM caches. This is neither OOM nor
a model/scientific failure, and a same-resource requeue would repeat the work.

The replacement campaign policy therefore keeps exact-HLT cells at 8 CPUs,
96 GiB, six hours, and one GH200, while P0 cells use the same CPU/RAM/GPU with
a sixteen-hour walltime. Tasks bind explicit `training_hlt` or `training_p0`
resource classes from their registered input domain; tests assert all four
exact command rows. The four-cell graph, inputs, seeds, losses, schedule, and
reports are unchanged. The old campaign must be cancelled only through the
exact IDs in its immutable submission ledger, and the replacement must use a
fresh source-pinned root.

## HCWDL-UJ canonical pilot-parent role-count repair (2026-08-12)

The first waived v2 300k dry-run attempt stopped safely before campaign
publication or submission with `HCWDL-UJ parent role counts differ`. The
canonical HCWDL 300k parent correctly records 300,000 train, 100,000
validation, and 100,000 sealed final-test rows; the validation-only U/J child
correctly records 300,000 train, 100,000 validation, and zero final-test rows.
`authenticate_parent()` had incorrectly compared the parent against the child
projection. It now authenticates the parent with its own HCWDL mode contract
and retains the existing child-side zero-final-test checks. A regression locks
both populations explicitly. The already published waiver and source parity
cannot be reused because this validator is semantic source; the corrected
commit requires new parity and waiver identities. The partial v2 smoke exact
IDs were cancelled only after its completed infrastructure evidence had been
bound into the old waiver, and no 300k jobs were submitted.

Local compilation and `git diff --check` pass. Focused pytest cannot collect
in the current Windows Python because NumPy is not installed; the pinned
Tigris environment must run the focused parent-population and waiver tests
before constructing the corrected dry run.

That corrected attempt then reached the next independent validator boundary
and stopped before publication with `dense D0c control has different HCWDL
parent lineage`. The implementation plan requires the selected coarse cold
`D0c` from the primary HCWDL graph; dense10/dense5 reports are separate,
possibly empty contextual imports. The canonical primary report does not and
should not carry the dense-supplement `parent_campaign_spec_sha256` field.
The importer now authenticates canonical path, primary graph/node payload,
unweighted recipe, split and source snapshot, assignment and qualification
locks, sole D25c teacher report, wrapper/engine equality, and checkpoint
bytes through `validate_completed_hcwdl_node()`. A focused regression locks
that schema. The second failed dry run again published no campaign and
submitted no 300k jobs.

## HCWDL-UJ v1 operational-evidence carry-forward (2026-08-12)

The human rejected repeating all graph-thinned v2 smoke fits after the
completed 80-fit v1 production-worker smoke. That decision is now represented
honestly by `HCWDL_STRUCTURAL_FEATURE_OPERATIONAL_EVIDENCE_WAIVER/v1`, not by
relabeling the partial v2 run as complete. The waiver binds the v1 campaign
and completion, completed v2 installed-Weaver parity, coupling/endpoint/TOFF
target/graph locks, the exact new source and semantic hashes, the 8 CPU/96G/
six-hour/GH200 pilot request, and the exact human authorization phrase. It
does not change any scientific graph, loss, dataset, endpoint, or final-test
rule. A completed measured v2 resource profile remains an alternative, not a
mandatory gate when this explicit waiver is present.

No jobs were cancelled or submitted by the implementation. The operator may
cancel only the exact partial-v2 smoke ledger IDs after publishing the waiver,
then create/dry-run/submit a new source-pinned 300k root. The two unrelated
worktree entries remain untouched.

## HCWDL-UJ reduced v2 graph (2026-08-12)

After the successful 80-fit v1 production-worker smoke, the user selected a
smaller primary screen before any 300k U/J submission. The implemented v2
graph retains ten equal nominal-L1 predecessor-KD transitions per primary
path: factorized `U020..U100 -> D80F..D0F -> M1F`, and joint
`J010..J100 -> M1J`. Its controls are five stationary D100 fits, ten
temperature-2 stationary HLT fits plus temperature-1 `S0_11`, and the same
seven direct/adapter controls with `U020P0KD`. The exact registry is 45 fits
and 66 total tasks. The completed v1 smoke remains immutable historical
evidence and is not relabeled as v2 acceptance.

Graph-bound coordinate, node, graph, recipe, graph-lock, campaign, command,
aggregate, completion, and both recovery families now use explicit `/v2`
contract identities. Coupling, endpoint, view, target, node-wrapper/runtime,
resource, and resume contracts remain v1 because their underlying semantics
did not change. Cache miniature keys, full-role displacement coordinates,
TOFF target consumers, seed aliases, temperature routing, dependencies,
report comparisons, and same-input metadata all derive from the reduced v2
registry. The earlier comparison-report bug that marked several exact-input
controls `false` is fixed by comparing authenticated input-domain signatures.

The requested 300k `gpu_training` row is enforced in campaign creation and
validation as exactly 8 CPUs, 96 GiB, six hours, and one GH200. The v2 smoke
must still demonstrate 25% resource headroom before its profile can authorize
the pilot. No Slurm job was submitted by this change.

Final local evidence from the implementation tree:

- focused homotopy contracts/graph/campaign/reporting: 45/45 passed;
- complete HCWDL/CLI focus: 107/107 passed;
- complete repository suite: 423/423 passed with the 14 existing
  Matplotlib/Pyparsing warnings;
- bounded synthetic end to end: all 45 fits completed, no final-test access;
- 13/13 changed graph-bound v2 contract identities and 7/7 primary CLI help
  surfaces passed;
- Python compilation and `git diff --check` passed (line-ending notices only).

The existing unrelated `.worktrees/hcwdl-rkd-local` modification and
`.worktrees/hcwdl-u-rkd-local/` directory remain untouched. No donor code or
donor commit was used. The next acceptance layer is one clean, pushed,
source-pinned v2 Tigris smoke using the already authenticated HCWDL parent;
only after completion, measured resource profiling, and a complete dry run may
the user separately authorize the 300k pilot.

## HCWDL architecture-factorial BF16 repair (2026-08-11)

The first genuine Tigris architecture-factorial smoke at source `c2a5510d`
authenticated its parent and passed architecture-check job `82058`. Unified
cells `H_U` (`82059`) and `O_U` (`82061`) completed, while split cells `H_S`
(`82060`) and `O_S` (`82062`) failed before producing scientific results. In
production BF16 autocast, each Weaver encoder returned a BF16 embedding while
`SplitScoutingParticleTransformer._encode_nonempty()` allocated its scatter
destination from the FP32 cached input. PyTorch correctly rejected the
cross-dtype indexed assignment. This is an implementation-only failure; it
does not implicate either authenticated input view or the factorial design.

The split encoder now allocates from the encoded embedding and uses
differentiable `index_copy`, preserving the active autocast dtype and gradient
flow. The architecture preflight now executes both models under the same BF16
autocast policy as training and records input, autocast, and output dtypes. A
focused regression exercises the split model's FP32-cache/BF16-output path.
Source compilation and `git diff --check` pass. The local Windows Python does
not provide PyTorch, so focused/full pytest cannot run in this checkout; the
replacement source-pinned Tigris smoke must first pass its strengthened GPU
preflight and then all four two-update cells. The completed old unified cells
are not mixed with repaired-source cells: the scientifically clean smoke is a
new campaign root, and the old failed campaign's aggregate/completion jobs may
be cancelled only by their exact IDs `82063` and `82064`.

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
