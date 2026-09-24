# CMS2JC2 CPU response-development study

Date: 2026-09-23. Implementation-authoritative, separately authorized development
extension to the three-family response plan. This is not its full science,
selection, confirmation, classifier, or JetClass2 transfer stage.

## 2026-09-24: authorized CPU64 execution extension

The user requested 64-CPU preparation after the 16-CPU A/C and B jobs showed
only roughly 9 and 7 active cores in a short sample. This section supersedes
the resource/file-scheduling defaults below **only for new DEV_STAGE64/v1
comparisons**. Existing DEV_STAGE/v1 artifacts, workers pinned to old commits,
and all scientific definitions remain unchanged.

The new `create-compare64` CLI creates a fresh, disjoint study/comparison root
at a clean pushed commit. It imports authenticated samples, histogram ranges,
and the completed 20k association-confirmation report from the previous study.
It verifies byte-identical scientific source, CMS inputs, compatibility,
numerical environment and canonical memberships. Only the explicit development
execution/CLI/documentation source allowlist may differ. It does not import
old models, claims, pending jobs or acceptance; it neither cancels jobs nor
changes the old roots. The original preparation, pilot and 20k confirmation
are **not rerun**. All original held-out access boundaries remain sealed.

Both fit allocations request **64 CPUs, 128 GiB, 24h, no GPUs**, debug by
default. The comparison still has 17 tasks with the same independent family
evaluation/report dependencies. The conservative sum-of-requests CPU upper
bound is 143, replacing 47 for this execution only; scheduler admission is
checked before submission. No speedup or full-time CPU utilization is promised
without a genuine SPORC measurement.

Each fit allocation loads the same 16k location and 4k residual pairs once per
source file, using the existing authenticated read capability. Files are read
in parallel, not reopened and rehashed for every small chunk. Selected pairs
exist only in RAM; the registered array/key payload guard is 8 GiB (not a claim
about total process RSS). The reader stage remains file-parallel and is not
claimed to occupy all 64 cores. No raw or calibration data are spilled to disk.

After reads finish, both roles share a 64-process association/record pool with
eight jets per work item and at most 128 outstanding items. Any completed chunk
releases another work item: slow files no longer hold whole CPU lanes. Native
libraries inside each preprocessing process are single-threaded. Completed raw
chunks are released. The final few chunks, I/O, startup and fitting may use
fewer than 64 CPUs; the allocation is not an assertion of 100% utilization.

Every chunk uses its original **per-file** record quota; each file's chunk
reservoirs merge back to the same stratified bottom-hash sample. Population
counts are summed before N/k inclusion weights are computed. No per-chunk
quota, different source mixture, looser association limits, approximation,
or removal of unresolved jets from evaluation is introduced. Canonical file
order and ordered identity hashes reproduce legacy calibration reports. A/C
reuse the prepared records; B's separate allocation prepares the same records.

The parent logs startup, read completion and global completed-jet/chunk counts,
plus 15-second wait heartbeats. Counts are completed work, not a guessed ETA.
Completed-jet progress includes unresolved jets; `resolved` is a separate
association diagnostic, not the fraction of the job completed.
An eight-jet chunk still in progress is not yet counted. Normal module-level
fit/publication messages follow preparation. A/C numerical fitting retains
16 threads; **B fitting retains one thread** because the 16-thread diagnostic
segfaulted. The 64 CPUs address preprocessing, not a claim of parallelizing B's
single numerical optimizer. No running allocations are resized or resumed.

Create/dry-review at the new pushed checkout, then separately authorize:

```bash
python -s "${PROJECT_DIR}/scripts/cms2jc2_response_dev.py" create-compare64 \
  --parent-spec /home/ryreu/atlas/HLT_Classification/checkpoints/cms2jc2_dev_de9890b2_r1/stages/confirm_search_r1/stage_spec.json \
  --project-dir "${PROJECT_DIR}" --source-commit "${NEW_COMMIT}" \
  --root "${NEW_ROOT}" --partition debug

SPEC="${NEW_ROOT}/stages/compare64_r1/stage_spec.json"
python -s "${PROJECT_DIR}/scripts/cms2jc2_response_dev.py" dry-run --spec "${SPEC}"
python -s "${PROJECT_DIR}/scripts/cms2jc2_response_dev.py" submit \
  --spec "${SPEC}" --execute \
  --authorization-phrase "AUTHORIZE CMS2JC2 CPU DEVELOPMENT COMPARE EXACT PLAN" \
  --reviewed-plan-hash "${REVIEWED_PLAN_HASH}"
```

Use the existing `monitor` and `results` commands on this **new** SPEC. Keep
the original metadata/worktree paths available for authenticated reuse. Any
decision to cancel the two older fits and their descendants is separate and
must use their exact original study-bound IDs; these tools never cancel them.

## Purpose and evidence

The SPORC 20k probes at f966dd804ed9ca8ca93c9c3227d1a7f036ef4434 completed
A_L and C_L; B_L exited with signal 11 after logging additional_count.
Only 10570/16000 location and 2607/4000 residual jets were fully resolved by
the association algorithm. Unresolved jets supplied no fitting records.
Downstream evaluation jobs were cancelled. These are execution/coverage results,
not evidence of CMS closure. The exact B crash cause is not established.

The user authorized CPU development on RC with thousands of jets, fixed
label-independent randomness, common CMS/JetClass2 fields, and incremental plots.
Existing jobs, roots, raw files and scientific contracts remain untouched.

## Frozen population

Import only authenticated CMS inventory and response-role metadata from a
completed preparation. Validate its receipt, exact bytes, historical training
boundary and source semantics. Do not import its old September-10 JC2 inputs,
models, fitted statistics or acceptance. Record the intended transfer release
as jetclass2_10M_20260918_dzfix. This study reads **no JC2 particles**; subsequent
transfer requires a new dzfix inventory/profile-bound spec and authorization.

Create a development registry inside response_fit only. Keep original
fit_residual files for 4000 residual-calibration jets. From fit_location files,
reserve up to two whole files per production-source category for 10000 development
evaluation jets; choose by SHA256(seed, file hash). Remaining files supply
16000 location jets. Require each source category in all three roles and at
least one remaining location file per source. The actual signal source has only
two location files, so reserve one, not two; report this limited source-group
coverage honestly. Never borrow response_select,
response_confirm, historical validation or final-test rows to meet capacity.

Apportion each exact jet budget to source categories using the original
response_fit source mixture, then proportionally to eligible files within each
category. Thus the different role file counts cannot change the source mixture.
Select
bottom SHA256(seed, file hash, tree cycle, entry) rows within each file. The
2000-jet pilot is a nested location subset; the 20k confirmation of association
settings is the union of location and residual rows. Freeze four disjoint
2500-jet evaluation shards using a second label-independent hash. All candidates
use the same masks and canonical CMS jet identities. Labels are not read.
File grouping does not certify unavailable cross-file event independence.

These evaluation files are development data after inspection. They are not an
independent final confirmation, and cannot authorize a qualified detector claim.

## Stage 1: pilot, explicit small CPU allocation

One prepare task validates inputs, publishes frozen masks and fits diagnostic
histogram ranges from the 16000 location jets only. Then four independent
association jobs process the same 2000 pilot jets, at gate 0.10 with unchanged
cost, tie-breaking, dustbins and merge/split hypotheses:

| Policy | Component objects | Hypotheses/component | Search nodes/component |
| --- | ---: | ---: | ---: |
| BASE | 64 | 4096 | 100000 |
| SIZE | 256 | 16384 | 100000 |
| SEARCH | 64 | 4096 | 1000000 |
| BOTH | 256 | 16384 | 1000000 |

Exhausted searches remain unresolved, not approximate truth. No quality-based
failure or automatic winner selection. Report per-jet count, offline momentum,
crowding, displacement summaries, resolution/reasons/search effort/runtime;
include resolved/unresolved conditional coverage and distributions. These
compact diagnostics are not raw constituent libraries.

Two separate B_L additional-count diagnostic jobs use BASE and the pilot rows,
with 1 and 16 numerical threads respectively. Their ROOT/association processing
is allocation-bounded. Enable Python faulthandler, disable core dumps and print
the module, data shapes and numerical environment. No automatic fallback or
claim that single-threading repairs the observed native crash.

Pilot tasks request prepare 1 CPU/16 GiB/4h; association 8 CPUs/32 GiB/12h;
B diagnostics 16 CPUs/128 GiB/12h. Maximum concurrent CPUs: 64.

## Stage 2: association confirmation on 20k development-fitting jets

After inspecting the pilot, explicitly create a new confirm stage choosing one
of its four registered policies. Require its authenticated pilot output; the
stage runs that policy on all 16000+4000 fitting/calibration jets. Request
16 CPUs/128 GiB/24h. Publish timing/RSS/coverage even below 99%. This stage
confirms computational behaviour only; it does not open response_confirm.
No automatic downstream submission. Different policies/attempts need fresh roots.

## Stage 3: response comparisons and independent family diagnostics

Explicitly choose numerical B threads (1 or 16) after inspecting its pilot
diagnostics. A successful small fit is not evidence that the larger fit cannot
crash; an unsuccessful pilot remains visible, not a gate that blocks A/C.
A combined
fit_AC allocation prepares common location/residual records once in RAM and
fits A_L then C_L, publishing each response and its receipt immediately.
A separate fit_B allocation fits B_L. Both request 16 CPUs/128 GiB/24h; never
run several multithreaded fits inside each allocation. Calibration reservoirs
remain bounded at 2M records per role/module, with recorded inclusion weights.

Each family has four independent CPU evaluation shards, 1 CPU/32 GiB/12h,
followed by its own report task (1 CPU/16 GiB/2h). Evaluation uses afterany on
its fit allocation but must authenticate its own completed model receipt before
reading data. Thus a later C failure cannot prevent evaluation of published A.
An absent/corrupt model fails that family's evaluation, never fabricates a row.
Because A/C share a fit allocation to reuse RAM records, A evaluation waits for
that allocation to finish/fail; early A publication is not an automatic Slurm
release trigger. It does make A independently usable when that allocation ends.
An authenticated but statistically unestimable model instead produces a
successful explicit unestimable report, with zero evaluated jets and no scores.
No all-family barrier controls per-family reports. At most 47 CPUs can overlap.

Use replicas 0/1/2 with the existing frozen common-random-number generator.
Every evaluation jet contributes, including empty outputs and jets that would
be unresolved by association. Fast development diagnostics do not rerun the
combinatorial association on each proxy: they compare whole collections,
PID counts, particle pT/E/eta/phi/radial distributions, valid displacement/error/
significance distributions, availability, summed/vector jet pT, mass, axes,
width, two-point correlators and jet-summary correlations. They do **not** claim
to replace the full six-block association/transition-based science score.

Freeze log1p positive quantities, asinh signed tracking (mm scale 1), and shared
location-fitted histogram ranges. Count every underflow/overflow and missing
quantity. Publish exact raw means and full-observation histogram counts alongside
approximate histogram TV and mean shifts. Plot per-PID counts and all registered
scalar distributions for offline, real HLT and each proxy replica. Conditional
panels use offline count (<50, 50..99, >=100), summed pT (<500, >=500 GeV),
median R=.05 crowding (<3, >=3) and max valid |d0| (<.1, >=.1 mm), with empty/
missing states separate. Per-jet correlation moments are included; particles
or replicas are not independent jets. All figures are exploratory, with no
automatic physics winner or qualification. Later source-group uncertainty and
the untouched selection/confirmation stages remain required.

## Execution, lineage and storage

Dedicated cms2jc2_dev_<commit8> roots and c2jd_ job prefix. Partition is explicit
at creation (debug default), account reu-aisocial, QoS qos_tier3, one node/task,
no GPU request. Validate partition availability and sbatch --test-only for
every resource shape before any live submission. Record actual allocation,
exact source, input checksums, numerical environment, walltime and RSS. Do not
pretend a selected partition is CPU-only hardware without scheduler evidence.

All stages have canonical dry plans and exact authorization phrases. Immutable
submit intents/receipts, exclusive worker claims and output receipts prevent
duplicate writers; ambiguous acknowledgements require explicit reconciliation.
No cancellation/hold/requeue/remote installation commands. Failed attempts are
retained; use fresh roots for changed source. Source and numerical environment
must match at task start/end. No mutable latest aliases or timestamp identities.

Reuse raw readers/bridge/RNG/families without relaxing their contracts. Producer
messages confirm CMS cm versus Delphes mm; the existing conversion already
matches. Exact sign/reference/weight equivalence stays provisional. Only common
offline fields and their derived neighbourhoods condition generation; no truth,
labels, CMS-only track quality or JC2 HLT information is available to models.

Persist memberships, fitted models, compact histograms/metrics and bounded plots,
not raw particle caches, synthetic datasets, fitting records or core dumps.
Existing 12-GiB total/2-GiB reports+figures caps apply across this study's stages,
including failed attempts, with atomic publication and free-space headroom.
This development study cannot authorize full-size science or transfer.

## Testing and release boundary

Require focused tests for nested sampling, file/role exclusion, corrupt masks,
source/import drift, exact CPU-only DAG/dependencies, ambiguous submission,
same-record A/C fitting, independent family outputs, all-jet diagnostics,
underflow/overflow, fixed replicas, immutable publication and actual tiny ROOT
through fitting/evaluation/report. Tiny tests are not RC resource acceptance.
Queue readiness means tools and reviewed plans are available; the first pilot
provides real SPORC development measurements, not an already-passed gate.

## Exact stage interface

After committing/pushing these changes, use a **clean detached worktree at that
exact pushed commit** on SPORC and activate atlas_kd_sporc. Do not run from an
old campaign worktree. Set PROJECT_DIR to that worktree, DEV_COMMIT to its full
commit, and DEV_ROOT to a new sibling checkpoint directory. The established
preparation is:

`/home/ryreu/atlas/HLT_Classification/checkpoints/cms2jc2_response_prep_f966dd80_r1/preparation_spec.json`

The following illustrates the explicit first-stage API; no command cancels or
changes another campaign. Run in a subshell if using shell fail-fast options so
a guard failure cannot close an interactive SSH session.

```bash
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PROJECT_DIR}/src"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
CLI="${PROJECT_DIR}/scripts/cms2jc2_response_dev.py"
python -s "${CLI}" create \
  --project-dir "${PROJECT_DIR}" --source-commit "${DEV_COMMIT}" \
  --preparation-spec "${PREPARATION_SPEC}" --root "${DEV_ROOT}" --partition debug
python -s "${CLI}" stage --study "${DEV_ROOT}/study_spec.json" \
  --stage pilot --name pilot_r1
SPEC="${DEV_ROOT}/stages/pilot_r1/stage_spec.json"
python -s "${CLI}" dry-run --spec "${SPEC}"
```

Review its commands, source, CPU/memory/time requests and content_hash. Then:

```bash
python -s "${CLI}" submit --spec "${SPEC}" --execute \
  --authorization-phrase "AUTHORIZE CMS2JC2 CPU DEVELOPMENT PILOT EXACT PLAN" \
  --reviewed-plan-hash "${REVIEWED_PLAN_HASH}"
python -s "${CLI}" monitor --spec "${SPEC}"
python -s "${CLI}" results --spec "${SPEC}"
```

Submission checks current site admission with scontrol and sbatch --test-only;
it does not assume debug's current limits or that CPUs are immediately free.
Use a different explicit partition at **study creation** only after reviewing
its CPU limits; do not silently edit a queued spec. A rejected request submits
no live jobs. Existing scheduler jobs are never altered by this interface.

After all pilot jobs are terminal and results inspected, create confirm with
`--parent-spec <pilot spec> --policy BASE|SIZE|SEARCH|BOTH --stage confirm
--name confirm_r1`. Dry-run and separately submit with the CONFIRM phrase.
After its 20k report exists and jobs are terminal, create compare with
`--parent-spec <confirm spec> --b-threads 1|16 --stage compare --name compare_r1`.
Dry-run and separately submit with the COMPARE phrase. Each has its own reviewed
plan hash; no stage auto-queues the next. Later developmental attempts use fresh
stage names, with the same pinned study/source/populations. Changed source needs
a new study root. Unknown scheduler states or ambiguous intents block new
allocations until inspected; `reconcile --spec ... --task ... --job-id ...`
authenticates one explicit existing job, never resubmits it.

The results command prints report/figure locations without reading particle
data; a missing terminal report is not labeled a successful or failed physics
result. The standalone monitor queries only that stage's exact journaled IDs.
