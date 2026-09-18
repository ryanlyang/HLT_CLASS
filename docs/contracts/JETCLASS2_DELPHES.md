# JetClass2 Delphes migration contracts

Authority: [active migration plan](../plans/JETCLASS2_DELPHES_DATASET_MIGRATION_IMPLEMENTATION_PLAN.md)
and its [20260918 dz-fix snapshot transition](../plans/JETCLASS2_DELPHES_DZFIX_500K_MIGRATION_PLAN.md).
Implementation: `src/hlt_classification/jetclass2_delphes/`. This is a new
benchmark, not a compatibility mode for the FullSim campaign.

## Implemented local contract surface

All families below use the `JETCLASS2_DELPHES_` prefix, canonical content
hashes, and explicit parent identities. Publication uses the repository's
atomic immutable JSON/NPZ primitives. Absolute dataset paths and timestamps do
not determine row identity; copies are verified against relative paths, file
SHA256, latest ROOT tree cycle, entry count, and branch schema. The inventory
records its actual implementation-file hashes and Git HEAD separately: HEAD
alone is explicitly not proof of clean source. Provenance-free initial local
previews are not eligible inputs to the finished inventory validator.
Families use `/v1` unless explicitly versioned below. The original full-data
foundation is retained as `/v1` but is not the selected production population.

| Family | Meaning |
| --- | --- |
| `ASSUMPTIONS`, `LABEL_MAP`, `SELECTION` | Provisional producer semantics and frozen benchmark |
| `IMPLEMENTATION_SOURCE`, `INVENTORY`, `SPLITS` | Source bytes, latest cycles, complete file-group partition |
| `SPLIT_DESIGN`, `ROLE_MEMBERSHIP`, `SPLIT_REGISTRY`, `SPLIT_PROFILE` | Exact nested training subsets and shared validation/test membership inside those reservoirs |
| `INPUTS`, `VIEWS`, `MODEL`, `METRICS` | New raw/model/physics interface |
| `FOUNDATION_SPEC`, `ASSIGNMENT_SHARD`, `FOUNDATION_LOCK`, `SAMPLE_AUDIT` | Preparation intent, compact arrays, coverage and bounded checks |
| `RECIPE`, `CAMPAIGN_PLAN` | Shared optimizer and fresh-reference four-spine graph |
| `PROBABILITY_BANK` | Parent/role/temperature/class/ordered-identity-bound KD probabilities |
| `KERNEL_TRAINING_REPORT`, `WEAVER_PARITY`, `LOCAL_OR_REMOTE_ACCEPTANCE` | Training-component evidence; not a production execution authorization |
| `EXECUTION_SITE`, `READINESS_SPEC`, `RESOURCE_MEASUREMENTS` | Explicit site, readiness-only dependency chain, diagnostic measurements (not execution authorization) |
| `PROFILE_ATTEMPT_SPEC`, `PROFILE_ATTEMPT_RESULT`, `RUNTIME_PROFILE/v3` | Profile-only reuse of completed preparation, source/job/result lineage, explicit debug A100 measurement to tier3 production transfer |
| `INSTALLED_ENVIRONMENT/v2`, `RUNTIME_PROFILE/v2`, `CAMPAIGN_SPEC/v3` | Installed Weaver source bytes/numerical versions, genuine site-bound measured execution evidence, exact pinned production graph |
| `TASK_REPORT`, `AGGREGATE`, `COMMAND_PLAN`, `MONITOR`, `RECOVERY`, `SUBMISSION_INTENT` | Verified selected outputs, validation results, exact Slurm dependencies, terminal-state recovery and ambiguous-submission protection |
| `PARTIAL_SNAPSHOT_PLAN`, `PARTIAL_SNAPSHOT_TRANSFER`, `PARTIAL_SNAPSHOT_EXTRACTION` | Capacity-proved partial file selection, deterministic archive bytes, safe verified extraction and mandatory destination re-inventory |
| `SALIENCE_DATASET_TRANSITION_PLAN`, `SALIENCE_DATASET_TRANSITION_RECEIPT` | Old-live/new-dry lineage proof, exact active old job IDs, and all-old-jobs-terminal evidence before new submission |

### Dz-fix partial snapshot and transition contracts

The September 18 dz-fix producer directory is live input, not an identity. A
candidate local inventory first authenticates every currently copied stable
file. `PARTIAL_SNAPSHOT_PLAN/v1` binds that parent inventory and takes the first
source-proportional deterministic file prefix whose actual seed-20260910
whole-file split has every class and at least the registered train/evaluation
capacities plus the declared headroom. Its projected inventory and split hashes
must be reproduced at the destination.

`PARTIAL_SNAPSHOT_TRANSFER/v1` binds a normalized USTAR archive to that exact
plan and inventory. `PARTIAL_SNAPSHOT_EXTRACTION/v1` proves exact archive bytes,
member coverage/order/metadata and extracted file hashes. Neither contract is a
destination inventory: ROOT cycles, schema, paths and hashes must still pass a
fresh inventory and split replay on SPORC.

`SALIENCE_DATASET_TRANSITION_PLAN/v1` is valid only after both old and new
production specs validate, the old ledger is a complete live ledger, the new
ledger is the canonical complete dry ledger, inventories and roots differ, and
the scientific graph/role counts agree exactly. It records current states and
only exact active old-ledger IDs. It performs no cancellation or submission.
After the user executes the printed exact-ID cancellation, the corresponding
receipt requires every old-ledger job to be terminal. Unknown or absent states
fail closed. The new live submit remains a separate explicit authorization.

## Selection and splitting

Default: require `hlt_matched=True`; map labels from `jet_label`, never a
filename. Collapse raw labels 161–187 into QCD; retain the ten available neutral
signal types at IDs 0–3 and 9–14. The model class order is QCD, X_bb, X_cc, X_ss,
X_qq, X_gg, X_ee, X_mm, X_tauhtaue, X_tauhtaum, X_tauhtauh.

QCD-labelled rows in `train_higgs2p` are excluded by default; including them is
an explicit differently hashed policy. Signal labels are not renamed or
silently discarded just because their file is under `train_qcd`. Unknown raw
labels fail closed. No inherited tight-ID, jet-rank, kinematic, or 16-particle
selection is applied. Retained jets must have nonempty HLT and offline sides.

The seed-20260910 file-group splitter first establishes class coverage, then
greedily minimizes normalized class-count deficits against 60/20/20 targets.
Ordering uses largest class-relative file mass and seeded path hashes, not
filename ranges. Every class must occur in each role. Duplicate content,
duplicate/case-colliding paths, group overlap, and counts inconsistent with
inventory fail. File/content disjointness is proven; cross-file generator-event
independence remains the authorized provisional assumption.

Final-test labels/count metadata are part of pre-split QA. Once roles are
frozen, ordinary readers, caches, assignment tasks, probability banks and
training cannot access that role. There is no public `allow_test=True` switch.

### Registered training-data scaling study

The full-file partition is now a reservoir, not a request to train on all rows.
The production study has four profiles:

| Profile | Train | Validation | Final test |
| --- | ---: | ---: | ---: |
| TRAIN_500K | 500,000 | 1,000,000 | 1,000,000 |
| TRAIN_1M | 1,000,000 | same identities | same identities |
| TRAIN_1P5M | 1,500,000 | same identities | same identities |
| TRAIN_2M | 2,000,000 | same identities | same identities |

The registry is campaign-independent and selects no models. It binds the
existing inventory and whole-file reservoir split, class order, exact quotas,
seed 20260911, selection algorithm and actual producer-file hashes. A ROOT
file never changes its outer role; unused evaluation rows are not training
capacity. File disjointness remains proven, event independence provisional.

Within each role/class, sort candidates by SHA256 of the UTF-8 string
`JC2/subset/v1/{seed}/{inventory_hash}/{role}/{relative_path}/{tree_key}/{entry}`.
Lexicographic digest order uses all 32 bytes, with inventory file index and
entry as tie breakers. The first subset reserves one row per class then assigns
the remaining budget proportionally to remaining class capacity using exact
integer Hamilton remainders, ties by class index. Each larger training subset
adds a proportionally allocated increment from remaining capacity and extends
the SAME classwise ordering. No RNG implementation or read chunk affects it.
The same allocation rule independently fixes each evaluation set. Sizes that
exceed capacity or cannot cover every class fail, never silently downsize.

Each `ROLE_MEMBERSHIP/v1` stores exact masks: base64 of NumPy-style `packbits`
with little bit order, bit i meaning latest-cycle entry i. Bytes are decoded
canonically; nonzero trailing padding bits, wrong lengths and duplicate/missing
file records fail. Per-file and total class counts are authenticated and
checked against inventory capacities. These are explicit membership sets,
not a seed-only recipe or dense feature arrays. Canonical identities remain
inventory/file/tree/entry hashes, independent of profile and physical root.
Readers iterate inventory file order then increasing selected entry; the
training sampler shuffles only those rows.

`SPLIT_REGISTRY/v1` contains four training memberships and one shared pair of
evaluation memberships. Exported `SPLIT_PROFILE/v1` artifacts bind the registry
hash and contain only the chosen training membership plus the shared pair.
Registry validation proves nesting with packed-mask subset checks; profile
validation checks parents, exact quotas, masks and inherited roles. Full
`verify` additionally replays scalar-metadata selection against raw source
checksums and compares every mask; it also authenticates exported profiles.
Metadata-only generation/replay can read final-test `jet_label` and
`hlt_matched`, never particle branches or predictions. This is explicitly
recorded and does not unlock ordinary test access.

Subset foundations use `FOUNDATION_SPEC/v2` and schedule matching ONLY selected
nonempty ordinary-role file subsets. `CAMPAIGN_PLAN/v2` exposes profile,
registry, membership hashes and counts; `CAMPAIGN_SPEC/v2` introduced a profile
foundation. The production creator rejects a legacy full-data reservoir, even
if older full-population resource evidence exists. Caches, acceptance sampling,
assignment joins and bank storage estimates consume exact subset counts.
Every profile needs its own matching foundation and measured acceptance in the
current implementation; cross-profile artifact reuse is not silently assumed.
Site-bound production now uses CAMPAIGN_SPEC/v3 and preserves those same
membership requirements; old v2 executions/artifacts are not rewritten.

HLT CE, offline CE and all ladder nodes train fresh on the chosen profile.
No full-data teacher is imported into a smaller-size comparison. Optimization,
batch size, early stopping and per-pass validation stay unchanged; differing
update counts/compute must be reported. Report absolute metrics alongside
recovery relative to that size's fresh references. Test remains sealed.

## Inputs and views

Raw fields are stored p4, charge, five exclusive PID flags, d0/error and
dz/error. Required values must be finite, pT/energy positive, charge physical,
PID/charge applicable and errors nonnegative. Zero error is unavailable
uncertainty, not a request to erase a finite displacement or divide by epsilon.
The unresolved interpretation and stored-mm convention remain provisional.
No missing quality, track-fit or lost-hit fields are fabricated.

The model sees exactly the documented 17 JetClass-style features plus its own
p4 and mask. Analytic logarithm offsets/scales and clipping follow
`data/part_inputs.py`; they are not old-data fitted normalization. Relative
angles use the visible constituent-sum axis of the supplied view, including eta
reflection and wrapped phi. Errors are clipped directly to [0,1]; displacements
use tanh. Stored producer-relative angles are not substituted into an
intermediate view with a different axis. No identity, matching index, label,
source category, or association diagnostic becomes a model channel.

Capacity is the pre-split selected-count maximum rounded up to a multiple of
16 (240 for the audited snapshot). Overflow fails rather than truncates.
RAM caches pack only real tokens; batches pad to their own maximum, at least
16. Weaver's random training-time sequence trimming is explicitly **disabled**
for every new reference and student, to preserve the no-truncation policy.
The otherwise canonical Weaver configuration has 17 inputs and 11 outputs.

The existing exact integer full-cardinality matrix solver is reused, with
global p4 directions, quantized delta-R and the existing tie hierarchy. Old
schema-dependent 21-field adapters are not reused. Poor matching distances are
reported, never grounds to reject a valid jet.

Maximum cardinality leaves residuals on at most one side. Consequently there
are only insertion/removal edits here, whose existing mass rule simplifies to
`round_half_up(1e6 * (3 + 4*pT_fraction + 2*energy_fraction))`. No residual
substitution-scale fit is needed. Mass-balanced circular switches are stratified
by edit kind, PID, charged applicability and uncertainty validity; their
hash-domain is new and label/teacher/branch independent. Rational coordinates
give nested structural edits across grids. Intermediate order is HLT slots,
then remaining native offline tails. Exact U000 returns native offline order;
exact D000 returns the original HLT object without touching offline inputs.

D interpolates matched p4 and applicable numeric measurements. Charge/PID
switch atomically. A charged/neutral applicability change makes all measurement
fields follow that same identity choice. An uncertainty-validity change switches
its value/error pair atomically. Unmatched HLT slots remain native HLT. This
avoids inventing intermediate zero-error significances or fractional identities.

## Persistence and execution boundary

Assignment arrays contain only ordered 32-byte row digests, integer offsets,
and int32 HLT-to-offline maps. Particle views and training caches stay in RAM.
Preparation uses bounded ordered process futures and one numerical thread per
worker; a conservative allocation bound must fit the caller's explicit RAM
ceiling before processing starts. Model batches are padded in RAM, never spilled
to memmap. Workers must separately budget train plus validation caches.

Probability shards have at most 100,000 rows (approximately 7.6 MB of array
payload), FP32 eleven-class probabilities and 32-byte identities. Training
targets are T=2; validation probabilities are T=1. Readers validate every parent,
checksum, contiguous shard boundary and ordered row join. All 26 proposed banks
over the old full train+validation reservoir implied roughly 13.2 GiB. The
new 500k+1M profile needs about 2.76 GiB; 2M+1M needs about 5.52 GiB, before
assignments/checkpoints. Neither is a promise of zero disk use. The production wrapper verifies the selected
checkpoint and training-report output inventory before publishing teacher banks.

The kernel uses C25P75, forward KL with one T² factor, AdamW, batch 256 and the
registered warmup/hold/decay/floor schedule. CE references receive no targets.
Validation selects by AUC, CE, log-R50, then earliest update; patience uses
meaningful AUC improvement while selection still records smaller improvements.
Selected weights are retained/restored in RAM. No rolling optimizer state or
resume checkpoint is written. Short acceptance reports are explicitly
nonscientific and cannot substitute for full training reports.

The scientific graph has two fresh references, 29 downstream fits and 26
teacher publications. It imports no old weights or banks. The original preview
CLI remains non-executable. `scripts/jetclass2_delphes_production.py` provides
the separate source-pinned execution wrapper: 31 train + 26 reduce + aggregate
+ completion = 59 jobs. No live submission happens by default.

Preparation arrays enumerate only registered ordinary-role files. Every shard
records producer source hashes; reuse requires the same semantic source bytes.
Exact pushed source and a matching foundation lock are required. A genuine
selected-site probe runs installed-Weaver forward/input-gradient/parameter-gradient
parity, bounded CE/oracle/KD checks, then one **nonscientific** full-profile
offline pass and a worst-capacity batch-256 GPU memory probe. Train/validation
cache ceilings divide 75% of allocated host RAM proportionally to each role's
conservative resident/worker/IPC bound; 25% is reserved for other runtime use.
This replaces the full-reservoir 53%/22% split, which starved validation when
training shrank to 500k but validation remained 1M. CPU/worker counts, actual cache sizes, GPU peak, and a conservative
walltime estimate are recorded. This is a resource estimate, not a throughput
guarantee. Every production GPU task verifies the same allocation, installed
Weaver Python source hashes and numerical-library versions. Preparation,
acceptance and production roots are separate and may not be inside raw data.

Production persists only one selected model-weight checkpoint per successful
fit, small reports, compact assignment arrays and probability shards. Failed
attempts get separate directories; no optimizer or rolling resume is saved.
Small/compact partial outputs remain recoverable rather than being deleted
automatically. They must be counted in storage checks if failures accumulate.
The submitter requires free space for twice the planned bank/checkpoint payload
plus 1 GiB and an exact immutable dry run before accepting the explicit live
authorization phrase. No existing campaign is modified or cancelled.

Same-source recovery validates checksums and preserves completed outputs. It
restarts only unfinished tasks from zero, after ALL old attempts are terminal;
live/unknown jobs in another recovery under the same root also block it.
There is no automatic cancellation or source hot-patching. Code-changing
recoveries require a separately designed lineage transition, not editing a
queued source snapshot. A campaign-wide submission claim prevents concurrent
submitters. A pre-sbatch intent without a recorded receipt fails closed and
requires exact-job reconciliation; it is never blindly resubmitted. Final-test
evaluation/finalist publication is deliberately not provided by this validation
campaign. Do not treat a preview or a unit-test profile as live-run evidence.

Optional installation extra: `.[delphes]` declares Torch, Weaver, scikit-learn
and threadpoolctl alongside the existing ROOT-reader dependencies. The existing
RC environment must be checked before making package changes in a shared
environment used by other running campaigns.

### SPORC execution and readiness-only submission

The selected EXECUTION_SITE/v1 is `sporc_a100`: cluster `sporc`, partition
`tier3`, account `reu-aisocial`, QoS `qos_tier3`, `gpu:a100:1`, one node/task,
and `/home/ryreu/miniconda3/envs/atlas_kd_sporc` on x86-64. The explicit
`tigris_gh200` alternative retains its own ARM environment. No automatic site
detection, environment fallback or modification of old Slurm defaults occurs.
The worker checks real Slurm metadata, BF16 support and one visible registered
GPU. Production additionally requires the measured GPU identity, CPU/RAM
allocation and installed environment. Environment v2 includes Python, host
architecture, Weaver distribution/source bytes, Torch/CUDA/cuDNN, NumPy, Uproot,
Awkward, SciPy, sklearn and threadpoolctl versions. These contracts do not
permit substituting an A100 run for an old GH200 acceptance artifact.

READINESS_SPEC/v1 is independent from CAMPAIGN_SPEC/v3. The new CLI
`prepare_jetclass2_delphes_sporc.py` creates its canonical dry ledger before
allowing live submission with the readiness-only authorization phrase. Four
commands form sample -> assignment array -> lock -> profile; array `afterok`
requires every registered element to succeed. Only train/validation membership
is enumerated. Default array concurrency is 16. CPU jobs request 1 CPU/4 GiB,
30 minutes for sample/lock and 120 minutes per assignment. The A100 profile
starts with 8 CPUs/8 workers/80 GiB/240 minutes. These requests are unmeasured
starting envelopes, not production certification. No science auto-launch is
present. The full campaign remains a separate 59-job dry-run/authorization step.

The profile times full U000/U050/D050 train+validation cache construction,
one offline training+validation pass, and reducer inference. Fit walltimes use
1.75 times the worst cache time plus 100 passes; reducer walltimes use twice
the worst cache time plus inference. Minimum requests are respectively 60 and
30 minutes; the default explicit planning ceiling is 2880 minutes. Excessive
estimates fail without a runtime profile and leave RESOURCE_MEASUREMENTS/v1
diagnostics. They are never silently truncated to a desired time. The
capacity-240/batch-256 backward must stay within 85% of measured GPU memory;
there is no hidden batch reduction. Probe views, predictions and checkpoints
are not durable (only the tiny bounded-acceptance probability bank is retained).

Slurm uses explicit `--no-requeue`: restarts follow the recorded restart-zero
workflow, not scheduler restarts that collide with immutable output paths.
Both readiness and scientific submission require exact dry ledgers and durable
pre-sbatch intent/receipt journals. Lost acknowledgements require reconciliation;
no broad cancellation, holding, reprioritization or changes to other projects
are part of either submitter. Initial raw-file verification reads checksums,
not final-test particle arrays or predictions.

### Debug profiling exception (no preparation rerun)

EXECUTION_SITE/v1 adds `sporc_a100_debug` without changing either existing site
object. It differs from `sporc_a100` only in name and partition (`debug`);
the same account, QoS, A100, Conda prefix and conservative resource envelope
apply. Debug is a measurement site only. RUNTIME_PROFILE/v2 cannot carry this
transfer. RUNTIME_PROFILE/v3 requires the exact debug measurement site, tier3
execution site and `sporc_debug_to_tier3_same_a100_environment_resources_v1`
policy. No arbitrary site transfer or debug scientific execution is accepted.
The existing CAMPAIGN_SPEC/v3 can embed either validated runtime version;
new profile hashes distinguish these executions. Its science plan always uses
the runtime's production site. GPU identity, host environment, CPU/RAM, workers
and all ordinary gates are unchanged; cross-partition timing is only an estimate.

PROFILE_ATTEMPT_SPEC/v1 binds a canonical old readiness spec/hash, its exact
foundation/hash and completed lock/hash, new pushed source/worktree, disjoint
attempt/evidence paths, resources, measurement/production sites, zero scientific
fits and no preparation/existing-campaign mutation. Reuse authenticates array
checksums, coverage and the unchanged assignment semantic source bytes, then
compares the complete existing lock to those reports without republishing it.
Missing/stale/corrupt preparation fails before submission; old Git HEAD alone
does not force recomputation of semantically identical assignments.

The attempt has exactly one profile task, no array or scheduler dependencies,
its own dry ledger and guarded submission journal, and a separate authorization
phrase. No old job is cancelled or moved automatically. Outputs are small
specifications, logs, acceptance reports, the tiny acceptance bank and a runtime
profile; no dense views, training checkpoints or matching copies are written.
PROFILE_ATTEMPT_RESULT/v1 binds successful runtime evidence to the attempt,
original lock and actual job. Timeout recovery uses a fresh directory, while
the original foundation remains read-only and reusable.

## Metrics

Accuracy and unweighted macro eleven-class OVR AUC use the common validation
population. Each signal's QCD rejection uses `p_signal/(p_signal+p_QCD)`;
both-zero probability gives score zero. The 50% threshold is the descending
signal rank `ceil(N_signal/2)`, including ties. Zero empirical background
passing yields a null/censored rejection and a labelled one-event resolution,
not an infinite JSON value or invented finite measurement. Macro R50 is the
geometric mean over ten signals, null if any component is censored. Recovery
uses fresh HLT=0% and fresh U000=100%, in linear rejection space. Undefined or
zero-denominator recovery is null; scientifically poor scores never fail jobs.
