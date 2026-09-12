# JetClass2 Delphes Offline/HLT Dataset Migration Implementation Plan

Date: 2026-09-11. Execution-site update: 2026-09-12.

Status: migration and source-pinned production code implemented and locally
tested; remote execution is not yet certified. Inventory/splits, the reduced
reader, matching/views, compact preparation, RAM caches, metrics, CE/KD training,
checkpoint/bank publication, Slurm dry/live submission, monitoring and same-source
restart-zero recovery now exist. Full submission remains gated on new-data
Weaver validation on the selected execution site and measured resources. See the
[implementation contracts](../contracts/JETCLASS2_DELPHES.md) and section 15.
This document does not authorize job submission or certify publication rights.

User decision, 2026-09-11: proceed under the explicitly provisional label and
zero-uncertainty policies in section 3.1 while awaiting Luka's reply. The user
also accepts using fewer available fields. Producer confirmation is not a
prerequisite for implementing these declared policies or for later separately
authorized exploratory fits that pass the ordinary execution gates.

**Superseding execution decision, 2026-09-12:** use SPORC `tier3`, account
`reu-aisocial`, QoS `qos_tier3`, one A100 and the isolated x86-64 environment
`/home/ryreu/miniconda3/envs/atlas_kd_sporc`. The real A100 readiness gate
replaces the historical genuine-Tigris/GH200 gate for this new benchmark only;
it is not a waiver of installed-Weaver parity or a real production-worker
miniature. Old campaigns and `sbatch/common.sh` defaults are unchanged. The
user authorized local preparation up to this gate, not live job submission.
The initial population remains TRAIN_500K / 1M validation / 1M sealed test.

Luka has now confirmed that the numeric labels were not changed from upstream
JetClass2. His supplied `offline-and-hlt/FatJetMatching.h` at inspected reference
commit `69c58b243476412b7489ddd80fd7d1897ec27b8a` agrees with the existing table.
This records new supporting evidence without rewriting the immutable inventory,
split masks or original provisional policy hashes. It is not the exact production
commit/card confirmation. Charged zero-error interpretation remains unresolved;
the previously authorized finite value/error policy is unchanged.

## 1. Objective and comparison boundary

Move HLT-only classification and offline-supervised ladder experiments to Luka
Lambrecht's paired JetClass2 Delphes production. Preserve the research question:
can a sequence of increasingly HLT-like teacher/student views transfer useful
offline information into a deployable classifier that consumes only HLT inputs?

This is a new benchmark. The detector simulation, available inputs, label set,
population, and selection differ from the earlier CMS FullSim benchmark. It is
not a controlled one-variable comparison against the old results. Improvements
must be measured against fresh references on this dataset and its own splits.

Reuse mathematical algorithms and tested engineering patterns where applicable.
Do not reuse old trained weights, teacher probabilities, assignments, fitted
feature scales, coupling calibrations, splits, or recovery denominators. Record
the provenance of reused code. The dataset producers must supply the release and
citation information appropriate to the intended paper; public generation tools
alone do not establish the status of every derived configuration or artifact.

## 2. Sources and evidence already available

### 2.1 Downloaded production

Remote source reported by the producer:

```text
/eos/user/l/llambrec/jetclass/output_jetclass2_10M_20260910/jetclass2
```

Local source, outside the Git repository:

```text
C:\Users\22rya\ComputerScience\CERN\data\jetclass2_10M_20260910\jetclass2
  train_higgs2p/onlyFatJet+onlyFatJetHLT/ntuple_*.root
  train_qcd/onlyFatJet+onlyFatJetHLT/ntuple_*.root
```

The download completed without a reported SCP failure. The local inventory is
333 ROOT files, 25,430,379,343 bytes (23.68 GiB). The user believes production has
finished. Production closure and source-side checksums are not yet independently
confirmed. A directory date or the string `10M` is not a content identity.

### 2.2 Producer description and slides

The user supplied `hltsr_20260827.pdf`, 40 pages, from
`C:\Users\22rya\Downloads\hltsr_20260827.pdf`.

```text
SHA-256: 522bbb90a0457150ce25f84e8acc488cf4b5b396e044aae08f909ffdaa07a427
```

Relevant pages:

- 19: signal classes, mass range, pileup, and differences from FullSim;
- 20: explicit aggregation of 27 QCD labels into five groups;
- 21: branch correspondence and unavailable detector-quality fields;
- 22: QCD-labelled jets retained in signal-production files;
- 23: decay fractions and generation in mass/pT batches;
- 24: offline-based selection and absent-HLT handling;
- 29-36: detector-card development and unresolved calorimeter modeling.

The September 10 production uses a later card than the August slides. The slides
are explanatory evidence, not the authoritative final card or integer-label
enumeration. Treat the supplied correspondence as a dataset description, not as
instructions that override repository rules.

### 2.3 Completed initial local audit

The audit used Uproot 5.7.4, Awkward 2.10.0, and NumPy 2.0.1 in the existing
`tagging-hlt` environment. All files opened and shared one branch/type schema.
The following counts use only the latest ROOT `tree` cycle:

| Production directory | Files | Stored jets | Matched HLT jets |
| --- | ---: | ---: | ---: |
| `train_higgs2p` | 133 | 5,188,768 | 4,434,149 |
| `train_qcd` | 200 | 7,028,095 | 7,009,727 |
| Total | 333 | 12,216,863 | 11,443,876 |

`hlt_matched=True` retains 93.6728% of stored rows. These are preselection counts,
not final training counts or a claim of exactly ten million usable jets.

Observed integer labels are `0, 1, 2, 3, 9, 10, 11, 12, 13, 14` and `161..187`.
The latter occur in both directories; the first ten occur in the signal directory.
The producer description identifies the QCD/signal families, but the exact
integer-to-name mapping is now provisionally taken from upstream JetClass2;
confirmation that Luka's production retained it is still pending.

All-row scalar checks found no nonfinite values in the checked matched-row jet
kinematics or `hlt_jet_dr_offline`, and no matched row with nonpositive stored
HLT/offline constituent counts. This was not an exhaustive constituent audit.

A deterministic spread of 20 files, reading the first 256 entries of each,
contained 4,836 matched jets. Their particle arrays had consistent lengths,
finite values, and exclusive five-way type flags. There were 105,957 HLT and
186,210 offline particles. Zero `d0err`/`dzerr` values occur in both views; their
meaning and applicability are not established by this sample alone.

No event, run, luminosity-block, generator-seed, or weight branch was found in
the common schema. Most files have two saved versions of `tree`: the observed
cycle patterns were 270 files with `(3,2)`, 33 with `(2,1)`, 29 with `(4,3)`,
and one with `(1)`. Enumerating every tree cycle would count overlapping entries.

There are 3,614,983 matched jets with fewer than 16 HLT constituents. Inspection
of `scouting/inputs.py::deployment_length` confirms that the old value 16 is a
padding floor. The old `scouting/labels.py::baseline_mask` requires a positive
count, not at least 16 particles. Do not confuse padding with sample selection.

The exploratory audit ran before a new split existed and produced no model
predictions. Its console observations need reproduction in an immutable audit
artifact; they are not a substitute for a dataset lock or remote checksum check.

## 3. What is understood, and what remains unresolved

There is no unidentified algorithm needed to implement the loader, masked
features, exact assignment, teacher banks, or ladder DAG. There are producer
facts we cannot infer safely and scientific choices the migration must expose.

| ID | Question | How to resolve it | What it blocks |
| --- | --- | --- | --- |
| E1 | Exact producer commit, local patches, cards, generation settings, and completed file set | Obtain production configuration and inventory from Luka; bind hashes | Producer-verified provenance, not local-snapshot exploratory execution |
| E2 | Whether production retained the upstream integer label enumeration | Adopt the explicit upstream table under section 3.1; request confirmation | Confirmed interpretation, not provisional labelled fits |
| E3 | Event relationships between files and generation jobs; retries or duplicated batches | Use whole-file groups provisionally; obtain job/event grouping or event-ID sidecar | An unconditional event-independence claim, not file-disjoint exploratory execution |
| E4 | Momentum/length units, impact-parameter signs, zero-error semantics, p4/PUPPI conventions, and stored angular definitions | Producer source plus train-only numerical checks; use section 3.1's declared zero-error policy | Fully verified feature semantics; explicit finite input rules still required before fits |
| E5 | Offline jet cuts, HLT jet association rule, thresholds, possible reuse of one HLT jet for several offline jets, and event weights | Producer code/config; confirm that absent weights imply an intended unweighted sample | Selection and sample interpretation |
| E6 | Release/citation status of production and final tuned card | Producer/advisor confirmation and recorded citations | Public-release claims, not local parser development |
| D1 | Include QCD-labelled rows from signal generation? | Choose a named source-selection policy before fits | Selection lock |
| D2 | Exact feature list, units, normalization, masking, and particle capacity | Implement the proposed common feature family; complete train-only audits and freeze | Input lock and model parity |
| D3 | Split fractions, grouping, and class/source balance | Freeze deterministic assignment with confirmed grouping or section 3.1's provisional file-group policy | Split lock |
| D4 | Initial ladder support variant | Recommended: original non-persistent pure-offline U000 path; record explicit variant | Graph/view contract |

E1-E6 do not prevent writing the inventory reader, raw schema validator,
synthetic fixtures, and configurable adapters. They must not be replaced by
silent guesses when publishing executable scientific artifacts. The recommended
D1-D4 values below are planning defaults, not previously approved immutable
campaign settings.

### 3.1 Authorized provisional assumptions

The user explicitly chose to proceed while producer confirmation is pending.
This supersedes the earlier requirement to wait for E2 or a zero-error reply
before implementing a labelled adapter. It does not claim those replies arrived.

**A1: unchanged upstream label enumeration.** Use the explicit standard
[JetClass2 label definition](https://github.com/jet-universe/jetclass2_generation/blob/main/delphes_analyzers/FatJetMatching.h#L690),
not the rank of a label among the observed IDs. Freeze the reference source
revision/content digest and the complete effective map in the implementation's
label artifact; never fetch a mutable branch during a training job.

| Raw ID | Signal name |
| ---: | --- |
| 0 | X_bb |
| 1 | X_cc |
| 2 | X_ss |
| 3 | X_qq |
| 9 | X_gg |
| 10 | X_ee |
| 11 | X_mm |
| 12 | X_tauhtaue |
| 13 | X_tauhtaum |
| 14 | X_tauhtauh |

Raw IDs 161 through 187 map to the 27 QCD names in their upstream order and
collapse to the classifier's QCD output. Original raw IDs remain in audit
metadata. Codes outside the declared supported set fail closed rather than
being silently mapped to QCD or shifted into a contiguous signal range.

**A2: zero uncertainty is unavailable for significance calculations.** For
each impact-parameter component independently, a finite error equal to zero
is an unavailable/not-applicable uncertainty sentinel, including on a charged
particle. It is not infinite precision. Preserve finite raw values and zero
errors in the raw reader and proposed value/error feature representation;
zero uncertainty alone does not prove the associated displacement is zero or
justify dropping a particle. Carry uncertainty applicability in construction
and audit metadata. Do not divide by zero or replace the denominator with a
small epsilon. If a later declared input contract includes significance, its
invalid cases require an explicit finite fill and validity indicator. Negative
errors and nonfinite required data remain validation failures, not newly
authorized sentinels. Tests must include charged particles with zero errors.

The bounded follow-up check on `train_higgs2p/.../ntuple_0.root` found 224
matched jets among the first 256 entries: 2,478/5,091 HLT constituents had zero
d0 error, all neutral in that sample. Offline zero-error cases included one
non-neutral constituent. This supports careful applicability handling, not an
assertion that every zero in the full production has a confirmed explanation.

**A3: whole-file split groups as the working event-containment assumption.**
Assign a file wholly to one role, check duplicate file content, and report
file disjointness as established while event independence across files remains
assumed. Do not switch to arbitrary row splitting or claim duplicate-content
checks establish independence of differently serialized events.

**A4: reduced available-feature input.** The user accepts fewer fields. Missing
detector-quality information need not be fabricated or block the migration.
Use the same explicit available-feature family for fresh references and ladder
nodes; final transforms/capacity still need testing and freezing. This does
not assert that the numerical size of KD gains will be unchanged.

Bind the effective policies and evidence status (`provisional`, not
`producer_confirmed`) to every derived scientific artifact. Preserve the
downloaded raw snapshot and its hashes. The exact producer cards/revision may
be added as separate evidence later without rewriting old immutable artifacts.
If Luka confirms the assumptions, record confirmation against the existing
policy hashes. If his answer changes labels, feature interpretation, or event
grouping, create a new affected contract/split and rebuild dependent artifacts;
never silently relabel old metrics or describe a corrective rerun as the same
experiment. Existing unrelated campaigns remain untouched. Source pinning,
tests, resource acceptance, dry runs, and explicit live-submission authorization
are still required; this assumption decision does not itself submit jobs.

A compact producer request is:

> Please share the code revision/local patches and exact offline/HLT cards used
> for the September 10 production, including the label enum, units and sentinel
> conventions, jet cuts/matching, and any event-weight semantics. Does each
> ntuple correspond to an independent generated-event batch, with all jets from
> an event confined to that file? Can batches span files or be duplicated by
> retries? Please also confirm production is closed and provide the intended
> release/citation information.

## 4. Dataset identity, inventory, and reader

Create a new `JETCLASS2_DELPHES_* /v1` family (without spaces in actual contract
names). Never identify this dataset as the older synthetic `fixed_hlt_v3`
profile or CMS Scouting schema merely because some fields have similar names.

The source manifest records every normalized relative file path, byte size,
SHA-256, ROOT logical tree name, selected cycle number, entry count, branch/type
schema digest, generation group, and producer-config parent when available.
Unconfirmed producer configuration is explicitly marked pending; do not invent
a parent hash or mislabel the local byte snapshot as producer-confirmed. Compare local
hashes with an authoritative source inventory when available. If it is absent,
state that the manifest authenticates the local snapshot only.

Use bounded reads of the latest `tree`, recording its selected cycle. ROOT
cycles are previous versions, not extra independent partitions. Zero-length,
unreadable, truncated, or schema-incompatible files fail inventory validation.
No skip-on-error production behavior is permitted.

Canonical row identity should bind the frozen dataset digest, relative file
identity, selected tree/cycle, and entry index. Store labels and label-map hash
separately so relabelling cannot evade an overlap check. Include the directory
in file identity: `ntuple_0.root` exists under both production sources. Relocating
the byte-identical dataset to Tigris must preserve scientific row identity.

Read raw files in place. Do not copy ROOT content into Git, checkpoint roots,
or a second persistent dense particle-cache format. Newly appearing or changed
files produce a new dataset identity, not an in-place mutation of a frozen run.

## 5. Labels and selections

### 5.1 Proposed class order

Keep one combined QCD class and ten supported signals:

```text
QCD, Xbb, Xcc, Xss, Xqq, Xgg, Xee, Xmm,
Xtauhtaue, Xtauhtaum, Xtauhtauh
```

This is an 11-output classifier. Map these names through section 3.1's explicit
upstream integer table, provisionally pending E2; do not assign names by sorting
observed IDs. Unknown IDs fail closed.
Retain raw labels and production-source metadata for audits, outside model inputs.
Do not synthesize absent Xbs/Xbc/Xcs/charged-Xud classes or reuse the old head.
`Xqq` here covers the supplied neutral component described in the slides.

Preserve the five-group QCD aggregation as diagnostic metadata, with the page-20
name groups exactly as follows; provisional numeric membership uses upstream
order, with ranges 161-169, 170-178, 179-181, 182-184, and 185-187 respectively:

| Group | Source label names |
| --- | --- |
| QCD_bb | QCD_bbccss, QCD_bbccs, QCD_bbcc, QCD_bbcss, QCD_bbcs, QCD_bbc, QCD_bbss, QCD_bbs, QCD_bb |
| QCD_b | QCD_bccss, QCD_bccs, QCD_bcc, QCD_bcss, QCD_bcs, QCD_bc, QCD_bss, QCD_bs, QCD_b |
| QCD_cc | QCD_ccss, QCD_ccs, QCD_cc |
| QCD_c | QCD_css, QCD_cs, QCD_c |
| QCD_light | QCD_ss, QCD_s, QCD_light |

### 5.2 Proposed first selection

- Require `hlt_matched=True` before consuming HLT dummy branches as inputs.
- Require supported labels, internally consistent particle arrays, and valid
  nonempty physical endpoint views. Invalid matched rows are reported and stop
  production; they are not silently discarded as if they were ordinary cuts.
- Retain the producer's offline-selected population initially; do not invent
  old tight-ID, jet-rank, HLT pT/mass, or multiplicity cuts from missing branches.
  Any additional kinematic cut must be explicit and shared by every comparison.
- Recommended D1: initially use supported signal labels from signal production
  and QCD labels from QCD production. Keep QCD-labelled signal-production rows
  on disk and in diagnostics, but exclude them from this first benchmark to
  align with the previous production-source convention.
- An alternative `all_sources_qcd_by_label` policy includes those rows as QCD.
  It is valid but changes the background mixture and needs a separate selection
  identity. Never infer an individual row's label from its filename.

Using the observed label families, the recommended policy would retain 1,987,133
matched signal rows plus 7,009,727 matched QCD-production rows: 8,996,860 rows
before further cuts. The all-sources policy would add 2,447,016 matched
QCD-labelled signal-production rows, yielding 11,443,876. Revalidate these
provisional calculations when E2 is confirmed or corrected.

Report class/source counts and HLT match efficiency by class, pT, eta, and
multiplicity. Classification on matched jets is conditional performance; it
does not measure the efficiency of recovering an absent HLT jet. Do not present
the label imbalance as a bug or infer expected training yields from `10M`.

## 6. Split construction and evaluation access

**Superseding user decision, 2026-09-11:** the initial campaign is TRAIN_500K,
not full population. Register nested TRAIN_500K, TRAIN_1M, TRAIN_1P5M and
TRAIN_2M populations (500,000 / 1,000,000 / 1,500,000 / 2,000,000 training
jets), all with the SAME 1,000,000 validation and SAME 1,000,000 final-test
jets. Each size trains fresh HLT/offline references and all ladder nodes on
its own training subset. A full-data teacher is not part of this comparison.
Keep the optimization protocol unchanged; record passes, updates and runtime
because equal passes across sizes do not imply equal compute. Recovery is
relative to each size's fresh references; compare absolute metrics as well.

The existing seed-20260910 whole-file SPLITS/v1 manifest remains the immutable
outer reservoir partition. Subsample only WITHIN those roles; unused validation
or test rows never enter training. The uploaded ROOT snapshot is unchanged.
The explicit registry is independent of model campaigns and contains six
membership artifacts: four nested training selections and two shared evaluation
selections. Each stores per-file little-bit-order, base64-encoded entry masks,
exact per-class/total counts and source parents. Mask bit i selects ROOT entry i
from the authenticated latest tree cycle. This is explicit row membership, not
a seed-only promise, and does not store features or predictions.

Selection uses a deterministic SHA256 priority per inventory/file/tree/entry,
role and selection seed (20260911). Within each class, smaller training profiles
are prefixes of the same ordering. Allocate the first subset by reserving one
jet per class then Hamilton largest remainders over remaining class capacities;
allocate later increments by largest remainders over remaining capacities.
Integer arithmetic and class-index tie breaks fix exact counts and monotonic
quotas. Evaluation uses the same class-proportional rule in its own reservoirs.
Masks are iterated in canonical file/entry order; training shuffles remain a
separate recipe. No numerical-library RNG or physical source root sets identity.

Registry generation may read ONLY label/matched scalar metadata in test files
to establish membership; it cannot decode test particle arrays, construct test
matches, train or evaluate a model. Ordinary readers continue to reject test.
Publish new SPLIT_REGISTRY/v1, ROLE_MEMBERSHIP/v1, SPLIT_DESIGN/v1 and
SPLIT_PROFILE/v1 families. Subset foundations use FOUNDATION_SPEC/v2; never
make old full-population artifacts masquerade as subset artifacts. Production
requires an explicit subset profile; old full-data foundations remain readable
for compatibility but cannot silently be submitted by the new production path.
Local split generation/testing is complete. The September 12 execution update
authorizes preparing the SPORC deployment tooling; actual remote matching and
genuine Weaver/A100 checks still require the user's readiness-only submission.

The outer roles are 60% train, 20% validation, and 20% sealed final test
by approximate retained-row mass, assigning indivisible generation groups.
Exact counts need not equal those fractions after grouping. Freeze the split
seed and grouping/stratification algorithm before inspecting model results.

A relative file/entry identity proves row disjointness, not event independence.
If E3 establishes that each file is an independent generation batch, whole-file
groups are sufficient for event containment. If a batch spans files, those files
must share one group. If producer sidecars provide event identities, bind them
and enforce event disjointness as well. Retries/duplicate batches must be detected
before splitting; a renamed byte-identical file is not new independent data.

Until these guarantees are available, section 3.1 permits a frozen provisional
whole-file split whose evidence states the event-containment assumption.
Kinematic fingerprints can flag duplicates but cannot prove unique generator-
event identity. Do not describe an arbitrary row-hash split as safe. A later
contradiction requires a new grouping/split and reassessment of dependent fits.

Where grouping allows, balance allocation by production source and class counts;
audit generated mass/pT coverage and observed kinematics. The slides describe
generation batches by mass/pT, so splitting contiguous filename ranges without
checking coverage is unsuitable. Missing required classes in a role require
revising the split design before fits, not changing it after seeing scores.

All baselines and ladders use identical selected row identities, labels, and
weights within each role. Proposed initial fits use natural class proportions
and unweighted CE; oversampling or weighting would be a separate declared recipe.
Normalization and coupling calibration use train only. Validation selects
checkpoints/early stopping and is labelled accordingly in preliminary reports;
it is not an untouched final performance estimate. Later hyperparameter screens
must declare their selection and reporting roles explicitly.

Initial all-file schema/integrity checks are recorded as pre-split QA. After
freezing roles, ordinary model execution cannot access final test. Final-test
predictions require the repository's finalist and execution locks.

## 7. Raw branch adapter and proposed model inputs

### 7.1 Source correspondence

| Quantity | Offline | HLT | Rule |
| --- | --- | --- | --- |
| Four-vector | `part_px/py/pz/energy` | `hlt_part_px/py/pz/energy` | Preserve native precision in source audit; canonicalize explicitly |
| Charge | `part_charge` | `hlt_part_charge` | Integer/physical-domain validation |
| Five particle identities | `part_isElectron/isMuon/isPhoton/isChargedHadron/isNeutralHadron` | Same suffixes under `hlt_part_` | Validate exact flags and applicability |
| Impact parameters | `part_d0val/dzval` | `hlt_part_d0val/dzval` | Confirm units and sign conventions |
| Their uncertainties | `part_d0err/dzerr` | `hlt_part_d0err/dzerr` | Confirm zero/sentinel semantics; these are not significances |
| Stored relative angles | `part_deta/dphi` | `hlt_part_deta/dphi` | Audit against producer axis convention; never mix axes across views |
| Constituent counts | `jet_nparticles` | `hlt_jet_nparticles` | Must agree with every jagged constituent branch |
| Labels | `jet_label` | Shared paired-row label | Supervision only |
| Jet association | `hlt_matched`, `hlt_jet_dr_offline` | Shared paired-row metadata | Selection/audit only; forbidden model inputs |

Do not fabricate missing `quality`, `normchi2`, `lostInnerHits`, or PF/lost-track
identities. Some old b-tag kinematics may be derivable, but equivalence requires
the exact formula; do not fabricate those by naming a proxy identically.

### 7.2 Proposed common feature family

The first implementation should start from the available JetClass-style
17-feature family in `data/part_inputs.py`:

```text
log pT, log energy, log relative pT, log relative energy, axis delta-R,
charge, five particle-type flags,
d0 value, d0 uncertainty, dz value, dz uncertainty, signed deta, wrapped dphi
```

This is a proposal for the new input contract, not authorization to import the
old module unchanged. That module also binds a ten-class schema. The Scouting
model and view helpers instead hardcode 21 features, 15 classes, and several
200-token assumptions. Both require explicit adaptation or a new implementation.

Specify each feature's units, axis definition, exact transform, clipping,
missing-value treatment, order, dtype, and any fitted train-only parameters.
Use one shared frozen transform for HLT, offline, and intermediate views.
Preserve documented analytic transforms where suitable; fit any required
calibration only on this dataset's train role. Do not import FullSim-fitted
numbers without distinguishing analytic choices from fitted artifacts.

The proposed 17-channel family retains errors directly. If significance is
added in a later explicit feature contract, derive value/error only where
validity permits, with a finite defined representation otherwise. Never divide
by zero or silently erase nonfinite values. Section 3.1 authorizes treating zero
errors as unavailable uncertainty pending confirmation of their physical meaning;
this is explicit behavior, not a claim that the producer has confirmed it.
If meaningful missing states require explicit model-visible masks, freeze their
channels and the resulting new feature count before the first fit.

Construct kinematic derived features from the supplied view's four-vectors and
its own axis. Confirm the treatment of zero-pT/zero-energy particles, massive
constituents, PUPPI weights, phi wrapping, and the relationship between stored
jet p4 and constituent sums. HLT endpoint inference must not consult offline
jet axes, offline normalization, matching metadata, or alternate views.

Particle ordering and capacity are scientific definitions. Recommended starting
policy: preserve all valid endpoint constituents and native source indices,
with deterministic intermediate carrier ordering. Audit train maxima and choose
an explicit adequate capacity; fail clearly on overflow rather than silently
truncating. Any top-k alternative is a declared change with lost-count/energy
diagnostics. Padded slots are zero and masked, and low-multiplicity jets remain
valid. The old 128/200 limits must not be assumed appropriate here.

Build raw intermediate particle states first, then apply the common feature
transform. Establish exact equality between standalone HLT inference and D000,
and between standalone offline inference and the proposed pure-offline U000.

## 8. Matching and U/D semantics

Recommended first variant: full-cardinality lexicographic bottleneck matching
with the original non-persistent `replace_source_with_target_v1` support meaning.
Persistent-HLT, MT20, salience matching, ensembles, and fusion are separate
experiments; their presence in the repository does not select them for this
migration. Confirm D4 explicitly in the final campaign specification.

For each paired jet with valid raw counts `n_h` and `n_o`, assign exactly
`min(n_h,n_o)` one-to-one pairs. Preserve the existing exact quantized delta-R
objective and deterministic tie hierarchy if importing that solver. Report
forced-pair tail quality separately from integrity. Large delta-R and low model
performance are scientific results, not reasons to skip data or downstream fits.

Match global particle directions derived from p4; independently centered HLT and
offline `deta/dphi` values cannot be subtracted as though they shared an axis.
No truth-correspondence probability is inferred from full cardinality.

Reimplement the schema-dependent endpoint preparation and field interpolation
under new contracts. In the proposed path:

- U000 is pure offline input, trained fresh on the new paired-jet population.
- U changes support while common paired content stays at the offline endpoint.
  Offline-only tails disappear; if `n_h > n_o`, unmatched HLT particles must
  enter. U is therefore not universally a particle-removal-only sequence.
- U100 has HLT slot support, offline content for paired slots, and native HLT
  content for unavoidable unmatched HLT slots. Slot support does not imply HLT
  particle kinematics at U100.
- D retains HLT support and moves paired fields from offline to HLT. The
  retained-offline coordinate is `alpha=1-f`; D000 has `f=1` and exact HLT.
- Both count-imbalance directions, empty-side handling, atomically coupled
  identity/charge/measurement applicability, and endpoint ordering have tests.

Use new train-derived coupling/switch calibration where required. Synthetic
reference tests must establish the adapted interpolation and exact endpoints;
old 21-channel parity is not evidence for new features. Assignment and source
indices are used only by construction/joins, never by the classifier.

## 9. Fresh references and proposed ladder execution

The proposed first science sequence is:

1. Fresh CE-only HLT ParT reference.
2. Fresh CE-only pure-offline U000 ParT reference on the same rows/classes.
3. Direct U000-to-D000 LOGIT KD control.
4. Coarse, dense, and ultradense immediate-parent single-model branches.

The two references can train in parallel once their gates pass. All four KD
branch heads can run after the new U000 probability bank is ready. The direct
control is the DIRECT branch, not an extra duplicated pilot. There is no score
threshold requiring the direct control to win before another registered branch
may run.

Proposed inherited paths:

```text
DIRECT:     U000 -> D000
COARSE:     U000 -> U050 -> U100 -> D066 -> D033 -> D000
DENSE:      U000 -> U033 -> U066 -> U100 -> D080 -> D060 -> D040 -> D020 -> D000
ULTRADENSE: U000 -> U020 -> U040 -> U060 -> U080 -> U100
                 -> D090 -> D080 -> D070 -> D060 -> D050
                 -> D040 -> D030 -> D020 -> D010 -> D000
```

The graph has 29 downstream fits. With fresh offline and HLT references, there
are 31 fits. The 25 nonterminal downstream teachers plus U000 need 26 teacher
publications. These counts are a proposed full migration target, not a queued
campaign. Do not import old checkpoints to recover the previous 29-fresh-fit
accounting.

The optimization reference is the single-GPU four-spine protocol (now A100):

- one process/GPU, global batch 256, fresh initialization at each edge;
- CE-only references; constant C25/P75 LOGIT KD with T=2 on KD edges;
- passes 1-3 warmup to 3e-4, 4-45 hold, 46-60 cosine decay to 1.5e-5,
  61-100 fixed-floor refinement;
- minimum 60 passes, maximum 100, patience 15, reset threshold AUC gain >5e-5;
- patience accumulates before pass 60; exact best-checkpoint selection is
  independent of the patience threshold;
- AUC-first checkpoint selection, then lower CE, higher logR50, earlier update;
- paired terminal initialization/sampler seeds across HLT CE and direct/ladder
  D000 comparisons; explicit coordinate seed aliases for the other rungs;
- no rolling optimizer resumes; failed fits restart from zero under exact
  source/ledger recovery.

Freeze optimizer parameters, loss normalization/T-squared convention, numeric
precision, partial final-batch behavior, seed derivation, and every schedule
boundary in a new recipe before implementation handoff. The schedule is a
reasonable reference from prior work, not known to be optimal on Delphes.
Class/feature changes require fresh installed-Weaver parity and real execution
acceptance. Do not inherit the older campaign's evidence waiver.

## 10. Metrics and claim boundaries

Compute accuracy, CE, macro OVR AUC over the frozen 11 classes, and per-signal
QCD rejection using `p_signal/(p_signal+p_QCD)` on the identical evaluation rows.
For each signal-versus-QCD curve, restrict those rows to that signal and QCD;
other signal classes are not part of its background denominator.
Freeze tied-threshold rules and zero-QCD-pass handling. Report achieved signal
efficiency and passing/total background counts; finite display ceilings are
not measurements of unlimited rejection.

Macro R50 is the exponential of the mean log per-signal QCD rejection, over
the explicit ten-signal set. Label it as a geometric mean, not an arithmetic
average. Any convention for zero-pass cases and undefined metrics must be
versioned and shared across references and candidates.

For a metric X with defined nonzero reference gap:

```text
recovery(X) = 100 * (X_model - X_HLT_CE) / (X_U000_CE - X_HLT_CE)
```

Use linear rejection values for R50 recovery. Report raw metrics and denominators;
do not clamp negative or >100% recovery. If the gap is zero, mark recovery
undefined; if small or negative, display the gap and avoid interpreting recovery
as a stable measure of recovered information. U000 is a reference, not a
mathematical upper bound. Never mix old FullSim and new Delphes denominators.

Report per-class Xbb, Xcc, Xqq and all other supported signals, class counts,
selected/completed passes, runtime, and upstream teacher identity. Preserve
actual node IDs so teacher-name substrings cannot mislabel D000-from-U100 as
a U100 student. Event/group dependence must also be reflected in uncertainty
estimation; one trained seed does not establish seed-to-seed significance.

## 11. RAM, disk, and throughput

Keep particle views, representations, optimizer state, and dense matching
matrices in RAM/device memory only. Persist raw ROOT files once, compact sharded
assignments, selected checkpoints, compact teacher probabilities, reports,
manifests, and bounded logs. No RSET/RREL target banks are part of this LOGIT
migration. No copied legacy checkpoint tree is a dependency.

Plan memory using actual retained row counts, feature count, and capacity. For
example, features plus p4 alone require `N * capacity * (F+4) * 4` bytes in
FP32, before masks, source arrays, workers, temporary copies, or model state.
For the historical full reservoir's 7,186,283 train+validation rows, 17 features and capacity
240 imply roughly 135 GiB for a fully padded feature+p4 representation alone.
The implementation packs real tokens, but conservatively budgets worst-capacity
storage plus worker copies before preparation. Historical 320-GiB requests are
not automatically certified for this dataset.

A compact probability row with 11 FP32 probabilities and a SHA-256 identity
costs 76 bytes before labels/metadata. One bank over the frozen train+validation
reservoir population is about 0.51 GiB; 26 teacher banks alone were about 13.2 GiB.
The selected TRAIN_500K profile instead has 1.5M ordinary rows (about 2.76 GiB
for 26 banks); TRAIN_2M has 3M ordinary rows (about 5.52 GiB). These are array
payload estimates, not acceptance evidence. Include
assignments, reports, models, temporary publication overlap, and a free-space
reserve in the final storage budget. Separate raw-source bytes from generated
campaign-output bytes.

Prefer chunked multi-process CPU preparation and bounded workers with one
numerical thread each. Avoid multiplying the full dataset by worker count.
Measure preprocessing, inference, train, and validation separately; print
progress during long phases. Persistent views, hidden disk-backed spills,
unbounded queues, and automatic changes to batch size are not speed fixes.

The raw snapshot was uploaded and checksum-verified at
`/home/ryreu/atlas/datasets/jetclass2_10M_20260910/jetclass2` and is visible from
SPORC; do not reupload it or use old `PracticeTagging` data. Transfer only the
compact registry/profile metadata. Use the registered SPORC execution site,
`PYTHONNOUSERSITE=1`, the isolated Conda library path and absolute helpers.
Workers discard inherited Python/loader paths from other projects. Numerical
libraries use one thread each; bounded preprocessing uses actual processes.

Initial readiness requests, explicitly **unmeasured**: sample and foundation
lock jobs use one CPU, 4 GiB and 30 minutes; assignment elements use one CPU,
4 GiB and 120 minutes, with at most 16 elements in flight. The real A100
profile job requests 8 CPUs, 8 workers, 80 GiB and 240 minutes. These are
finite starting envelopes, not assertions about observed throughput. The 80-GiB
request exceeds the native 500k-profile conservative bound (63.01 GiB including
the 25% reserve) and allows four such requests within the reported 340000-MiB
A100-node envelope, unlike four 96-GiB requests. Failures
or timeouts do not authorize changing batch size, data membership or the
scientific recipe. Examine exact logs before a source-pinned retry.

For train and validation, compute the conservative cache bounds already used
by `prepare_cache`, including worker/IPC copies. Divide 75% of allocated host
RAM in proportion to those two bounds and reserve 25% for other runtime use.
The old fixed 53%/22% split is NOT valid for the new 500k/1M ratio. Reject an
insufficient envelope before any cache builds. Both caches remain RAM-only.

The genuine profile runs installed-Weaver parity, bounded HLT CE/offline CE/KD,
one full-selected-population offline training+validation pass, worst-capacity
batch-256 GPU backward, real train/validation inference, and full train/validation
U050 and D050 cache construction. No intermediate views or probe checkpoints
are persisted. Let C be the maximum measured U000/U050/D050 cache construction
time, P the one-pass training+validation time and I the reducer inference time.
Production fit walltime is ceil(1.75*(C+100*P)/60), at least 60 minutes;
reducer walltime is ceil(2*(C+I)/60), at least 30 minutes. These are estimates,
not guarantees, and use a default explicit maximum envelope of 2880 minutes.
Never clamp an excessive estimate down: retain diagnostic measurements and
fail without an eligible runtime profile. A failed real GPU-memory probe also
blocks production without silently reducing the batch or token capacity.

## 12. Implementation and artifact map

Recommended isolated package namespace and thin CLI prefix:

```text
src/hlt_classification/jetclass2_delphes/
  contracts.py       # content/parent/role validators
  inventory.py       # ROOT versions, file hashes, producer snapshot
  schema.py          # raw branch types, applicability, labels
  reader.py          # bounded paired-row reads and canonical identities
  selection.py       # explicit class/source/matched policy
  splits.py          # generation groups, deterministic role assignment
  split_registry.py  # explicit nested entry masks and shared evaluation sets
  inputs.py          # shared feature transforms and deployment interface
  views.py           # endpoint preparation, U/D fields and support
  foundation.py      # assignment/calibration artifacts and locks
  reporting.py       # class-map-aware metrics and fresh recovery references
  campaign.py        # new graph/recipe bindings
  runner.py          # training/reduction integration

scripts/audit_jetclass2_delphes.py
scripts/create_jetclass2_delphes_split_registry.py
scripts/create_jetclass2_delphes_foundation.py
scripts/create_jetclass2_delphes_spine4_campaign.py
```

These core modules and thin CLIs now exist, with additional `cache.py`,
`model.py`, `banks.py`, `acceptance.py`, `provenance.py`, `production.py` and
`submission.py`. The preview CLI still emits only a scientific graph.
`scripts/jetclass2_delphes_production.py` separately provides `prepare`,
`profile`, `create`, `run`, `submit`, `monitor`, `recover`, and `results` modes.
Two thin absolute-path Slurm workers cover preparation/profiling and campaign
execution. The full plan has 59 jobs: 31 fits, 26 banks, aggregate and completion.
Kernel/acceptance entry points cannot substitute for the production gates.

| Existing reusable surface | Reuse intent | Adaptation boundary |
| --- | --- | --- |
| `data/cache_contracts.py` | Hashing and atomic artifact primitives | New dataset/schema/role parents |
| `data/part_inputs.py` | Reference 17-feature mathematics | Old ten-class assumptions and raw-field contracts cannot be inherited |
| `models/scouting_particle_transformer.py` | Weaver factory/parity patterns | Hardcoded 21/15 interface requires a separate new contract/model wrapper |
| `scouting/hcwdl_fullcard_bottleneck_matcher.py` | Exact assignment solver | New raw-particle preparation and new data binding |
| `scouting/hcwdl_homotopy.py`, `hcwdl_unified_balanced.py` | Support and nested-switch semantics | Raw schema, field groups, calibrations, capacity, and exact endpoints |
| `scouting/view_cache.py` | Bounded RAM/process preparation patterns | New reader, identities, feature count, and measured resource profile |
| `scouting/hcwdl_mhpe_tri60_training.py` | Optimizer/logging/selection patterns | Audit every 15-class, source-recipe, and reporting assumption |
| `scouting/hcwdl_tri100_spine4_*` | Graph, source pinning, ledgers, recovery patterns | Fresh references and new contracts; no old campaign-parent imports |
| `evaluation/metrics.py` and campaign metric callers | Rank/threshold algorithms | Explicit 11-class/QCD semantics; no default ten-/15-class assumptions |

Baseline repository HEAD inspected for this plan:
`fd1ed1d01d54bf2ad4d42ffa6311432263a14770`. The workspace also contains unrelated
uncommitted work; that commit is not a claim that the whole workspace is clean.
Record actual donor files, donor commits, local patches, and semantic changes
in `docs/LEGACY_SOURCE_MAP.md`. The completed implementation's donor reuse and
explicit adaptations are recorded there.

Proposed contract families include source inventory/provenance, raw schema,
label map, selection, generation groups, split manifest, input transform,
assignment/endpoint/coupling locks, runtime acceptance, campaign/recipe/graph,
probability bank, training report, aggregate, and recovery. Every consumer
validates hashes, semantic versions, shape/dtype, ordered identities, and roles.

## 13. Implementation stages and acceptance

| Stage | Deliverable | Evidence before marking complete |
| --- | --- | --- |
| A | Durable source inventory and reproducible audit CLI | All 333 files/cycles/counts reproduced; corrupted-file and duplicate-path fixtures rejected; snapshot provenance labelled accurately |
| B | Schema, labels, selections, and generation groups | E2-E5 evidenced or covered by explicit authorized provisional policies; every observed ID covered; mixed-source, zero-error, and dummy-HLT cases tested |
| C | Frozen split and input contracts | Event/group disjointness, capacity/class coverage, units, sentinel handling, deterministic transforms, train-only fitting |
| D | Pairing and U/D foundation | Exact smaller-side coverage; rectangular/reference solver parity; exact raw/model endpoints; stable output across chunk/worker changes |
| E | Fresh reference and direct-KD execution | Real installed-Weaver FP32 parity plus genuine selected-site production-worker training/reducer miniature with measured RAM/disk/GPU use |
| F | Isolated four-spine execution plan | 31-fit/26-teacher-publication proposed graph reconciled with final spec, exact source, full dry run, sealed test, restart-zero recovery, explicit submission authorization |

Local tests must cover real synthetic ROOT files with multiple cycles, repeated
basenames in different directories, unmatched dummy HLT rows, label-map mismatch,
jagged-length mismatch, zero errors, type/charge applicability, phi wrapping,
overflow, source relocation, corrupt parents, missing classes, and group overlap.
Endpoint tests must perturb offline inputs while evaluating D000 to demonstrate
that deployable logits cannot depend on offline data. Verify U000 independently
of the assignment, and both directions of particle-count imbalance at U100.

Do not skip measured production acceptance merely because old FullSim workers
ran successfully. Poor scientific scores cannot fail acceptance or remove a
registered row. Fail closed for invalid inputs, leakage, nonfinite required
quantities, corrupt lineage, or forbidden access. Keep runtime acceptance short
and distinct from full scientific fits; its outputs are not extra scored studies.

## 14. Documentation authority and explicit conflicts

This migration plan governs only the new Delphes namespace. The original
[repository transfer plan](../../REPOSITORY_TRANSFER_PLAN.md),
[data contract](../DATA_CONTRACT.md), and
[experiment contract](../EXPERIMENT_CONTRACT.md) describe historical defaults
including filename labels, ten outputs, fixed budgets, and resume behavior.
Their cross-cutting integrity principles remain useful; their dataset-specific
defaults cannot silently define the new benchmark.

The [four-spine plan](HCWDL_TRI100_FOUR_SPINE_LOGIT_IMPLEMENTATION_PLAN.md) and
[four-spine contract](../contracts/HCWDL_TRI100_FOUR_SPINE_LOGIT.md) supply the
proposed path/schedule reference. Their imported-U000 requirement is replaced
here by a fresh Delphes U000. The
[full-cardinality plan](HCWDL_TRI100_FOUR_SPINE_FULL_CARDINALITY_BOTTLENECK_MATCHING_IMPLEMENTATION_PLAN.md)
supplies the proposed solver objective, not permission to reuse old assignments
or its 21-channel endpoint contracts. Authorized early stopping and no rolling
resume are explicit campaign policies, distinct from historical fixed-budget
baseline rules.

Existing persistent-support, MT20, salience, RSET/RREL, and fusion studies retain
their own scope. This plan does not modify them or adopt their latest settings
implicitly. Existing files, datasets, worktrees, and queued jobs remain under
their original identities. Update the [handoff](../HANDOFF.md) with actual
implementation/test evidence as stages finish, following [testing](../TESTING.md).

## 15. Current handoff and next concrete work

Implemented locally: the isolated adapter and v1 contract families, complete
333-file hash/cycle/label inventory, deterministic file-group splits, 17-feature
HLT/offline interface, exact full-cardinality matching and U/D kernels, compact
assignment workers, RAM-only ragged caches with bounded process preparation,
class-aware metrics, row-joined probability shards, and CE/KD training kernels.
The graph contains 31 fresh fits and 26 probability publications. A separate
production wrapper materializes it only after the required measured evidence.

The immutable reservoir population has 8,996,860 rows: 5,376,107 train,
1,810,176 validation and 1,810,577 final test. It now feeds the four explicit
training-size profiles in section 6, with fixed 1M validation and 1M final test;
the initial intended run is TRAIN_500K, not full population. Counts inspected before role sealing imply
a nontruncating capacity of 240 (HLT maximum 176, offline maximum 231). The
shared model disables random sequence trimming, so no hidden model-side token
discard defeats that policy. No new-data scientific fit has been launched.

Stage D has synthetic reference and bounded native-row evidence, not a claim
that all production assignments have been built. Full assignment publication
and its reducer remain pending execution. Stage E has unit-tested training and
bank components and a real-Weaver acceptance CLI; the local scientific Python
environment does not currently contain installed Weaver. Its CPU toy-model
tests are explicitly not genuine-Weaver parity. Stage F now supplies a tested
source-pinned Slurm workflow, but its deployment evidence is still outstanding.

The user has uploaded the unchanged raw snapshot and reservoir manifests to RC
and supplied successful checksum verification for all 333 files. Next: transfer
the new subset registry/profile metadata after local validation, commit/push
the migration, create the clean pinned RC worktree, build the chosen profile's
matching foundation, run genuine Weaver/A100 acceptance and measured resources,
then create/audit the production dry run and obtain explicit live authorization.
The initial deferral during split integration is superseded by section 15.1's
SPORC tooling preparation. No upload or job was initiated by this implementation.
Do not reuse old FullSim runtime evidence as permission to launch this benchmark.
Producer configuration/zero-error/event-group details remain provisional as
authorized; the subsequent unchanged-label confirmation is recorded above. Whether
the ladder preserves the old scientific gains is an outcome to measure, not a
criterion for passing implementation checks.

### 15.1 SPORC readiness boundary

`scripts/prepare_jetclass2_delphes_sporc.py create` publishes a fresh,
source-pinned `READINESS_SPEC/v1`, the chosen subset foundation and exact dry
ledger. Its only tasks are sample -> assignment array -> foundation lock ->
A100 profile. Slurm `afterok` on the array requires all elements to succeed.
It never submits any of the 31 scientific fits. Live readiness submission
requires the separate phrase `AUTHORIZE JETCLASS2 DELPHES SPORC READINESS ONLY`.
It verifies snapshot bytes and storage headroom before submission, journals
exact IDs, refuses ambiguous acknowledgements and never cancels other jobs.
There is no auto-launch at the end of this gate.

New operational contracts are EXECUTION_SITE/v1, READINESS_SPEC/v1 and
RESOURCE_MEASUREMENTS/v1, plus RUNTIME_PROFILE/v2, INSTALLED_ENVIRONMENT/v2 and
CAMPAIGN_SPEC/v3. The scientific plan, recipe, inputs, masks and foundations
are unchanged. Previous runtime artifacts cannot be reinterpreted as A100
evidence. Production binds the measured site, GPU identity, CPU/RAM allocation,
workers, Python/Weaver/numerical versions and installed Weaver source hashes.
The explicit legacy Tigris site remains selectable for future fresh profiling;
SPORC is the selected site here. Full scientific submission still requires
the real profile, a separate 59-job dry run and its own explicit authorization.
