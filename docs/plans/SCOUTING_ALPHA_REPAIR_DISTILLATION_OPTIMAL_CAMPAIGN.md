# Scouting Particle-Matched Alpha-Repair Distillation: Optimal Campaign Plan

Status: **active; fitted-strict selective campaign implemented locally**.
Activated by the user's explicit invocation of the PMARD implementation
mandate on 2026-08-05. This status does not claim real-data validation,
installed-Weaver acceptance, measured resources, or a scientific result.

Short name: **PMARD** (Particle-Matched Alpha-Repair Distillation).

The reusable implementation mandate is
[`pmard_implementation_agent_prompt.md`](../../pmard_implementation_agent_prompt.md).
The user invoked that mandate on 2026-08-05. This document therefore has
campaign-level scientific authority for PMARD implementation and execution.

### 2026-08-05 selective-matcher amendment

The working constituent matcher is now the calibrated `fitted_strict`
algorithm at confidence threshold `0.9828147479721088`, specified by
[`docs/contracts/FITTED_STRICT_MATCHER.md`](../contracts/FITTED_STRICT_MATCHER.md).
It intentionally accepts only a high-confidence subset (64.86% of HLT tokens
and 67.21% of aggregate HLT pT in its untouched canonical test sample).
Unmatched HLT tokens remain exactly HLT. This decision supersedes statements
below that make 100% assignment a requirement of the primary scientific view.

The new primary contract is `SELECTIVE_FULL_PARTICLE_ENDPOINT/v1`. At alpha 1,
every accepted match replaces p4 and the complete projected 21-field record;
every unmatched token remains exactly HLT, including order, count, mask, raw
values, and derived model inputs. Intermediate alpha uses continuous
interpolation plus deterministic identity-bound nested discrete/validity
switches only on accepted tokens. The old `FULL_PARTICLE_ENDPOINT/v1` remains
available as a strict complete-assignment diagnostic and was not weakened.

The production graph computes fitted-strict assignments once per selected jet,
publishes immutable per-source sparse CSR/DEFLATE shards with uint8 HLT index,
uint16 offline index, and uint16 calibrated confidence, and reloads them through
a bounded source LRU. Dense assignment datasets and repaired particle datasets
remain forbidden. The assignment authorization binds the matcher artifact,
exact threshold, split, deterministic row selection, and assignment manifest.

This amendment supersedes all later passages that require M0--M5 cross-fitting,
100% HLT-token coverage, online rematching by training jobs, or
`FULL_PARTICLE_ENDPOINT/v1` as the campaign primary. Mechanism ablations reuse
the same fixed assignment cache so they vary repair semantics rather than
silently changing correspondence.

Three initial oracles are registered on identical selected rows: the HLT
baseline, the native offline TOFF model, and T100 trained/evaluated on the
matched-offline endpoint. The last is deliberately a selective hybrid oracle,
not a claim to reconstruct the native offline jet.

The campaign has three modes. Smoke selects 4,096 train and validation jets and
does not open final test. Pilot selects exactly 300,000 train, 100,000
validation, and—only after finalist/execution locks—100,000 final-test jets.
Production uses every mapped jet. Bounded selections allocate per-class counts
by largest remainder from the role population and choose identities by stable
SHA-256 rank, preserving class proportions without copying ROOT data.

This document specifies a new campaign on the paired CMS ScoutingAK8 2024
dataset. It is independent of the repository's other planned or active
campaigns. Making PMARD active requires an explicit documentation-authority
change, new versioned Scouting data/model contracts, an immutable campaign
specification, focused tests, and a real Tigris miniature.

The operational source for the dataset is:

```text
C:\Users\22rya\ComputerScience\FCV\SCOUTING_AK8_USAGE_GUIDE.md
```

The operational-guide snapshot used to instantiate this plan has SHA-256
`c80b3f3459848c4f634f8f3a861fc83da0bfa530f71e314826123d8dc6774ac9`.
The guide remains explanatory provenance; the repository-local contracts and
manifests frozen below are authoritative for execution.

The canonical read-only research-compute data root is:

```text
/home/ryreu/cms/data/ScoutingAK8_native_compact/2024/train
```

The campaign does not convert, rewrite, pad, or permanently cache the ROOT
dataset. It uses projected Uproot/Awkward reads and constructs constituent
matches, repaired views, and teacher outputs in bounded process memory or
job-local RAM-backed storage.

## 1. Executive decision

The primary experiment is:

```text
paired HLT/offline jet row
        |
        +-- HLT constituent set ------------------------------> HLT student
        |
        +-- inferred one-to-one constituent correspondence
                  |
                  +-- alpha-repaired HLT constituent set ----> teacher

teacher logits + ordinary HLT-teacher logits + hard label
        |
        +-----------------------------------------------------> HLT-only student
```

The teacher is not a full offline Particle Transformer and is not restricted
by a residual adapter. It is the same style and capacity of Particle
Transformer as the HLT model, but it receives an HLT-cardinality input in
which every model-visible HLT constituent has a unique offline counterpart.
Continuous quantities move a fixed fraction `alpha` toward that counterpart;
discrete quantities use deterministic nested endpoint switches.

For every view:

- HLT constituent count, ordering, padding, and mask are unchanged;
- the primary full-particle family requires every visible HLT constituent to
  have a unique accepted offline assignment and fails closed otherwise;
- extra offline constituents are never added;
- at `alpha = 1`, all 21 feature channels and the p4 in every visible slot are
  the matched offline endpoint, with no retained HLT model value;
- matching indices, confidences, residuals, and offline masks never enter the
  teacher or deployable student;
- `alpha = 0` is exactly the canonical HLT input, not a numerically similar
  reconstruction;
- deployable inference accepts HLT inputs only.

The central hypothesis is that an intermediate privilege level yields a
teacher whose improvement is both real and more recoverable by an HLT-only
student than the improvement of a full offline teacher.

## 2. Scientific questions

PMARD answers the following questions in order.

1. Can HLT and offline constituents in an already paired Scouting/offline jet
   be associated with sufficiently high precision for controlled input
   repair?
2. Does uniformly moving all accepted HLT constituents toward their offline
   counterparts produce a smooth and useful teacher-quality ladder?
3. At which privilege strength is the largest fraction of teacher gain
   transferable back to an HLT-only student?
4. Does a mixture of hard-label CE, ordinary HLT self-distillation, and
   privileged-teacher distillation outperform ordinary self-distillation?
5. Is any improvement specifically caused by the offline-directed repair,
   rather than extra compute, target smoothing, changed resolution, matching
   artifacts, or teacher ensembling?
6. Do late representations or relations transfer additional information
   beyond logits when teacher and student token identities are aligned?
7. Can the resulting HLT-only student seed another bounded generation of
   privileged-teacher training and yield further gains?
8. How do gains depend on match confidence, particle category, multiplicity,
   jet density, kinematics, and class?

### 2.1 Primary estimand and multiplicity

The primary deployable contrast is the five-seed, validation-selected PMARD
student versus the five-seed ordinary HLT self-KD student `K1`, evaluated on
the identical sealed final-test HLT rows. Selection of alpha, loss mixture,
and representation variant uses validation only under the frozen selector.

The primary effect is reported for macro mean log QCD rejection and
multiclass CE, with the direction of both declared in advance. Accuracy,
macro OVR AUC, calibration, and per-class results are secondary outcomes. The
complete alpha curve is a validation result; final-test inference is not run
for every alpha merely to find a more favorable point.

The locked diagnostic final-test wave contains only:

- the CE HLT baseline;
- ordinary HLT self-KD;
- the locked PMARD finalist or predeclared two finalists;
- the selected alpha teacher, `T0`, and the full offline oracle required to
  interpret the privilege gap;
- the specific null control predeclared for final confirmation.

All other ablation conclusions remain validation/confirmation conclusions.
If multiple deployable finalists are locked, family-wise reporting identifies
both comparisons and applies the predeclared multiplicity procedure rather
than presenting each as an isolated primary test.

## 3. Claims and non-claims

Allowed claims are deliberately conditional on the inferred matcher.

PMARD may establish that:

- offline-directed, partially repaired Scouting inputs improve a same-capacity
  ParT teacher;
- some of that improvement is transferable into an HLT-only ParT;
- an intermediate privilege level transfers better than full offline KD;
- token-aligned late representation distillation adds or does not add value;
- iterative privileged distillation produces a bounded generational trend.

PMARD may not claim that:

- inferred matches are physical truth without independent truth evidence;
- the repaired view is a real detector configuration;
- offline reconstruction is unsmeared particle truth;
- a forced one-to-one match describes split or merged reconstruction objects;
- validation or test performance proves match correctness;
- a student has learned information absent from its HLT view;
- results on this dataset automatically transfer to another trigger or
  detector domain.

The formal term is **offline-directed alpha repair**. “Unsmearing” may be used
as intuition, but not as the artifact or contract name.

## 4. Dataset contract instantiated for PMARD

### 4.1 Immutable source

Expected source facts are:

```text
ROOT files                         53
ROOT tree                          tree
raw rows                           5,084,737
logical/physical branches          297
baseline-selected jets             4,638,163
baseline-selected mapped jets      4,635,175
baseline-selected unmapped jets    2,988
mapped signal jets                 1,292,811
mapped QCD jets                    3,342,364
classes                            15
compressed source bytes            41,059,470,126
compressed source size             38.239611 GiB
aggregate logical-content SHA-256  30ba783bf08be19c65e9dfeb0dcced1b9634ce8a6387bf066e65969393336630
```

The two source directories are simulated samples, not collision data:

```text
H0HpHm_mixed_new
QCD_PT-mixed_TuneCP5_13p6TeV_pythia8_new
```

The aggregate digest authenticates the verified logical ROOT content produced
by the lossless compaction. It does not replace the per-file byte digests in the
source manifest. The data lock records both the aggregate logical-content
digest and all 53 compact-file SHA-256 digests.

The compact files may contain both `tree;1` and `tree;2` keys whose latest
logical `tree` points to the same scientific baskets. Readers open the logical
name `tree` exactly once (`root_file["tree"]` or `Get("tree")`) and must never
enumerate key cycles as independent datasets. The source audit asserts one
logical tree, 297 branches, and the expected entry count for every file.

One ROOT row is already a matched **jet pair**:

- HLT/scouting jet and `scoutpfcand_*` constituents;
- offline jet and `cpfcandlt_*`, `npfcand_*`, and optional `sv_*` objects.

PMARD must not perform another jet-level nearest-neighbor association. Its
matching problem exists only within each paired row.

The raw files remain immutable. The campaign must never open them in update
mode or place outputs beneath the data root.

### 4.2 Split contract

The primary split is version 2 of the deterministic, source-file-disjoint
contract:

```text
seed                 12345
target fractions     0.6 train / 0.2 validation / 0.2 final test
indivisible unit     complete authenticated ROOT source file
balance variables    all 15 mapped class counts in every source file
file-count quotas    largest-remainder allocation of the 53 complete files
optimizer            exact mixed-integer minimax allocation
acceptance tolerance maximum 0.02 absolute deviation from any of the 45
                     requested class-by-role fractions
```

For the locked mapped population of 4,635,175 jets, the aggregate fractional
targets are 2,781,105 train, 927,035 validation, and 927,035 final-test jets.
Those are targets, not fabricated realized counts. Files are indivisible, so
exact integer 60/20/20 counts for every class and exact file disjointness are
not simultaneously guaranteed. The allocator therefore globally minimizes
the worst absolute deviation among the 45 class-by-role fractions, then
minimizes their aggregate deviation. With 53 files the canonical file-count
quota is 32 train, 11 validation, and 10 final test. If the exact optimizer
cannot keep every realized class fraction within two percentage points of its
target, split construction fails closed before campaign creation. The real
manifest publishes the optimum and every realized count, fraction, and
deviation; no training begins until that manifest validates.

The source audit makes one streaming scalar-only pass over the baseline and
label branches to authenticate per-file 15-class counts. This is the sole
pre-split privilege to inspect labels over the complete unsplit inventory. It
may influence only immutable split membership, never model selection, matcher
fitting, feature statistics, or hyperparameters. Once membership is published,
the source and split manifest hashes become campaign parents and the ordinary
final-test access seal applies. This exception is required for a genuinely
class-stratified file split and is not an authorization to inspect final-test
features or predictions.

Manifest paths are normalized POSIX paths relative to the authenticated data
root, never machine-specific absolute paths. The data lock records the split
generator source hash and environment as well as the seed; the manifests, not
the seed alone, define membership. Per-file and per-role baseline, mapped, and
15-class counts are materialized in the authenticated source/split reports so
the balance guarantee is independently auditable.

The exact `train.txt`, `val.txt`, `test.txt`, `split_manifest.csv`, source-file
hashes, optimizer diagnostics, and role class counts are immutable campaign
parents. The directory name `train` in the source dataset is not treated as a
supplied scientific split.

`event_no` is not a globally unique event identifier. PMARD therefore claims
file disjointness and row disjointness, not proven physical-event disjointness
across files.

Role permissions are:

| Role | Labels | HLT | Offline | Fit matcher/statistics | Select models |
|---|---:|---:|---:|---:|---:|
| `train` | yes | yes | yes | yes | no |
| `validation` | yes | yes | yes | no | yes |
| `final_test` before locks | manifest/checksum only | no branch reads or model output | no branch reads or model output | no | no |
| `final_test` after locks | yes for reporting | yes | oracle diagnostics only | no | no |

No matcher, response normalization, calibration threshold, class weight,
reweighting histogram, or feature statistic may be fit on validation or final
test rows.

The canonical paired-row identity is the tuple:

```text
(normalized source path relative to the immutable data root,
 logical tree name "tree",
 zero-based ROOT entry index)
```

The normalized path uses forward slashes and rejects `..`, case-normalization
ambiguity, aliases, and paths outside the authenticated data root. The source
file digest is a manifest parent but is not duplicated into every row key.
Labels are deliberately excluded from identity so inconsistent relabeling
cannot evade an overlap audit. HLT and offline arrays inherit exactly the same
row identity; any loss of within-entry alignment fails closed.

After the one-time pre-role stratification pass described above, no PMARD
worker opens a final-test particle or label branch before the finalist and
execution locks. Source checksum verification and split-manifest validation
are the only permitted pre-lock operations on assigned final-test files. This
is stricter than merely forbidding predictions and removes accidental matcher
or preprocessing feedback from final test.

### 4.3 Labels and primary task

The reusable baseline selection is exactly:

```text
(jet_tightId == 1)
& (jet_no < 2)
& (scoutfj_pt > 200)
& (scoutfj_pt < 5000)
& (scoutfj_sdmass > 0)
& (scoutfj_sdmass < 650)
& (n_scoutpfcands > 0)
```

`ak.num(scoutpfcand_px) > 0` is an allowed equivalent for the final term only
after the source audit reproduces equality with `n_scoutpfcands > 0`. The
historical `event_no % 7 != 0` and sample-dependent selection in the donor
Weaver YAML are not baseline cuts and do not enter PMARD. Selection is applied
before label mapping.

The primary task is the documented 15-class classification problem, in this
exact order and with these exact ROOT definitions and full-population reference
counts:

| Index | Output | Exact ROOT definition after baseline selection | Jets | Fraction |
|---:|---|---|---:|---:|
| 0 | QCD | `(fj_label >= 309) & (fj_label < 314)` | 3,342,364 | 72.109% |
| 1 | Xbb | `scoutfj_label == 17` | 89,046 | 1.921% |
| 2 | Xcc | `scoutfj_label == 18` | 88,789 | 1.916% |
| 3 | Xss | `scoutfj_label == 19` | 87,849 | 1.895% |
| 4 | Xqq | `(scoutfj_label == 20) & (scoutfj_gen_pid == 25)` | 88,318 | 1.905% |
| 5 | Xbs | `scoutfj_label == 22` | 88,310 | 1.905% |
| 6 | Xgg | `scoutfj_label == 24` | 89,405 | 1.929% |
| 7 | Xee | `scoutfj_label == 25` | 34,295 | 0.740% |
| 8 | Xmm | `scoutfj_label == 26` | 32,498 | 0.701% |
| 9 | Xtauhtaue | `scoutfj_label == 27` | 37,114 | 0.801% |
| 10 | Xtauhtaum | `scoutfj_label == 28` | 35,305 | 0.762% |
| 11 | Xtauhtauh | `scoutfj_label == 29` | 72,757 | 1.570% |
| 12 | Xbc | `scoutfj_label == 21` | 184,037 | 3.970% |
| 13 | Xcs | `scoutfj_label == 23` | 182,558 | 3.939% |
| 14 | Xud | `(scoutfj_label == 20) & (abs(scoutfj_gen_pid) == 37)` | 182,530 | 3.938% |

The mapper evaluates all 15 membership predicates, requires at most one true
membership, and emits `-1` when none is true. Exactly 2,988 baseline-selected
rows must map to `-1`; they are excluded before split-role training statistics
or loss weights are computed. QCD deliberately uses offline `fj_label` while
signals use `scoutfj_label`; substituting `scoutfj_label` for the QCD definition
changes the population and is forbidden. Truth labels, sample flags, generator
fields, existing classifier probabilities, and target mass factors are targets
or observers and never model inputs.

These are simulated training samples. The selected 72.109% QCD fraction is a
property of the stored production mixture, not a measured collider prior.
PMARD applies no cross-section or luminosity weighting. Unweighted validation
accuracy and CE therefore describe this selected MC mixture; macro metrics and
per-signal QCD rejection carry the primary cross-class interpretation.

Hbb-versus-QCD is a report-only secondary task unless a later immutable
campaign specification explicitly registers a separate binary campaign. It
must not be used to tune the 15-class primary campaign.

### 4.4 Native collection schema and length invariants

The immutable jagged collection contract is:

| Collection | Stored fields | Meaning | Per-row length |
|---|---:|---|---|
| `scoutpfcand_*` | 31 | HLT/scouting PF candidates | `n_scoutpfcands` |
| `cpfcandlt_*` | 44 | offline charged PF candidates plus lost-track entries | `n_cpfcands + n_lts` |
| `npfcand_*` | 18 | offline neutral PF candidates | `n_npfcands` |
| `sv_*` | 27 | offline secondary vertices | `n_sv` |

All members of a collection have identical length within a row. Any family
length disagreement fails the row and the source audit; it is never repaired by
clipping one branch. Scalar observer, label, jet, and event branches remain
separate from constituent inputs.

The primary matcher uses a canonical combined offline index namespace:

```text
-1                                      unmatched
0 <= j < n_cpfcandlt                    cpfcandlt local index j
n_cpfcandlt <= j < n_cpfcandlt+n_npfcand
                                        npfcand local index j-n_cpfcandlt
```

Here `n_cpfcandlt = ak.num(cpfcandlt_px)` and
`n_npfcand = ak.num(npfcand_px)`. The collection boundary and both native local
indices are retained in matcher diagnostics. Every assignment consumer checks
the index range and collection/category compatibility before dereferencing.

Particle categories are decoded without precedence. For HLT, exactly one of
`isEl`, `isMu`, `isChargedHad`, `isGamma`, and `isNeutralHad` must equal one and
the other four must equal zero. For `cpfcandlt_*`, exactly one of `isEl`,
`isMu`, and `isChargedHad` must equal one; for `npfcand_*`, exactly one of
`isGamma` and `isNeutralHad` must equal one. Nonfinite, nonbinary, missing,
zero-hot, or multi-hot category values make that object unmatchable. They are
counted by collection and split; no category priority silently coerces them.
The legacy conservative/P4 matcher requires category compatibility and charged
matches additionally require finite equal nonzero stored charge. It cannot
authorize `FULL_PARTICLE_ENDPOINT/v1`, because those hard gates make PID and
charge disagreements unrepairable by construction. The full-particle matcher
must instead allow such disagreements, establish identity without the disputed
field, validate them at its locked precision target, and produce a complete
unique assignment for every visible HLT token. That matcher design is a
separate required lock; the repair implementation fails closed until its
complete assignments exist.

Lost-track membership is not inferred from pT ordering. Before the matcher
design lock, Stage A must authenticate the producer semantics and cross-tabulate
`cpfcandlt_isLostTrack`, `n_cpfcands`, `n_lts`, native position, and the family
length invariant over the complete selected training role. The resulting exact
lost-track predicate and disagreement policy are stored in the native-offline
input contract. Until that predicate is locked, all candidate rows for which
lost-track status is ambiguous are unmatchable. The primary matcher excludes
every locked lost-track entry; an allow-lost-track arm requires a new matcher
and repair variant.

### 4.5 Sequence and truncation contract

The canonical HLT sequence is `scoutpfcand_*`, stored in descending HLT
candidate `pT`. The primary maximum length is 200. The observed selected HLT
maximum is 214 and only four documented selected jets exceed 200, so the
truncation is small but must still be reported.

The model-visible HLT sequence is exactly the first
`min(n_scoutpfcands, 200)` stored candidates. The primary matcher constructs
assignments only for those model-visible HLT indices. Candidates at stored HLT
indices 200 and above neither receive assignments nor enter matcher context;
their counts and categories are reported as truncation diagnostics. This keeps
the match table, repaired teacher, and HLT student on exactly the same token
universe.

Offline charged/lost-track and neutral collections may reach 309 combined
objects. The primary matcher cap is the complete current-row offline collection:
all `cpfcandlt_*` and `npfcand_*` entries are examined before category and
lost-track eligibility are applied. There is no smaller numerical offline cap
in the primary matcher. An accelerator-specific cap is permitted only as a
separately named matcher variant after a measured recall study, and it must
record every plausible partner excluded by that cap.

Offline storage order is preserved for identity and indexing; it is not assumed
to be globally pT sorted. In particular, `cpfcandlt_*` must never be truncated
under an implicit global-pT-order assumption. Any sorted offline modeling view
stores its explicit stable sort key and native-index inverse map and is not used
as the assignment-table index namespace.

## 5. Canonical Scouting Particle Transformer

PMARD requires a new Scouting-specific model/input contract. It does not
pretend that the repository's JetClass 17-feature, 10-class contract is
interchangeable with this dataset.

The HLT baseline uses:

```text
particle features        CMSSW V00 21-channel HLT feature order
particle vectors         px, py, pz, energy
particle mask            one valid/padding channel
maximum sequence         200
pair relation            Weaver standard-four
particle embed dims      [128, 512, 128]
pair embed dims          [64, 64, 64]
attention heads          8
particle blocks          8
class-attention blocks   2
activation               GELU
output classes           15
initialization           from scratch unless an arm says otherwise
```

The immutable upstream preprocessing source is commit
`37a104b51c8458bcda97a95ba52d304a5041e922`:

```text
https://github.com/cms-data/RecoBTag-Combined/blob/37a104b51c8458bcda97a95ba52d304a5041e922/Run3Scouting/GlobalParticleTransformerAK8/General/V00/preprocess.json
```

The verified documented local transcription has SHA-256
`2e346c5da8491046205f29cba21b0f6e434e273eb89dae068aa5dd40e2ca91f7`.
Before the data lock, that JSON is vendored into this repository as a
content-addressed contract; runtime must not depend on the mutable external
checkout named at the top of this plan.

The exact HLT feature order and transformation constants are:

| # | Model feature | Stored branch | Median | Factor | Output bounds |
|---:|---|---|---:|---:|---|
| 0 | `pfcand_quality` | `scoutpfcand_quality` | 0 | 0.2 | [-5, 5] |
| 1 | `pfcand_charge` | `scoutpfcand_charge` | 0 | 1 | [-1e32, 1e32] |
| 2 | `pfcand_isEl` | `scoutpfcand_isEl` | 0 | 1 | [-1e32, 1e32] |
| 3 | `pfcand_isMu` | `scoutpfcand_isMu` | 0 | 1 | [-1e32, 1e32] |
| 4 | `pfcand_isChargedHad` | `scoutpfcand_isChargedHad` | 0 | 1 | [-1e32, 1e32] |
| 5 | `pfcand_isGamma` | `scoutpfcand_isGamma` | 0 | 1 | [-1e32, 1e32] |
| 6 | `pfcand_isNeutralHad` | `scoutpfcand_isNeutralHad` | 0 | 1 | [-1e32, 1e32] |
| 7 | `pfcand_phirel` | `scoutpfcand_phirel` | 0 | 1 | [-1e32, 1e32] |
| 8 | `pfcand_etarel` | `scoutpfcand_etarel` | 0 | 1 | [-1e32, 1e32] |
| 9 | `pfcand_abseta` | `scoutpfcand_abseta` | 0.6 | 1.6 | [-5, 5] |
| 10 | `pfcand_pt_log` | `scoutpfcand_pt_log` | 1 | 0.5 | [-5, 5] |
| 11 | `pfcand_normchi2` | `scoutpfcand_normchi2` | 5 | 0.2 | [-5, 5] |
| 12 | `pfcand_dz` | `scoutpfcand_dz` | 0 | 180 | [-5, 5] |
| 13 | `pfcand_dxy` | `scoutpfcand_dxy` | 0 | 300 | [-5, 5] |
| 14 | `pfcand_dxysig` | `scoutpfcand_dxysig` | 0 | 1 | [-5, 5] |
| 15 | `pfcand_btagEtaRel` | `scoutpfcand_btagEtaRel` | 1.5 | 0.5 | [-5, 5] |
| 16 | `pfcand_btagPtRatio` | `scoutpfcand_btagPtRatio` | 0 | 1 | [-5, 5] |
| 17 | `pfcand_btagPParRatio` | `scoutpfcand_btagPParRatio` | 0 | 1 | [-5, 5] |
| 18 | `pfcand_dzsig` | `scoutpfcand_dzsig` | 0 | 0.9 | [-5, 5] |
| 19 | `pfcand_e_log_nopuppi` | `scoutpfcand_e_log` | 1.3 | 0.5 | [-5, 5] |
| 20 | `pfcand_lostInnerHits` | `scoutpfcand_lostInnerHits` | 0 | 1 | [-1e32, 1e32] |

Every feature uses replacement value zero for NaN, infinity, or raw values
outside `[-1e32, 1e32]`. The unnormalized vector order is exactly
`scoutpfcand_px`, `scoutpfcand_py`, `scoutpfcand_pz`,
`scoutpfcand_energy`. The mask is one for each retained real candidate and zero
for padding.

The exact CMSSW preprocessing order is retained for HLT inputs:

```text
replace nonfinite/out-of-domain -> subtract median -> multiply factor -> clip
```

Padding is zero and masked. The input contract must prove that no offline jet
axis, offline jet scalar, matching result, or construction field enters an
HLT-only forward.

Fixed-size training tensors use length 200. The deployment contract permits
dynamic length `clamp(raw_length, 16, 200)`; real values agree, and positions
beyond the dynamic deployment length are masked zeros. Byte identity is
required between the canonical HLT tensor builder and the `alpha=0` short
circuit in the same implementation. Independent CMSSW/Weaver parity is checked
in FP32 with versioned absolute and relative tolerances rather than an undefined
cross-runtime byte-comparison claim.

The alpha-repaired teacher has exactly the same architecture and tensor
shapes. Only its authorized repaired input values differ.

Required architecture controls are:

- HLT CE baseline at identical capacity;
- HLT self-distillation student at identical capacity;
- HLT ParT with the extra representation-projection parameters but their
  privileged loss disabled;
- full offline ParT oracle, reported as nondeployable and not capacity-matched
  unless its native charged/neutral encoders are explicitly matched;
- optional HLT ensemble teacher with similar validation accuracy to separate
  privileged information from generic better-teacher effects.

## 6. Constituent matching is a separate scientific component

The matcher is selected, locked, and validated before any alpha-teacher or KD
result is inspected. Downstream classification performance is forbidden as a
matcher-training loss or matcher-selection criterion.

### 6.1 Match taxonomy

PMARD distinguishes:

```text
charged PF <-> charged offline PF
electron   <-> offline electron
muon       <-> offline muon
photon     <-> offline photon
neutral hadron <-> offline neutral hadron
HLT object <-> offline lost track       initially forbidden
one-to-many or many-to-one              initially abstain
no credible partner                     unmatched
```

The primary v1 compatibility matrix is exact:

| HLT category | Allowed offline category | Extra hard constraint |
|---|---|---|
| electron | charged electron | equal charge; not lost track |
| muon | charged muon | equal charge; not lost track |
| charged hadron | charged hadron | equal charge; not lost track |
| photon | neutral photon | none |
| neutral hadron | neutral hadron | none |

Rows with invalid, non-exclusive, or missing required category indicators are
unmatchable rather than coerced. Cross-category matching, charge flips, and
offline lost-track partners require separately named variants and can never
enter the primary matcher through a soft score.

The primary view uses only high-confidence one-to-one assignments within a
compatible category. Offline lost tracks are not silently treated as normal
PF candidates. A later explicit ablation may allow them, but its match and
repair semantics require a new variant identifier.

### 6.2 Physics-space candidate construction

Matching operates on raw physical quantities before model normalization.
Absolute candidate direction is reconstructed from `(px, py, pz)` rather
than comparing only coordinates relative to two different jet axes.

Candidate edges use broad, type-dependent gates over legitimate quantities:

- wrapped angular displacement;
- log `pT` and energy response;
- charge agreement;
- compatible particle category;
- `dxy`, `dz`, their significances, and quality information for charged
  objects when definitions are audited as compatible;
- jet-region, local-density, and rank context;
- missingness and measurement validity.

The gate is only a computational and physical plausibility filter. It never
chooses a winner independently.

The matching branch allowlist is frozen before implementation. Its maximum v1
capability is:

```text
HLT candidate:
  px, py, pz, energy, pt, charge,
  isEl, isMu, isChargedHad, isGamma, isNeutralHad,
  dxy, dxysig, dz, dzsig, normchi2, quality, lostInnerHits

offline charged candidate:
  px, py, pz, energy, pt_nopuppi, charge,
  isEl, isMu, isChargedHad, isLostTrack,
  dxy, dxysig, dz, dzsig, normchi2, quality, fromPV,
  trackHighPurity, lostInnerHits

offline neutral candidate:
  px, py, pz, energy, pt_nopuppi, isGamma, isNeutralHad

context:
  collection multiplicities and quantities deterministically derived from
  the candidate allowlist
```

Only fields that pass a units, missingness, range, and definition audit are
enabled in a matcher version. Jet labels, generator fields, sample flags,
existing classifier outputs, mass-regression targets, and downstream model
predictions are structurally forbidden. `fj_*` and `scoutfj_*` classification
or truth fields cannot enter through a broad wildcard read.

The candidate graph remains sparse through physical gates and is decomposed
into connected components for assignment. No plausible edge is silently
dropped to satisfy a fixed candidate cap. A pathologically large component is
processed in a smaller matcher batch or explicitly marked unresolved; it is
not greedily truncated.

### 6.3 Train-only response bootstrap

Because exact constituent truth is unavailable, PMARD uses a conservative
bootstrap:

1. Construct ultra-pure seed pairs from isolated, mutually nearest,
   charge/type-compatible candidates with very tight angular, response, and
   track compatibility.
2. Estimate type-, `pT`-, `eta`-, and density-conditional residual models from
   those train-only seeds.
3. Estimate accidental-pair distributions from incompatible pairs, distant
   in-jet candidates, and event-mixed candidates.
4. Treat every other plausible candidate edge as **unlabeled**, not as a
   negative. Only incompatible, sufficiently distant, extreme-response, or
   event-mixed pairs are definite negatives.
5. Fit likelihood and contextual scorers on known positives and definite
   negatives; unlabeled edges do not enter binary cross entropy.
6. Calibrate edge logits on a disjoint held-out graph subset and solve the jet
   globally with unmatched states.
7. Add only Hungarian/OT-consensus, mutual-first assignments above calibrated
   probability `0.999` as pseudo-positives and repeat for exactly two rounds.

The bootstrap never expands until convergence based on classification. Its
iteration count, seed thresholds, feature list, binning, smoothing, and random
seeds are frozen before teacher training.

### 6.4 Cross-fitting and calibration isolation

Matcher overfitting could otherwise manufacture unusually clean repairs on
the same rows used to train the matcher. All teacher-training assignments are
therefore out-of-fold.

The source files in the training role are divided into five deterministic,
source-stratified matcher folds. For fold `k`:

1. fit seed-response models, edge scorer, calibration, and thresholds using
   only the other four folds;
2. construct assignments for fold `k` without updating from those rows;
3. record the fold model and out-of-fold aggregate diagnostics.

The complete train-role teacher view is the concatenation of those five
out-of-fold assignment streams. After all matcher architecture and threshold
choices are locked, one final inference matcher is fit on all training files
and applied unchanged to validation. It is applied to final test only after
the final-test locks.

Pseudo-label expansion is also fold-local: a row can never teach the scorer or
calibrator that later assigns that same row. Synthetic perturbations inherit
the source fold of their parent. Matcher hyperparameters are selected using
only matching diagnostics aggregated across held-out train folds; validation
classifiers and labels play no role.

The production matcher fit uses 10,000 known-correspondence synthetic jets and
200,000 native jets per fit. Native rows use equal per-source quotas sampled at
evenly spaced mapped-row ordinals during one bounded streaming pass; the
worker never retains graph objects for the complete 2.78M-jet train role. The
matching-only validation uses 2,000 adversarial synthetic jets and 10,000
native jets per fold/full report. Smoke budgets are 200/20,000 for fitting and
200/10,000 for validation. These budgets and the sampling algorithm are in the
immutable campaign registry and matcher report. Once the matcher is locked,
assignment construction still scans every applicable train or validation row.

### 6.5 Primary contextual matcher

The optimal matcher is a physics-informed sparse bipartite graph model:

```text
HLT nodes     -> node encoder ---+
                                 +-> 3 rounds candidate-edge message passing
offline nodes -> node encoder ---+                 |
candidate edge residuals -> edge encoder ----------+-> contextual edge logits
                                                       |
                                                       v
                                       Hungarian+dustbins (M4 primary)
```

Inputs include audited physical pair residuals, validity indicators,
track-quality residuals, local density, ranks, and candidate multiplicities.
Three sparse message-passing rounds make an edge aware of competitors that
share either endpoint without constructing a dense pair tensor.

Context is restricted to the candidate graph and bounded local neighborhoods.
The matcher has no global class token, pooled whole-jet classifier head, sample
identifier, or unrestricted global jet embedding. Event-mixed negatives are
balanced in the permitted kinematic variables so the network cannot solve the
task by recognizing two different sample populations instead of particle
identity. This locality rule reduces the risk that a powerful matcher becomes
an undeclared jet tagger whose match pattern carries class information.

Training supervision is limited to:

- ultra-pure native-source seed matches;
- high-confidence pseudo-labels from the locked physics likelihood solver;
- event-mixed and physically incompatible negatives;
- synthetic known-correspondence tasks formed by applying train-fitted
  response, loss, split, and merge perturbations to offline train jets;
- augmentation consistency and assignment-cycle consistency.

No class label, generator field, existing tagger score, validation statistic,
or downstream KD result is permitted.

The neural matcher is not automatically assumed better than the likelihood
matcher. It becomes primary only if it wins the predeclared matching-only
validation selector. Otherwise the physics likelihood plus global solver is
primary.

### 6.6 Global assignment

For edge log-likelihood ratios `s_ij`, solve a globally exclusive assignment:

```text
maximize   sum_ij M_ij s_ij - unmatched penalties
subject to each HLT and offline object participating in at most one hard match
```

`M4` uses deterministic Hungarian/min-cost assignment with explicit dummy
nodes as its primary hard extraction. `M3` is the independent unbalanced-OT
physics control. `M5` requires agreement between the contextual Hungarian
assignment and the independently scored OT construction.

Tie behavior, edge ordering, floating-point dtype, Sinkhorn tolerance and
iterations, dustbin costs, and hard-extraction rules are versioned. Greedy
nearest-neighbor matching is diagnostic only.

Candidate edges are lexicographically ordered by HLT index then offline index.
Calibration and neural inference use a locked dtype and deterministic runtime.
Before assignment, calibrated edge scores are canonicalized to a versioned
fixed-point grid (`1e-5` logit units in v2, coarser than measured single- versus
multi-graph FP32 launch drift). Scores within a declared guard band of a gate, threshold, or
quantization boundary abstain. Two independent constructions from identical
parents must yield byte-identical hard assignments and aggregate counters.

For an accepted edge, the global alternative margin is measured by re-solving
its connected component with that edge forbidden. Approximate local margins
may be used for screening, but the primary `ultra_pure` lock uses the exact
component re-solve definition.

### 6.7 Confidence and abstention

Edge probabilities are not assignment probabilities. After global assignment,
a held-out logistic calibrator predicts assignment correctness from the
selected-edge logit, exact forbidden-edge global margin, row/column margins,
mutual preference, solver agreement, competitor counts, and category. An
assignment is accepted only when it satisfies all locked conditions:

- high calibrated match probability;
- mutual HLT-to-offline and offline-to-HLT preference;
- sufficient margin over the best alternative global assignment involving
  that object;
- stability under independently sampled, physically meaningful HLT/offline
  perturbations, never a common rigid transform that preserves pair distances;
- compatible type-specific residual likelihood;
- agreement between the primary solver and an independent global solver at
  the ultra-pure operating point.

Primary repair uses a single hard acceptance threshold. Confidence is not
multiplied into `alpha`, because that would conflate repair strength with
matcher belief. Confidence-weighted and posterior-averaged repairs are
registered ablations only.

Three matcher operating points are frozen:

| Operating point | Intended use |
|---|---|
| `ultra_pure` | primary scientific claim and representation KD |
| `high_purity` | coverage sensitivity |
| `high_coverage` | explicit contamination stress test |

If the neutral channel fails its matching-only acceptance criteria, neutral
repair is disabled in the primary view rather than forced. Charged-only repair
then becomes the honest primary experiment.

### 6.8 Validation without exact constituent truth

No single diagnostic proves native-source match correctness. The matcher report
therefore combines:

- precision, recall, and calibration on held-out synthetic
  known-correspondence samples;
- stress samples with deliberately shifted response, higher density, extra
  loss, splitting, and merging;
- seed-match holdout recovery;
- event-mixing false-positive rate;
- assignment stability under independent response/direction perturbations;
- agreement between optimal-transport and Hungarian solutions;
- residual closure by particle type, `pT`, `eta`, multiplicity, and density;
- match fraction and ambiguity rate in every class, without using class in
  the matcher;
- manual event displays for a small predeclared, identity-hashed sample;
- any later independently available simulation association as report-only
  external validation.

The preferred operating point maximizes coverage subject to predeclared
precision and stability requirements. The initial target for the primary
operating point is a lower confidence bound of at least 99% precision on the
in-domain synthetic holdout and all required category-specific bounds. Exact
statistical interval and category thresholds must be frozen in the matcher
contract before the report is generated.

The v2 selector additionally requires event-mixing false-positive rate at
most `0.001`, independent-perturbation assignment stability at least `0.99`,
and synthetic accepted-assignment confidence Brier score at most `0.01` in
every held-out fold. Event-mixed partners are chosen from a kinematically
similar pool but remain a genuinely different jet.

The matching-only selector is lexicographic:

1. reject any arm failing a required synthetic precision/calibration,
   event-mixing false-positive, perturbation-stability, or category bound;
2. minimize the upper confidence bound on stress-sample false matches;
3. maximize native-source accepted coverage at the locked purity point;
4. maximize independent-solver agreement;
5. choose the lower matcher ID.

Class-stratified matching plots are generated only after this selection and
are report-only. They cannot change the matcher, thresholds, or allowed
particle categories.

If those requirements are not met, the campaign still reports the matching
result and runs the registered conservative charged-only and ultra-pure arms.
Poor coverage or quality is a scientific result, not a job failure.

### 6.9 Required matcher ablations

| ID | Edge score | Assignment | Purpose |
|---|---|---|---|
| `M0` | angular distance only | greedy | weak reference, never primary |
| `M1` | hand-scaled physical cost | Hungarian+dustbin | transparent reference |
| `M2` | fitted likelihood ratio | Hungarian+dustbin | global physics baseline |
| `M3` | fitted likelihood ratio | unbalanced OT+dustbin | solver comparison |
| `M4` | sparse contextual graph model | Hungarian+dustbin + post-assignment calibration | proposed primary |
| `M5` | `M3`/`M4` consensus | exact-margin hard abstention | maximum-purity option |

Matching arms publish only compact model parameters, configuration, aggregate
reports, and hashed event-display identities. Full-dataset assignments are
ephemeral.

## 7. Alpha-repaired HLT view

### 7.1 HLT-anchored construction

For every accepted one-to-one match from HLT constituent `i` to offline
constituent `pi(i)`, define a repaired four-vector:

```text
p4_i(alpha) = (1 - alpha) * p4_i(HLT) + alpha * p4_pi(i)(offline)
```

The endpoints are exactly:

```text
HLT       scoutpfcand_px, scoutpfcand_py, scoutpfcand_pz,
          scoutpfcand_energy

charged   cpfcandlt_px, cpfcandlt_py, cpfcandlt_pz,
offline   cpfcandlt_energy

neutral   npfcand_px, npfcand_py, npfcand_pz,
offline   npfcand_energy
```

`pt_nopuppi` is a consistency and matching observable, not substituted for the
stored Cartesian endpoint. The source audit must document precisely whether
each stored p4 is weighted or unweighted; the campaign describes the endpoint
as the **stored offline reconstructed p4**, not particle truth or an assumed
no-PUPPI four-vector.

The offline closure rule is not assumed in advance. Stage A must determine and
lock, separately for `cpfcandlt_*` and `npfcand_*`, whether
`hypot(px, py)` closes to `pt_nopuppi`, `puppiw * pt_nopuppi`, or another
producer-authenticated expression. The contract records the formula, units,
missing-weight behavior, tolerance, failure counts, and representative closure
plots. `pt_nopuppi` may enter a response gate only after that audit. Until the
closure contract passes, offline p4 matching and repair are disabled.

The v2 feature gate uses relative closure tolerance `1e-4` per object and
permits at most a `1e-4` failing-object fraction for the selected formula in
each of HLT, offline charged, and offline neutral collections, independently
in train and validation. Missing reference chunks or zero audited objects fail
closed.

Direct Cartesian four-vector interpolation is the primary parameterization.
It avoids angular branch cuts and remains inside the convex future-directed
four-vector cone when both inputs are physical. All downstream kinematics are
recomputed from the repaired four-vector.

Before matching or repair, both endpoint four-vectors must be finite, have
positive energy, and satisfy the versioned physical-tolerance check
`E >= |p|`. HLT stored `pt` must close against `(px, py)`; each offline
collection must close under its authenticated weighted/unweighted rule. An
invalid required endpoint is not repaired and is counted; a violation above the
predeclared source-audit allowance fails the view build. The implementation
must not silently project malformed endpoints onto a physical four-vector.

For unmatched HLT constituents:

```text
p4_i(alpha) = p4_i(HLT)
```

The primary alpha grid is:

```text
alpha = 0.00, 0.05, 0.10, 0.25, 0.50, 1.00
```

Every alpha for a role uses the identical locked assignment table, accepted
edge set, endpoint particles, and HLT ordering. Matching is
not rerun and its confidence threshold is not retuned as alpha changes. Thus
alpha is the only scientific difference along the privilege ladder.

`alpha = 1` is the **HLT-cardinality offline endpoint**, not the complete
offline jet. It retains HLT multiplicity and slot ordering and discards extra
offline constituents, but every model-visible slot contains only its matched
offline p4 and projected 21-channel offline representation.

Candidate order always remains the original descending-HLT-`pT` order. It is
never resorted by repaired `pT`, because resorting would expose an additional
offline-dependent permutation and break exact teacher/student token identity.

### 7.2 Feature-coherence rule

Repair is performed before CMSSW normalization. Every family declares whether
each channel is recomputed, interpolated, switched discretely, or retained.
ParT standard-four pair inputs always derive from the repaired p4.

Any stored derived feature whose exact formula and reference axis cannot be
authenticated must not be guessed. It remains HLT-valued until a formula audit
proves closure. The implementation must classify each of the 21 channels as:

```text
recomputed from repaired physical values
retained from HLT by design
inapplicable/zero only under an explicit versioned contract
```

At `alpha = 0`, the builder takes an exact HLT short circuit and must produce
byte-identical canonical HLT tensors. It may not rely on recomputation being
approximately equal.

The retained `P4_ONLY/v1` ablation channel map is fixed as follows:

| # | Channel | `P4_ONLY/v1` behavior |
|---:|---|---|
| 0 | quality | retain HLT |
| 1 | charge | retain HLT |
| 2--6 | five particle-category indicators | retain HLT |
| 7 | `phirel` | HLT value plus repaired-minus-HLT wrapped `phi` displacement |
| 8 | `etarel` | HLT value plus repaired-minus-HLT `eta` displacement |
| 9 | `abseta` | HLT value plus repaired-minus-HLT absolute-`eta` displacement |
| 10 | `pt_log` | HLT value plus `log(pt_repaired / pt_HLT)` |
| 11 | `normchi2` | retain HLT |
| 12--14 | `dz`, `dxy`, `dxysig` | retain HLT |
| 15--17 | three b-tag relative/ratio features | retain HLT |
| 18 | `dzsig` | retain HLT |
| 19 | `e_log` | HLT value plus `log(E_repaired / E_HLT)` |
| 20 | `lostInnerHits` | retain HLT |

The four vector supplied to ParT is the repaired `(px, py, pz, energy)`, and
standard-four pair inputs are rebuilt from it. The delta-transport definitions
for channels 7--10 and 19 preserve the stored HLT convention without guessing
an offline relative-axis convention. They use raw, pre-normalization values;
the ordinary pinned CMSSW transformation is applied afterward. Wrapped `phi`
uses the unique half-open interval frozen by the input contract. Required
positive denominators and log epsilon behavior are audited and versioned.

This fixed table prevents an implementation from opportunistically repairing
additional channels after seeing results. `FULL_PARTICLE_ENDPOINT/v1` is a
separate versioned family and does not silently change `P4_ONLY/v1`.

The primary `FULL_PARTICLE_ENDPOINT/v1` projects native charged and neutral
offline particles into the complete HLT 21-channel schema. Charged offline
fields populate their common track and identity channels. Neutral offline
fields populate neutral identity and kinematics; structurally inapplicable
charged fields use the authenticated canonical neutral value. Every field has
a fixed endpoint source and interpolation type in code.

Continuous quantities are repaired before normalization. Cartesian p4 uses a
convex interpolation, wrapped angles use their shortest arc, and ordinary raw
real-valued channels use endpoint interpolation. A validity/applicability
change is a discrete transition rather than interpolation through a sentinel.
PID and charge switch as one atomic identity bundle. Quality and lost-inner-hit
state use their own switches when applicability is unchanged. A
charged/neutral identity change switches all dependent track state with the
identity, preventing an intermediate neutral token with partially populated
charged-track fields.

For discrete group `g`, token `i`, and authenticated row identity `r`, a stable
`u(r,i,g)` is derived from the repair seed and SHA-256. The offline endpoint is
selected iff `u < alpha`. Views are therefore deterministic across epochs,
chunks, ranks, and workers; switches are nested as alpha rises; and every
group is offline at `alpha = 1`.

### 7.3 Repair families

The primary family is `FULL_PARTICLE_ENDPOINT`: an exact all-field offline
endpoint on the unchanged HLT token skeleton. `P4_ONLY` remains the clean
kinematics-only ablation.

Registered mechanism ablations are:

| Repair family | Changed quantities | Purpose |
|---|---|---|
| `FULL_PARTICLE_ENDPOINT` | p4 plus all 21 projected offline fields | primary complete uncorruption ladder |
| `P4_ONLY` | p4 and audited derived kinematics | clean universal repair |
| `TRACK_ONLY` | compatible charged-track observables | tracking contribution |
| `P4_PLUS_TRACK` | both above | best common-feature oracle |
| `DIRECTION_ONLY` | direction, fixed HLT momentum magnitude | angular contribution |
| `RESPONSE_ONLY` | magnitude/energy, fixed HLT direction | response contribution |
| `WRONG_DIRECTION` | equal-sized control displacement not aimed offline | generic perturbation control |
| `MATCH_SHUFFLED` | offline partners deterministically deranged | matching/null control |

Track repair is enabled only after a branch-semantics audit proves that the
HLT and offline stored quantities have compatible definitions. Discrete
charge, particle type, quality flags, lost-inner-hit counts, and similar
fields are never linearly interpolated.
The full family is gated by that real-data semantic audit and by complete
one-to-one assignment coverage.

`FULL_PARTICLE_ENDPOINT` is authorized by a distinct hash-chained
`full_endpoint` lock only when the matching-only selector passes, all five
particle categories pass their locked precision criteria, the selected full
matcher passes its own 99% precision lower bound, and a separate count-only
scan proves exactly complete assignments over every mapped train and
validation row. For each role, the signed report records the split-manifest
`expected_mapped_jets` and observed `scanned_mapped_jets`; authorization
requires exact equality. It also requires the visible and assigned category
counts to sum to their corresponding token totals and to agree category by
category. Thus perfect coverage over only a shortened prefix cannot authorize
repair. The bounded matching-validation samples inform precision and
selection but never authorize full-role completeness. The lock binds the exact
variant, threshold, split, fixed source-to-fold seed, five fold matcher hashes,
full matcher, validation, and full-role coverage hashes. Partial category
eligibility can support `P4_ONLY` ablations and matcher diagnostics, but can
never silently authorize the all-field endpoint.

### 7.4 No hidden match channel

The teacher receives neither an explicit match mask nor a confidence value.
Otherwise the pattern of matchability could itself become privileged input.

As a diagnostic, the report stratifies results by match coverage. A separate
matched-mask control may feed the mask to a nondeployable oracle to quantify
how informative matchability is, but that graph is never a teacher for the
primary student.

## 8. Teacher hierarchy

For each alpha, train a same-capacity ParT teacher from scratch:

```text
T_alpha : repaired HLT-skeleton view X_alpha -> 15 logits
```

The ordinary HLT teacher is:

```text
B_0 = T_0 : canonical HLT view X_0 -> 15 logits
```

Required teacher/oracle rows are:

| ID | Input | Role |
|---|---|---|
| `T0` | canonical HLT | ordinary baseline teacher |
| `T05` | `FULL_PARTICLE_ENDPOINT`, alpha 0.05 | weak privilege |
| `T10` | `FULL_PARTICLE_ENDPOINT`, alpha 0.10 | weak privilege |
| `T25` | `FULL_PARTICLE_ENDPOINT`, alpha 0.25 | intermediate privilege |
| `T50` | `FULL_PARTICLE_ENDPOINT`, alpha 0.50 | strong privilege |
| `T100` | `FULL_PARTICLE_ENDPOINT`, alpha 1.00 | complete HLT-count offline endpoint |
| `TOFF` | native offline constituents | unrestricted offline diagnostic |

All alpha teachers use the same architecture, initialization tensor, sampler,
optimizer, schedule, update budget, and checkpoint rule. `TOFF` uses the locked
native charged/neutral representation below and is never described as
capacity-matched unless parameter and compute parity are explicitly enforced.
`T0` and every alpha teacher use the identical train-only class-weight formula
defined in Section 13.1; privilege level never changes row weights or the
realized batch plan.

The primary `TOFF/v1` input contract is:

| Group | Features in exact order | Vectors | Length and padding |
|---|---|---|---|
| charged/lost-track | `pt_log_nopuppi`, `e_log_nopuppi`, `etarel`, `phirel`, `abseta`, `charge`, `isEl`, `isMu`, `isChargedHad`, `lostInnerHits`, `normchi2`, `quality`, `dz`, `dzsig`, `dxy`, `dxysig`, `btagEtaRel`, `btagPtRatio`, `btagPParRatio` | `px`, `py`, `pz`, `energy` | 90, constant zero padding and mask |
| neutral | `pt_log_nopuppi`, `e_log_nopuppi`, `etarel`, `phirel`, `abseta`, `isGamma`, `isNeutralHad` | `px`, `py`, `pz`, `energy` | 60, constant zero padding and mask |

Every name in the first row has prefix `cpfcandlt_`; every name in the second
has prefix `npfcand_`. Native stored order is retained independently in each
group. `TOFF/v1` includes the stored lost-track entries present in
`cpfcandlt_*`, but uses no `sv_*` group and no jet scalar. It is therefore named
the **native offline constituent oracle**, not a complete offline-detector
oracle. Any SV-enabled oracle is a separate `TOFF_SV` contract and experiment.

The documented donor YAML for this representation has SHA-256
`32da128ad0e82c6933a36cf409ae766523a116ae66df8d43153c671bc89f7c1d`.
Its feature normalization constants and bounds are vendored and authenticated
in the repository-local native-offline contract before `TOFF` training.

Teacher quality is characterized by:

- validation CE, accuracy, macro OVR AUC, and per-class metrics;
- KL divergence and class agreement with `T0`;
- examples where the alpha teacher corrects or breaks `T0`;
- gain versus alpha and match coverage;
- calibration and entropy;
- full-offline gap closure.

Monotonic teacher accuracy is not required. Non-monotonicity is a valid result.

## 9. Logit-distillation protocol

Let `S` be an HLT-only student, `B` the frozen ordinary HLT teacher, `T` the
frozen alpha teacher, and `y` the hard label. The primary loss is:

```text
L = lambda_CE   * CE(S, y)
  + lambda_HLT  * tau^2 * KL(q_B(tau) || q_S(tau))
  + lambda_PRIV * tau^2 * KL(q_T(tau) || q_S(tau))
```

where `q(tau) = softmax(logits / tau)`.

The primary mixture, motivated by the previously observed effectiveness of
25% CE plus 75% HLT-teacher KD, is:

```text
lambda_CE   = 0.25
lambda_HLT  = 0.60
lambda_PRIV = 0.15
```

The baseline KD temperature is first reproduced on this new dataset using the
predeclared grid `tau in {1, 2, 4}`. The same selected temperature is then used
for every primary alpha comparison. Temperature is selected on validation CE
with accuracy and earliest fixed experiment ID as exact tie breakers; no final
test output exists at this point.

Required loss controls at the selected alpha are:

| ID | CE | HLT KD | Privileged KD | Purpose |
|---|---:|---:|---:|---|
| `K0` | 1.00 | 0.00 | 0.00 | from-scratch CE |
| `K1` | 0.25 | 0.75 | 0.00 | established self-KD form |
| `K2` | 0.25 | 0.60 | 0.15 | primary dual-teacher mixture |
| `K3` | 0.25 | 0.50 | 0.25 | stronger privilege |
| `K4` | 0.25 | 0.00 | 0.75 | privileged teacher only |
| `K5` | 0.10 | 0.75 | 0.15 | original proposed mixture |
| `K6` | 0.25 | 0.60 | 0.15 full-offline | unattainable-teacher control |

The `K2` alpha sweep runs for every registered alpha teacher. Thus the
campaign does not choose an alpha using only teacher accuracy and then hide
the transfer curve.

The core derived quantity is the privilege recovery ratio:

```text
student gain over K1 / alpha-teacher gain over T0
```

It is reported for each metric with uncertainty and marked undefined when the
teacher denominator is zero or changes sign.

## 10. Representation and relation distillation

Representation KD is a second-stage mechanism study, not mixed into the first
logit result.

Because the teacher retains the HLT token skeleton, accepted tokens are
positionally aligned. Nevertheless, a teacher feature may encode unavailable
offline detail, so matching raw hidden states is not assumed beneficial.

Registered variants are:

| ID | Added target | Loss |
|---|---|---|
| `R0` | none | logits only |
| `R1` | final class token | normalized projected cosine/MSE |
| `R2` | pooled particle state | normalized projected cosine/MSE |
| `R3` | late particle block | masked learned projection |
| `R4_PAIR` | actual Weaver pair-geometry/attention-bias tensor | masked pairwise MSE after an identity-initialized 1x1 head projection |
| `R4_GRAM` | Gram matrix of normalized late particle states | masked pair-similarity MSE |
| `R5` | two selected late depths | weighted sum with separate projections |

Every representation arm has a matched-capacity control containing the same
projection modules with the privileged loss coefficient zero.

The v2 implementation freezes the normalized representation-loss coefficient
at `0.10` for every non-`R0` arm. This value and the corresponding zero-coefficient
matched-capacity controls are part of the training lock; changing it defines a
new registered training variant rather than a post-result adjustment.

Primary representation rules are:

- normalize teacher and student representations before comparison;
- use learned projections when dimensions or coordinate conventions differ;
- mask padding exactly;
- apply the primary token loss over all valid HLT token positions, so the
  training loss does not reveal which positions were matched;
- register accepted-match-only token loss as a separate training-time-mask
  ablation rather than silently using it;
- fit no normalizer on validation/test;
- retain logit KD in all promoted representation arms;
- report target predictability separately from classification utility.

The alpha selector is allowed to choose `alpha = 0`. In that case,
positive-coefficient representation KD exposes the exact same in-memory HLT
view under both `hlt` and `privileged` keys and retains the shared frozen T0
teacher. This is an identity alias, not a copied or reconstructed view: no
offline branch, matcher, assignment, or repaired input is read. A zero result
therefore remains scientifically meaningful and cannot terminate the
registered representation stage.

Layer-by-layer freezing is a registered exploratory arm only after the six
declared non-`R0` representation surfaces.
The preferred first implementation trains all student layers jointly. A
progressive arm may train late alignment, then progressively introduce earlier
targets, but it must have an equal-update, equal-capacity, no-freeze control.

## 11. Generational distillation

The generational hypothesis is tested for a fixed maximum of three transfer
generations. The campaign does not keep running until a visually pleasing
result appears.

Generation zero is:

```text
B_0 = ordinary trained HLT teacher
T_0 = selected alpha-repaired teacher
B_1 = from-scratch HLT student distilled from B_0 and T_0
```

For generation `g >= 1`, construct a privileged companion by initializing
from `B_g` and fine-tuning on the same locked alpha-repaired view:

```text
T_g = privileged fine-tune of B_g
B_(g+1) = from-scratch HLT student distilled from B_g and T_g
```

The privileged fine-tune uses CE plus an HLT-anchor KD term to prevent an
unbounded decision shift. This is not the primary teacher construction; it is
specific to the generational experiment.

Its frozen default is 25% class-weighted CE plus 75% KD to `B_g`, using the
selected campaign temperature, one quarter of the full student update budget,
and one tenth of the locked peak learning rate. Each `B_(g+1)` then uses the
primary 25% CE / 60% `B_g` KD / 15% `T_g` KD loss and the complete full budget.
Changing the anchor, fine-tune budget, or learning-rate fraction is a named
generational ablation, not an adaptive response to the observed gain.

Required generational controls are:

- self-distillation from `B_g` without `T_g`;
- repeated distillation from the fixed original `T_0`;
- same-compute CE training;
- privileged teacher trained from scratch rather than warm-started;
- identical student initialization and update budget across each comparison.

All registered generations `B_1`, `B_2`, and `B_3` run even if an earlier
generation is poor. If `B_3` still improves, extending the chain requires a
new campaign specification and cannot use final-test feedback.

If the locked alpha selector chooses zero, each companion `T_g` is initialized
from `B_g` and fine-tuned on the identical HLT endpoint with the same CE plus
anchor-KD recipe. `B_(g+1)` may then receive distinct HLT-domain logits from
`B_g` and `T_g`; both use the HLT identity alias for any aligned
representation surface. The companion and all later generations still run,
and no matcher or offline branch is introduced merely to avoid a zero-alpha
result.

## 12. Required mechanism and null controls

The following controls protect the scientific interpretation.

### 12.1 Matching controls

- greedy angular matching;
- likelihood Hungarian versus optimal transport;
- ultra-pure versus high-purity versus high-coverage thresholds;
- charged-only versus charged-plus-neutral matching;
- deterministic within-jet partner derangement;
- 0.5%, 1%, 2%, and 5% deliberate accepted-match corruption;
- removal of the densest jets from matcher fitting, then evaluation on them;
- match-model seed variation.

Corruption is identity-bound and applied after the primary match lock. At a
declared fraction of accepted edges, the offline endpoint is replaced by a
different compatible offline candidate from the same jet that is not already
assigned to another retained edge. Rows without such an alternative remain
uncorrupted and the achieved corruption fraction is reported. Selection is
independent of class and invariant to chunking. This preserves approximate
local kinematics while testing sensitivity to wrong identity; a separate
event-mixed corruption supplies the stronger null.

### 12.2 Repair controls

- `alpha = 0` exact identity;
- equal-magnitude random perturbation;
- wrong-direction perturbation;
- direction-only and response-only repair;
- p4-only versus track-only versus combined repair;
- fixed uniform alpha versus confidence-weighted alpha;
- Cartesian p4 interpolation versus audited log-response/angular
  interpolation.

For the equal-magnitude controls, displacement is measured in the exact
standardized repair-coordinate metric frozen from train only. Random and
wrong-direction controls preserve the empirical displacement norm by particle
type, alpha, and kinematic bin. They cannot use labels or validation/test
statistics. This prevents a weak null whose perturbations are simply smaller
than the offline-directed repair.

### 12.3 Teacher controls

- ordinary HLT teacher;
- same-capacity HLT ensemble teacher;
- full offline teacher;
- alpha teacher with shuffled offline partners;
- alpha teacher trained with identical compute but offline repair disabled;
- from-scratch versus HLT-checkpoint initialization.

### 12.4 Student controls

- CE only;
- ordinary HLT self-KD;
- privileged-only KD;
- dual-teacher KD;
- temperature and entropy-matched soft-label control;
- extra-parameter/no-representation-loss control;
- same-update and same-forward-compute controls where practicable.

No poor control result may cancel another registered row.

## 13. Training protocol

### 13.1 Sampling and class imbalance

The natural validation and final-test distributions are retained.

The primary train policy streams the natural selected train population without
permanent downsampling or oversampling. Every comparison receives identical
authenticated row cycles and batch plans. File order and within-chunk order
are deterministic functions of the epoch, rank, worker, and sampler seed.
Training round-robins chunks from eight source files and applies an
identity-preserving deterministic 32,768-row RAM shuffle buffer before model
batches are emitted. This mixes files while preserving the natural class
distribution. Validation and final-test streams remain ordered and unshuffled.

Train-only class counts `n_c` define square-root inverse-frequency weights:

```text
r_c     = 1 / sqrt(n_c)
kappa   = N_train / sum_c (n_c * r_c)
w_c     = kappa * r_c
```

Thus the natural-population mean example weight is exactly one. The per-row
weight `w_y` multiplies CE and every primary KD term before reduction; all
three loss components therefore optimize the same declared class emphasis.
Validation and final-test metrics are unweighted unless the metric itself is
explicitly macro-averaged.

Required sampling/weighting ablations are unweighted natural streaming and a
deterministic rotating-QCD-cap policy. They are mechanism controls, not silent
changes to individual arms. No pT/mass reweighting enters the primary campaign;
if studied later, its train-only histogram and mass-sculpting controls define a
new registered variant.

### 13.2 Optimization

Unless a Scouting miniature demonstrates a required dataset-specific change,
the campaign inherits:

- AdamW;
- update-based linear warmup for 5% of updates;
- cosine decay to 5% of peak learning rate;
- fixed update budgets;
- deterministic validation checkpoint selection by CE, then accuracy, then
  earliest optimizer update;
- eight approximately equidistant validation checks over the fixed update
  budget, including the final update, unless an explicitly locked miniature
  interval overrides the schedule;
- durable FP32 interval-mean training losses every 50 updates; unfinished
  intervals are included in exact-resume state and the completed compact
  history is retained in the immutable training report;
- rolling resume publication only at validation boundaries, explicit
  interruption, or Slurm preemption, rather than after every optimizer step;
  checkpointable Slurm tasks request `B:USR1@120`; the batch worker uses
  `exec` so Python becomes the signaled batch process and checkpoints at the
  next completed optimizer update;
- exact resume of model, optimizer, scheduler, scaler, sampler, epoch/update,
  worker/rank partition, and RNG state.

Screening and confirmation budgets are separate fixed constants. Their exact
values and batch size are set from throughput, memory, and learning-curve
measurements in a real miniature before the immutable campaign specification
is created. They may not be changed after comparison results are visible.

The budget-lock exercise is restricted to the CE HLT baseline and runs every
registered point in a small predeclared optimizer grid. The initial grid is:

```text
effective batch size   256, 512 where memory permits
peak learning rate     1e-4, 3e-4, 1e-3
training exposure      10 and 20 complete natural train-role passes
```

It chooses the lowest validation CE, then highest accuracy, then smaller
learning rate, batch size, exposure, and experiment ID. The resulting batch,
peak learning rate, and full comparison budget are frozen for every core
teacher and student. Broad mechanism screens may use a separately frozen
shorter budget, but any promoted arm is retrained from scratch at the full
budget, and results at unequal budgets are never treated as a paired model
comparison. Resource-only smoke runs may shorten updates but cannot influence
the scientific selector.

Training never terminates early for poor accuracy, poor matching coverage, or
failure to beat a baseline. Nonfinite required values, invalid inputs,
forbidden data access, source drift, corrupt lineage, or impossible masks fail
closed.

### 13.3 Seeds

Screening uses master seed `1337`. Confirmation uses master seeds
`11, 22, 33, 44, 55`. The baseline, ordinary self-KD, best logit PMARD arm,
best representation arm, selected generational arm, and every arm within the
predeclared validation promotion window receive all five confirmation seeds.

Each master seed expands through named domain separators into teacher,
student, sampler, matcher, corruption, dropout, and augmentation sub-seeds.
For a given master seed, `T0` and every alpha HLT-architecture teacher start
from the identical `teacher_initialization` tensor; `K1` and every PMARD
candidate use the same distinct `student_initialization` tensor. Thus the
privilege comparison is paired without accidentally initializing teacher and
student identically.
No screen checkpoint is relabeled as one of the confirmation seeds.

Seed derivation separately namespaces:

- model initialization;
- sampler and within-chunk shuffle;
- matcher training;
- synthetic matcher perturbations;
- corruption controls;
- dropout and augmentation;
- bootstrap statistics.

The paired metric bootstrap uses seed `8041`.

## 14. Staged experiment registry

### Stage A: source and capability audit

- authenticate all 53 ROOT files and projected branch schema;
- build and validate file-disjoint manifests;
- reproduce mapped class counts and baseline selection;
- audit raw physical validity, per-family jagged lengths, and category flags;
- lock weighted/unweighted p4 closure independently for HLT, offline charged,
  and offline neutral collections;
- lock lost-track decoding and combined offline index semantics;
- audit HLT/offline feature semantic compatibility;
- vendor and hash the HLT and `TOFF/v1` preprocessing contracts;
- measure I/O, RAM, GPU, and truncation behavior on a real miniature.

### Stage B: matcher development and lock

- build `M0`--`M5` on train only;
- run synthetic known-correspondence and native-source consistency validation;
- freeze match taxonomy, primary matcher, thresholds, and operating points;
- publish the matching report and immutable matcher lock;
- do not inspect teacher or classifier results before the lock.

### Stage C: canonical model baselines

- installed-Weaver FP32 parity for the 21-channel, 15-output ParT;
- CE baseline;
- baseline KD temperature screen;
- ordinary HLT self-KD;
- native offline oracle and optional HLT ensemble control.

### Stage D: teacher privilege ladder

- train all `FULL_PARTICLE_ENDPOINT` alpha teachers;
- report teacher gain, divergence, agreement, and match-coverage dependence;
- retain `P4_ONLY` as the kinematics-only mechanism ablation;
- retain every registered result regardless of quality.

### Stage E: primary logit transfer

- run `K2` for every alpha;
- run `K0`--`K6` at the predeclared selected alpha;
- calculate privilege recovery and compare against HLT self-KD and full
  offline KD;
- run deliberate matching-corruption sensitivity.

### Stage F: matching and repair mechanisms

- operating-point, particle-category, interpolation, wrong-direction,
  shuffled-match, direction-only, response-only, and track-feature controls;
- determine whether improvement follows offline direction rather than generic
  perturbation magnitude.

### Stage G: representation transfer

- run `R0`, `R1`, `R2`, `R3`, `R4_PAIR`, `R4_GRAM`, and `R5`, with a
  matched-capacity control for every non-`R0` arm;
- evaluate target fidelity and downstream classification separately;
- run progressive layer freezing only as the registered exploratory arm.

### Stage H: generations

- run `B_1`, `B_2`, and `B_3` plus same-generation controls;
- report gain, calibration, diversity, and teacher/student divergence at every
  generation;
- do not use final test to decide whether another generation would be useful.

### Stage I: five-seed confirmation

- run every predeclared promoted graph for five seeds;
- aggregate paired validation statistics;
- select finalist graphs using the frozen selector;
- create the finalist lock.

### Stage J: sealed final test

- create the execution lock and atomic execution claim;
- run HLT-only finalists once on final test;
- run teacher/oracle final-test diagnostics only in the same locked reporting
  wave;
- publish metrics, plots, uncertainty, and the interpretation table.

## 15. Selection rules

Checkpoint selection minimizes validation CE, then maximizes accuracy, then
chooses the earliest update.

Campaign finalist selection uses a versioned 15-class utility. The proposed
primary criterion is the macro mean over the 14 signal classes of log QCD
rejection at 50% signal efficiency, with exact tie/interpolation and
zero-background behavior frozen in tests.

For signal class `s`, its rejection curve ranks rows with the conditional
signal-versus-QCD discriminant
`P(s) / (P(s) + P(QCD))`; it does not use the class-independent `1-P(QCD)`.
QCD mass-sculpting diagnostics use the same per-signal discriminant.

Tie breakers are:

1. higher macro OVR AUC;
2. lower multiclass CE;
3. lower top-label ECE;
4. lower registered experiment ID.

The campaign may lock two finalists only if declared before confirmation:

- the utility finalist under the rule above;
- the lowest-CE HLT-only finalist.

No final-test model-derived output exists before both locks. Offline or alpha
teacher final-test performance is diagnostic and cannot reopen selection.

## 16. Metrics and statistical reporting

Required classifier metrics are:

- accuracy and multiclass CE;
- confusion matrix, per-class recall, and precision;
- per-class OVR AUC and macro AUC;
- per-signal QCD rejection at fixed efficiencies;
- macro mean log rejection;
- multiclass Brier score and 15-bin top-label ECE;
- metrics versus HLT and offline jet `pT` and mass;
- QCD mass sculpting;
- metrics versus HLT multiplicity, offline multiplicity, truncation, local
  density, match coverage, and match ambiguity;
- performance on QCD sublabels 309--313 as a diagnostic.

Required matching and repair metrics are:

- accepted fraction by category and operating point;
- ambiguity, dustbin, and solver-disagreement rates;
- synthetic precision/recall/calibration;
- physical residual distributions before and after repair;
- view displacement norm versus alpha;
- match corruption sensitivity;
- alpha-teacher divergence and class agreement with `T0`.

Required KD metrics are:

- student improvement over CE and ordinary HLT self-KD;
- teacher improvement over `T0`;
- privilege recovery ratio;
- student/teacher KL and agreement;
- teacher-correction recovery: cases where the privileged teacher fixes a
  baseline error and the student follows;
- teacher-break inheritance: cases where privilege breaks a correct baseline
  prediction and the student follows;
- seed-wise and paired-jet uncertainty.

Statistics use paired jet rows. Bootstrap resampling is stratified by the 15
classes with a frozen seed and exact quantile definition. Five-seed reports
show individual seeds, mean, standard deviation, and a hierarchical or
explicit seed-aware paired comparison. A result is not reduced to a single
best seed.

The final primary interval uses 10,000 deterministic nested bootstrap
replicates with seed `8041`:

1. resample the five confirmation master seeds with replacement;
2. within each sampled seed, resample paired jet identities independently
   within each of the 15 true classes while preserving observed class counts;
3. use the same resampled identities for both compared models;
4. calculate the seed-mean paired metric difference;
5. report NumPy linear 2.5% and 97.5% quantiles.

This is explicitly a **paired-row-level** interval conditional on the stored MC
sample. Because the files do not provide a globally unique physical-event key,
it is not described as an event-level sampling interval. The final report also
includes a source-file delete-one sensitivity and a deterministic
within-source-file contiguous-entry block bootstrap. Those diagnostics probe
file and local ordering dependence but do not manufacture a claim of known
physical-event clustering; any material disagreement is reported as a
dependence limitation.

Positive primary evidence requires the 95% interval for PMARD minus `K1` macro
mean log rejection to lie above zero and the interval for PMARD minus `K1` CE
to lie below zero. Failure of either condition is reported as inconclusive or
negative primary evidence even if a secondary metric improves. If two
deployable finalists are locked, their two primary comparisons use
Bonferroni-adjusted 97.5% two-sided percentile intervals to control the
two-finalist family at `alpha = 0.05`; ordinary 95% effects and intervals are
also shown transparently. Those adjusted endpoints are the linear 1.25% and
98.75% bootstrap quantiles. Requiring both co-primary outcomes to have the
declared direction is an intersection rule and is not relaxed by a strong
result on only one outcome.

This inferential rule is separate from the deterministic validation selector.
It cannot alter which predictions are generated after final-test execution
begins.

## 17. Streaming and minimum-durable-storage design

### 17.1 Source reads

ROOT remains the only authoritative dataset representation. Each worker:

- reads only projected branches needed by its task;
- round-robins bounded chunks across at most eight open source files for
  training and reads one file at a time for ordered evaluation;
- uses Awkward jagged arrays until batch construction;
- partitions files across distributed ranks and DataLoader workers;
- begins with two reader workers and tunes only from measured evidence;
- never enables whole-dataset `--in-memory` loading;
- never writes a padded full-dataset tensor cache.

The starting raw read chunk is 4,096 rows, the deterministic train shuffle
buffer is 32,768 selected rows, and the starting model batch is 512, subject
to miniature measurement. Increasing readers is not assumed to improve
throughput because it may multiply memory and random I/O.

### 17.2 Ephemeral match table

The matcher scans a role once at job start and writes a compact, row-ordered
assignment table to process shared memory or a validated RAM-backed directory
such as `$SLURM_TMPDIR` when it is RAM-backed, otherwise `/dev/shm`.

The v2 implementation uses a process-local NumPy `int16` table and a cached
identity index, both authenticated by their headers. Production full-role jobs
build it once and reuse it across epochs. Bounded `--max-rows-per-role` smoke
runs deliberately retain deterministic online matching. The sole exception is
the registered `ram_cache_miniature`, which builds a bounded assignment table
and bounded teacher-logit tables to exercise the production cache/join path
without scanning the full role. `CONFIDENCE_WEIGHTED` also remains online
because an assignment-only table cannot reproduce its calibrated confidence.

For a length-200 HLT sequence, a signed 16-bit offline index with `-1` as the
unmatched sentinel requires approximately:

```text
4.635 million rows * 200 tokens * 2 bytes = 1.85 GB
```

The train role alone is smaller. A quantized confidence byte, if required for
diagnostics, adds less than 1 GB over the complete selected population. The
primary fixed-threshold repair needs only the hard index table; aggregate
confidence diagnostics are published separately.

The ephemeral table binds source-file hash, entry range, split, matcher lock,
threshold, cross-fit fold or full-train inference model, feature contract, and
content hash. Its row table is indexed by canonical paired-row identity, not
by incidental batch arrival order. It is validated before use and deleted on
worker exit. It is never treated as a deployable artifact.

If shared RAM is unavailable, matching is recomputed deterministically per
chunk rather than written to durable campaign storage.

### 17.3 On-the-fly repaired views

Alpha views are constructed in the collating process from:

- current streamed HLT/offline raw arrays;
- the ephemeral compact assignment;
- the immutable repair configuration.

Only the current bounded batch is materialized as padded tensors. Multiple
alphas may be generated from the same raw batch during teacher inference, but
no repaired ROOT, NPZ, Parquet, or full padded tensor dataset is published.

### 17.4 Ephemeral teacher targets

Logit KD does not require durable teacher-output caches. At student-job start,
frozen teachers stream over the training role and place identity-keyed FP32
logits in process RAM with a content-authenticated header.

For 4.635 million rows and 15 FP32 logits, one complete table is about 278 MB
before metadata. Two teacher tables remain modest relative to available host
RAM. CE, temperature softmax, KL, and representation losses are evaluated in
FP32 even while ParT forward/backward uses BF16 autocast.

Late 128-dimensional float16 embeddings would require about 1.19 GB over the
complete population and remain feasible in RAM, but representation targets
may instead be computed online to avoid retaining unnecessary roles.

Teacher-target RAM is cleared at job exit. No training-role teacher logits or
embeddings are durably stored.

Teacher forward computation and RAM target storage are FP32-authoritative.
CE, KL, recursive representation structures, and pair/Gram construction and
reductions are FP32-authoritative even when their source ParT activations were
created under BF16 autocast.
Teacher targets are joined to shuffled training batches by
canonical row identity; positional coincidence with a previous stream order
is never accepted as a join.

Once identity-keyed logits exist, ordinary `R0` students and matched
zero-coefficient representation controls train from the HLT-only stream. They
do not reread offline branches, decode offline particles, rebuild repair
views, or retain the now-unused teachers/matchers on GPU. Only a non-`R0` arm
with a strictly positive representation coefficient retains the aligned
privileged stream and privileged teacher for per-update representation
targets. This changes execution cost only; the cached logits, HLT batches,
losses, row identities, and scientific budgets are unchanged.

### 17.5 Durable artifacts

Durable storage is limited to:

- split manifests and source verification reports;
- matcher configuration, small matcher weights, lock, and aggregate report;
- train-only normalization and sampling statistics;
- immutable campaign specifications and source snapshots;
- one rolling exact-resume checkpoint per active training job;
- compact selected model-only checkpoints;
- validation reports and compact validation logits when required for paired
  selection;
- finalist/execution locks;
- final-test logits for locked graphs;
- aggregate metrics, plots, and task/resource attestations.

Dense pair tensors, per-epoch representations, repaired datasets, persistent
assignment matrices, and duplicated source arrays are forbidden.

After a screening selector authenticates all reports, unpromoted screening
checkpoints may be removed by an exact campaign-bound pruning task if and only
if that retention policy was declared in the campaign specification. Their
configuration, source, result, and former checkpoint digest remain in the
report; they are then explicitly non-reusable. Promoted and finalist
model-only checkpoints remain durable.

A successful completed run deletes its rolling optimizer/RNG checkpoint only
after the immutable report and required model-only checkpoint authenticate.
An interrupted run retains exactly one rolling checkpoint for exact resume.

### 17.6 Storage and RAM gates

Before any full submission, a real miniature measures:

- peak durable storage;
- peak `/dev/shm` or `$SLURM_TMPDIR` usage;
- host RSS per worker and rank;
- GPU memory for student plus frozen teachers;
- ROOT throughput and wait fraction;
- size of every checkpoint and prediction artifact;
- cost of matcher construction versus reuse inside one job.

The count-only full-role matcher audit is intentionally not a bounded
miniature: both the smoke and production DAGs scan the complete mapped train
and validation roles. It therefore has an independent 48-hour GPU request
instead of inheriting the one-hour smoke GPU default. No bounded diagnostic
may produce the authorization artifact. The smoke resource evidence must
record this task's elapsed time, RSS, GPU memory, and ROOT I/O; that evidence
is reviewed before freezing the production request.

Production requests use those measurements plus a declared safety factor.
Large RAM availability is used to reduce durable duplication, not to load the
entire decompressed dataset into every worker.

### 17.7 Tigris worker contract

PMARD inherits the repository compute rules:

```text
account        reu-aisocial
partition      tigris
project        /home/ryreu/atlas/HLT_Classification
data           /home/ryreu/cms/data/ScoutingAK8_native_compact/2024/train
environment    atlas_kd_tigris
```

Every worker sets `PYTHONNOUSERSITE=1`, activates the declared environment,
prepends `${CONDA_PREFIX}/lib` to `LD_LIBRARY_PATH`, and sources helpers from
the absolute `${PROJECT_DIR}` path. It never resolves helpers relative to a
Slurm spool copy. Dataset dependencies are installed and recorded in the
declared environment rather than importing an unversioned checkout from
`/home/ryreu/cms`.

The initial Scouting Weaver target is the usage guide's pinned development
revision `6a084ec341f5af2e06f6d884e6a4ee4dab8bb4e4`. If the existing environment
cannot host that revision, the replacement exact revision and compatibility
evidence must be locked before model results; a moving branch is forbidden.

Submission uses `sbatch --parsable`, exact numeric dependencies, campaign-local
ledgers, and exact campaign-bound cancellation IDs. Full production requires
explicit authorization, a complete dry run, measured resources/storage, exact
pushed source, and a genuine streamed miniature.

## 18. Reproducibility and lineage

New versioned contracts are required for:

```text
Scouting source schema and selection
file-disjoint split manifest
21-channel HLT ParT input
native offline input
matcher training configuration and matcher lock
compact ephemeral assignment table
alpha-repair view
15-class ParT model
PMARD teacher/student checkpoint and training report
PMARD prediction and evaluation report
campaign specification
finalist and final-test locks

Current scientific surfaces introduced by the matcher and endpoint reviews:
hlt_classification_scouting_schema_v3
hlt_classification_scouting_feature_audit_v2
hlt_classification_pmard_ephemeral_assignment_v2
hlt_classification_pmard_matcher_report_v2
hlt_classification_pmard_matcher_validation_v2
hlt_classification_pmard_full_role_coverage_v2
hlt_classification_pmard_training_report_v4
hlt_classification_pmard_resume_checkpoint_v4
hlt_classification_pmard_lock_v4
hlt_classification_pmard_campaign_spec_v8
```

Every reusable artifact records its content hash, parent hashes, exact source
snapshot, dataset verification identity, split identity, branch and feature
order, class order, matcher/repair version, model configuration, optimizer
budget, and environment.

Scientific identities never use timestamps. Path existence alone never
authorizes reuse. Publication is atomic and consumers revalidate hashes and
parents.

### 18.1 Pre-result freeze sequence

This plan intentionally leaves hardware-dependent numbers to a measured
miniature, but it does not leave them available for post-result tuning. Locks
occur in this order:

1. **Data lock:** source inventory, checksums, split files, row identity,
   branch allowlists, feature/channel map, truncation, labels, and observers.
2. **Matcher-design lock:** candidate gates, folds, pseudo-label rules,
   architecture grid, solvers, score canonicalization, thresholds, validation
   metrics, selector, and matching seeds. This precedes the matching report.
3. **Matcher-result lock:** selected matcher, full out-of-fold assignments
   recipe, operating points, category eligibility, and repair compatibility.
   This precedes every teacher result.
4. **Full-role coverage audit and full-endpoint authorization lock:** scan all
   mapped train rows with their locked out-of-fold matchers and all mapped
   validation rows with the full matcher. Publish only counts and hashes, then
   require exact split-declared versus scanned jet totals, token/category
   conservation, selected execution lineage, all five eligible categories,
   full-matcher precision, and complete assignments. No assignment dataset is
   published. This lock must exist before any full-endpoint teacher.
5. **Training lock:** installed-Weaver model parity, class weights, optimizer
   grid outcome, batch size, update budgets, temperatures, loss coefficients,
   checkpoint selector, experiment registry, and model seeds. This precedes
   alpha-teacher and student comparisons.
6. **Screen/confirmation lock:** promotion window and full-budget confirmation
   rows. Existing registered rows are never cancelled because of poor results.
7. **Finalist lock:** exact HLT-only graphs, checkpoints, seeds, metrics, null
   control, and diagnostic oracles permitted in the final wave.
8. **Execution lock and claim:** exact committed source, environment, job DAG,
   final-test manifest, and authorized one-time execution.

Every lock records the complete prior lock as a hashed parent. A later change
creates a new campaign identity and cannot overwrite or masquerade as the
earlier campaign. Hardware/resource adaptation after the miniature may change
Slurm requests and bounded read sizes without changing row order, model math,
sampling, or scientific budgets; any change that affects those semantics
requires a new training lock.

## 19. Deployability and leakage audit

The exported student signature is exactly:

```text
HLT 21-channel features, HLT four-vectors, HLT mask -> 15 logits
```

The export/audit must prove that the graph, checkpoint, and inference loader
cannot access:

- offline particle arrays or jet scalars;
- match indices, confidences, residuals, or masks;
- alpha-repaired values;
- teacher logits or representations;
- labels or generator/truth fields;
- sample flags or existing classifier outputs;
- source row, file name, or construction metadata as model inputs.

All compared deployable graphs consume the identical authenticated HLT view.
Teacher and oracle graphs are explicitly nondeployable.

## 20. Required implementation surfaces

Reusable semantics belong under `src/hlt_classification/`, for example:

```text
src/hlt_classification/scouting/
    schema.py
    identity.py
    splits.py
    streaming.py
    inputs.py
    matching.py
    match_model.py
    assignment.py
    repair.py
    contracts.py
    campaign.py
    training.py
    evaluation.py

src/hlt_classification/models/
    scouting_particle_transformer.py
```

CLIs remain thin, configuration-driven wrappers. Expected commands include:

```text
validate_scouting_data.py
build_scouting_splits.py
audit_scouting_features.py
train_scouting_matcher.py
validate_scouting_matcher.py
train_pmard_teacher.py
train_pmard_student.py
evaluate_pmard.py
create_pmard_campaign.py
submit_pmard_campaign.py
monitor_pmard_campaign.py
resume_pmard_campaign.py
```

The existing `scouting_ak8_tools` package is a documented external operational
reference. Any migrated donor file or adapted logic must be recorded with its
exact donor path and commit in `docs/LEGACY_SOURCE_MAP.md`. PMARD runtime code
must not depend on an unversioned external checkout.

## 21. Required tests

### 21.1 Data and streaming

- exact 15-class definitions and class order;
- baseline selection and the 2,988 unmapped-row behavior;
- exact 53-file, 297-branch, raw/selected/mapped population facts and source
  digests;
- logical `tree` selection without duplicate ROOT key-cycle enumeration;
- file-disjoint split construction and manifest authentication;
- projected branch reads and bounded jagged chunks;
- bounded eight-file round-robin reads and deterministic cross-file RAM-buffer
  shuffling without changing the natural class population;
- rank/worker file partition without duplication;
- deterministic row identity from normalized file and entry;
- all four native collection length invariants;
- combined offline index encoding, decoding, bounds, and native-index inverse;
- exact first-200 HLT visibility, mask, padding, and truncation behavior;
- byte-identical canonical-HLT versus `alpha=0` tensor construction;
- tolerance-bounded FP32 preprocessing parity against the pinned CMSSW
  reference;
- native-offline `TOFF/v1` feature order, group lengths, masks, and no-SV rule;
- forbidden-input and observer separation.

### 21.2 Matching

- wrapped-angle and four-vector residual calculations;
- authenticated HLT and offline p4/pT/weight closure rules;
- type/charge compatibility and lost-track policy;
- dustbin and unequal-cardinality behavior;
- global exclusivity and deterministic ties;
- Hungarian and optimal-transport toy solutions;
- calibration and abstention thresholds;
- split/merge ambiguity handling;
- event-mixing negatives;
- synthetic known-correspondence precision/recall;
- chunk, batch, worker, and rank invariance;
- no label or final-role fit access;
- ephemeral assignment hash and cleanup.

### 21.3 Repair

- exact `alpha = 0` HLT identity;
- exact all-21-channel `alpha = 1` offline endpoints;
- physical and finite interpolated four-vectors;
- unchanged HLT count, order, and mask;
- complete one-to-one assignment enforcement with no HLT fallback;
- atomic, nested PID/charge, quality, hit-state, validity, and applicability switches;
- approved derived-feature recomputation;
- no guessed feature transformations;
- consistent pair inputs;
- deterministic alpha values independent of chunking;
- no match metadata in model inputs.

### 21.4 Model and KD

- installed-Weaver FP32 parity for 21 features and 15 outputs;
- HLT-only student signature;
- frozen teacher behavior;
- exact CE/KD coefficient and temperature semantics;
- hard-label, HLT-teacher, and privileged-teacher gradient paths;
- representation masking and projection controls;
- zero privileged coefficient parity;
- exact uninterrupted/resumed training;
- nonfinite fail-closed behavior;
- poor-performance completion.

### 21.5 Campaign and storage

- source/split/matcher/repair/checkpoint lineage rejection;
- no durable repaired data or train teacher-output cache;
- validated RAM-root location outside project and data roots;
- cleanup only of exact job-owned ephemeral paths;
- rolling-checkpoint retention and post-success removal;
- screening-checkpoint pruning only after authenticated selection;
- dry-run nonmutation;
- exact Slurm IDs, dependencies, monitor, resume, and cancellation;
- final-test locks and atomic execution claim.

### 21.6 Real acceptance

Before production:

1. validate the native source dataset and exact split manifests;
2. run the matcher on real paired rows and publish its report;
3. pass installed-Weaver FP32 model parity;
4. complete a real GPU forward/backward with student plus frozen teachers;
5. complete a streamed miniature through matching, repair, teacher, student,
   validation, exact resume, and reporting;
6. measure resources and storage;
7. dry-run the complete production DAG against exact committed source.

No full campaign is authorized merely because synthetic tests pass.

## 22. Interpretation table

| Outcome | Interpretation |
|---|---|
| Intermediate alpha teacher improves and student recovers gain | supports teachable bounded privilege |
| Teacher improves monotonically but student peaks at intermediate alpha | supports a transferability frontier |
| Full offline teacher is best but transfers poorly | supports the original unattainable-target concern |
| Alpha teacher improves but dual KD does not beat self-KD | privilege is useful to teacher but not recoverable from HLT |
| Wrong-direction/random repair performs similarly | gain is likely generic augmentation or regularization |
| Shuffled matches perform similarly | claimed constituent correspondence mechanism is unsupported |
| Charged-only works and neutral-inclusive degrades | neutral matching is the limiting component |
| Representation KD adds gain | aligned late structure carries information beyond logits |
| Representation KD hurts | privileged internal state is less attainable than outputs |
| Generations improve then plateau | bounded iterative self-distillation is supported |
| Generations drift or regress | compounding confirmation bias dominates |
| No arm beats HLT self-KD | bounded offline repair provides no additional deployable value |
| Matching cannot reach required purity | constituent-level repair claim is not presently supportable |

Every row is a valid scientific result. None is an execution failure.

## 23. Optimal initial production shortlist

The full plan defines all controls, but the first scientifically complete
shortlist should remain bounded:

1. CE HLT ParT baseline.
2. Ordinary 25% CE / 75% HLT self-KD.
3. `FULL_PARTICLE_ENDPOINT` teachers at all six alpha values, with `P4_ONLY`
   retained as the kinematics-only mechanism ablation.
4. Primary 25% CE / 60% HLT KD / 15% privileged KD at all six alphas.
5. Privileged-only KD and full-offline KD at the selected intermediate alpha.
6. Ultra-pure charged-only and charged-plus-neutral matcher operating points.
7. Wrong-direction, shuffled-match, and 1% match-corruption controls.
8. Final-class-token and late-relation representation KD.
9. Three fixed generations plus self-KD controls.
10. Five-seed confirmation of the baseline, self-KD, best logit arm, best
    representation arm, and best generation under the frozen selector.

This shortlist directly answers the central claim while keeping model count,
durable checkpoints, and final predictions controlled. Broader ablations are
configuration rows, not separate copied pipelines.

## 24. Definition of success

### Scientific success

Minimum scientific success is a trustworthy matching report plus a complete
teacher/student alpha-transfer curve, even if no deployable improvement is
found.

Positive mechanism evidence requires all of:

- the alpha teacher beats its identical HLT teacher across confirmation
  seeds;
- the HLT-only dual-KD student beats ordinary HLT self-KD under paired
  validation statistics;
- offline-directed repair beats equal-magnitude wrong-direction and shuffled
  controls;
- the result survives the locked high-purity matcher and deliberate small
  corruption tests;
- the exported student passes the HLT-only deployability audit.

Generational success is secondary and requires improvement beyond the best
single-generation student under equal student compute and five-seed evidence.

### Production readiness

PMARD is production-ready only after:

- new contracts and implementations pass focused local tests;
- native-source MC and split audits pass;
- the matcher lock and independent validation report exist;
- installed-Weaver parity passes;
- a genuine minimum-storage Tigris miniature completes;
- RAM, GPU, I/O, wall-time, and durable-storage use are measured;
- exact committed source and immutable campaign specification exist;
- the full submission dry run passes;
- explicit full-campaign authorization is given.

Until then, the repository implementation is not production-ready evidence and
does not establish that PMARD's scientific hypothesis is true. The local code,
contracts, synthetic tests, and execution preparation are implementation
evidence only; the real acceptance items above remain mandatory.
