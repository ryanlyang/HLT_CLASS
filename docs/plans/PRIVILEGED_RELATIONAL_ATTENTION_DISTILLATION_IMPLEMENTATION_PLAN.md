# Privileged Relational Attention Distillation Implementation Plan

Status: active implementation plan for the PRAD campaign.

This plan binds the user-supplied execution specification
[`prad_implementation_agent_prompt.md`](../../prad_implementation_agent_prompt.md)
and scientific proposal
`privileged_relational_attention_proposal(1).md`
to the standalone repository contracts. The source documents remain the
scientific specification; this document records repository-specific facts and
the fail-closed adaptations required to implement them without inventing data.

## 1. Objective

Train an offline Particle Transformer whose learned pair relation affects its
classification attention, distill that relation into an HLT-only student, and
test whether feeding the predicted relation into later HLT attention improves
jet tagging beyond ordinary supervision, logit KD, auxiliary supervision, and
matched-capacity controls.

Deployable inference is exactly:

```text
authenticated HLT tokens -> PRAD student -> ten-class logits
```

Offline particles, matches, targets, teacher outputs, labels, and target masks
are training-only and must be unreachable from the exported runtime graph.

## 2. Current-source capability audit

The canonical JetClass source exposes one jet row with:

```text
particle px, py, pz, energy
particle charge
five particle-category indicators
d0, d0err, dz, dzerr
ten one-hot jet labels
```

It does not expose:

```text
physical event identifier
particle or track identifier
direct HLT-to-offline association
reconstructed vertex assignment
original jet-radius metadata
truth ancestry or decay-branch identity
```

Consequences:

- jet identity is the strongest available split and pairing key;
- physical-event overlap cannot be tested and is reported as unavailable;
- the vertex objective is registered with coefficient zero for this source;
- HLT-to-offline association uses the specified deterministic Hungarian
  fallback and is stored only in a training-only artifact;
- exclusive C/A targets operate on the already selected jet constituents.
  The source jet radius is recorded as unavailable; no radius is guessed.

The synthetic HLT-v3 generator may retain construction indices internally for
diagnostics, but PRAD matching must not read or serialize them. This keeps the
matching study meaningful and prevents a construction shortcut.

## 3. PRAD split contract

PRAD defines a new three-role split contract rather than reusing the baseline
five-role population:

```text
train  500,000
val    150,000
test   500,000
seed   1337
```

Each role is capacity-aware proportionally class-stratified against the source
inventory, sampled without replacement, globally shuffled,
and bound to canonical jet identities. Exact counts, class counts, zero
identity/location overlap, file inventory, schema, and manifest checksums are
authenticated. Raw input moments and semantic positive weights are fit and
recorded from `train` only. The model retains the canonical Weaver fixed
feature transformations instead of substituting a newly data-normalized input,
because baseline parity is a required control; this adaptation is explicit in
the statistics artifact.

The test role maps to the repository's sealed `final_test` capability. It may
be cached before selection only as input preparation; model-derived test output
requires finalist and execution locks.

## 4. Required scientific adaptations

The following repository invariants override execution shortcuts without
changing the central PRAD mechanism:

1. Oracle performance is diagnostic. It never cancels E3--E10 or causes a job
   failure; the prompt's percentage bands guide interpretation and optional
   variant priority only.
2. Training uses a fixed declared maximum budget. Validation selects a
   checkpoint but does not terminate registered work early.
3. As required by E10, shuffled-relation controls derange teacher relation
   targets only among the jets in each realized training batch without
   preserving class. The derangement is identity-bound and deterministically
   seeded by run, epoch, and batch-plan position so checkpoint resume is exact;
   hard labels and the E8 semantic targets remain unchanged.
4. Final-test inference remains sealed until graph selection and execution
   locks exist. Baseline and teacher test predictions are produced only in the
   locked final wave.
5. Poor tagging, relation fidelity, calibration, or oracle headroom remains a
   successful scientific result.

These adaptations are mandatory for consistency with `AGENTS.md` and the
versioned reusable experiment contract.

## 5. Model contract

The canonical base remains the repository's Weaver standard-four Particle
Transformer. A parity-safe split-forward adapter exposes particle state after
the configured contextualization depth and resumes the ordinary blocks and
class-attention path.

Default PRAD graph:

```text
context depth              2 particle blocks
relation input             h_i+h_j, abs(h_i-h_j), h_i*h_j,
                           standard-four pair features, symmetric scalar pairs
relation MLP               input -> 256 -> 128 -> 16
activation/dropout         GELU / 0.1 after both hidden layers
relation normalization     LayerNorm(16)
bias projection            3*tanh(Linear(16, attention_heads))
bias centering             valid keys per query and head
injection                   every particle block after context depth
gate                        tanh(raw), separate by layer and head, raw=0
ordinary ParT pair bias     retained
```

With gates zero, FP32 evaluation logits, masks, feature-input gradients,
shared parameter gradients, and shared state tensors must match the canonical
baseline under their registered tolerances. Installed Weaver's standard-four
pair construction has nonfinite auxiliary Lorentz-input derivatives even for
all-valid inputs; training does not differentiate those inputs, so parity
requires exact finite/NaN/+Inf/-Inf topology and matching finite entries for
that diagnostic surface instead of requiring it to be finite. All outputs and
gradients required by training remain strictly finite. The deployable student
forward accepts only canonical HLT Particle Transformer inputs.

## 6. Training-only targets

- frozen teacher relation bottleneck `r_T`;
- frozen centered teacher bias `B_T`;
- frozen teacher class logits and true-class confidence;
- exclusive C/A same-cluster labels for K=2,3,4;
- common-vertex label only under a future schema that authenticates vertex
  assignments;
- deterministic Hungarian HLT-to-offline match and derived pair mask.

The production campaign uses the versioned
`prad_minimum_durable_storage_v1` profile. Paired offline/HLT views, compact
assignments, cluster IDs, and teacher jet outputs are rebuilt under
`$SLURM_TMPDIR` (falling back to `/dev/shm`) for the task that consumes them
and are deleted when that task exits. The worker rejects a temporary root
inside the project or campaign tree. Standalone CLIs also support direct
process-memory reconstruction. None of these large arrays consume durable
campaign quota.

Only the split, train-only statistics, reports, compact best/final model
checkpoints, locks, and compact predictions survive successful tasks. A full
optimizer/RNG checkpoint is a single rolling `last.pt` per active run so a
preempted job remains exactly resumable; it is deleted after its immutable
training report and both required model-only checkpoints are published.
Prediction shards store only ten float32 logits. Their stable identities are
reconstructed from authenticated split row ranges rather than duplicated in
every file.

The teacher-output CLI retains an explicit `--dense-pairs` diagnostic mode,
but the campaign never enables it. Dense relation and centered-bias tensors
are recomputed in bounded batches.

## 7. Experiment registry

The immutable core registry contains E0--E10 exactly as specified. V1--V10 are
configuration variants of the same model/trainer, not copied scripts. All rows
predeclare role, allowed data capabilities, losses, gate behavior, initialization,
seed, budget, and selection eligibility.

Broad screening uses one registered seed. Confirmation uses seeds
`11,22,33,44,55` for the baseline, full PRAD, PRAD without KD, the selected
variant, and every graph within the predeclared one-percent validation window.

## 8. Metrics and selection

PRAD adds a campaign-versioned validation metric implementing per-class OVR
background rejection at 50% signal efficiency with the prompt's declared ROC
interpolation and macro mean log rejection. Its exact threshold/interpolation,
zero-background, tie, and absent-class behavior must be frozen in tests before
results are inspected.

Required reports also retain the repository metrics so PRAD remains comparable
with the canonical baseline campaign. Final statistical reporting is paired by
canonical identity and seed and uses authenticated test predictions only.

## 9. Implementation order

1. contracts, split compiler, capability/data audit;
2. Hungarian matching and compact matching cache;
3. exclusive C/A targets and train-only normalizers/weights;
4. relation module, centered bias, gates, and parity-safe model extension;
5. teacher cache, losses, fixed-budget staged trainer, exact resume;
6. configuration-driven E0--E10 and V1--V10 registry;
7. validation metrics, selection, final locks, plotting, and reporting;
8. thin CLIs and Tigris Slurm DAG;
9. local test ladder, installed-Weaver parity and PRAD runtime mechanics, real
   miniature, then authorized full campaign.

## 10. Production readiness

Implementation files and synthetic tests are not campaign completion. Full
acceptance additionally requires real JetClass capacity, a genuine Tigris
miniature through the production workers, measured storage/resources, exact
committed source, all registered full-data rows, five-seed confirmation, and
the sealed 500k final evaluation.

The minimum-storage estimator covers the deliberately uncapped one-percent
confirmation rule and therefore assumes every eligible graph advances. Its
current conservative durable peak is about 29 GiB, down from the disk-backed
cache design's roughly 119 GiB. The ordinary case should be smaller, but only
the new real miniature may replace this bound with measured evidence.
