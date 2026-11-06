# Next supervision: Group A science v1, execution v2

Authority: [implementation plan](../plans/LUKA_SEVEN_CANDIDATE_IMPLEMENTATION_PLAN.md)
and [candidate catalog](../plans/LUKA_NEXT_SUPERVISION_CAMPAIGN_IDEAS.md).
Implementation is additive under `luka_next_supervision_hydra`, based on
`ddcdd5fa22ce2455b12a62d61a926caa07969159`. New envelopes are
`LUKA_NEXT_SUPERVISION_{KIND}/v1`, schema1, canonical content/parent hashes and
the existing ALLJETS exploratory flags. Execution v2 versions SPEC,
RESOURCE_CHECK and ACCEPTANCE as `/v2`, schema2; the scientific RECIPE, raw
transfer, normalization, targets and other envelopes stay v1. Failed execution
v1 artifacts must be read in their pinned original source; they are not
accepted as a v2 setup. Old contracts are unchanged.

This is the A-scope contract. The shared implementation additionally supports
the separately registered [Group B v1 scope](LUKA_NEXT_SUPERVISION_GROUP_B.md).
That extension does not rewrite the original A recipe, specifications or
running worktrees. The separately registered
[Group C v1 scope](LUKA_NEXT_SUPERVISION_GROUP_C.md) now implements D/S;
additive A/B/C authorization does not release a new combined full40 execution.

## Scientific registration

The registry preserves all40 planned row IDs and their order. Only22 Group A
configurations and C0/C1 are executable. Groups B/C and full40 release raise
explicit unsupported-scope errors. A separately authorized Group A release
filters the global order; it never claims coverage of all seven candidates.

Every student is native, fresh HLT ParT8x128,17 inputs/11 classes, batch128,
30 full epochs, original AdamW/LR/selection and CE+5*original U100 SmoothL1.
Native/classifier/U100-head initialization and model RNG remain unchanged.
Seed1 uses the particle-teacher campaign's offset1000003 for initialization and
sampler; the original U100 predictor keeps seed20261019 in both replicates.
Auxiliary heads initialize inside an isolated CPU RNG context. New losses ramp
linearly over three epochs. All heads disappear from selected native exports.

J5 uses the original dot-product head seed2026103101,1024 original OFFLINE mix
negatives, cosine temperature0.1 and original CE reduction. J's candidate streams
delegate to SAMPLER/v1, including root seed2026103102+replicate and identity ties.
J6 keeps exactly those OFFLINE-mined candidate identities, but scores frozen
M0HLT keys. Learned J uses two normalized64D projections and a joint MLP; no
learned temperature, division by candidate count, T-squared or logit KD is added.

P/G use the full native128D jet embedding and detached measured candidates.
Particle heads are independent MLPs with LayerNorm64. G averages encoded real
members and symmetric pair descriptors; singleton relation output is zero.
G5 averages independent member logits on **exactly the same sampled groups**.
The balanced loss is one half mean-positive BCE plus one half mean-negative BCE,
then mean over jets. Sigmoids are not calibrated source-membership posteriors.

## Raw transfer and descriptors

RAW_TRANSFER authorizes **TRAIN only**,299234 paired rows, physical p4, charge,
six-way category, four tracking fields in mm, and four measurement-validity
flags. The original prepared conventions and source before/after SHA checks
are reused. File shards preserve ordered identities/labels; every consumer
joins them to authenticated D000 TRAIN. No model input is reconstructed from
clipped normalized cache fields. The export also copies the original M0HLT
checkpoint for J6. No new teacher fit is trained and no validation particle
target bank is exported. Existing OFFLINE keys come from the authenticated
contrastive transfer. Existing U100 targets are reused byte-for-byte.

P/G descriptors have22 fields: log pT and energy fractions, parent-local delta
eta/sin(phi)/cos(phi), charge, six category indicators, signed log1p tracking,
asinh significances, and four validity flags. Invalid tracking/significance is
zero with explicit validity. Significance requires valid value/error and
positive error. Fractions use scalar sums over each candidate's own parent
collection; the axis uses its summed four-vector. There is no absolute lab
axis, total parent momentum, parent multiplicity, rank, ID or pool position
input. NORMALIZATION records TRAIN real-particle FP64 population mean/std,
floor1e-3, for continuous columns only. Categories/charge/validity are unscaled.
The raw representation remains available for later Q/S coordinate transforms.

The10 G relation channels are deltaR, absolute deltaEta, cos(deltaPhi), sum and
product of pT fractions, valid absolute differences in transformed d0/dz, their
valid sign agreements, and mean joint d0/dz validity. Encode each unordered
real pair with the relation MLP, then average encoded pairs; do not apply a
nonlinear MLP only after averaging raw pair measurements. The result is
permutation invariant and zero for a singleton.

## Sampling closure fixed before scientific execution

RNG is identity/epoch/source/replicate/component keyed PCG64 with root2026110508.
It never consumes model dropout or data-order RNG. Same-family weight variants
share candidate streams; G1/G3/G5 share group streams. Source controls use the
same law, applied to their corresponding measured HLT collection.

- P:500 unique particles, min(32,parent N) positives, uniform without replacement;
  negatives select parents first, then particles; cap16 per negative parent.
- G:128 unique unordered intact groups, min(16,choose(N,g)) positives,
  g=min(requested2/3,N), cap8 negative groups per parent. Every negative is
  intact from one other parent. Small-N anchors are never dropped.
- M quotas split equally G/S/K/F, with remainders in that order. G draws a
  uniform eligible source class, then parent; S is same-class. Classes without
  remaining eligible support are excluded explicitly. K uses TRAIN HLT pT20
  and multiplicity10 quantile bins, the nearest non-exhausted Chebyshev radius,
  then a uniform eligible parent. Expansions are reported.
- P-F indexes actual measured particles. Its discrete signature is PID, charge
  and the four-bit validity pattern; numeric bins are log-pT fraction edges
  [-8,-6,-4,-2,0] and radius [.02,.05,.1,.2,.4,.8]. First expand numeric
  Chebyshev radius within the exact signature; when exhausted, visit increasing
  discrete Hamming distance, restarting the numeric expansion. Choose a uniform
  eligible parent in the first available tier, then a uniform eligible particle.
  No scorer sees signatures or the positive used to define the tier.
- G-F fixes a bounded proposal law:4096 parent-first broad draws of uniform
  intact groups for each anchor/epoch, deduplicated. Match to a sampled positive
  using sorted member signatures: minimize discrete Hamming distance, then
  numeric Chebyshev radius; select parent uniformly within that tier, then
  group. This is **matching within a bounded reservoir**, not an exhaustive
  search over every combination of every TRAIN jet. Report this limitation and
  the selected tier. It bounds the cost without fabricating mixed-parent groups.
- P9: floor(n_negative/2) M negatives;4096 further unique M proposals excluding
  them, cap16 in the proposal universe. Detached current MLP scores rank them;
  ties use canonical parent identity/local slot. Select the remaining negatives
  uniformly without replacement from ranks257..1024 while retaining the final
  cap16. No rank widening, optimizer step, buffer mutation or native re-forward.

Capacity proofs use all TRAIN source counts, exclusion groups when authenticated,
and sufficient worst-exclusion class/global bounds. Particle class capacity
also covers the4096 proposal's G-before-S draws and the separate final pool,
not only a500-particle pool. Sampling never relaxes
quotas, duplicates padding, drops anchors or changes after seeing validation.
Historical data have no authenticated cross-file event groups; record that
limitation rather than asserting physical particle matches or event independence.

## Artifacts and admission

SPEC binds clean pushed source, immutable recipe/order/scope, original A6000
qualification, environment, raw transfer, OFFLINE keys, U100 bank, same-seed
reuse proofs and exact remaining rows. Local REUSE runs the original validator in its
own worktree and binds complete selected fit/report/receipt/prediction bytes,
seeds, population, U100 bank and source. C0/C1 are required compatible reused
references. J5 defaults to its completed original result inside the authenticated
contrastive transfer: original recipe/seeds/U100/OFFLINE parents, original-source
export completion proof, report/receipts/predictions and metric replay must match.
This is explicitly labeled SPORC result reuse; its checkpoint is neither copied
nor re-inferred on Brown. If no completed transferred J5 exists, queue a new fit;
an explicit local original-worktree J5 proof may also be supplied. Incompatible
completed references fail closed rather than silently rerunning them.

HLT_KEYS records frozen M0 extraction, TRAIN standardization/L2 normalization
and ordered identities. CAPACITY/NORMALIZATION bind TRAIN-only preparation.
BASELINE_PARITY checks installed-Weaver native tensors, FP32 logits and original
U100 predictor against AuxiliaryParT. No new performance threshold is used.

TRAINING_REPORT/FIT/EPOCH bind all30 epochs, objective components, native
full-validation selection, native checkpoint replay and selected saved
probabilities. Parameter counts distinguish training-only capacity. TRAIN_PROBE
records first-shuffled-TRAIN-batch source diversity/expansions and correct,
zero-query and within-class shuffled-query scores; probes are adaptive diagnostics,
not independent evaluation. A no-swap shuffle is disclosed by its swapped fraction.

ACCEPTANCE requires genuine warmed production full/partial miniatures, full
candidate universes and measured preparation/mining costs. Distinct source,
sampler, group-size, interaction and dot/judge paths are measured separately;
only scalar-weight/replicate variants with identical execution support share a
measurement. Four CPU-only Linux fork workers share TRAIN arrays and start
before the live student graph; they never call CUDA. Identity-keyed sampling
is unchanged by chunk/order/process assignment. P9 proposal construction and
rank selection run on CPU; detached scoring batches up to8 queries in the
parent GPU process. Host accounting conservatively adds child proportional
RSS to the parent's peak. Production measurements include IPC and preparation.

Execution v2 uses compact five-byte-per-particle signature indices, immutable
exact-cell parent/particle inventories prepared before fork, and sparse
eligible-parent position draws instead of rescanning the parent population.
Stable cell/parent ordering and PCG64 streams are preserved; regression pools
replay the original0a6721ed indices, labels and evidence byte-for-byte. Expanded
regions retain their separate64MiB bounded caches. Descriptor arrays are
preallocated/normalized in chunks; membership drops the redundant physical
measurement copy after authenticating and constructing its descriptors.

The published correct/zero/shuffled-query probes run on the first TRAIN batch
of each epoch only. These deterministic observation-only forwards consume no
model RNG and their removal from other batches preserves the optimizer update.
Technical reports retain synchronized timings for native forward, sampling and
mining, auxiliary forward, probes, backward/optimizer, and input/dispatch. The
projection includes regular updates, native validation and one probe per epoch;
probe time is not multiplied by the number of training batches. Every phase
must be finite/nonnegative and its sum must reproduce the complete batch time.
Neither probes nor target construction are omitted from memory measurement.

Projection is2x full30 fit estimate + preparation +900seconds;
limits48h and85% host/GPU memory under one A6000,6CPUs,32000MiB. Technical weights
are discarded. Local synthetic tests do not substitute for real Hydra acceptance.

PLAN/SUBMIT_CLAIM/TASK use exact hashes/IDs, canonical dry runs, immutable command
journals and exclusive claims. Setup is reviewed/submitted separately. Only
after exact COMPLETED/0:0 setup accounting and authenticated acceptance can
science be reviewed. **Every science command has an
empty dependency list and no dependency/hold/nice/array/lane flags.** The whole
filtered interleaving is submitted without waiting for earlier fits; reused
rows have no new job. No automatic retries or dependent report job are added.
Completed stages are idempotent; partial/ambiguous journals require inspection.

RESULTS authenticates saved joins/metrics and same-seed controls; progress reads
provisional epochs and last markers without inference/writes. Failed, pending,
unsubmitted and incomplete rows remain explicit. Recovery uses historical
M0HLT/OFFLINE anchors; original50k overlaps validation. There is no independent
test. Poor finite metrics never reject admission, cancel fits or skip rows.
Invalid lineage/source, forbidden capabilities, corrupt/nonfinite required data
or measured resource violations fail closed and preserve diagnostics.
