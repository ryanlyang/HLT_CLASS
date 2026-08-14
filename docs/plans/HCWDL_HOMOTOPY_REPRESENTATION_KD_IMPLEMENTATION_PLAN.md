# HCWDL Factorized Homotopy with Representation KD

Status: **implementation-authoritative supplemental plan; the v2 twenty-point
revision was authorized for local implementation on 2026-08-12; live
submission is not authorized by this document**.

Short name: **HCWDL-U-RKD**.

This plan specifies two validation-only representation-distillation variants
of the existing factorized HCWDL structural/feature homotopy:

```text
logit KD + RSET
logit KD + RREL
```

It freezes twenty-percentage-point structural and feature steps. Each variant
has exactly 10 homotopy-transition students plus one terminal born-again HLT
student. The two variants therefore add 22 fits in total. The already
registered logit-only factorized path is the paired control and is not
silently retrained, relabeled, or absorbed into this graph.

This v2 graph supersedes the unexecuted ten-point HCWDL-U-RKD v1 candidate.
No v1 artifact may be relabeled or imported as v2. The completed U/J parent
remains unchanged; v2 imports only its matching twenty-point subset and keeps
each selected rung's original U/J seed alias.

This plan governs only the new HCWDL-U-RKD artifacts. It does not mutate or
reinterpret completed HCWDL, HCWDL-UJ, dense5/dense10, representation-dense,
architecture-factorial, PMARD, or final-test artifacts.

## 1. Authority and implementation prerequisite

The implementation must reconcile two already implemented source lines before
adding campaign code:

1. the structural-feature homotopy authority and implementation introduced by
   repository commit `5faf848daf0b5c12d6139a3cf40c4d352941499f`, including
   `docs/plans/HCWDL_STRUCTURAL_FEATURE_HOMOTOPY_IMPLEMENTATION_PLAN.md`;
2. the matching-free representation-KD implementation whose active plan is
   [HCWDL Matching-Free Privileged Representation-KD Dense Descents](HCWDL_MATCHING_FREE_REPRESENTATION_KD_ASCENTS.md),
   including the corrected `HCWDL_REPRESENTATION_RECIPE/v5` semantics.

Those source lines currently diverge from a common repository ancestor. The
future implementation branch must merge them into one clean committed tree.
Runtime imports from a sibling worktree, path injection into another checkout,
or copying only convenient fragments are forbidden. Every merge conflict in
model surfaces, training, checkpoints, recipes, runtime binding, Slurm, and
documentation must be resolved against this plan and the two parent
authorities rather than by choosing whichever file is newer.

After the merge, documentation authority for the new campaign is:

1. this plan for the combined U/D plus RSET/RREL graph;
2. the structural-feature homotopy plan for `P0`, coupling, `V(s,f)`, exact
   endpoints, and logit-only controls;
3. the controlling dense-descent amendment in the representation-KD plan for
   RSET/RREL mathematics, taps, calibration, schedules, and v5 RREL meaning;
4. new versioned HCWDL-U-RKD contracts;
5. the immutable campaign specification for one execution.

An incompatibility between the two parent implementations is a blocker. It
must not be resolved by weakening either endpoint equality or representation
semantics.

## 2. Scientific question

The logit-only factorized homotopy asks whether knowledge can be carried across
small changes in available particle structure and fields. This supplement asks
whether carrying internal representation information as well as logits helps
the student retain more privileged performance along the same path.

The ordered hypotheses are:

1. **RSET benefit:** the complete RSET track improves validation macro OVR AUC
   at exact D100, exact HLT D0, or terminal M1 relative to the paired
   logit-only factorized track.
2. **RREL benefit:** the complete RREL track improves those endpoints relative
   to the paired logit-only factorized track.
3. **Relation allocation:** RREL outperforms RSET when both receive the same
   total representation budget, supporting the complete relation-inclusive
   allocation rather than an extra unbudgeted loss.
4. **Retention shape:** representation KD reduces the loss of macro AUC and
   background rejection as support and fields are removed rung by rung.

These are complete-path comparisons. At rung `k>1`, RSET and RREL have already
trained different predecessors, so their difference includes accumulated
history. It is not an isolated causal estimate of only the current rung's
representation term.

Finite poor performance is a scientific result. It never fails a task,
shortens a branch, changes a teacher, selects a different view coordinate, or
suppresses a descendant.

## 3. Scope, data, and access

The primary screen is the same validation-only population as the exact
unweighted HCWDL-UJ pilot:

| Role | Rows | Use |
|---|---:|---|
| train | 300,000 | optimization, representation calibration, teacher targets |
| validation | 100,000 | every-pass checkpoint selection and comparisons |
| final test | 0 | absent from the command plan and worker capabilities |

Campaign creation binds one exact parent HCWDL-UJ campaign specification,
source commit, split, row selection, all-ones `HCWDL_RECIPE/v4`, coupling
manifests, coordinate table, endpoint-equality lock, graph-recipe lock, TOFF
checkpoint, and the complete logit-only factorized node registry.

The representation tracks may start after the parent coupling, endpoint, and
graph locks exist. They need not wait for every logit-only fit to finish.
However, the combined aggregate depends on every imported logit-only
factorized report and every new RSET/RREL report. Thus training can proceed in
parallel without allowing a partially completed control path to masquerade as
a comparison.

For a live concurrent launch, the child freezes the parent's exact submitted
`campaign_complete` job ID and adds it only to the aggregate's Slurm
dependencies. It is not a dependency of target construction or either
representation-training track.

The aggregate may read only the exact contextual report list frozen at new
campaign creation. It cannot discover later reports by path existence.

No task may request final-test identities, assignments, labels, features,
checkpoints, predictions, metrics, locks, or capabilities. A final-test path
appearing in a rendered command plan is a contract failure.

## 4. Exact path and the U000 boundary

The imported generalized view is unchanged:

```text
V(s,f)

s = structural conversion from projected-native P0 support to D100 support
f = matched-field conversion from exact offline fields to exact HLT fields
```

The exact endpoints remain:

```text
V(0,0) = P0 projected-native-offline particle multiset
V(1,0) = byte-identical D100 Shell Exact endpoint
V(1,1) = byte-identical canonical HLT input
```

`U000` is **not a trained model**. It is the semantic name for `V(0,0)` and is
used by endpoint audits and existing P0 adapter diagnostics. The first primary
student is `U020`, taught directly by native TOFF.

This choice is mandatory for a paired comparison. Adding a trained U000/P0KD
root would add one extra optimization/KD generation to the representation
paths but not the logit-only control. Existing `P0CE`, `P0KD`, and
`U010P0KD` results remain contextual adapter diagnostics; none becomes a
teacher in the primary graph.

The factorized coordinate registry is exactly:

| Transition | View | `s` | `f` | Shell alpha | Meaning |
|---:|---|---:|---:|---:|---|
| 1 | `U020` | 0.20 | 0.00 | 1.00 | first 20% structural mass switched |
| 2 | `U040` | 0.40 | 0.00 | 1.00 | first 40% structural mass switched |
| 3 | `U060` | 0.60 | 0.00 | 1.00 | first 60% structural mass switched |
| 4 | `U080` | 0.80 | 0.00 | 1.00 | first 80% structural mass switched |
| 5 | `U100` | 1.00 | 0.00 | 1.00 | exact D100 |
| 6 | `D80` | 1.00 | 0.20 | 0.80 | fixed support, 20% field conversion |
| 7 | `D60` | 1.00 | 0.40 | 0.60 | fixed support, 40% field conversion |
| 8 | `D40` | 1.00 | 0.60 | 0.40 | fixed support, 60% field conversion |
| 9 | `D20` | 1.00 | 0.80 | 0.20 | fixed support, 80% field conversion |
| 10 | `D0` | 1.00 | 1.00 | 0.00 | exact HLT |
| 11 | `M1` | 1.00 | 1.00 | 0.00 | separate born-again exact-HLT student |

The structural coordinate is the existing authenticated, nested,
information-mass coordinate. A structural percentage means that percentage
of its frozen label-free edit mass, not that percentage of particles. The
feature coordinate is
the existing Shell Exact alpha convention. No coupling, endpoint, switch,
confidence, field, identity, p4, carrier, or raw-length semantic changes.

## 5. Exact new graph

There are two new cold-start strategies and no warm branch:

```text
TOFF-native
  -> F_RSET_U020 -> F_RSET_U040 -> ... -> F_RSET_U100
  -> F_RSET_D80  -> F_RSET_D60  -> ... -> F_RSET_D0 -> F_RSET_M1

TOFF-native
  -> F_RREL_U020 -> F_RREL_U040 -> ... -> F_RREL_U100
  -> F_RREL_D80  -> F_RREL_D60  -> ... -> F_RREL_D0 -> F_RREL_M1
```

Canonical node IDs are registry values, not names inferred by string parsing:

```text
F_RSET_U020 F_RSET_U040 F_RSET_U060 F_RSET_U080 F_RSET_U100
F_RSET_D80  F_RSET_D60  F_RSET_D40  F_RSET_D20  F_RSET_D0
F_RSET_M1

F_RREL_U020 F_RREL_U040 F_RREL_U060 F_RREL_U080 F_RREL_U100
F_RREL_D80  F_RREL_D60  F_RREL_D40  F_RREL_D20  F_RREL_D0
F_RREL_M1
```

The complete edge registry is:

| Index | RSET node | RSET teacher | RREL node | RREL teacher | Student domain | Seed alias |
|---:|---|---|---|---|---|---|
| 1 | `F_RSET_U020` | `TOFF` | `F_RREL_U020` | `TOFF` | `u020` | `transition_02` |
| 2 | `F_RSET_U040` | `F_RSET_U020` | `F_RREL_U040` | `F_RREL_U020` | `u040` | `transition_04` |
| 3 | `F_RSET_U060` | `F_RSET_U040` | `F_RREL_U060` | `F_RREL_U040` | `u060` | `transition_06` |
| 4 | `F_RSET_U080` | `F_RSET_U060` | `F_RREL_U080` | `F_RREL_U060` | `u080` | `transition_08` |
| 5 | `F_RSET_U100` | `F_RSET_U080` | `F_RREL_U100` | `F_RREL_U080` | `u100`/exact D100 | `transition_10` |
| 6 | `F_RSET_D80` | `F_RSET_U100` | `F_RREL_D80` | `F_RREL_U100` | `d80f` | `transition_12` |
| 7 | `F_RSET_D60` | `F_RSET_D80` | `F_RREL_D60` | `F_RREL_D80` | `d60f` | `transition_14` |
| 8 | `F_RSET_D40` | `F_RSET_D60` | `F_RREL_D40` | `F_RREL_D60` | `d40f` | `transition_16` |
| 9 | `F_RSET_D20` | `F_RSET_D40` | `F_RREL_D20` | `F_RREL_D40` | `d20f` | `transition_18` |
| 10 | `F_RSET_D0` | `F_RSET_D20` | `F_RREL_D0` | `F_RREL_D20` | `hlt`/exact HLT | `transition_20` |
| 11 | `F_RSET_M1` | `F_RSET_D0` | `F_RREL_M1` | `F_RREL_D0` | `hlt`/exact HLT | `transition_21` |

The seed resolver is the imported homotopy resolver with replicate seed 1337:

```text
derive_seed(1337, "hcwdl_uj/<purpose>/<seed_alias>")
```

`<purpose>` remains the existing closed purpose vocabulary. The new graph may
namespace representation-head/resource seeds separately, but it may not
change the imported deployable-backbone, sampler, dropout, or trimmer seed
values for a transition alias.

Each strategy has 11 fits:

- 5 structural U students;
- 5 feature D students;
- 1 terminal M1 student.

The combined supplement has 22 fits: 20 view-changing students plus two M1
students across both strategies.

For each strategy independently:

- `U020` teacher: imported TOFF-native checkpoint on TOFF-native inputs;
- every later U teacher: immediate same-strategy U predecessor on its own U
  view;
- `D80` teacher: same-strategy `U100` on exact D100;
- every later D teacher: immediate same-strategy D predecessor on its own D
  view;
- `M1` teacher: same-strategy `D0` on exact HLT.

No node may use a logit-only teacher after the TOFF root, cross from RSET to
RREL, skip a predecessor, average teachers, add a persistent TOFF anchor, or
select a teacher based on validation results.

Every node is cold-started. It initializes a fresh canonical unified
21-channel Scouting ParT. No deployable backbone parameters, optimizer state,
scheduler state, scaler state, RNG state, or representation heads are loaded
from the predecessor. Only detached teacher targets cross an edge.

## 6. The paired logit-only control

The exact imported logit-only factorized chain is the primary control:

```text
TOFF-native -> U020 -> U040 -> ... -> U100 -> D80F -> ... -> D0F -> M1F
```

At each transition index, the control, RSET, and RREL students must share:

- deployable-backbone initialization seed;
- data sampler and row order;
- dropout, trimmer, and model stochastic streams;
- optimizer and schedule configuration;
- student view tensors and identities;
- update and validation budgets.

Representation-only random resources and projection-head initialization use
separate namespaced streams. They may not advance any shared model/data RNG.

The imported control is reusable only if all of those seed aliases and every
view/data/recipe parent validate exactly. Otherwise the new campaign must
register and run a fresh paired logit-only factorized chain under a bumped
graph contract. It may not call an unpaired historical result â€œthe control.â€

`rho_repr=0` wrapper equivalence is a required implementation regression, not
a third 11-fit scientific track. It must reproduce the ordinary logit-only
loss, backbone gradients, optimizer update, RNG state, and checkpoint bytes
where serialization permits.

## 7. Model surfaces and architecture boundary

All U, D, and M1 students use the canonical unified 21-channel Scouting ParT.
Representation taps come from the same single forward that produces logits:

- `particle_block_2`: contextual token states after particle block 2;
- `jet_penultimate`: 128-dimensional state before the final classifier;
- logits, masks, four-vectors, persistent token IDs, and family codes.

The public deployable forward remains unchanged. Persistent IDs and family
codes are training-time side channels propagated through the same trimmer
call and removed before model embedding.

TOFF-native keeps its registered separate charged 19-feature and neutral
7-feature encoders and fusion classifier. Its target forward exposes:

- charged and neutral block-2 token states separately;
- the native-offline penultimate jet state;
- charged/neutral masks, p4, token IDs, family codes, and logits.

The first `TOFF-native -> U020` edge therefore crosses both input support and
architecture. This is disclosed. The campaign does not claim that its first
20% structural step isolates structure from adapter conversion.

The ordinary and TOFF representation-exposing forwards must pass the existing
installed-Weaver FP32 surface-parity contracts after the two implementations
are merged. The merge cannot introduce a second student forward or second
trimmer call.

## 8. Matching-free representation targets

The student view is still constructed from authenticated high-coverage
assignments and residual coupling. The representation loss itself must not
read or use:

- HLT-slot-to-offline assignment indices;
- O/R residual-coupling pairs;
- carrier correspondence;
- structural switch ranks;
- token-index equality between teacher and student.

Teacher and student are joined only by exact jet identity. Token supervision
is an unordered distribution comparison and supports unequal particle counts.

For an ordinary unified teacher, one target forward records:

- detached FP32 logits;
- detached pooled `jet_penultimate` target;
- detached weighted token-set kernel mean from `particle_block_2`;
- detached RREL relation sketches and eligibility metadata.

For TOFF-native, charged and neutral token/set/relation summaries remain
separate. The student is partitioned by the existing representation-family
contract for comparison; cross-family cosine relations are forbidden.

RSET consumes logits, pooled jet targets, and token-set summaries. RREL
consumes the same common bytes plus relation summaries. A physical target bank
is always the relational superset so common target values cannot differ merely
because the consumer strategy differs.

Teacher targets are generated from the predecessor's selected checkpoint on
the predecessor's own exact view. Running the teacher on the child view is
forbidden. Target arrays are detached; no teacher gradient graph survives
publication or RAM loading.

## 9. Exact loss

### 9.1 Base classification/distillation loss

For every U and D transition:

```text
L_base = 0.25 * CE_unweighted_natural_population_mean
       + 0.75 * KD(predecessor, temperature=2)
```

For terminal M1:

```text
L_base = 0.25 * CE_unweighted_natural_population_mean
       + 0.75 * KD(D0, temperature=1)
```

KD is the existing FP32 temperature-squared-corrected KL reduction. The model
forward may use BF16 autocast; CE, KD, representation losses, calibration, and
reported scalar accumulation are FP32 or the explicitly registered FP64 host
reductions.

No class weights, second teacher, privileged anchor, confidence weight,
per-token match weight, or validation-tuned coefficient is permitted.

### 9.2 RSET representation package

The exact v5 RSET package is:

```text
L_repr_RSET = 0.40 * Lhat_jet
             + 0.60 * Lhat_set
             + 1e-3 * L_orth
```

`Lhat_jet` is the calibrated paired-jet direct-plus-Gram objective.
`Lhat_set` is the calibrated, weighted, matching-free finite-spectral-kernel
token-set objective after an identity-initialized linear student projection.
`L_orth` regularizes only training-time projection heads.

### 9.3 RREL representation package

The exact corrected v5 RREL package is:

```text
L_repr_RREL = 0.30 * Lhat_jet
             + 0.45 * Lhat_set
             + 0.25 * Lhat_rel
             + 1e-3 * L_orth
```

`Lhat_rel` compares finite-kernel summaries of cosine relations among raw
FP32 L2-normalized `particle_block_2` states. No latent projection is applied
before the cosine statistic. Raw geometry selects the fixed pair strata but is
not itself optimized. A regression must prove that RREL gradients reach the
student backbone while the token projection receives no relation gradient.

RREL reallocates one quarter of the same representation budget. It does not
receive an extra unbudgeted term. Thus RREL-minus-RSET compares two complete
equal-budget packages, not â€œRSET plus relations.â€

### 9.4 Total auxiliary coefficient and schedule

For both strategies:

```text
L_total = L_base + 0.10 * scheduled(L_repr)
```

`rho_repr=0.10` is an auxiliary coefficient, not a probability-mixture weight.
It does not rescale or renormalize the 0.25/0.75 base coefficients.

Let `e` be fractional completed passes. The shared jet/set ramp is:

```text
r_js(e) = 0                 e <= 2
          (e - 2) / 4       2 < e < 6
          1                 e >= 6
```

The RREL relation ramp is:

```text
r_rel(e) = 0                e <= 4
           (e - 4) / 4      4 < e < 8
           1                e >= 8
```

RSET uses:

```text
scheduled = r_js * (0.40 Lhat_jet + 0.60 Lhat_set + 1e-3 L_orth)
```

RREL uses:

```text
w_common = r_js - 0.25 * r_rel
scheduled = w_common * (0.40 Lhat_jet + 0.60 Lhat_set)
          + 0.25 * r_rel * Lhat_rel
          + r_js * 1e-3 * L_orth
```

The base loss remains constant for all …883 tokens truncated…tained metadata; cleanup completion is
published afterward. Missing bytes without an authorization are corruption.

Only train-role targets are stored. Validation runs classification inference
without representation-target banks because representation loss does not
select checkpoints.

Target shards bind:

- teacher node/checkpoint and teacher-own view coordinate;
- source commit, architecture/tap parity, class order, and runtime signature;
- train selection identity set/order and source-shard lineage;
- coupling, coordinate, endpoint, base-recipe, representation-recipe, and
  combined-recipe hashes;
- exact FP32 dtype, shapes, logical hashes, and finite/support metadata.

Student train/validation views are constructed once per job into authenticated
process-local RAM caches and replayed across passes. Reconstructed U/D particle
datasets are never durable artifacts. Teacher targets are loaded once into RAM
and joined by canonical jet identity.

Ragged endpoint preparation is linear in selected chunk population: every
required offline and HLT branch is materialized a fixed number of times per
source chunk, never once per selected row. The same prepared builder serves
student caches and non-TOFF predecessor target banks. A bounded ordered worker
pool may use at most the CPUs granted by Slurm while preserving canonical
source/entry order and byte-identical view and target semantics. Worker count
is operational and is excluded from scientific identity.

## 13. Contracts and immutable locks

Existing contract identities are reused only through their exact validators
and hashes. Their meanings are not broadened. The combined campaign adds a
new supplemental v2 family. The unexecuted ten-point v1 family remains a
historical identity and is not accepted by v2 validators:

```text
HCWDL_HOMOTOPY_REPRESENTATION_PARENT_IMPORT/v2
HCWDL_HOMOTOPY_REPRESENTATION_RECIPE_COMPATIBILITY/v1
HCWDL_HOMOTOPY_REPRESENTATION_PREREQUISITE_BUNDLE/v1
HCWDL_HOMOTOPY_REPRESENTATION_INTEGRATION_ATTESTATION/v2
HCWDL_HOMOTOPY_REPRESENTATION_NODE_SPEC/v2
HCWDL_HOMOTOPY_REPRESENTATION_GRAPH/v2
HCWDL_HOMOTOPY_REPRESENTATION_RECIPE/v2
HCWDL_HOMOTOPY_REPRESENTATION_GRAPH_RECIPE_LOCK/v2
HCWDL_HOMOTOPY_REPRESENTATION_TARGET_SPEC/v2
HCWDL_HOMOTOPY_REPRESENTATION_TARGET_GENERATION/v2
HCWDL_HOMOTOPY_REPRESENTATION_TARGET_SHARD/v2
HCWDL_HOMOTOPY_REPRESENTATION_TARGET_MANIFEST/v2
HCWDL_HOMOTOPY_REPRESENTATION_TARGET_CLEANUP_AUTHORIZATION/v2
HCWDL_HOMOTOPY_REPRESENTATION_TARGET_CLEANUP_COMPLETION/v2
HCWDL_HOMOTOPY_REPRESENTATION_CALIBRATION/v2
HCWDL_HOMOTOPY_REPRESENTATION_RESUME_STATE/v2
HCWDL_HOMOTOPY_REPRESENTATION_TRAINING_REPORT/v2
HCWDL_HOMOTOPY_REPRESENTATION_SELECTED_CHECKPOINT/v2
HCWDL_HOMOTOPY_REPRESENTATION_DEPLOYABLE_EXTRACTION/v2
HCWDL_HOMOTOPY_REPRESENTATION_AGGREGATE/v2
HCWDL_HOMOTOPY_REPRESENTATION_CAMPAIGN_SPEC/v2
HCWDL_HOMOTOPY_REPRESENTATION_COMMAND_PLAN/v2
HCWDL_HOMOTOPY_REPRESENTATION_RUNTIME_BINDING/v2
HCWDL_HOMOTOPY_REPRESENTATION_TASK_ATTESTATION/v2
HCWDL_HOMOTOPY_REPRESENTATION_SUBMISSION_LEDGER/v2
HCWDL_HOMOTOPY_REPRESENTATION_MONITOR_REPORT/v2
HCWDL_HOMOTOPY_REPRESENTATION_SOURCE_RECOVERY/v2
HCWDL_HOMOTOPY_REPRESENTATION_RESOURCE_RECOVERY/v2
HCWDL_HOMOTOPY_REPRESENTATION_CAMPAIGN_COMPLETE/v2
```

The combined recipe references, rather than copies approximately:

- exact `HCWDL_RECIPE/v4` hash;
- exact `HCWDL_REPRESENTATION_RECIPE/v5` hash;
- exact homotopy coupling/coordinate/endpoint/graph-recipe locks;
- exact representation tap, parity, kernel-resource, and numerical-acceptance
  artifacts;
- the complete per-node loss, temperature, seed, view, teacher, and target
  table.

The integration attestation proves that the merged implementation preserves:

- public ordinary and TOFF logits/gradients;
- Shell Exact and `V(s,f)` endpoint bytes;
- v5 RSET/RREL mathematics and raw-state RREL gradients;
- deployable extraction and checkpoint selection;
- no runtime import from either source worktree.

### 13.1 Pre-campaign asset bootstrap

The v2 implementation must provide one non-training bootstrap for sites where
the four reusable representation assets have not yet been materialized. The
bootstrap authenticates a training-ready U/J parent plus the frozen historical
unweighted HCWDL campaign and publishes the source-compatible v5
representation recipe, installed-Weaver architecture attestation, numerical
acceptance, and committed deterministic kernel envelope. It may not train a
model or read a final-test role. The historical and U/J v4 recipes are
authenticated independently. Their complete execution policy (optimization,
schedule, losses, temperatures, duration, checkpointing, unweighted
reduction, architecture, and exact class weights) must match. Only evidence
hashes, natural train class counts, and row-selection lineage may differ. A
versioned compatibility artifact binds both exact recipe hashes and files,
the equal policy projection, and the explicit difference allowlist. This is
not authority to translate or approximate a scientific policy. A failed
bootstrap is immutable partial evidence and is retried in a new root.

The installed-Weaver architecture attestation is deliberately narrower than
parent-loss authority. It may audit the authenticated pre-loss-semantics
`b3154d67` wrapper/checkpoint bytes because that artifact proves only strict
model state compatibility. It validates the legacy wrapper's original
content hash, engine lineage, checkpoint hash, and strict state load without
adding current loss fields or changing its meaning. Recipe compatibility
separately proves the exact-one unweighted execution policy. Such a legacy
wrapper remains categorically invalid wherever corrected parent-loss
authority is required; architecture evidence may not be reused as a loss
attestation.

Changing rung spacing, node order, teacher routing, representation strategy,
loss weights, calibration, taps, target sketches, coupling/view semantics, or
seed aliases requires a new scientific contract. CPU/RAM/walltime changes
that preserve scientific runtime signatures may use resource recovery.

## 14. Campaign DAG

The logical dependency graph is:

```text
authenticate training-ready parent UJ locks + source integration
  -> validate coupling/coordinate/endpoint locks
  -> validate ordinary/TOFF surface parity + kernel resources
  -> publish combined graph/recipe lock
  -> build shared TOFF target generation
       -> F_RSET_U020 -> build its target -> F_RSET_U040 -> ... -> F_RSET_M1
       -> F_RREL_U020 -> build its target -> F_RREL_U040 -> ... -> F_RREL_M1
  -> wait for imported logit-only factorized reports
  -> aggregate all three tracks
  -> campaign-complete report
```

RSET and RREL run concurrently when resources permit. Each strategy is
strictly sequential because every child consumes its immediate predecessor's
selected checkpoint. Target generation for a child begins only after the
teacher report and selected checkpoint validate.

The 22 fits and 21 logical target generations are fixed scientific counts.
Physical target-build arrays are source-shard dependent. Their exact array
rows, task IDs, dependency IDs, and total Slurm-job count are frozen in the
immutable command plan after the authenticated source inventory is known;
array-index ordering cannot define scientific meaning.

Every task has an immutable task attestation binding command bytes, source,
parents, declared inputs/outputs, resource class, and scheduler identity.
Scientific result values never control dependency construction.

## 15. Slurm and resource policy

The requested pilot training envelope is:

```yaml
CPUs:     8
RAM:      96G
Walltime: 06:00:00
GPU:      gpu:gh200:1
```

The same envelope is the initial target-builder request unless the genuine
smoke measures a smaller safe class. These are operational values, not
scientific tuning knobs. The final campaign spec must bind genuine Tigris
measurements proving peak CPU memory, CUDA memory, walltime, I/O, target-bank
storage, and headroom. A measured requirement above this envelope is reported
rather than hidden by changing batch size or scientific execution.

CPU-only lock/aggregate tasks use a separately measured smaller resource
class. Arrays are uncapped by default. An operational concurrency cap may be
added only in the command plan and may not change row order, RNG, targets, or
scientific identities.

Every worker uses account `reu-aisocial`, partition `tigris`, a clean detached
worktree at the exact pushed commit, `atlas_kd_tigris`,
`PYTHONNOUSERSITE=1`, and `${CONDA_PREFIX}/lib` first in
`LD_LIBRARY_PATH`. Workers receive absolute `PROJECT_DIR`, campaign-spec, and
task paths. Checkpointable GPU jobs request `--signal=B:USR1@120`, and the
batch shell ends with `exec python -s`.

No job is submitted by implementation, plan publication, smoke construction,
or dry run. Live smoke and pilot submissions each require a separate explicit
user authorization bound to an exact immutable candidate.

## 16. Modes and acceptance sequence

### 16.1 Local synthetic mode

A bounded synthetic graph exercises all 22 nodes, both teacher-domain types,
every loss component, calibration state, target lifecycle, exact resume,
aggregate, and no-final-test invariant. It is not scientific evidence.

### 16.2 Genuine Tigris smoke

The smoke imports an authenticated completed HCWDL-UJ smoke and its exact
coupling/endpoint locks. It uses that smoke's 4,096/4,096 train/validation
population, but each fit performs exactly two optimizer updates. It invokes a
non-scientific full-strength representation probe because the production
ramps remain zero during two updates.

The probe must exercise:

- ordinary and TOFF target construction;
- RSET jet/set gradients;
- RREL raw-state relation gradients and all fixed strata;
- variable-support U and fixed-support D views;
- projection exclusion from RREL;
- target cleanup, USR1 delivery, and exact fresh-process resume.

Smoke science metrics do not authorize a pilot. Successful execution,
lineage, parity, finite gradients, endpoint equality, and measured resources
do.

### 16.3 300k pilot

After the smoke, the implementation renders a complete nonmutating 300k dry
run from exact pushed source. Live submission requires a new phrase-bound
authorization naming the exact campaign-spec, command-plan, source, parent,
resource, storage, and recovery hashes.

The pilot trains all 22 fits for 60 passes. There is no performance early
stop, no final-test task, and no automatic scientific pruning.

## 17. Reporting and comparisons

The primary classification metric is validation macro OVR AUC. Reports also
retain unweighted CE, accuracy, per-class AUC, macro mean log QCD rejection at
50% signal efficiency, geometric-mean R50, and calibration diagnostics.

For every matched view coordinate, report:

```text
RSET - logit-only
RREL - logit-only
RREL - RSET
```

Primary endpoint comparisons are predeclared at:

```text
U100 / exact D100
D0   / exact HLT path endpoint
M1   / exact HLT born-again endpoint
```

For metric `m` where higher is better, recovery against paired M0 and TOFF is:

```text
recovery_m(node) = (m_node - m_M0) / (m_TOFF - m_M0)
```

For CE, use the sign-corrected form:

```text
recovery_CE(node) = (CE_M0 - CE_node) / (CE_M0 - CE_TOFF)
```

Undefined or near-zero denominators are reported as undefined, never replaced
by zero. R50 recovery is secondary because tail rejection is noisier; reports
show both logR50 and `exp(logR50)`.

Representation diagnostics include component losses/scales/status, gradient
ratios, projection conditioning, token/family/stratum support, teacher-target
coverage, and pathwise retention. None selects a checkpoint or a strategy.

The one-seed pilot is an exploratory validation screen. Paired validation
bootstraps are descriptive after repeated checkpoint selection. A future
confirmation must predeclare seeds and rerun each selected strategy's complete
causal prefix plus its paired logit control. Retraining only D0 or M1 with a
new seed while importing screening-seed predecessors is forbidden.

## 18. Failure and recovery

Execution fails closed on source drift, wrong parent or view, corrupt target
hashes, identity coverage mismatch, forbidden branch access, nonfinite
required tensors/losses, endpoint inequality, wrong teacher domain, stale
calibration, invalid resume, hidden truncation, or deployable-state mismatch.

Finite weak representation support publishes an inactive component with exact
reason and continues. Finite bad metrics, zero gains, poor conditioning below
the declared numerical-failure boundary, or failure of a hypothesis also
continue.

Source recovery must:

1. bind the original ledger and immutable monitor;
2. compute the exact failed/downstream closure;
3. preserve completed reports, checkpoints, targets, and cleanup evidence;
4. use a new clean source commit only for reviewed execution corrections;
5. preserve graph, recipes, views, targets, seeds, resources, and output root;
6. publish a separate recovery spec, command plan, attestations, and ledger;
7. support repeated recovery without retraining valid completed nodes.

Resource-only recovery may change CPUs, RAM, walltime, or operational
concurrency after measured OOM/time-limit evidence. It cannot change batches,
updates, precision, view caches, target values, or any scientific field.

Cancellation uses exact campaign-bound job IDs. Broad job-name cancellation,
deleting checkpoints, releasing dependency-never-satisfied jobs, or ordinary
requeue under known-broken source is forbidden.

## 19. Implementation map

The future implementation should add new integration-specific modules rather
than expand either existing graph contract in place:

| Path | Responsibility |
|---|---|
| `src/hlt_classification/scouting/hcwdl_homotopy_representation_contracts.py` | new contract identities and validators |
| `src/hlt_classification/scouting/hcwdl_homotopy_representation_graph.py` | exact 22-node registry, teacher/view/seed table |
| `src/hlt_classification/scouting/hcwdl_homotopy_representation_recipe.py` | v4 base plus v5 representation overlay |
| `src/hlt_classification/scouting/hcwdl_homotopy_representation_targets.py` | teacher-own-domain target generations and cleanup |
| `src/hlt_classification/scouting/hcwdl_homotopy_representation_training.py` | one node, surfaces, calibration, loss, resume, extraction |
| `src/hlt_classification/scouting/hcwdl_homotopy_representation_campaign.py` | campaign spec, command plan, resources, DAG |
| `src/hlt_classification/scouting/hcwdl_homotopy_representation_workflow.py` | thin exact task dispatch |
| `src/hlt_classification/scouting/hcwdl_homotopy_representation_reporting.py` | paired rung/endpoints/recovery aggregate |
| `src/hlt_classification/scouting/hcwdl_homotopy_representation_recovery.py` | source and resource closure recovery |

Reuse without duplicating semantics:

- homotopy coupling, coordinate, endpoint locks, `V(s,f)` stream, and RAM
  view cache;
- ordinary and native representation taps, finite kernel resources, RSET/RREL
  losses, calibration, schedule, and deployable extraction;
- generic metric, checkpoint selection, immutable envelopes, task
  attestations, monitor, exact-ID cancellation, and Slurm journal code.

Thin command surfaces should cover campaign creation, target build/finalize/
cleanup, one task worker, dry run, submit, monitor, exact cancellation,
source recovery, resource recovery, aggregate, and local synthetic smoke.
Workers must contain no scientific registry duplicated from Python modules.

## 20. Required tests

### 20.1 Graph and recipe

- exact 11 RSET and 11 RREL nodes, coordinates, teachers, domains, and edges;
- no trained U000 and no warm node;
- exact 22 fits, 21 target banks, and two terminal candidates;
- same transition backbone/data RNG aliases across logit/RSET/RREL;
- strategy-specific auxiliary RNG cannot perturb shared streams;
- exact base temperatures: 2 for U/D, 1 for M1;
- all 22 engine loss configurations bind v4, v5, and combined recipe hashes;
- graph JSON round-trip identity and no string-parsed scientific semantics.

### 20.2 Views, surfaces, and losses

- U100/D0 endpoint equality remains byte-exact after model-surface merge;
- one-forward logit/surface parity for ordinary and native architectures;
- variable support, padding, unknown family, empty family, and partial batch;
- RSET permutation invariance and no token correspondence access;
- RREL fixed top-32/pair/strata/ESS semantics;
- raw-state RREL backbone gradient and zero token-projection relation gradient;
- RSET/RREL equal total scheduled representation budget;
- hand-calculated ramps before/at/after passes 2, 4, 6, and 8;
- calibration one-forward behavior, RNG restoration, weak-support continuation,
  and nonfinite failure;
- `rho_repr=0` logit-only parity.

### 20.3 Targets and training

- exact 21-bank registry and shared TOFF two-consumer bank;
- teacher evaluated only on predecessor domain;
- logits and representations originate from one teacher execution;
- full train identity coverage with no validation/final targets;
- immutable shard/logical hashes, corruption rejection, atomic generation,
  authorization-before-cleanup, and interrupted cleanup recovery;
- view and target constructed/loaded once rather than per epoch;
- 60 passes, 60 validations, final partial batch, macro-AUC tie order;
- cold initialization and fresh projection heads at every rung;
- exact resume including heads/calibration/target generation;
- extracted D0/M1 logits equal training wrapper logits with heads inactive.

### 20.4 Campaign and operations

- clean-source integration attestation and cross-worktree runtime rejection;
- immutable parent/control lists and no path discovery;
- exact dependency closure with two parallel sequential tracks;
- nonmutating dry run and task-command reconstruction;
- poor finite rows retain descendants;
- final-test task/branch/capability absence;
- exact-ID cancel, monitor, source recovery, resource recovery, and repeated
  recovery;
- local synthetic all-node execution, Tigris full-loss smoke, USR1, and
  measured resource reports.

## 21. Implementation sequence

1. **Integrate source lines.** Merge homotopy and representation implementations
   into one clean branch; run both complete focused suites before modification.
2. **Freeze contracts and graph.** Add the new v2 family, exact 22-node table,
   combined recipe, seed aliases, and fail-closed validators.
3. **Prove surfaces/endpoints.** Resolve model-forward conflicts, rerun
   installed-Weaver parity, RREL-v5 gradient tests, and all `V(s,f)` endpoint
   tests.
4. **Adapt target banks.** Add teacher-own-homotopy-domain construction,
   shared TOFF bank, 20 strategy banks, cleanup, and recovery.
5. **Implement node training.** Wire cold starts, calibration, schedules,
   one-time RAM caches, selection, resume, and extraction.
6. **Implement campaign/reporting.** Add thin CLIs/workers, immutable command
   plan, aggregate, exact monitoring/cancellation, and both recovery paths.
7. **Local closure.** Run focused tests, bounded all-node synthetic smoke,
   complete repository suite, CLI/help, contract inventory, Markdown links,
   shell syntax, and `git diff --check`.
8. **Real closure.** Push exact source, run installed-Weaver parity and the
   separately authorized Tigris smoke, measure resources/storage, and render
   the complete nonmutating 300k dry run.
9. **Pilot authorization.** Request explicit user authorization for the exact
   candidate. Do not infer it from implementation or smoke permission.

## 22. Definition of done

Implementation is complete only when:

- both parent source lines exist in one clean runtime tree;
- all new contracts reject stale, cross-strategy, cross-view, and
  cross-campaign artifacts;
- the exact 22-node graph and 21 target-bank lifecycle execute end to end;
- RSET/RREL mathematics are unchanged from v5 and RREL uses raw block-2 states;
- coupling/view semantics are unchanged and endpoints remain byte-exact;
- all nodes are cold, run 60 passes, validate every pass, and select by macro
  AUC first;
- no target/view is recomputed per epoch or persisted as a repaired dataset;
- D0/M1 extraction is physically HLT-only;
- poor finite results continue and invalid execution fails closed;
- local focused/full tests and the complete synthetic graph pass;
- installed-Weaver parity, genuine Tigris smoke, USR1 resume, measured
  resources/storage, and a complete pilot dry run pass;
- `docs/HANDOFF.md` records exact evidence and the remaining authorization;
- the final pushed commit is clean and the user separately authorizes the
  immutable 300k candidate.

Until then, the correct status is **planned or locally implemented, not pilot
authorized**.

## 23. Claims and limitations

Allowed claims:

- matching-free representation KD across a fixed authenticated U/D homotopy;
- RSET and corrected raw-state RREL v5 complete-path comparisons;
- exact D100 and HLT endpoint comparisons;
- cold-start predecessor knowledge transfer;
- HLT-only terminal deployment.

Forbidden claims:

- representation loss uses or validates physical particle correspondence;
- U000 is a trained model or native-TOFF tensor/architecture identity;
- the first U edge isolates structure from architecture conversion;
- RREL is â€œfree extra supervisionâ€ beyond RSET;
- RREL-minus-RSET isolates only the current relation term;
- the 20 view-changing transitions exhaust the campaign fit count or make the
  two terminal M1 fits unnecessary;
- twenty-point spacing is proven optimal relative to ten- or five-point
  spacing;
- one validation seed is confirmatory;
- success reconstructs genuinely unavailable offline particles at inference;
- failure proves an information-theoretic ceiling.

The twenty-point design is chosen as the first manageable, compute-bounded
test. A ten- or five-point representation study requires a new graph/version
and scientific authorization after this screen; it is not an operational
expansion of the same campaign.

## 24. Authorization boundary

This plan authorizes documentation and, only after a separate user request,
local implementation. It does not authorize:

- a Git merge, commit, push, or worktree deletion;
- live Slurm submission or cancellation;
- rerunning assignments or changing coupling coordinates;
- final-test access;
- pilot launch merely because a smoke or dry run exists;
- retrospective changes after validation results are inspected.

Every real scheduler mutation remains separately phrase-bound to an exact
source-pinned candidate.
