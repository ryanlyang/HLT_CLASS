# HCWDL Full-Data Three-Track 60-Pass Ensemble-Compression Plan

Status: **implementation-authoritative additive plan; implementation and live
submission are not authorized by this document**.

Short name: **HCWDL-MHPE-TRI60-FULL**.

This plan combines the strongest current HCWDL ideas into one deadline-aware,
full-population campaign:

1. a multi-horizon triangular LOGIT-KD ladder;
2. a shorter triangular LOGIT+RSET ladder;
3. a shorter triangular LOGIT+RREL ladder;
4. fixed probability ensembling within every track;
5. fixed probability ensembling of the three terminal M1 models; and
6. one final exact-HLT M2 that compresses the three-track ensemble.

All fresh fits run for 60 complete natural-population passes. The campaign
keeps the implemented unified 21-channel Particle Transformer, projected
offline `U000`, balanced structural homotopy, uniform deterministic feature
homotopy, unweighted population loss, exact HLT endpoint, and macro-AUC-first
checkpoint selection. It changes the graph, adds representation supervision
to two branches, makes representation targets ephemeral and RAM-only, and
disables rolling optimizer-state resumes for this campaign.

This plan is additive. It does not modify, resume, relabel, or reinterpret any
completed HCWDL-UJ, HCWDL-UB, HCWDL-MHPE, HCWDL-U-RKD, dense, schedule-screen,
or final-test execution.

## 1. Authority and implementation prerequisite

Documentation authority for this campaign is:

1. this plan;
2. the implemented
   [multi-horizon projection-ensemble plan](HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_IMPLEMENTATION_PLAN.md)
   for project-then-ensemble semantics;
3. the implemented
   [unified-balanced homotopy plan](HCWDL_UNIFIED_BALANCED_HOMOTOPY_IMPLEMENTATION_PLAN.md)
   for `U000`, balanced U support changes, uniform D field changes, and exact
   endpoints;
4. the corrected `HCWDL_REPRESENTATION_RECIPE/v5` RSET/RREL implementation
   and its implementation-authoritative representation-KD plan;
5. the new versioned contracts required by this plan; and
6. the immutable `campaign_spec.json` for one execution.

The representation implementation currently lives on a separate repository
source line. Before campaign code is added, the implementation must merge the
MHPE and corrected representation-KD source lines into one clean committed
tree. Runtime imports from `.worktrees`, `PYTHONPATH` injection into a sibling
checkout, or copying only convenient fragments are forbidden. The integration
attestation must bind the exact clean source commits and prove that:

- unified-balanced `V(s,f)` endpoint bytes are unchanged;
- ordinary model logits and representation surfaces come from one forward;
- RSET and corrected raw-state RREL v5 mathematics are unchanged;
- probability-target KD remains numerically identical to implemented MHPE;
- public deployable inference remains the ordinary HLT-only model; and
- the fixed-chunk ragged preprocessing repair and batched representation-loss
  speedups are present.

The explanatory representation guide observed in the development worktree is
useful evidence, but an uncommitted worktree path is not a scientific parent.
Implementation must first bind a clean committed representation source. If
the two source lines cannot be reconciled without changing endpoint or
representation semantics, that is a genuine blocker and requires a plan
revision rather than a silent choice.

### 1.1 Exact representation source map

The corrected implementation to integrate is currently located under the
representation development source line at:

```text
.worktrees/hcwdl-u-rkd-local/src/hlt_classification/
```

That path is for locating and reviewing source only. It must not appear in a
runtime import, campaign spec, Slurm command, artifact parent, or published
scientific claim. Implementation binds the eventual clean commit containing
these files.

| Source file | Symbols/semantics to integrate |
|---|---|
| `models/scouting_particle_transformer.py` | `HCWDLScoutingSurfaces`, one-forward ordinary surface capture, block-2 token state, penultimate jet state |
| `models/hcwdl_representation.py` | `HCWDLRepresentationHeads`, `HCWDLRepresentationStudent`, `publish_hcwdl_deployable_extraction`, strict head-free deployable checkpoint |
| `scouting/hcwdl_representation_graph.py` | `HCWDL_REP_SET/v1`, `HCWDL_REP_REL/v1`, strategy identities and paired seed conventions |
| `scouting/hcwdl_representation_recipe.py` | corrected `HCWDL_REPRESENTATION_RECIPE/v5` payload and fixed kernel/schedule meanings |
| `scouting/hcwdl_representation_kernels.py` | committed spectral resources, device caching, finite spectral features, weighted kernel means, reference oracles |
| `scouting/hcwdl_representation_losses.py` | `jet_representation_loss`, `ordinary_set_representation_loss`, `build_ordinary_token_targets`, `build_relation_topology`, `build_student_relation_sketches`, `build_teacher_relation_targets`, `relation_representation_loss`, `scheduled_representation_loss` |
| `scouting/hcwdl_representation_calibration.py` | train-only identity selection, gradient-scale calibration, state/RNG restoration, pass-2/pass-4 activation |
| `scouting/hcwdl_representation_targets.py` | target array schema, identity hashing, target validation, kernel dimensions and support metadata |
| `scouting/hcwdl_representation_target_runtime.py` | `prepare_target_generation_in_memory`, one-forward teacher surface/target construction, target sentinels and audits |
| `scouting/hcwdl_representation_training.py` | representation student initialization, batch normalization, target joining, `_raw_representation_components`, `compute_node_loss`, diagnostics and extraction |
| `scouting/hcwdl_homotopy_stream.py` | corrected fixed-work-per-source-chunk prepared endpoint streaming |
| `scouting/hcwdl_homotopy_representation_training.py` | reference wiring from a homotopy view and selected teacher to representation targets and one student fit |
| `scouting/hcwdl_homotopy_representation_targets.py` | durable-bank reference implementation used only as a numerical oracle for the new RAM path |

The historical rolling-resume publisher and durable representation-target
publisher are **reference implementations, not runtime dependencies** of this
campaign. Their validators and bounded fixtures are reused to prove numerical
equivalence, but this plan's workers do not write their payloads.

### 1.2 Exact MHPE integration points

The ensemble side comes from the tracked main source line:

| Main-source file | Required reuse |
|---|---|
| `scouting/hcwdl_mhpe_graph.py` | source-qualified specialist and stage-registry pattern |
| `scouting/hcwdl_mhpe_targets.py` | probability-bundle arithmetic, identity binding, manifest/lock validation, no-double-temperature rules |
| `scouting/hcwdl_mhpe_runner.py` | unified-balanced student-view construction, probability-target KD, selected-checkpoint inference |
| `scouting/hcwdl_mhpe_workflow.py` | specialist/reducer dispatch and immutable publication pattern |
| `scouting/hcwdl_mhpe_campaign.py` | full-data foundation authentication, source-pinned spec and command-plan construction |
| `scouting/hcwdl_mhpe_recovery.py` | exact failed/downstream closure and completed-stage preservation |

Integration must not bolt RSET/RREL onto an already averaged representation.
For each representation specialist, the new runner performs two independent
joins for the same canonical train identities:

```text
declared MHPE probability bank
    -> base C25P75 or C10P90 logit/probability KD

declared single carrier checkpoint on the carrier's own view
    -> RAM-only jet/set/relation targets
    -> scheduled RSET or RREL auxiliary
```

Both terms enter one `compute_node_loss` call and one backward pass. The
student logits and student representation surfaces come from one forward.
The implementation must not run a second student forward, redraw trimming,
or let representation code choose the teacher edge.

The graph module owns all teacher and carrier routing. The MHPE probability
module owns only probability construction/validation. The representation
module owns only carrier target construction and auxiliary math. The training
runner combines their already validated outputs. This separation prevents
node-name parsing or implementation convenience from redefining the science.

## 2. Scientific objective

The campaign tests whether three complementary knowledge-transfer mechanisms
can retain privileged performance and then be compressed into one deployable
HLT model:

- **LOGIT:** classification-distribution supervision only;
- **RSET:** logit KD plus jet-state and unordered token-set supervision;
- **RREL:** logit KD plus an equal-budget jet/set/relation package.

The triangular teacher lattice addresses the telephone-game failure of a
single predecessor chain. At each lower coordinate, every earlier declared
teacher is independently projected into a fresh model on the same target
view. Those same-view specialists are then combined by fixed probability
averaging. The ensemble, rather than one arbitrarily selected specialist,
becomes the next local teacher.

The ordered hypotheses are:

1. a 60-pass full-data triangular LOGIT track retains more of the
   `M0paired -> U000` gap than the existing 20-pass full-data lattice;
2. the shorter RSET and RREL tracks produce complementary exact-HLT models
   even when their individual metrics do not beat LOGIT;
3. each fixed same-view ensemble exceeds its mean specialist and preferably
   its best specialist;
4. the fixed `M1_LOGIT + M1_RSET + M1_RREL` ensemble exceeds each member;
5. one fresh exact-HLT `M2` compresses most of that ensemble gain; and
6. the final single model remains fully HLT-only at inference.

These are hypotheses, never execution gates. Any finite poor metric completes
the graph, remains in every predeclared ensemble, and is reported.

## 3. Population and access boundary

The campaign uses the authenticated all-mapped HCWDL full-data population:

| Role | Approximate size | Authorized use |
|---|---:|---|
| train | 2.6 million | optimization, teacher probabilities, RAM-only representation targets, calibration |
| validation | 1 million | every-pass checkpoint selection and fixed-ensemble evaluation |
| final test | 1 million | sealed; absent from ordinary campaign capabilities |

The approximate counts above are descriptive only. The campaign spec stores
the exact integer counts, ordered identity-set hashes, source inventories,
split hash, and all-mapped selection hash inherited or rebuilt from the
authenticated full-data foundation.

Ordinary tasks may not open, assign, couple, project, cache, infer, or report a
final-test row. The train/validation graph must always report
`final_test_accessed: false`. A later sealed evaluation requires a separately
versioned finalist lock, explicit human execution lock, and one consumed
execution claim. Test metrics may not change checkpoints, members, ensemble
weights, or the designated M2.

## 4. Exact homotopy coordinates

Every student uses the implemented unified-balanced view:

```text
V_UB(s,f)

s = balanced structural progress from projected-offline support to D100
f = uniform matched-field progress from offline values to HLT values
```

The required coordinates are exact rationals:

| Label | Exact view | Meaning |
|---|---|---|
| `U000` | `V_UB(0,0)` | projected-native-offline unified model input |
| `U050` | `V_UB(1/2,0)` | half of frozen structural information mass switched |
| `U100` | `V_UB(1,0)` | byte-exact authenticated D100 input |
| `D066` | `V_UB(1,1/3)` | two-thirds offline matched-field strength |
| `D050` | `V_UB(1,1/2)` | half offline matched-field strength |
| `D033` | `V_UB(1,2/3)` | one-third offline matched-field strength |
| `D000` | `V_UB(1,1)` | byte-exact canonical HLT input |

`D066` and `D033` are names for the exact fractions above, not binary floats
`0.66` and `0.33`. Specs store numerator/denominator pairs and float-hex
encodings. No match confidence enters the D coordinate. Continuous fields
move uniformly, and discrete field groups use the existing deterministic,
nested population switch. No coupling, support, PID, charge, validity,
ordering, mask, padding, p4, or raw-length semantic changes.

## 5. Shared U000 root

`U000` is one fresh, shared, CE-only model trained on the all-mapped train
population for 60 passes. It is not native TOFF and is not imported from a
20-pass full-data campaign. It consumes the projected-native-offline particle
population in the unified 21-channel architecture.

Its exact recipe is:

```text
unweighted per-jet CE
fresh canonical unified 21-channel ParT
60 complete natural-population passes
validation after every pass
macro AUC -> CE -> logR50 -> earliest update selection
batch size 256
AdamW, peak LR 3e-4, weight decay 0.01
5% warmup, cosine decay, 5% LR floor
BF16 model forward; FP32 loss and metric arithmetic
```

U000 is trained once. All three tracks consume the exact same selected
checkpoint and probability bank. Retraining a nominally similar U000 per
track is forbidden.

## 6. Exact graph and fit registry

### 6.1 LOGIT track

The full triangular LOGIT track is:

```text
U000
  -> LOGIT_U050E
  -> LOGIT_U100E
  -> LOGIT_D066E
  -> LOGIT_D033E
  -> LOGIT_D000E
  -> M1_LOGIT
```

Each `E` is formed only after all source-qualified specialists at that target
view finish:

| Target ensemble | Fresh specialists and KD teachers |
|---|---|
| `LOGIT_U050E` | `LOGIT_U050_from_U000 <- U000` |
| `LOGIT_U100E` | `LOGIT_U100_from_U000 <- U000`; `LOGIT_U100_from_U050E <- LOGIT_U050E` |
| `LOGIT_D066E` | from `U000`, `LOGIT_U050E`, `LOGIT_U100E` |
| `LOGIT_D033E` | from `U000`, `LOGIT_U050E`, `LOGIT_U100E`, `LOGIT_D066E` |
| `LOGIT_D000E` | from `U000`, `LOGIT_U050E`, `LOGIT_U100E`, `LOGIT_D066E`, `LOGIT_D033E` |

`M1_LOGIT` is one fresh exact-HLT student taught by `LOGIT_D000E`.

The LOGIT track owns 15 view-projection specialists plus one M1 fit.

### 6.2 RSET track

The shorter triangular RSET track is:

```text
U000
  -> RSET_U100E
  -> RSET_D050E
  -> RSET_D000E
  -> M1_RSET
```

| Target ensemble | Fresh specialists and KD teachers |
|---|---|
| `RSET_U100E` | `RSET_U100_from_U000 <- U000` |
| `RSET_D050E` | `RSET_D050_from_U000 <- U000`; `RSET_D050_from_U100E <- RSET_U100E` |
| `RSET_D000E` | from `U000`, `RSET_U100E`, `RSET_D050E` |

`M1_RSET` is one fresh exact-HLT student taught by `RSET_D000E`.

Every RSET edge receives both the declared logit/probability target and the
declared deterministic representation carrier target. The track owns six
view-projection specialists plus one M1 fit.

### 6.3 RREL track

The RREL track has the identical topology, coordinates, seeds, and base loss:

```text
U000
  -> RREL_U100E
  -> RREL_D050E
  -> RREL_D000E
  -> M1_RREL
```

| Target ensemble | Fresh specialists and KD teachers |
|---|---|
| `RREL_U100E` | `RREL_U100_from_U000 <- U000` |
| `RREL_D050E` | `RREL_D050_from_U000 <- U000`; `RREL_D050_from_U100E <- RREL_U100E` |
| `RREL_D000E` | from `U000`, `RREL_U100E`, `RREL_D050E` |

`M1_RREL` is one fresh exact-HLT student taught by `RREL_D000E`.

The track owns six view-projection specialists plus one M1 fit.

### 6.4 Terminal ensemble and M2

After all three M1 models finish:

```text
M1E = mean_probability(M1_LOGIT, M1_RSET, M1_RREL)
M1E -> M2
```

`M2` is a fresh exact-HLT unified model. It receives only fixed M1E
probability KD and labels. It receives no representation auxiliary because an
ensemble has no unique latent representation and because the scientific goal
is one ordinary deployable HLT model.

### 6.5 Counts

| Component | Fresh fits |
|---|---:|
| shared U000 | 1 |
| LOGIT specialists + M1 | 16 |
| RSET specialists + M1 | 7 |
| RREL specialists + M1 | 7 |
| terminal M2 | 1 |
| **total** | **32** |

There are 12 fixed probability reducers: five LOGIT rung ensembles, three
RSET rung ensembles, three RREL rung ensembles, and `M1E`. A one-member `E`
artifact remains a real authenticated probability bank so every downstream
consumer uses one target contract.

The complete canonical fit registry is:

```text
U000

LOGIT_U050_from_U000
LOGIT_U100_from_U000
LOGIT_U100_from_U050E
LOGIT_D066_from_U000
LOGIT_D066_from_U050E
LOGIT_D066_from_U100E
LOGIT_D033_from_U000
LOGIT_D033_from_U050E
LOGIT_D033_from_U100E
LOGIT_D033_from_D066E
LOGIT_D000_from_U000
LOGIT_D000_from_U050E
LOGIT_D000_from_U100E
LOGIT_D000_from_D066E
LOGIT_D000_from_D033E
M1_LOGIT

RSET_U100_from_U000
RSET_D050_from_U000
RSET_D050_from_U100E
RSET_D000_from_U000
RSET_D000_from_U100E
RSET_D000_from_D050E
M1_RSET

RREL_U100_from_U000
RREL_D050_from_U000
RREL_D050_from_U100E
RREL_D000_from_U000
RREL_D000_from_U100E
RREL_D000_from_D050E
M1_RREL

M2
```

The complete reducer registry and component lists are:

```text
LOGIT_U050E = {LOGIT_U050_from_U000}
LOGIT_U100E = {LOGIT_U100_from_U000, LOGIT_U100_from_U050E}
LOGIT_D066E = {LOGIT_D066_from_U000, LOGIT_D066_from_U050E,
               LOGIT_D066_from_U100E}
LOGIT_D033E = {LOGIT_D033_from_U000, LOGIT_D033_from_U050E,
               LOGIT_D033_from_U100E, LOGIT_D033_from_D066E}
LOGIT_D000E = {LOGIT_D000_from_U000, LOGIT_D000_from_U050E,
               LOGIT_D000_from_U100E, LOGIT_D000_from_D066E,
               LOGIT_D000_from_D033E}

RSET_U100E = {RSET_U100_from_U000}
RSET_D050E = {RSET_D050_from_U000, RSET_D050_from_U100E}
RSET_D000E = {RSET_D000_from_U000, RSET_D000_from_U100E,
              RSET_D000_from_D050E}

RREL_U100E = {RREL_U100_from_U000}
RREL_D050E = {RREL_D050_from_U000, RREL_D050_from_U100E}
RREL_D000E = {RREL_D000_from_U000, RREL_D000_from_U100E,
              RREL_D000_from_D050E}

M1E = {M1_LOGIT, M1_RSET, M1_RREL}
```

## 7. Teacher probabilities and representation carriers

An ensemble has one class-probability distribution but no coherent hidden
state. The implementation therefore separates two concepts:

- **distribution teacher:** the full declared selected checkpoint or
  probability ensemble;
- **representation carrier:** one deterministic selected model whose own
  domain and internal surfaces are used for RSET/RREL targets.

This is not a learned carrier choice and never depends on validation metrics.

### 7.1 Exact carrier table

For either representation strategy `X in {RSET,RREL}`:

| Distribution teacher | Representation carrier |
|---|---|
| `U000` | `U000` |
| `X_U100E` | `X_U100_from_U000` |
| `X_D050E` | `X_D050_from_U100E` |
| `X_D000E` | `X_D000_from_D050E` |

Consequently:

- a specialist taught by `U000` uses U000 for probabilities and surfaces;
- a specialist taught by `X_U100E` uses ensemble probabilities but surfaces
  from `X_U100_from_U000`;
- a specialist taught by `X_D050E` uses ensemble probabilities but surfaces
  from `X_D050_from_U100E`; and
- `M1_X` uses `X_D000E` probabilities and surfaces from
  `X_D000_from_D050E`.

The carrier is the local-predecessor projection specialist because it is the
member most directly aligned with the ensemble's causal path. Changing a
carrier, averaging representations, selecting a carrier by performance, or
using a different carrier per row changes scientific meaning and requires a
new graph version.

The teacher/carrier is always evaluated on its own authenticated view. It is
never run on the child's view. Targets join to the student solely by exact jet
identity. Particle correspondence, residual coupling indices, and HLT/offline
match confidence are forbidden from representation losses.

## 8. Losses and optimization

### 8.1 View-changing specialists

Every LOGIT, RSET, and RREL U/D specialist uses:

```text
L_base = 0.25 * CE_unweighted
       + 0.75 * T^2 * KL(p_teacher,T || p_student,T)

T = 2
```

The teacher may be a selected model or an ensemble probability bank. An
ensemble probability is consumed directly and is never treated as a logit or
temperature-transformed twice.

### 8.2 RSET auxiliary

RSET adds the corrected v5 package:

```text
L_total = L_base + 0.10 * scheduled(L_RSET)

L_RSET = 0.40 * Lhat_jet
       + 0.60 * Lhat_set
       + 0.001 * L_orth
```

`Lhat_jet` and `Lhat_set` use the exact train-only gradient calibration,
matching-free unordered token-set target, fixed kernels, taps, validity, and
support rules in `HCWDL_REPRESENTATION_RECIPE/v5`.

### 8.3 RREL auxiliary

RREL adds the equal-budget corrected v5 package:

```text
L_total = L_base + 0.10 * scheduled(L_RREL)

L_RREL = 0.30 * Lhat_jet
       + 0.45 * Lhat_set
       + 0.25 * Lhat_relation
       + 0.001 * L_orth
```

Relation cosines use raw FP32-normalized block-2 states. The set projection is
not applied to relation geometry. The exact top-32, pair strata, ESS support,
kernel resources, and no-match semantics remain v5.

### 8.4 Representation schedules

Let `e` be fractional completed passes. The shared jet/set ramp is:

```text
r_js(e) = 0                 e <= 2
          (e - 2) / 4       2 < e < 6
          1                 e >= 6
```

The relation ramp is:

```text
r_rel(e) = 0                e <= 4
           (e - 4) / 4      4 < e < 8
           1                e >= 8
```

RSET multiplies jet/set/orthogonality by `r_js`. RREL uses the exact v5
reallocation:

```text
w_common = r_js - 0.25 * r_rel
scheduled = w_common * (0.40 * Lhat_jet + 0.60 * Lhat_set)
          + 0.25 * r_rel * Lhat_relation
          + r_js * 0.001 * L_orth
```

### 8.5 M1 and M2 compression

All three M1 models use the same base compression recipe:

```text
0.10 CE + 0.90 probability KD
temperature 1
60 passes
```

`M1_RSET` and `M1_RREL` retain their respective scheduled representation
auxiliary with coefficient 0.10 and their frozen carriers. `M1_LOGIT` has no
representation auxiliary. Using the same base recipe and paired seeds keeps
the track comparison interpretable.

M2 uses:

```text
0.10 CE + 0.90 M1E probability KD
temperature 1
no representation auxiliary
60 passes
```

### 8.6 Common optimization recipe

Every fresh fit uses:

```text
microbatch size:            256
gradient accumulation:      1
effective batch size:       256
optimizer:                  AdamW
peak learning rate:         3e-4
betas:                      (0.9, 0.999)
epsilon:                    1e-8
weight decay:               0.01
gradient clipping:          disabled
schedule:                   5% warmup + cosine to 5% floor
model forward:              BF16 autocast
CE/KD/representation math:  FP32 except declared FP64 host reductions
passes:                     60
validation:                 after every pass
performance early stop:     forbidden
```

Checkpoint selection is highest macro OVR AUC, then lowest CE, then highest
macro mean log QCD rejection at 50% signal efficiency, then earliest update.

All models are cold-started. No backbone, projection head, optimizer,
scheduler, scaler, sampler, or RNG state is inherited from a teacher.

## 9. Ensemble numerical contract

For selected specialist logits `z_k(x)` and declared temperature `T`:

```text
p_E,T(x) = (1/K) * sum_k softmax(z_k(x) / T)
```

The component list is lexical by canonical node ID. Each softmax is
max-subtracted FP32. Components are accumulated and divided in FP64 in that
canonical order, then rounded once to little-endian FP32. Every row must be
finite, nonnegative, and normalized within the frozen tolerance.

Weights are exactly uniform. They cannot depend on labels, validation
metrics, class, confidence, entropy, disagreement, row, rung, or carrier.
Raw logits, model weights, and representations are never averaged.

Every U/D ensemble publishes T2 train probabilities for children and T1
validation probabilities for metrics. Exact-HLT D000 and M1 ensembles publish
T1 train probabilities for compression and T1 validation probabilities.
Any additional temperature is diagnostic only and cannot silently become a
teacher.

## 10. RAM-first target and view lifecycle

### 10.1 What may persist

The campaign may persist:

- campaign, graph, recipe, source, endpoint, and resource locks;
- selected and final model envelopes;
- compact FP32 train probability banks for multi-consumer selected models or
  ensembles;
- compact validation probabilities needed for fixed ensemble reporting;
- target manifests, identity registries, logical hashes, support audits, and
  cleanup evidence;
- validation histories, metrics, runtime profiles, task attestations, ledgers,
  and reports.

Probability banks are intentionally durable. For 2.6 million rows and 15
classes, one FP32 bank is about 156 MB before identities and metadata. Their
few-gigabyte campaign total is small compared with repaired particle views,
representation targets, or rolling optimizer states and prevents repeated
teacher inference across several consumers.

### 10.2 What must never persist

The campaign must not persist:

- reconstructed U/D particle datasets;
- full train or validation view caches;
- token states, jet states, set sketches, or relation sketches for an
  RSET/RREL edge;
- model activations;
- per-pass optimizer, scheduler, scaler, or RNG resume generations; or
- partial checkpoints eligible for warm or resume reuse.

### 10.3 Per-job representation target construction

Every RSET/RREL training job performs this exact one-time sequence:

1. authenticate its campaign, selected distribution teacher, probability
   bank, representation carrier, view coordinates, identities, and kernels;
2. stream the carrier's train view in canonical identity order;
3. run the carrier once and build logits, jet targets, token-set sketches,
   relation sketches, and support metadata into process-local host RAM;
4. validate full train identity coverage and logical hashes;
5. release the carrier model and transient teacher chunks;
6. build the student train and validation views once into process-local RAM;
7. calibrate representation components on the frozen train-only calibration
   identities without optimizer mutation;
8. train for 60 passes by replaying the student cache and RAM targets; and
9. release all ephemeral targets/views when the process exits.

Representation targets are computed once per job, not once per pass. There
are no separate durable RSET/RREL target-build jobs. A small immutable audit
records the carrier report/checkpoint, identity registry, target array shapes,
support counts, logical hashes, dtype, peak RAM, and `durable_payload: false`.
It cannot reconstruct target bytes and does not authorize cross-job reuse.

Carrier logits in this sequence are parity and lineage diagnostics. The
classification KD term always consumes the separately authenticated
distribution-teacher probability bank. Carrier logits may replace that bank
only in the degenerate single-model case where the graph proves the carrier
and distribution teacher are the same checkpoint and temperature.

Validation checkpoint selection needs classification metrics only. No
validation representation bank is constructed or stored.

The teacher stream and student cache should share the corrected prepared-
endpoint builder. The implementation should avoid holding complete teacher
and student view caches simultaneously: stream the carrier into compact RAM
targets, release it, then build the student cache.

## 11. No-resume execution policy

This campaign deliberately disables rolling training resumes:

```text
resume_policy: disabled_restart_from_zero_v1
routine_persistent_training_state: false
partial_checkpoint_reuse: forbidden
```

During a fit, model, optimizer, scheduler, RNG, current best state, and final
state live in process memory. The best checkpoint may be retained as an
in-memory CPU copy. On successful completion, the worker atomically publishes
only:

- the selected deployable checkpoint;
- the final deployable checkpoint;
- the selected training-wrapper metadata needed to prove extraction parity;
- the complete 60-row validation history; and
- the immutable training/runtime report.

A timeout, node failure, preemption, cancellation, process crash, or invalid
input loses that fit's in-memory progress. Recovery restarts the exact fit
from update zero using the same scientific seeds and inputs. It may not load a
partial checkpoint or shorten the remaining schedule.

GPU jobs still request `--signal=B:USR1@120`. The handler publishes only a
small interruption attestation with task identity, last completed pass/update,
reason, and no scientific completion claim, then exits nonzero so descendants
remain blocked. It does not serialize model or optimizer tensors.

This is an explicit reliability/storage tradeoff. The historical 300k
representation campaign used about 22.2 GiB for rolling resume generations;
removing them reduces disk and per-pass fsync/hash overhead but increases the
cost of a late failure. The plan does not claim otherwise.

## 12. Resource and deadline policy

Batch size remains 256. A larger batch could improve throughput but would
change optimization noise and the scientific recipe, so this campaign does
not use 512 or 1024.

Initial Tigris envelopes are:

| Task class | CPUs | RAM | Walltime | GPU |
|---|---:|---:|---:|---|
| U000, LOGIT specialists, LOGIT M1, M2 | 16 | 256 GiB | 3-00:00:00 | `gpu:gh200:1` |
| RSET specialists and M1 | 16 | 384 GiB | 6-00:00:00 | `gpu:gh200:1` |
| RREL specialists and M1 | 16 | 384 GiB | 6-00:00:00 | `gpu:gh200:1` |
| probability reducers | 16 | 192 GiB | 1-00:00:00 | `gpu:gh200:1` when inference is required |
| locks, aggregate, completion | 4 | 32 GiB | 02:00:00 | none |

### 12.1 Full-population preprocessing execution repair

The immutable original campaign retains the 16-CPU envelopes above. Runtime
evidence from full-data reducer job `90660` showed that the expensive ROOT,
selection, assignment, and coupling stages were upstream of the existing
view-transform pool: 16 CPUs were reserved while source ingestion was
effectively serial. This is an execution defect, not a scientific result or a
reason to change rows, views, ordering, seeds, losses, or batching.

Source-repaired recoveries request all 72 effective CPUs measured on the
GH200 nodes for GPU fit, probability-reducer, and representation classes.
This is a deadline-oriented execution allocation: the repaired cache builder
can consume the bounded allocation, while the subsequent single-GPU
optimization or inference stage is expected to leave most CPUs idle. The
cache builder partitions the bounded allocation between
independent source producers and per-source view transforms. With at least
eighteen nonempty sources, the default 72-CPU plan is eighteen source
producers and three transform workers per producer. A bounded queue holds at most two
batches per source worker. One coordinator alone writes the final RAM cache,
using immutable source-specific slices derived from the authenticated split
and row-selection counts.

Source completion order is deliberately irrelevant. Within every source,
entry identities must be strictly increasing; across sources, the coordinator
restores exact split-record order. Balanced views therefore retain the serial
builder's canonical identity order. The older HLT cache could be initially
materialized in shuffled/interleaved order, so its storage-order digest may be
canonicalized by this repair; its identity set, labels, all particle arrays,
and every epoch's sampler replay must remain byte-identical. Missing,
duplicate, cross-source, out-of-order, or excess rows fail closed.
There is no durable repaired view dataset and no second full-size cache.

The source-reader count may be reduced through the bounded
`HCWDL_UB_VIEW_SOURCE_WORKERS` execution setting if real Tigris measurements
show filesystem contention. It may never exceed the allocated CPU or source
count, and it does not alter scientific artifacts. A source-pinned recovery
must use the explicit `--logit-cpus 72`, `--reducer-cpus 72`, and, when it
restarts representation fits, `--representation-cpus 72` resource overrides.
The speedup is accepted only after real-job phase timings and `sstat` prove
CPU utilization and no regression in canonical cache hashes; no numerical
speedup is promised in advance.

### 12.2 Split-ledger composite recovery

If independent source repairs leave LOGIT and representation work in separate
immutable ledgers, recovery must not select one stale ledger and duplicate
healthy work from the other. An
`HCWDL_MHPE_THREE_TRACK_60E_COMPOSITE_RECOVERY_SPEC/v1` binds both recovery
specifications, both exact live ledgers, and fresh immutable monitor reports.
Task ownership is fixed: LOGIT tasks come from the LOGIT subject; RSET, RREL,
and the shared tail come from the representation subject. Every task admitted
to the new closure must be terminal or exactly cancelled in its owning
ledger. Completed outputs remain immutable, and running upstream
representation fits remain external exact-ID `afterok` parents.

The composite closure is the full canonical downstream closure of all failed
owned tasks across the original graph. Thus the new `reduce_M1E` depends on
the newly registered `M1_LOGIT`, `M1_RSET`, and `M1_RREL`; no task may retain a
dependency on a cancelled `906xx` job or duplicate a running `910xx` fit. The
spec stores per-task source ownership, superseded exact job IDs, active parent
IDs, and the complete dependency plan. Its command plan is recomputed during
validation, and all recovered GPU tasks use the separately authenticated
72-CPU execution envelope. This changes execution only, never graph, recipe,
rows, seeds, losses, views, passes, or result interpretation.

The memory profile must include simultaneous train and validation student
caches, complete RAM representation targets, probability targets, model,
optimizer, calibration workspace, prepared source chunks, and allocator
headroom. Authorization fails before training if measured peak usage exceeds
75% of the request or if local disk projection exceeds the locked free-space
reserve.

The intended critical path is approximately:

```text
U000
  + max(LOGIT path, RSET path, RREL path)
  + M1E reducer
  + M2
```

Specialists within one rung run in parallel, and all three tracks run in
parallel after U000. Existing measurements suggest a roughly 4.5-6 day
compute critical path after queue latency when no task fails, but this is an
operational estimate, not a promise or scientific contract. Queue delay,
filesystem contention, a six-day timeout, or restart-from-zero recovery can
push completion beyond 6.5 days.

Resource recovery may increase CPUs, RAM, walltime, or an operational array
cap while preserving GPU class. It may not change rows, batch size, passes,
losses, coordinates, seeds, precision, or targets.

## 13. Scheduler DAG

The campaign is one scientific graph with one canonical spec, one command
plan, one primary ledger, and one monitor. It is not six loosely associated
campaigns.

```text
authenticate source + all-mapped foundation
  -> integration/endpoint/resource/graph/recipe locks
  -> train U000
  -> publish U000 T2/T1 probability bank
       |
       +-> LOGIT rung specialists -> reducer -> next LOGIT rung ... -> M1_LOGIT
       |
       +-> RSET rung specialists  -> reducer -> next RSET rung  ... -> M1_RSET
       |
       +-> RREL rung specialists  -> reducer -> next RREL rung  ... -> M1_RREL
  -> M1E reducer
  -> M2
  -> validation aggregate
  -> campaign-complete report
```

At a triangular rung, all specialists start together after every required
teacher probability bank and carrier checkpoint validates. The reducer starts
only after every declared specialist succeeds. Poor metrics cannot remove a
member. The next rung starts only after the reducer lock exists.

All workers use account `reu-aisocial`, partition `tigris`, a clean detached
full-commit worktree, `atlas_kd_tigris`, `PYTHONNOUSERSITE=1`, and
`${CONDA_PREFIX}/lib` first in `LD_LIBRARY_PATH`. Workers source helpers from
the absolute `${PROJECT_DIR}` and end with `exec python -s`.

## 14. Pairing and seeds

At the same target coordinate, all specialists in a track share the same
backbone initialization, sampler, dropout, trimmer, optimizer, validation
order, and repair seed aliases. Only the declared teacher differs.

Across LOGIT, RSET, and RREL, nodes at the same coordinate share the same
backbone/data seed alias where their topology permits a paired comparison.
Representation head, random Fourier resource, and relation resource seeds use
strategy-specific namespaces and may not advance shared streams.

The three M1 nodes share one exact-HLT backbone/data seed alias. M2 has a new
terminal seed alias. Reducers are deterministic and have no scientific RNG.

Changing a seed alias or allowing job order/array index to define a seed is a
scientific version change.

## 15. Contracts and artifact family

Implementation introduces an additive family, provisionally:

```text
HCWDL_MHPE_THREE_TRACK_60E_GRAPH/v1
HCWDL_MHPE_THREE_TRACK_60E_NODE_SPEC/v1
HCWDL_MHPE_THREE_TRACK_60E_RECIPE/v1
HCWDL_MHPE_THREE_TRACK_60E_INTEGRATION_LOCK/v1
HCWDL_MHPE_THREE_TRACK_60E_FOUNDATION_LOCK/v1
HCWDL_MHPE_THREE_TRACK_60E_ENDPOINT_RESOURCE_LOCK/v1
HCWDL_MHPE_THREE_TRACK_60E_PROBABILITY_SHARD/v1
HCWDL_MHPE_THREE_TRACK_60E_PROBABILITY_MANIFEST/v1
HCWDL_MHPE_THREE_TRACK_60E_PROBABILITY_LOCK/v1
HCWDL_MHPE_THREE_TRACK_60E_EPHEMERAL_REP_AUDIT/v1
HCWDL_MHPE_THREE_TRACK_60E_TRAINING_REPORT/v1
HCWDL_MHPE_THREE_TRACK_60E_STAGE_REPORT/v1
HCWDL_MHPE_THREE_TRACK_60E_CAMPAIGN_SPEC/v1
HCWDL_MHPE_THREE_TRACK_60E_COMMAND_PLAN/v1
HCWDL_MHPE_THREE_TRACK_60E_RUNTIME_PROFILE/v1
HCWDL_MHPE_THREE_TRACK_60E_AGGREGATE/v1
HCWDL_MHPE_THREE_TRACK_60E_FINALIST_LOCK/v1
HCWDL_MHPE_THREE_TRACK_60E_EXECUTION_LOCK/v1
HCWDL_MHPE_THREE_TRACK_60E_CAMPAIGN_COMPLETE/v1
HCWDL_MHPE_THREE_TRACK_60E_RECOVERY_SPEC/v1
HCWDL_MHPE_THREE_TRACK_60E_RESOURCE_RECOVERY_SPEC/v1
```

The recipe references exact hashes of the authorized 60-pass HCWDL base
recipe, `HCWDL_REPRESENTATION_RECIPE/v5`, unified-balanced view contracts,
probability-ensemble numerical policy, and no-resume execution policy. It
contains a complete per-node table of view, teacher, carrier, loss,
temperature, auxiliary, seed aliases, pass count, and deployability.

Existing assignment, residual-coupling, balanced-sidecar, uniform-D,
checkpoint-selection, metrics, submission-ledger, task-attestation, and
monitor contracts may be reused only after exact hash validation. Existing
MHPE or U-RKD contracts must not be broadened to accept this graph.

## 16. Recovery and cancellation

Source-pinned recovery must exist before live submission. It binds the
original spec, command plan, ledger, immutable monitor, exact source, graph,
recipe, foundation, probability locks, and all completed reports.

Recovery computes the exact failed/downstream closure:

- a failed specialist preserves completed siblings but invalidates its rung
  reducer and descendants;
- a failed reducer preserves all component checkpoints;
- a failed RSET/RREL fit restarts that fit from zero and rebuilds its
  representation targets in RAM;
- a failed track does not cancel independent work already running in the
  other tracks;
- a failed M1E or M2 preserves all three completed M1 models; and
- repeated recovery binds the complete prior recovery chain.

Source repair may change only reviewed execution files under a separately
versioned, exact-file allowlist and must prove scientific semantics unchanged.
Resource recovery changes only declared resources. Neither may import partial
training state.

Cancellation accepts exact job IDs from one campaign-bound ledger. Broad job
name matching is forbidden. A cancellation report records every exact ID and
whether it was pending, running, or already terminal.

## 17. Reporting

The aggregate reports every specialist, ensemble, M1, M1E, and M2 with:

- CE, accuracy, balanced accuracy, macro OVR AUC;
- per-class OVR AUC;
- macro mean log QCD rejection and geometric-mean R50;
- per-signal Hbb, Hcc, H4q, Hq, Hgg, and other registered class rejections;
- Brier score, ECE, confusion matrix, selected pass, and full history;
- GPU-hours, walltime, cache/target preparation time, peak RSS, peak CUDA
  memory, probability-bank bytes, and durable campaign bytes; and
- `final_test_accessed`.

For every same-view ensemble it reports:

- ensemble minus mean specialist;
- ensemble minus best specialist;
- local-carrier specialist versus skip specialists;
- every fixed leave-one-out diagnostic without changing primary weights;
- pairwise probability correlation, Jensen-Shannon divergence, prediction
  disagreement, and classwise disagreement; and
- carrier identity and representation support diagnostics.

Primary endpoint comparisons are:

```text
LOGIT_D000E - best LOGIT D000 specialist
RSET_D000E  - best RSET D000 specialist
RREL_D000E  - best RREL D000 specialist

M1_LOGIT - LOGIT_D000E
M1_RSET  - RSET_D000E
M1_RREL  - RREL_D000E

M1E - best M1 member
M2  - M1E
M2  - M0paired
M2  - existing same-population exact-HLT controls
```

Recovery of the paired `M0paired -> U000` gap is reported without clipping.
Macro AUC is primary. R50 and per-class rejection are essential companions
but are treated as noisier tail metrics. The imported full-data `M0paired`
is a contextual denominator, not a pass-matched 60-pass control; reports must
show its pass count and may not attribute an M2-minus-M0 difference solely to
KD. Adding a fresh 60-pass CE-only HLT control would be a separately
registered, nonblocking ablation rather than an undeclared 33rd fit. One seed
remains exploratory.

One additive post-hoc validation diagnostic is also predeclared after the
LOGIT and RSET D000 reducers complete. It reads the immutable
`LOGIT_D000E` and `RSET_D000E` validation probability banks, assigns each
exact rational weight 1/2, accumulates in lexical order and FP64, casts the
average once to FP32, and evaluates it on the authenticated validation rows.
`M0CE60` is the zero-recovery reference and `U000` is the one-recovery
reference; R50 recovery is calculated in linear rejection space. This
diagnostic is exploratory and cannot select a campaign model, alter the graph,
create a fit or deployable artifact, add a scheduler dependency, or access
final test. Its result is scientifically distinct from either track's internal
same-view ensembles.

After observing the predeclared equal-family result, one fixed member-count
follow-up is registered without changing the campaign graph. `LOGIT_D000E`
contains five uniformly weighted specialists and `RSET_D000E` contains three.
The follow-up therefore assigns the durable family banks weights 5/8 and 3/8,
giving all eight underlying specialists nominal effective weight 1/8. The
weights come only from the frozen member counts and are not selected using
validation performance. The report embeds the 50/50 result as a comparator,
uses the same `M0CE60`/`U000` recovery references, and explicitly records the
small FP32 family-bank rounding boundary. It remains validation-only,
post-hoc, non-selection-eligible, and isolated from all ladder scheduling and
artifacts.

## 18. Deployability and final evaluation

All U nodes and nonzero-D nodes are privileged training-time models. Every
`D000` specialist, `D000E`, all three M1 models, `M1E`, and M2 consume exact
canonical HLT inputs and are technically HLT-compatible.

The designated single-model result is M2. Its deployable extraction contains
only the ordinary unified 21-channel HLT Particle Transformer. Training-only
projection heads, carrier IDs, homotopy coordinates, assignments, offline
fields, probability targets, and representation targets are physically
absent.

The validation campaign does not automatically access final test. A later
finalist plan should register at minimum `M0paired`, the three D000 ensembles,
the three M1 models, M1E, and M2. The exact finalist list must be frozen before
the execution lock; no validation result may cause an undeclared model to be
silently added after test access begins.

## 19. Implementation map

Preferred reusable modules are additive:

| Surface | Responsibility |
|---|---|
| integration module | authenticate merged MHPE, UB, and representation source semantics |
| graph module | exact 32-fit registry, 12 reducers, teacher/carrier/view/seed table |
| recipe module | 60-pass base, representation v5 overlay, M1/M2 rules, no-resume policy |
| probability module | T1/T2 reduction, shards, manifests, locks, cleanup, identity joins |
| ephemeral representation module | carrier streaming, RAM targets, calibration, audit, release |
| training runner | LOGIT/RSET/RREL cold fits, in-memory best state, terminal extraction |
| campaign/workflow | foundation, task dispatch, DAG, command plan, source/resource locks |
| reporting | track, ensemble, diversity, compression, recovery, runtime, storage tables |
| recovery | exact restart-from-zero failed/downstream closure |
| Slurm | thin absolute-path workers and durable submission journal |

The implementation should reuse the corrected prepared-endpoint stream,
batched set/relation kernels, one-forward surfaces, probability-KD adapter,
metrics, and immutable publication helpers. It must not duplicate scientific
math in worker scripts.

### 19.1 Required integration sequence

Implementation proceeds in this order:

1. **Commit and bind the representation source.** Produce a clean commit from
   the corrected representation worktree and record its full hash. Do not use
   its current uncommitted path as lineage.
2. **Merge the source lines.** Bring the representation model surfaces,
   kernels, losses, calibration, target runtime, and training wrapper into the
   current MHPE source tree. Resolve conflicts in favor of this plan and the
   two parent scientific authorities, not file recency.
3. **Prove unchanged parents.** Before adding the new graph, run the existing
   MHPE probability tests and the representation math/model/training tests.
   Prove old MHPE logits/gradients, RSET/RREL losses/gradients, and U/D
   endpoint bytes remain unchanged.
4. **Add the new contracts and graph.** Register all 32 fits, 12 reducers,
   exact teacher/carrier table, coordinates, losses, seeds, resources, and
   access capabilities as data. No scientific edge may be inferred from an
   ID string.
5. **Add the ephemeral target adapter.** Reuse
   `prepare_target_generation_in_memory` and the target schemas, but return an
   authenticated in-process object rather than calling the durable shard
   publisher. Validate it against the old durable-bank reference on bounded
   identical batches.
6. **Add the combined specialist runner.** Start with MHPE's probability-
   target/student-view runner, wrap only RSET/RREL students in
   `HCWDLRepresentationStudent`, load the graph-selected carrier, generate
   RAM targets, and pass base plus auxiliary terms through the existing
   representation loss implementation.
7. **Add no-resume terminal publication.** Reuse validation and checkpoint
   selection, retain current-best and final state in RAM, and publish selected
   and final envelopes only after all 60 passes. Do not call the rolling
   resume-generation writer.
8. **Add reducers and terminal compression.** Reuse the MHPE probability
   reducer unchanged for all 12 ensembles. RSET/RREL training heads never
   enter a reducer. Train M2 from the fixed T1 M1E probability bank.
9. **Add campaign operations.** Implement one source-pinned campaign spec,
   command plan, Slurm DAG, monitor, exact-ID cancellation, restart-from-zero
   recovery, aggregate, completion, and sealed-test denial.
10. **Close acceptance.** Run focused tests, the complete repository suite,
    installed-Weaver parity, bounded production-worker RAM/runtime profiling,
    and the complete nonmutating all-data dry run before requesting live
    authorization.

The focused source-integration baseline must include at least:

```text
tests/test_hcwdl_representation_math.py
tests/test_hcwdl_representation_model.py
tests/test_hcwdl_representation_training.py
tests/test_hcwdl_homotopy_representation.py
tests/test_hcwdl_homotopy_representation_profile.py
tests/test_hcwdl_mhpe.py
tests/test_hcwdl_unified_balanced.py
tests/test_hcwdl_unified_balanced_full.py
```

If file names change during the clean merge, the integration attestation maps
the old authoritative test to its new tracked replacement rather than simply
dropping coverage.

## 20. Required tests

### 20.1 Graph and recipe

- exact 32 fits, 12 reducers, and complete node list;
- exact LOGIT `1+2+3+4+5`, RSET `1+2+3`, and RREL `1+2+3` specialist counts;
- exact teachers, carriers, coordinates, temperatures, losses, and seeds;
- one shared fresh U000 and no imported 20-pass U000;
- 60 passes and 60 validations for every fit;
- C25P75/T2 for all U/D specialists;
- C10P90/T1 for all M1 and M2 nodes;
- RSET/RREL auxiliary only on their own nodes;
- no warm starts, hidden teachers, result-driven pruning, or string-parsed
  scientific routing.

### 20.2 Views, surfaces, and representation math

- exact rational coordinates and nested balanced/discrete switches;
- U100 byte equality to D100 and D000 byte equality to HLT, including raw
  lengths above 200;
- ordinary one-forward logit/surface parity;
- RSET permutation invariance and zero particle-match access;
- RREL raw-state gradient reaches backbone and not set projection;
- exact v5 schedules, kernels, strata, support, calibration, and weak-support
  continuation;
- carrier always runs on its own view and joins only by jet identity.

### 20.3 Probability and ephemeral targets

- hand-calculated one- through five-member probability ensembles;
- canonical FP64 accumulation and one FP32 publication;
- T1/T2 semantics and no double-temperature transform;
- identity/class/order/normalization/corruption checks;
- exact carrier table for every RSET/RREL consumer;
- RAM target equality to the existing durable-bank reference on bounded
  fixtures;
- representation targets constructed once per job and never written;
- cleanup/release and peak-memory accounting;
- no validation/final representation target access.

### 20.4 No-resume training

- no rolling-state files after successful, failed, or interrupted fits;
- in-memory best selection equals the durable-resume reference on a bounded
  deterministic run;
- USR1 writes only an interruption attestation and exits nonzero;
- recovery restarts at update zero with exact seeds;
- partial checkpoint injection is rejected;
- selected/final publication is atomic and deployable extraction has logit
  parity with the training wrapper.

### 20.5 Campaign and operations

- exact topological dependencies and track/rung parallelism;
- clean-source/full-commit validation and no sibling-worktree imports;
- all-mapped identity coverage and zero final-test capability;
- resource projection includes every simultaneous RAM resident;
- dry-run nonmutation and durable partial-submission journal;
- poor finite metrics preserve descendants;
- exact-ID cancellation and repeated source/resource recovery;
- CLI help, contract inventory, shell syntax, Markdown links, compilation,
  full repository tests, and `git diff --check`.

## 21. Acceptance and launch boundary

Implementation is queue-ready only after:

1. the two source lines are merged into one clean committed runtime tree;
2. focused MHPE, UB, and representation suites pass before and after changes;
3. the complete local synthetic 32-fit/12-reducer graph passes with tiny data;
4. RAM-only targets match the existing target-bank reference numerically;
5. no-resume interruption and restart-from-zero recovery pass;
6. installed-Weaver parity passes on Tigris;
7. one bounded real production-worker acceptance/profile proves the new
   RAM-only target and no-resume path, peak memory, and walltime projection;
8. the complete repository suite, CLI/help, contract-version, shell/static,
   Markdown-link, and diff checks pass;
9. a full nonmutating all-data dry run renders exact resources, dependencies,
   paths, and IDs; and
10. `docs/HANDOFF.md` records exact evidence and remaining authorization.

No standalone reduced scientific ladder or 300k pilot is required by this
plan. The bounded production-worker acceptance in item 7 is an operational
test of genuinely new execution behavior, not a second research campaign.

This document does not authorize a Git push, Slurm submission, cancellation,
or final-test access. Live submission still requires a clean pushed commit,
an authenticated immutable candidate, adequate free storage, a complete dry
run, and the user's explicit phrase-bound authorization.

## 22. Claims and limitations

Allowed claims after successful validation execution are limited to:

- full-data 60-pass LOGIT, RSET, and RREL triangular knowledge transfer on
  the declared views;
- fixed, uniform same-view probability ensembles;
- deterministic single-carrier representation supervision;
- validation-only three-track complementarity and compression; and
- HLT-only D000/M1/M2 inference.

The campaign may not claim that:

- an ensemble has a literal averaged internal representation;
- the carrier is the uniquely correct representation of an ensemble;
- RSET/RREL uses physical particle correspondence;
- U000 is native TOFF or an offline deployable model;
- representation KD alone causes ensemble diversity;
- a 60-pass gain is caused only by duration rather than total optimization;
- uniform weights are optimal;
- success reconstructs unavailable offline information at inference;
- failure proves an information-theoretic ceiling; or
- the 6.5-day target is guaranteed.

The central scientific result is the complete path and its fixed controls,
not a retrospective choice of whichever track happens to score best.

## 23. Definition of done

The implementation is complete only when the exact graph, contracts, merged
source, probability banks, RAM-only representation targets, no-resume
training, Slurm DAG, reports, recovery, and tests above exist and pass. Until
then the correct status is **planned, not queue-ready**.
