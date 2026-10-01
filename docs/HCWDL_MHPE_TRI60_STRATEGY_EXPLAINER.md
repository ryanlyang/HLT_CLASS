# The HCWDL-MHPE TRI60 strategy

## A reader's guide to the full-data, three-track triangular KD ensemble ladder

This document explains the scientific strategy behind the full-data,
60-pass HCWDL multi-horizon projection-ensemble campaign (TRI60). It is meant
to make the campaign understandable from the outside: what problem it is
trying to solve, what every rung means, why there are multiple students at one
rung, what the `E` suffix means, how representation supervision is added, and
how the final result becomes one deployable HLT-only model.

This is a descriptive guide, not a new scientific authority. Exact semantics
remain governed by the
[implementation plan](plans/HCWDL_MHPE_THREE_TRACK_60E_FULL_IMPLEMENTATION_PLAN.md),
[versioned contract](contracts/HCWDL_MHPE_THREE_TRACK_60E_FULL.md), immutable
campaign specification, and
[research-compute runbook](HCWDL_MHPE_THREE_TRACK_60E_FULL_RUNBOOK.md), in that
order.

## 1. The idea in one paragraph

TRI60 starts with a strong unified Particle Transformer trained on the richest
projected-offline particle view, then progressively moves the model's input
toward exact HLT particles. At every lower-information coordinate, it does not
trust only the immediately preceding model. Instead, it trains one fresh
student from every declared earlier teacher horizon that can illuminate that
target view. Those students all see the same target inputs, but inherit
different predictive perspectives through knowledge distillation. Their class
probabilities are averaged with fixed equal weights, and that same-view
ensemble becomes a teacher for subsequent rungs. Three variants perform this
transfer: logits alone, logits plus unordered set-level representation
supervision, and logits plus relation-aware representation supervision. The
three exact-HLT endpoints are finally ensembled and distilled into one ordinary
HLT-only `M2` model.

The strategy is therefore not merely a longer KD chain. It is a triangular,
multi-horizon projection system followed by ensemble compression.

## 2. The problem the ladder is designed to solve

Offline particles contain information that is absent or degraded in HLT
particles. A model trained directly on the richer view can therefore form a
better decision function, but that model is not deployable if it needs offline
inputs at inference.

A simple solution would be a sequential ladder:

```text
rich input -> slightly degraded input -> ... -> exact HLT input
```

with each model distilling into the next. That helps make each individual
transition smaller, but it creates a telephone-game failure mode. Every model
inherits the strengths and mistakes of only one predecessor. Information lost
at an early edge may never be recovered, and a locally weak predecessor can
become a bottleneck for every later rung.

TRI60 replaces that single chain with a triangular teacher lattice. At a given
target view, it asks several distinct questions in parallel:

- What can the original high-information teacher project directly onto this
  target?
- What can a nearer teacher transfer after already adapting to part of the
  domain change?
- What can each intermediate ensemble contribute after combining several
  projection histories?

The answers become separate, freshly initialized specialists. Since they all
consume exactly the same target view, their output probabilities can be
combined without mixing input domains.

## 3. The input homotopy: what U and D mean

Every view is an exact unified-balanced coordinate

```text
V_UB(s, f)
```

where:

- `s` controls structural progress from the projected-native-offline particle
  support toward the complete D100 support; and
- `f` controls uniform progress of matched particle fields from their offline
  values toward their HLT values.

The important views are:

| Name | Exact coordinate | Interpretation |
|---|---|---|
| `U000` | `V_UB(0,0)` | richest projected-native-offline unified input |
| `U050` | `V_UB(1/2,0)` | half the frozen structural information mass has switched |
| `U100` | `V_UB(1,0)` | full D100 support, with offline matched-field values |
| `D066` | `V_UB(1,1/3)` | matched fields retain two-thirds offline strength |
| `D050` | `V_UB(1,1/2)` | matched fields are halfway from offline to HLT |
| `D033` | `V_UB(1,2/3)` | matched fields retain one-third offline strength |
| `D000` | `V_UB(1,1)` | byte-exact canonical HLT input |

The naming reflects how much privileged information remains, not the numeric
value of `f`. Thus `D066` means approximately 66% offline matched-field
strength even though its exact stored progress toward HLT is `1/3`.

The U transition changes particle support in a balanced, deterministic order.
It is calibrated by switched information mass rather than by blindly removing
the same number of particles at every rung. Its frozen edit list includes
substitutions, insertions, and removals. Each edit receives a positive mass
from its particle count, momentum/energy share, and endpoint disruption, then
a deterministic balanced switch coordinate. At structural progress `s`, all
edits at or below that coordinate are active. The active set is nested, so an
edit switches once and never reverses; `U000` explicitly applies none and
`U100` applies all. A single indivisible edit can still cause a local jump,
which is measured rather than hidden.

The D transition then moves all matched continuous fields together along the
declared interpolation and uses deterministic nested switches for discrete
fields. Match confidence does not decide which D fields change first.
Consequently, later D coordinates are strict, reproducible continuations
rather than newly sampled corruptions.

`U000` is not native TOFF. It is a fresh unified 21-channel Particle
Transformer trained on the richest projected-offline view. `D000` is the
critical deployment boundary: it consumes exact HLT particles and no offline
input.

## 4. How to read a node name

Consider:

```text
LOGIT_D066_from_U050E
```

It means:

- train a fresh model at the `D066` input coordinate;
- give it ordinary classification KD from the `LOGIT_U050E` probability
  teacher; and
- do not initialize it from the teacher's weights.

The arrow is supervision, not parameter continuation. Every student is cold
started. Optimizer state, model weights, sampler state, and RNG state are not
inherited.

The suffix `E` denotes a durable class-probability ensemble artifact. It is not
another trained model. For example:

```text
LOGIT_U050_from_U000   = one trained specialist
LOGIT_U050E            = its packaged one-member probability teacher
```

The singleton artifact is intentional. It gives every downstream edge the
same authenticated probability-bank interface, whether the preceding rung has
one member or five. A tiny difference between a specialist's reported raw-logit
metrics and its singleton `E` metrics can arise because the reducer evaluates
normalized probabilities; it does not imply a second training run.

## 5. The triangular operation at one rung

Suppose the LOGIT track has reached `D066`. Three earlier teacher horizons are
available:

```text
U000
LOGIT_U050E
LOGIT_U100E
```

TRI60 trains three fresh `D066` students in parallel:

```text
LOGIT_D066_from_U000
LOGIT_D066_from_U050E
LOGIT_D066_from_U100E
```

The students use the same `D066` view, architecture, initialization seed
alias, sampler, optimizer, pass count, and validation ordering. Their declared
teacher is the controlled difference. After all three finish, the reducer
forms:

```text
LOGIT_D066E = uniform_probability_mean(
    LOGIT_D066_from_U000,
    LOGIT_D066_from_U050E,
    LOGIT_D066_from_U100E,
)
```

That ensemble is now a same-view `D066` teacher. It joins the permanent set of
teacher horizons available to lower rungs.

This creates two kinds of protection:

1. **Long skip teachers** test whether a distant strong teacher can still
   project useful knowledge directly onto the target view.
2. **Local teachers** test whether a nearby domain-adapted teacher makes KD
   easier and preserves finer information.

No validation result decides which specialist survives. Every predeclared
member remains in the ensemble even if its individual score is poor. This is
essential: otherwise the ensemble and downstream graph would be selected on
the same validation data used to report them.

## 6. The complete LOGIT triangle

The LOGIT track uses ordinary class-distribution KD only. Its primary path is:

```text
U000
  -> LOGIT_U050E
  -> LOGIT_U100E
  -> LOGIT_D066E
  -> LOGIT_D033E
  -> LOGIT_D000E
  -> M1_LOGIT
```

The apparent line hides the widening triangle underneath:

| Target ensemble | Fresh specialists at that target |
|---|---|
| `LOGIT_U050E` | from `U000` |
| `LOGIT_U100E` | from `U000`, `LOGIT_U050E` |
| `LOGIT_D066E` | from `U000`, `LOGIT_U050E`, `LOGIT_U100E` |
| `LOGIT_D033E` | from `U000`, `LOGIT_U050E`, `LOGIT_U100E`, `LOGIT_D066E` |
| `LOGIT_D000E` | from `U000`, `LOGIT_U050E`, `LOGIT_U100E`, `LOGIT_D066E`, `LOGIT_D033E` |

This is the `1 + 2 + 3 + 4 + 5` construction: 15 view-projection specialists.
It is deliberately dense in teacher horizons rather than only dense in input
coordinates.

At `D000`, all five specialists consume identical byte-exact HLT inputs. Their
diversity comes from their supervision histories, not from different deployed
features. `LOGIT_D000E` therefore measures whether several projection histories
can collectively preserve more privileged knowledge at the exact HLT endpoint
than one history alone.

`M1_LOGIT` is then a fresh exact-HLT model trained to compress
`LOGIT_D000E`. It is not an additional homotopy coordinate. It asks whether a
single ordinary model can absorb the ensemble's distribution.

## 7. The RSET and RREL triangles

LOGIT KD transfers only the teacher's final class distribution. It does not
explicitly communicate how the teacher organizes particles internally. TRI60
therefore runs two shorter parallel tracks that retain the same probability-KD
base but add matching-free representation supervision.

### 7.1 RSET

RSET transfers:

- the global jet representation; and
- an unordered summary of the teacher's token set.

It does not require particle-to-particle correspondence. The track is:

```text
U000
  -> RSET_U100E
  -> RSET_D050E
  -> RSET_D000E
  -> M1_RSET
```

with a `1 + 2 + 3` triangle:

| Target ensemble | Fresh specialists at that target |
|---|---|
| `RSET_U100E` | from `U000` |
| `RSET_D050E` | from `U000`, `RSET_U100E` |
| `RSET_D000E` | from `U000`, `RSET_U100E`, `RSET_D050E` |

### 7.2 RREL

RREL transfers the same broad jet and set information, then adds a relational
description of token geometry from the corrected raw block-2 states. Its
topology is identical:

```text
U000
  -> RREL_U100E
  -> RREL_D050E
  -> RREL_D000E
  -> M1_RREL
```

Again, the endpoint is a three-member exact-HLT ensemble.

RREL does not use HLT/offline matching indices as a shortcut. Relations are
matching-free summaries, joined to the student only by the authenticated jet
identity.

## 8. Why ensembles and representation carriers are separate

A probability ensemble has a coherent prediction:

```text
mean of member class probabilities
```

It does not have one coherent internal token representation. Averaging hidden
states from separately trained models would require arbitrary alignment and
would falsely suggest that the ensemble owns a literal latent state.

For this reason, every RSET/RREL edge has two independent supervision sources:

1. the declared full ensemble supplies class-probability KD; and
2. one frozen, deterministic carrier model supplies jet/set/relation targets.

For a track `X` equal to RSET or RREL, the carriers are:

| Distribution teacher | Fixed representation carrier |
|---|---|
| `U000` | `U000` |
| `X_U100E` | `X_U100_from_U000` |
| `X_D050E` | `X_D050_from_U100E` |
| `X_D000E` | `X_D000_from_D050E` |

The carrier is the local-predecessor projection specialist: the member most
directly aligned with the causal path into the ensemble. It is fixed in the
graph, never selected by validation score, and always evaluated on its own
authenticated input view. The carrier and student are joined by the same
32-byte jet identity.

This separation is one of the campaign's most important design choices. It
lets the student receive the ensemble's broader predictive consensus without
pretending that an ensemble has an averaged representation.

## 9. Losses and training recipe

Every view-changing LOGIT, RSET, and RREL specialist uses:

```text
L_base = 0.25 * CE
       + 0.75 * T^2 * KL(teacher_T || student_T)

T = 2
```

CE is unweighted per-jet cross entropy. The KD target may come from one
checkpoint or from a durable probability bank. An ensemble probability is
consumed directly; it is not converted back into invented logits or subjected
to temperature twice.

RSET adds an auxiliary term with total coefficient 0.10:

```text
L_RSET = 0.40 * calibrated_jet_loss
       + 0.60 * calibrated_set_loss
       + 0.001 * orthogonality_loss
```

RREL uses the same total auxiliary budget but allocates it across jet, set,
and relation information:

```text
L_RREL = 0.30 * calibrated_jet_loss
       + 0.45 * calibrated_set_loss
       + 0.25 * calibrated_relation_loss
       + 0.001 * orthogonality_loss
```

Jet/set supervision is off through pass 2 and ramps to full strength by pass
6. Relation supervision is off through pass 4 and reaches full strength by
pass 8. RSET simply applies the jet/set ramp to its components. RREL uses the
exact v5 budget-preserving reallocation: while the relation term ramps in, its
weight is removed from the common 40/60 jet/set package instead of being added
on top. This prevents a large randomly initialized representation head from
dominating early classification optimization and keeps the intended auxiliary
budget controlled.

Every fit uses:

- fresh initialization;
- 60 natural-population passes;
- validation after every pass;
- batch size 256;
- AdamW with peak learning rate `3e-4` and weight decay `0.01`;
- 5% warmup followed by cosine decay to a 5% learning-rate floor;
- BF16 model forward with FP32 loss and metric arithmetic; and
- no performance early stopping.

Checkpoint selection is lexicographic: highest macro OVR AUC, then lowest CE,
then highest macro mean log-R50, then earliest update.

## 10. Exact probability ensembling

For `K` selected specialists with logits `z_k(x)`, the ensemble is:

```text
p_E,T(x) = (1/K) * sum_k softmax(z_k(x) / T)
```

The weights are always exactly `1/K`. They cannot depend on validation score,
class, confidence, entropy, disagreement, row, or rung. Components are ordered
canonically, accumulated in FP64, and rounded once to FP32 for publication.

This is probability averaging, not logit averaging, checkpoint averaging, or
representation averaging. Probability averaging allows independently trained
models to express different uncertainty structures while producing a valid
class distribution.

U/D ensembles publish the temperature-2 train targets required by their
children and temperature-1 validation probabilities for evaluation. M1E and
M2 use temperature 1. The temperature transform is applied exactly once.

## 11. Terminal compression: M1E and M2

Each track first compresses its exact-HLT endpoint ensemble into one fresh
exact-HLT model:

```text
LOGIT_D000E -> M1_LOGIT
RSET_D000E  -> M1_RSET
RREL_D000E  -> M1_RREL
```

The three M1 models share the same base compression recipe and paired
backbone/data seed alias:

```text
0.10 CE + 0.90 probability KD
temperature 1
60 passes
```

RSET and RREL retain their declared representation auxiliaries during their
M1 compression. LOGIT does not.

The terminal ensemble is:

```text
M1E = uniform_probability_mean(
    M1_LOGIT,
    M1_RSET,
    M1_RREL,
)
```

Finally:

```text
M1E -> M2
```

`M2` is a fresh ordinary exact-HLT Particle Transformer trained with 10% CE
and 90% temperature-1 KD from M1E. It receives no representation auxiliary,
because M1E has no unique representation carrier. This last step is the
deployment compression: it attempts to retain the complementary gains of all
three tracks in one model with one forward pass.

## 12. Why this strategy is scientifically attractive

Several ideas reinforce one another:

### Multi-horizon transfer limits compounding loss

A pure chain exposes every later model to every earlier mistake. The triangle
keeps long-range teachers alive, so lower rungs can recover knowledge that a
local predecessor failed to preserve.

### Near teachers make difficult domain transitions easier

A local teacher has already adapted to a similar view. Its decision boundary
may therefore be easier for the student to imitate than the richer root's
decision boundary. TRI60 tests both the long-range and local hypotheses rather
than assuming one is universally correct.

### Same-view specialists isolate supervision history

Members of one ensemble consume the same bytes and use paired training seeds.
Their principal controlled difference is teacher history. Ensemble diversity
therefore has an interpretable source.

### Probability ensembles turn several imperfect projections into one teacher

Different projection histories may preserve different class boundaries or
different regions of phase space. Fixed probability averaging can reduce
idiosyncratic errors and create a smoother downstream KD target.

### Representation tracks probe knowledge that logits may omit

Two teachers can produce similar class probabilities while organizing their
particle representations differently. RSET asks whether global and unordered
set structure adds useful supervision. RREL asks whether pairwise relational
geometry adds something further.

### The final model remains operationally simple

The campaign may use offline information, ensembles, and representation
targets during training, but the designated endpoint is one standard unified
HLT Particle Transformer. Complexity is spent at training time rather than at
trigger inference time.

## 13. Scale of the experiment

The campaign contains exactly:

| Component | Fresh fits |
|---|---:|
| shared `U000` | 1 |
| LOGIT specialists plus `M1_LOGIT` | 16 |
| RSET specialists plus `M1_RSET` | 7 |
| RREL specialists plus `M1_RREL` | 7 |
| terminal `M2` | 1 |
| **Total** | **32** |

It also has 12 fixed probability reducers: five LOGIT rung ensembles, three
RSET rung ensembles, three RREL rung ensembles, and M1E. The shared U000
probability publication is additional infrastructure, not another fit.

The authenticated full-data population contains approximately 2.78 million
train jets and 0.96 million validation jets in the current foundation. The
ordinary campaign has no final-test capability.

## 14. Storage and execution boundaries

The campaign persists compact class-probability banks, selected/final model
checkpoints, small reports, hashes, locks, ledgers, and attestations.

It does not persist reconstructed particle views or RSET/RREL representation
targets. Representation targets are generated once per job in host RAM from
the fixed carrier, used for all 60 passes, and released when the job exits.
Student train and validation views are likewise RAM-resident. This avoids the
multi-gigabyte durable target banks that would otherwise exhaust research-
compute storage.

Rolling training resumes are intentionally disabled. A successful fit writes
only terminal selected and final checkpoints. If a fit fails, an authenticated
recovery restarts that fit from update zero with the same scientific seeds;
completed sibling fits and probability banks remain immutable.

Full-population view construction is source-process parallel. That operational
optimization changes walltime, not view bytes, ordering, sampler replay, or
scientific identity. The subsequent single-GPU training or inference phase is
expected to use far fewer CPUs.

## 15. What the 300k predecessor experiments already showed

### 15.1 Multi-horizon LOGIT ensembling

TRI60 was not motivated only by theory. Its closest completed predecessor was
the 300k-train/100k-validation, 60-pass C25P75 multi-horizon LOGIT campaign.
That experiment used the same central idea: train several same-view students
from different teacher horizons, average their probabilities uniformly, and
use the result as the next teacher. It did not yet contain the full TRI60
RSET/RREL branches, so it is evidence for the triangular LOGIT mechanism, not
advance evidence that every TRI60 component will succeed.

The relevant validation progression was:

| Rung | Accuracy | Macro AUC | Macro R50 | AUC recovery |
|---|---:|---:|---:|---:|
| `M0paired` | 0.791330 | 0.936951 | 1230.3 | 0.0% |
| `U000` | 0.804820 | 0.948409 | 1677.8 | 100.0% |
| `U050_from_U000` | 0.805280 | 0.948017 | 1662.6 | 96.6% |
| `U100E` | 0.802460 | 0.946350 | 1553.3 | 82.0% |
| `D066E` | 0.800910 | 0.944528 | 1525.0 | 66.1% |
| `D033E` | 0.798740 | 0.943178 | 1483.9 | 54.3% |
| `D000E` | 0.796270 | 0.940796 | 1435.6 | 33.6% |
| `M1` | 0.795890 | 0.941237 | 1432.0 | 37.4% |

Here, recovery used the experiment's paired `M0paired -> U000` AUC gap. The
important observation is not that performance remained flat; information was
still lost as the input became HLT-like. The important observation is that the
multi-horizon ensemble consistently retained more than the corresponding
basic local KD edge.

For a direct comparison, define **basic local KD** at a rung as the fresh
student taught only by the nearest preceding ensemble. Then:

| Target | Basic local-KD specialist | Local AUC | Ensemble AUC | AUC gain | Local recovery | Ensemble recovery | Recovery gain |
|---|---|---:|---:|---:|---:|---:|---:|
| `U100` | `U100_from_U050` | 0.945745 | 0.946350 | +0.000605 | 76.7% | 82.0% | +5.3 pp |
| `D066` | `D066_from_U100E` | 0.943989 | 0.944528 | +0.000539 | 61.4% | 66.1% | +4.7 pp |
| `D033` | `D033_from_D066E` | 0.941924 | 0.943178 | +0.001254 | 43.4% | 54.3% | +10.9 pp |
| `D000` | `D000_from_D033E` | 0.939914 | 0.940796 | +0.000882 | 25.9% | 33.6% | +7.7 pp |

The same comparison in R50 was also favorable:

| Target | Basic local-KD R50 | Ensemble R50 | Ensemble gain |
|---|---:|---:|---:|
| `U100` | 1496.6 | 1553.3 | +56.7 |
| `D066` | 1472.6 | 1525.0 | +52.4 |
| `D033` | 1389.7 | 1483.9 | +94.2 |
| `D000` | 1411.7 | 1435.6 | +23.9 |

Moreover, every multi-member ensemble beat its best individual member in
macro AUC. The gains over the best member were approximately `+0.000605` at
U100, `+0.000539` at D066, `+0.001025` at D033, and `+0.000882` at D000. That
matters because it shows that the benefit was not merely caused by one lucky
skip teacher hiding inside the ensemble. The fixed average extracted useful
complementarity from the different projection histories.

The endpoint compression supplied a second encouraging result. Distilling
`D000E` into the single `M1` model increased macro AUC from `0.940796` to
`0.941237`, moving recovery from 33.6% to 37.4%. R50 changed slightly in the
opposite direction, from 1435.6 to 1432.0, so the correct conclusion is that
compression improved the primary AUC metric without dominating on every
metric. It nevertheless demonstrated that a single HLT model can sometimes
generalize beyond the literal average that taught it.

These are meaningful but preliminary results. They came from one 300k
training campaign, one seed configuration, validation rather than sealed
final-test data, and LOGIT KD only. They support the claim that the triangular
ensemble idea has real empirical teeth relative to a nearest-teacher KD chain.
They do not guarantee that the larger full-data run, the RSET/RREL branches,
or the terminal `M1E -> M2` compression will produce the same gains.

### 15.2 LOGIT, RSET, and RREL are complementary

A separate 300k representation-KD screening campaign supplied the direct
motivation for TRI60's three parallel tracks. It trained matched LOGIT, RSET,
and RREL models along the same factorized homotopy and then evaluated a fixed
equal-weight probability ensemble:

```text
p_three_way = (p_LOGIT + p_RSET + p_RREL) / 3
```

There was no member selection, validation-tuned weighting, temperature fit,
or stacking model. All three selected checkpoints were rescored in common
FP32 on the same authenticated validation rows before their probabilities
were averaged.

The recovery anchors for this earlier experiment were:

```text
M0:   AUC = 0.935643, R50 = 1217.6
TOFF: AUC = 0.945136, R50 = 1494.5
```

Its common-FP32 comparison was:

| Rung | Predictor | AUC | AUC recovery | R50 | R50 recovery | Delta AUC vs LOGIT | Delta R50 vs LOGIT |
|---|---|---:|---:|---:|---:|---:|---:|
| D40 | LOGIT KD | 0.940716 | 53.4% | 1384.0 | 60.1% | 0 | 0 |
| D40 | LOGIT+RSET+RREL | **0.942329** | **70.4%** | **1466.7** | **90.0%** | **+0.001613** | **+82.7** |
| D20 | LOGIT KD | 0.939813 | 43.9% | 1389.0 | 61.9% | 0 | 0 |
| D20 | LOGIT+RSET+RREL | **0.941410** | **60.7%** | **1439.9** | **80.3%** | **+0.001597** | **+50.9** |
| D0 | LOGIT KD | 0.938660 | 31.8% | 1326.0 | 39.1% | 0 | 0 |
| D0 | LOGIT+RSET+RREL | **0.940464** | **50.8%** | **1434.2** | **78.2%** | **+0.001804** | **+108.2** |

The exact-HLT D0 result is the most important. Adding the RSET and RREL
members to LOGIT KD almost doubled R50 recovery, from 39.1% to 78.2%, while
raising AUC recovery from 31.8% to 50.8%. In absolute terms, the ensemble
added `+0.001804` macro AUC and `+108.2` R50 over LOGIT KD alone. D40 and D20
showed the same direction, so the endpoint gain was not an isolated numerical
accident at one rung.

Crucially, RSET and RREL were not consistently stronger standalone models.
Their value appeared primarily through different errors, particularly in the
score-distribution tails that determine R50. The result therefore supports a
more precise statement than "representation KD is better": representation
objectives created useful predictive diversity, and fixed probability
averaging converted that diversity into a substantially stronger HLT-only
endpoint.

This finding explains the architecture of TRI60. TRI60 does not force one
representation strategy to replace LOGIT KD. It preserves all three as paired
tracks, compresses each track's endpoint ensemble into an M1 model, averages
those three M1 probability distributions at `M1E`, and then asks whether M2
can retain the cross-track gain in one deployable forward pass.

The evidence remains exploratory: it used one validation seed, did not yet
include a matched three-independent-LOGIT ensemble control, and evaluated a
three-model ensemble rather than a compressed single model. It nevertheless
provides strong empirical evidence for the exact type of cross-objective
complementarity that TRI60 is designed to exploit.

## 16. How to read the TRI60 results

A useful report should show every specialist as well as every ensemble. The
important questions are not only whether the final M2 wins, but where knowledge
is preserved or lost:

- Does a nearer teacher beat a distant teacher at the same target view?
- Does the ensemble beat its mean member or best member?
- Which teacher horizons contribute complementary errors?
- Does RSET or RREL help at D000 even if it does not win at an earlier rung?
- Does each M1 compression retain its endpoint ensemble?
- Does M1E beat all three M1 members?
- How much of M1E can M2 compress into one deployable model?

The immutable TRI60 scientific plan also registers the imported full-data
`M0paired` anchor for historical continuity. Because that anchor used a
different pass count, the separately registered exact-HLT CE control
`M0CE60` is the cleaner pass-matched baseline for asking how much the 60-pass
campaign itself recovers. When it is available, recovery is conveniently
reported as:

```text
AUC recovery = 100% * (AUC_model - AUC_M0CE60)
                      / (AUC_U000 - AUC_M0CE60)
```

and analogously in linear R50 space. Under this convention, `M0CE60` is 0%
and U000 is 100%. Recovery may exceed 100% or become negative; it is not
clipped. R50 recovery is informative but noisier because it depends on a tail
operating point.

Poor finite metrics never fail a job or prune a declared ensemble member.
Scientific disappointment is a result, not an operational error.

## 17. What the campaign can and cannot establish

If successful, TRI60 can show that a fixed triangular, multi-horizon KD system
retains privileged information across declared homotopy views; that LOGIT,
RSET, and RREL training produce complementary exact-HLT predictors; and that
their ensemble can be compressed into one HLT-only model.

It cannot by itself prove that:

- uniform weights are optimal;
- an ensemble owns an averaged latent representation;
- its fixed carrier is the uniquely correct carrier;
- representation KD alone caused model diversity;
- unavailable offline information was literally reconstructed at inference;
- one exploratory seed generalizes to every initialization; or
- success or failure establishes an information-theoretic limit.

The clean scientific claim is narrower and stronger: the graph is fixed in
advance, all specialists are reported, the deployment boundary is exact, and
the campaign tests whether multiple teacher horizons and multiple kinds of
knowledge can cooperate to produce a stronger single HLT model.

## 18. The complete strategy at a glance

```text
                         shared CE-only root
                               U000
                                 |
          +----------------------+----------------------+
          |                      |                      |
       LOGIT                  LOGIT+RSET             LOGIT+RREL
   1+2+3+4+5 triangle         1+2+3 triangle         1+2+3 triangle
          |                      |                      |
    LOGIT_D000E             RSET_D000E             RREL_D000E
          |                      |                      |
      M1_LOGIT                M1_RSET                M1_RREL
          +----------------------+----------------------+
                                 |
                  uniform probability ensemble M1E
                                 |
                    10% CE + 90% KD compression
                                 |
                                M2
                                 |
                   one exact-HLT deployable model
```

That is what makes TRI60 special: it combines a physically meaningful input
homotopy, permanent skip teachers, same-view probability ensembles,
matching-free representation transfer, and final ensemble compression in one
predeclared and reproducible experiment.
