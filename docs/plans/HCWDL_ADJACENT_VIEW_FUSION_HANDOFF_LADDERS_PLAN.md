# HCWDL Adjacent-View Fusion and Performance-Constrained Handoff Ladders

Implementation status (2026-09-02): Strategy A is implemented under
`src/hlt_classification/scouting/hcwdl_adjacent_output_handoff_*.py`, with thin
create/run/submit/monitor/recovery/print CLIs and source-pinned Slurm workers.
Its normative contract companion is
`docs/contracts/HCWDL_ADJACENT_OUTPUT_FUSION_HANDOFF.md`. Strategy B remains a
planned follow-up and is not executable in this artifact family.

Status: implementation-authoritative scientific design. This document freezes
the best current versions of two proposed adjacent-view transfer strategies.
It does not authorize implementation, live submission, final-test access, or
reuse of a path merely because it exists.

## 1. Purpose

The full-cardinality studies motivate a narrower question than the existing
four-spine and offline-context plans. The rich `U100` endpoint can remain very
close to `U000`, yet the eventual transition to exact HLT `D000` still loses
substantial performance. A conventional single-view ladder asks each new
student to recover its teacher using only one poorer view. These studies ask
whether an explicitly overlapping bridge can make every information handoff
safer.

Two mechanisms are kept separate:

1. **output-fusion handoff:** a rich-view teacher and its poorer-view student
   are mixed at the prediction level; the poorer view receives as much weight
   as validation-supported non-inferiority permits, and that mixture is then
   compressed into a fresh poorer-view carrier;
2. **learned two-view handoff:** a lower-view primary Particle Transformer is
   allowed to consult the adjacent richer view through removable one-way
   cross-attention; the richer residual is then withdrawn and the extracted
   lower-view model becomes the next carrier.

Both mechanisms use the same registered coordinates, populations, outer seed
domains, training budget, reporting partitions, and endpoint controls. They
may therefore be compared, but their artifacts are not interchangeable.
The broader
[offline+HLT concatenation, fusion, and withdrawal plan](HCWDL_OFFLINE_HLT_CONCATENATION_FUSION_WITHDRAWAL_IMPLEMENTATION_PLAN.md)
defines the architectural donor semantics; this document alone defines the
new adjacent-coordinate ladders.

## 2. Canonical coordinate vocabulary

The established downward degradation path is:

```text
U100 -> D080 -> D060 -> D040 -> D020 -> D000
```

Earlier discussion sometimes called the first coordinate `D100`. That name is
not part of the established graph. In all implementation, artifacts, reports,
and claims, the rich boundary coordinate is exactly `U100`. The `Dxxx` values
retain the existing `HCWDL_TRI100` coordinate constructors and meanings;
neither plan is allowed to reinterpret or recreate them.

Let the ordered coordinates be:

```text
v_0 = U100
v_1 = D080
v_2 = D060
v_3 = D040
v_4 = D020
v_5 = D000
```

`T_i` denotes the selected single-view carrier that consumes only `v_i`.
`B_i` denotes a cold-started bridge student at `v_i`. `E_i` denotes an output
mixture used only as a teacher. `F_i` denotes a learned two-view model whose
primary branch consumes `v_i` and whose removable context branch consumes
`v_(i-1)`.

## 3. Shared scientific boundaries

- The final primary artifact is one ordinary exact-HLT `D000` Particle
  Transformer. It accepts no richer coordinate, second branch, source label,
  degradation coordinate, construction index, matching index, or assignment
  metadata.
- `U100`, `D080`, `D060`, `D040`, and `D020` are training-time teacher or
  privileged bridge inputs only. Their availability never broadens deployable
  inference.
- Train and validation use the same authenticated full mapped populations,
  label order, view construction, and full-cardinality lineage as the source
  campaign. Final test remains sealed.
- Every transition is registered before execution. Poor performance, a zero
  selected mixture weight, or unsuccessful withdrawal is a completed
  scientific result and cannot remove a later registered row.
- Each new single-view carrier is cold-started from the registered matched
  initialization domain. Model or optimizer weights are not continued across
  coordinates. The within-rung withdrawal phase is the sole declared
  warm-start exception because it begins from that rung's frozen unrestricted
  fusion checkpoint.
- The primary specialist loss is the established `C25P75`, `T=2` logit KD.
  Exact T-squared scaling, FP32 loss evaluation under BF16 model autocast, and
  identical label supervision apply to both ladders.
- The fixed optimization budget is shared across comparable fresh fits:

  ```text
  passes 1-3:      linear warmup
  passes 4-45:     hold at 3e-4
  passes 46-60:    cosine decay to 1.5e-5
  passes 61-100:   constant 1.5e-5 refinement
  minimum passes:  60
  patience:        15, beginning after pass 60
  selection:       exact best registered checkpoint
  global batch:    256
  ```

  If a later completed schedule study supersedes this recipe, one exact recipe
  must be chosen and locked before either ladder is created. It must change for
  both ladders together and cannot be selected separately per rung.
- Particle views and hidden activations are RAM-only. Compact 15-class
  probability banks with exact identity digests may be durable. Rolling
  optimizer resume is forbidden; incomplete fits restart from update zero.

## 4. Validation partition and anti-selection-leakage policy

Repeatedly selecting mixture weights on the same validation events used for
headline reporting would create adaptive validation overfitting. The complete
authenticated validation role is therefore deterministically partitioned by
canonical 32-byte jet identity into three disjoint, label-stratified roles:

```text
V_checkpoint   checkpoint selection and early stopping
V_blend        output-family calibration and mixture-weight selection
V_report       untouched primary scientific reporting
```

Within each class, rows are sorted by a graph-bound hash of the canonical
identity and assigned round-robin to the three roles. The partition rule,
seed, identity hash, per-class counts, and disjointness proof are immutable
parents of both campaigns. Approximate thirds are used; no row is dropped or
duplicated. `M0CE60`, `U000`, `U100`, and every other control are reevaluated
on `V_report` so recovery and paired differences refer to identical events.

Full-validation metrics may be published as explicitly secondary descriptive
numbers after every choice is frozen. They are never used to choose a
checkpoint, fusion family, mixture weight, rung, or endpoint. Final test is
not substituted for `V_report`.

## 5. Strategy A: performance-constrained output-fusion handoff

### 5.1 Scientific question

Can each richer single-view carrier transfer its prediction function through
a temporary ensemble while progressively moving almost all teacher weight to
the adjacent poorer view, and can a fresh poorer-view model retain that
mixture without carrying an ensemble into the next rung?

### 5.2 Exact transition

For each `i` from 1 through 5, starting with authenticated carrier `T_(i-1)`:

1. Cold-start `B_i` on view `v_i`.
2. Train `B_i` with `C25P75/T=2` KD from `T_(i-1)`.
3. Evaluate `T_(i-1)` and `B_i` once on `V_blend` and construct every
   preregistered output mixture.
4. Select the mixture with the greatest admissible weight on `B_i` under the
   non-inferiority rule below.
5. Freeze the selected mixture `E_i` as a compact teacher distribution.
6. Cold-start a second model `T_i` on the same poorer view `v_i` and train it
   with `C25P75/T=2` KD from `E_i`.
7. Use the selected single-view `T_i` checkpoint as the only carrier into the
   next transition.

The complete graph is:

```text
T_U100
  -> B_D080 -> E_D080 -> T_D080
  -> B_D060 -> E_D060 -> T_D060
  -> B_D040 -> E_D040 -> T_D040
  -> B_D020 -> E_D020 -> T_D020
  -> B_D000 -> E_D000 -> T_D000
```

There are ten fresh fits after `T_U100`: one bridge and one compression fit per
transition. Output mixture construction is evaluation, not training. This
ten-fit chain is the first implementation target; Strategy B remains deferred
until the complete output-handoff result exists.

### 5.3 Registered output fusion families

Two families are evaluated at every transition without adding model fits. The
calibrated centered-logit-space family is the primary requested mechanism;
the arithmetic probability family is its mandatory simpler control.

The arithmetic probability control is:

```text
p_prob(alpha) = (1 - alpha) * p_rich + alpha * p_poor
```

where both inputs are temperature-one class probabilities and accumulation is
performed in deterministic lexical FP64 order before a single FP32
publication cast.

The paired calibrated centered-logit-space mixture is constructed from the
durable probabilities without requiring a second raw-logit artifact:

```text
ell(p) = log(max(p, 2^-126))
ell_rich_c = ell(p_rich) - mean_class(ell(p_rich))
ell_poor_c = ell(p_poor) - mean_class(ell(p_poor))

p_logit(alpha) = softmax(
    (1 - alpha) * ell_rich_c / tau_rich
    + alpha * ell_poor_c / tau_poor
)
```

`tau_rich` and `tau_poor` are fit independently on `V_blend` by a bounded,
deterministic one-dimensional NLL calibration with `0.25 <= tau <= 4`. Raw,
uncentered, uncalibrated logit interpolation is forbidden because arbitrary
logit offsets and scales would change the meaning of `alpha`.

After a temperature-one mixture is selected, its `T=2` KD target is defined
exactly as `softmax(ell(p_selected) / 2)`. The mixture is therefore a complete
teacher distribution rather than an invented average hidden representation.

Both families use the exact grid:

```text
alpha in {0/40, 1/40, ..., 39/40, 40/40}
```

The `alpha=0`, `alpha=0.5`, and `alpha=1` rows are always reported. The search
does not assume that 50/50 improves over the richer carrier and does not
assume monotonic performance in `alpha`.

### 5.4 Non-inferiority and weight selection

For every candidate, define paired macro-AUC change on `V_blend` relative to
`T_(i-1)`. Use 2,000 label-stratified paired Gaussian-multiplier bootstrap
replicates over exact paired AUC influences under one graph-bound seed. A
candidate is admissible only if:

```text
point delta macro AUC       >= -0.0001
one-sided 95% lower bound   >= -0.0003
Hbb R50 point ratio         >= 0.95
Hcc R50 point ratio         >= 0.95
Hqq R50 point ratio         >= 0.95
all probabilities and metrics are finite
```

The thresholds are fixed scientific choices, not retry tolerances. Among all
admissible probability and calibrated-logit candidates, select in order:

1. greatest `alpha`;
2. greatest macro AUC;
3. greatest linear macro R50;
4. calibrated-logit fusion before probability fusion;
5. lexical candidate identifier.

`alpha=0` reproduces the rich teacher and is therefore the defined fallback.
If it is the only admissible row, the rung records that no safe output handoff
was observed, but `E_i = T_(i-1)` and the registered poorer-view compression
fit still runs. Scientific results never control DAG completion.

### 5.5 What must be reported at every rung

The aggregate contains `T_(i-1)`, `B_i`, all mixture curves, selected `E_i`,
and compressed `T_i` on untouched `V_report`. It explicitly separates:

```text
ensemble gain       = metric(E_i) - metric(T_(i-1))
compression gap     = metric(T_i) - metric(E_i)
net handoff change  = metric(T_i) - metric(T_(i-1))
direct transition   = metric(B_i) - metric(T_(i-1))
```

Thus a strong temporary ensemble cannot conceal failed compression, and a
strong final carrier cannot be attributed solely to carrying an ensemble
forward.

### 5.6 Rung-level seed pairing

Each coordinate transition receives a unique rung seed, but the direct and
compression students inside that rung use the same initialization, sampler,
view, augmentation, and dropout seed domains:

```text
U100 -> D080:  OUTPUT_DIRECT_D080      seed S080
               OUTPUT_COMPRESSION_D080 seed S080

D080 -> D060:  OUTPUT_DIRECT_D060      seed S060
               OUTPUT_COMPRESSION_D060 seed S060

D060 -> D040:  OUTPUT_DIRECT_D040      seed S040
               OUTPUT_COMPRESSION_D040 seed S040

D040 -> D020:  OUTPUT_DIRECT_D020      seed S020
               OUTPUT_COMPRESSION_D020 seed S020
```

`S080`, `S060`, `S040`, and `S020` are distinct. Pairing within a rung makes
the selected teacher distribution the principal controlled difference;
changing the seed between rungs avoids forcing every member of the sequential
ancestry through one identical stochastic trajectory.

### 5.7 Five-seed terminal D000 study

The `D020 -> D000` transition is expanded to five predeclared paired outer
seeds. All ten terminal students share the same selected `OUTPUT_T_D020`
teacher, train/validation identities, view constructor, recipe, and update
budget. For replicate `r` in `{1,2,3,4,5}`:

```text
OUTPUT_DIRECT_D000_Sr
    D000, seed Sr, C25P75/T2 KD from OUTPUT_T_D020

OUTPUT_MIX_D000_Sr
    selected output mixture of OUTPUT_T_D020 and OUTPUT_DIRECT_D000_Sr

OUTPUT_COMPRESSION_D000_Sr
    D000, the same seed Sr, C25P75/T2 KD from OUTPUT_MIX_D000_Sr
```

The main ladder's terminal pair is exactly `S1`; it is not retrained. `S2`
through `S5` add four direct and four compression fits. `S1` through `S5` are
distinct, while direct/compression randomization is paired within each
replicate.

Five matched CE-only controls are also registered:

```text
CE_D000_Sr
    D000, seed Sr, C100P0, otherwise the identical training recipe
```

These controls are deliberately exact rather than relying only on the older
60-pass CE5 study. The historical `CE5E` and `CE5_KD` remain contextual
references, but schedule or checkpoint differences prevent them from
replacing the new matched controls.

For each family, deterministic uniform probability ensembles are evaluated
for every prefix size:

```text
CE_D000_E1 ... CE_D000_E5
OUTPUT_DIRECT_D000_E1 ... OUTPUT_DIRECT_D000_E5
OUTPUT_COMPRESSION_D000_E1 ... OUTPUT_COMPRESSION_D000_E5
```

The seed order is the immutable lexical order `S1` through `S5`; the most
favorable subset is not searched. Every member has exact weight `1/N`, FP64
lexical accumulation, and one FP32 publication cast. The prefix curves reveal
whether improvement comes from ordinary seed diversity or from the
fusion-handoff teacher.

The three five-member ensembles are each distilled into one final D000 model:

```text
CE_D000_E5                 -> FINAL_CE_SEED_D000
OUTPUT_DIRECT_D000_E5      -> FINAL_DIRECT_D000
OUTPUT_COMPRESSION_D000_E5 -> FINAL_HANDOFF_D000
```

All three final students use one common, separately registered initialization
and stochastic seed, `C10P90/T=1`, the same full-data optimization recipe, and
the same checkpoint-selection role. Their sole scientific difference is the
teacher ensemble. `FINAL_HANDOFF_D000 - FINAL_DIRECT_D000` is the primary
answer to the objection that a direct-KD seed ensemble followed by ordinary
compression is sufficient. `FINAL_HANDOFF_D000 - FINAL_CE_SEED_D000` answers
the still simpler CE-seed-ensemble objection.

This terminal replication tests the final handoff, not complete-ladder seed
uncertainty. Five fully independent ladder replicates remain a later
confirmation and cannot be claimed from this study.

## 6. Strategy B: learned adjacent-view fusion with exact withdrawal

### 6.1 Scientific question

Can a poorer-view Particle Transformer absorb useful context from the
adjacent richer view inside the representation, then retain that performance
after the richer context encoder is removed exactly?

This is the adjacent-view analogue of the removable offline-context design in
the broader concatenation/fusion/withdrawal plan. It is not symmetric fusion:
the lower-view branch is privileged as the future deployable carrier from its
first update.

### 6.2 Frozen architecture

For transition `v_(i-1) -> v_i`, `F_i` contains:

```text
v_(i-1) -> rich context adapter -> eight context blocks -> K,V context
                                                               |
                                                               | rich -> poor
                                                               v
v_i       -> canonical primary adapter -> eight primary blocks
              ^              ^              ^              ^
           after 2        after 4        after 6        after 8
                -> standard class-attention head -> logits
```

The lower-view primary branch alone owns the classifier head. One-way
cross-attention residuals enter after primary blocks 2, 4, 6, and 8:

```text
h_l = PRIMARY_BLOCK_l(h_(l-1))
r_l = CROSS_ATTN_l(Q=LN(h_l), K=LN(c_l), V=LN(c_l), masks, pair bias)
h_l = h_l + alpha * sigmoid(g_l) * W_l(r_l)
```

The registered pair bias uses only standard-four geometry computed directly
between active particles. It receives no matching identity, assignment,
coordinate, or source-row metadata. Output projections `W_l` begin at exact
zero, so the initial `alpha=1` wrapper reproduces the initialized lower-view
primary network before optimization.

When `alpha=0`, dispatch does not load the richer view, run its encoder,
construct cross-pair features, or execute cross-attention. Extraction removes
every context adapter, context block, residual gate, cross-attention module,
and pair-bias module. The result is one ordinary `v_i` Particle Transformer.

### 6.3 Exact transition protocol

For each `i` from 1 through 5:

#### Phase A: unrestricted adjacent-view acquisition

1. Cold-start the lower-view primary branch under the same matched seed as
   Strategy A's `B_i`.
2. Cold-start newly introduced context and cross-attention parameters under
   graph-bound architecture seeds.
3. Train `F_i(alpha=1)` with `C25P75/T=2` KD from single-view carrier
   `T_(i-1)` while both adjacent views are available.
4. Select its unrestricted checkpoint only on `V_checkpoint`.
5. Freeze an immutable copy as privileged teacher `Q_i`.

#### Phase B: removable-residual withdrawal

1. Initialize withdrawal student `W_i` from `Q_i`; optimizer and scheduler
   state are not inherited.
2. Evaluate the exact lower-only `alpha=0` route on every update.
3. Train the privileged, curriculum, and lower-only paths under the fixed
   schedule:

   ```text
   passes 1-10:     alpha = 1
   passes 11-60:   cosine alpha from 1 to 0
   passes 61-100:  alpha = 0
   minimum passes: 60
   patience:       15 after pass 60
   selection:      alpha=0 validation only
   ```

4. Use frozen `Q_i(alpha=1)` probabilities for KD. The lower-only path gets
   CE, teacher KD, and directed consistency from the curriculum path; the
   curriculum path receives CE and teacher KD. Exact coefficients are inherited
   from the primary withdrawal loss in the broader plan and must be versioned
   if changed.
5. Extract the selected `alpha=0` primary branch as single-view carrier `T_i`.
6. Start the next rung from a new cold initialization; do not copy `T_i`
   weights into the next fusion architecture. `T_i` supplies logits only.

The complete learned-fusion graph is:

```text
T_U100
  -> F(U100,D080) -> withdraw U100 -> T_D080
  -> F(D080,D060) -> withdraw D080 -> T_D060
  -> F(D060,D040) -> withdraw D060 -> T_D040
  -> F(D040,D020) -> withdraw D040 -> T_D020
  -> F(D020,D000) -> withdraw D020 -> T_D000
```

Every arrow ends in a real single-view carrier. A two-view network is never
silently relabeled as a lower-view model.

### 6.4 Mandatory within-rung diagnostics

Each unrestricted and withdrawal checkpoint is evaluated as:

```text
both views, alpha=1
the same lower primary with graph-bound zero context
the same lower primary with identity-permuted richer context
lower primary only, using exact alpha=0 dispatch
```

The report records the gain supplied by richer context, the gap already
present at alpha zero before withdrawal, and the recovered fraction after
withdrawal. Learned residual gates and alpha-dependent validation curves are
diagnostic only; only lower-only metrics can select the extracted carrier.

## 7. Matched controls and the D000+D000 question

At terminal `D000`, the first output campaign registers the five matched
cold CE-only controls from Section 5.7:

```text
CE_D000_Sr = D000 student, C100P0, seed Sr, matched optimization recipe
```

Nonterminal CE-only views may be reported from exact authenticated artifacts,
but the first output campaign does not add four more CE fits merely to populate
those contextual rows. Such artifacts cannot replace the terminal controls.

At every transition each strategy also includes its own cold direct-KD
control because its incoming carrier can differ after the first rung:

```text
OUTPUT_DIRECT_i  = v_i student, C25P75/T2 KD from OUTPUT_T_(i-1)
LEARNED_DIRECT_i = v_i student, C25P75/T2 KD from LEARNED_T_(i-1)
```

Strategy A's `B_i` is `OUTPUT_DIRECT_i`; it is not duplicated. Strategy B
must train or exactly authenticate `LEARNED_DIRECT_i` against its own parent.
At the first transition the two direct controls may share one artifact only
if their `U100` teacher, recipe, seed, and probability lineage are identical.

The first and final learned-fusion transitions additionally include:

```text
FUSION_LOW_LOW
    identical two-branch architecture, both branches receive the lower view

LOW_WARM_CONTINUE
    ordinary lower-view carrier continued for the same updates with CE

LOW_PARAMETER_MATCHED
    lower-view-only network with comparable trainable parameter count
```

`Fusion(D000,D000)` is a useful control but is not the primary endpoint. Three
different experiments must not be conflated:

1. averaging the same D000 checkpoint with itself is an identity test and
   must produce identical predictions;
2. averaging two independently trained D000 checkpoints measures ordinary
   seed ensembling;
3. a two-branch trainable network that consumes D000 twice measures extra
   architecture/capacity and must be labeled `FUSION_LOW_LOW`.

Both principal ladders end in a single extracted or compressed `D000` model.
The optional `FUSION_LOW_LOW` row may outperform it, but that does not make the
two-branch model evidence of successful privilege transfer.

## 8. Seed design and the reviewer-facing ensemble decomposition

The first output-fusion execution uses unique seeds across coordinate rungs,
paired seeds within each direct/compression comparison, and five paired seeds
at the terminal coordinate as frozen in Sections 5.6 and 5.7. Strategy B, when
implemented, uses the same rung-seed registry for its lower primary branch and
matched direct-KD control; architecture-only parameters use separately named
deterministic domains. Every node records the master registry, semantic seed
domain, and all derived seeds.

The terminal study separates three effects using exact equal-member controls:

```text
CE_D000_E5
    ordinary random-seed diversity

OUTPUT_DIRECT_D000_E5
    random-seed diversity after ordinary direct KD from one shared T_D020

OUTPUT_COMPRESSION_D000_E5
    the same seed count after performance-constrained handoff compression
```

The three paired final distillers then test how much of each ensemble survives
compression into one model. This is stronger than comparing an ensemble with
a single model, because member count, terminal seeds, data, endpoint view, and
final student initialization are held fixed.

If the terminal result warrants confirmation, rerun each complete ladder under
at least three predeclared outer master registries. The confirmation aggregate
compares:

```text
one ordinary D000 model
ordinary independent-seed D000 ensemble with N members
one fusion-handoff ladder endpoint under its registered rung-seed map
N independent full-ladder D000 endpoints averaged uniformly
```

`N` is identical for the two ensemble rows. The first campaign can establish
that the terminal handoff is more than ordinary terminal seed ensembling. Only
the later full-ladder replication can establish uncertainty and diversity for
the complete sequential method.

## 9. Primary reporting

All controls, bridge students, temporary teachers, fusion models, extracted
carriers, and terminal models are reported on untouched `V_report` with:

- accuracy and macro one-vs-rest AUC;
- linear macro background rejection at 50% signal efficiency;
- Hbb, Hcc, Hqq, Hgg, top, and every other available per-class rejection;
- AUC and linear-R50 recovery with `M0CE60 = 0%` and pure-offline `U000 =
  100%`;
- paired bootstrap intervals for every adjacent transition and final-endpoint
  difference;
- selected pass, complete validation history, elapsed time, peak CPU RAM,
  peak GPU memory, and durable bytes;
- Strategy A mixture curves, selected family and alpha, ensemble gain,
  compression gap, and net handoff change;
- Strategy B both/rich-only/lower-only metrics, alpha trajectory, withdrawal
  gap, extracted parity, and residual-gate diagnostics;
- `final_test_accessed: false`.

The primary comparisons are:

```text
OUTPUT_T_i  - OUTPUT_DIRECT_i
LEARNED_T_i - LEARNED_DIRECT_i
LEARNED_T_i - OUTPUT_T_i
terminal T_D000 - M0CE60
terminal T_D000 - established single-view ladder D000
```

No metric threshold controls campaign completion. Non-inferiority controls
only Strategy A's already-registered teacher mixture choice.

## 10. Artifact, storage, and execution model

The two strategies use separate versioned contract families, roots,
source-pinned worktrees, Slurm namespaces, command plans, ledgers, monitors,
and restart-zero recoveries. They may share immutable authenticated source
views, controls, and compact probability banks only through exact parent
hashes.

Durable artifacts are limited to:

- graph, recipe, source, population, seed, calibration, and selection locks;
- selected/final model checkpoints without rolling optimizer generations;
- uint8 identity digests and float32 15-class probability banks;
- mixture and withdrawal reports, aggregates, task attestations, ledgers, and
  completion markers.

Particle tensors, concatenated views, cross-attention pair features, hidden
states, attention maps, and representation targets remain process-local RAM.
No full-dataset hidden-state bank is authorized. Resource requests must be
measured using the production workers on a genuine Tigris miniature before a
full campaign is authorized.

The long-term conceptual DAG is sequential within each strategy but the two
strategies and matched controls may run concurrently:

```text
authenticate
  -> population/partition lock
  -> probability and memory audit
  -> genuine installed-Weaver preflight
  -> OUTPUT rung 1 -> ... -> OUTPUT rung 5
  -> LEARNED rung 1 -> ... -> LEARNED rung 5
  -> aggregate
  -> completion
```

No task may cancel, hold, reprioritize, mutate, or depend on a running source
campaign. Only exact campaign-bound IDs may be monitored or recovered.

### 10.1 First implementation scope and fit count

The first executable campaign contains Strategy A only. It imports one exact
authenticated `T_U100` and registers:

```text
main adjacent handoff chain                         10 fits
additional D000 direct/compression seeds S2-S5      8 fits
matched CE-only D000 seeds S1-S5                    5 fits
three five-member ensemble-compression students     3 fits
                                                    --------
total                                               26 fits
```

The four nonterminal direct/compression pairs are sequential. At D000, all
five direct fits and all five CE controls may run concurrently after
`OUTPUT_T_D020`; each compression fit waits only for its paired direct fit and
mixture selection. The three final ensemble-compression students run in
parallel after their family reducers. There is no M1, final test, learned
two-view model, representation KD, or full-ladder multi-seed replication in
this first campaign.

## 11. Required acceptance tests

Before Strategy A queue readiness, implementation must prove:

- exact reuse of established `U100`, `D080`, `D060`, `D040`, `D020`, and
  `D000` view constructors and identical canonical jet identities;
- disjoint, exhaustive, label-stratified validation partitions;
- exact probability and calibrated centered-logit mixture formulas, endpoint
  identities, alpha grid, bootstrap seed, feasibility rules, and tie breaks;
- identity-joined teacher banks and invariance to row ordering;
- all selected teacher probabilities sum to one and are finite;
- unique inter-rung seeds, paired within-rung direct/compression seeds, five
  exact terminal seed triplets, and one paired final-distiller seed;
- exact lexical nested ensembles for `N=1..5` and no favorable-subset search;
- final CE, direct-KD, and handoff ensemble distillers differ only by teacher;
- cold initialization and semantic seed pairing at every new carrier;
- same-checkpoint D000 self-averaging is prediction-identical;
- poor results and alpha-zero fallbacks do not alter DAG coverage;
- no rolling optimizer resume or durable hidden/particle artifacts;
- final-test capability is absent;
- focused local tests, neighboring regressions, installed-Weaver parity, a
  genuine Tigris miniature, measured resources, exact dry runs, and exact
  pushed source all pass before live science submission.

Before later Strategy B queue readiness, implementation must additionally
prove lower-view ownership of the learned-fusion class head, one-way
rich-to-lower information flow, fully masked padding, exact zero-residual
initialization, exact alpha-zero dispatch, absence of rich-view reads and
context allocation in the extracted route, and wrapper/extracted checkpoint
reproduction under registered FP32 and BF16 tolerances.

## 12. Implementation and decision order

The output-fusion ladder is implemented first because it uses existing
single-view training and compact probability machinery and introduces no new
deployable architecture.

1. Implement the shared validation partition, probability-bank, calibration,
   bootstrap, non-inferiority, and output-mixture contracts.
2. Implement the complete 26-fit Strategy A graph from Section 10.1. All rows
   and dependencies are registered before execution; intermediate metrics
   choose only a mixture artifact and never prune the graph.
3. Prove both fusion formulas, the `U100 -> D080` handoff, terminal paired-seed
   logic, nested ensemble reducers, and final teacher-only distiller
   differences using local synthetic tests.
4. Run installed-Weaver parity and a genuine Tigris miniature that traverses
   one direct fit, one mixture selection, one compression fit, all three
   terminal ensemble families, and one final distillation update.
5. Measure resources, materialize the complete nonmutating dry run, bind exact
   pushed source, and only then request authorization for the full output
   campaign.
6. Report the complete output result on untouched `V_report`, including every
   failed or unfavorable row.
7. Only after that review, implement the matched learned-fusion pilot and
   decide whether full learned-fusion or full-ladder multi-seed confirmation
   is justified.

This order preserves the causal story. Strategy A first asks whether
function-space mixtures enable a safe handoff and whether that survives both
per-member and final ensemble compression. Strategy B later asks whether
removable representation-level interaction enables a better handoff. In both
cases, success is measured only by the resulting single-view carrier,
especially the final ordinary `D000` model.

## 13. Strategy A implementation blueprint

### 13.1 Contract and module family

The first campaign introduces a new
`HCWDL_ADJACENT_OUTPUT_FUSION_HANDOFF_*/v1` artifact family. It must not reuse
the identity of TRI60 probability ensembles, TRI100 single-component
reducers, endpoint-mixture diagnostics, or the planned learned-fusion
campaign.

Reusable semantics belong under `src/hlt_classification/scouting/`:

```text
hcwdl_adjacent_output_handoff_contracts.py
hcwdl_adjacent_output_handoff_graph.py
hcwdl_adjacent_output_handoff_source.py
hcwdl_adjacent_output_handoff_fusion.py
hcwdl_adjacent_output_handoff_runner.py
hcwdl_adjacent_output_handoff_workflow.py
hcwdl_adjacent_output_handoff_campaign.py
hcwdl_adjacent_output_handoff_monitor.py
hcwdl_adjacent_output_handoff_recovery.py
```

Thin create, run, submit, monitor, print, and recovery CLIs belong under
`scripts/`; thin source-pinned workers belong under `sbatch/`. Existing TRI100
view construction and single-GPU training are called through explicit
authorities rather than copied. Any semantic extension to their reports or
teacher interfaces receives a versioned contract and regression tests.

### 13.2 Required artifact identities

At minimum, versioned artifacts cover:

- source, population, validation-partition, control, graph, recipe, and seed
  locks;
- temperature-calibration reports and probability/logit-family mixture curves;
- non-inferiority bootstrap reports and immutable selected-mixture manifests;
- training reports, selected/final checkpoints, probability shards,
  manifests, and locks for every fresh fit;
- nested CE, direct, and handoff ensemble reports;
- three final-distillation comparison reports;
- validation aggregate, command plan, submission journal/ledger, task
  attestations, monitor, restart-zero recovery, and completion marker.

Every reusable artifact carries a content hash, exact parent hashes, source
commit, population identity, and `final_test_accessed: false`. Path existence
alone never authorizes reuse.

### 13.3 Exact task topology

The command graph has these phases:

```text
authenticate
  -> partition_validation
  -> audit_sources_and_storage
  -> preflight
  -> in parallel, reevaluate M0CE60 and pure-offline U000 on V_report
  -> for D080,D060,D040,D020:
       train_OUTPUT_DIRECT
       publish_direct_probabilities
       select_OUTPUT_MIX
       train_OUTPUT_COMPRESSION
       publish_compression_probabilities
  -> in parallel at D000:
       train five OUTPUT_DIRECT seeds
       train five matched CE seeds
  -> select five OUTPUT_MIX_D000 teachers
  -> train five OUTPUT_COMPRESSION_D000 seeds
  -> reduce CE/direct/compression E1..E5
  -> in parallel:
       train FINAL_CE_SEED_D000
       train FINAL_DIRECT_D000
       train FINAL_HANDOFF_D000
  -> aggregate
  -> campaign_complete
```

Probability publication is a compact reducer, not a fresh fit. A selected
mixture manifest records the full candidate curve even when `alpha=0` wins.
Downstream training consumes only exact identity-joined selected teacher
probabilities.

### 13.4 Submission and recovery

Campaign creation requires exact source authentication, full dry run, measured
resources, installed-Weaver parity, a genuine production miniature, explicit
live authorization, and a clean detached worktree at the pushed commit.
Science jobs use their own root and job prefix and have no scheduler
dependency on a still-running source campaign.

Monitoring queries only ledger-bound job IDs. Recovery is created only after
all subject jobs are terminal, inherits only content-valid completed tasks,
and retries the failed downstream closure from update zero under a new pinned
source. It cannot change views, teachers, fusion grids, thresholds, seeds,
losses, schedules, populations, or final-test capability.

### 13.5 Source prerequisites and creation interface

Campaign creation accepts explicit paths rather than searching for the
highest-scoring or newest directory:

```text
--source-campaign-spec
--u100-training-report
--u100-selected-checkpoint
--m0ce60-training-report
--pure-offline-u000-training-report
--campaign-root
--project-dir
--source-commit
```

The imported `T_U100` must come from the intended completed full-cardinality
lineage and authenticate its exact `U100` coordinate, train/validation
population, selected checkpoint, training recipe, seed, and final-test
boundary. Overall source-campaign completion is not required when those exact
parents are independently complete and reusable. A different `U100` cannot be
chosen after looking at this campaign's results.

If the source does not already expose compatible compact train and validation
probability locks, the new DAG registers source-pinned inference reducers
before the first handoff. Those reducers write only canonical identity digests
and 15-class probabilities; they do not retrain `U100`, alter its report, or
write particle views. Imported `M0CE60`, pure-offline `U000`, and historical
CE5 artifacts are reporting controls and never training teachers unless a
separate graph edge explicitly says so.
