# PMARD Pilot: Preliminary Validation Results and Implementation Map

Status: **preliminary, validation-only, single screening seed**  
Snapshot date: **2026-08-07**  
Campaign root on Tigris:
`/home/ryreu/atlas/HLT_Classification/checkpoints/pmard_pilot_c3e40850_prefix_recovery_r5`

This document is a self-contained technical brief for reviewing the first
Particle-Matched Alpha-Repair Distillation (PMARD) pilot. It records the
results transcribed from the immutable Tigris campaign reports, explains what
was actually trained, points to the implementation of every major component,
and separates observations from claims that still require confirmation.

The authoritative scientific specification remains the
[active PMARD campaign plan](plans/SCOUTING_ALPHA_REPAIR_DISTILLATION_OPTIMAL_CAMPAIGN.md).
The fitted matcher is defined by the
[fitted-strict matcher contract](contracts/FITTED_STRICT_MATCHER.md). This
brief does not change either contract.

## 1. Executive summary

The pilot gives consistent but small evidence that privileged offline-directed
repair helps an HLT-only ParT student:

- teacher quality improves as alpha increases, and alpha `1.0` wins the
  predeclared alpha selector;
- the primary K2 student beats the ordinary K1 self-KD control in both the
  alpha sweep and the separately trained K0--K6 control sweep;
- the privilege-specific gain is small: K2 versus K1 improves CE by about
  `0.00082`, macro AUC by `0.000134`, and the exponentiated macro mean log
  rejection by about `1.2%`;
- the selective matched endpoint T100 recovers only about 21% of the
  native-offline CE gap, 16% of the macro-AUC gap, and 18% of the macro
  mean-log-rejection gap;
- direct KD from the substantially stronger native-offline oracle K6 improves
  CE and AUC slightly beyond K2, but produces essentially the same rejection;
- the completed P4-only mechanism row is close to the alpha-zero null, while
  full 21-field selective repair is slightly better. This is preliminary
  evidence that non-kinematic fields carry the useful signal.

The result is directionally encouraging, but it is not yet a confirmed
scientific result. The final test is sealed, multi-seed confirmation has not
run, representation KD and generational KD have not yet reported, and the
observed differences are small enough that selection and run-to-run variation
must be quantified.

## 2. Dataset, split, and evaluation population

The source is the paired CMS ScoutingAK8 2024 dataset. One ROOT row already
contains an HLT/scouting jet and its corresponding offline jet; PMARD infers
only constituent-level associations within that paired row. It does not run a
second jet-level nearest-neighbor match.

The pilot selects:

| Role | Selected jets | Use in this snapshot |
|---|---:|---|
| Train | 300,000 | optimization and teacher/student fitting |
| Validation | 100,000 | checkpoint, alpha, and model selection |
| Final test | 100,000 | sealed until finalist and execution locks |

Rows are selected class-proportionally from the file-disjoint 60/20/20 split
without copying the ROOT dataset. The split implementation is in
[`splits.py`](../src/hlt_classification/scouting/splits.py), and the stable
class-proportional bounded selection is implemented by `build_row_selection`
in
[`selective_assignment.py`](../src/hlt_classification/scouting/selective_assignment.py).

The validation distribution is natural and highly imbalanced: QCD is roughly
72% of the mapped population. Therefore the reported `accuracy` is ordinary
overall accuracy, not balanced accuracy. CE, macro one-versus-rest AUC, and
macro mean log QCD rejection are more informative for this campaign.

The 15-class definitions, 21 HLT channels, native-offline branches, and
collection limits are frozen in
[`schema.py`](../src/hlt_classification/scouting/schema.py). Fixed-shape HLT
and native-offline ParT tensors are built in
[`inputs.py`](../src/hlt_classification/scouting/inputs.py).

## 3. Model definitions

### 3.1 Deployable HLT model

T0 and every deployable student use the same 21-input, 15-output Weaver
Particle Transformer adapter:

```text
input_dim              21
pair_input_dim          4
embed_dims              [128, 512, 128]
pair_embed_dims         [64, 64, 64]
attention heads         8
particle blocks         8
class-attention blocks  2
activation              GELU
maximum HLT tokens      200
```

The implementation and its HLT-only forward signature are in
[`scouting_particle_transformer.py`](../src/hlt_classification/models/scouting_particle_transformer.py),
especially `ScoutingParticleTransformer`. A deployable graph accepts only
`features`, `vectors`, and `mask`; it cannot accept offline arrays, match
indices, or confidence.

### 3.2 Native-offline oracle

TOFF is a nondeployable diagnostic using separate charged and neutral ParT
encoders followed by a classifier. Its implementation is
`NativeOfflineParticleTransformer` in
[`scouting_particle_transformer.py`](../src/hlt_classification/models/scouting_particle_transformer.py).
TOFF is not the deployable architecture and should not be described as a
strictly capacity-matched teacher. It is an upper-reference diagnostic for the
information present in the native offline collections.

### 3.3 Training budget

The pilot budget grid selected:

```text
experiment       budget_b256_lr0.0003_p10
effective batch  256
peak LR          3e-4
updates          11,720
nominal passes   10
temperature      1.0
screening seed   1337
```

The budget and temperature grids are registered in `pmard_tasks` and executed
by `Workflow.run` in
[`campaign.py`](../src/hlt_classification/scouting/campaign.py) and
[`workflow.py`](../src/hlt_classification/scouting/workflow.py). Training,
validation checkpoint selection, BF16 model execution, FP32 CE/KL math, and
exact resume are implemented by `train_pmard` in
[`engine.py`](../src/hlt_classification/scouting/engine.py).

Train rows use square-root inverse-frequency class weights with unit natural-
population mean. The formula is `sqrt_inverse_class_weights` in
[`training.py`](../src/hlt_classification/scouting/training.py). Reported
validation metrics are unweighted.

## 4. What the selective offline-directed endpoint means

The production matcher is the frozen `fitted_strict` algorithm implemented by
`ConstituentMatcher` in
[`fitted_strict.py`](../src/hlt_classification/scouting/fitted_strict.py). It
uses empirical likelihood-ratio tables, strict PID and rounded-charge gates,
a rectangular Hungarian assignment with private dummies, post-assignment
diagnostics, calibrated confidence, and the fixed acceptance threshold
`0.9828147479721088`.

The matcher deliberately abstains. On its canonical untouched reference
sample, it accepts 64.86% of HLT tokens, representing 67.21% of aggregate HLT
transverse momentum. The exact pilot assignment coverage should be read from
its assignment manifest; this brief does not substitute the reference
coverage for a missing pilot-specific count.

Assignments are computed once per selected jet and stored as compact sparse
per-source shards. Construction, validation, and bounded reload are in
[`selective_assignment.py`](../src/hlt_classification/scouting/selective_assignment.py),
with the thin builder in
[`build_pmard_assignment_cache.py`](../scripts/build_pmard_assignment_cache.py).
The student never consumes assignments as features.

The primary repair is `SELECTIVE_FULL_PARTICLE_ENDPOINT/v1`, implemented by
`build_alpha_repaired_inputs` and
`build_selective_matched_offline_endpoint_inputs` in
[`repair.py`](../src/hlt_classification/scouting/repair.py). Its semantics are:

- HLT count, token order, padding, and mask never change;
- accepted matched tokens move toward their assigned offline particle;
- continuous raw quantities interpolate with alpha;
- discrete identity, validity, quality, and hit groups use deterministic
  nested endpoint switches rather than fractional categories;
- at alpha `1`, every accepted token receives its complete projected offline
  p4 and 21-field endpoint;
- every unmatched token remains byte-identical to HLT at every alpha;
- extra offline constituents are never inserted.

Consequently, T100 is a selective HLT/offline hybrid on the HLT skeleton. It
is not the native offline jet and does not imply that every HLT constituent
has a true inferred counterpart.

Streaming assembly and assignment joining are in
[`pmard_stream.py`](../src/hlt_classification/scouting/pmard_stream.py). The
reported recovered pilot is an immutable streamed execution. The newer
process-local view cache in
[`view_cache.py`](../src/hlt_classification/scouting/view_cache.py) and its
[v1 contract](contracts/PMARD_EPHEMERAL_VIEW_CACHE.md) optimize later
campaign-spec-v10 executions and did not reinterpret this pilot.

## 5. Teacher and oracle results

The teacher ladder uses the same HLT-style ParT architecture, initialization,
sampler, optimizer, and update budget. Only alpha changes.

| Teacher | Alpha/input | CE | Accuracy | Macro AUC | Mean logR50 | exp(mean logR50) |
|---|---:|---:|---:|---:|---:|---:|
| T0 | HLT | 0.680137 | 0.787210 | 0.936530 | 6.997916 | 1094.4 |
| T05 | 0.05 | 0.679418 | 0.787000 | 0.936653 | 6.991735 | 1087.6 |
| T10 | 0.10 | 0.679039 | 0.787390 | 0.936696 | 7.012073 | 1110.0 |
| T25 | 0.25 | 0.677499 | 0.786850 | 0.937207 | 7.022193 | 1121.2 |
| T50 | 0.50 | 0.675302 | 0.788250 | 0.937461 | 7.027202 | 1126.9 |
| T100 | 1.00 | 0.672082 | 0.789230 | 0.937811 | 7.029308 | 1129.2 |
| TOFF | native offline | 0.642329 | 0.797980 | 0.944426 | 7.179346 | 1312.0 |

CE and AUC improve almost monotonically with alpha. Mean log rejection is
noisier at small alpha but is strongest at alpha 1 among the selective
teachers. The large remaining T100-to-TOFF difference shows that the selective
endpoint retains only a minority of the native-offline advantage.

The separately executed authenticated oracle validation reproduces the
training-report values to rounding:

| Oracle | CE | Accuracy | Macro AUC | Mean logR50 | exp(mean logR50) |
|---|---:|---:|---:|---:|---:|
| HLT baseline | 0.680136 | 0.787240 | 0.936530 | 6.997916 | 1094.4 |
| Matched endpoint | 0.672080 | 0.789240 | 0.937812 | 7.029709 | 1129.7 |
| Native offline | 0.642331 | 0.797990 | 0.944426 | 7.179346 | 1312.0 |

Using these authenticated values, T100 closes approximately 21.3% of the
T0-to-TOFF CE gap, 16.2% of the macro-AUC gap, 18.8% of the overall-accuracy
gap, and 17.5% of the macro-mean-log-rejection gap.

Teacher construction is dispatched by the `teachers` and `oracle_validation`
branches of
[`workflow.py`](../src/hlt_classification/scouting/workflow.py). The thin
trainer and evaluator are
[`train_scouting_pmard_teacher.py`](../scripts/train_scouting_pmard_teacher.py)
and
[`evaluate_scouting_pmard.py`](../scripts/evaluate_scouting_pmard.py).

## 6. Alpha-sweep student results

Every alpha student uses K2:

```text
0.25 * hard-label CE
+ 0.60 * KD from T0
+ 0.15 * KD from the corresponding alpha teacher
```

At alpha zero, both frozen teachers resolve to the HLT endpoint, making the row
an exact no-offline-information control. The six results are:

| Student | CE | Accuracy | Macro AUC | Mean logR50 | exp(mean logR50) |
|---|---:|---:|---:|---:|---:|
| K2 alpha 0 | 0.676319 | 0.787690 | 0.937296 | 7.023200 | 1122.4 |
| K2 alpha 0.05 | 0.675792 | 0.787950 | 0.937371 | 7.023852 | 1123.1 |
| K2 alpha 0.10 | 0.675665 | 0.787800 | 0.937379 | 7.018688 | 1117.3 |
| K2 alpha 0.25 | 0.675651 | 0.787960 | 0.937407 | 7.021657 | 1120.6 |
| K2 alpha 0.50 | 0.675263 | 0.788190 | 0.937429 | 7.023913 | 1123.2 |
| K2 alpha 1.00 | 0.675242 | 0.788130 | 0.937435 | 7.033900 | 1134.4 |

Alpha 1 improves over alpha 0 by `0.001077` CE, `0.000139` macro AUC, and
`0.010700` mean log rejection. Exponentiating the log-rejection difference
corresponds to approximately 1.08% higher geometric-mean rejection.

The selector prioritizes higher macro mean log rejection, then higher macro
AUC, lower CE, lower calibration error, and stable ID tie-breaking. It selected
alpha `1.0`. The exact selector is `utility_key`/`select_alpha` in
[`selection.py`](../src/hlt_classification/scouting/selection.py), and the
sweep dispatch is the `k2_alpha_sweep` branch in
[`workflow.py`](../src/hlt_classification/scouting/workflow.py).

## 7. Registered logit-KD controls

The fixed loss mixtures live in `KD_MIXTURES` and are evaluated by
`pmard_loss` in
[`training.py`](../src/hlt_classification/scouting/training.py). KD is forward
KL on temperature-scaled class probabilities; CE, KL, and reductions execute
in FP32.

| Arm | CE weight | T0 KD | Privileged KD | Privileged source |
|---|---:|---:|---:|---|
| K0 | 1.00 | 0.00 | 0.00 | none |
| K1 | 0.25 | 0.75 | 0.00 | none |
| K2 | 0.25 | 0.60 | 0.15 | T100 |
| K3 | 0.25 | 0.50 | 0.25 | T100 |
| K4 | 0.25 | 0.00 | 0.75 | T100 |
| K5 | 0.10 | 0.75 | 0.15 | T100 |
| K6 | 0.25 | 0.60 | 0.15 | native TOFF |

| Arm | CE | Accuracy | Macro AUC | Mean logR50 | exp(mean logR50) |
|---|---:|---:|---:|---:|---:|
| K0 | 0.676165 | 0.787550 | 0.937008 | 7.016325 | 1114.7 |
| K1 | 0.675880 | 0.787740 | 0.937334 | 7.022452 | 1121.5 |
| K2 | 0.675060 | 0.787720 | 0.937468 | 7.034357 | 1135.0 |
| K3 | 0.674463 | 0.788050 | 0.937547 | 7.029742 | 1129.7 |
| K4 | 0.675812 | 0.787790 | 0.937214 | 7.034689 | 1135.3 |
| K5 | 0.676623 | 0.787760 | 0.937032 | 7.015861 | 1114.2 |
| K6 | 0.674396 | 0.787940 | 0.937601 | 7.034321 | 1134.9 |

The primary privilege-specific comparison is K2 versus K1:

| Quantity | K2 minus K1 | Direction |
|---|---:|---|
| CE | -0.000821 | favorable |
| Accuracy | -0.000020 | effectively unchanged |
| Macro AUC | +0.000134 | favorable |
| Mean logR50 | +0.011905 | favorable |
| exp(mean logR50) | about +1.20% | favorable |

This agrees closely with the independent alpha-1 versus alpha-0 sweep
contrast. That agreement is the strongest evidence that the direction is not
an isolated table fluctuation, although both are still screening-seed results.

Relative to the T100 teacher advantage over T0, the K2-minus-K1 student gain
is approximately 10.2% for CE, 10.5% for macro AUC, and 37.4% for mean log
rejection. These are descriptive ratios across separate trained runs, not
confidence intervals or formal causal estimators.

K3 shows that increasing T100 weight from 15% to 25% improves CE and AUC but
trades away some rejection. K4 shows that removing the T0 anchor does not
dominate. K5 suggests that reducing CE to 10% is harmful in this setup.

K6 is especially diagnostic. Replacing T100 with the substantially stronger
native-offline TOFF teacher improves CE by another `0.000664` and macro AUC by
another `0.000133` relative to K2, but changes mean log rejection by only
`-0.000036`. Better privileged information therefore still helps probability
quality, but stronger teacher logits alone do not create a large rejection
gain at the current 15% coefficient.

Student CLI wiring, teacher loading, cached-logit joining, and HLT-only student
training are in
[`train_scouting_pmard_student.py`](../scripts/train_scouting_pmard_student.py).
The K0--K6 campaign dispatch is in the `kd_controls` branch of
[`workflow.py`](../src/hlt_classification/scouting/workflow.py).

## 8. Preliminary mechanism controls

At the captured snapshot, mechanism rows M00 and M12 had completed:

| Row | Actual meaning | CE | Accuracy | Macro AUC | Mean logR50 | exp(mean logR50) |
|---|---|---:|---:|---:|---:|---:|
| M00 | P4-only repair at selected alpha | 0.675837 | 0.788030 | 0.937353 | 7.019930 | 1118.7 |
| M12 | alpha-zero identity/null | 0.675835 | 0.787780 | 0.937331 | 7.022366 | 1121.4 |
| selected K2 | full selective 21-field repair | 0.675242 | 0.788130 | 0.937435 | 7.033900 | 1134.4 |

M00 and M12 are extremely close. Full repair improves over M00 by `0.000595`
CE, `0.000082` macro AUC, and `0.013970` mean log rejection, corresponding to
about 1.41% higher geometric-mean rejection. This is consistent with useful
information residing in track, PID, impact-parameter, quality, or other
non-kinematic fields rather than in p4 repair alone. It is not yet a confirmed
feature attribution because only one screening seed and two mechanism rows
are represented.

The actual 22-row mechanism registry is in the `mechanism_controls` branch of
[`workflow.py`](../src/hlt_classification/scouting/workflow.py). After adoption
of the persistent fitted-strict assignment cache, the workflow deliberately
overrides matcher variant, threshold, and category selection with the one
authorized canonical assignment. As a result, M00--M07 and M13--M14 are
effectively repeated P4-only rows rather than distinct matcher/threshold/
category studies. The genuinely distinct remaining rows test shuffled
partners, direction-only, response-only, wrong/random direction, explicit
match corruption, log-angular interpolation, and confidence-weighted repair.
This redundancy is an execution-efficiency limitation and should be accounted
for when interpreting the completed array.

Repair-family behavior is implemented in
[`repair.py`](../src/hlt_classification/scouting/repair.py), including
`P4_ONLY`, `MATCH_SHUFFLED`, `DIRECTION_ONLY`, `RESPONSE_ONLY`,
`WRONG_DIRECTION`, `RANDOM_DIRECTION`, `LOG_ANGULAR`, and
`CONFIDENCE_WEIGHTED`.

## 9. Representation and generational stages still pending

The representation array is registered only after all mechanism jobs finish.
The six positive arms add coefficient `0.10` to the K2 objective:

| Arm | Distilled surface |
|---|---|
| R1 | final class-token representation |
| R2 | masked mean of final particle states |
| R3 | final per-particle states |
| R4_PAIR | learned pair-geometry representation |
| R4_GRAM | normalized particle-representation Gram matrix |
| R5 | two late particle-block depths |

Each positive arm has a matched-capacity zero-coefficient control. There is no
registered initial-embedding KD arm in this pilot. The representation taps and
projections are implemented by `RepresentationScoutingParticleTransformer`
in
[`scouting_particle_transformer.py`](../src/hlt_classification/models/scouting_particle_transformer.py),
and the recursive FP32 loss is `representation_kd_loss` in
[`training.py`](../src/hlt_classification/scouting/training.py). Execution is
the `representation` branch of
[`workflow.py`](../src/hlt_classification/scouting/workflow.py).

Generation 1 uses the selected alpha student as the new HLT anchor. It
initializes and fine-tunes a selected-alpha companion teacher for one quarter
of the original updates at one tenth of the LR, then trains a new K2 student
from the previous HLT model and the companion. Generation 2 repeats this once.
Each generation also trains a same-generation self-KD control. These paths are
implemented in the `generation_1`/`generation_2` branch of
[`workflow.py`](../src/hlt_classification/scouting/workflow.py).

At the captured scheduler snapshot:

```text
44082  teachers             completed
44083  oracle_validation    completed
44084  K2 alpha sweep       completed
44085  alpha selection      completed
44086  K0--K6 controls      completed
44087  mechanism controls   in progress
44088  representation KD    pending on 44087
44089  generation 1         pending
44090  generation 2         pending
```

Poor performance in any registered arm does not cancel later scientific rows;
the dependency only enforces artifact availability and lineage.

## 10. Metric implementation and interpretation

All metrics are implemented in
[`evaluation.py`](../src/hlt_classification/scouting/evaluation.py), especially
`classification_metrics`:

- CE is ordinary unweighted validation cross-entropy;
- accuracy is natural-population overall accuracy;
- macro AUC is the mean of 15 one-versus-rest AUCs;
- each signal-versus-QCD discriminant is
  `p(signal) / (p(signal) + p(QCD))`;
- per-class rejection at 50% signal efficiency is
  `N_QCD / max(1, N_QCD passing)`;
- `macro_mean_log_qcd_rejection_at_50pct_signal` averages the logarithm of
  rejection across the 14 signal classes.

The displayed `R50` in this brief is
`exp(macro_mean_log_qcd_rejection_at_50pct_signal)`. It is therefore the
geometric mean of the 14 per-signal rejections, not the rejection of a single
binary tagger and not the arithmetic mean rejection.

Authenticated label-free inference and report publication are implemented in
[`inference.py`](../src/hlt_classification/scouting/inference.py). The pilot
results above came from immutable `training_report.json` and
`evaluation_report.json` artifacts, not from parsing transient training logs.

## 11. Artifact map for independent inspection

With `ROOT` equal to the campaign root printed at the top of this document:

| Artifact | Location |
|---|---|
| Immutable campaign specification | `${ROOT}/campaign_spec.json` |
| Training lock | `${ROOT}/locks/05_training.json` |
| T0 reference | `${ROOT}/training/teachers/T0_reference.json` |
| Alpha teachers | `${ROOT}/training/teachers/T05` through `T100/training_report.json` |
| Native-offline oracle | `${ROOT}/training/teachers/TOFF/training_report.json` |
| Oracle validation | `${ROOT}/evaluation/oracles/validation/*/evaluation_report.json` |
| Alpha students | `${ROOT}/training/k2_alpha/{0,0.05,0.1,0.25,0.5,1}/training_report.json` |
| Alpha selection | `${ROOT}/training/alpha_selection.json` |
| KD controls | `${ROOT}/training/kd_controls/K0` through `K6/training_report.json` |
| Mechanism rows | `${ROOT}/training/mechanisms/00` through `21/` |
| Representation rows | `${ROOT}/training/representation/` once released |
| Generation rows | `${ROOT}/training/generations/` once released |
| Slurm submission lineage | `${ROOT}/submission_ledger.json` |

Reusable artifacts are content-hashed and validated before reuse. The lock
chain and selective-assignment authorization are implemented in
[`locks.py`](../src/hlt_classification/scouting/locks.py).

## 12. Current interpretation

The pilot supports three narrow observations:

1. Increasing the amount of accepted all-field offline repair improves the
   same-capacity selective teacher, with alpha 1 preferred over weaker alpha.
2. Adding the selected teacher to self-KD gives a small, directionally
   consistent HLT-only student improvement.
3. P4-only repair does not reproduce the full-repair result in the first
   completed mechanism comparison.

It also exposes two ceilings:

1. **Endpoint ceiling:** T100 retains only about 16--21% of the native-offline
   advantage, consistent with a view where roughly one third of tokens remain
   HLT and extra offline particles are absent.
2. **Transfer ceiling:** K6 shows that even direct native-offline logit KD at
   15% weight transfers only a modest additional CE/AUC improvement and no
   clear additional rejection.

Matching coverage is therefore not the only plausible limitation. Improving
high-purity coverage or constructing an explicitly named set-transport
completion could strengthen the teacher, but better teacher optimization,
loss weighting, representation transfer, and generational assistants may also
be required to transfer that strength into the HLT-only model.

## 13. Questions for an external reviewer

1. Are the agreement between the alpha-sweep and K1/K2 contrasts, and their
   size relative to run-to-run controls, persuasive enough to continue?
2. Does the weak T100-to-TOFF gap recovery point primarily to selective match
   coverage, HLT-cardinality truncation, mixed-domain inconsistency, or teacher
   optimization?
3. Does K6's better CE/AUC but unchanged rejection imply that logit KD is the
   dominant bottleneck for the primary metric?
4. Should the next endpoint diagnostic use the highest-pT `N_HLT` native
   offline particles without inferred correspondence to isolate cardinality
   from matching?
5. Would a high-confidence anchored transport completion for unmatched HLT
   slots create a scientifically cleaner alpha-1 endpoint than leaving them
   HLT, provided it is described as transport rather than physical matching?
6. Should teacher training begin from T0, use an alpha curriculum, train
   longer, or receive TOFF logit supervision before investing in a higher-
   coverage matcher?
7. Which representation surface is most plausible when teacher/student token
   identity is only valid for the accepted subset?
8. Are the current validation differences likely resolvable with the planned
   five confirmation seeds, or should the screening/confirmation design be
   revised before final-test access?

## 14. Claims that are not yet authorized

This snapshot does not establish that:

- inferred constituent matches are physical truth;
- repaired particles are unsmeared truth particles;
- K2 significantly outperforms K1;
- the gain generalizes across seeds or datasets;
- any representation or generational method helps;
- T100 approximates the complete native offline jet;
- final-test performance improved.

The final test remains sealed. Statistical confirmation, paired inference,
and final-test evaluation must follow the registered lock sequence in the
[PMARD runbook](PMARD_RUNBOOK.md).
