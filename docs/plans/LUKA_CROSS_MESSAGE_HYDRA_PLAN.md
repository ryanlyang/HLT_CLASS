# U100 cross-attention messages on Hydra — v1

Authority: the user approved two privileged teachers and twelve fresh students,
then requested implementation through queue readiness on Hydra. This is an
additive campaign. Existing joint/frozen adapters, controls, jobs and data stay
unchanged. Hydra uses `gpus/default/normal`, one L40 per fit.

## Scientific registration

The primary population is the existing ALLJETS TRAIN299234/validation74895.
Original50k overlaps validation and is report-only. Historical M0HLT/OFFLINE
define recovery; the saved U100-w5 reference and its Hydra replay are reused.
There is no independent test and no new native reference fit.

Two teachers, `teacher_m32` and `teacher_m16`, train a fresh HLT ParT and a
privileged message branch jointly. The original selected COARSE_U100 encoder
is frozen, in evaluation mode; its eighth particle-block features (before class
attention) supply keys and values. HLT features after particle block4 supply
queries. A four-head, width128 cross-attention branch reads all valid U100 slots,
with separate LayerNorms and learned Q/K/V projections. Its output is projected
to a 32- or16-dimensional message, then mapped to128 and added to the HLT token
residual before block5. No positional IDs or matching/degradation metadata enter
either branch. Existing native physics pair attention stays unchanged.

Only the message bottleneck differs between teachers. The expansion projection
starts at zero; earlier message parameters are normally initialized. The base
initially matches native ParT without a dead all-zero branch. Teachers use
`0.25 CE + 0.75 * T^2 KL(q_U100(T) || p_teacher(T))`, T=2, the original batch128
30-epoch AdamW schedule. U100 logits are frozen TRAIN targets; validation labels
and logits never train the teacher. Teachers select by full validation macro AUC.

Each student has CE1 plus the unchanged U100 jet-embedding SmoothL1 weight5.
An HLT message predictor reads contextual block4 tokens, own-view17 features
and a masked pooled HLT jet context. LayerNorm/128-wide GELU MLP predicts the
compact message. A separately initialized learnable expansion aligns it with
the student's native128 space. No privileged teacher projection/weights are
copied to any student or architecture control. The message predictor always
uses HLT inputs, including during training; no teacher forcing occurs.

Supervision is SmoothL1 in the selected teacher's compact message space, divided
by per-channel TRAIN RMS (first4096 authenticated TRAIN rows, floor0.05), averaged
over channels, valid particles within each jet, then jets. Its coefficient ramps
linearly by update over the first3 epochs. Compact messages are not a claimed
physical quantity or an identifiable universal coordinate system. Injection CE
gradients train both the prediction and student expansion; auxiliary-only loss
also reaches the first four native blocks. Native classification/embedding losses
reach the entire encoder. All supervised targets are detached.

| Task | Width | Mode | Message weight | Seed index |
|---|---:|---|---:|---:|
| m32_both_l0p3_s0 |32|prediction + injection|0.3|0|
| m32_ce_s0 |32|HLT architecture control|0|0|
| m32_aux_l0p3_s0 |32|prediction, no injection|0.3|0|
| m32_both_l0p1_s0 |32|prediction + injection|0.1|0|
| m32_aux_l0p1_s0 |32|prediction, no injection|0.1|0|
| m32_both_l1_s0 |32|prediction + injection|1|0|
| m32_aux_l1_s0 |32|prediction, no injection|1|0|
| m16_both_l0p3_s0 |16|prediction + injection|0.3|0|
| m16_ce_s0 |16|HLT architecture control|0|0|
| m16_both_l1_s0 |16|prediction + injection|1|0|
| m32_both_l0p3_s1 |32|prediction + injection|0.3|1|
| m32_ce_s1 |32|HLT architecture control|0|1|

Seed0 preserves the matched U100-w5 native initialization and sampler seeds.
Seed1 adds1000003 to both. Predictor/expansion initialization uses its own forked
RNG, with the same seed for a matched architecture pair. Predictors/expansions
are never loaded from a teacher. Frozen U100 message-source weights are used
only on oracle paths. Controls read the original embedding bank, not oracle
messages, U100 particles or the frozen encoder.

At deployment, remove embedding heads in all students and message heads in
auxiliary-only students. Injected/control students retain HLT message prediction
and expansion. Teacher checkpoints are explicitly privileged and nondeployable.
Selected teachers additionally report message-disabled predictions using the
same checkpoint, no reselection. This is a reliance diagnostic under distribution
shift, not a trained no-message control. Poor finite scores never block other fits.

## Data, compute and execution

Reuse the verified Brown bundle, frozen-adapter validation-U100 delta, environment,
and precision diagnostic. Transfer only the original COARSE_U100 checkpoint and
its saved validation predictions in a new authenticated small delta. Bind both
to the original imported teacher record and original-source replay. Do not
retrain U100, substitute another checkpoint, or require a SPORC GPU for export.

Frozen U100 tokens/logits are built once per allocation into bounded process-local
ragged FP32 RAM, with ordered identities and exact slot alignment. Teachers use
TRAIN and validation; supervised students only TRAIN; controls neither. No large
persistent token bank or per-pair target bank is written. The frozen encoder is
released before fitting. Frozen selected message teachers produce TRAIN targets
online, using the same cached U100 tokens, so the target depends on the teacher's
own HLT context, not the student's changing coordinate system.

Shared setup authenticates files, measures frozen-feature preparation, executes
all14 production miniature fits with real worst-multiplicity full/partial batches,
and reloads selected checkpoints. It replays the U100 oracle with the original
AUC portability threshold5e-5; probability drift remains diagnostic. It reuses
the accepted Hydra precision evidence, with no separate preflight job. Actual
RAM/GPU/time limits, environment, data and source lineage are technical checks;
scientific performance is never a submission threshold.

Submission: shared setup, two teachers, twelve students and report. Controls
have no teacher dependency; each supervised student waits only for its selected
width-specific teacher. No lanes or concurrency cap. Ordering expresses priority,
not a promised Slurm start order. Exact pushed source, complete dry review and
measured setup acceptance precede science. Preserve claims on failure; an
idempotent `finish` supports a stopped login watcher without rerunning setup.

Output budget is8GiB plus real filesystem headroom, not all available user quota.
Read-only results include historical references, matched architecture controls,
all completed fits, pending rows and per-epoch progress. Second-seed differences
use their matched architecture control; historical seed0 U100 remains context.

Implementation: `src/hlt_classification/luka_cross_message_hydra/`.
