# Complementary OFFLINE teachers on Hydra

Active scientific plan, execution v2 updated 2026-11-10. Implement three final OFFLINE teachers and
four native HLT ParT students. Three additional out-of-fold (OOF) preparation
fits make the error-focused targets honest with respect to TRAIN membership.
Ten training jobs total; setup, durable CPU release and CPU report are separate.
This campaign is independent of the summary-token experiment.

## Fixed comparison

| Task | Training objective / role |
|---|---|
| T0 | Fresh OFFLINE CE teacher |
| F0, F1, F2 | Fresh CE reference fits on two of three TRAIN folds |
| TE | Fresh OFFLINE CE plus explanation-diversity penalty relative to T0 |
| TF | Fresh OFFLINE CE with bounded OOF difficulty weights |
| S0 | Fresh HLT CE + 5 embedding(T0) |
| SE | Fresh HLT CE + 2.5 embedding(T0) + 2.5 embedding(TE) |
| SF | Fresh HLT CE + 2.5 embedding(T0) + 2.5 embedding(TF) |
| SEF | Fresh HLT CE + (5/3) each embedding(T0, TE, TF) |

No U100 or IT embedding objective is present. Teachers have the same native
8-block, 128-wide ParT architecture. Students share native initialization and
sampler seeds, architecture, batch128 and original full30 AdamW/LR schedule.
T0/TE/TF and fold references share the original OFFLINE initialization/sampler
seeds +200000. Each target has a separately seeded 128-GELU-128 predictor,
initialized without advancing the native RNG. Export each selected teacher's
clean native pooled128 preclassifier representation. Independently normalize
TRAIN banks with FP64 population moments, std floor .001; mean SmoothL1(beta1)
over jets and channels. Total auxiliary weight is five in all four students.
Deploy only native HLT weights; heads and privileged inputs are absent.

Final teachers and students select checkpoints by full development-validation
AUC using existing tie rules. Weak finite teachers still export targets and
release all registered students. No performance gate or early stopping.

## Explanation diversity: explicit adaptation, not embedding rotation

Inspired by [Ross et al., Right for the Right Reasons](https://arxiv.org/abs/1703.03717),
penalize sensitivity on input coordinates used strongly by T0. This is an
adaptation to native ParT, not a reproduction of the paper or a guarantee of
causal explanations, disjoint information, or improved HLT recovery.

The explanation score is the true-class logit minus logsumexp of the other
ten logits. Tap both native input streams: 17 particle features before input
BatchNorm, and the four deterministic pairwise physics features before their
input BatchNorm. The pair tap is before any learned pair embedding, not the
attention weights. Its detached leaf avoids singular derivatives through
four-vector-to-pair construction while leaving forward bytes unchanged.
No artificial reconstruction of attention is used. Both streams include only
valid particles/pairs; padding is excluded. Symmetric pairs are counted once.

Compute deterministic FP32, math-SDPA, eval-mode gradients for T0 and TE on
the same jets. Eval mode prevents batch-dependent explanations and extra BN
updates/dropout. Multiply each input derivative by the frozen T0 input-BN
standard deviation, floor .001, separately for particle and pair channels.
Within each jet, mark coordinates with absolute reference sensitivity at
least .67 times the largest absolute sensitivity across both streams; zero
reference coordinates are unmarked. The per-jet penalty is the sum of TE's
squared scaled sensitivities on that mask, divided by the reference masked
energy floored at 1e-4. Average over jets, coefficient1. Gradients flow through
TE's input derivatives to its actual classification parameters. T0, its scale
and mask are frozen. All input fields remain available to TE.

CE uses the original training-mode BF16 path. Accumulate the explanation term
in microbatches of 64, each scaled by its fraction of the training batch;
one optimizer update follows the complete CE and explanation gradients. This
bounds second-order memory without changing the registered per-jet objective.
All models use the same sampler; explanation passes preserve training RNG.
Native eval/inference paths and exported weights remain unchanged.

Execution v2 changes only the grouping of the deterministic explanation passes
from eight jets to 64. Hydra setup160118 completed all miniatures, but its
2.143-second TE step projected to5058 minutes with the registered safety margin,
above2880 minutes. Larger microbatches reduce repeated launch/synchronization
overhead. The per-jet objective, reference, coefficient, full128 CE batch,
single optimizer update and30 epochs are unchanged. Finite-response probes
use the final up-to-eight jets of the CE batch, independently of microbatch
size. For partial batches this also fixes a diagnostic-only dependence on the
last microbatch remainder. New v2 receipts and fresh setup are required;
v1 artifacts retain their original meaning in their pinned worktree. Runtime
and memory ceilings, and the2x timing margin, are not increased.

This measures sensitivity to transformed input coordinates, including relaxed
categorical channels and redundant pair information, not physically independent
detector interventions. Log reference-mask coverage and penalty, and at
epochs1,3,10,20,30 compare finite standardized input perturbations at the same
two taps. This diagnostic can expose a local-gradient penalty that does not
change finite responses. It is never an acceptance gate. CE and embedding
gradient norms/cosines are also diagnostics, not adaptive loss weighting.

## Error focus without in-sample difficulty leakage

Assign three TRAIN folds by class-stratified sorting of SHA256(identity plus
fixed fold seed), round-robin. Identity is metadata only. Each fold reference
starts fresh and trains exactly30 epochs on the complement. It never reads
the held-out fold during training or selects a checkpoint with that fold or
development validation. Export the **last** epoch and predict its held-out
TRAIN fold once afterward. Predictions are authenticated with fold membership,
trained-ID digest, checkpoint, source, and full TRAIN lineage. This is jet-ID
exclusion; it makes no unverified claim of event-level independence.

Combine the three disjoint OOF predictions in original TRAIN order. Raw weight
is `1 + 2*(1-p_true)` (range1..3); divide by the TRAIN class mean weight so each
class keeps its original total mass. TF uses the mean of weight times per-jet
CE (not batch-renormalized). All jets retain positive weight. Labels appear
only in TRAIN loss/preparation. No reporting-validation error influences weights.
T0 is trained on all TRAIN; the fold helpers estimate difficulty of its recipe,
not of the exact full-data T0 weights. This additional computation is explicit.

## Execution and authentication

Reuse the original teacher-sweep specification and native OFFLINE delta in
its pinned worktree; no new SPORC transfer is needed. Preserve original
299234 TRAIN and 74895 validation joins and prepared fingerprint. Historical
M0HLT/OFFLINE anchors are reference values, not matched controls; S0 is the
new matched student control. Original50k overlaps validation. Single-seed
exploratory comparisons cannot establish significance or independent-test gains.

Hydra overrides generic Tigris defaults: A6000, gpus/default/normal,6 CPUs,
32000MiB, setup12h, fit ceiling48h. CPU release/report compute/default/normal,
one requested CPU/4000MiB, accept observed two-CPU allocation rounding. Reserve
8GiB outputs plus1GiB headroom. Fresh clean pushed source and separate root.

Setup performs real native/installed-factory parity, historical BF16 replay,
two-epoch production miniatures, fold exclusion/OOF merge, selected-bank export,
all objective paths including second-order backward, partial and longest-support
batches, and warmed timing with validation/export/diagnostic costs. Memory
limits stay85% host and88% GPU. Fits are admitted with2x timing margin plus
startup/I/O allowances. Technical targets/weights are discarded. TE and students
also run disposable miniatures with their actual completed parents before science.

Queue T0 and independent F0/F1/F2 first; S0 after T0, TE after T0, TF after all
fold helpers; SE after T0/TE, SF after T0/TF, SEF after T0/TE/TF. Report afterany
all ten fits. Dependencies express data requirements only. One bootstrap
review authorizes setup, durable CPU afterok release, the bounded full DAG and
report. Exact-ID journals and exclusive claims stop ambiguous retries. No login
polling. Real setup success is mandatory before automatic science release.

Every artifact is immutable, content-hashed and parent-bound. Wrong role,
missing/nonfinite gradients, corrupt files, stale source and lineage mismatches
fail closed; low accuracy, large finite penalty and weak targets remain results.
Read-only results/progress expose teacher/fold/student roles, selected AUC,
recovery, delta S0, complete/incomplete tasks and per-epoch weighted losses.
