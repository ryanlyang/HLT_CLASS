# Frozen HLT privileged adapters on Hydra — v1

This is the active scientific implementation plan. The first campaign contains
two privileged-adapter fits and twelve fresh HLT student fits. It extends the
verified U100 pair-transfer population; it does not modify another campaign.

## Question and comparisons

Can privileged information improve a *fixed* HLT computation through a small
adapter, and can an HLT-only student learn useful approximations of that
adapter? A strong oracle does not establish that its corrections are predictable.

Reuse the selected historical M0HLT checkpoint. Freeze all its parameters and
buffers, including normalization statistics; keep its transformer in evaluation
mode throughout oracle training. Train only a new adapter by CE. Two independent
oracles intervene at particle block index 6 (the seventh of eight blocks):

* `teacher_bias`: a bounded, query-row-centered residual added to the existing
  eight-head attention bias. Native HLT pair bias and Q/K remain present.
* `teacher_edge`: four bounded pair-message coefficients, aggregated with each
  head's native HLT attention probabilities, projected to its 16 value channels,
  and passed through the existing attention output projection without its bias.
  This adds messages before the block residual. It does not replace native values.

Both adapters see normalized contextual HLT tokens at that block, native HLT
pair features, and slot-aligned U100 particle features/pair features. U100 keeps
HLT particle slots and retains HLT values for unmatched particles. No source
index, matching flag, or degradation-construction metadata is an input.

## Small first architecture

For each output channel, use a rank-four ordered token-pair interaction plus a
small pair-geometry MLP (16 hidden units). Token context is projected through
32 hidden units. Native pair features are log(kt), log(z), log(delta-R), and
log(mass squared), with the existing physical-pair conventions. Divide these
log features by 10; this fixed scale is not fitted on validation. Diagonal
geometry features are zero. Apply `2*tanh(raw/2)` and mask padding. Bias residuals
are subsequently centered over valid keys, removing softmax's constant-shift
ambiguity. Edge coefficients have four channels shared across heads, with a
learned per-head 4-to-16 message basis. Pair computation uses query chunks and
activation checkpointing; no persistent quadratic target bank is created.

Zero output layers make each oracle initially exactly M0HLT. The edge basis is
small random, not also zero, so CE can train the coefficients immediately.
Bias-only backward through a frozen backbone uses PyTorch's math SDPA backend
for the intervened attention operation only, to support additive-mask gradients
in the pinned Torch version. Earlier frozen context and inference retain the
native backend. Kernel roundoff means zero-adapter identity is numerical, not a
cross-backend bitwise claim.

## Twelve students

For each site (`bias`, `edge`) and embedding weight (0, 5), run:

| Mode | Predict frozen adapter target | Inject HLT-only correction | Purpose |
| --- | --- | --- | --- |
| `aux` | yes | no | representation supervision alone |
| `ce` | no | yes | architecture/capacity control trained by CE |
| `both` | yes | yes | supervision plus useful retained correction |

All students start from scratch with the historical M0HLT initialization and
minibatch seeds. Their predictor sees their own contextual HLT tokens and HLT
pair features only. The target is the selected frozen oracle's correction for
the same TRAIN jet. An eight-by-eight head mixer (identity initialized) for bias,
or a learned per-head message basis for edge, adapts teacher coordinates to the
fresh student's computation. Supervision applies **before** this mapping; CE
trains both the mapping and predictor in injected modes. Never match final
attention matrices or force new heads to have the oracle's interpretation.

The loss is `CE + E*U100_embedding_SmoothL1 + ramp*correction_SmoothL1`.
`E` is 0 or 5. Correction weight is 1 in `aux`/`both`, 0 in `ce`; ramp increases
linearly over the first three epochs. Correction loss is averaged over channels
and valid ordered particle pairs within each jet, then over jets. Targets are
detached and divided by channel RMS measured on the first 4096 authenticated
TRAIN rows using the selected teacher (floor 0.05). No logit KD. Embedding loss
and target bank retain the historical U100-w5 definition. Poor target RMS or
oracle validation quality is reported, not used to cancel fits.

Use the original batch128 AdamW full30 recipe and CE coefficient 1. Oracle
adapters use AdamW with constant LR 3e-4 for 30 epochs; the frozen model is never
unfrozen. Select each checkpoint by full validation macro AUC with the original
tie-breaking rule. Oracle validation explicitly uses privileged U100 inputs;
student validation and deployment use only D000. Persist full selected oracle
weights, target RMS, model/recipe identities, per-epoch losses, predictions,
selected deployment checkpoints, and artifact receipts. Drop embedding and
unused auxiliary-only predictors at inference; retain injected predictors and
their alignment maps. This changes inference cost for injected students.

## Inputs and execution

Reuse the already authenticated Brown pair bundle and completed Hydra precision
diagnostic. Export a separate small delta from SPORC containing only the original
M0HLT checkpoint and validation U100 shards. Bind it to original pair campaign
evidence and hashes; do not retransfer TRAIN data, raw datasets, or U100 targets.
Validation U100 is admitted only for oracle selection, never student inference.

One shared setup authenticates inputs, replays M0HLT, runs production-kernel
miniatures (including backward, checkpoint publication/reload and all six
student modes), and measures time/memory on the L40. It is not a separate
preflight. Two oracle fits and twelve students follow. There are no scheduling
lanes or concurrency cap. Target-supervised students depend only on their own
teacher; CE architecture controls do not. An after-any report includes failures
as pending/incomplete rather than inventing results. No existing job is changed.

Hydra uses `gpus`, `default`, `normal`, one `nvidia_l40` GPU, six CPUs and
120000 MiB. It has no SPORC `debug` partition. Reserve eight GiB for new outputs
plus one GiB filesystem headroom. Setup measures each family against a reviewed
24-hour envelope; resource failures are technical failures, never quality gates.

Historical SPORC M0HLT and OFFLINE remain the recovery anchors. Include matched
U100-w5 and its Hydra replay; explicitly identify mixed-site and single-seed
comparisons. This is exploratory development validation, with no independent
test; original50k overlaps validation and is report-only. No promise of a gain.

Recommended early order: bias both+w5, edge both+w5, then their auxiliary-only
and CE controls, then standalone counterparts. Dependency eligibility and the
scheduler, rather than submission order, determine actual start times.
