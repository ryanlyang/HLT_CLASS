# Group C v1: embedding distributions and measured-set denoising

Authority: [active plan](../plans/LUKA_SEVEN_CANDIDATE_IMPLEMENTATION_PLAN.md)
and [candidate catalog](../plans/LUKA_NEXT_SUPERVISION_CAMPAIGN_IDEAS.md).
Internal implementation donor: `3b839fc3aef7e1e24e886b9dd512a6a8114b5473`.
This extends the shared `luka_next_supervision_hydra` package. Original A/B
recipes, specs, data, worktrees and running jobs retain their meaning.

## Scope and immutable roles

Seven new fits, in filtered global order: `D1 S1 D2 S2 D5 S5 D3`.
D1/D2 use K4, lambda0.3, seeds0/1; D3 uses K4/lambda0.1/seed0;
D5 uses K1/lambda0.3/seed0. S1/S2 use full OFFLINE sets/lambda0.1/seeds0/1;
S5 uses the corresponding full HLT-set target/lambda0.1/seed0.
Every fit retains native `CE + 5*original standardized U100 SmoothL1`,
batch128, full30 epochs, original optimizer/schedule/selection and seed offset.
No additional teacher fit, warm start, reference rerun or logit KD is introduced.

The from-scratch HLT encoder, classifier and original w5 head stay unchanged.
New auxiliary heads use the existing isolated seed2026110507 plus replicate
offset1000003. New loss weights ramp linearly over three epochs. All auxiliary
modules are removed for native-only checkpoint replay and deployment.

New PROBABILISTIC_RECIPE/SPEC/TARGETS and SET_BANK envelopes are v1 with canonical
hashes, exact parent hashes, source/environment lineage and immutable publication.
Shared resource/acceptance envelopes remain execution v2. Training reports bind
the frozen Group C target manifest as an additional parent. A/B recipe hashes
and scientific target definitions are unchanged.

Only full TRAIN299234 raw HLT/OFFLINE sets and standardized U100 targets may
prepare new targets. Their identities/labels join the same authenticated D000
TRAIN order; labels are never decoder inputs or mixture-initialization inputs.
Native development validation74895 selects checkpoints. No new validation
embedding bank, raw validation set target, held-out density audit or independent
test is admitted. Original50k overlaps validation and remains report-only.

## D: coherent whole-vector Gaussian mixture

Separate GELU MLP `128 -> 256 -> 256` predicts K means, K diagonal scales and K
mixture logits for 128D U100 targets. One shared component index describes the
whole vector. Compute coordinate-summed Gaussian log probabilities, add log
mixture weights, logsumexp over components, divide by128, then average jets.
Everything in this likelihood runs in FP32 outside BF16 autocast. Valid negative
differential NLL, weighted NLL and total loss are accepted if finite; CE and the
original embedding loss still must be nonnegative. Nonfinite required values fail.

Scales are `0.1 + 1.9*sigmoid(raw)`, initialized at1. Mixture logits initially
zero; mean residual weights have normal std0.001. Biases initialize from frozen,
label-free, TRAIN-only MiniBatchKMeans centers for K4 or K1. Exact sklearn settings:
k-means++ initialization, random_state2026110509, n_init3, batch_size4096,
max_iter100, max_no_improvement20, reassignment_ratio0, tol0. Initialization
centers are not fixed subclasses, balanced assignments or a diversity penalty.
Collapsed/duplicate centers and unused components are scientific observations.

## S: full mixed-field velocity denoising

Raw fourteen-field physical sources are used directly; clipped model features
are never inverted. The observable frame is the paired HLT sum-p4 eta/phi axis
and its vector pT (floor1e-12). All particles are retained, regardless of rank.
The numerical target has22 channels:

- Five kinematic channels: log(pT/HLT-pT), log(E/HLT-pT), eta offset,
  sine and cosine of phi offset. Positive log floors are1e-12.
- Nine composition channels: six measured PID categories including unknown,
  and one-hot charge -1/0/+1.
- Eight tracking channels: asinh(d0/mm), asinh(dz/mm), log(d0err/mm),
  log(dzerr/mm), and their four effective validity indicators. Errors require
  declared validity and strictly positive reported error; unavailable values
  contribute neither normalization statistics nor value loss.

Continuous channels are standardized with source-specific full-TRAIN masked
particle moments in FP64 and std floor0.001. One-hot/validity channels retain
their numeric encoding. Streaming two-pass preparation publishes normalized
FP32 memory-mapped values, packed four-bit tracking masks, ordered identities,
offsets and normalization metadata. No dataset-wide raw duplicate is loaded in
each worker. Missing value coordinates are encoded as zero after normalization.

For each anchor/epoch, an isolated NumPy stream (seed2026110510, replicate,
epoch, TRAIN index) permutes its rows and samples t uniformly in[0.2,0.95].
With alpha=cos(pi*t/2), sigma=sin(pi*t/2), independent Gaussian epsilon gives
`Yt=alpha*Y0+sigma*epsilon` and `v=alpha*epsilon-sigma*Y0`.
Every numeric encoding, including composition and validity, is noised.
This is continuous-encoding denoising, not a categorical likelihood or a
calibrated detector simulator. No iterative sampling enters training.

The removable decoder has four width128, four-head self-attention blocks,
FF width256, pre-LayerNorm, GELU and zero dropout. Each block has learned FiLM
scale/shift from pooled HLT h plus(t,alpha,sigma). There are no particle positional
IDs, HLT/offline particle matches, clean target coordinates or class inputs.
Only noisy target rows, t and the real-padding mask enter the decoder.
Independent chunks of up to8 jets trim padding to their own maximum while
retaining every real row. This changes memory use, not particle support.

Squared velocity error is averaged over valid channels/particles within each
jet and each of the three groups; groups receive equal weight, then jets.
Absent value groups use safe denominators and contribute zero. Padding is masked
before arithmetic; tracking validity channels remain scored when values are absent.
A separate h-only count head predicts0..source TRAIN Nmax with categorical CE
divided by log(Nmax+1), weighted0.1. Its initialization seed2026110511 is isolated
and its frozen output width is configured before optimizer construction.
The original classifier never receives the noisy set, its cardinality or mask.

## Admission, diagnostics and independent release

Register a fresh C-only spec from the existing pinned A input spec, validating
the latter in its own source process. Reuse completed C0/C1 proofs. No A/B fit
or B target-preparation output is needed. Reserve8GiB after existing inputs and
environment plus1GiB filesystem headroom. Target preparation also checks its
projected bytes before allocation and reserves1GiB of the budget for outputs.

One explicit A6000 setup (6CPUs/32000MiB/up to12h) prepares targets once and
checks installed-Weaver FP32/native-w5 initialization parity. Genuine warmed
full128 and partial batches cover four paths: K4, K1, OFFLINE sets, HLT sets.
Set miniatures include128 largest source sets,128 greatest valid-track support,
native extremes and a real partial batch; there is no32/64-particle crop.
Acceptance measures target loading, all backward/optimizer paths, probe cost,
preparation peak host memory, GPU peak and full30 walltime with the existing
2x+900s margin,85% memory limit and48h fit ceiling. Actual Nmax/tails, channel
support, target identities and initialization settings appear in science review.

The exact successful setup is required before reviewing/submitting all seven
independent fits. C uses its own authorization token/job prefix, exact journals,
claims and idempotence. No fit dependencies, holds, nice adjustments, family
lanes, arrays/concurrency caps or automatic retries are added. Combined full40
release is not inferred from additive C authorization; the registry reports
all40 methods implemented, with separate A/B/C execution scopes.

First-TRAIN-batch probes record raw/weighted auxiliary jet-gradient norms,
mixture entropy/responsibilities/scales/mixture-mean error or grouped velocity
error/count CE. Identical targets/noise are compared with zero h, within-class
shuffle and within-class/coarse-HLT-pT/count shuffle (bins floor(log vector-pT)
and floor(real HLT count/16)); actual swapped fractions
are disclosed. These are adaptive TRAIN mechanism probes, not held-out evidence.
Finite poor targets/conditioning/AUC never cancel or shorten a registered fit.
Read-only results/progress retain all incomplete rows and matched-seed deltas.
