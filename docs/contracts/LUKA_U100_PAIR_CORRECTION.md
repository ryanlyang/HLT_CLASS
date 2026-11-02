# U100 particle/pair correction contract v1

Scientific authority: [the four-run plan](../plans/LUKA_U100_PARTICLE_PAIR_CORRECTION_PLAN.md).
Implementation: [luka_u100_pair_correction](../../src/hlt_classification/luka_u100_pair_correction/).

## Fixed comparison

Exactly four fresh students, original 299,234 TRAIN / 74,895 validation jets,
physical batch128, all real particles up to the inherited capacity512,
30 complete epochs, original AdamW and three-epoch warmup/cosine schedule.
Every student retains **CE1 + U100 embedding5**, without student logit KD.
The completed `u100_w5_s0` job21993056 is the matched saved reference, never
retrained or used to initialize a student. Original50k is report-only and overlaps
validation. Historical ALLJETS disclosures and sealed-test boundaries propagate
to every artifact.

| Arm | Retained prediction | Additional full-strength loss |
| --- | --- | --- |
| `pair_u100` (A) | Direct normalized pair residual | pair |
| `p4_u100` (B) | Physical corrected particles, derived pairs | particle |
| `p4_pair_u100` (C) | Same predictor as B | 0.5 particle + 0.5 derived-pair |
| `dual_u100` (D) | Independent particle and direct-pair branches | 0.5 particle + 0.5 direct-pair |

Each new term uses SmoothL1(beta1), mean over four fields, then eligible items
within each jet, then **all jets**, including zero-eligible jets. Multiply the
new budget by `rho=min(1, update/(3*steps_per_epoch))`. CE and embedding
coefficients never change. D's pair loss does not supervise its particle branch;
its attention correction is the half-sum of the two branches.

## Geometry and model

U100 targets use the original authenticated matching assignments and HLT slot
order. No offline tail or unmatched replacement is introduced. Only matched
real slots and unique matched pairs receive physical supervision; all real HLT
slots receive predictions at inference. Mapping indices, matching flags, labels,
IDs and offline inputs are absent from the model signature.

Pair features are native Weaver `[log(kT), log(z), log(delta), log(m2)]`, using
rapidity and wrapped phi. The differentiable companion preserves native forward
values at exact collinearity; below angular distance1e-8 its square-root
derivative is stopped to avoid the singularity. No native attention calculation
is replaced. Per-particle targets are `[log(pt), rapidity, phi, log1p(m2/pt2)]`.
The decoder uses the HLT axis sign, bounded residuals and the plan's softplus
mass parameterization with1e-8 floor. It preserves positive pt/energy and
physical mass; no unit conversion is assumed. Materially nonphysical source
vectors fail. Tiny negative mass squared within `1e-6*max(E2,p2,1)` is counted
and projected only for the auxiliary coordinate transform.

Frozen TRAIN scales are zero-centered RMS residuals, giving equal weight to
eligible jets. Particle floors are `[1e-3,1e-4,1e-4,1e-4]`, pair floor1e-3.
Pair-MLP input mean/std uses all real HLT unique pairs, equal weight per eligible
jet, std floor1e-3. The original U100 embedding normalization is unchanged.

The predictor reads native block1 output (128 channels). Particle head:
LN128–Linear64–GELU–Linear4. Direct branch: LN128–Linear32–GELU, followed by
a symmetric sum/absolute-difference/product/native-pair input and
Linear100→64→32→4 with GELUs. Final prediction layers are zero-initialized;
hidden layers are not. Isolated seeds2026110201/2026110202 preserve backbone,
embedding-head and sampler RNG streams. Pair-MLP activations are checkpointed
in4096-pair chunks; there is no dense retained N×N×100 feature tensor.

One zero-initialized bias-free Linear4→8 maps soft-capped residuals to
`alpha*log(2)*tanh(W(6*tanh(c/6)))`. The resulting symmetric bias is added to
the existing native bias at blocks2–7, once per forward, with zero added
diagonal/padding. Alpha follows rho in training. **Validation and deployment
always use alpha1**, so selection evaluates the deployed model. Class-attention
blocks and native classification head are unchanged. CE and embedding loss
backpropagate through retained corrections.

## Artifacts and validation

All new artifacts use `LUKA_U100_PAIR_CORRECTION_<KIND>/v1`, canonical content
hashes and explicit parent hashes. Kinds include RECIPE, REFERENCE, SPEC,
PARTICLE_BANK, BUNDLE, EPOCH, TRAIN_PROBE, TRAINING_REPORT, DEPLOYMENT, FIT,
PROFILE, RESOURCE_DIAGNOSTIC, ACCEPTANCE, TASK, RESULTS and submission records.
No old schema is changed. File references include size and SHA256; publication
is atomic and immutable.

The REFERENCE is authenticated through the original contrastive source and
exact accepted job/receipts. It binds original U100-w5 results, teacher,
prepared population, full existing embedding payload and installed environment.
No target re-export occurs. New physical targets are compact ragged p4,
matched masks, offsets, identities and labels; pair targets are computed on demand.
No validation physical-target bank is built. Missing old input blocks may be
reconstructed in RAM only if their original byte hashes and view fingerprints
match; present corrupt blocks fail.

Selection uses the original full-validation AUC/CE/rejection/update tie-break.
The selected checkpoint contains native weights, active correction heads,
bias projector and frozen scales. Only the jet embedding predictor is removed.
Reloaded deployment must reproduce selected validation probabilities/metrics.
It needs only HLT features, vectors and padding mask; no TRAIN path is required.
Its manifest binds arm, tap/injection blocks, geometry, scales, alpha and timing.

TRAIN probes preserve RNG/modes/buffers and record errors, coverage, bounded
correction magnitudes, branch disagreements and encoder-gradient norms/cosines.
Poor finite AUC or alignment is a result, never an admission threshold.
Nonfinite required losses/gradients, invalid inputs and lineage corruption fail.

## Execution

SPORC `debug`, `reu-aisocial`, `qos_tier3`, one A100, six CPUs,120000MiB.
One shared setup authenticates reuse, builds compact physical targets and runs
real production miniatures for all four variants, largest-support and largest
matched-pair backward passes, full validation and representative timing.
It records diagnostics before resource admission. Setup requests12h; measured
fit requests cannot exceed24h. Report requests2h. Storage allowance8GiB plus
1GiB free headroom is fixed, not the caller's entire free-space reading.

The exact full graph and resource envelope are dry-reviewed before setup.
An authorized detached dispatcher queues the four fits after successful setup,
without a second interactive step. C/A start the two lanes; D follows C and B
follows A via `afterany`. Report follows all four via `afterany`. Six jobs total;
no standalone preflight or reference fit. Student failures do not cancel other
students. Setup failure prevents new fits. Claims/journals prevent duplicate
submission and preserve ambiguous partial acceptance for explicit diagnosis.

Local tests do not establish SPORC feasibility. Genuine shared-setup acceptance
is required before those scientific jobs are automatically submitted.
