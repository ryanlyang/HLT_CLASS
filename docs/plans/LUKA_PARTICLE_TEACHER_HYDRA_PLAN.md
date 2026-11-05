# HLT-aligned teacher particle prediction on Hydra

Scientific registration v1, authorized in chat on 2026-11-05. Twelve fresh
students reuse the completed cross-message teacher_m16 (original A6000 source
199928da991f8a4194dcf42df009130cfcf84118, selected checkpoint authenticated by its
original worktree). No new teacher fit. This plan governs the new campaign only.

## Scientific question

Does predicting the full contextual representation of each HLT particle in a
strong privileged teacher improve a native HLT classifier beyond U100-w5?
The teacher takes the same HLT particle sequence, injects a privileged U100
cross-read after block4, then processes it through blocks5-8. Targets are its
full block6/block8 outputs, not the compact injected residual. Extraction must
call the teacher's privileged `oracle` forward with its injection enabled.
Calling a native base-class forward would bypass that scientific definition.

Each student uses the native eight-block Particle Transformer, fresh weights,
CE and original frozen COARSE_U100 jet-embedding prediction at weight5. Every
particle head reads the student's final block8 output. Each head is a shared
128->128->128 GELU MLP applied separately to contextual particle tokens. No
student attention injection, retained message predictor, logit KD or privileged
student inputs. Deployment strips all jet/particle prediction heads.

TRAIN-only teacher channel means and population standard deviations (FP64,
real particles, floor1e-3) standardize each teacher depth separately. Prediction
heads output these standardized coordinates. SmoothL1(beta=1) is averaged over
128 channels, valid particles within each jet, depths if present, then jets.
Two depths use separate heads and mean losses, not a doubled total coefficient.
The centered variant uses half full-token loss and half loss between each
prediction/target and its own valid-token jet mean. Single-particle jets have
zero centered term. Padding never enters either mean or loss.

`L = CE + 5 * original_jet_embedding_SmoothL1 + lambda(t) * particle_loss`, with
`lambda(t) = lambda * min(t/3, 1)` at continuous epoch position t. Both teacher
and U100 encoder are frozen/eval. Auxiliary gradients reach the student encoder;
the student's hidden states are not directly equated to teacher coordinates.

## Twelve fits in preferred submission order

| Task | Teacher targets | lambda | Seed |
|---|---|---:|---:|
| final_l1_s0 | block8 | 1 | 0 |
| control_s0 | none; native CE+U100-w5 | 0 | 0 |
| final_l0p3_s0 | block8 | .3 | 0 |
| multi_l0p3_s0 | block6 and block8, mean loss | .3 | 0 |
| final_l1_s1 | block8 | 1 | 1 |
| control_s1 | none; native CE+U100-w5 | 0 | 1 |
| middle_l0p3_s0 | block6 | .3 | 0 |
| final_l3_s0 | block8 | 3 | 0 |
| centered_l1_s0 | block8, full/centered mixture | 1 | 0 |
| multi_l1_s0 | block6 and block8, mean loss | 1 | 0 |
| middle_l1_s0 | block6 | 1 | 0 |
| final_l0p1_s0 | block8 | .1 | 0 |

All use the authenticated D000 population (299234 TRAIN,74895 validation),
batch128,30epochs, existing AdamW and learning-rate schedule. Seed0 preserves
the original U100-w5 backbone/jet-head initialization and sampler; seed1 adds
1000003 to both native seeds. Particle-head construction preserves native RNG.
Existing shared-message controls are not this architecture and are not reused.
Select on HLT-only full-validation AUC under the existing selection key; use
same-seed controls for treatment deltas. Historical SPORC anchors define recovery.
Original50k overlaps validation and is report-only; no independent test exists.

## Targets, resources and queueing

Reuse Brown's authenticated input bundle, encoder delta, adapter delta, original
environment and completed parent setup/teacher proofs. Source is additive from
cc55765a5e6a80ed04b660d99dc182a4aaf4b017. Preserve all existing source/specs/jobs.
No additional transfer, teacher training, or privileged validation extraction.

Dense FP32 targets are built once per supervised allocation, streaming frozen
U100 encoder and oracle forward in fixed128-row batches. Do not retain a second
full U100-token bank. Precheck array/RAM bounds before allocating; actual setup
measures preparation peaks. Free teacher, encoder and U100 raw views before
student fits. Persist only content-hashed target identities, hashes and TRAIN
statistics; every rebuilding job must reproduce them. This avoids large home
disk caches. Controls do not load privileged particles or teacher weights.

Use Hydra gpus/default/normal, one RTX A6000,6CPUs,32000MiB, no node pin or fit
dependencies. Shared setup performs fresh disposable warmups followed by fresh
measured full/partial production fits for all twelve arms. The largest128 rows
stay together; partial batches have at least two jets. Schedule position7 makes
all auxiliary weights active in miniatures. Include preparation overhead once,
2x measured runtime plus900s, 48h fit ceiling,85% memory bounds. Only technical
validity gates release; poor finite accuracy/target variance never rejects fits.

Queue setup plus a durable CPU release (compute/default/normal, one requested
CPU,4000MiB; accept verified scheduler rounding to two allocated CPUs). After
authenticated acceptance and exact dry review, release submits twelve independent
fits and a CPU report with afterany dependencies. Exact-job journals, immutable
claims, source/environment hashes and idempotent complete-stage recovery are
mandatory. Ambiguous partial submissions require inspection, never blind retry.
Four GiB output allowance plus one GiB headroom; no automatic cleanup.

## Evidence and diagnostics

Record raw/weighted CE, jet/particle loss, per-depth/full/centered losses and
prediction/target within-jet variance. These describe learning, not admission.
Tests must cover teacher forward routing, exact HLT slots, frozen gradients,
TRAIN-only normalization, mask/permutation invariance, native inference parity,
matched initial seeds, all twelve objectives, CUDA full/partial production paths,
checkpoint replay, hashed target reuse, and exact dry/live/recovery submissions.
Local tests are not real-data Hydra acceptance. The setup job remains required
before automatic scientific release; no performance gain is assumed.
