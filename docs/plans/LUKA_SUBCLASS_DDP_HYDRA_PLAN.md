# Two-GPU FP32 U100-w5 plus subclass campaign

Active production plan, 2026-11-10. The user authorized implementing the ten
subclass weights and a fresh matched U100-w5 control on two GTX 1080 Ti GPUs.
This separate plan supersedes the benchmark-only boundary for this new execution;
the original BF16 and benchmark specs/readers remain unchanged.

Hydra benchmark job160948 completed in17m24s. User-supplied output reports
two-card step times0.164-0.168s, longest-support success,6.6-7.2GiB allocated per
card and passing numerical probes; one card failed longest-support memory.
These are benchmark observations, not production acceptance or accuracy results.

Use the original299234/74895 population, native128 ParT and frozen TRAIN U100
targets. Rebuild the same within-class subclass bank from authenticated T00.
Retain global batch128,30 epochs, original initialization/sampler seeds, AdamW,
LR and three-epoch subclass ramp. New C0 has subclass coefficient0; the other
weights are.03,.1,.3,.5,1,2,5,10,40,100. C0 keeps the same isolated unused head
to match initialization; its zero-weight term contributes no encoder gradient.
All heads are removed from deployment. Fresh C0 is the primary comparison;
historical BF16 C0 and SPORC anchors are explicitly labeled historical.

Production precision is FP32. One process per GPU uses DDP and SyncBatchNorm.
Build padding from the global batch before splitting. Odd batches use the
sample-weighted rank loss before DDP averaging; no rows are duplicated/dropped.
Gather validation scores with unique row indices, restore original order and
evaluate every validation jet exactly once. Only rank0 publishes artifacts.
Checkpoint selection uses the original native validation/tie rules. Strict
reload into an ordinary HLT-only ParT must reproduce selected predictions.

Gradient probes at epochs1,3,10,20,30 use a separate unwrapped SyncBatchNorm
forward, manually average weighted per-term gradients across ranks, and restore
RNG and buffers before the training forward. They do not change optimization.
Log subclass agreement/entropy and raw/weighted losses as observations only.

One fresh setup requests two GPUs on one node,6 CPUs,48000MiB and4h. It rebuilds
targets, checks native factory and zero-subclass equivalence, measures full
validation replay, executes short full production fits, and profiles all11
weights on real TRAIN batches, partial batches and the longest-support batch.
Memory accounting includes both ranks and launcher overhead. Walltime estimates
include validation, diagnostics, startup and2x safety; each fit is capped48h.
CPU release/report use compute/default/normal,1 requested CPU,4000MiB,1h.

The bounded14-job DAG is setup, CPU afterok release,11 independent fits and
CPU afterany report. Queue C0 first, then the original interleaved weight order.
Successful authenticated setup and exact COMPLETED/0:0 accounting are required
for release. One initial authorization covers the whole DAG. No automatic retry,
quality gate, inference-time offline input or final-test access is permitted.
Real Hydra production miniatures replace legacy Tigris-only qualification for
this user-requested execution; local tests establish queue readiness only.

Implementation and artifacts use the new `luka_subclass_ddp_hydra` namespace,
`LUKA_SUBCLASS_DDP_HYDRA_*/v1`, exact source and immutable parent/file hashes.
Original target recipe hash is preserved. Scientific recipe is new because
precision, normalization and the fresh control differ. Finite poor AUC or target
quality cannot cancel, fail or skip registered scientific runs.
