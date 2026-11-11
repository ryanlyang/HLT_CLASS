# Two-GPU subclass execution contract v1

Authority: [active plan](../plans/LUKA_SUBCLASS_DDP_HYDRA_PLAN.md).
Artifacts use `LUKA_SUBCLASS_DDP_HYDRA_*/v1`, content hashes, exact parent
hashes, inherited evaluation disclosures and atomic publication. The original
BF16 execution and bounded benchmark contracts cannot satisfy this execution.
The frozen TRAIN target recipe retains hash
`2794d55556ae6d031f1347c00437df918e4261bea5f0c10016d38af3c980e60c`.

## Scientific semantics

Eleven independently initialized native128 HLT students use the same
299234 TRAIN and74895 validation jets, original seeds/order/AdamW/LR, global
batch128 and30 epochs. C0 uses CE +5×U100 SmoothL1; the ten subclass arms add
conditional KL with maximum weights0.03,0.1,0.3,0.5,1,2,5,10,40,100 and the
original three-epoch linear ramp. The auxiliary subclass head exists in C0
at zero weight to preserve architecture and execution; its gradient to the
encoder is zero and it is absent from exported inference.

Targets remain the original small-data T00 CE OFFLINE TRAIN embeddings,
class-conditional clustering and calibrated soft assignments. They are built
once, frozen, and bound to the source checkpoint, bank, preprocessing, ordered
jet identities and labels. U100 targets are the original authenticated bank.
No validation target fitting, new teacher, FullStat target substitution, or
final-test access is permitted. Bad finite scientific performance is a result.

Two local ranks use FP32, DDP and SyncBatchNorm, with TF32 disabled. Both see
the identical global permutation and padding before splitting the jet axis.
Each local mean loss is scaled by `world_size * local_rows / global_rows`
before DDP averages gradients. Normal training dropout uses deterministic
rank-local streams. Exact dropout equality with the old single-GPU BF16 run
is not asserted; fresh FP32 C0 is the primary comparison.

Diagnostic epochs1,3,10,20,30 measure the CE, weighted U100 and weighted
subclass gradients on the shared encoder. The observation pass bypasses DDP's
reducer, retains SyncBatchNorm collectives, reduces parameter gradients
explicitly, and restores RNG state and buffers before the actual update.

## Evaluation and artifacts

Validation partitions every global evaluation batch without duplicate sampler
padding. An empty final local shard is allowed. Gathered row indices must be
exactly the complete ordered population, with finite normalized probabilities.
Rank0 computes metrics and publishes epochs/checkpoints/receipts. Selection
uses the original native full-validation tie rules. The selected checkpoint
must reload into ordinary HLT-only ParT and reproduce selected metrics. It
contains `mod.*` tensors only; the full training checkpoint also retains heads.

Every EPOCH binds spec, recipe, U100 bank and subclass bank and records update,
LR, loss components, subclass diagnostics and timing. FIT authenticates the
training report, selected checkpoints, ordered probabilities and deployment
profile. TASK authenticates the exact accepted Slurm job, environment and
output inventory. A missing TASK receipt is incomplete, even if weights exist.
Historical BF16 C0 is explicitly labeled; missing fresh C0 never falls back
to it for a matched contrast. Original50k overlaps validation and is not an
independent test. Progress reports selected-so-far and same-epoch fresh-control
contrasts separately.

## Execution and release

GPU jobs request one Hydra node, one Slurm task, six CPUs,48000MiB, and
`gpu:geforce_gtx_1080_ti:2`. The launcher starts exactly two torchrun processes;
the worker rejects a different allocation/device family/precision. CPU release
and report use compute/default/normal, one requested CPU and4000MiB; normal
Slurm CPU rounding is accepted. Source/environment hashes and typed GPU claims
are checked before work. Python user packages and bytecode are disabled.

Setup runs cache/target preparation in a disposable process, then exercises
the production distributed worker. Required evidence includes native factory
FP32 parity, original U100 C0 AdamW-update parity, one-rank/two-rank numerical
parity at C0/S0p5/S100, historical checkpoint FP32 replay, all eleven actual-target
two-epoch miniatures, ordinary native reload, longest-support and partial
batches, twelve warmed timing samples per arm and full validation timing.
An additional paired S100 update checks the diagnostic with normal dropout
enabled: CPU/device RNG must match, loss/buffer differences stay below1e-6,
and gradient/parameter/AdamW-moment relative differences below5e-6.
The historical replay tolerance is an absolute AUC difference of5e-5 on the
same checkpoint; this checks numerics, not new-model quality.

Host memory counts both rank high-water RSS values plus launcher RSS, compared
with85% of48000MiB; each GPU must stay below88% of its measured physical memory.
Single-process preparation and miniature peaks are also included. Walltime uses
measured training, validation, startup, probes and loading with a2× multiplier
and30 minutes total fixed allowance, capped at48h. No cached benchmark estimate
can replace these measurements. Miniature weights are discarded.

The initial review covers setup, its `afterok` CPU release, an upper bound of
eleven independent fits, and the `afterany` report. Only measured walltimes may
change during release. Exact completed setup accounting, verified receipt and
recomputed acceptance are required. Original authorization binds this bounded
DAG; there is no second interactive prompt. Journals, exclusive live claims,
no-requeue and exact-ID recovery prevent blind duplicate submissions. Failed
setup or ambiguous partial submission preserves evidence and releases nothing.

This Hydra-specific real setup is the production acceptance gate under the
active plan, superseding the generic legacy Tigris miniature requirement for
this campaign only. Local CPU/single-GPU tests establish queue readiness, not
two-GPU research qualification.
