# FullStat hybrid logit-KD / embedding ladder, Oscar, v1

User-authorized 2026-10-23. A NEW five-fit experiment, not a mutation or resume
of the CE-only cumulative ladder. Existing jobs and artifacts remain untouched.

## Exact science

Use the authenticated FullStat 3,000,000 TRAIN / 500,000 validation split,
unchanged paired views, proportions and selection cuts. Final-test1,500,000
and unused105,325 remain sealed. Event-level independence is still unverified.

| Fresh stage | Input | Frozen logit teacher | Frozen embedding teachers | Embedding coefficient each |
| --- | --- | --- | --- | --- |
| UE000 | U000 | original OFFLINE | OFFLINE, original M0HLT | 2.5 |
| UE100 | U100 | new UE000 | OFFLINE, original M0HLT | 2.5 |
| DE066 | D066 | new UE100 | OFFLINE, new UE100, original M0HLT | 5/3 |
| DE033 | D033 | new DE066 | OFFLINE, new UE100, new DE066, original M0HLT | 1.25 |
| DE000 | D000 | new DE033 | OFFLINE, new UE100, new DE066, new DE033 | 1.25 |

All UE/DE teachers are selected models from THIS new campaign. In particular,
UE100 consumes UE000 logits but does not predict UE000 embeddings. The final
DE000 does not predict M0HLT. No earlier CE-only or original logit-ladder model
may substitute for a new ancestor. Every rung starts from scratch, not a warm start.

For N embedding teachers, the objective is

`0.25 CE + 0.75 T^2 KL(q_teacher(T) || p_student(T)) + (5/N) sum_j SmoothL1(g_j(h), z_j)`.

T=2, forward KL batchmean, temperature applied to both logits before softmax,
exactly one T-squared factor. CE is evaluated at T=1. Each SmoothL1(beta1) is
independently mean-reduced over batch and128 dimensions. The 75/25 mixture
does NOT scale the embedding loss. Sum of auxiliary coefficients is5; this
does not mean identical gradient norms across stages. All targets are detached.

Maintain20 complete epochs, physical batch128 with final partial batch,
coordinate-paired initialization/sampling seeds, original FullStat20 AdamW
groups and two-epoch warmup/cosine schedule. BF16 native CUDA forward; FP32
heads and losses. No early stopping, adaptive weights, attention, fusion,
relational loss, batch fallback or pretraining/warm starts.

Each embedding is the frozen selected teacher's native128-D pre-classifier
representation, never an auxiliary prediction. Use separate128-GELU-128 heads,
isolated head initialization streams and independent TRAIN-only FP64 population
normalization (scale floor1e-3), storing FP32 standardized targets. Preserve
existing OFFLINE/UE100/DE066/DE033 head seeds; give M0HLT its own fixed stream.
Embeddings and predictions never feed the classifier. Only native weights are
saved as selected checkpoints. UE000/UE100/DE066/DE033 are richer-input oracle
diagnostics; DE000 deploys unchanged native HLT-only ParT with no heads.

Reuse the original OFFLINE embedding bank byte-for-byte. In the first job,
export original M0HLT embeddings from D000 TRAIN and OFFLINE temperature-two
probabilities from OFFLINE TRAIN, loading/releasing these caches sequentially.
After each selected non-final fit, export its native temperature-two TRAIN
probabilities using the original BF16-CUDA/FP32-softmax reducer convention.
UE100/DE066/DE033 additionally export native FP32/eval embeddings. UE000 does
not export an unused embedding bank. Reuse each fit's TRAIN cache for outgoing
exports. No ROOT reconstruction, new particle cache or validation/test targets.

Selection remains full-validation AUC, then CE, logR50 and earliest update.
One selected checkpoint is used for all R30/R50 comparisons. Report all20
epochs, update counts, CE, T-squared KD, per-head losses, their weighted terms,
progress, selected metrics and recovery. Compare to authenticated historical
M0HLT/OFFLINE, original DIRECT_D000/COARSE_D000, U100-w5 and four-teacher model;
no new control fits. Single-seed adaptively inspected validation evidence is
not an independent-test result or a compute-matched comparison. Poor finite
science is always published and releases technically valid descendants.

## No standalone preflight: bounded acceptance reuse

Reuse the completed b21c2d4e four-teacher campaign, original native/KD
seven-coordinate Oscar acceptance, exact successful jobs and frozen parent
lineage through original-source authentication. Source bytes, installed
environment, backend, L40S, batch128 and support/population summaries must match.
This explicitly replaces a NEW standalone preflight/cluster miniature for
this requested experiment; it does not claim the combined objective was timed.

Queue exactly five scientific jobs, afterok chain UE000 -> UE100 -> DE066 ->
DE033 -> DE000. Bind all prior tasks as dependencies for transitive lineage.
No preflight, setup, reducer, aggregate job or login-node controller. Export is
inside each applicable job and task completion is published LAST. A fit receipt
alone is insufficient. Fail technical corruption, not disappointing accuracy.

Request6CPUs,180000MiB,L40S. Inherit max(original accepted training request,
four-teacher request)+120min; first fit gets180min extra instead for its original
exports. Reject requests above48h rather than clamping. Reserve16GiB personal
durable headroom. Check conservative cache/bank/export host-memory bounds.
The new objective/runtime is unmeasured and these requests are not guarantees.

Inside every scientific job, perform three disposable maximum-TRAIN-support
backward/optimizer updates with the actual KD/head set, max-validation-support
forward, native tap equality, finite gradients and memory checks. No benchmark,
technical epoch or full-validation probe. Discard that model and reset fresh
scientific initialization before the20-epoch fit. Never change science to fit
an allocation. Old active/cancelled jobs are not changed by this experiment.

## Provenance and acceptance

New additive `luka_fullstat_hybrid_embedding` namespace, version1 contracts,
thin CLI/helper/Slurm wrapper and exact dry/live hash-reviewed5-job submission.
Exclusive claims/journals prevent duplicate or ambiguous retries. Clean pushed
execution source required. No jobs submitted by implementation alone.

Implementation donor: cumulative ladder at
d98359f76abe3a46c70f948aa4b99d51c82787e9. Runtime acceptance anchor:
b21c2d4e01211a31e57d879c5a86ca90aab2ca0b. Integration baseline:
6b76a6777952c8faa2b155084ef560518c6fa55a. Existing runtime files are unchanged.
The explicit, unimported concurrent trial `luka_particle_prediction` file list
may be newly added in the same commit chain; no existing runtime modification
is exempted. All committed bytes still belong to the pinned source snapshot.

Require exact KD formula/gradients/update parity, all teacher maps and TRAIN
joins, temperature/roles/lineage corruption failures, native export/readback,
fresh-seed checks, failed-parent blocking and poor-score continuation. Exercise
all five production workers and100 scientific epochs on a local synthetic
installed-Weaver GPU miniature; report mocked cluster boundaries explicitly.
Local evidence does not constitute new Oscar/Tigris resource certification.

See [contract](../contracts/LUKA_FULLSTAT_HYBRID_EMBEDDING.md) and
[queue runbook](../LUKA_FULLSTAT_HYBRID_EMBEDDING_RUNBOOK.md).
