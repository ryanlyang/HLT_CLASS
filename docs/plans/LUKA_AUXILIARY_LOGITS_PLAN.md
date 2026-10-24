# c50 auxiliary-logit DE000 follow-up, v1

User authorized 2026-10-24: four fresh DE000 fits, debug, no separate preflight.
This plan supersedes a new standalone preflight for THIS follow-up only. Reuse
authenticated completed c50 execution and perform a bounded disposable safety
check inside target export. No parent jobs, files, models or source are changed.

## Registered experiment

Reuse selected frozen OFFLINE and **c50** UE100, DE066, DE033. Never substitute
c25/c75/anneal or historical COARSE/LOGIT teachers. Require completed c50 DE000
as the existing same-teacher comparison. Do not wait for other parent arms.
Retired parent TRAIN payloads are not needed: re-export four compact paired
embedding/probability banks once from the retained selected native models.

All four new students are native HLT D000 ParT, initialized from scratch with
the parent's c50 DE000 encoder seed, sampler, batch 128, exactly 30 epochs and
warmup3/cosine30 AdamW recipe. No warm start. Same 299,234 TRAIN / 74,895 validation;
original50k report-only, former test already retired into TRAIN, NO independent
test. Poor finite performance always completes the registered experiments.

Loss: `0.5 CE(main,y) + alpha/4 sum_i 4 KL(q_i(T2)||p_i(T2))
+ 5/4 sum_i SmoothL1(g_i(h),z_i)`.
Arms: `k0=0`, `k025=.25`, `k05=.5`, `k1=1` total auxiliary-logit coefficient.
Main native 128->11 head receives CE ONLY. Its coefficient remains 0.5 to
match the parent; this is not a CE-coefficient-1 experiment. Four separate
linear 128->11 logit heads and four separate 128-GELU-128 embedding heads
share the differentiable native fc-input 128-D representation.
Each head has an isolated fixed CPU RNG
stream identical across arms. Embedding initialization matches parent c50.
No extra dropout, projection bottleneck, teacher-weight initialization, averaged
embedding axes, token loss, attention loss or logit KD on the main head.

Frozen teacher probabilities use existing T2 BF16-native/FP32-softmax export;
embeddings use native FP32 eval fc-input, independent TRAIN-only FP64 moments
and 1e-3 scale floor, FP32 standardized targets. KL batchmean with one T-squared
factor; SmoothL1 beta 1, mean over batch and 128 axes. Targets are detached.
At alpha 0 logit heads are evaluated for diagnostics but excluded from loss
(no zero-gradient AdamW updates); encoder/head RNG is still paired.

Only main-head full-validation AUC selects native-only checkpoints using the
existing CE/logR50/earliest-update tie rules. Auxiliary predictions never feed
the main head. All auxiliary heads are removed at deployment. Save all 30 raw
and weighted per-teacher losses, validation histories, selected native models,
validation predictions, cost and lineage records. References: original M0HLT,
OFFLINE, OFFLINE-w5, U100-w5, and parent c50 DE000. This tests the combined
multi-logit/head-separation strategy, not an isolated architectural ablation.

## Execution and storage

Six exact hash-reviewed afterok jobs: targets, four independent fits, CPU report.
Submit the fits in descending auxiliary-logit weight: k1, k05, k025, k0.
All four still depend only on targets, not one another. This is submission
order, not a guarantee of Slurm start/completion order; parallel starts remain
allowed. Order is bound in the immutable spec and reviewed plan hashes.
No standalone preflight, full technical epoch, controller, automatic retry or
retraining of earlier rungs. Require completed original-source c50 receipts,
selected checkpoints and exact accepted-job accounting. Preserve old runtime
bytes and parent evidence; newly exported banks live only in the new root.

Debug/A100, 6 CPU / 90,000 MiB, each GPU job 8h minimum or larger inherited fit
allowance (reject above 24h). CPU report: 2h / 16 GiB. These are conservative allowances,
not remotely measured new-objective timings. Exact environment/GPU/allocation
must match parent acceptance. In targets: installed-native output parity,
CE-only classifier-gradient check, all four objectives at TRAIN maximum support
(physical batch 128, three disposable updates), and maximum-validation forward. Reject
nonfinite/memory violations; discard all technical models before science.

Reserve 3 GiB personal durable headroom. Four compact TRAIN banks are shared by
all fits and retired only after every fit and saved report validate. Delete only
their four exact canonical newly owned `targets.npz` leaves, with hashes and
retirement journal; keep models/results/manifests/moments/source/logs. Interrupted
retirement fails closed. No old source, datasets, Fresh_check or payload deletion.

## Validation

New `luka_auxiliary_logits` namespace and v1 contracts, additive to donor commit
72490a9d6de4acabe9cbea518c5b100d893e6866. Test loss/head gradient isolation,
temperature and normalization, paired RNG, native extraction/parity, c50-only
reuse before/after parent retirement, data joins, exact DAG, startup failures,
poor-score completion, four full local production fits and own-target retirement.
Local synthetic tests do not certify a real SPORC execution. No submission,
commit or push is part of implementation authorization.
