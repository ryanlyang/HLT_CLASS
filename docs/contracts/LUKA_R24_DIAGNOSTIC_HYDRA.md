# R24 diagnostic Hydra contracts v1

Authority: [active scientific plan](../plans/LUKA_R24_DIAGNOSTIC_HYDRA_PLAN.md).
Artifacts use `LUKA_R24_DIAGNOSTIC_<KIND>/v1`, schema version 1, canonical
content hashes, explicit parent hashes and the existing exploratory-data flags.
They cannot be reused as curriculum or original R24 artifacts.

`RECIPE` fixes fourteen cells, their order, paired controls, worker, native
precision, independent seed offsets, global batch 128, 30 epochs, U100 weight5
and original R24 delay3/ramp3/.3. `SPEC` binds recipe, exact pushed clean source,
original 58d0c0ed spec/source, completed 1094dea4 benchmark and receipt, installed
environment, full TRAIN/validation identities, U100 bank and target support.
The new-worker implementation is migrated from a549fef1; it is not imported
from a mutable curriculum checkout. No existing artifact is overwritten.

`INITIALIZATION` records native/predictor/decoder tensor hashes, independent
seeds, rank dropout seeds and the full first-epoch training-order hash. The
hash operation consumes no RNG. Offsets apply to exactly one source at a time.
The predictor and decoder use their original forked initialization, including
in the PI arms. Original-worker fits keep their donor initialization semantics.

`EPOCH`, `TRAINING_REPORT`, `DEPLOYMENT` and `FIT` bind new-worker records to
the spec, recipe and original parent. Validation runs on native HLT inputs
only at the arm's registered precision. Selection uses the donor's complete
AUC/CE/rejection/earliest-update rule. Native checkpoints must reload and
reproduce selected validation probabilities and contain no auxiliary tensors.
`ORIGINAL_FIT` wraps the unchanged donor report/receipt in a diagnostic parent
binding without relabeling its scientific schema. Its validation uses the
unchanged donor validator. Task receipts bind every output to the accepted job.

`QUALIFICATION` on Pascal requires 12 successive updates at positions
1,2,2.9,3,3.01,3.5,4,4.5,5,5.9,6,7 for both control and R24, alternating
16- and 15-row global batches. The reference is serial FP32; the comparison
uses four real CUDA ranks, global BatchNorm/padding and weighted gradient SUM.
Dropout is disabled for this diagnostic only. Every step must satisfy:

| Quantity | Maximum difference |
|---|---:|
| Scalar objective, absolute | 1e-3 |
| Gradient, relative L2 | 5e-4 |
| Parameters, relative L2 | 5e-4 |
| AdamW first/second moments, relative L2 | 5e-4 |
| Buffers, maximum absolute | 1e-4 |

Reference snapshots are hashed and retained under setup. Failure preserves
the numerical evidence and prevents that lane's release. A6000 qualification
checks the new BF16 native route against the original forward on the same
model/input exactly. Both lanes replay the authenticated historical U100
checkpoint over full development validation in FP32, with fixed AUC replay
tolerance 5e-5. These are numerical checks, not scientific quality thresholds.

Each `RESOURCES` record covers every fit in its lane using the actual worker:
production miniatures and reload, full-batch longest HLT and OFFLINE support,
partial11, the real tail batch, two warm updates and twelve full timed updates
at active reconstruction weight, full validation including metrics, and
aggregate peak host/per-device allocated memory. Original miniatures use the
donor's one-epoch technical path; new miniatures use two epochs at positions
7 and 20. Technical state is discarded. Time requests include 30 full epochs,
validation, probes, loading, startup/reload/export allowance, a 2x margin and
30 minutes overhead. The hard cap is 48h. Host limit is 85% of requested memory;
GPU limit is 88% of each device's memory. Invalid/nonfinite evidence fails closed.

`PLAN`, submission journals and ledger use exact task-bound job IDs. Bootstrap
contains two independent setup/afterok-release pairs. Each release can submit
only its lane's preauthorized commands; only measured walltime may shrink.
Pascal science requests 4 typed GTX1080 Ti, 12 CPUs and 64000 MiB; A6000 science
requests 1 typed A6000, 6 CPUs and 32000 MiB. Report jobs use afterany; no fit
waits for another fit. CPU requests are 1 CPU/4000 MiB, allowing verified Slurm
allocation rounding to 2 CPUs. Controller reads retry three times; scientific
work and ambiguous submissions are never retried automatically.

`RESULTS` validates completed fits; `PROGRESS` reads only saved science epoch
records. Both label matched controls, precision/worker and provisional rows.
Missing controls remain missing. Neither performs inference, alters jobs or
substitutes an historical control into a fresh paired comparison.
