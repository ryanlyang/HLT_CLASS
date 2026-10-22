# FullStat four-teacher representation prediction, v1

Scientific authority for one new Oscar experiment. This is an alternative to,
not implementation of, the relational-embedding proposal. User-approved teachers
are OFFLINE, COARSE_U100, COARSE_D066 and COARSE_D033, with coefficient 5/4 each.
The completed U100-w5 fit is reused as a verified reference; no new control fit.

## Population and model

Use the authenticated FullStat jet-row split: 3,000,000 TRAIN and 500,000
development-validation jets. Keep paired HLT/offline rows aligned. The 1,500,000
final-test and 105,325 unused rows remain sealed. Event independence is unverified.
Do not rebuild views, rebalance classes, change cuts, or use validation/test
particles for target normalization. All comparisons use the same validation rows.

Train a fresh native HLT-only ParT from the original M0HLT initialization and
sampling seeds, with physical batch 128 and the registered 20 complete epochs.
Use the FullStat20 AdamW parameter groups, per-update two-epoch warmup/cosine LR,
BF16 backbone forward on CUDA, and retain partial final batches. No warm start,
logit KD, attention KD, contrastive/relational loss, fusion or early stopping.

Each frozen validation-selected teacher uses its registered TRAIN coordinate:

| Teacher | Input coordinate | Coefficient |
| --- | --- | ---: |
| OFFLINE | OFFLINE | 1.25 |
| COARSE_U100 | U100 | 1.25 |
| COARSE_D066 | D066 | 1.25 |
| COARSE_D033 | D033 | 1.25 |

Capture each teacher's native 128-dimensional vector immediately before its
classification layer in frozen FP32 evaluation mode. Independently standardize
each coordinate with that teacher's TRAIN population mean and standard deviation
(FP64 moments, population variance, standard-deviation floor 1e-3, FP32 targets).
Do not average representations: independent teachers' axes need not align.

Reuse the original verified OFFLINE and U100 target banks, keeping their original
contracts. Export only D066 and D033. Authenticate teacher checkpoints, reports,
payload hashes, prepared-cache identity, and ordered TRAIN identities and labels.

Attach four independent Linear(128,128)-GELU-Linear(128,128) prediction heads to
the student's native vector. U100 retains its previous predictor seed; the other
heads use separately registered CPU RNG streams. Adding heads must not change
native initialization or the native RNG state.

The exact training objective is:

`CE + 1.25 * sum_t mean(SmoothL1_beta1(g_t(h_HLT), standardized_teacher_t))`.

Each mean is over batch and 128 coordinates. All four auxiliary gradients reach
the shared encoder; teachers/targets receive no gradients. Weight5 is a sum of
four coefficients, not a promise that gradient norms equal the U100-only run.
Neither real nor predicted teacher embeddings enter the classification head.
Deployment removes all four heads and strictly loads into the original native
HLT-only model. No additional inference cost or offline inputs are introduced.

## Selection and comparisons

Select among all 20 epochs by full-validation AUC using the original tie-breaks
(CE, finite mean log rejection, earlier update). Publish native selected weights,
all epoch metrics and individual/weighted auxiliary losses, predictions, receipts
and a comparison against authenticated completed M0HLT, OFFLINE and U100-w5.
This is adaptive single-seed development validation, not independent test evidence
or a fresh matched-control experiment. Poor performance never fails admission or
suppresses publication. Do not use previously pasted/hand-corrected numbers.

## Queue and admission

One approved dry-reviewed submission queues exactly `setup -> train_multi4` with
Slurm `afterok` and `kill-on-invalid-dep`. No login controller or manual second
submission is needed. Setup authenticates/exports targets, checks maximum-support
teacher taps, three maximum-support student updates, all four finite gradient
paths, separately stresses validation support, times 512 ordinary training
updates and a full validation pass, then discards the probe model. It does not
run a separate full training epoch. Scientific training always starts afresh.

Record the benchmark as an estimate, never a measured full epoch. Conservative
training estimate = 1.5 times 20 estimated train+validation epochs plus 90 minutes
for startup, authentication and publication. The reviewed cap is 1.5 times the
accepted U100 request, capped at 48 hours. Setup receives 12 hours. Both jobs use
the original Oscar L40S environment, 6 CPUs and 180000 MiB. Admission also requires
peak host RAM plus 8 GiB below 85% of allocation and peak allocated GPU memory below
90% of capacity. A failed setup/resource check blocks training; it never silently
changes epochs, batches or losses. Runtime variation can still cause timeout.

Require exact pushed clean source, immutable new root, 12 GiB personally available
durable output headroom, immutable plan review, exact authorization, exclusive
submission claim, job journal, exact successful setup accounting and receipt
revalidation in the student. No automatic retries, cancellation or old-artifact
mutation. Local tests/installed-Weaver miniature are necessary but do not replace
Oscar measurements. No final-test command exists.

See [contract](../contracts/LUKA_FULLSTAT_MULTITEACHER.md) and
[queue runbook](../LUKA_FULLSTAT_MULTITEACHER_RUNBOOK.md).
