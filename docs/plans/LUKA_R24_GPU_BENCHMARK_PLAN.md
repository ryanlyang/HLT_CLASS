# R24 reconstruction: two, three and four GTX 1080 Ti GPUs

The user authorized this bounded hardware benchmark on 2026-11-12. It does
not authorize a new scientific sweep or changes to existing jobs. Scientific
targets, decoder and schedules remain those of the
[R8 follow-up](LUKA_DIRECT_OFFLINE_RECONSTRUCTION_FOLLOWUP_PLAN.md).

One Hydra allocation requests four GTX1080Ti devices on one node, twelve CPUs,
64000MiB and two hours. Within it, separate processes run a one-card FP32
numerical reference on small actual-target batches, then two-, three- and
four-card measurements. Unused cards remain reserved during smaller phases;
active-card projections are not the bill for this four-card benchmark job.
No node is pinned. Device availability is established by Slurm at submission.

All scientific code and original data validation are imported from the clean
58d0c0edf5f41cff2398fb168fc16fc92e169294 worktree. The new benchmark package
is independently source-pinned. Reuse its completed setup/targets and completed
R24 report; never edit or regenerate them. Registration authenticates the
full saved R24 fit, setup, source and environment. A6000 comparisons report
observed thirty-epoch training/validation timings and, when available, Slurm
elapsed time. They compare practical FP32 Pascal against BF16-native/FP32-decoder
A6000 execution, not equal-precision hardware speed.

Native model, U100-w5 and the complete four-block FP32 decoder are replicated
per GPU. Global batch128 splits64/64,43/43/42 or32/32/32/32; global padding is
preserved. Native training uses SyncBatchNorm. Decoder chunks retain size8,
exact SciPy matching and bounded VJP backward. Local mean gradients are scaled
by local_rows/global_rows, then summed explicitly in bounded all-reduce
buckets after both backward paths. This deliberately avoids DDP reducer hooks
around the decoder's multiple backward calls. Gradients, buffers, AdamW moments
and parameters are compared with a single-card reference. FP32/TF32-off is
explicit. Normal dropout uses rank-local seeds; stochastic trajectory equality
with A6000 is not asserted. Solver threads per rank are12/world (six, four,
three); no rank silently creates six threads when four ranks share twelve CPUs.

Parity uses real15/16-row batches, dropout disabled and schedule positions3,
4.5 and7 (zero, partial and full reconstruction weights). Mismatches are saved
as unsuccessful numerical checks. Nonfinite inputs/gradients and corrupt
lineage fail closed. No accuracy threshold or science release exists.

Preparation publishes identical padded TRAIN batches spanning the epoch-one
permutation, U100 targets and complete OFFLINE particle targets. K remains the
original TRAIN maximum (298 in the observed campaign). Include the actual
final TRAIN batch, eleven rows, longest HLT support and largest OFFLINE counts.
No labels or OFFLINE metadata enter native forward. Validation has native HLT
inputs only. Thirty-two spaced validation batches estimate full validation
inference cost; no AUC, checkpoint selection, final test or teacher fitting.

Each multi-card phase runs three warmup and twenty-four timed updates at
position7 with ordinary dropout; timing includes local slicing/transfers,
exact matching, both backward paths, gradient synchronization and AdamW.
It measures a production diagnostic pass separately, validation inference,
initialization and longest-support memory. Sample input construction time from
real caches is recorded separately because timed updates use prepared arrays.
Projections include training, estimated input construction, once-per-epoch
diagnostics, validation and initialization/preparation. They exclude queue wait,
metric calculation, production checkpoint I/O and additional source audits;
they cannot directly authorize production walltimes. Full phase sample lists,
GPU peaks, aggregate rank RSS and actual allocation duration remain visible.
Preparation includes publishing the benchmark samples, so that startup term is
conservative for a future production worker. Diagnostic shuffles are local to
each rank; their timing does not establish equivalence to the original
global-shuffle scientific diagnostic. The A6000 training-speed comparison
includes estimated input construction and diagnostics on the Pascal side.

The report compares active-GPU-hours and idealized fits/day for a fixed GPU
budget, explicitly assuming no additional contention. OOM/failed phases remain
visible and later phases are still attempted in fresh process groups. No blind
retry or automatic scientific release is permitted. A separate production
profile and matched FP32 control remain necessary if Pascal is selected.

Local tests establish queue readiness only. Real multi-GPU Hydra checks are
the authoritative remote benchmark for this task, superseding the generic
legacy Tigris execution requirement for this bounded Hydra-only measurement.
