# Six-teacher subclass-off contract v1

Authority: [active plan](../plans/LUKA_FULLSTAT_SIX_TEACHER_S0_PLAN.md).
New scientific/execution artifacts use `LUKA_FULLSTAT_SIX_TEACHER_S0_<KIND>/v1`.
Every artifact has canonical content and parent hashes, development-validation
scope, sealed final test, and no claimed event independence or old-trial disjointness.

`BASELINE` pins the exact original2fb09f38 spec, clean source snapshot,
COMPLETED/0:0 job8848995, selected report/checkpoint/predictions/task receipt,
and original CONCAT/FUSION export inventories. Original-source replay validates
completion using the historical worker before reuse. The selected control
weights are comparison artifacts only and never initialize the new student.

`RECIPE` is explicitly distinct from six-teacher/v1. It keeps all numerical
settings except subclass coefficient zero: CE1, embedding15 across the same
six teachers, KD1 at T2, contrastive0.3, original ramps/seeds,3M/500k,20 epochs.
Nested `SUBCLASS_OBJECTIVE` binds the old subclass definition and coefficient0.
The frozen subclass bank retains its original identity and bytes; it is not
regenerated or given a different scientific meaning.

`SPEC` binds source, BASELINE, original supervision reuse, resources, one native
HLT task and optional explicit same-source recovery. Parent source and immutable
references must remain present. Dedicated new output root cannot overlap any
protected input/source/control/recovery root. Minimum free result allowance is
12GiB; resources are one L40S/6CPU/192000MiB/48h. No teacher exports, teacher fits,
new control or standalone preflight jobs are registered.

The model imports the exact original six-teacher Student and initialization.
The zero coefficient is enforced by the new objective boundary; all other loss
arithmetic is the original implementation. The raw subclass forward/loss remain
available for diagnostics, and the subclass gradient from the total objective
must be zero. Optimizer groups and head initialization stay matched. Auxiliary
predictions and teacher inputs never enter the deployed CE classifier.

CONCAT/FUSION `KD_BANK` and payload bytes are copied from the completed control.
Validate them with the original schema and exporter lineage, including the
control's export-recovery chain. Exact role, temperature, source/model/view,
prepared population, ordered identities/labels and payload hashes remain
mandatory. Corruption fails; the implementation has no re-export fallback.
All other embedding/KD targets and OFFLINE matching keys use unchanged readers.

`STARTUP` requires the original installed-Weaver/sampler/CE-only-classifier/
deployment checks and measured full-objective workload, plus
`subclass_gradient_zero=true`. Active auxiliary heads remain connected. The
post-exec worker-memory policy is imported unchanged. Finite poor metrics do
not reject training; failed identity, numerical or resource checks do.

`EPOCH` binds full optimizer/model/RNG state and validation/loss history to all
target identities and the new recipe. `weighted_subclass` and its gradient norm
must equal zero; weighted gradient cosine is undefined. Diagnostics remain
first-batch native encoder+classifier measurements at epochs1/3/10/20.
`TRAINING_REPORT` requires20 contiguous epochs and original deterministic
best-AUC selection. `TASK` binds the accepted exact job, published native
model/predictions/report and complete epoch inventory. Reporting R30/R50 uses
the same AUC-selected checkpoint, without class-based reselection.

`RESUME` permits only same-source, same-recipe, same-baseline continuation after
an authenticated terminal unsuccessful job. Preserve original evidence; use
a fresh root and exclusive claim. Complete epochs resume with optimizer/RNG;
twenty completed epochs permit publication-only recovery. Target copies always
come from the same authenticated control. Changed-source or completed-job
recovery, implicit retries and ambiguous resubmission are forbidden.

`PLAN` and `LAUNCH_*` register one job with a saved dry review, explicit bounded
authorization, durable append-only journal and detached launcher. DETACHED
only establishes that the launcher started; LAUNCH_COMPLETE plus EXIT0 proves
submission. New source is pinned to a pushed clean descendant of 27a83412 with
only this namespace/entrypoints changed in runtime directories. Historical
contracts, workers, numerical kernels and results remain unchanged.
