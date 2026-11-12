# Oscar six-teacher subclass-off ablation

This is the active scientific plan for `luka_fullstat_six_teacher_s0`.
The user authorized one fresh Oscar fit, with the existing six-teacher run
as its control. No separate preflight, new control, or teacher training.

## Only scientific change

Use the exact recipe of six-teacher job8848995, source
`2fb09f38a3a1da85cdceff9df311963fdb3aa872`, with subclass coefficient zero.
Keep CE1, OFFLINE/CONCAT/FUSION embeddings10/3 each, hybrid UE100 and previous
split-head DE066/DE033 embeddings5r/3 each, six auxiliary logit objectives r/6
each (T2, T-squared once), and pure-OFFLINE contrastive0.3r. Here r is the
original three-epoch update ramp. Do not include original U100 as a target.

Train from scratch on the identical authenticated3M TRAIN/500k development
validation population,20 epochs, global batch128, original optimizer, LR,
initialization and sampling seeds, precision, candidate sampler and1024 negatives.
Select by the original validation-AUC rule. No warm start from the control.

Keep the subclass head, its isolated initialization, raw diagnostic loss and
frozen targets, but multiply its loss by exactly zero. This preserves model
construction, optimizer parameter groups and stochastic execution. It supplies
no optimization gradient to the encoder or classifier. All auxiliary heads,
including the inactive subclass head, are removed from deployed HLT inference.
The zero-weight head can still undergo AdamW decay; it cannot feed the main
classifier. Do not change other auxiliary objectives or introduce gradient surgery.

## Reuse and execution

Authenticate the completed control in its original clean source, including
COMPLETED/0:0 accounting for job8848995, published reports, epoch states, frozen
banks and saved prediction population. Reuse its exact six embeddings, all six
logit banks, OFFLINE keys and subclass targets. Copy completed CONCAT/FUSION
logit exports byte-for-byte into the new output root, preserving exporter
source, manifest identity and payload checksums; never rerun their forwards.
The control's trained student is a reference only, never an initializer.

Register new `LUKA_FULLSTAT_SIX_TEACHER_S0_* /v1` artifacts (without the space).
The completed six-teacher and all donor contracts/implementations stay unchanged.
The new spec binds its own pushed source, original control, identical frozen
supervision, recipe, resources and exclusive submission records. A new namespace
replaces only orchestration and coefficient/acceptance handling; shared numerical
implementation is imported unchanged wherever possible.

Request one Oscar L40S,6 CPUs,192000MiB,48 hours, with12GiB available result
quota. Preserve current-address-space worker memory accounting and the existing
headroom. Integrated real-allocation parity, zero-subclass-gradient checks,
full-objective stress and timing happen before scientific training. This user-
authorized workflow replaces the generic separate Tigris miniature gate for
this Oscar experiment; local fixtures do not certify remote production.

The detached launcher performs authentication, saved dry review and exactly
one authorized submission after logout. No existing jobs are cancelled.
Recovery is explicit, same-source/same-recipe, fresh-root, exact terminal job
accounting, with complete epoch/optimizer/RNG continuation and no implicit retry.
Completed epochs and selected predictions are content authenticated and atomic.
Poor finite scientific metrics never reject the run; final-test access stays sealed.

## Evaluation and evidence

Compare against the completed six-teacher subclass0.5 control, plus historical
Multi3+contrastive+subclass, Multi3, U100-w5, M0HLT and OFFLINE. Report selected
accuracy/AUC, recovery and R50/R30 on the same validation population. Maintain
per-epoch weighted loss totals and gradient diagnostics, with subclass weighted
norm exactly zero and undefined weighted cosine. Single-batch gradient snapshots
are diagnostics, not evidence of a guaranteed validation improvement.

Required local checks: unchanged non-subclass recipe/seeds, zero-gradient
isolation, CPU/CUDA losses, identical deployed graph, frozen-bank byte reuse,
source/metadata corruption rejection,20-epoch continuation/publication, strict
one-job detached submission, and installed-Weaver startup. Record actual outcomes
and remaining real-Oscar acceptance in the validation record; do not submit here.
