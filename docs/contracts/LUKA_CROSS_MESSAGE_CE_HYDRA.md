# Cross-message CE-only controls, artifact contract v1

Authority: [plan](../plans/LUKA_CROSS_MESSAGE_CE_HYDRA_PLAN.md).
Artifacts use `LUKA_CROSS_MESSAGE_CE_HYDRA_<KIND>/v1`, schema1, canonical hashes,
explicit parent hashes and the existing ALLJETS exploratory scope flags.

`SPEC` binds exactly two seed0 CE-only tasks, original donor commit, recipe,
source snapshot, parent A6000 setup proof, environment, HLT population and bundle,
output budget2GiB and measured job minutes. No mutable parent task result is a
training dependency. Parent source/qualification artifacts must remain authentic.
`RECIPE` fixes the thirty-epoch schedule and predictor architecture with all
privileged losses zero. No teacher/student warm start is admitted.

`PLAN` and the shared exact-DAG journal bind two independent Slurm commands.
The full dry plan must precede a live exclusive submission claim. A complete
ledger can be read idempotently; active/partial claims require inspection.
Workers verify their task-specific job ID, allocation and source before running.

`RESOURCE_CHECK` binds the actual production miniature, parent acceptance,
environment/GPU, memory peaks and runtime estimate. Failed checks publish the
same diagnostic with `passed=false`; no performance score gates admission.

`EPOCH`, `TRAINING_REPORT`, `FIT` and `TASK` bind task/spec/recipe/prepared inputs,
fresh seeds, CE-only objective, all epochs, exact selected epoch, validation
predictions and HLT-only checkpoint publication/reload. Training reports have
no embedding, teacher or encoder target parent. Validation regenerates metrics
from saved predictions and verifies checksums and exact-job ownership.

`RESULTS` is read-only unless explicitly saved by a future command. Recovery
uses historical SPORC M0HLT/OFFLINE anchors. It reports full validation and the
overlapping original50k separately, with the same selected checkpoint. Optional
completed parent CE+embedding controls are authenticated before comparison;
unfinished parents are pending rather than silently replaced with partial fits.
