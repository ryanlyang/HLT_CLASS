# LUKA_CE_WARMSTART/v1

Scientific authority: [active plan](../plans/LUKA_CE_WARMSTART_PLAN.md).

New RECIPE, SPEC, PLAN, CLAIM, REPORT, FIT_REPORT, CHECKS, READBACK, MEASUREMENT,
TASK and RESULTS records use content hashes, schema_version=1, parent hashes
and the ALLJETS exploratory/no-independent-test flags. They are not historical
from-scratch or embedding-pretrain artifacts.

SPEC binds source snapshot, original fixed30 anchor, frozen U100 teacher,
historical OFFLINE-w5 reference and exact M0HLT initialization file reference.
The unchanged U100 TEACHER/v1 and BANK/v1 contracts retain their own validators
and hashes; importing them does not relabel or mutate them.

Each fit REPORT binds the new recipe, prepared population, initialization
SHA256 and, only for the treatment, TRAIN bank hash. It records arm, node,
counts, all 30 epoch metrics/LRs/losses/timings, selected epoch, fresh optimizer,
native-only selected state and full native warm start. FIT_REPORT also binds
the exact initialization and selected checkpoint file references and successful
native validation probability readback. Control bank parent is null and its
auxiliary loss is exactly zero; it must never consume privileged targets.

TASK receipts enumerate and hash all retained outputs and depend on exact SPEC
and setup hashes. Setup validates measured resource admission and both technical
fits before publication. Worker job IDs must match the exact submission journal
and dependencies must have successful accounting. A permanent directory claim
prevents duplicate execution. Uncertain submission failures require inspection.

Read-only RESULTS revalidates completed receipts, predictions and population
alignment, computes full-validation and original50k comparisons, and marks
uncompleted arms pending. Predictions and training reports are retained even
when the other arm fails. No result quality threshold can suppress a row.
