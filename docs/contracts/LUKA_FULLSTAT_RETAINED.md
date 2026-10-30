# FullStat retained predictor contracts v1

Authority: [active plan](../plans/LUKA_FULLSTAT_RETAINED_PLAN.md).
All artifacts use canonical content hashes, explicit parent hashes and the
inherited sealed final-test/development flags. Prior native-only checkpoints
are not interchangeable with this deployed graph.

| Kind (LUKA_FULLSTAT_RETAINED_* /v1) | Required meaning |
| --- | --- |
| RECIPE | Original FullStat20 recipe parent; CE1/E5 retained and matched CE-only graph |
| REUSE | Corrected db9afe8b readout acceptance, original U100 evidence, exact source map, file references |
| SPEC | Reuse/source parents, exact populations/nodes/resources/order/runtime and quota |
| PLAN | Two jobs, train_retained then train_control afterany; no preflight/setup |
| STARTUP | Spec/task, parity, finite gradients, real graph readback, maximum support and resource witnesses |
| EPOCH | Spec/prepared/recipe and retained-only bank parent; row plus full state file hash |
| TRAINING_REPORT | Same fit parents, all 20 history entries, combined-classifier selection, retained deployment |
| INFERENCE_COST | Spec/task, parameter/state sizes, bounded forward timing and incremental CUDA memory for retained/native graphs; saved pre-measurement peak |
| TASK | Spec/task/job/source, full output inventory, report content hash |
| RESULTS | Spec, completed model/reference rows, explicitly pending tasks, per-metric recovery |
| LAUNCH_INTENT/APPROVAL/COMPLETE/FAILURE, CLAIM | Bounded authorization, dry-plan binding and exclusive submission evidence |

The retained arm uses only the original authenticated TRAIN embedding bank;
the control's fit parents exclude that bank and its worker never loads it.
All reference/prediction comparisons require identical validation identities
and labels. Strictly load the entire selected state via
`luka_fullstat_retained.model.load_deployed(node, task, state, device)`.
Removing predictor/combined weights is an invalid checkpoint.

Only a fully validated task receipt plus exact accepted job with successful
scheduler accounting denotes completion. No receipt is emitted after failed
publication. Canonical dry review, pushed clean source and immutable exact-ID
submission journals are required. There is no implicit retry or resume.
