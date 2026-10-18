# LUKA_COLD_GATE_* /v1

Authority: [plan](../plans/LUKA_COLD_GATE_TIER3_PLAN.md).

Distinct canonical-hashed RECIPE, ANCHOR, GATE, PREFLIGHT, SPEC, PLAN, TASK,
TRAINING_REPORT, SELECTED_OUTPUTS, RESULTS, OWNER, RETIREMENT_INTENT, RETIREMENT
and opt-in FOLLOWUP artifacts bind source, recipe, populations and parents.
Existing gated BANK/v1 numerical schema is reused on owned local scratch only.

Each TRAINING_REPORT binds a deterministic fresh initialization digest. No
student-construction API accepts teacher weights. Reports enforce all30 epochs,
no frozen backbone, full validation AUC selection and fresh optimizer. The
selected checkpoint and native/gated inference readback must match retained
predictions. Ordered identities, labels and original50k membership are saved.

TASK binds the immediate teacher's selected checkpoint and report hashes,
its completed task (or immutable U100 anchor), target bank manifest, explicit
ephemeral payload hashes, selected weights, report, predictions and costs.
Retirement is limited to a private owned scratch directory and an exact
authenticated inventory. The durable selected-output proof is validated before
deletion. A subsequent retirement receipt records deleted bytes/files. Reporting
validates this evidence without requiring erased scratch payloads; retained
checkpoints/reports/predictions still receive full checksum and metric validation.
Unrecorded absence, reappearing retired files and different consumers fail closed.

Preflight requires all12 one-pass fits,24 maximum-support stress cases, native
parity in FP32/BF16, initialization evidence, exact TRAIN/validation alignment,
per-fit resource projections and verified retirement. Poor scores never fail.
Scratch and persistent budgets are distinct; only new scratch banks are retired.
Production outputs remain below a conservative2GiB admission reserve; snapshots
are not claims about personally available filesystem-wide quota.

Both phases bind tier3 A100/6CPU/128000MiB and the exact installed environment.
Science has12 fits plus aggregate, measured fit requests no longer than24h.
Optional follow-up binds a distinct reviewed outer plan and allows only measured
times and spec identity to replace the registered ceilings; failed admission,
partial/ambiguous submission or changed source never trigger an automatic retry.

Archived reports require retained persistent data and provenance, not live Slurm
accounting or scratch. They explicitly disclose single seed, development-only
validation, no independent test and oracle versus HLT-only inputs.
