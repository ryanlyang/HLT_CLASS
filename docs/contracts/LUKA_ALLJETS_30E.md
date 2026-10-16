# LUKA ALLJETS fixed30 contracts

Authority: [plan](../plans/LUKA_ALLJETS_30E_PLAN.md).

`LUKA_MATCHED_BATCH128_30E_RECIPE/v1` binds the original batch128 recipe hash,
exact30 passes, warmup3/cosine-to30, disabled early stopping and unchanged
non-schedule fields. Shared runner accepts this exact recipe only.

`LUKA_ALLJETS_FUSION_{GATE,SPEC,PREFLIGHT,PLAN,TASK,RESULTS,DIAGNOSTICS,
SUBMISSION_CLAIM}/v6` is separate from versions1--5. There is no v6 GPU PROBE.
GATE binds the original v4 gate, sacct records, every replay input, original
candidate hash, source compatibility and registered target schedule/budget.
PREFLIGHT preserves original candidate/recipe/time failure separately from
target recipe/requests, CPU admission allocation and original GPU accounting.
Technical acceptance and target-time admission are both mandatory. Technical
predictions/checkpoints never become scientific teachers.

Science has17 fresh fits,11 local TRAIN banks,30 jobs and no imported weights
or targets. Reports require exact30 validation entries and target recipe hash;
old100-pass reports are not interchangeable. Both validation summaries use the
same full-validation-selected weights. All ordinary content/parent/source hashes,
immutable publication, exact dry plan/journal and no-independent-test flags apply.

Resource reuse only changes schedule. Count cache once,30 complete measured
training+validation passes, and one measured selected-model inference/readback;
retain1.75 safety factor and reducer factor2. Host/GPU checks and debug1440min
maximum are unchanged. Missing evidence, structural mismatch, time or resource
excess fails closed; bad scientific performance never gates completion.
