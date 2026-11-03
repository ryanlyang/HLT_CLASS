# U100 identification: single-lane continuation with validation patience

Date: 2026-11-02. Scientific amendment authorized in this chat: preserve the
twenty original configurations, prioritize the remaining fits, use one active
debug fit, and stop unproductive continuation successfully using patience.
This additive plan supersedes the original plan's fixed30/no-early-stop rule
**only for this new execution**. Existing campaign specs and historical results
remain unchanged. This is exploratory, adaptively prioritized development work.

## Evidence and scope

Original successful execution: source
`431341db5eced8eaa3e52b37ba047f8cd4beed50`, root
`/home/ryreu/atlas/HLT_Classification/checkpoints/luka_u100_contrastive_431341db_debug_r1`.
Setup21993008 completed. User-provided histories show control38.6%, mixed0.3
44.4%, random0.3 42.9%, mixed1 39.4%, and provisional kinmix0.3 45.6% AUC-gap
recovery. Kinmix's lead over mixed0.3 is only0.0000464 AUC. These are single-seed,
small-data development results, not evidence of equivalent high-data gains.

Every new fit retains CE1 + U100 embedding5 + its original identification
coefficient, the same targets, population, initialization, batch ordering,
candidate sampler, independent diagnostic RNG, three-epoch matching ramp,
optimizer, precision and warmup3/cosine30 schedule. Inference remains native
HLT only. No new teachers, weights, samplers or configurations are introduced.
In particular, the suggested kinmix0.1 is **not** silently added to this grid.

## Common stopping policy

Track full-validation macro OVR AUC from epoch1. An improvement of at least
0.00001 above the last meaningful best resets patience. Smaller improvements
accumulate relative to that meaningful best; they do not reset it individually.
Do not terminate before completing epoch15. At each subsequent epoch boundary,
stop after five consecutive epochs without meaningful improvement. Maximum30.
Keep the original30-epoch learning-rate function, including fractional update
positions; do not compress its schedule when a fit stops early.

Checkpoint selection is independent: use the original selection key on every
completed validation pass, saving every actual improved native checkpoint,
even a gain below the patience delta. Save immutable `fit/best/epoch_NN.pt`
and hash-bound metadata immediately after validation, before TRAIN diagnostics.
Restore the selected native model, perform the original prediction/reload checks,
and publish the report and successful receipt on a patience stop. Persist actual
epochs, maximum epochs, policy, first stopping decision, selected epoch and all
observed history. A failed/interrupted fit's best files are preserved but are
not misrepresented as a completed fit or optimizer-resume checkpoint.

This is an authorized successful stopping policy, not a scientific-quality
failure gate. Finite poor models also receive a completed result. No row is
cancelled or skipped for its score; nonfinite execution and corrupt evidence
still fail closed. Negative recovery alone never stops a fit. The policy is
identical for experimental fits and new seed1 control. Historical30 fits are
explicitly labelled in the combined report, not retroactively called early stops.

## Remaining priority order

| Priority | Arm |
| --- | --- |
| 1 | `mix_l0p1_s0` |
| 2 | `hard_l0p1_s0` |
| 3 | `cross_l0p3_s0` |
| 4 | `mix_l0p3_s1` |
| 5 | `u100_w5_s1` |
| 6 | `random_l0p1_s0` |
| 7 | `hard_l0p3_s0` |
| 8 | `cross_l0p1_s0` |
| 9 | `mix_l0p03_s0` |
| 10 | `random_l1_s0` |
| 11 | `cross_l1_s0` |
| 12 | `hard_l1_s0` |
| 13 | `random_l3_s0` |
| 14 | `mix_l3_s0` |
| 15 | `mix_l10_s0` |

The original five already attempted arms are retained in the registry as
fallbacks, in order kinmix0.3, mixed0.3, controlS0, random0.3, mixed1. Authenticate
and reuse their complete results. If any original attempt is terminal without
a completed receipt, the reviewed transition permits a fresh attempt in the
new root, preserving every original byte. Never resume from an unvalidated
partial file. A COMPLETED job with missing receipt is inconsistent and stops
the transition for investigation, rather than being silently retrained.

## Transition and execution

The original code publishes selected weights only after training. Therefore
never cancel a running original student. Review the original source/spec,
successful setup, exact accepted job ledger, target payloads, completed fits
and terminal accounting. Review the prospective single-lane graph and exact
cancellation IDs before the detached dispatcher starts.

The dispatcher uses `scancel --me --state=PENDING` with **only explicit IDs from
the authenticated original campaign ledger**, including its report. Slurm's
state filter protects a job that starts during the transition race. It waits
for any original running allocations to finish and for exact terminal accounting,
then replays original authentication to reuse newly completed fits. The final
graph is the reviewed order minus authenticated completions. All other science
rows remain. Both old controls/results and partial outputs are untouched.

Queue all remaining fits plus report with a **single afterany chain**. Only one
fit in this campaign can run at a time; a failed fit still releases its successor.
The report records failures/incomplete rows. No further submit-host process is
needed after all jobs are accepted. The detached dispatcher can survive logout
while waiting; it is not guaranteed to survive a submit-host reboot. Journals
and exclusive claims prohibit blind retries or duplicate submission.

No new setup, target export or preflight is queued. Reuse the authenticated
successful six-family A100 miniatures and measured full30 resource upper bounds.
Updates/model/input/sampler are unchanged; local production tests establish
prefix parity for the added stopping loop. Keep debug/A100, six CPUs,120000MiB,
`reu-aisocial`, `qos_tier3`, and exact environment/GPU checks. Each fit retains
its original conservative full30 walltime request. One active fit reduces
concurrent allocation demand; early stopping does not authorize tighter walltime.

Reserve8GiB persistent storage plus1GiB free filesystem headroom for independent
new results and durable improvement checkpoints. Targets remain external hash-bound
references. This allowance is separate from the original campaign and other
ongoing work; no filesystem cleanup occurs.

## Reporting and validation

Combined results preserve original30 and patience5 labels, actual epoch counts,
selected epochs and stop reasons. All model probabilities use the same full
74895-jet validation join. Original50k overlaps and is report-only. Final test
remains sealed. Matched-seed deltas are descriptive; historical controls retain
their original stopping budget, and no identical-budget replication is claimed.

Test policy edge cases, tiny and cumulative gains, warmup, first stop/max cap,
finite poor outcomes, CPU/CUDA installed-Weaver batch/update parity, native
reload, durable files before interruption, corrupted lineage, explicit-ID
cancellation, pending-to-running races, completion during transition, one-lane
dependencies, failed predecessors, and duplicate/partial submissions. Existing
contrastive runtime and its tests must remain unchanged.

Implementation: [runtime](../../src/hlt_classification/luka_u100_contrastive_serial/),
[contract](../contracts/LUKA_U100_CONTRASTIVE_SERIAL.md),
[runbook](../LUKA_U100_CONTRASTIVE_SERIAL_RUNBOOK.md).
