# Report-only batch128 recovery, execution artifacts v5

Authority: [recovery plan](../plans/LUKA_BATCH128_REPORT_RECOVERY_PLAN.md).

`LUKA_ALLJETS_FUSION_{GATE,SPEC,PREFLIGHT,PLAN,TASK,RESULTS,DIAGNOSTICS,SUBMISSION_CLAIM}/v5`
separates CPU report recovery from original v4 GPU execution. There is no v5
PROBE: all twelve original PROBE/v4 records remain in the failed job's directory.
The scientific recipe is still `LUKA_MATCHED_BATCH128_RECIPE/v1`.

GATE/v5 binds the exact failed 922bd972/21898881 source, original gate and
submission journal, terminal sampler traceback, all original payload digests,
normalized sacct job/batch accounting, reconstructed-candidate hash and a named
source compatibility manifest. Fresh roots cannot overlap old sources, raw
data, caches, preparation or any previous campaign. Current source must be
clean, committed and pushed. The original package validates its own evidence
in a CPU-only subprocess. No original file is rewritten and no particle array
or checkpoint is decoded. Original successful checkpoint/bank readback evidence
is authenticated through the twelve child records and payload hashes.

Replay requires canonical twelve phase markers and exactly twelve matching
logged successful PROBE objects, unique child PIDs, exact original runtime/job,
full population/support coverage, and no child failure. The coordinator's final
stack must show the deferred `ProcessLookupError` from `Measurement.__exit__`.
Successful child exits are inferred from the original coordinator reaching that
specific exit after processing the complete loop; the report labels this evidence.
Missing, reordered, altered or unrelated-failure evidence stops recovery.

Child memory fields must be present integers (not booleans or coerced strings).
Host high-water must be positive for every child. GPU high-water must be positive
for every stress/measure child and both fusion parity children. The pinned v4
producer intentionally performs no CUDA parity work for `parity/U000_probe` and
`parity/FINAL_DIRECT_D000_probe`: those two authenticated phases must have empty
`parities` and may report integer zero GPU allocation. Missing, negative or
malformed measurements are never treated as zero. This corrects the ea7cfdfc
replay implementation's overstrict positive-only check; it changes neither the
v5 artifact schema nor the scientific recipe or resource thresholds.

The original aggregate candidate exists only in the new report, explicitly
labelled as a reconstruction. Never insert it into the original GPU job.
PREFLIGHT/v5 binds both original execution source and recovery source, original
FAILED accounting, new CPU allocation/job, reconstructed v4 candidate, input
hashes and memory-evidence method. `job_id` identifies the new CPU report job;
`original_gpu_job_id` identifies 21898881. Keep environment/GPU measurements
from the original probes, not the CPU recovery host.

Require a one-node, one-task Slurm batch step with valid nonzero MaxRSS and the
exact original CPU/memory/account/partition/job name. Conservatively SUM batch
MaxRSS and the maximum sequential child process-tree peak. Preserve the incomplete
coordinator-sampler and sampled-not-exact caveats. GPU peak is the maximum child
peak. The original strict technical validator, 85% host / 90% GPU headroom and
100-pass * 1.75 / debug24h projection remain unchanged. Any admission error
publishes diagnostic measurement.json only, never an accepted preflight.json.

SPEC/v5 still requires the new exact reviewed gate journal and COMPLETED/0:0
accounting for the CPU report job, plus replay-validated admission. It does NOT
reinterpret the failed GPU job as successful. All 17 fits, 11 banks and 30
science tasks are fresh batch128 execution; nothing imports technical weights
or targets. V1--v4 artifacts and scientific meanings remain unchanged.

Gate: ONE CPU-only debug job, 1 CPU / 32768 MiB / 4h. Science: separate dry/live
30-job plan with measured resources. Both require exact saved plan hash and
`AUTHORIZE LUKA BATCH128 REPORT RECOVERY EXACT PLAN`. No automatic followup,
retry, cancellation, root overwrite, backend change or GPU rerun.
