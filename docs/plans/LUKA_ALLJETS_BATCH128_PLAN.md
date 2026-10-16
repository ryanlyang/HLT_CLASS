# Matched ALLJETS batch-128 experiment

Execution-only amendment 2026-10-16: the
[report recovery plan](LUKA_BATCH128_REPORT_RECOVERY_PLAN.md) permits authenticated
CPU-only recovery of job21898881's twelve completed probes after its sampler
failure. New execution v5 preserves this scientific recipe; v4 is unchanged.

Authorized 2026-10-15: true batch128, accumulation1; rerun matched controls,
ordinary DIRECT/COARSE and BOTH fusion endings. This new GATE/SPEC v4 supersedes
the batch256 and imported-control/teacher clauses of LUKA_FULLSIM_ALLJETS_FUSION_PLAN
ONLY for this experiment. Earlier versions retain their original meaning.

Same 299234 TRAIN / 74895 validation, original50k secondary reporting, cuts,
units, cached seven views, full cardinality, seeds, models, BF16/FP32 losses,
optimizer/LR and 60--100-pass schedule. No new preprocessing. No independent
test (the historical holdout was already retired into TRAIN). Batch128 is a
scientific recipe change: different BatchNorm population and more updates per
pass; accumulation is NOT used to imitate batch256. Inference batches also128.

Cold-start all17 fits: M0HLT, OFFLINE, U000, DIRECT_D000, five ordinary COARSE
nodes; five adjacent FUSION nodes; FINAL_DIRECT_D000, FUSION_D000_D000,
FINAL_BRIDGE_D000. All11 scientific TRAIN banks are produced within this new
campaign. No previous weights, banks or result rows enter the comparison.
With aggregate/complete there are30 science jobs, separately hash-authorized.
The original U000 supervises both ladders only in historical batch256 runs;
in v4 the NEW batch128 U000 bank supplies DIRECT and both new ladders.

Authenticate failed isolated gate21898735 with its original5cab1830 source and
exact journal; preserve all old roots. This binds original prepared inputs and
software/GPU; it is NOT acceptance. Exact named source differences permit only
version dispatch, the batch128 recipe/kernel opt-in and new recovery policy.
The synchronous pair model, parity tolerances, physics, caches and split code
remain byte-identical. No backend or environment upgrade, truncation or tiling.

One new debug/A100/6CPU/256000MiB/8h preflight, fresh CUDA children. D000/D000
first; adjacent U050/U000, single-HLT D000, then largest single U000. Separate
parity and full128/three-update longest-TRAIN stress; validation stress is
inference-only. All stresses precede full-population technical passes. The
technical KD targets are uniform11-class probabilities, NOT old scientific
banks; the U000 probe is CE-only. Technical banks never become science teachers.
Require strict installed/native and paired FP32/BF16 parity, complete role-specific
support coverage, selected-checkpoint/TRAIN-bank roundtrips and actual timings.
Retain85%host/90%GPU headroom and conservative1.75*100-pass debug24h admission.
Failure preserves diagnostics, never admits science or silently retries.
Batch128 is a testable remedy, not a claim that the CUDA failure is solved.

Implementation does not authorize commit, push, submission or automatic followup.
Gate and science each require clean pushed source, saved dry plan and explicit
exact-hash live authorization. Poor model quality never fails or skips a row.
