# U100 particle/pair corrections on Brown CS Hydra

This additive execution port implements the user's request to move the four
fits in [the scientific plan](LUKA_U100_PARTICLE_PAIR_CORRECTION_PLAN.md) to Brown
CS Hydra. It does not change that experiment's losses, models, data, seeds,
30-epoch schedule, selection, or HLT-only inference. No new reference fit or
teacher fit is authorized. Existing SPORC and ALBERT jobs remain untouched.

The user verified the 2,287-file, 2.913 GiB export on Brown. Archive SHA256:
`2fbb97036df805967c651b9a15a4a14f89ef3a9a75fab1f9e96f1819ecc986f5`.
Manifest: `b87e76450441c78892af092a1cb67d6abeb23963b4c983eaf756cd4c4535116c`.
These are the only admitted inputs. Original manifests remain byte-identical;
old absolute paths are provenance, never runtime paths on Brown. Missing files
fail without raw-data reconstruction or new target inference.

Hydra uses `gpus`, account `default`, job QoS `normal`, one `nvidia_l40` GPU,
six CPUs and 120000 MiB per job. This is a site change from SPORC/debug, not an
assumption that Hydra has a debug partition. Two fit lanes preserve C/A/D/B
ordering, with `afterany` ordering dependencies. One shared setup precedes the
four fits and report. No separate preflight job. Setup measures all four real
production miniatures and support extremes before releasing fits, as before.

Allocation checks compare the running controller job and available batch
environment fields. An unspecified controller task count requires an explicit
batch task count; known conflicting counts always fail. Missing cluster or
memory environment exports require controller configuration/allocation evidence.
The literal cluster export `(null)`, observed in Hydra job 131385, is also
unspecified: query the controller and require `ClusterName=hydra`. Never replace
an unknown name with the expected one or accept a concrete different name.
The same one-node, one-task resource request remains mandatory. Whitelisted
allocation details are logged before admission so a later controller purge
does not erase the evidence. This is metadata compatibility, not a change to
the registered resources or scientific experiment.

Create a separate Python 3.10.21 environment with the recorded eight packages,
PyTorch 2.5.1+cu118, cuDNN90100, and exact installed Weaver Python-source hashes.
Do not modify system Python or unrelated environments. Environment equality is
required, including on workers. The new execution namespace owns site/source
validation; original scientific modules are imported unchanged.

For the original execution v1, reload the transferred native control checkpoint and evaluate the
same 74,895 validation rows in batch128 using the original BF16 prediction
routine. Require exact row/label order, finite normalized probabilities,
maximum absolute probability deviation <=0.005, RMS deviation <=0.0002 and
absolute macro-AUC difference <=0.00005 against the saved SPORC predictions.
These fixed, symmetric numerical portability tolerances are not scientific
quality thresholds or evidence of bitwise equality. Record deviations before
checking them. Failure requires diagnosis; never silently widen tolerances or
rebase the control. The explicitly authorized v2 policy below supersedes this
cross-site check for new campaigns; it does not reinterpret old artifacts.
Native geometry, zero-correction forward and deployment
reload checks remain part of genuine setup.

The extracted bundle is shared immutable input, with an atomic completion
receipt. Its archive, manifest and each file are authenticated. Campaign outputs
live separately and reserve eight GiB plus one GiB filesystem headroom; extraction
and the environment consume additional space. No automatic deletion or retries.
Final-test access remains sealed; original50k overlaps validation and is report-only.

Queue only clean pushed source after a full dry review. The detached dispatcher
may submit the four fits automatically within the reviewed resource envelope
after successful setup. It may reduce measured walltimes, not alter the science
or expand resources. All submission and worker claims bind exact job IDs.

Implementation status and measured evidence belong in HANDOFF. Local tests do
not establish Hydra feasibility; actual environment creation and shared setup
must succeed on Brown.

## Authorized inference-only precision diagnostic (2026-11-02)

After setup 131410 failed the original probability tolerances while its AUC
difference was only 0.0000018135, the user requested a Hydra check. Run one
separate diagnostic job using the existing bundle, environment and native
matched-control checkpoint. No SPORC GPU, TRAIN arrays, offline arrays, teacher
fit or reference retraining is needed. This does not rerun shared setup.

Use Hydra `gpus/default/normal`, one L40, six CPUs, 120000 MiB and a one-hour
walltime. Numerical libraries use one thread. Reserve 256 MiB for outputs plus
one GiB filesystem headroom. This is a bounded inference request, not measured
evidence for production fits. Use a fresh output namespace and exact accepted
job ID. A printed and saved one-job dry plan precedes submission; invoking the
dedicated queue helper authorizes only this diagnostic. No detached dispatcher
or follow-on jobs are involved.

First replay all 74,895 validation jets in original batch128 BF16 order and
padding. Save the new probabilities separately; compare to the immutable saved
SPORC probabilities. Report macro and per-class AUC, accuracy, CE, probability
differences, top-one disagreements, and QCD rejection at 30% and 50% signal
efficiency, including passing counts and zero-count censoring. Each probability
set uses its own signal threshold under the same original metric convention.

Then reuse four evenly spaced complete original validation batches (starting
at rows 0, 24832, 49792, 74752) and the final 15-row partial batch starting at
74880. These 527 rows are fixed without examining numerical differences or
labels. Preserve their original batch composition and dynamic padding, rather
than batching the selected rows together. Replay the same checkpoint with:

1. GPU BF16 using the original default backend, including comparison against
   its own full-validation predictions for these exact batches.
2. GPU BF16 with math SDPA, TF32 disabled and reduced-precision BF16 reductions
   disabled.
3. GPU FP32 under those same math/TF32 settings.
4. CPU FP32 under the same requested math settings.

Record the actual backend flags, original batch starts/lengths/padded lengths,
row identities, timings, probabilities and pairwise differences. Restore every
modified backend setting even on error. Evaluation mode and inference mode
prevent optimization or checkpoint updates. Save full-validation evidence
before the smaller precision probes so a later error preserves useful output.

Finite discrepancies, even outside the old limits, are diagnostic results and
do not fail this job. Invalid inputs, misaligned identities, nonfinite outputs,
environment/source/allocation mismatches still fail. Original science admission
tolerances were unchanged by the diagnostic implementation. Diagnostic completion publishes neither ACCEPTANCE
nor a setup receipt, cannot release training, and cannot be substituted for
setup evidence. Its artifacts explicitly state `admission=false`.

CPU/L40 FP32 agreement would establish local consistency, not A100 FP32 parity.
The BF16/math comparison changes both attention backend and reduction settings;
it cannot alone attribute differences to one kernel. No sampled AUC or recovery
claim is made on the 527-row probe. Full-population agreement and rejection
counts inform a later explicit portability decision; they are not independent
test evidence. See [the diagnostic runbook](../LUKA_U100_PAIR_HYDRA_PRECISION_RUNBOOK.md).

## Authorized Hydra execution v2 after diagnostic 131484 (2026-11-03)

The user supplied successful results from precision job 131484, source
`ad55d3cd32d7826d6e16a475228fece3c6099da4`, and requested the full campaign after
discussion of those results. This authorizes the following explicit execution
policy change; four-fit science, precision, data and reference weights remain
unchanged. It requires no SPORC GPU or additional diagnostic job.

Authenticate that exact completed diagnostic in its original clean source.
Pin its spec, report, receipt, full and probe predictions, submission evidence,
checkpoint/reference hashes, environment, GPU and transfer identity in the new
SPEC/v2. Preserve all original files. Recompute metrics and probe differences
from the saved arrays, including exact validation identities, labels, fixed
batch indices and original CPU/GPU backend flags. The production worker must
use the diagnostic's same L40 identity, environment and backend configuration.

The v2 numerical qualification requires:

- CPU/GPU FP32 math probability differences <=1e-5 maximum and <=1e-6 RMS on
  the fixed 527-row probe. These are local consistency bounds, not a claim of
  A100 FP32 agreement. Observed values were 1.3113e-6 and 5.4447e-8.
- Exactly repeated BF16 probabilities for the same original batches. Observed
  deviation was zero.
- The original symmetric full-validation macro-AUC difference bound <=5e-5
  for this same saved checkpoint. Observed difference was 1.8135e-6.
- Finite normalized outputs and unchanged authenticated population/checkpoint
  lineage. Corruption, stale evidence, reordered rows or missing modes fail.

Cross-site BF16 maximum/RMS differences are now **report-only** under v2; the
original v1 constants remain unchanged. This is justified by the local backend
probe exceeding those old bounds while FP32 agrees tightly. Accuracy, rejection
and CE changes remain visible, with no new quality gate. No student's scientific
performance is used for job admission or cancellation.

Reuse the completed full Hydra BF16 control probabilities. Do not rerun the
reference fit or control inference. Add `U100_W5_HYDRA` to reports (validation
AUC 0.96209461), keep the saved SPORC matched row (0.96209280), and measure the
four students' primary AUC deltas against the Hydra replay. Recovery continues
to use explicitly labelled historical SPORC M0HLT/OFFLINE anchors. Original50k
is sliced from these same saved arrays and remains report-only, not independent.
The control was trained on SPORC: replaying its inference on Hydra does not
establish equivalence of training trajectories across GPUs or create a newly
matched Hydra-trained control. Preserve that limitation when interpreting gains.

Create a fresh campaign root and SPEC/v2 with an explicit PORTABILITY_POLICY/v2
and imported PORTABILITY_EVIDENCE/v2. Shared setup verifies the qualified
population/GPU/backend, records PORTABILITY/v2, reuses U100 embeddings, exports
the still-required particle targets, and runs **all four original production
miniatures and support/resource checks**. The diagnostic is not resource
acceptance. Only then may the reviewed dispatcher release the four 30-epoch
fits in C/A/D/B order across two L40 lanes, followed by the report. Use the
distinct V2 submission authorization and the existing full dry-run/resource
envelope. Preserve previous failed roots, and do not resubmit their job IDs.

See [v2 commit and queue instructions](../LUKA_U100_PAIR_HYDRA_V2_RUNBOOK.md).
