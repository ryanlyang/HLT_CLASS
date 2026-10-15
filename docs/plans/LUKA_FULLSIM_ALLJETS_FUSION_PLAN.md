# Luka all-eligible fusion continuation, both single-HLT endings

Authorized 2026-10-15. Add a separate read-only-parent continuation to the
completed ALLJETS regular campaign. This plan supersedes the old fusion plan's
100k/50k population, raw-view rebuilding and fresh control fits, only for this
new namespace. Preserve the parent, its clean worktree, journals and every byte.

Use exactly 299234 TRAIN / 74895 validation, original-50k secondary reporting,
natural proportions, original cuts, provisional mm/error-only zero masking,
seven cached views, full support512 and no truncation. The historical test is
already retired into TRAIN; there is NO independent test. Reused validation and
single-seed comparisons are exploratory; cross-file event independence unverified.

Authenticate the completed parent with its ORIGINAL pinned source in a separate
Python process. Require ALLJETS SPEC/v1 and exact counts, split, recipe, all nine
committed fits, five banks and completion. Authenticate old/new source snapshots;
all old tracked src bytes must be identical, with only the five named fusion
production modules plus the explicitly diagnostic-only `alljets_fusion_diagnostic`
module added. Freeze parent receipt hashes and both-population result rows.
Reuse existing controls, DIRECT and regular COARSE as comparison rows and the
scientific U000 TRAIN probability bank as the first teacher. Never import
initial model weights or a technical probe bank. Do not refit controls.

New cold-start nodes (context first in this description):

U000 -> FUSION_U050(U000,U050) -> FUSION_U100(U050,U100)
-> FUSION_D066(U100,D066) -> FUSION_D033(D066,D033)
-> FUSION_D000(D033,D000), then BOTH:

- FINAL_DIRECT_D000: single D000 student of FUSION_D000;
- FUSION_D000_D000: two encoders of the SAME D000, then FINAL_BRIDGE_D000,
  a single D000 student of this bridge.

Eight fresh fits, six TRAIN-only reducers, aggregate and complete = 16 new
science jobs. U000's old reducer is an authenticated external dependency, NOT a
Slurm dependency on an aged-out ID. Retain the previous fusion graph's seeds,
context-to-primary blocks2/4/6/8, alpha=1, lossless pair saved-tensor CPU offload,
batch256/BF16, FP32 C25/P75 T=2 KL, and unchanged 60--100-pass recipe. Final single
HLT endpoints have identical initialization/sampler seeds to the original M0HLT.
Full validation selects; original50k reports the same checkpoint. Save validation
predictions for all new fits. Label offline-containing fusion as oracle and
D000/D000 as two-encoder HLT. Never call either a single-HLT deployed endpoint.

Load authenticated durable inputs, at most two coordinates per role. D000/D000
aliases the same cache. No ROOT reconstruction or new data upload/preparation.
One NEW real SPORC debug GPU gate: 6 CPUs, 256000 MiB, A100, 8h. Old single-encoder
acceptance is not fusion resource evidence. Probes cover worst-support paired
fusion, identical-input HLT fusion and single-HLT, including separate longest
TRAIN backward and longest validation inference-only stress, batch256/3 steps;
installed-Weaver/native masks and FP32/BF16 offload parity; full-population
technical passes and checkpoint/TRAIN-bank readback. The imported U000 TRAIN
bank supplies technical KD targets; technical outputs never feed science.
Prove every science view's support is bounded by measured families separately
by role. Measure full cached load/inference/pass times; unchanged conservative
100-pass projection must fit debug24h. Otherwise preserve diagnostics and stop:
no automatic resource, partition, batch, support or schedule change.

New exact source, contracts and roots; saved dry plans, explicit hash-authorized
live submission, exclusive claims and per-job journals for gate and science.
No automatic followup/retry/cancellation or commit/push/submission during local
implementation. Poor science never fails jobs or suppresses any registered row.

## 2026-10-15 CUDA failure investigation (not science admission)

User authorized investigating/fixing gate21896409. Preserve its original
e9b0b560 worktree, failed output and all parent/data/cache artifacts. CUDA illegal
access occurred in D000/D000 batch256 stress; earlier U050/direct technical passes
completed. The captured stack alone does not identify the originating kernel.
PyTorch2.5.1+cu118/cuDNN9.1 warns of a large-convolution fallback, but that is not
proof of the illegal-access cause. No OOM kill was reported.

One separate 1-hour SPORC debug diagnostic allocation (6 CPUs, 256000MiB, A100)
may run four fresh child processes: original D000/D000 asynchronous, original
D000/D000 launch-blocking, synchronous saved-tensor restoration D000/D000, and
synchronous restoration U050/U000. The sole candidate change is waiting for
the H2D restore; same tensors, dtype, shape/strides, BN/reductions, batch256 and
three longest-TRAIN steps. Candidate runs must first pass existing strict
FP32/BF16 three-update installed-Weaver/offload parity. A proposed convolution
tiling alternative failed local optimizer/gradient parity and was discarded;
do not relax tolerances or microbatch the model.

No validation arrays, model-quality selection, probability-bank publication,
scientific checkpoints or full-population training pass in this diagnostic.
Trace operation names/shapes/strides and exceptions. A failed child must not
prevent collecting the other explicitly registered diagnostic cases; each uses
a new Python/CUDA context. Overall worker exits nonzero if any case fails, even
if the comparison report was published. No automatic retry/followup. Reports
always say science_admissible=false, including an all-pass diagnostic.

The new source authenticates the failed gate using its ORIGINAL validator and
exact submission journal, with old source immutable. Relative to e9b0b560 only
the named new diagnostic src module and source allowlist may change; existing
scientific factories and workers remain byte-identical. A separate pushed
source, fresh diagnostic root, dry review and explicit exact-plan authorization
are required. No in-place environment upgrade. Production-size A100 evidence
and a separately registered full preflight remain necessary before any repair
is promoted into science. Instrumented timings cannot replace throughput
measurements or bypass the existing debug24h admission.

## 2026-10-15 synchronous restoration recovery /v2

User authorized promotion after diagnostic **21896886**, source658cafa5, report
`60abf23c62b03b2b627d8afc5823b69f65cedbf3564ab1edb104f7b0a48bea8b`.
Original asynchronous D000/D000 failed with CUDNN_STATUS_EXECUTION_FAILED;
launch-blocking original failed with illegal memory access. Synchronous restore
D000/D000 and U050/U000 passed strict FP32/BF16 parity and batch256/three-step
longest-TRAIN stress. MaxRSS135163040KiB of250GiB. This is A100 candidate evidence,
not a complete scientific admission or proof of the exact underlying CUDA bug.

For a NEW v2 execution only, use `alljets_fusion_sync.SynchronousFusionModel`:
same old full-population pair embedding and physical-span pack, same stats,
checkpoint layout, tensors, strides, dtype and seed; H2D restore uses
`non_blocking=False`. No production global monkeypatch. Old shared model and
factories remain unchanged. The canonical three-update parity harness temporarily
selects the actual new subclass for reference/offloaded constructors and restores
that constructor even on error; tolerances/BN/optimizer checks are unchanged.
Its explicit policy is `pair_saved_tensors_cpu_sync_v3`, not the old async v2.

This supersedes the preceding byte-identical-execution rule ONLY for new v2
recovery: add named sync/recovery modules and permit version-dispatch edits only
to the five existing alljets_fusion orchestration modules. All other old src bytes
(data, caches, splits, kernels, diagnostic, scientific graph/seeds/recipe helpers)
must match658cafa5. Reauthenticate the diagnostic with its original source, which
in turn authenticates the failed gate and completed regular parent with theirs.
Bind the exact report hash, all case payloads/parities, original and diagnostic
submission journals, terminal FAILED/1:0 accounting at creation, and old/new
source digests. A fresh nonoverlapping root is mandatory; never modify old roots.

Queue only ONE replacement FULL technical preflight: debug A10040GB, original
software/cuDNN,6CPUs,256000MiB,8h. Use existing TRAIN/validation cache and original
scientific U000 bank, no ROOT rebuild or control refits. Both paired families and
single-HLT must repeat full population passes, longest-role stresses, actual
production-subclass parity, checkpoint/bank readback and uninstrumented timings.
Use the same new factory in all subsequent eight science fits and reducers.
No CUDA_LAUNCH_BLOCKING instrumentation in v2 production.

Version GATE/SPEC/PREFLIGHT/PLAN/TASK/RESULTS/DIAGNOSTICS/SUBMISSION_CLAIM to v2
for this execution; require matching versions across dependencies. Original v1
artifacts are never relabeled as repaired evidence. Scientific data/population,
schedule60--100,batch256/BF16,seeds,loss,branches and all16 jobs remain unchanged.
The1.75 conservative100-pass projection and debug24h limit remain mandatory.
If runtime does not fit, save measurement.json but NO admitting preflight.json:
no automatic partition, schedule, safety-margin or batch changes. The CUDA
repair alone does not resolve the independently observed approximately60h bound.

## 2026-10-15 process-isolated recovery /v3

Job21898311 (source f979171c) disproved sufficiency of the synchronous-copy
workaround in the sequential preflight: U050 and direct completed, but D000/D000
failed full256 backward stress after passing small FP32/BF16 parity. There was no
OOM or runtime-admission failure. Fresh traced diagnostic success does NOT prove
uninstrumented full-run stability. The originating CUDA kernel remains unknown.

User authorizes fixing execution without changing the scientific experiment.
Use the byte-identical v2 synchronous model and all existing kernels. Change only
the preflight process lifecycle: one fresh Python/CUDA process per family/phase,
sequentially within ONE allocation. Run D000/D000, U050/U000, then single-HLT
stress, all BEFORE any population passes. Run each full-population technical pass
and checkpoint/bank roundtrip in its own fresh process as science workers do.
No TorchDispatch tracing, launch blocking, retries, skipped checks or fallback.
All cases still require original full256/three-step stress, role-specific support,
strict both-precision parity and uninstrumented timings. Small local tests are not
production-support acceptance. If isolated D000 still fails, stop immediately.

This supersedes v2's sequential in-process gate execution ONLY for a new v3 root.
Freeze f979171c's failed gate, original validator/journal/job21898311, runtime,
failure phase and Slurm log. All prior src files remain identical except the five
fusion orchestration modules; add only isolated_preflight and isolated_recovery.
The synchronous model, original recovery, diagnostic, physics, data/cache/kernel,
seeds, loss, paired BN population, schedule and 16-job graph remain unchanged.

One debug/A100/6CPU/256000MiB/8h job; original software environment. Children
must exit zero and have authenticated source/runtime/population/output evidence.
Small parity runs in its OWN child, never the subsequent stress process; each
full-size stress starts without preceding model work in that CUDA context.
Only the coordinating gate can admit science after all nine children pass.
Peak resource evidence includes the coordinator's process-tree RSS and every
child's sampled/high-water memory. Child failure never publishes admission;
do not query a poisoned CUDA context during exception reporting. Both successful
and failed child logs are preserved. No automatic retries or science followup.
The original100-pass/1.75-margin/debug24h admission remains unchanged. If it does
not fit, preserve measurements and request a separately authorized scheduling
change. Process isolation is an execution repair candidate, not a guaranteed
fix for the still-unlocalized A100 error or the independent walltime constraint.
