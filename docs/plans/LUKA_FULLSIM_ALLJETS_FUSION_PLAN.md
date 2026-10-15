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
