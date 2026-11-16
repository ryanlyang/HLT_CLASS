# Six-class Oscar implementation validation

Status: queue-ready after commit/push; no Oscar jobs submitted by this work.

## Population evidence

The production scalar foundation and frozen unused-only extension were replayed
locally from all 6,996 authenticated membership files. The new production build
and independent load/replay agreed on population hash
`5092ce94a5e32fac10bdbd5b91b1becddc3fe9123affc41a93a0b312db2ea616`.
The latter replay took 70.23 seconds locally. This read no raw particle arrays
and performed no model inference or test evaluation.

Artifacts (ignored local audit output):
`artifacts/audits/sixclass_balanced_oscar_v1/`. They are reproduced from verified
scalar metadata during detached registration on Oscar; they need not be uploaded.

Counts: TRAIN 1,658,937, validation 328,589, sealed test 822,313. Outside validation
237,443. All 120 TRAIN cells are nonempty; minimum 3,413, maximum expected exposure
4.050535/epoch. Exact deterministic epoch-1 draw: 1,658,937 draws, 689,989 unique
rows, 968,948 repeated draws. Replacement sampling does not promise one visit to
each eligible jet per epoch.

Maximum HLT/offline/CONCAT particle support is TRAIN 280/316/596, primary validation
224/251/467, outside validation 240/291/531. No constituent truncation was added.

## Local checks

Before changes, the original Oscar and runtime-restoration suites passed:
68 tests in 128.03 seconds. New tests cover six-class population masks/replay,
all 120 cells and boundary behavior, six-output main/auxiliary heads, exact
main-KD classifier gradients, all six miniature 20-epoch production-worker
publications, teacher-bank checksums/row joins, selected R50/R30, fresh-anchor
recovery, interrupted optimizer/RNG continuation, export-only recovery,
submission journals, exact Slurm identity, duplicate prevention, and SSH-detached
shell behavior. Combined new-campaign/runtime suites: **76 passed in 141.34s**,
followed by the added actual detached-bootstrap lifecycle test: **1 passed**.
That is 54 new-campaign tests and 23 runtime-restoration tests, with no skips.
All five new Bash scripts pass `bash -n`. All 47 canonical donor Git blobs match
the manifest at `d28c403360f7b4662f679f5564529a048d8db5e2`.

Actual installed Weaver 0.5.3 and the local NVIDIA RTX 2000 Ada Generation Laptop
GPU exercised the unmodified production-sized six-output factory in all six
startup paths: FP32 forward/input/parameter parity, BF16 forward, two-epoch
miniature/reload, direct classifier gradients, native export, serial/optimized
contrastive equivalence and bounded CPU workers. Synthetic constituent lengths
are small; these tests do not measure full-data Oscar resource usage. CPU-worker
twenty-epoch fit tests reduce encoder depth but keep the true 128-dimensional
embedding and complete head/optimizer/export semantics.

## Production boundary

No live Oscar L40S qualification, Slurm allocation, package downloads, full-data
training or final-test inference has been performed here. Actual maximum-support
GPU memory and timing acceptance executes inside each science job before its
fresh fit; probes are discarded. No separate preflight jobs. Tigris was not used
for this Oscar-specific campaign. Queue readiness does not claim production
acceptance has already passed or that a six-class model will improve recovery.
