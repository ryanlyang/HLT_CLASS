# Balanced reconstruction Hydra: queue-readiness evidence

Evidence date2026-11-19. This implements the
[600k plan](plans/LUKA_BALANCED_RECONSTRUCTION_HYDRA_PLAN.md), not a completed
physics campaign. No remote jobs were submitted, changed or cancelled. Existing
datasets, scientific checkpoints and other working-tree edits were preserved.

## 2026-11-20: Input-reader repair and measured local parity

Operator-supplied217200 log: D000 reached64370 TRAIN+7063 validation rows after
8470.7s across57697 tiny spans, then repeated parent-side waits on an unfinished
Executor result until the6h Slurm limit. This identifies input preparation as
the blocked stage; worker-level traces were unavailable, so its precise I/O or
kernel cause is not established. Qualification and science had not begun.

New source batches up to128 selected rows per same-file/role CPU task, retains
at most128MiB of temporary numeric basket buffers per process, finalizes only
original authorized spans to numeric NumPy arrays, and preserves FP64 physical
conversion and per-jet normalization order. Four spawned workers have own
stack samples and600s submitted-batch deadlines; failed pool shutdown terminates
owned workers. New specs bind INPUT_EXECUTION_POLICY/v1; scientific recipes,
resource limits, source donor pins and initial-only HOME policy are unchanged.

Focused pre-edit baseline:7 passed in17.94s. Final complete campaign suite:
**50 passed,1 skipped in97.13s**. The skip requires four actual CUDA/NCCL GPUs.
Tests cover multi-basket boundaries, cache hits/eviction/oversize nonretention,
authorized final row bounds, and byte-for-byte equality with the53bab0c5 reader
for all three coordinates, mixed22 targets and normalization. Interleaved
synthetic roles include test/unused/quarantined rows; the reconstruction case
uses all four spawned CPU workers. A real stalled child test verifies timeout
propagation and no remaining owned children. Old specs lacking the new policy
fail closed. All existing objective/reload/resume/HOME-policy tests still pass.

An authenticated local replay retained population
21175a03adc3d847e62d84a447b04525fc813207685a37d0b0dcb630acafd1f1. A discarded
technical probe read64 selected TRAIN jets from a merged Higgs source and64
from a QCD source, checking source hashes before/after. Original versus final
reader:2.906s versus0.453s, **6.415x** local serial speedup. Features, vectors,
masks and raw mixed22 target/frame/aggregate arrays were bitwise equal. No model
inference, validation particles or test evaluation. This small local probe does
not measure full Hydra/NFS throughput or the additional task-batching benefit.
Evidence (ignored local artifact):
`artifacts/audits/balanced_reader_repair_uk2m4b5g/evidence.json`, content hash
`c45a23e4493329a112b01fe9fded906530b6420f755927a26dde384a30803b0f`.

All531 committed donor payloads match their pins. Three working-copy byte
differences are solely Windows CRLF conversion; no dependency changes are
required or included. Scoped whitespace and package syntax checks pass.
No new scientific donor migration or Tigris job is involved. Actual Hydra
four-GPU qualification remains mandatory; no remote job/source/data mutation
was performed here. Next action: push this scoped revision and queue a fresh
study using its source-derived root, preserving prior records and other jobs.

## Operator-reported silent qualification timeout and diagnostic revision

The operator reports4a264700 prepare216290 COMPLETED/0:0 (5m24), qualify216291
TIMEOUT (6h00m25), release216292 CANCELLED, no qualification working directory,
and no arm proofs or qualification/task receipts. No scientific fits were
released. Zero CPU accounting is inconclusive. No independent remote inspection
has identified an input-reader, import or GPU-driver cause.

The subsequent read-only SSH probe to gpu1705 was reset during key exchange.
It supplied no GPU-health or scratch-manifest evidence. Do not interpret this
connection failure as proof of a bad GPU or completed input construction.

The follow-up adds flushed stage/progress logs and nonfatal scheduled stack
samples, and bounds the common initial import probe to300s with15s termination
grace. The parent waiting for ranks and ordinary scientific epochs disable
their stack timers. Both resource limits, all scientific donor pins, population,
objectives and readouts remain unchanged. Source changes require a fresh study.
Focused pre-edit baseline:8 passed in18.88s. Revised full campaign suite:
45 passed,1 skipped in86.46s. The skip requires four actual CUDA/NCCL devices.
New tests reproduce a blocked subprocess and observe its real stack sample
without aborting work, inject an import-probe timeout and verify the CLI is not
entered, and preserve exceptions without publishing a successful stage. Existing
reader role/target checks, source/recipe equality, actual local Weaver/CUDA
objectives, restart equivalence and no-runtime-HOME-check tests also pass.
Both changed Bash helpers pass `bash -n`; scoped `git diff --check` passes.
No real four-card acceptance or remote bottleneck fix is claimed by this change.

## Population and storage

The downloaded verified new dataset was audited locally using scalar membership
metadata. New subset build and independent replay both produced
`21175a03adc3d847e62d84a447b04525fc813207685a37d0b0dcb630acafd1f1`.
Evidence artifacts are ignored local files under
`artifacts/audits/balanced_reconstruction_600k_v1/`; no dataset enters Git.

| Class | TRAIN | Validation |
| --- | ---: | ---: |
| QCD | 427186 | 106429 |
| X_bb | 35806 | 8937 |
| X_cc | 34645 | 8689 |
| X_ss | 33939 | 8682 |
| X_qq | 33774 | 8488 |
| X_gg | 34650 | 8775 |
| Total | 600000 | 150000 |

All120 TRAIN cells survive. Minimum cell1234; maximum expected appearances is
4.051863857 per epoch. TRAIN HLT/OFFLINE support280/316; validation221/247.
Decoder support316 is TRAIN-derived. Test and reduced-out roles stay unchanged.
This subset audit read no particles and performed no inference.

A separate discarded real-data technical probe replayed the same membership,
authenticated the referenced ROOT checksums and decoded12 selected TRAIN jets,
two per class. It ran a production R24 CUDA update with full decoder support316;
these rows had maximum HLT/OFFLINE multiplicity131/143. All losses were finite.
Normalization fitted only those12 TRAIN rows and the embedding target was a
technical synthetic substitute, not a scientific teacher bank. No test particles
or validation inference were used, and no scientific checkpoint was published.
Local evidence:`real_train_probe.json` in the same ignored audit directory.

Initialized production tensor/row inventory estimates conservative new outputs
at4,828,964,049 bytes (4.50GiB), within the10GiB allowance. Native selected
states617,306,112 bytes; prediction arrays691,200,000; outgoing teacher
banks739,200,000; largest full restart estimate142,007,988. The estimate includes
staging and extra metadata margin. It is an inventory estimate, not a measured
remote disk peak. Physical/declared free space,2GiB reserve and the unspent
existing ladder14GiB reservation are checked before launch and throughout fits.

Budget-sizing follow-up: the current initial output allowance is8GiB, replacing
the original10GiB above. The complete initialized316-support inventory remains
4,828,964,049 bytes, leaving over3GiB additional allowance plus the separate2GiB
reserve. The protected ladder allowance stays14GiB. The initial maximum is24GiB;
it is an estimate/admission rule, not physical storage reservation. A dedicated
test reinitializes all18 production models and verifies the inventory and exact
scientific recipe equality with1e602e33. No further deletion is required merely
to clear admission at the user's last reported24.539GiB free snapshot.

Operator update2026-11-19: the paragraph above describes the original release.
The revised source checks HOME admission at registration and explicit manual
resume only. Its versioned execution policy removes submission/release, GPU-job
startup and epoch checks. Genuine filesystem write errors propagate; the previous
committed checkpoint remains available. The scientific recipe and donor pins
are unchanged. See the updated runbook for a fresh-commit launch after the
original01324411 attempt stopped at admission before registration/submission.

Revision checks: **51 passed,1 skipped in75.99s** across the reconstruction and
new retirement suites (41 reconstruction tests plus10 cleanup tests). The skip
requires four real CUDA/NCCL GPUs. Tests make the HOME checker raise if invoked
during scientific training, GPU startup or automatic submission, and verify those
paths still complete. Simulated`ENOSPC` at epoch2 propagates while epoch1 remains
committed; resuming reproduces uninterrupted selected native tensors exactly.
Cleanup tests verify one-file-only deletion, unchanged retained model/prediction/
manifest bytes, content-hash/checksum/lineage rejection, alias-normalized transitive
dependencies, unknown jobs, newly arriving consumers and exclusive retirement
records. No remote deletion was performed by these local tests. Scientific
Weaver/CUDA and full-support checks remain as described below.

## Local automated checks

Before implementation, the focused existing reconstruction-stability,
EMA-control and six-class Oscar suites passed **130 tests,1 skipped** in509.21s.
The skip required four real CUDA/NCCL GPUs.

The final combined new-campaign and existing Hydra-tail suites passed
**65 tests,1 skipped** in154.99s:39 new campaign tests plus26 existing tail tests.
The new skip also requires four real CUDA/NCCL GPUs; this check is enforced by
the production Hydra qualification before science release.

Coverage includes proportional deterministic membership, exact replacement-draw
clock/EMA, TRAIN-only BN membership, synthetic ROOT reads that forbid test rows
and undeclared OFFLINE branches, and all nine arm objectives. An independent
full-graph reference checks chunked losses, gradients, AdamW moments and
parameters, including delayed/asymmetric decoder schedules. Four actual CPU/Gloo
processes verify gradient SUM for uneven shards and partial batches.

Production fit functions run all nine arm types in compact fixtures, including
new OFFLINE/U100 banks, all four native readout exports, checksum/row-order
tamper checks and compact retirement. Failure injection verifies exact selected
tensor reproduction after an interrupted epoch, and final teacher-export recovery
without repeated training updates. Submission tests exercise durable intent,
accepted-ID bindings, ambiguous replies, duplicates and teacher dependencies.
Retirement cannot delete outside its own fit; allocation guards reject wrong GPU
counts/requests. All531 canonical donor pins match their committed bytes.

Actual Weaver0.5.3 forward/input/parameter parity and all nine new native/CUDA
paths passed locally with FP32 and TF32 disabled, including live token taps,
full-graph/chunked decoder comparisons at both schedule positions, and exact
native deployable exports. Local environment: Python3.10.19, PyTorch2.5.1,
CUDA12.1, NVIDIA RTX2000 Ada laptop GPU, Weaver0.5.3 source under the existing
`tmp/r8-followup-weaver` harness. This is single-card evidence, not Pascal or
four-GPU acceptance. Four new shell scripts passed Git Bash`bash -n`.

## Required remote acceptance

The new plan adapts generic Tigris validation to this authorized Hydra workflow.
No Tigris or real Hydra miniature has been run from this workstation. The queued
four-GPU qualification uses actual selected TRAIN/validation views, installed
Weaver parity, four-rank SyncBN/full-graph comparisons, both-seed discarded
miniatures, native/EMA reload checks and complete-support stress. It measures
full validation (including gathered metrics), BN calibration and training, then
checks a conservative48h projection and measured CPU/GPU memory margins.

Shared CPU arrays use PSS accounting so mmap pages are counted once across
ranks, plus private transient-peak, elastic-agent and input-launcher allowances. Resource proof is
specific to the requested four GTX1080Ti GPUs,12CPUs,64000MiB and this population.
It cannot be substituted by local Ada timing or old small-dataset runs.

Qualification must pass before automatic release. Every scientific allocation
also runs a discarded miniature with its real frozen teacher bank. No miniature
loss threshold gates scientific performance. Poor recovery is retained; invalid
inputs, nonfinite quantities, lineage drift, ambiguous submissions, missing
resources and corrupt artifacts stop execution. No final-test API was added.
