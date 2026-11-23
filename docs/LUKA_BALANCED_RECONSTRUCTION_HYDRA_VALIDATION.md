# Balanced reconstruction Hydra: queue-readiness evidence

## 2026-11-23: Actual sbatch script parsing after de6e9309 rejection

The operator's first live normalization submission was rejected with `This
does not look like a batch script`. The preceding scheduling estimates did
not submit jobs. The owned wrapper incorrectly placed `bash` in sbatch's
script-file position, while dry review used `--wrap=true` and missed it.
Pending original reconstruction/report cancellations preceded this rejection;
the failed recovery spec/claim/intent/log must remain intact.

The wrapper now supplies the absolute shebang script directly for all phases.
Dry review sends the real script and its arguments to `sbatch --test-only`
before cancellation. Every command also validates its script/header before
live submission. Actual scheduler return code/stdout/stderr are retained in
intent-bound SUBMISSION_RESPONSE/v1 metadata, including rejections.

Baseline focused suite: 19 passed in 14.76s. After repair: **24 passed in
17.12s**. Added coverage exercises normalization/release/fit/report script
boundaries, resolved release dependency, malformed interpreter/header/CRLF
rejection before scheduler access, actual-script dry review, and rejection
evidence without releasing later phases. Combined reconstruction, recovery,
startup and transport regression: **108 passed, 1 skipped in 160.33s**. The
skip requires four real local CUDA/NCCL devices; installed-Weaver/CUDA, actual
four-process Gloo, decoder spawn and unchanged 531 donor checks passed. Bash
syntax, rank CLI and scoped diff checks passed.
No remote job or file was changed by the agent. A new pushed commit/root
continues the same original scientific campaign and retains its completed fits.

## 2026-11-23: Frozen normalization and existing-study recovery

Reported jobs 219507, 219508, 219509 completed their early GPU steps in 77--83s,
then failed before any epoch at `TRAIN reconstruction normalization changed
after qualification`. Operator confirmed qualification 219148/gpu1905 and the
expected c06fa2b1d349e0a8615d0ef14132975bd6f1b543a46d3df3ebe2c455da9ffc59 hash.
The failed record bodies were not retained by the old code; exact numerical
cause remains unknown. No claim that the difference is harmless was made.

The operator subsequently captured the complete record from running R22 job
219510. Local canonical-hash and TRAIN-scope validation confirms it matches
the exact qualified hash. Its source-bound JSON is now the queue wrapper's
default candidate: a CPU stage authenticates/publishes it without input replay.
It is not a new normalization estimate. R22 remains untouched while running.

New normalizer execution preserves the exact qualified record, casts only after
applying its FP64 values, validates exact membership/count/support/frame and
records local moment comparisons. Substantive drift still fails. The legacy
adapter preserves completed/running fits and requires exact old-hash recovery
before any replacement release. It does not regenerate a new arbitrary normalizer.
No original source/spec/accepted receipt/epoch/result is rewritten. New fit
parents and appended submission attempts explicitly bind the repair source.

Focused original baseline: 7 passed in 36.18s. Expanded combined regression:
**101 passed, 1 skipped in 164.09s**. Coverage includes actual synthetic ROOT
preparation through 4 spawn workers with the isolated publication module,
unchanged native arrays, exact frozen-normalized targets before float32,
substantive/scope/tamper rejection, lost-reference fail-closed behavior,
candidate recovery without particle reads, retained live/completed models,
teacher-only replacement edges, pending cancellation races, old-journal
compatibility, intent preservation and duplicate refusal. Existing 531 donor,
CPU/Gloo and single-card CUDA/Weaver regressions pass. The one skip requires
four local CUDA/NCCL devices. Bash syntax, rank CLI and scoped diff checks pass.

Final focused regression after worker guards, candidate launcher and captured
record updates: **21 passed, 54 deselected in 18.46s**. This adds exact captured
record verification, CPU candidate dependency checks, inherited scheduler
environment isolation and preexisting scratch protection.

Remote capture was performed by the operator; actual replacement jobs have not
been launched by the agent. Recovery must authenticate the exact qualified hash;
otherwise its saved full replay record is the next diagnostic, with no fits
released. Scientific allocations retain their actual early GPU and real-bank
miniature acceptance before training. See the dedicated recovery runbook.

Evidence date2026-11-19. This implements the
[600k plan](plans/LUKA_BALANCED_RECONSTRUCTION_HYDRA_PLAN.md), not a completed
physics campaign. No remote jobs were submitted, changed or cancelled. Existing
datasets, scientific checkpoints and other working-tree edits were preserved.

## 2026-11-21: Remote no-P2P pass and pinned production transport

Operator's original-helper -Status output confirms on gpu1905:

| Profile | Job | Result | Runtime |
| --- | --- | --- | --- |
| Original NCCL control |219050|FAILED/1:0; root tensor SUM incorrect|2m34|
| Gloo control, default P2P |219051|FAILED/1:0; ranks1--3 tensor SUM incorrect|2m08|
| Gloo control, P2P disabled |219052|Saved four-rank pass; COMPLETED/0:0|1m27|

The Gloo-only root reached object gather but timed out because peers had
already rejected incorrect tensor values. These failures precede real model
data/training and show that a control-group change alone is insufficient.
Passing profile covers known SUM/broadcast, host control and independent
OFFLINE_s10 full-graph/four-rank parity with synthetic inputs. Helper SHA is
4a19da6c85172343141bbfbc43fa494bad59482439be6edee2f8435c56bf0466, verified unchanged
locally. This is operator-supplied remote evidence; the agent did not submit
or independently fetch the remote jobs. Root hardware/driver cause is unproven.

New owned transport.py and WORKER_LAUNCH_POLICY/v3 pin Gloo control plus
NCCL_P2P_DISABLE=1 for NCCL tensors. The common shell exports it before Python;
rank/allocation guards reject a missing/different value before CUDA use.
EARLY_GPU_POLICY/v2 and completion evidence bind that transport. Both early
srun and full torchrun receive it. Scientific objectives/SyncBN/FP32 SUM, all
531 donor pins, sampler, epochs and HOME policy remain unchanged. No automatic
fallback or environment edits to other campaigns. Actual all9-arm early GPU
checks still precede full decoding; real-input/resource acceptance still gates
science release. The old diagnostic helper/source/results are preserved.

Pre-change focused baseline:9 passed in11.20s. Final combined regression:
**84 passed,1 skipped in126.83s**, covering campaign, isolated diagnostics and
startup/transport. The skip requires four actual CUDA/NCCL devices. New checks
exercise shell export before the first Python call (overriding inherited0),
rank rejection before CUDA for missing/invalid flags, policy/evidence identity,
unchanged original helper hash and environment propagation to both launches.
Actual four-process Gloo ordering and local CUDA/Weaver all9-arm parity also
pass. Bash syntax and scoped diff checks pass. All531 scientific donor pins
remain unchanged. The operator's remote four-Pascal OFFLINE check is separate
from local all-arm single-card evidence. Remote all9-arm early acceptance and
full real-input qualification remain required in the fresh execution.

## 2026-11-21: NCCL broadcast timeout and bounded diagnostic workflow

Operator's a2b634c7 job218567 log shows four ranks completed D000/OFFLINE/U100
mmap validation; rank0 completed installed-Weaver parity3.3s and entered OFFLINE
full-weight parity. Ranks1--3 timed out after1200s in NCCL BROADCAST (2 elements)
inside donor Context.main. No arm proof was written. Torchrun reported SIGABRT;
job FAILED/1:0 in6h06m16 under12h, then allocation scratch was cleaned. The
subsequently supplied earlier stack shows rank0 waiting in SyncBatchNorm while
peers remained in the preceding object broadcast. This establishes different
collective positions, not a proven hardware/P2P fault.

New isolated diagnostic helper/uploader compares original, Gloo-control and
Gloo/no-P2P profiles on gpu1905, each4 GTX1080Ti/12CPU/64000MiB/5m, serialized
afterany. The test first exercises Context.main, NCCL tensor sums/broadcast and
object gather, then old production OFFLINE parity on16 synthetic jets using
authenticated untrained starts. Source/spec remain a2b634c7. No real particle
access, cached-array construction or scientific fit/release. Per-rank trace
files,45s collective deadlines and5m allocation limits replace another blind
full-input retry. No-P2P remains diagnostic-only. The owned worker repair uses
Gloo for control objects/barriers and retains native NCCL SyncBN/tensor SUM.
WORKER_LAUNCH_POLICY/v2 requires a separate five-minute srun step before
population replay/full decoding, with actual four-rank synthetic communication,
Weaver and native/full-weight/auxiliary parity. Qualification covers all9 arm
types, including delayed schedules; each scientific allocation checks its arm.
Failed processes or missing/corrupt receipts stop before data work. No old
source/spec, scientific kernel, recipe, donor pin or HOME policy is changed.

Focused existing baseline3 passed in9.39s. New14 tests passed in6.62s, covering
job limits/profile dependencies, source/hash/lineage, uncertainty/duplicate
protection, CPU control routing, no data/training calls, missing rank proof and
rank-driver branches with mocked CUDA. Actual GPU probes remain unperformed
here. Next: operator runs only the short diagnostic launcher, then fetches
statuses/rank traces before a fresh-source full campaign attempt.

Expanded local regression: **78 passed,1 skipped in146.42s**, covering the
existing campaign, isolated diagnostic and early-startup suites. The skip is
the four-real-CUDA/NCCL test. After tightening installed-Weaver proof checks
and adding rank-driver ordering coverage, the final startup suite passed
**10 tests in29.22s**. This includes actual four-process Gloo object/tensor
ordering and error propagation, all9 native/auxiliary paths with the new
synthetic inputs on local CUDA/Weaver, unchanged full registered decoder
execution in the remote guard, bad/missing completion rejection and proof
that failed early execution cannot begin population/particle loading.
PowerShell parser, CLI rank --startup-only help and scoped diff checks pass.
The old531 donor-pin regression and full scientific recipe equality pass.
These checks do not certify actual four-GTX1080Ti operation. Remote diagnostics
were not submitted here; the existing authenticated terminal/key is needed.

## 2026-11-21: Torchrun boundary failure and qualification envelope

Operator log217775 establishes all three input views completed and were published
to allocation scratch. U100 alone took13547.9s. The subsequent torchrun parser
rejected child `--node` as ambiguous with --node-rank/--node_rank and exited2;
the worker raised CalledProcessError, removed allocation scratch and exited1.
No GPU ranks/arm proofs were created. Slurm FAILED after5h49m28 was neither the
earlier nonfatal300s stack timer nor a six-hour scheduler TIMEOUT. Prepare217774
completed in5m10. No scientific fits were released.

Reproduced that exact parser error with local installed torch2.5.1. The new
launch module inserts `--` before the rank script, verifies the actual installed
parser's script/arguments/allocation both before registration data/population
work and before GPU input work, and binds WORKER_LAUNCH_POLICY/v1 in fresh specs.
Qualification requests12h because the observed prep nearly consumed its former
6h budget; science48h,4 GPUs/12 CPUs/64000MiB, all reader/scientific semantics and
technical acceptance remain. No persistent cache or dependency change. A fresh
pushed source/study is required; old58b9e655 source/spec/receipts stay untouched.

Focused pre-edit baseline3 passed in7.23s. New focused launcher checks5 passed
in13.95s; final entire campaign suite55 passed,1 four-CUDA skip in106.58s.
The actual installed parser reproduces old failure and accepts fixed transport.
Four real CPU worker processes receive exact child arguments/rank/world values,
including paths with spaces. The Windows-only CPU smoke selects the real legacy
TCPStore constructor because the local wheel lacks libuv; production Linux
torchrun invocation is unchanged. Parser failure injection proves registration
stops before raw authentication/population, and worker stops before input reads.
Allocation checks accept12h qualification and reject old6h; science remains48h.
Existing tests still verify recipe equality,531 donor pins, reader equality,
all arm objectives, no automatic HOME checks, four-rank weighted gradients and
exact interrupted-training resume. Actual Hydra/NCCL acceptance is still required.

No remote files/jobs changed here. Next action: scoped commit/push and fresh
source-derived queue. Temporary arrays were cleaned after the failure, so they
must be reconstructed; neither timing nor a remote completed qualification is
claimed by local tests. The12h allocation is headroom, not a measured guarantee.

## 2026-11-20: Operator-reported partial cleanup and continuation

User-supplied journal reports15 completed removals and no complete.json.
Two old input/cache roots and13 old source worktrees were removed. Git returned255
for code_cross_message_73abecfd; original captured stderr was not printed, so
the underlying reason is unresolved. Old run roots were not yet attempted.
The user reports workspace usage65G; this is not a new measured free-space quota.

New helper/uploader explicit resume preserves that checkout and authenticates
the existing archive, scope, source and per-target receipts before continuing.
Old run/input payloads precede optional source removal. New continuation records
leave the original archive/receipts intact. Captured stderr is now printed for
Git failures. No scientific source, runtime, model or data contract changes;
the original58b9e655 campaign remains the kept execution source.

Focused baseline21 passed in1.49s. Extended clean-slate suite38 passed in3.14s,
covering the original failure, archive/journal preservation, partial recovery,
recreated paths, corrupt receipts, changed metadata, shared locks, queue changes,
preserved worktrees, missing-path attribution and different NFS client devices.
PowerShell parameter sets bind fresh Commit versus Resume independently; parser
and embedded Bash syntax pass. These are local filesystem tests, not remote
deletion or scientific/GPU acceptance. Next: operator runs local -Resume, waits
for COMPLETE with free-space/deferred-path report, then queues the original
kept source. Actual scientific four-GPU qualification remains required.

## 2026-11-20: Standalone storage policy and authorized workspace retirement

The operator supplied names-only Hydra inventory confirming the common Git store
at `code_65952ed3/.git`, the new raw/foundation dataset and the current runtime.
They authorize retiring all earlier workspace campaigns/inputs except what this
new study needs. That supersedes the narrower stability-only retirement scope.
No remote deletion has been performed by this implementation.

STORAGE_POLICY/v2 removes the14GiB other-campaign reservation and all reads of
the old six-class execution records. Fresh admission is8GiB own outputs plus
2GiB spare. Automatic HOME checks stay disabled. No scientific recipe, population,
readout, model, schedule, objective, input reader or resource limit changes.
Old specs fail closed; use new source and a fresh root. The old v1 budget evidence
later in this document is historical rather than the current admission rule.

Pre-change focused baseline:3 passed in12.06s. Updated storage tests:4 passed
in10.31s. Full campaign suite:50 passed,1 four-CUDA skip in100.24s. It includes
the initialized production inventory/unchanged recipe, original input parity,
no-automatic-HOME-check paths and filesystem-failure/resume checks. Actual
Hydra four-GPU qualification remains necessary; no new Tigris check is involved.

The separate cleanup authenticates kept pushed source/Git/interpreter and uses
a bounded CPU runtime import probe. It archives compact old metadata before
retiring old binaries without decoding or hashing them. Its focused tests cover
keep-set preservation, queue changes, altered runtime/keep contracts, external
imports, changed directories/reports, corrupt archives, partial deletion journals
and rerun refusal. No local test opens the real remote filesystem or submits jobs.
See [the cleanup contract](contracts/LUKA_HYDRA_CLEAN_SLATE.md).
Final new+existing cleanup regression:33 passed,1 Windows directory-symlink
privilege skip in3.67s (21 new clean-slate tests). PowerShell parser and embedded
Bash/queue-helper syntax checks pass. No remote cleanup or new qualification
has been performed; older payloads remain until the operator executes cleanup.

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
