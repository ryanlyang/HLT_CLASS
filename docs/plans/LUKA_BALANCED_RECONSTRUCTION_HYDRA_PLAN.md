# Balanced six-class reconstruction screen on four GTX 1080 Ti GPUs

Authorized 2026-11-19. This is a new 600k/150k study, not a continuation of the
old 299234/74895 reconstruction trajectories or the full-size balanced ladder.
Scientific workers may change only their own execution. The separately
authorized clean-slate operation below may retire prior inactive Hydra artifacts.
Oscar artifacts and scheduler state remain outside that operation.

## Normalization repair and continuation, 2026-11-23

Submission correction after de6e9309: pass the absolute shebang batch script
directly to sbatch for normalization, release, replacement fits and report.
Dry review must parse that actual script and its arguments before cancellation
or live submission. Preserve the failed recovery root; a new pushed commit
creates a fresh recovery execution against the same scientific study. No
scientific fit, qualified normalizer or original journal is reset.

Jobs219507/R24,219508/D20,219509/R20 passed early GPU acceptance but rejected
their rebuilt normalization hash before the scientific miniature; zero epochs
were committed. Underlying numerical/statistical drift remains unproven. Original
qualification219148 on gpu1905 retained only hash
c06fa2b1d349e0a8615d0ef14132975bd6f1b543a46d3df3ebe2c455da9ffc59.

The operator subsequently captured the full record from running R22 job 219510.
Its hash matches the qualification exactly. Preserve that captured record in
the repair source and use it by default, avoiding another input replay or GPU
placement wait. Replacement fits must still authenticate it against the old
qualification and compare their local TRAIN statistics before using it.

Persist the full qualified TRAIN record and apply that exact mean/scale to all
raw FP64 targets before float32 publication. Fresh campaigns bind
INPUT_EXECUTION_POLICY/v2 and NORMALIZATION_EXECUTION_POLICY/v1. Observe the
local FP64 statistics separately: exact population/count/support/frame joins,
rtol1e-10/atol1e-12 mean/scale agreement, and full comparison evidence. Tolerance
is diagnostic only; the applied artifact hash remains exact. Preserve source
hashes, all selected TRAIN rows, original per-jet accumulation and test sealing.

For the existing dc5920f0 execution, use the explicit
[normalization recovery contract](../contracts/LUKA_BALANCED_RECO_NORMALIZATION_RECOVERY.md).
Recover a full record matching the old hash exactly, from a captured candidate
or one original-source replay on the original qualification node. A mismatch
saves the full record and releases no fits. Reuse original qualification only
with that exact record and unchanged scientific kernel/recipe/runtime. Keep
actual early acceptance and real-bank miniature in every replacement allocation.

An audited execution adapter runs new campaign-bound IDs while preserving old
source/spec/results. New fit parents record repair-source and comparison hashes.
It cancels only exact bound pending reconstruction/report IDs, preserving all
running/completed fits and queued baselines/teachers. New replacements cover
terminal reconstruction attempts with no committed epochs; partial histories
remain protected. Replacement siblings have only real teacher dependencies,
superseding the historical serial edges for these new attempts. The final report
includes retained and repaired fits. Review full prospective resources before
any mutation/submission. No new scientific method, sample, loss, epoch budget,
test capability, persistent particle cache or automatic HOME gate is introduced.

## Passing transport diagnostic and fresh-source recovery, 2026-11-21

Operator accounting and saved diagnostic status show three allocations on
gpu1905: original219050 FAILED/1:0 (2m34), Gloo-only219051 FAILED/1:0 (2m08),
Gloo/no-P2P219052 COMPLETED/0:0 (1m27), saved four-rank pass. With P2P enabled,
the known eight-element tensor SUM returned incorrect results on rank0 in
original and ranks1--3 in Gloo-only; the latter's root then timed out gathering
control objects because peers had failed. Thus Gloo alone is insufficient.
The no-P2P pass covers known tensor SUM/broadcast, host object control and
original OFFLINE_s10 independent four-rank parity on16 synthetic inputs.
It does not establish the root hardware/driver defect or all scientific arms.
Helper SHA:4a19da6c85172343141bbfbc43fa494bad59482439be6edee2f8435c56bf0466.

WORKER_LAUNCH_POLICY/v3 now pins GPU_TRANSPORT_POLICY/v1: Gloo host control,
NCCL tensor operations and NCCL_P2P_DISABLE=1. The campaign's common shell
sets the flag before its first Python import; the allocation/rank guards reject
missing or different values before CUDA use. Both early srun and scientific
torchrun inherit the fixed environment. This is an explicit campaign-local
transport choice, with no automatic fallback or changes to other campaigns.
The EARLY_GPU_POLICY/v2 proof includes the transport identity. All9 arms and
both reconstruction schedule positions must pass the five-minute step before
particle decoding on each qualification allocation; every fit checks its arm.
The production real-input qualification still gates release of18 fits and
measures timing/memory with this transport. Do not infer full-runtime speed
from the87s diagnostic. SyncBN, FP32 gradient SUM and all scientific recipe
settings remain unchanged; bitwise equality across transport choices is not
promised. Use a fresh pushed source/spec; old a2b634c7 and diagnostic helper/
receipts remain unchanged. The old three diagnostics need not be repeated.

### Historical failure and v2 control-only repair

The operator reports qualify218567 failed after6h06m16 in the12h allocation.
The launch boundary repair worked: four ranks reached mmap validation. Rank0
passed Weaver parity and entered OFFLINE full-weight parity; ranks1--3 timed
out in the initial Context.main NCCL broadcast after20 minutes. The subsequent
earlier-stack excerpt shows rank0 inside native SyncBatchNorm while peers were
still in that preceding broadcast. The ranks reached different collectives;
hardware/P2P failure is not established. No completed arm proof or science fit.

Before any further long attempt, run the isolated
[five-minute diagnostics](../LUKA_HYDRA_GPU_DIAGNOSTIC.md) on the failed node.
They compare original control transport, Gloo CPU control with NCCL tensor
reductions, and a diagnostic no-peer-transfer profile. All use synthetic inputs
and canonical untrained states; no real particle cache/data preparation or
scientific release. This supersedes earlier instructions to immediately retry
the whole qualification after a local-only launch/parser test. Keep all old
source/spec/logs. Probe success is not full qualification.

The current owned worker repair binds WORKER_LAUNCH_POLICY/v2: Gloo CPU
control plus a host barrier after root callbacks, with native SyncBatchNorm
and tensor/gradient collectives unchanged on NCCL. A separate five-minute
srun step runs actual four-rank synthetic parity before any population replay
or full particle preparation, in every GPU allocation. Qualification probes
all9 arms; science probes the current arm. Require valid completion and zero
exit before decoding. Preserve real-input qualification before science release,
all531 donor pins, scientific recipe, draws/epochs, readouts, initial-only HOME
admission and no persistent particle cache. See the
[v2 launch contract](../contracts/LUKA_BALANCED_RECONSTRUCTION_HYDRA_LAUNCH.md).
Remote acceptance remains the next action; no long retry is approved by local
tests alone.

## Worker argument boundary revision, 2026-11-21

Operator evidence: prepare217774 completed in5m10; qualify217775 exited1 after
5h49m28. All three input views completed. U100 construction took13547.9s; then
torchrun rejected worker `--node` as ambiguous with its node-rank options.
GPU ranks, parity, miniatures and scientific fits had not started. The earlier
five-minute stack samples were diagnostic; this final argparse error was fatal.

Put `--` between torchrun options and the rank script, and parse the real command
with installed PyTorch both before registration population work and before GPU
worker input work. Bind WORKER_LAUNCH_POLICY/v1 in fresh specs. No separate
preflight job is added. Increase qualification's allocation from6h to12h because
observed input preparation used almost the whole former limit. Keep science48h,
four GTX1080Ti,12 CPUs/64000MiB and every technical acceptance check. The12h
envelope provides headroom rather than a measured completion guarantee.

This execution revision supersedes later historical6h descriptions. Science,
source donor pins, reader/sampler, normalization, objectives,18 fits, seeds,
20 epochs, readouts and initial-only HOME policy stay unchanged. No persistent
particle cache is introduced. The failed job already removed its temporary
arrays; a fresh pushed source/study reconstructs them. Preserve58b9e655 logs,
population, canonical starts and submission records; never patch that checkout
or rewrite its spec to use new source. See the
[launch contract](../contracts/LUKA_BALANCED_RECONSTRUCTION_HYDRA_LAUNCH.md).

## Standalone storage revision, 2026-11-20

The operator now authorizes retiring previous Hydra campaigns and old inputs,
keeping what this new study needs. This supersedes the historical preservation
and protected-ladder reservation requirements later in this document. The new
study trains fresh OFFLINE/U100 teachers; it requires no old300k data, old model
weights, imported Oscar teacher bundle or earlier ladder execution records.

Fresh execution specs bind `STORAGE_POLICY/v2`, schema2: own8GiB output allowance
plus2GiB spare, zero other-campaign reservation, `protected_campaign=null` and
`other_campaign_dependency=null`. Initial fresh admission requires10GiB, credited
for own existing outputs on explicit resume. There are still no automatic HOME
rechecks at release, worker startup or epoch boundaries and no runtime output cap.
The model, population, sampling, objectives,18-fit recipe, readouts, output
inventory, reader repair and actual four-GPU acceptance requirements are unchanged.
Older policy-v1 executions are not migrated; use a new pushed commit/study.

The separate [clean-slate contract](../contracts/LUKA_HYDRA_CLEAN_SLATE.md)
keeps the whole new dataset/foundation, whole runtime/interpreter, shared Git
store in `code_65952ed3/.git`, newly pushed source, every `balanced_reco_600k_*`
root/log and existing retirement records. It archives compact text evidence
before retiring prior campaign models/predictions/inputs, including other
full-size new-dataset campaigns. Clean, pushed, registered old worktrees may be
removed; dirty, separate-store, unpublished or unknown source paths stay.
No cleanup runs inside training. The [operator commands](../LUKA_HYDRA_CLEAN_SLATE.md)
perform no job submission/cancellation and require an empty queue throughout.

## Population

Use the verified new QCD Run-2-combined/Higgs dataset and frozen six-class
600 <= offline jet pT < 2600 GeV population, hash
`5092ce94a5e32fac10bdbd5b91b1becddc3fe9123affc41a93a0b312db2ea616`.
QCD, X_bb, X_cc, X_ss, X_qq, X_gg retain labels 0..5. Twenty 100 GeV bins give
120 class/pT cells. Quarantine raw label 206 and exclude the other classes.

Allocate exactly 600000 TRAIN and 150000 validation rows proportionally across
their respective original class/pT cell counts using largest remainders, ties
by cell code. Select rows within each cell by a fixed SHA256 identity ranking;
selection is independent of the model seed. Preserve canonical parent order.
The 600k TRAIN counts are [427186,35806,34645,33939,33774,34650]. The smallest
cell has 1234 rows; 600000/(120*1234) = 4.051864 expected appearances/epoch.
These are replacement-sampling expectations, not maximum observed counts.

TRAIN draws are uniform cell then uniform row, 600000 draws/epoch, global
batch128, 4688 updates/epoch, final batch64. All arms of a seed use identical
draws. Validation retains natural class/pT frequencies, is never resampled or
loss weighted. Reduced-out TRAIN/validation rows retain their old roles. The
822313 in-range and 610553 outside-range test rows remain sealed; no test
particle or prediction API is introduced. This campaign evaluates only its
150k primary validation subset, not outside validation or a relabeled test set.

For efficient ROOT decoding, a reader may read bounded contiguous slices of
the original six-class in-range TRAIN/validation roles and discard rows outside
the small subset before feature construction, fitting statistics or training.
It may never logically read test or unused particle rows. Reconstruction targets
are selected TRAIN only; HLT-only CE inputs never request OFFLINE branches.

## Eighteen scientific fits

Fresh seed indices10/11, ordered seed-major with the priority comparisons first:

| Arm | Inputs | Training objective |
| --- | --- | --- |
| OFFLINE | Pure OFFLINE | CE |
| M0HLT | HLT only | CE |
| U100 | Matched U100 hybrid on HLT support | .25 CE + .75 T² KL from same-seed OFFLINE EMA05, T=2 |
| U100_W5 | HLT only | CE + 5 SmoothL1 to same-seed U100 EMA05 embedding |
| R24 | HLT only | U100_W5 + reconstruction, delay3/ramp3/max.3 |
| D20 | HLT only | U100_W5 + prepared decoder, delay3/cosine-ramp6/max.3 |
| R20 | HLT only | U100_W5 + reconstruction, delay0/ramp3/max2 |
| R22 | HLT only | U100_W5 + reconstruction, delay0/ramp3/max10 |
| R28 | HLT only | U100_W5 + block4+block8 reconstruction, delay0/ramp3/max.3 |

U100 is a newly declared single predecessor-KD teacher, not the historical
COARSE_U100 or full ladder's multiple-target UE100. OFFLINE and U100 are trained
on the same 600k subset as the students. Their selected EMA05 checkpoint supplies
one fixed bank per seed shared by every downstream comparison. No CONCAT,
contrastive matching, subclasses or auxiliary logit heads are included here.
The later combination with the full balanced ladder is a separate experiment.

Use the installed Weaver six-output native ParT and original independent
128->128->128 embedding predictor. Canonical untrained native/predictor tensors
per seed are loaded by every arm; independent decoder initialization preserves
native/dropout RNG. R28 differs in captured memory, not decoder parameters.
All fits use four GPUs, FP32/TF32-off, synchronized BatchNorm, sample-weighted
gradient SUM, global padding and global batch128. Decoder chunks8 preserve the
complete per-jet loss and all real constituents. Three solver threads per rank
fit the twelve CPU allocation. No optimizer reset, gradient clipping/projection,
batch accumulation or automatic learning-rate scaling.

AdamW retains the existing betas/eps/weight decay. Use peak3e-4, floor1.5e-5,
warmup2 then cosine through20 epochs. Schedule/EMA clocks count actual draws:
epoch-1+(draws_consumed/600000), including the final64 batch. These are adapted
20-epoch recipes, not claims to reproduce historical 30-epoch trajectories.

Reconstruct complete OFFLINE TRAIN particle sets in the paired HLT frame using
the original measured14->mixed22 transforms, TRAIN-only FP64 moments with .001
scale floor, validity masking, detached exact one-to-one assignment, equal
kinematic/composition/tracking groups, balanced presence and .1 normalized
count CE. No aggregate consistency term. K is the maximum selected TRAIN HLT
or OFFLINE multiplicity, not the old fixed298 and not a validation-derived cap.
D20 trains decoder parameters at .3 from the start, but its encoder VJP follows
the delayed cosine ramp. R24 has no decoder-only preparation during its delay.
During zero-weight delay, preserve the donor's zero decoder gradients, AdamW
moment/step advancement and decay. Explicit zeros avoid a costly zero-weight
decoder forward/backward without changing that optimizer bookkeeping.

## Readouts and fair reporting

Every fit yields separately selected RAW, RAW_BN, EMA05 and EMA10. EMA05 is the
predeclared primary; averaging starts at draw-position3, half-lives .5/1 epoch.
Only native parameters are averaged. Copies recalibrate BN on the same frozen
8192 balanced TRAIN draws (replacement, separate RNG); this intentionally adapts
the older uniform-TRAIN calibration to the new training distribution. Dropout
is disabled and live buffers/CPU-CUDA RNG are preserved during readouts.

Select by macro six-class OVR AUC, then CE, macro log R50 and earliest update.
Report accuracy, AUC, class R50/R30, passing-QCD counts and per-pT diagnostics
at each stream's same selected checkpoint. No R50/R30 reselection. Compute
recovery using the same seed and same readout of new M0HLT/OFFLINE. Missing,
censored or nonpositive gaps remain unresolved. Pair R24/D20/etc with U100_W5
and U100_W5 with M0HLT; report individual seeds, mean/min/sample SD and fixed
late epochs16..20 separately. Do not compare recovery with old eleven-class
anchors or select the maximum across readout streams as the primary result.

## Execution and bounded storage

Reader-repair follow-up2026-11-20: the diagnostic run217200 timed out while
building D000, before any GPU qualification checks. The last progress marker
shows64370 TRAIN+7063 validation rows after8470.7s, across57697 tiny spans;
subsequent samples show the parent waiting for a process-pool future. This
establishes slow preparation and then a nonreturning worker result, not its
underlying OS/I/O cause. Do not attribute the timeout to model quality or GPU
training, or repeat the original reader under a larger walltime alone.

Bind a new INPUT_EXECUTION_POLICY/v1 in fresh specs. Bundle consecutive spans
by source file/role into at most128 selected rows per CPU task. Preserve the
original single-role contiguous logical request boundaries, acceptance order,
all particle support and TRAIN-only statistics. Cache at most128MiB of Uproot
temporary numeric basket buffers per process; its final interpreter still
trims only to authorized entry bounds. No finalized test/unused row, new target,
whole-file array or HOME particle cache is permitted. Four bounded spawned
workers, own stack samples,600s submitted-batch deadlines and terminating pool
shutdown replace tiny Executor futures and indefinite failed-future waits.
Require bitwise original-reader feature/target/normalization parity and actual
local selected-TRAIN probes. See the
[input execution contract](../contracts/LUKA_BALANCED_RECONSTRUCTION_HYDRA_INPUTS.md).
Scientific recipe/resource limits remain unchanged; remote speed/acceptance is
still measured by Hydra qualification. Preserve all earlier roots and jobs.

Startup-diagnostic follow-up2026-11-19: the operator reports prepare216290
COMPLETED, qualify216291 TIMEOUT at6h, and release216292 CANCELLED. No GPU
working directory or arm proofs were published. CPU accounting is zero/absent;
this does not establish idle GPUs, a specific slow reader or a CUDA fault.
The failure occurred before the qualification working-directory marker, which
follows imports, source/runtime/allocation checks, population replay, construction
of all three views, NCCL setup and rank-side mmap validation.

Add flushed shell/import/stage markers, per-view decoded-row progress and
five-minute scheduled stack samples during startup and qualification. Limit
the common initial import probe to300s plus15s termination grace. This bounds
that probe only, not input construction or all startup. Stack samples do not
abort healthy work or certify qualification. Disable the parent's stack timer
while it waits on child ranks, and disable rank timers before ordinary science
epochs. Preserve the six-hour qualification and48h science limits until measured
evidence supports a resource revision. No scientific recipe, input selection,
target, readout, resource or artifact-contract change is introduced.
Use a fresh pushed source/study for this diagnostic retry; preserve4a264700's
spec, journals, completed preparation and logs. The new source cannot be used
to resume that source-pinned execution. Qualification remains required before
release; no old science fits need repeating because none were released.

CPU preparation authenticates source/runtime/data, builds the frozen population,
and publishes canonical untrained starts. A four-GPU qualification job measures
all nine arm types using actual selected TRAIN/validation inputs and complete
support. It performs independent full-graph update parity, discarded two-epoch
miniatures and native reload/EMA checks. It uses temporary untrained teacher
outputs only for technical probes; they never become scientific banks.

A dependent CPU release authenticates successful qualification and measured
48h envelope before submitting the eighteen fits and an afterany report. Each
fit repeats a short integrated acceptance with its real frozen teacher inputs.
No scientific metric threshold admits or suppresses a row. The first queued
bootstrap does not imply real four-card acceptance has already passed.

GPU fits: four `geforce_gtx_1080_ti`, twelve CPUs, 64000 MiB, 48h envelope.
Temporary input/target arrays are shared read-only through allocation-local
scratch, not replicated persistent HOME caches; the filesystem and host-memory
envelopes are measured. The decoder and target statistics are discarded probes
during qualification. Science starts always reload canonical untrained states.

Serialize scientific allocations, with afterany ordering plus afterok edges for
required teachers. Failures of an unrelated arm do not suppress other arms;
missing/corrupt required teachers fail closed. Resume requires exact terminal
accounting and preserves accepted submission journals; no automatic blind retry.

Reserve8 GiB for new outputs plus2 GiB operational headroom. Preserve the
existing14 GiB ladder-output reservation, crediting its authenticated committed
outputs when determining the unspent reservation. Registration must see enough
actual and declared current free space for both remaining reservations. Output
size is estimated from actual initialized tensors and exact selected token/row
inventory; fail before submission if this envelope is insufficient.

Operator revision2026-11-19: HOME admission applies only at registration and
explicit manual resume. Remove automatic HOME free-space/allowance rechecks
from submissions (including automatic release), GPU-job startup and epoch
boundaries. Record this as `STORAGE_POLICY/v1` in new execution specs. The
operator accepts genuine filesystem write failures if disk space runs out.
The nominal8GiB output allowance is an initial estimate, not a runtime cap.
Node-local input capacity, CPU/GPU qualification, lineage and atomic artifact
publication requirements remain. Existing execution specs are not rewritten;
launch this revision from a fresh committed source/study root.

Budget-sizing follow-up: the verified complete initialized inventory is
4,828,964,049 bytes (4.50GiB), including native readouts, prediction arrays,
teacher banks, restart staging,15% margin and2GiB metadata/overhead allowance.
Reduce this campaign's original10GiB initial allowance to8GiB, leaving over3GiB
above that conservative inventory. The separate2GiB spare and full14GiB
protected ladder allowance stay unchanged: at most24GiB initially required.
This is an execution budget revision only; model, population, objectives,
epochs, readouts and scientific recipe hashes remain unchanged. Existing
committed executions retain their original policies; use a new source/study.

The separate authorized pre-launch cleanup may retire only the inactive old
`contrast_delta_54473f10/keys/keys.npz` derived TRAIN export. Verify both manifests,
payload checksum/teacher lineage and current transitive job dependencies; retain
the manifests and a retirement receipt. Old contrastive replay needs that bank
restored or re-exported. This exception does not authorize changing raw datasets,
existing model/prediction outputs, source checkouts or scheduler state.

Follow-up: the live dependency audit protected that old key bank, and it was not
deleted. The separately authorized inactive cross-message miniature cleanup
completed with126MiB retired, according to supplied operator output. It removed
only authenticated one-epoch `setup/mini_*/selected.pt` weights, retaining live
dependencies, scientific models/predictions and original metadata. Its receipts
are under `retired_campaigns/cross_message_setup_weights_v1`; do not rerun it.
This file operation did not change the1e602e33 scientific execution or sources.

Checkpoint retention is current full restart state plus selected native states,
not twenty full optimizer backups. Each epoch publishes immutable metrics and a
full restart state, atomically commits it, then journals retirement of the old
state. Retain full per-rank CPU/CUDA/dropout RNG, optimizer, native/predictor/
decoder state, EMA values, best readouts and sampling cursor. Completed fits keep
native checkpoints, predictions, reports and manifests; restart payload retirement
occurs only after authenticated final publication. Never retire another campaign.

Source/runtime/model environment are immutable, pinned and rechecked. Exact
pushed source, dry scheduler review, durable intent/accepted-ID journals and a
detached launcher are required. Local tests and single-GPU CUDA checks establish
queue readiness; real Hydra four-GPU qualification remains mandatory before
science release, superseding generic Tigris validation for this Hydra workflow.
