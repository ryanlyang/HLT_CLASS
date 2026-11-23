# Balanced reconstruction normalization recovery: v1

Authority: the 2026-11-23 repair in the active balanced reconstruction plan.
Original execution is `balanced_reco_600k_dc5920f0_r1`, source
`dc5920f0ed95beeb288816276beb36c485332561`, qualification219148 on gpu1905.
Its qualified TARGET_NORMALIZATION/v1 hash is
`c06fa2b1d349e0a8615d0ef14132975bd6f1b543a46d3df3ebe2c455da9ffc59`.
Failed219507/R24,219508/D20 and219509/R20 saved no scientific epochs. The
observed error is a normalization hash mismatch; its numerical cause is not
yet established. The recovery must not invent replacement statistics.

New `LUKA_BALANCED_RECO_NORMALIZATION_RECOVERY_<KIND>/v1` records bind the old
spec/qualification file hashes, exact clean pushed repair source, original job
IDs, module payload hashes and policy. Original source/spec and completed
results are immutable. A fresh recovery root holds full statistics, source
provenance, reviewed commands, intents, cancellation records and comparisons.

Only exact qualified-hash records may be frozen. A candidate must validate
TRAIN scope, population,600k rows, complete token/support counts,22 fields,
scale floor, maximum and HLT frame. Without a candidate, replay the unchanged
original input builder once on gpu1905, requesting one GPU slot/4 CPUs/64000 MiB/
4h; this stage performs CPU work, not training. Save its complete record even
on hash mismatch, then fail without releasing fits. A matching candidate can
skip decoding. All temporary arrays stay in owned `/dev/shm` scratch and are
retired afterward; only small metadata persist. Release requires normalization
completion-last evidence, the exact qualified hash and COMPLETED/0:0 accounting.

The operator captured the complete record from R22 job 219510 on 2026-11-23.
Its verified content hash equals the qualification hash above. The source-bound
copy in `configs/operations/balanced_reco_dc5920f0_normalization.json` is the
queue wrapper's default candidate. This path uses a CPU normalization allocation
and no particle decoding. Validation against the original population and
qualification remains mandatory; the transcript is not treated as a fresh
normalization fit or a replacement qualification.

Every replacement applies the frozen FP64 mean/scale to raw FP64 targets before
the original float32 cast. It never reconstructs raw targets from standardized
float32 arrays. The observed record remains diagnostic. Population/rows/tokens/
support/maximum/role/frame must match exactly. Observed means/scales must be
within rtol1e-10/atol1e-12 of the frozen record; this check allows only numerical
drift and never changes the applied artifact or its exact hash. Save both full
records and differences before substantive mismatch failure. Raw source hashes,
membership, target shape and canonical row joins remain enforced. This does
not prove that the original mismatch was numerical roundoff.

Execution correction after de6e9309's submission rejection: sbatch receives
the absolute `run_luka_reconstruction_normalization_recovery.sh` filename
directly, followed by the existing project/spec/phase/task arguments. No shell
interpreter precedes the script filename. Validate its Bash shebang and Unix
line endings locally, then run `sbatch --test-only` against the actual script
and arguments, removing only unresolved dependency options. Do not substitute
a `--wrap=true` probe. Validate every command before cancellation, and again
before submission intents. Normalization, release, fits and report use this
same script boundary. New SUBMISSION_RESPONSE/v1 metadata retain the actual
return code/stdout/stderr linked to the intent before handling acceptance or
failure. Uncertain responses still stop without a blind retry.

The reported first normalization submission was rejected by sbatch's script
parser and created no accepted job. Keep its recovery spec, claim, intent,
cancellation records and launch log. Launch the corrected pushed source in
its fresh commit-derived recovery root. Existing cancelled original attempts
remain in the scientific journal and are eligible for authenticated replacement;
completed/running fits and queued original teachers are preserved as before.
This fixes orchestration under the existing v1 recovery schemas; science,
normalization and original execution identities are unchanged.

The execution adapter imports scientific modules only from the original clean
checkout and verifies the original full source/runtime/qualification. The new
normalization and input-publication modules are loaded by exact file hashes.
Spawned decoders retain the original importable worker functions; the parent
alone selects/applies the frozen statistics. Original model/kernel/training,
18-fit recipe, draws/epochs, seeds, targets, native/EMA readouts, GPU transport
and accepted per-allocation early/real-bank miniatures remain unchanged.

Replacement Slurm attempts use new IDs. Append ordinary original-format intent/
accepted records to a fresh submission attempt, explicitly recording the repair
hash and actual external adapter command. New fit INITIALIZATION/EPOCH/METRICS/
REPORT parents additionally contain `execution_recovery` and
`normalization_comparison`; they are visibly distinct from unmodified fits.
No original accepted receipt, epoch, report or checkpoint is rewritten.

Before submission, review the bootstrap plus prospective replacement resource
requests. An exclusive original campaign submission lock and recovery claim
prevent duplicate controllers. Cancel only exact, authenticated PENDING
reconstruction/report IDs, using a Slurm PENDING filter to preserve newly
running jobs. Preserve all running/completed fits and queued baselines/teachers.
Unrecognized new submissions stop recovery. Submissions write intent before
sbatch; uncertain replies stop without blind retry. Replacement candidates are
terminal reconstruction fits with no scientific epoch commits. Preserve and
stop on partial histories rather than resetting them. Required teachers must be
completed/authenticated or retain afterok dependencies on their live IDs.
Replacement siblings have no artificial serial edges. The report waits afterany
for the new attempts and preserved live fits. No performance threshold gates
rows, and no automatic retry is added.

Old source resumes must not be used for reconstruction after this operation.
A later failed original job may need another explicitly reviewed recovery;
this operation never duplicates a job that was still running at release.
The original measured full-input four-GPU qualification is reused only because
the exact original normalization is recovered and model/recipe/runtime semantics
are unchanged. Actual early GPU checks and real-bank miniature still run in
each replacement allocation before scientific training. No final-test access,
inference on test, persistent particle cache or automatic HOME recheck is added.
