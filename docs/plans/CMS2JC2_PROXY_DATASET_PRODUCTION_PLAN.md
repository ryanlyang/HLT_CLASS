# Frozen CMS-calibrated JetClass2 proxy-HLT production

## Authority and scope

The user authorized implementing a 2,250,000-jet offline-to-proxy-HLT build on
Tigris with at most 16 concurrent 36-CPU generation tasks, no GPU requests.
This is a new campaign. It neither resubmits nor modifies older studies. It
supersedes the direct-generation engineering plan's 10k TRAIN-only boundary
**only for this separately authorized dataset build**. Older APIs stay sealed.

The product is a controlled CMS-calibrated proxy, NOT genuine detector HLT or
a claim that every CMS observable is accurately reproduced. The frozen JOINT
mapping, historical calibration, replica zero, random-key domains, physical
precision, and generation algorithm are unchanged. No fitting or selection is
performed here. Cross-platform compatibility is tolerance-based; exact replay
is required within the pinned Tigris numerical environment.

## Required evidence

The user-authorized [36-of-57 confirmation amendment](CMS2JC2_REDUCED_CONFIRMATION_PLAN.md)
adds a separate reduced-evidence route. It requires explicit
`--allow-reduced-confirmation`, a reviewed reduced report hash, and new study
and dataset contract names. It records incomplete full-population support even
if the subset checks pass. It does not change the original full-confirmation
route, frozen mapping, storage budget, retained writer pilots, or test seal.

Import the completed Tigris direct gate (64 jets, serial/process exact replay
and SPORC compatibility) and a completed frozen CMS confirmation. The operator
must review and supply the confirmation report content hash and acknowledge
the controlled-proxy limitations. Supported, rejected, or inconclusive physics
outcomes are recorded, not turned into execution failures. Wrong ancestry,
corruption, missing evidence, unauthorized reads, and nonfinite outputs fail
closed. Do not advertise physics qualification from technical completion.

Full historical authentication runs in a scheduled preflight CPU task. Its
receipt pins a copied frozen bundle, input metadata, confirmation report, gate,
source and numerical environment. Generation workers verify those copies and
receipts, without reconstructing the historical fits on every task.

## Population and identity

Use the exact authenticated dzfix snapshot and TRAIN_1M split profile from
the SAME split registry/inventory as the direct gate. Preserve its 1,000,000
training rows, nested existing training subset, and 1,000,000 final-test rows.
Select 250,000 validation rows by ascending SHA256 of the fixed domain
`CMS2JC2_PROXY_VAL_250K/v1` and the existing row identity, with row identity as
tie breaker, restricted to that profile's validation membership. Store packed
entry masks, source hashes, original ROOT tree keys, and ordered row digests.
No label or particle reads occur during membership construction. Existing
native-HLT eligibility conditioning in the source split is inherited and
disclosed; this is not an unconditioned new sample.

File roles are disjoint and never reassigned. Generate from offline branches
only; native JetClass2 HLT and class labels are not response-model inputs.
Jet identity remains the original inventory/path/tree/entry identity, so
sharding, recovery, and process scheduling do not change random draws.

## Execution

1. Create an immutable spec, source snapshot, population, and pilot plan in a
   new operator-named persistent project directory. Require an explicit
   persistent-storage declaration, an output budget, and an operator-reported
   available quota; filesystem free space alone is not a quota check.
2. Dry review then explicitly submit preflight and a two-element TRAIN pilot
   array (36 CPUs each). Choose two source-diverse files deterministically,
   retaining up to 10k rows each as actual production shards. The first 64
   rows of each are regenerated serially and compared exactly with process
   outputs. All pilot outputs are retained, not repeated by the bulk build.
3. Once pilot jobs terminate successfully, explicitly create/review/submit
   bulk. Derive resource requests conservatively from both pilots, bounding
   memory at 32/64/128 GiB and walltime at 1--8 hours. Reject an envelope above
   these bounds instead of silently underrequesting resources. Other shards
   have at most 25k rows and do not span ROOT files. An array `%16` bounds
   generation to 576 allocated CPUs. Child numerical libraries use one thread.
4. Bulk creation requires explicit sealed-test materialization authorization
   and writes an immutable build lock before any test particle is read.
   This authorizes dataset construction only, NOT final-test classifier
   inference, physics plots, hyperparameter selection, or response refitting.
5. The final aggregation job runs after all array elements finish (afterany).
   It verifies every shard's byte hashes, array shapes, exact row identities,
   membership counts, non-overlap, source lineage and prior physical-schema
   verification receipt. Full particle-level validation occurs in each
   generation worker before committing its receipt; the finalizer does not
   serially repeat millions of those checks. Missing or invalid
   shards prevent a complete dataset manifest. Job success alone is not proof
   of a complete dataset.

One process pool per array element uses the existing ordered, bounded process
generator with 36 workers and chunks of 32. ROOT reads use bounded windows;
the writer emits lossless stored NPZ blocks of at most 1k jets, checks exact
readback, and publishes a shard receipt last. Float64 physical arrays are not
downcast. Construction keys, association indices, and degradation decisions
are excluded from the physical bank.

Bounded ROOT windows may decode unselected rows of the same outer-role file;
only frozen selected rows are converted or generated. File checksum reads
necessarily hash the entire ROOT byte stream, but native HLT/label branches
are never requested as particle arrays.

## Persistence, budgets, and recovery

The output must be outside the raw data, old campaigns, and code worktree.
Do not default to the nearly full home directory or ephemeral scratch. This
tool cannot promise backups or indefinite retention; the project storage
allocation remains the operator's responsibility.

Each attempt reserves a bounded output allowance under a filesystem lock.
Reservations, including failed-attempt reservations, count against the total
budget; partial files are never deleted automatically. Blocks and JSON are
published atomically without overwriting different bytes. Logs and metadata
have a separate 2 GiB allowance; free filesystem headroom is checked before
publication. Final aggregation checks actual total usage as well.

Recovery requires all jobs from every preceding attempt to be terminal and
unambiguous. Fully authenticated shard receipts are reused; only unfinished
shards get fresh attempt directories. No live jobs are canceled, no existing
claims are erased, and no completed blocks are overwritten. Interrupted or
ambiguous scheduler submissions stop for explicit reconciliation, not blind
resubmission. The array cap applies campaign-wide because attempts cannot
overlap.

Array identity checks use `sacct`'s `JobID` (parent plus element index), not
`JobIDRaw`, which reports individual raw job IDs; see the official
[Slurm accounting field definitions](https://slurm.schedmd.com/sacct.html).
The array `%16` limit follows the official
[job-array concurrency mechanism](https://slurm.schedmd.com/job_array.html).

## Validation and honest status

Local tests cover population/role invariants, unchanged physical replay,
atomicity and corruption, test-read authorization, durable limits, dependency
plans, scheduler identity, incomplete aggregation, and recovery exclusion.
The existing 64-jet Tigris result validates the response kernel, not this new
writer at production scale. The retained TRAIN pilot is the required genuine
production-worker miniature and resource measurement before bulk release.
No remote campaign is submitted by implementation work.

## CLI sequence (after pushing an exact commit)

Use a clean detached Tigris worktree at that commit and the existing
`/home/ryreu/miniforge3-aarch64/envs/atlas_kd_tigris` environment. All commands
are in `scripts/cms2jc2_proxy_dataset.py`; run long setup commands under
`nohup` with a log outside the code worktree to survive SSH disconnects.

1. `review-confirmation --spec CONFIRMATION_STAGE_SPEC` prints the completed
   decision and report hash. This is read-only and can take several minutes.
2. `create --gate-spec TIGRIS_GATE_SPEC --confirmation-spec CONFIRMATION_STAGE_SPEC
   --confirmation-hash REVIEWED_REPORT_HASH --profile TRAIN_1M_PROFILE
   --project-dir PROJECT_DIR --source-commit EXACT_COMMIT --root NEW_DATASET_ROOT
   --persistent-parent EXISTING_PROJECT_STORAGE --budget-gib 50
   --available-quota-gib OPERATOR_VERIFIED_FREE_QUOTA
   --persistent-attested --acknowledge-proxy` freezes the population and pilot.
   `NEW_DATASET_ROOT` must be a new direct child of the persistent parent.
3. `bash scripts/queue_cms2jc2_proxy_dataset.sh PILOT_ATTEMPT_SPEC` prints the
   dry plan. Repeat with `--execute REVIEWED_PLAN_CONTENT_HASH` to submit.
   Normal initial attempt path: `ROOT/attempts/pilot_000/attempt_spec.json`.
4. After both pilot elements finish, `bulk --study ROOT/study_spec.json
   --authorize-sealed-test-materialization` authenticates retained shards,
   freezes measured resources and writes `pilot_admission.json` (including a
   queue/contention-excluding timing estimate). Review/submit the returned
   attempt with the same helper. Normal first bulk path is
   `ROOT/attempts/bulk_001/attempt_spec.json`; use the actual returned name if
   a pilot recovery was necessary.
5. `status --study ROOT/study_spec.json` is read-only. Add `--verify-shards`
   for full physical revalidation of committed shards (an explicitly
   authorized integrity read of materialized test data, not physics analysis).
6. Only after all earlier scheduler IDs are terminal, `recover --study
   ROOT/study_spec.json --authorize-sealed-test-materialization` makes a new
   dry plan for missing shards, or a finalize-only retry if all shards already
   committed. Invalid existing receipts stop recovery; they are never silently
   deleted or regenerated. Ambiguous submissions and dependency-stranded live
   jobs require inspection/explicit retirement before recovery.

Output is a lossless NPZ **physical feature bank**, not a drop-in replacement
ROOT ntuple. `cms2jc2_production.output.read_role(root, 'train'|'validation')`
returns authenticated row identities and physical arrays; labels can be
joined from the original authenticated split using identity, outside the
response model. Adapting a particular classifier's input pipeline is a
separate downstream task. Test inference remains sealed.
