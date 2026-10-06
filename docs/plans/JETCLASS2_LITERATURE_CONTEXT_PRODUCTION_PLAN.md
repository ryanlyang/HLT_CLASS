# Frozen CONTEXT_V1: 2.25M paired synthetic jets

## Decision, evidence and authority

The user authorizes full dataset generation with the completed context pilot's
recipe, not another tuning round. This plan extends the training-only scope of
`JETCLASS2_LITERATURE_CONTEXT_PILOT_PLAN.md` to deterministic materialization;
it does not modify any pilot equations, rates, random keys or prior artifacts.

Approved pilot: commit `6cb5f32ab6e8d179f846f0cfeb4511ef1a0b6ab4`, spec content
hash `5e4fcd1297606421c957644ecbd4d06ff1ff1f01dd44f7a59309b6d998270f60`, at
`/home/ryreu/atlas/HLT_Classification/checkpoints/jc2_context_pilot_6cb5f32a_r1/study_spec.json`.
User-supplied Tigris job 225085 completed in 2m09s on 2026-10-06. It processed
20,000 training jets, preserved the 38.0021 mean and exact process replay,
transformed all 343,888 eligible particles, and measured raw inverse scaled
error 3.84e-15 and new-frontend inverse error 4.87e-7. Those are pilot results,
not full-production acceptance or evidence of classifier/KD improvement.
At registration authenticate the actual saved spec, receipt, report and full
parent chain, rather than trusting console text or recreating rounded rates.

## Frozen scientific construction

Produce **CONTEXT only**. Do not emit additional full NOISE_V3 or LOW_NOISE
datasets. The explicit composition is the unchanged low-noise
`literature_context.kernel.generate` followed by the unchanged
`literature_context.transform.transform`. Never use the low-noise generator
alone and call it CONTEXT. Reuse full-precision COUNT38 calibration from the
approved pilot, not a new population fit. Pin the pilot scientific source
files and input representation, as well as the production execution source.

Preserve identity-keyed random streams, post-response visible-p4 context,
all six value couplings and both error-scale maps, missing-track semantics,
PID conversions, drops and merges. Generation order, role, shard, process
count and retry must not change physical output. Raw inverse reconstruction
is checked per jet as an implementation invariant, not a performance gate.
The mean on the full population need not equal 38 exactly. No selection,
resampling, clipping or recalibration is allowed to force a desired mean.

This remains an artificial, literature-inspired mechanism benchmark, not CMS
or native Delphes HLT. Encoded tracking scales are not calibrated detector
uncertainties. New physical d0/dz/error columns are stored losslessly; the
required downstream 17-input asinh/log1p contract is recorded, not applied
destructively to the physical banks. All future classifier arms and teachers
must use that same representation. No classifier is changed or submitted here.

## Population, storage and pairing

Exactly **1,000,000 train / 250,000 validation / 1,000,000 final test**.
Reuse the approved pilot's dzfix offline source, inventory, existing TRAIN_1M
registry and the metadata-only 250k validation hash subset used by the previous
2.25M dataset. Inherit original profile eligibility; do not claim the source
selection was independent of native-HLT eligibility. No new split is created.

Read only the original offline particle columns and jet particle count from
ROOT; no labels, native-HLT particle branches or CMS response model. Pair by
canonical inventory/file/tree/original-entry identity. Preserve original
`part:{index:08d}` keys internally for RNG parity, but never persist these keys,
ancestry or latent draws as model inputs. Store the same offsets/identity/p4/
charge/category/tracking/valid physical-bank layout as literature production.
The distinct `JC2_CONTEXT_PRODUCTION_*/v1` namespace prevents confusing it
with the old recipe despite the common bank layout.

Create a fresh persistent sibling dataset root under the operator's confirmed
directory, normally `/home/ryreu/atlas/datasets`. Do not overwrite existing
CMS proxy, NOISE_V3, offline data or the Oscar copy. Default output budget is
20 GiB plus 2 GiB headroom; fresh available quota and persistence must be
attested. Filesystem free space is checked separately. Budgeted reservations
include preserved failed attempts. Never delete data automatically.

## Full queued DAG and acceptance

One reviewed submission schedules **preflight -> array -> finalizer**.
Preflight and generation elements: Tigris/reu-aisocial, 36 CPUs, 64 GiB, 2h,
no GPU. At most 16 generation elements (576 allocated CPUs) run concurrently.
Finalizer: 1 CPU, 16 GiB, 2h. File-local shards at most 25k jets, ROOT windows
at most 512 entries, banks at most 1k jets and bounded ordered process queues.
All native numerical libraries use one thread per process.

The first job is the genuine production-worker miniature: authenticate the
approved pilot and source, require matching Python/architecture/NumPy, read
up to 64 saved training identities through the production ROOT reader, and
compare serial/process/writer output exactly against saved **CONTEXT** banks.
Record measured wall time, process-tree RSS and bytes; check conservative time,
RAM and output projections before admitting the array. No second user action
is needed after a successful preflight. The new production preflight has not
yet run merely because the earlier pilot succeeded.

Bulk generation requires its exact successful preflight and submission
identity. Per-jet inverse checks produce integrity maxima only; no test
distributions or task metrics are computed. Finalizer uses afterany, validates
all receipts and physical block hashes, publishes train then validation role
releases and publishes a complete manifest only at exact full totals.
Missing or failed shards must never look like a completed dataset.

## Test boundary and recovery

The user's authorization includes final-test **materialization**, not final-
test inference, class metrics, distribution comparisons, recipe tuning or
classifier selection. A study-bound test-build lock grants only materialization.
Test receipts and complete manifests truthfully mark materialization/access
and always mark evaluation false. The public consumer exposes train/validation
only; test inference still needs a separately authorized evaluation campaign.

Submission is dry by default, requires exact pushed clean source, a reviewed
full plan hash, storage attestation and site test-only checks. Durable intents
and scheduler responses prevent duplicate/ambiguous submission. Same-source
recovery preserves completed shards and uses a fresh attempt for missing ones,
only after every prior exact job/array element is terminal and absent from the
live queue. No broad cancellation or silent source-changing repair.

## Verification and next action

Local tests must include the real synthetic context-pilot ancestry and ROOT
reader, saved CONTEXT parity, serial/process/chunk invariance, lossless banks,
approval/source/calibration drift rejection, role pairing, test sealing,
storage/corruption/incomplete publication and exact DAG/array recovery rules.
Weaver parity is not applicable to generation alone. Local fixtures do not
claim real production measurements. Push the new source, attest current
storage, review and queue the complete DAG on Tigris; then check preflight and
final manifests. Oscar transfer and classifier campaigns are separate steps.

## Operator entry points

After committing/pushing the scoped implementation and creating a clean detached
Tigris worktree at that exact commit, use
`scripts/queue_jetclass2_literature_context_dataset.sh`. Arguments are:

```text
COMMIT ROOT AVAILABLE_QUOTA_GIB [--execute PLAN_HASH]
```

Use a fresh persistent root such as
`/home/ryreu/atlas/datasets/jetclass2_literature_context_v1_2250k_<commit8>_r1`.
The helper requires `CONTEXT_PERSISTENT_STORAGE=YES` and a fresh quota check;
the default 20-GiB budget requires at least 22 GiB of available user quota.
The default approved pilot and TRAIN_1M paths are already set in the helper.
Never create a new split or substitute the Oscar NOISE_V3 dataset as its input.

First call without `--execute` to authenticate and create the immutable study
and complete dry plan. Review the three submissions (preflight, generation
array, finalizer), then repeat with `--execute` and that plan's exact 64-character
content hash. This submits the entire dependency graph, not just its initial
job. Run the helper through nohup with a fresh log on an unstable SSH connection;
do not run another helper concurrently. Inspect the existing submission journal
before retrying an interrupted or ambiguous submission.

The thin `scripts/jetclass2_literature_context_dataset.py` CLI provides `status
--spec ROOT/study_spec.json` for metadata progress, `release --spec ... --role
train` (or validation) to authenticate a completed role before test finishes,
and explicit `recover --spec ... --name NEW_ATTEMPT` for missing shards only
after all prior jobs are terminal. Recovery also uses a fresh reviewed plan.
Never modify old manifests, source pins or receipts to force reuse.
