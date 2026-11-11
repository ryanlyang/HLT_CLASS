# Embedding bridge / response / region validation

## 2026-11-11 original frozen-export batching and completed preparation reuse

User evidence: setup162131, source92397736, completed both data roles and
published PREPARATION/data manifests in8816.77s (host peak15088398336 bytes).
The following U100 replay failed:4096 rows, maximum normalized-coordinate
difference0.0014567375183105469, atol/rtol1e-4. This was not another historical
input hash failure or a storage error. Original saved TRAIN targets remain
unchanged. The original exporter uses consecutive128-row batches; the failed
panel extraction regrouped rows and split each batch across ranks. That
execution difference is established; its responsibility for all remote drift
is not established without a new Brown witness.

Execution v3 preserves original contexts for frozen U100 inference, records
both original-batch and regrouped diagnostics, and retains the strict tolerance.
Fresh setup runs this witness before expensive preparation. Explicit recovery
from the completed92397736 preparation authenticates source/terminal jobs,
completion manifests, all payloads and exact original U100 arrays. It links
49 immutable arrays into a fresh root with new manifest parents; old data,
claims/logs and failed witness remain. No scientific acceptance is reused.

Local evidence (same installed-Weaver environment below):

- Baseline:56 passed,1 skipped in55.84s.
- New recovery/extraction suite:19 passed in24.38s.
- Combined campaign/native/recovery regression:75 passed,1 skipped in66.34s.
- Final actual-CUDA original128/tail replay and early-failure/recovery ordering:
  3 passed in9.55s. These add two cases to the combined suite:77 distinct
  passing cases, one two-GPU/NCCL skip.
- Tests cover nonconsecutive panel order, original batch neighbours, whole-batch
  distribution, tails, duplicate/invalid indices, unchanged tolerance with bad
  targets rejected, payload corruption/partial inventories/lineage rejection,
  terminal accounting, shared payload inodes, and storage-credit accounting.
- Syntax:24 Python files and4 Bash runbook blocks pass.

SPEC/v3 and U100_REPLAY/v2 are versioned; PREPARATION_REUSE/v1 is new.
RECIPE/v1 and both input transfer formats are unchanged. No imported numerical
donor is edited. No repaired two-GTX1080Ti run or Tigris miniature was executed
locally; the numerical replay and remaining full GPU qualification are still
required on Brown. No commit/push or remote mutation/submission was performed
by the agent. Queue readiness here means readiness to run that recovery and
qualification, not a promise that every remaining production check passes.

## 2026-11-11 exact historical-input repair

Remote evidence supplied by the user: setup161306, source0a00441f, failed
before training at the TRAIN D000 fingerprint check. A first-difference scan
found one float32 ULP in relative eta (5.684341886080802e-14). A later full
fingerprint pass found different hashes for D000/U100/D033/D066/OFFLINE, with
all other summary fields equal. Only the first D000 numerical discrepancy
was measured; platform math/layout remains a hypothesis, not a proven cause
or a bound on all differences. Disk reported26GiB free, with no ENOSPC error.

The repair copies authenticated original tensors instead of recomputing
historical views on Brown. A minimal native delta fills five missing
role/coordinate pairs; only the new quarter views use physical reconstruction.
SPEC/v2 binds the delta. Tests explicitly compare original RECIPE/v1 and
TRANSFER/v1 against the original0a00441f implementation: unchanged hashes.

Before queueing, `main` advanced to2fb09f38 with the separate Oscar six-teacher
repair. The embedding-bridge source pin now freezes that exact commit; the
diff contains no changes to imported numerical donors. Its runtime allowlist
remains restricted to this campaign. User cleanup evidence confirms22 failed
partial arrays removed (12.87GiB), with hashes retained, failed root328KiB
and filesystem free space39GiB. This is user-supplied remote evidence, not an
agent-executed deletion or a repaired production acceptance.
After this pin update, the focused source-guard and original-recipe checks
passed:2 tests in6.92s; scoped whitespace checks also passed.

Evidence using the same local environment below:

- Before editing:2 focused preparation/source tests passed.
- New export/preparation/source tests:9 passed in24.03s. Synthetic ROOT
  replay succeeds with all required source cache files removed, reproduces
  original NPZ bytes and leaves source caches absent. Existing corruption
  and injected one-ULP replay changes both stop before completion manifest.
- Full campaign plus native-transfer suite: **56 passed, 1 skipped in50.68s**.
  Includes exact source routing, identity/offset rejection, resealed manifest
  tampering rejection, original U100 preservation, all scientific loss paths,
  real local one-GPU Weaver forward/backward and submission failure handling.
- Final source guard, canonical preparation and recipe checks:3 passed in8.19s.
- Syntax checks:23 Python files,3 Bash runbook blocks; local documentation links
  and CLI help pass. No imported donor runtime changed.

The skipped test needs two CUDA GPUs/NCCL. No real repaired SPORC export or
Hydra/Tigris miniature has run here; local fixture success is not remote
acceptance. The [runbook](LUKA_EMBEDDING_BRIDGE_HYDRA_RUNBOOK.md) specifies the
fresh v2 execution, reuses the completed physical transfer, and retains
measured two-GPU setup plus automatic release. Output reservation remains
33GiB after the extra input transfer; failed partial arrays are not promoted.

## Original implementation evidence (2026-11-10)

Implementation completed locally on 2026-11-10 for queueing real Hydra setup.
No Brown production setup, remote export, commit, push or research submission
was performed here. No result or runtime improvement is claimed.

Scientific authority: [plan](plans/LUKA_EMBEDDING_BRIDGE_RESPONSE_REGION_HYDRA_PLAN.md).
The [runbook](LUKA_EMBEDDING_BRIDGE_HYDRA_RUNBOOK.md) queues real two-GTX1080Ti
qualification and automatic release; [contract](contracts/LUKA_EMBEDDING_BRIDGE_HYDRA.md).
The exact internal donor baseline and reused files are in the
[source map](LEGACY_SOURCE_MAP.md): `45075fe6c2d8ccdcf7463cd3a99cbe2fe6976a08`.

## Implemented scope

- All 30 scientific fits and 22 native HLT endpoints; R0/P0 alias fresh C0.
- Three assistant ladders in unchanged U100 axes; shared six-condition TR;
  independently encoded physical-crop TP; all nine R and nine P objectives.
- Original assignments and native frontend; authenticated physical export,
  quarter-view preparation, unchanged TRAIN U100 arrays, evaluation-only
  targets and frozen TRAIN statistics in typed memory-mapped stores.
- Two-rank FP32/SyncBatchNorm execution, true parent dependencies, genuine
  production setup/startup, resource admission and automatic release/report.
- Epoch losses, throughput, memory, target coverage, fixed-panel diagnostics,
  checkpoint selection/replay, stripped native export and read-only contrasts.

## Local evidence

Python: `C:/Users/22rya/miniconda3/envs/tagging-hlt/python.exe`.
Tests used the repository's existing installed-Weaver fixture at
`tmp/hydra-weaver-fixture-20261102`, one thread per numerical library,
`PYTHONNOUSERSITE=1`, `PYTHONDONTWRITEBYTECODE=1`, and CUDA deterministic
workspace configuration `:4096:8`.

1. Baseline donor suite before implementation: **106 passed, 1 skipped**.
2. Combined campaign, two-GPU subclass donor and selective-teacher regression
   suite: **151 passed, 2 skipped in 111.99s**. Both skips require two CUDA
   GPUs/NCCL. This also exercised the donor's real two-process CPU Gloo checks.
3. After final reporting, CPU admission and epoch-observation additions, the
   complete new campaign suite: **48 passed, 1 skipped in 38.18s**. The skip is
   the genuine two-CUDA/NCCL producer pipeline and all-family SyncBN test.
4. After adding measured producer-chain cost reporting: **6 focused tests
   passed in 8.19s**, covering chain accounting, historical references,
   CPU measurements, automatic release and failed-setup handling.

Static checks passed for all 28 campaign files, Python syntax, both shell
scripts, all three Bash runbook blocks, 43 local documentation links and all
58 recorded donor paths at the exact baseline. New files have one final
newline and no trailing whitespace. The CLI help loads successfully.

Reproduction of the final campaign suite:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONNOUSERSITE='1'
$env:PYTHONPATH='tmp/hydra-weaver-fixture-20261102;src'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
& C:/Users/22rya/miniconda3/envs/tagging-hlt/python.exe -B -s -m pytest -q -p no:cacheprovider tests/test_luka_embedding_bridge_hydra.py
```

Meaningful campaign checks include:

- Exact original U100 AdamW loss, native state and optimizer-update parity;
  auxiliary heads absent from native reload; every loss reaches the encoder.
- Actual two-epoch native fits for all eight producers, target publication,
  reload and consumption by all final families; full30 C0/R5/P5 fixture fits.
- All original coordinate replays and new quarter views through the real
  physical frontend; changed original assignments rejected despite a newly
  valid outer shard hash.
- Response algebra and scale floors; crop nesting/permutation/isolation;
  valid-region/scale/jet reductions; source-bound targets and controls.
- Technical-bank rejection by scientific readers, corruption/role/hash
  checks, checkpoint selection, exact DAG and duplicate/ambiguous submission
  refusal; CPU/GPU/time/coverage failures preserve diagnostics.
- Full-depth eight-particle/two-class-block **one-GPU** FP32 forward/backward
  for C0, R8, P7, TR and TP with global batch128 and tails98/11, including
  observation gradients. This local card is an RTX2000 Ada laptop GPU,
  not a GTX1080Ti and not a two-rank acceptance run.
- CPU allocation rounding for release/report; measured CPU report kernels
  and admission arithmetic; optional S2IT/S6 ownership/corruption checks;
  missing fresh C0 never replaced by historical BF16 C0; producer-chain
  costs include all assistants and distinguish missing cost evidence.

Synthetic training fixtures reduce depth for fast integration checks; the
separate CUDA cases retain production depth. Neither uses research data.

## Required remote evidence

The runbook pins clean pushed source, exports original physical rows and
U100 weights on SPORC, verifies transfer bytes and available quota on Brown,
then submits setup and durable release. Setup must establish:

- Original view/target/checkpoint reproduction and installed-Weaver parity.
- Genuine two-GTX1080Ti NCCL/SyncBatchNorm loss, gradient, buffer and update
  parity; diagnostic RNG/state neutrality on every scientific family.
- All 30 production miniatures with actual technical parent-bank exports,
  every teacher condition/source/scale, largest support and tail batches.
- Warmed update timings, full validation and bounded serialized export costs,
  summed host/process peaks and per-card peaks under the registered bounds.
- CPU authentication/report measurement and explicit full-campaign estimate.

Each scientific fit additionally authenticates its actual scientific parent
and runs a disposable startup before fresh initialization. Technical weights
and banks cannot be scientific parents. A weak but finite teacher is allowed
to complete and release its dependent students.

Queue readiness is readiness to perform this remote qualification. It does
not mean that its resource estimates, two-rank parity or study outcomes have
already passed on Brown. The next task is the scoped commit/push and runbook;
the implementation changed no existing jobs or donor scientific runtime.
