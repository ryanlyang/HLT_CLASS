# Full-population RAW D20 campaign: implementation evidence

Final evidence date2026-11-26; implementation began2026-11-24.
The sections below distinguish supplied remote execution records from local
tests. Full four-GTX1080Ti Hydra resource/miniature qualification remains
pending. No remote files/jobs, final-test inference or working-branch
commit/push were performed by the agent.

## Activation-memory repair after qualification227995

The user's supplied accounting/log records show qualification227995 failed
after17:02:29 on gpu1703; release227996 was cancelled. All three full input
views completed. OFFLINE full-weight parity, seed miniatures and serialized
continuation passed; the maximum-support resource stress then failed in1.2s
with CUDA OOM on every rank. Each64-jet local batch repeated the longest316-
particle TRAIN jet. The reported GPU capacity was10.90GiB, with10.34GiB allocated
and an additional196MiB allocation failing. This is genuine VRAM exhaustion,
not a48h timeout or the earlier field-name bug. The four-rank stage ran394.7s;
most of the allocation was input preparation. No scientific fit was released.

The owned `activation.py` adapts the existing bridge recomputation helper from
donor d534fcfea4f11a4bc845127d03c31b779e391a8c, SHA256
eba048422ddc509af95369be15543d8ca3e12e00e20f01bf361acdba9ab329b6.
It invokes installed native module implementations, preserves RNG, and replays
BN/SyncBN with disposable buffers. Taps are explicit checkpoint outputs so a
recomputed forward cannot replace the live reconstruction VJP endpoints.
The independent reference disables recomputation. No parameter/state keys,
scientific recipe/population, dtype, batch or target support change.

ACTIVATION_EXECUTION_POLICY/v1 is mandatory; EARLY_GPU_POLICY advances to/v3.
The early actual four-card srun has15minutes and performs two global256 steps
at authenticated maximum native/decoder support before population replay or
particle preparation. Both synthetic early admission and later real stress
retain88% per-GPU memory limits. A failed early check never enters data builds.
Existing roots/specs remain immutable; deploy a new pushed source/fresh study.

Before edits,4 focused policy/DAG/scratch checks passed in29.30s. The existing
installed-Weaver CUDA13-arm/asymmetric-gradient and exact D20 resume test
passed after model adaptation in85.70s. New focused policy, failure-order,
CPU/CUDA dropout/gradient/AdamW/BN-state and native-export checks:12 passed in
28.73s. They verify original and recomputed RNG/buffers after ordinary updates
and both sides of the delayed reconstruction schedule.

Standalone local CUDA stress used the production kernel/model for two full
updates at local batch64: OFFLINE316 particles peaked at9,107,302,400bytes
(8.48GiB), D20 HLT280/full316 targets at7,318,498,816bytes (6.82GiB).
Evidence: `tmp/d20_full_checkpoint_memory.json`. The local device is an RTX2000
Ada laptop with Windows memory management; these are PyTorch allocation peaks,
not proof of physical Pascal fit or transferable speed measurements. Genuine
four1080Ti acceptance remains pending. Recomputing trades extra work for lower
retained activation memory; actual timing must still pass the48h envelope.

The broad repaired suite ran188.05s:73 passed and the clean-deployment case
caught a pre-existing shared donor change from the separate Oscar recovery.
Only `luka_sixclass_wide_b256_oscar/contracts.py` drifted: it added SPEC/v2--v5
execution helpers and72h CONCAT support. Its v1 recipe, population artifact,
flags, constants and default tasks remain exactly equal to the d534fcfe source.
The reviewed file now pins committed b94cb3d23c4fee6c05dbf6d1401d44474366e119,
SHA25623ab1b38578e1f5a3d5cd269df18ba3ca21f95259aa3382d5736287b62a07c06,
as an explicit `reviewed_overrides` entry;595 original pins are unchanged.
The shared file itself was not altered or reverted. The new compatibility test
and complete clean-deployment/stale-source test then2 passed in47.41s, verifying
all75 cases across the broad and focused runs. All596 current donor bytes,
owned Python syntax, documentation links and5 Bash blocks authenticate/parse.

## Input-schema repair after qualification226613

Repair base:0051dc8e0da4b9e791d9a2b6fbf7ba9b26318267. The supplied2026-11-25
logs/accounting identify root `d20_full_0051dc8e_r1`: prepare226612 completed in
5m17s, qualify226613 failed in7m57s on gpu1705, release226614 was cancelled.
All four early GPU ranks passed transport, installed-Weaver and the five base
arm technical checks; that stage took69.2s. Scalar membership replay completed
in294.3s. Full D000 input allocation then failed at once with
`KeyError: 'particles'`, before any particle construction or scientific epoch.
Neither qualification completion artifact was present. These are the user's
supplied records; the remote scheduler/payloads were not independently reread.

Four lookups retained the miniature population's `particles` field even though
the imported full population exposes `role_particles`. Corrected all four in
the owned input adapter: native allocation, target allocation, native coverage
and target coverage. No population alias or rehash, recipe change, input
cropping, normalization change or donor-source mutation was introduced.

Baseline focused input/metadata checks3 passed in15.76s. New real synthetic ROOT
roundtrips for D000 with/without reconstruction, OFFLINE and U100 all reproduced
the original KeyError before the fix (4 failed in17.14s); after repair,4 passed
in16.95s. They exercise the pinned wide inventory schema without `particles`,
full mmap allocation/decode/publication/reload, complete particle counts,
ordered IDs and labels, exact native arrays, TRAIN-only target values/masks/
frames/aggregates, sealed test/unused spans, and exact frozen-normalizer reuse.
The related donor ROOT/basket/worker-pool regression8 passed in41.13s.

The full repaired owned suite **64 passed in157.42s**, including the clean
deployment closure/stale-source test, all objective-gradient cases, real
installed-Weaver single-CUDA parity and exact serialized D20 decoder/AdamW/RNG
next-update restoration. Together with the related donor regression,72 tests
passed. Edited Python syntax, owned Markdown links, all5 runbook Bash blocks
and scoped `git diff --check` passed. None of these local tests claims complete
real full-population four-GPU qualification.

The original local implementation evidence below remains historical. The
repaired source requires a new pushed commit and fresh study; old source/spec
and accepted-job records remain unchanged. Because no science began, repeat
the small preparation task and full real qualification rather than inventing
cross-source reuse. The early pass at the old source does not authorize release
under the repaired source. See the [runbook](LUKA_D20_FULL_HYDRA_RUNBOOK.md).

## Original implementation evidence

Before edits, focused donor/benchmark baseline75 passed,3 deselected in212.02s.
The combined new-campaign/donor/startup/transport/benchmark regression passed
**159 tests,1 skipped, in675.19s**. The skip requires four actual CUDA/NCCL GPUs.
It includes real installed-Weaver single-CUDA parity for all13 objective arms
and exact D20 decoder/AdamW/RNG next-update restoration on CUDA.

After final role, scratch, late-schedule and journal refinements, the focused
suite passed **20 tests,40 deselected, in185.18s**. Those40 earlier objective /
CUDA cases had passed in the broader regression. The owned collection totals
60 tests. Separate late-schedule/submission checks passed12 tests in41.61s;
scratch/deployment assembly checks passed2 in64.63s.

The clean deployment fixture assembles the entire required source/document
closure in a separate local Git repository, verifies it, then confirms stale
source is rejected. It contacts no remote and does not commit the user's
working branch. All24 owned Python files parse; all596 donor files still match
Git/Linux bytes; owned document links, whitespace, CLI help, four Bash syntax
checks and Unix shebang/newline checks pass. No working-branch commit/push or
remote job submission occurred.

Coverage includes independent full-graph gradients for all coefficients,
fresh/matched starts, teacher-only dependencies and all92 segment endpoints,
real ROOT role-bounded reads and absence of privileged validation targets,
twenty-epoch RAW selected checkpoint equality with/without segmentation,
full D20 decoder/AdamW/RNG serialization and exact next-update replay, failed
write recovery and export-only continuation, all-unique TRAIN bank export,
streaming FP64 standardization, primary RAW versus teacher EMA05 semantics,
rehash-resistant scientific policy, real installed torchrun parsing, actual
script dry review and uncertain/duplicate submission evidence.

Production guards are explicitly replaced in synthetic fixture tests; these
tests do not claim real Slurm acceptance or four-GPU SyncBatchNorm parity.
No accuracy, speed or recovery gain is claimed from local technical tests.

The authenticated scalar wide-population report confirms2,488,148 TRAIN,
497,954 validation,144 cells/minimum805 rows and maximumHLT280/OFFLINE316.
Qualification final numeric arrays total approximately51.07GiB; D000 target
build peak approximately49.69GiB before runtime/private overhead. Request96,000
MiB, then measure full host/GPU peaks before science release. Allocations
repeat early communication/installed-model checks and actual-bank miniatures.

Initialized full-support output inventory14,748,932,625bytes includes all
concurrent restart/staging allowances. Initial HOME admission26GiB comprises
24GiB results plus2GiB spare; no automatic HOME checks or runtime cap.

Scientific adapters originate at d534fcfea4f11a4bc845127d03c31b779e391a8c.
`adapted_donors.json` records18 adapted modules; `donors.json` pins596 repository
Python files in the transitive imported closure. Their Git/Linux bytes match
the donor; no semantic working-tree difference was detected. Old source,
contracts/specifications and historical artifacts remain untouched.
The aggregate-only audited population JSON is bundled for portable metadata
tests; production still authenticates/replays the real row payload and roles.
