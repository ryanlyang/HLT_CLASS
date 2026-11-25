# Full-population RAW D20 campaign: implementation evidence

Final evidence date2026-11-25; implementation began2026-11-24.
The sections below distinguish supplied remote execution records from local
tests. Full four-GTX1080Ti Hydra resource/miniature qualification remains
pending. No remote files/jobs, final-test inference or working-branch
commit/push were performed by the agent.

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
