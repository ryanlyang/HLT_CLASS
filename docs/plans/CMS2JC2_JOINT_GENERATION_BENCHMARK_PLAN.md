# Frozen JOINT generation throughput study

## Authority and boundary (2026-09-29)

Ryan asked to determine the fastest practical way to generate 1M training,
250k validation and 1M test proxy jets. This separate engineering study permits
**10,000 JetClass2 dzfix TRAIN jets only**, while CMS confirmation continues.
It does not qualify the mapping, change its parameters, select a new mapping,
read confirmation/validation/test particles, or authorize 2.25M production.
The older frozen CMS plan's no-JC2 boundary remains unchanged for that campaign;
this plan adds a distinct training-only benchmark capability, not a bypass.

Reuse the completed JOINT development selection, fitted response and joint map
with byte-identical scientific source and numerical environment. Use replica 0,
never select a lucky replica. Preserve the provisional GeV/mm/sign interface.
No native JC2 HLT or labels may be read or used as predictors. Historical split
masks retain their original HLT-match conditioning; that limitation is disclosed.
The user's future 1M/250k/1M budget is a projection only. Existing split profiles
are not edited, and no final-test capability is implemented.

## Fixed population and work

Require an authenticated dzfix inventory and explicit split profile. The exact
documented locations admitted are `jetclass2_10M_20260918_dzfix/jetclass2` and
`jetclass2_10M_20260918_dzfix_partial_v1/jetclass2`. The latter is the existing
SPORC partial snapshot, documented in `JETCLASS2_DZFIX_SALIENCE_MATCHING_HANDOFF.md`;
it is not an incomplete ROOT file or permission to substitute the larger local
inventory. Do not rename/symlink data to bypass the guard or accept arbitrary
`dzfix*` names. Keep the actual root, its own inventory and its own split profile
frozen. These two file sets and role assignments are not interchangeable.
The name is an operator release assertion, not proof of producer revision;
actual input identity is the inventory hash and per-file bytes. This operational
location correction does not change v1 membership, reader or response semantics.
Validate all split metadata, then select up to eight
training files deterministically, preferring different source groups, with at
least four files and sufficient eligible capacity. Distribute 10,000 rows evenly;
choose one hash-positioned consecutive window of eligible entries per file.
This limits repeated sparse ROOT reads. It is an engineering sample, not a
class-balanced physics or transfer-quality test; disclose the source mix.
Freeze membership before particle access. Runtime checks exact tree, file bytes,
entry masks, jet identities, no duplicate/omitted rows and offline branches only.
For efficient ROOT I/O, decoding uses at most 512-entry windows inside TRAIN
files. Gaps in the selected mask can decode other TRAIN rows; these never enter
the generator or output. Windows stop at the final selected entry. No window
opens a validation/test file. The 10k (gate: 64) bound is on generated unique jets,
not ROOT basket bytes or unused in-window training rows.

Use the existing Generator and JOINT map without changes. Load them once per
process, native thread pools limited to one. Bounded spawn queues preserve row
order. Each output has float64 p4/tracking, int8 charge/category, bool validity,
int64 offsets and uint8 jet identities. No labels, offline/source constituent
keys, topology traces or matching indices enter the output particle cache.
Generation keys are bound only in a separate parity digest. This benchmark
format is not yet an authorized deployable/classifier dataset contract.
Write fixed 1,000-jet NPZ output blocks atomically, validate checksums and perform
numeric readback. No lossy quantization, truncation, skipped jets or relaxed
validity checks. Empty outputs remain explicit. Persist artifacts under a new
root only. No automatic deletion or retries after ambiguous submission.

## Two separately authorized stages

1. `generation_gate`: one 4-CPU/32-GiB/4h job replays the first 64 frozen training
   rows serially and with four processes, including write/readback parity and
   exact trace-on versus trace-off output equality. Measure load, streaming,
   generation, output I/O, RSS and whole-task overhead. Estimate worst single-job
   resource/storage demand conservatively before admitting the 10k study.
2. `generation_screen`: two repeats each of five frozen settings, all using
   exactly the same 10k rows: 4/8/16 CPUs with process chunks of 32, 16 CPUs with
   chunks of 128, and 16 CPUs/chunks 128 with lossless deflate compression.
   Stored (uncompressed) NPZ is the other four settings. Jobs request 64 GiB/8h;
   the summed independent allocation ceiling is 120 CPUs. One 1-CPU/32-GiB/4h
   report depends on all ten jobs. No next stage is launched automatically.

Gate projections are deliberately conservative (no assumed 4-to-16 speedup).
For the 10k screen, reserve 1h authentication plus twice the serial generation
and input-read time scaled from 64 rows; RAM is six times sampled gate-tree RSS
plus 2 GiB, allowing for 4-to-16 child expansion. Storage is twice measured
stored bytes/jet times 10k times ten runs plus 2 GiB. These are conservative
admission estimates, not guarantees about unseen high-multiplicity inputs.
All files, including interrupted outputs, count toward the existing 12-GiB root
cap, and free-space headroom is required before screen creation. If the budget
does not fit, stop with measurements; do not silently reduce the sample.
Poor throughput or support/clamp rates are reported, not job-failure criteria.
Nonidentical outputs, invalid particles, source drift and forbidden reads fail.

One explicit partition per fresh study, default tier3; debug remains selectable.
Read-only probes show actual request shapes on both partitions but never treat
a single-job estimate as a campaign completion prediction. Reuse durable exact
submission journals, claims and allocation validation from CPU development.
No cancellation, partition migration or writes to old studies.

## Measurements and finite endpoint

Report total worker wall time (including authentication), processing wall time
(ROOT read/authentication, generation, writing and readback), summed generation
CPU-section time, serialization/write/readback times, sampled tree RSS/CPU,
output bytes/jet and throughput. Worker CPU and phase times overlap under
parallelism; do not sum them into a wall-time decomposition. Two repeats expose
variation but do not establish confidence intervals. Record hostname, allocation
and filesystem; cold cache effects and other users' contention remain caveats.
The total worker timer starts at Python worker entry and ends after final
source/environment checks; interpreter startup and final small JSON receipt
publication are outside that timer. These overheads and scheduler wait are not
claimed to be measured. Full-file byte hashing remains part of processing even
though only selected offline branches/rows are decoded. The gate times its
input-read separately because its 64 in-memory rows are replayed four times.

Require every setting/repeat's ordered physical-output and generation-key
digests to agree. Report fastest observed elapsed setting and best CPU-hour
efficiency separately. Give conditional 2.25M projections for 32/64/128 available
CPUs, with integer job packing and **no queue-time guarantee**. Include storage
projection; generated particle storage is not assumed to fit the current home
quota. Never multiply one-jet microbenchmark speed into a production promise.

Stop after report. A later production design needs completed CMS assessment,
JC2 training support/plausibility review, explicit new split sizes, storage and
resource locks, and separately authorized sealed-test materialization. This
study makes engineering progress while leaving all those scientific decisions
and the current confirmation untouched.

## Verification

Focused tests cover training-only source/membership, forbidden branches/roles,
source/map/receipt authentication, allocation/claim-before-read, serial/spawn,
chunk/compression parity, readback/corruption, storage/time projections, exact
dry/live authorization, unchanged old workers and incomplete report refusal.
Synthetic tests cannot supply SPORC throughput evidence; the real gate is the
next queueable step, not permission to submit the full dataset generation.
