# Direct Tigris frozen-JOINT generation benchmark

## Authority and distinction from the portable export route

On 2026-09-29 Ryan requested no further SPORC runs and supplied read-only
evidence that Tigris can access the existing GEN study, frozen bundle, completed
64-jet gate report, dzfix partial ROOT dataset and ARM Conda environment.
Implement a separate TG_* v1 route. Do not weaken GEN or PORT contracts or
pretend a 10k SPORC reference exists. No existing job or artifact is changed.

The completed SPORC gate is the reference, not a new export. Fully validate its
GEN lineage, output receipts and bundle before reuse. Freeze its exact TRAIN
10k membership, inventory/profile refs, data root, review, bundle and reference
result hash. Require byte-identical donor scientific source; only the existing
execution-only dispatcher/CLI allowlist may differ. Pin the new pushed clean
source and a distinct Tigris numerical environment. Never impersonate SPORC's
environment or silently replace missing dependencies. Both old and new source
identities and numerical environments are retained.

Full donor fit/map ancestry is authenticated once when creating TG_IMPORT.
That immutable import pins the donor stage/study, completed gate hash, output
receipt and materialized bundle. Subsequent workers revalidate those pinned
files, bundle, metadata/membership and output bytes without reconstructing the
historical fit/selection on every run. This avoids repeated old-model audit
cost; it does not skip authentication of the data or model actually reused.

## Two separately authorized Tigris stages

1. `tigris_direct_gate`: one CPU-only 4-CPU/32-GiB/4h Tigris job. Read the same
   first 64 TRAIN jets directly from authenticated ROOT files. Replay serial,
   trace-on and four-process, with exact within-Tigris identity/physical/key/
   flag parity and output readback. Compare each variant to the saved SPORC
   serial output. Discrete arrays, identities/order, particle counts, keys and
   flags must agree exactly; p4/tracking use the existing portable tolerances
   atol=1e-12, rtol=1e-10, with exact byte agreement reported separately.
   No rounding, resampling or relaxed tolerance on failure.
2. `tigris_direct_screen`: only created after that real gate passes. Two repeats
   at each of 16, 36 and 72 processes. Each job directly reads the identical
   10k TRAIN population, generates JOINT replica zero, writes 1,000-jet lossless
   float64 NPZ blocks and verifies exact readback. Process chunk=32 and native
   library threads=1. Repeats within a CPU setting are sequential; the three
   settings may overlap, allocation ceiling 124 CPUs. A report waits for all
   six runs and refuses missing/corrupt/nonidentical results. No automatic next
   stage and no production stage. 144 CPUs are not registered in this route.

Both stages use `tigris`, account `reu-aisocial`, one node, no GPU and no guessed
SPORC QoS. Use `/home/ryreu/miniforge3-aarch64/envs/atlas_kd_tigris` and the
absolute project worker path. Exact reviewed dry-plan hash and authorization,
durable submission intent/receipt and exclusive execution claim are mandatory.
Creation is metadata-only; workers check source, environment and allocation
before any particles. An explicit new TRAIN reader capability reuses the
unchanged bounded offline-only GEN reader. No labels, native HLT, CMS particle
data, validation, confirmation or test particles are opened. Existing matched
TRAIN-profile conditioning remains disclosed, not silently rebuilt.

## Resource admission and storage

Use gate serial generation/write/readback PLUS separately measured ROOT read
time, scale to 10k, double, add one hour. Round requests up to whole hours
(2--8 maximum) and RAM to 32/64/128 GiB. RAM estimate is twice the 4-process gate
peak scaled to 72 processes plus 2 GiB. Disk estimate is twice measured stored
bytes/jet times 10k times six runs plus 2 GiB. Refuse advancement beyond 8h,
128 GiB, the 12-GiB root cap or measured free headroom (twice remaining planned
bytes plus 5 GiB). Check headroom again on each screen worker. All old artifacts
remain read-only; no data copy, input cache, export packet, cleanup or deletion.
The user's 28-GB free-space snapshot is not a reservation or quota guarantee.

## Interpretation

Cross-site evidence covers **64 jets only**. Screen exact parity covers all
10k within the pinned Tigris environment and across worker counts/repeats, not
10k agreement with SPORC. Require screen 64-row physical and identity prefix digests
to equal the Tigris gate as well (different output block sizes do not matter).
Generation keys remain exact across all complete screen repeats/settings; no
per-row construction keys are added to the particle cache for the prefix check.

Report processing time (including ROOT input/file authentication and output
write/readback), authentication-inclusive worker time, CPU hours/million,
storage/jet, host/architecture, environment, per-repeat timing, fastest elapsed
and best CPU efficiency separately. The processing pipeline overlaps stages;
do not add summed child wall/CPU sections as a wall-time decomposition. This
measures raw-ROOT ingestion for this fixed sample; it does not establish
filesystem scaling, queue latency, production support or scientific fidelity.

No new fit, mapping selection or transfer qualification. The controlled
proxy-HLT is not genuine detector HLT. Future 2.25M generation needs its own
split/storage/source authorization and completed scientific assessment.

## Verification

Test frozen donor/membership/source/bundle/receipt corruption; no changed old
environment contract; disjoint roots; claim and allocation before reads;
serial/spawn and cross-site tolerance versus discrete drift; all six repeat
identities and prefix parity; resource/free-space admission; CPU-only exact
plans and submission idempotence/ambiguous refusal; helper stdin/exit behavior.
Synthetic tests are not real Tigris throughput or compatibility evidence.
