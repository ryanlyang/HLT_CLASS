# Frozen JOINT Tigris portability and throughput test

## Authority and scope

Ryan authorized this isolated test after Tigris accepted CPU-only probes and
reported 144-CPU CPU-only nodes in addition to 72-CPU GH200 nodes. Availability
is a user-supplied snapshot, not a reservation. Do not migrate, cancel or change
the CMS confirmation, old mapping fits, or SPORC benchmark. No GPUs, fitting,
validation/test particles, native JetClass2 HLT, labels or production generation.

This additive plan deliberately does NOT relax the original benchmark's exact
SPORC numerical-environment contract. New PORT_* v1 artifacts authorize a
different-environment replay with fixed tolerances. The original frozen JOINT
parameters, particle interface, keys, replica zero and scientific source remain
unchanged. This is an engineering gate, not mapping qualification.

## Three separately authorized stages

1. On SPORC, after the existing real `generation_gate` passes, create a separate
   `port_export` study/root. One 16-CPU/64-GiB/8h CPU job authenticates the original
   10,000 TRAIN rows and exports physical offline inputs plus reference JOINT
   outputs, in lossless 1,000-jet blocks. Reuse the existing exact gate population;
   never select a more convenient sample. Export is a one-time cost, timed and
   reported separately. Run only in the donor's exact numerical environment.
   Inputs contain physical fields, ordered identities and reconstructed `part:i`
   keys only inside the engineering reader. They are not classifier inputs.
   Output caches never contain construction keys or labels.
2. Copy only the completed packet directory, preserving its manifest and bytes,
   or use it directly if shared storage really is mounted. Record the manifest's
   SHA-256 from the completed SPORC result and explicitly supply that trusted
   fingerprint on Tigris. Never trust path existence or an unanchored self-hash.
   Create a fresh Tigris `port_gate` study with the exact same pushed source
   commit/files. Pin its own installed numerical environment. A 4-CPU/32-GiB/2h
   job replays the first 64 registered rows serially, trace-on, and four-process,
   and compares against SPORC. It records architecture, platform, dependency
   versions/native-library hashes, allocation, sampled RSS and filesystem.
3. Only after the real Tigris gate passes, explicitly create `port_screen` in
   that new study. Default: 16 and 36 workers, two repeats each. Freeze measured
   requests at screen creation: round projected runtime up to whole hours
   (minimum 2, maximum 8), and RAM up to 32, 64 or 128 GiB. Do not request eight
   hours/128 GiB automatically when the real gate supports a smaller envelope.
   Fresh creation can instead register a maximum of 72 or 144, adding those
   worker counts; this cannot be changed inside an existing study. Chunks are
   fixed at 32 and output blocks at 1,000, stored NPZ with exact readback.
   Repeats of a setting run sequentially. 16/36/72 settings can overlap (124-CPU
   ceiling); 144-worker repeats wait for all smaller settings, keeping the study
   at <=144 CPUs. Report follows every registered repeat; no cherry picking.

Export and all stage submissions require clean pushed code, exact dry-plan hash,
explicit authorization, immutable journals, allocation identity and an exclusive
claim before data access. No retry of an ambiguous submission and no cleanup.
All roots must be disjoint. The imported packet may be on a different mount, but
its relative paths and byte hashes remain fixed. Source archives/old absolute
ancestry paths are evidence only on Tigris; no CMS or ROOT reader is invoked there.

## Replay contract and admission

Across architectures require exact jet identity/order, particle counts/order,
PID/charge/validity arrays, generation-key digest and diagnostic flags. Compare
every p4/tracking coordinate against the SPORC reference with
`abs(actual-reference) <= 1e-12 + 1e-10 * abs(reference)`; all must be finite.
Report exact-byte equality separately; tolerance agreement is NEVER called
bitwise identity. Within one pinned Tigris environment, serial/process/trace
gate outputs and all full-screen repeats/settings must be bitwise identical.
No reference-dependent rounding or correction is applied to generated outputs.
Any discrete or excessive floating drift fails closed and requires investigation,
not an automatic tolerance increase. The full screen compares all 10,000 rows,
not just the 64-row gate. Every output still passes the physical particle schema.

Conservative screen admission uses serial gate time scaled to 10k, doubled plus
one hour overhead, without assuming speedup. RAM is 2 * max_workers/4 times
sampled gate peak RSS plus 2 GiB; disk is twice projected output bytes for every
screen repeat plus 2 GiB. Enforce 8h, 128 GiB, 12-GiB root cap, 2-GiB reports and
free-space headroom; refuse expansion if measurements do not fit. Poor speed is
a reportable result. Allocation, lineage, nonfinite or replay violations fail.

## What the measurement does and does not mean

This is cached-input mapping + output-write/readback throughput, including worker
startup in the measured processing section. It is NOT raw ROOT ingestion or a
full production end-to-end speed prediction. Export separately records ROOT
reading and reference generation together; transfer and queue wait are excluded.
Report both processing and worker-authentication-inclusive rates, per-repeat
times, allocated CPU hours, bytes/jet, host/architecture/filesystem and flags.
Do not assume an idle CPU total is one contiguous allocation. Compare actual
request shapes with test-only probes; select partition/site manually.

Even a successful Tigris screen does not unlock 2.25M production, final-test
materialization, mixed-environment cache reuse, or the claim that proxy HLT is
genuine detector HLT. A later production decision still needs CMS confirmation,
JetClass2 support review, split authorization, measured ROOT I/O, parallel
filesystem throughput, storage and a single frozen execution environment.

## Verification

Tests must exercise portable-byte corruption, missing blocks, explicit external
fingerprint, wrong source/roles/environment, exact discrete and tolerant float
comparison, serial/spawn output, CPU profiles/caps, gate-before-screen,
canonical dry/live journal behavior and disjoint roots. Real SPORC export and
Tigris replay remain required; synthetic tests do not establish portability.

## Operational entry points

Use a clean, pushed worktree at the same commit on both clusters. The existing
`queue_cms2jc2_generation_benchmark.sh` prepares/submits the original SPORC
`generation_gate`; do NOT submit its full screen merely to obtain this packet.
It requires explicit authenticated dzfix inventory, TRAIN profile and data root.
The older SPORC handoff's pre-dzfix dataset paths are not valid substitutes.

After that gate completes, in the SPORC environment:

```bash
python -s "${PROJECT_DIR}/scripts/cms2jc2_response_dev.py" create-portable-export \
  --parent-spec "${GEN_GATE_SPEC}" --root "${EXPORT_ROOT}" \
  --project-dir "${PROJECT_DIR}" --source-commit "${COMMIT}" --partition tier3
bash "${PROJECT_DIR}/scripts/queue_cms2jc2_portable_benchmark.sh" \
  "${COMMIT}" "${EXPORT_ROOT}/stages/port_export_r1/stage_spec.json"
```

Both helper calls here are dry review. Explicitly repeat the helper with
`--execute --reviewed-plan-hash HASH` to submit exactly the displayed plan.
When export completes, `portable-benchmark-results --spec EXPORT_SPEC` prints
the manifest path and trusted SHA-256. Transfer the `packet/` directory only
(or verify it is accessible on shared storage); do not copy 2.25M raw jets.

On Tigris, activate `atlas_kd_tigris`, pin the same source, then:

```bash
python -s "${PROJECT_DIR}/scripts/cms2jc2_response_dev.py" create-tigris-benchmark \
  --packet "${PACKET_DIR}/manifest.json" --packet-sha256 "${PACKET_SHA256}" \
  --root "${TIGRIS_ROOT}" --project-dir "${PROJECT_DIR}" \
  --source-commit "${COMMIT}" --max-workers 36
bash "${PROJECT_DIR}/scripts/queue_cms2jc2_portable_benchmark.sh" \
  "${COMMIT}" "${TIGRIS_ROOT}/stages/port_gate_r1/stage_spec.json"
```

Register 72 or 144 instead only at fresh creation if desired. The gate remains
four CPUs, and its measured memory projection determines whether the registered
screen can proceed. `probe-portable-benchmark --spec SPEC` tests actual stage
shapes without submission. After the gate passes, `advance-tigris-benchmark
--parent-spec GATE_SPEC` creates only `port_screen_r1`; separately review/submit
it using the same helper. `portable-benchmark-results --spec SCREEN_SPEC` reads
the final authenticated speed report. Existing `monitor --spec SPEC` applies.
No operation creates a production stage. Use the helper under nohup with a log
when SSH is unreliable; its heartbeat does not itself survive a disconnected
foreground shell. Never repeat an ambiguous live submission blindly.
