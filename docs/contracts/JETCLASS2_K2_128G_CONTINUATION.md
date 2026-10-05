# K2 128-GiB checkpoint-preserving continuation

## Authority and evidence

Execution-only amendment to
[the segmented continuation](JETCLASS2_K2_SEGMENTED_CONTINUATION.md), authorized
by the 2026-10-05 request to allow 128 GiB while continuing D025's training.
This concerns the original SPORC native-Delphes K2 campaign, not OSCAR NOISE-K2.

The user supplied these accounting observations from the original segmented
campaign `jc2_dzfix_concat_k2_segmented_71c1bde2_r1`:

| Task | Job | State | Elapsed | Batch-step MaxRSS |
| --- | --- | --- | --- | --- |
| Preflight | 21798257 | COMPLETED | 00:41:09 | 34.78G |
| D025 part 1 | 21798741 | COMPLETED | 22:04:39 | 23.48G |
| D025 part 2 | 21798744 | COMPLETED | 22:07:44 | 31.48G |
| D025 part 3 | 21798745 | PENDING | 00:00:00 | unavailable |

The preflight's main-process peak was 19.66 GiB; the original request was
320000 MiB (312.5 GiB). These observations motivate a 128-GiB request, not a
guarantee: main-process RSS and Slurm task-level MaxRSS are not measurements
of the aggregate simultaneous memory of every process. A fresh genuine
128-GiB allocation/cache/resume gate must pass before migrated science.

## Immutable identities and narrow cutover

The scientific donor remains
`c891da0d45dd3251dea9ea72df975bb96bae3570`. The only supported execution donor is
the original segmented v1 campaign at
`71c1bde2d759b11cddb1f5cfbc1d199847bd6eef`, after two successfully completed,
advancing D025 segments. Paths come from authenticated specs/ledgers, not job
names or a broad checkpoint-directory search. Both donor checkouts and roots
must remain available, unchanged and read-only.

New registration is `K2_SEGMENTED_CAMPAIGN_SPEC/v2` (schema version 2), with a
content-hashed `K2_SEGMENTED_MEMORY_RESUME_IMPORT/v1` descriptor. It binds:

- Original scientific spec/ledger and completed-prefix hashes.
- Segmented donor spec raw/content hashes, exact pushed source, genuine gate,
  gate/science ledgers and completed preflight/part1/part2 receipt hashes.
- Exact eleven unfinished donor task/job IDs, separately from completed IDs.
- The part2 endpoint's epoch, checkpoint hash, full-state binding and completion
  flag, plus raw hashes and sizes of every committed epoch payload/manifest.
- Additional final report/selected-weight files if D025 finished in part2.

Reject running, failed, unknown or completed remainder jobs, missing successful
prefix accounting, checkpoint advancement, a remainder claim/execution/receipt,
corrupt artifacts, altered scientific identities or an unreviewed training
kernel. The new `k2_segmented/training.py` must be byte-identical to the donor's.
Registration and the default queue helper perform no Slurm writes.

Live retirement requires explicit authorization and affects only the eleven
registered pending remainder IDs, using the PENDING state filter. All must be
CANCELLED before replacement submission. Completed D025 segments, completed
scientific rungs, unrelated attempts and other campaigns are never cancelled.
If a donor job starts during cutover, fail closed; do not kill or restart it.

## State preservation and graph

During the new preflight, copy every committed D025 payload/manifest into the
fresh root, byte-for-byte, without links, rewriting, deletion or rollback.
Check source and destination hashes and free space. Publish an immutable
`K2_SEGMENTED_RESUME_COPY/v1` receipt only after verifying the copied chain.
Conflicting target files are never overwritten. Copying all prior states costs
additional disk space; preserve the 32-GiB checkpoint budget and 4-GiB free-space
reserve. Authentication may take minutes on shared storage.

The in-progress D025 fit deliberately retains its **donor training binding**,
including original campaign, ordered dataset and D050 teacher identities.
The v2 spec, import receipt, execution/task receipts and new final report record
the new execution lineage. This is an explicitly registered cross-execution
continuation, not relabelling old weights as new or silently weakening hashes.
D000/compression are new fits bound to the new execution spec.

Resume current weights, AdamW, all RNG states, absolute epoch/update, complete
history, selected best weights and early-stopping state. Recompute the real
ordered cache/teacher bindings before accepting the imported state. Never
initialize D025 anew or reset its LR/patience clock. If part2 already completed
D025, part3 authenticates the copied final outputs and exits as a spare without
loading science caches or optimizing again.

The new full dry plan contains **13 jobs**:

1. Fresh native preflight and CPU after-gate launcher.
2. D025 **part3 only**, then its T2 reducer.
3. D000 parts1/2/3, then its T2 reducer.
4. HLT-x1 compression parts1/2/3, aggregate and complete.

Thus science contains **11 jobs**, of which seven are training segments. No
D025 part1/part2 is registered. All science waits for the passed new gate.
Reducers still consume only a completed selected fit, never a partial state.
If the last segment cannot finish, preserve state and fail closed for reviewed
recovery; the migration does not grant extra epochs or an unbounded segment.

## Resources, gate and scientific invariants

All new GPU jobs request **131072 MiB = 128 GiB** host RAM, one A100 and four
CPUs. Training segments keep 1380 minutes (23h); preflight/reducers keep 360
minutes (6h). CPU launcher/aggregate/complete keep one CPU, 8192 MiB and 240
minutes. Default partition debug, with pending **partition-only** debug/tier3
moves allowed and recorded; account/QoS/resources remain fixed. Operator
confirmation that RC permits segmented production on debug remains required.

Native preflight rebuilds the full ordinary D025 caches under the actual
128-GiB request, authenticates the copied state against their identities, and
runs the existing genuine installed-Weaver native-vs-extended and
uninterrupted-vs-reconstructed resume miniature. It checks the unchanged
GPU/software fingerprint, positive main-process RSS <=80% of 128 GiB, positive
CUDA peak <=90% capacity, and checkpoint storage. Acceptance records the new
memory request and exact resume-import hash. The old 312.5-GiB gate alone
cannot release this campaign. This is not a repeat of every old stress test;
the original expanded-input acceptance is also required. Local synthetic tests
cannot replace remote acceptance or guarantee future peak usage/queue speed.

Everything scientific stays fixed: 500k/1M/1M memberships, frozen matching and
features, physical/inference batch128, no accumulation, lossless pair storage,
seeds, C25P75/T2, AdamW, absolute LR schedule, max100/min60/patience15 starting
at60, checkpoint selection and final-test seal. Poor finite metrics are not
execution failures. Host RAM is reduced, not GPU capacity or the data budget.

## Operator interfaces

Use a fresh clean pushed executor checkout. The original v1 queue/worker
contract remains at 320000 MiB; changing its memory with `scontrol` alone is
not supported. Additive CLI mode:

```bash
python -s scripts/jetclass2_k2_segmented.py create-128g \
  --donor-spec "$K2_SEGMENT_DONOR_SPEC" \
  --campaign-root "$K2_128G_ROOT" \
  --source-commit "$K2_128G_COMMIT"
```

The dry-by-default `scripts/queue_jetclass2_k2_128g.sh` requires `PROJECT_DIR`,
`K2_SEGMENT_DONOR_SPEC`, fresh `K2_128G_ROOT` and full `K2_128G_COMMIT`. It prints
the authenticated resume epoch, completed status, full dry DAG and exact
retirement IDs. An explicit `--execute` additionally requires
`K2_DEBUG_POLICY_CONFIRMED=yes`, retires only the registered pending remainder,
then queues the new preflight and authorized automatic after-gate launcher.
Use the existing `gate`, `monitor` and `results` commands with the new spec.
No matching or completed fit is rerun. A fresh preflight can still take tens
of minutes and rebuilds transient caches; it does not advance scientific D025.
