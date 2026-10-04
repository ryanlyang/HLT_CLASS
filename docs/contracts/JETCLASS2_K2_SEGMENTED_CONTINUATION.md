# K2 segmented continuation / v1

## Scope and scientific authority

Execution-only amendment authorized by the 2026-10-03 request to split the
remaining full K2 fits into two or three debug allocations. This is not a new
pilot or matching run and never changes an existing immutable campaign.

Only donor scientific source `c891da0d45dd3251dea9ea72df975bb96bae3570` is
supported. The CLI imports scientific modules from that original clean,
pushed checkout, not newer main; the additive executor is pinned separately.
Authenticate both commits, original spec/ledger, completed receipts and output
bytes. No production scientific modules are monkeypatched.

Seven completed fits and the D050 bank are read-only imports. New fits are
`CONCAT_K2_D025 -> CONCAT_K2_D000 -> HLT_X1_COMPRESSED`, with the immediate
parent's completed T2 bank. Only the first segment of each fit cold-starts.
Matching, foundations, validation partition, controls and completed rungs are
not rebuilt. If the original remainder advances after registration, fail
closed rather than duplicate a completed fit.

Keep the original 500k/1M/1M membership, validation checkpoint/diagnostic/report
roles, capacity, matching, features, masks, seeds, C25P75/T2, AdamW, physical
batch128, inference batch128, no accumulation, and pair-tensor CPU storage.
Keep max100/min60, significant-AUC delta5e-5, patience15 with clock starting
at60, absolute-epoch warmup/hold/decay/floor schedule, selection tie-breakers
and best-weight restoration. Earliest ordinary stop remains75. Poor finite
metrics never fail a job; final test remains sealed.

## Graph, resources and time budget

The full dry plan contains15 jobs: native preflight, CPU after-gate launcher,
then13 science jobs (nine segments, two reducers, aggregate, complete).
Live `full` submission is forbidden. A passed native gate and durable operator
authorization permit the after-gate worker to submit science automatically.

Each fit has three sequential `afterok` segments: one A100,4CPUs,320000MiB,
23h each. Preflight/reducers use the same GPU/CPU/RAM with6h; metadata uses
1CPU,8192MiB,4h. Account `reu-aisocial`, QoS `qos_tier3`, no requeue, default
partition **debug**. Pending partition-only moves `debug <-> tier3` are
accepted and recorded. CPU/RAM/GRES/time/account/QoS must remain unchanged.
Moving a launcher does not change descendants' default submission partition.
The operator must confirm RC permits segmented production use of debug;
faster scheduling is not guaranteed.

Before starting another epoch, require room for1.5 times the longest of the
last five epochs plus30min for saving, final report evaluation and shutdown.
The first epoch uses a1h estimate before the1.5 factor. Slurm elapsed time
includes source checks and cache rebuilding. Pause only after a full epoch,
save atomically, publish a receipt and exit successfully. Never deliberately
hit the time limit: TIMEOUT/OOM/cancellation/failure does not release `afterok`.

Every completed epoch is saved. Retain all committed state files; no automatic
deletion. Native acceptance measures payload size. A1.5 size margin times100
passes times3fits must fit32GiB and leave4GiB free. Every save rechecks space
including an atomic-publication reserve. Source artifacts are never scratch.

Three23h allocations are a finite budget, not guaranteed completion. Caches
are rebuilt per active segment; preparation, queueing, validation and IO add
overhead. Previous ~43h fits suggest two or three allocations but do not prove
all future fits fit within69h. If part3 cannot finish, preserve state and fail
closed for an explicitly reviewed extension; do not shorten the epoch budget.

If a fit finishes in part1/part2, spare prequeued parts authenticate the final
outputs and exit without cache preparation or optimization. They still need
a Slurm allocation and can add queue delay. Reducers wait for part3's receipt,
which carries the same completed fit when spare.

## Full-state resume and publication

New namespace `K2_SEGMENTED_*/v1`; checkpoint commit
`K2_SEGMENT_CHECKPOINT/v1`; payload `K2_FULL_TRAINING_STATE/v1`.
Original artifacts retain their old contract and `rolling_resume=False`.

Each immutable `epoch_NNN.pt` contains:

- Current weights and buffers, including BatchNorm, plus complete AdamW state.
- Python/NumPy/CPU Torch/CUDA RNG, completed epoch and optimizer update count.
- Complete validation/loss history, selected best weights/metrics/key/epoch,
  significant-AUC value, patience clock and completion status.
- Precision/determinism/thread policy, checked rather than silently changed.
- Original/new campaign, node/recipe, ordered populations and teacher lineage.

No GradScaler is used by the original BF16 recipe. LR is computed from absolute
epoch/update position; the sampler retains the original epoch seed derivation.
Construct model/optimizer and caches first, restore RNG last. Resume **current**
weights; restore selected best weights only at final reporting/export.

Publish payload atomically, then a content-hashed commit naming payload hash,
size, subject and preceding commit. Verify the manifest chain, latest payload
bytes and preceding segment's exact checkpoint hash. Reject corrupt,
cross-subject, wrong-epoch, orphaned or ambiguous state without silent rollback.
Failed jobs retain committed checkpoints but are not automatically considered
clean handoffs. Persistent claims/ambiguous submission intents need explicit
reconciliation. No mid-epoch resume or blind Slurm requeue is supported.

Only completed fits publish selected weights, REPORT metrics, HLT-only
deployment metadata and T2 banks. Neither partial nor acceptance weights can
teach science. Original identity joins and final-test prohibition remain.

## Native gate and retirement safeguards

The completed original acceptance supplies expanded-input stress evidence.
A new genuine installed-Weaver A100 resume gate remains mandatory: rebuild
ordinary D025 caches, run a short native KD miniature at physical batch128,
compare original vs new uninterrupted kernel, then uninterrupted vs serialized
checkpoint reconstructed into a fresh model/optimizer. Compare current/best
weights, AdamW, RNG, counters, metrics and scientific history. The scoped
deterministic parity backend is restored before science. Ordinary CUDA keeps
the original backend; bitwise equality across nondeterministic GPU runs is
not promised.

This is a resume-interface acceptance, not a repeat of every old stress test.
Require identical installed environment/GPU, original batch/capacity,
CPU/GPU headroom and measured checkpoint-storage bounds. CPU tests cannot
certify remote acceptance. A failed gate blocks automatic science.

Workers authenticate exact IDs against immutable submission journals, actual
Slurm resources and both checkouts. Execution receipts record actual partition
and source identities. Fresh output roots cannot overlap original artifacts,
data or either checkout. All original science artifacts remain read-only.

Derive the seven old unfinished IDs from the authenticated donor live ledger.
Retirement is dry by default. Live retirement accepts only PENDING or already
CANCELLED and uses `scancel --state=PENDING` on those exact IDs; a race to
RUNNING or unknown accounting fails closed. New live submission requires all
seven CANCELLED. Other K2 attempts and all jc2fc/jc2salp/CMS campaigns are out
of scope. No broad-name cancellation or automatic partition migration.

## Interfaces and evidence boundary

`scripts/jetclass2_k2_segmented.py`: create, retire, submit, run, gate, monitor,
results. `scripts/queue_jetclass2_k2_segmented.sh` is dry by default; it requires
`PROJECT_DIR`, `K2_SOURCE_SPEC`, `K2_SEGMENT_ROOT`, `K2_SEGMENT_COMMIT`.
Existing roots must match these identities. Its explicit `--execute` also
requires `K2_DEBUG_POLICY_CONFIRMED=yes`, retires the exact old pending
remainder and submits the gate plus automatic after-gate launcher.

Authorizations: `CANCEL ONLY REGISTERED PENDING K2 REMAINDER` and
`AUTHORIZE K2 SEGMENTED DEBUG CONTINUATION EXACT SPEC`. Record authorization
against the new spec. Use a clean pushed executor checkout while retaining
the old scientific checkout/root. This implementation supplies queue tooling;
native resume acceptance is still required remotely before science.
