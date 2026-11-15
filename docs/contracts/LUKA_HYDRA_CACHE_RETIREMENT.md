# Completed natural-baseline cache retirement and balanced Hydra recovery

Operational contract v1, 2026-11-15. Scientific authority remains
[the balanced campaign plan](../plans/LUKA_NEW_BALANCED_HYDRA_PLAN.md).

## Exact scope

This operation applies only to these existing executions below
`/cs/home/rlyang/hlt_u100_pairs`:

- Natural baseline: `runs/new_fullstat_offline_a865e5b6_r1`, completed train job
  `174643`, source `a865e5b65bc0271d1359644b15956d98f94caca3`.
- Disposable prepared inputs: `cache_new_fullstat_offline_a865e5b6_r1`.
- Balanced campaign: `runs/new_balanced_4a61a34c_r1`, source
  `4a61a34c97888d3ad5f479e29a5b94f9b6afa22a`, spec content hash
  `3b6fdd3e0dab78645fc7630575282bf7aa15c69aee5f66b31283abfcafc9b6d9`.

The user authorized retirement of the completed baseline's regenerable inputs
and recovery of the balanced campaign. Preserve raw data, scalar memberships,
cache metadata, all epoch states, selected models, predictions, reports,
source checkouts, environments and unrelated jobs. No recursive deletion.
No inference, final-test evaluation, new sampling policy or scientific fit
configuration is introduced by the operation itself. The user subsequently
requested cleanup first: `cleanup` has its own authorization token and stops
after retirement. It never invokes submission, command generation or scheduler
resource review. `recover` is the separate submission-capable action.

## Retirement records

Records live in the balanced root's
`operations/baseline_cache_retirement_v1/`. Each is immutable, atomically
published and canonically SHA256 hashed. New contracts are
`LUKA_HYDRA_CACHE_RETIREMENT_{SPEC,INTENT,COMPLETE,EXECUTION,EXECUTION_COMPLETE}/v1`.
All declare development validation, sealed final test and unchanged science.
Existing scientific v1 artifacts retain their original identities.

The SPEC binds the operational source snapshot/commit, both original specs,
baseline completion and output hashes, raw foundation, cache manifest and
preparation intent, exact disposable payload references, current quota
declaration, completed balanced stages and two named runtime overrides.

Before the first unlink, require:

1. Both original sources and the separate operational checkout are authentic;
   installed runtime matches the registered balanced environment.
2. Original accepted jobs are unambiguously terminal. The completed balanced
   OFFLINE passes its original full artifact validation. Natural OFFLINE has
   20 committed epochs, a completed publishing job, an authenticated selected
   model/predictions/report and consistent checkpoint selection.
3. Every original raw source and membership matches its saved checksum. These
   are opaque byte checks, with no ROOT decoding or sealed-row evaluation.
4. No other saved `*spec*.json` under `runs/` directly references this cache and
   no current user job declares the baseline cache, result or source path.
   This check cannot prove absence of arbitrary unregistered external readers.
5. Every registered block matches its bytes/hash and strict manifest path
   `blocks/NNNNN_NNNNN_{train,validation}.npz`. Refuse symlinks, hardlinks,
   foreign owners, escaping paths and unregistered files in `blocks/`.
6. For the combined `recover` action, Hydra accepts a test-only resource review
   of the replacement worker commands. Cleanup alone has no replacement jobs
   and skips that review, while retaining terminal-job/consumer checks.

Publish an INTENT listing the exact files before unlinking them. Partial
retirement can continue only with that same intent after repeating checks;
missing files without an intent are an error. Publish COMPLETE last. Reappeared
payloads invalidate the retired-cache credit. Preserve live claims and journals
on error. A Unix advisory lock prevents concurrent operational launchers.

## Storage and original-runtime adapter

The old quota guard assumes the natural baseline may grow to 17 GiB. Removing
its cache without this amendment would increase that reserve again. After
authenticated retirement, its **future write reserve is zero**. Preserve the
balanced 14 GiB result budget, subtract only original committed-output credit,
and check actual free space plus a conservative operator quota declaration.
The declaration is capped at its recorded pre-cleanup value; the adapter does
not invent additional quota from the amount deleted. Check again in each job.

The standalone CLI imports the operational module by path, then inserts only
the pinned original scientific checkout into Python's import path. Already
loaded science from another checkout is rejected. A scoped adapter replaces
only `campaign.storage` and `campaign.commands`. The latter swaps the last four
worker arguments for the new absolute wrapper and operational spec; resource
requests, DAG, job names, old work directory and original spec comments remain
identical. The originals are restored when the scope exits.

Original source/environment checks, submission intent/receipts, scheduler
allocation checks, population replay, teacher validation, integrated acceptance,
training, recovery and publication all remain active. Each replacement job
also publishes a separately source-bound operational execution record. Thus
scientific outputs keep the original source identity while the additional
execution record documents the storage/submission adapter that actually ran.

Do not resubmit OFFLINE. Expected remaining fits are M0HLT, CONCAT, UE100,
DE050 and DE000. The original journal handles their dependencies and any
authenticated epoch continuation. Ambiguous sbatch acceptance stops recovery;
this operation never guesses job IDs or cancels jobs.

## After retirement

Saved baseline metrics and prediction-based comparisons remain available.
Input-dependent baseline commands require regenerating the prepared inputs.
The old preparation fast path will reject missing blocks despite retained
metadata: rebuild into a new empty cache using the original source and raw
foundation, then explicitly plan any restoration or new input-consuming job.
Do not restore blocks during this balanced recovery or invoke the old launcher
whose reservation assumes an active baseline. The operational continuation
path, after journal inspection, preserves the retirement/accounting binding.

Focused tests cover deletion boundaries, corruption, interruptions, retained
outputs, active/external consumers, storage credit, original import isolation,
unchanged request/DAG generation and shell syntax. Real Hydra cleanup and
submission are not claimed by local tests.
