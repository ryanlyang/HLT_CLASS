# LUKA_SIXCLASS_FULL_RECOVERY/v1

Authority: [analysis plan](../plans/LUKA_SIXCLASS_FULL_RECOVERY_PLAN.md).
Implementation: `operations/sixclass_full_recovery.py`.

All artifacts have canonical content hashes, schema1, policy and parents.
Original scientific artifact families remain unchanged.

* SPEC binds both campaign hashes, clean analysis and scientific source
  snapshots, runtime, pooled validation identity digest and class counts,
  selected anchor result hashes/metrics, original20 and snapshotted new epoch
  receipts. Each original receipt retains job, state hash and teacher ancestry.
* EPOCH binds analysis and original epoch-commit hashes, family/epoch, full
  macro/binary metrics, reproduction evidence, analysis job and duration.
* REPORT binds SPEC and publishes both fixed anchors, complete snapshot epoch
  rows, population evidence and new epoch count, immutably and completion-last.

Anchor prediction IDs/labels/pT must match registered in/out rows. Validation
has566032 unique32-byte IDs with exactly the declared class counts. Payloads
require checksums. Invalid probabilities, corrupt states and source/runtime/
ancestry drift fail closed. Scientific scores and negative recovery do not.

Resume authenticates source/results again, verifies the saved epoch prefix,
and cannot silently incorporate later training epochs. Old selected and new
first epochs repeat to qualify the current allocation; other saved receipts
are authenticated and reused. A Linux advisory lock prevents concurrent
analysis and releases on exit. Queue commands are journaled before submission;
duplicate launches stop, explicit resume requires the prior exact job terminal
and unsuccessful, and ambiguous submissions retain all records.

The CLI imports science from the actual lower-LR execution checkout and loads
the analysis adapter separately. Historical teachers authenticate in original
source; reused historical modules are pinned by the lower-LR donor manifest.
Selected prediction replay, per-epoch in-range metrics and new saved full-Hbb
diagnostics must reproduce exactly. No final-test inference or reselection.
