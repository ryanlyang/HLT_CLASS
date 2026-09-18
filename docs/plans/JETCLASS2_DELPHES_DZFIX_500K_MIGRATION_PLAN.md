# JetClass2 Delphes 20260918 dz-fix 500k migration plan

Status: active staged migration; local capacity-planning code is implemented,
but no new bytes have yet been downloaded or authenticated and no old job has
been cancelled by this work.

## Scientific scope

The new producer directory is
`/eos/user/l/llambrec/jetclass/output_jetclass2_10M_20260918_dzfix/jetclass2`.
It retains the all-reconstructed-particle PUPPI-neighbourhood setting from the
preceding producer version and fixes HLT `dz` to be stored relative to the
primary vertex rather than the origin. It still uses PUPPI. A future no-PUPPI,
displacement-dependent tracking production is a different prospective dataset,
not part of this migration.

The target remains TRAIN_500K with 1,000,000 validation rows and a sealed
1,000,000-row final test. The salience matching foundations, salience U100
screen, selected matching lock, and direct/coarse/dense production campaign are
all rebuilt from scratch on the new snapshot. Old assignments, split identities,
matching locks, model weights, probability banks, and metrics are not reusable.
The model graph, C25P75/T=2 logit KD, optimization, feature schema, HLT-only
deployment boundary, and final-test seal otherwise remain unchanged.

## Why the partial snapshot is larger than 2.5M rows

The authenticated design first partitions complete files into approximately
60/20/20 train/validation/final-test reservoirs, then selects exact row
memberships inside each role. A 500k + 1M + 1M campaign therefore cannot be
supported by a 2.5M-row raw subset: each 20% evaluation reservoir must itself
contain 1M rows. The partial snapshot planner requires the existing four nested
training profiles through 2M, both 1M evaluation roles, and 5% capacity
headroom. Per-class feasibility is proved by constructing the actual outer
split, not inferred from total rows or a filename count.

## Frozen-snapshot stages

1. Download the currently available producer files into a new local directory.
   Production closure is not required. The completed local copy, not the live
   EOS directory name, becomes the candidate byte snapshot.
2. Run the established read-only inventory over every copied ROOT file. It
   hashes every file twice around latest-cycle schema/count inspection and fails
   if a byte changes, a ROOT file is truncated, schemas differ, or a path/content
   collision exists.
3. Run `plan_jetclass2_delphes_partial_snapshot.py`. The plan deterministically
   orders files within each producer source by a parent-inventory-bound SHA256,
   interleaves proportional source prefixes using exact rational comparisons,
   and takes the first prefix whose real seed-20260910 split clears every role
   capacity plus 5% headroom.
4. `transfer_jetclass2_delphes_partial_snapshot.py build` authenticates every
   selected source file and writes one deterministic uncompressed USTAR archive.
   UID/GID, names, mode and mtime are normalized; the archive itself is hashed
   and bound to the inventory and plan. Upload that archive and its compact
   manifests to a new SPORC dataset root.
5. The matching `extract` mode requires the exact archive hash and member
   coverage/order, rejects non-regular or altered members and path escapes,
   verifies every extracted file hash, refuses an existing destination, and
   publishes a completion receipt. An interrupted extraction remains visibly
   incomplete and must use a fresh destination rather than being trusted.
6. Re-run the inventory and split builder at the destination using the exact
   pushed source. `verify-destination` requires destination inventory and split
   hashes to equal the projected hashes, exact path coverage to agree, and all
   destination ROOT bytes/cycles/schema to verify.
7. Build the ordinary four-profile split registry and select TRAIN_500K. Only
   after this gate may preparation, salience foundations, and the salience U100
   screen be recreated.

The partial plan reads authenticated scalar inventory metadata only. It does
not read final-test particle arrays, construct matching, train, infer, submit,
or cancel jobs.

## Queue transition

Keep the old scientific jobs untouched during download, inventory, transfer,
destination verification, split-registry construction, preparation, and the new
salience screen. Once the new selected matching lock and production dry run are
authenticated, enumerate old jobs from the exact old submission ledger, prove
each candidate is still pending/running under that ledger, and cancel only those
exact IDs. Never cancel by name or a broad user query. Create the new production
under a distinct dataset/campaign root and pushed-source identity.

`transition_jetclass2_delphes_salience.py plan` implements that final boundary.
It authenticates both campaign specifications, the old live ledger, the new
canonical 30-task dry ledger and command plan; requires different inventories
and roots but an identical scientific graph and role counts; queries every old
ledger ID; refuses missing/unknown states; and prints only the exact active IDs.
It never calls `scancel` or submits the new campaign. After the printed command
is run explicitly, `verify-terminal` requires every old-ledger job to be in a
terminal state and writes a receipt before the new live submission is allowed.

## Implementation and tests

- `src/hlt_classification/jetclass2_delphes/partial_snapshot.py`: deterministic
  planning, replay validation, and destination binding.
- `scripts/plan_jetclass2_delphes_partial_snapshot.py`: create/inspect/verify CLI.
- `src/hlt_classification/jetclass2_delphes/partial_snapshot_transfer.py` and
  `scripts/transfer_jetclass2_delphes_partial_snapshot.py`: deterministic
  archive construction, safe extraction, content verification and receipt.
- `tests/test_jetclass2_delphes_partial_snapshot.py`: capacity, replay,
  insufficiency, tamper, and destination-parent checks.
- `tests/test_jetclass2_delphes_partial_snapshot_transfer.py`: reproducible
  archive, changed source/archive and extra-member rejection.
- `src/hlt_classification/jetclass2_delphes/salience_transition.py`,
  `scripts/transition_jetclass2_delphes_salience.py`, and their focused test:
  exact old-ledger cancellation planning and terminal-state receipt.

This code is additive. It changes no old inventory, split, campaign, matching,
or submission contract and imports no legacy donor.
