# FullStat DE033 post-export finalization v1

Execution-only extension of the [split-head plan](../plans/LUKA_FULLSTAT_SPLITHEAD_PLAN.md)
and [first recovery](LUKA_FULLSTAT_SPLITHEAD_RECOVERY.md), authorized by the
user's request to recover DE033 and run its DE000 descendant (2026-10-26).
No objective, seed, population, selection, model or training schedule changes.

## Evidence and scope

Exact parent: 0c4a71a55e8bc163ea3c4805b72f06d18e640cb5 recovery execution.
7938791 recover_DE066 completed; 7938792 train_DE033 finished all twenty epochs,
native fit publication and both native TRAIN exports, then FAILED/1:0 while
starting original-source authentication: Python could not load encodings.
7938793 train_DE000 was CANCELLED/0:0 with zero elapsed time. The parent DE033
TASK receipt is absent. These statements come from user-supplied logs; creation
must independently authenticate the actual remote files and accounting.

New namespace LUKA_FULLSTAT_SPLITHEAD_FINALIZE_<kind>/v1 (SPEC, EVIDENCE,
PLAN, CLAIM, READBACK, TASK). All retain sealed-role flags, canonical hashes,
exact source lineage and immutable atomic publication. No old artifact is
edited or relabeled as a successful failed job. The only parent-root addition
is an exclusive finalize_DE033_claim_v1 directory binding this execution.

Two afterok jobs: finalize_DE033 -> train_DE000. The first verifies the saved
native DE033 checkpoint with byte-identical validation predictions, authenticates
the original TRAIN bundle and both saved outgoing TRAIN arrays including ordered
identity/label joins, and writes new provenance receipts for imported DE066 and
finalized DE033. No optimizer, startup updates, retraining, target regeneration,
checkpoint reselection, or test access occurs in finalization. New DE000 uses
the unchanged twenty-epoch worker, seeds and original teachers/targets. Its
ancestor receipts bind the successful finalizer and original successful DE066,
not successful accounting for the failed DE033 job.

Original source is replayed in its clean exact pinned worktree. Bind successful
DE066 ancestry, exact failed/cancelled job states/owners/account/names, original
fit/report/predictions/startup/bundle, every target manifest/payload, and the
specific post-export encodings-failure log. Refuse any parent DE033 TASK receipt,
DE000 fit/attempt/bundle, partial target inventory, corruption or changed evidence.
Unknown/active/started descendants stop recovery; no automatic cancellation.

## Runtime, resources and storage

Only new wrappers activate the user-verified replacement:
/oscar/scratch/rlyang/hlt_classification/runtime_repair_v1/atlas_kd_oscar.
Require the exact resolved prefix and original environment fingerprint (including
Weaver Python sources). Original wrappers, worktrees and damaged environment stay
unchanged. This replacement is still scratch storage, NOT durable or purge-proof.
No chmod, sudo, symlink swap, package upgrade or environment modification by jobs.

Reuse original Oscar L40S, six CPUs, 180000 MiB and original per-stage walltime
ceilings. No separate preflight. Finalization's request inherits DE033's ceiling;
this is not a measured duration claim. DE000 retains integrated startup checks.
Only one new ~50 MiB selected-fit publication, small receipts/logs and temporary
serialization copies are written; old multi-GiB banks stay in place. Register
a conservative 2 GiB personal durable result-headroom minimum for THIS closure,
plus filesystem free-space checking. Old campaigns retain their 16 GiB rule.

Exact source boundary: 028c3314394949706787c6d1ef5e632f53ca157c. Only scoped
campaign/worker/submission/KD-locator dispatch changes and the new finalization
modules/wrappers are permitted. Scientific training/model/recipe files stay
byte-identical. Require clean pushed source, saved complete dry review, explicit
AUTHORIZE_FULLSTAT_SPLITHEAD_FINALIZE_V1 and exact plan hash. Claims and journals
refuse duplicate/ambiguous submissions, including across new roots.

Finite poor science never blocks a descendant. Any integrity, environment,
source, allocation or memory failure fails closed. Preserve partial outputs.
