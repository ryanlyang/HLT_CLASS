# LUKA_FULLSTAT_EMBEDDING_ORDER_* /v2

Authority: [ordered handover plan](../plans/LUKA_FULLSTAT_EMBEDDING_ORDER_PLAN.md).
All new records use canonical JSON SHA256, schema2 and `final_test_accessed=false`.
Original scientific contracts/specifications are not versioned or modified.
Legacy schema1 `/v1` records remain immutable in `ordered_followup_v1`; new
records use `ordered_followup_v2` and explicit ORDER_V2 authorization.

- AUTHORIZATION_PLAN binds original gate/followup/armed hashes, unchanged
  preflight ID, stopped process host/UID/PID/start-ticks/argv, clean pushed
  helper commit/file hashes, exact ordered ceiling and maximum requests. V2
  additionally binds the successfully probed pidfd backend and either no prior
  attempt or the exact known pre-signal v1 plan/authorization/failure hashes.
- AUTHORIZATION and RETIRED record the explicit reviewed handover; retirement
  uses a pidfd, not an unverified numeric PID or a Slurm cancellation.
- PLAN binds the original measured plan hash and exact eight commands. The only
  modifications are command ordering and six `afterany` scheduler edges. The
  original aggregate still requires success of all seven fits.
- ORIGINAL_SUBMIT_DISABLED marks the permanent old science live-claim; it is
  deliberately not an original independent-plan submission ledger.
- MEASURED_REVIEW binds authorization, original science spec and measured
  ordered plan; original receipt/resource/source validation must succeed first.
- Standard exact-DAG journals use the ordered PLAN hash as identity, recording
  every accepted numeric job ID and resolved command before proceeding.
- COMPLETE means all eight jobs were accepted, not training completion.
  FAILURE preserves diagnostic evidence and prohibits blind retries.

The original controller cannot remain live during submission. An existing
science directory, unexpected process, changed source/authorization, invalid
technical receipt or ambiguous/partial submission fails closed. No test access,
scientific performance gate, dataset modification or automatic artifact deletion.

V2 accepts an earlier v1 failure only from helper commit
`b356267787a1ce543db6d327781008315608b12c` with exact RuntimeError
`Linux Python with pidfd required; no unsafe PID fallback`, matching stopped
controller identity, gate/armed parents and requested order, valid canonical
plan hash and reconstructed authorization/failure records. The directory must
contain only those three records and an empty live claim, without symlinks.
This is a separately reviewed recovery, not automatic retry or reuse of a claim.

Both native Python wrappers and the Linux x86_64 LP64 syscall adapter retain
kernel pidfd identity protection. Unsupported ABI/kernel fails during dry review
before creating a claim. Retirement rechecks process identity after opening the
descriptor and verifies process exit before any science submission can proceed.
