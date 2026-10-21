# LUKA_FULLSTAT_EMBEDDING_ORDER_* /v1

Authority: [ordered handover plan](../plans/LUKA_FULLSTAT_EMBEDDING_ORDER_PLAN.md).
All records use canonical JSON SHA256, schema1 and `final_test_accessed=false`.
Original scientific contracts/specifications are not versioned or modified.

- AUTHORIZATION_PLAN binds original gate/followup/armed hashes, unchanged
  preflight ID, stopped process host/UID/PID/start-ticks/argv, clean pushed
  helper commit/file hashes, exact ordered ceiling and maximum requests.
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
