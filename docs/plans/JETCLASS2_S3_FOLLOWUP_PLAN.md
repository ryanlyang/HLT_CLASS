# One-time automatic S3 science continuation

## Authorization and precedence

On 2026-10-08 the user explicitly requested the pending S3 comparison to launch
after its preflight succeeds. This operational addendum supersedes **only** the
manual-follow-up boundary in [the frozen S3 plan](JETCLASS2_S3_LADDER_PLAN.md).
It does not change the scientific recipe, source or immutable SPEC/PLAN v1
`automatic_followup=false` fields. A separate authorization records the newly
requested continuation. All other scientific and hardware gates remain required.

User-supplied pending job: SPORC `21826026`, `jc2s3_preflight`, debug/A100.
The controller discovers exactly one matching live ledger; it authenticates
that ledger, its exact-DAG journal and the regenerated original gate plan.
No assumption that a matching name or existing path proves identity.

## Execution boundary

The seven-job science plan needs measured train/reducer walltimes from the
preflight. Therefore it cannot yet be materialized as seven ordinary `afterok`
jobs with the current scientific implementation. A separate clean, pushed
controller worktree runs a one-time `nohup` process on SPORC, checking accounting
every 60 seconds for up to 14 days. It does not change or resubmit the preflight.
SSH disconnects do not stop it; host reboot/process termination can.

Require exact job ID, name, user, account, partition and QOS; wait on missing
accounting or registered active states. Require `COMPLETED` with `0:0` before
continuation. Failed/cancelled/timed-out/OOM/nonzero or unrecognized terminal
states stop without science submission. Reauthenticate the controller, old
source, spec, gate plan and journal after waiting. The saved genuine preflight
must pass the original S3 validator and name this exact Slurm job.

Then run the **original pinned** S3 CLI in a separate interpreter to produce
the full canonical science dry run. Automatically review the precise graph,
measured resource requests and scope against the user's bounded authorization:

- Same 100k train/50k development validation, frozen S3 and OFFLINE controls.
- Four fresh KD fits, two reducers, one summary; no U000/dense/new baseline.
- DIRECT D000 and COARSE D066 start independently, then the registered coarse
  D066 reducer -> D033 fit -> D033 reducer -> D000 fit. Summary follows all fits.
- Debug, reu-aisocial/qos_tier3, A100/6 CPUs/90000 MiB for GPU jobs; the original
  one-CPU/16000-MiB/60-minute summary. Measured GPU limits at most 24 hours.
- No dataset generation, final-test access, cancellation or unrelated changes.

Publish the authorization and exact-plan review, then invoke the original
submitter once with the reviewed content hash. Its site feasibility checks,
canonical dry ledger, exclusive live claim and per-job durable journal remain
authoritative. Authenticate the complete live ledger **and journal** afterward.
A partial/ambiguous live submission is not automatically resumed. Re-running an
already completed controller authenticates its receipts and submits nothing.
This is automatic submission of the original DAG, not an extra scientific run
or a new Slurm job depending on an old historical ID.

## Verification

Local tests must cover dry-only nonmutation; accounting identity/failure/missing
states; exact original gate ledger/journal binding; complete success and reuse;
source/spec changes during waiting; acceptance job mismatch; graph/resource
expansion; interrupted submission; duplicate controller exclusion; and corrupt
receipts. No new model/data semantics or new GPU acceptance is introduced here.
The genuine pending S3 preflight remains mandatory. New source is not yet
deployed merely because local tests pass.
