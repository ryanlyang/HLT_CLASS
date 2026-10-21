# Trial embedding sweep: tier3 to debug transfer

Authorized 2026-10-21: move the eight pending trial `lukaes` weight fits to
SPORC debug and cancel the remaining jobs of the specified `lukaag` alternating
campaign. Do not touch Fresh_check, evmsg, other campaigns, datasets or results.
This extends the [sweep plan](LUKA_EMBEDDING_SWEEP_PLAN.md) operationally; its
original tier3-only contract and immutable specifications remain unchanged.

Reuse the accepted eight-weight preflight through its original clean pinned
source, exact gate accounting, full technical receipt and payload validation.
The new source must preserve every measured scientific runtime file. Only the
registered same-SPORC/A100/environment/resources partition transfer is allowed:
tier3 -> debug, same account/QOS, one A100, 6 CPUs, 90000 MiB and each measured
walltime <=24h. GPU identity and installed environment are checked at runtime.
No repeated GPU preflight, shortened training, recipe change or relaxed check.

Keep weights 3,5,20,50,100,200,500,1000, CE plus weighted embedding prediction
without KD, identical frozen OFFLINE targets, seeds, native HLT inputs,
batch128/fixed30, 299234/74895 split and validation selection. Original50k is
report-only; there is no independent test. Poor results never block an arm.

Register separate DEBUG/v1 SPEC/PLAN/SCIENCE/RESULTS, under the original gate's
`science_debug` directory. Original FIT records still bind the measured ARM.
Reuse the original 2 GiB allowance and shared inputs/target bank read-only.
No deletion, copies of particle arrays, old-spec edits or automatic retries.

Before replacement submission, authenticate the original exact eight-job journal
and require all original fits cancelled without ever starting and without arm
output directories. If an original starts, stop for a separate decision; never
discard its progress automatically. A completed automatic follow-up must have
published its matching completion record before transfer.

Cancellation is a separately reviewed, exact-ID operation bound to the user's
two known scientific spec hashes. Review scheduler identity (owner, account,
partition, QOS, job name, comment, worker, work directory and log), not a name
substring or numeric range. Preserve completed alternating fits; cancel only
its pending/running jobs. The running alternating fit loses unfinished work.
Recheck immediately before each cancellation. Record a permanent intent before
the first mutation; ambiguous/partial outcomes require inspection, not retry.
Use two bounded exact-ID controller requests, with a server-side PENDING
filter for the sweep; never cancel a newly started sweep fit deliberately.
Submission independently requires accounting confirmation for all eight old
sweep jobs and the cancellation receipt. Missing accounting cannot admit it.

The helper first registers and dry-reviews all eight debug commands, then
separately reviews/confirms cancellation, and finally asks for explicit debug
submission authorization. No guarantee of immediate debug availability.
Local tests cover original-source replay, unchanged scientific execution,
source/receipt drift, wrong job identity, started-original refusal, exact
cancellation scope, partial/uncertain submission and compact results. Local
fixtures are not a new remote resource measurement.
