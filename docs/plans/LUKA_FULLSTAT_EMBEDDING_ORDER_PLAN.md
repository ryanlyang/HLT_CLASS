# FullStat embedding sweep: scheduling-only ordered handover

Authorized order on 2026-10-21: **50, 500, 1000, control (0), 100, 10, 2**.
This additive operational plan overrides only the original independent Slurm
scheduling policy, not the scientific dependencies, immutable gate/specification,
source, acceptance requirements, resources, inputs, targets, seeds or fixed20
training. Original [scientific plan](LUKA_FULLSTAT_EMBEDDING_SWEEP_PLAN.md) applies.

The user paused original submit-host Python controller PID300411 on login010;
it was confirmed `Tl` with no science directory. Existing preflight7690397
continues. The replacement authenticates its original gate journal and original
FOLLOWUP plan/authorization/armed records. It requires that exact same process
remain stopped, owned by the user on its original host. Its argv and process
start ticks enter a separately reviewed authorization plan. On explicit execution,
Linux pidfd binds retirement to the verified process; only it receives TERM/CONT.
No preflight, training or other controller jobs are cancelled.

The replacement waits with original accounting logic, then uses original clean
`fd687181a60c1557f98d7498f577e26d6dbb35a7` code to authenticate all completed gate
evidence and create the exact original science specification. Failed, cancelled,
missing, nonfinite or corrupt acceptance cannot launch science. There is no new
GPU preflight, copied cache, recreated embedding bank or scientific code change.

Order is enforced using scheduling-only `afterany` edges between independent
fits; an unsuccessful weight does not cancel later weights. Aggregation retains
its original `afterok` dependency on all seven fits. No metric threshold is added.
Serialization can increase time to complete the whole sweep relative to parallel
execution; it changes the order of availability, not model initialization.

Keep separate `ORDER_*/v2` provenance under `ordered_followup_v2`, including
the original-plan hash, tool-source identity, retire record, measured dry review,
exact command plan, immutable accepted-job journal and completion/failure record.
Never rewrite original followup records to pretend the independent plan ran.
Claim the original science submit path to prevent accidental manual duplicate
submission; workers/results still use the original scientific spec/source.
Claims are permanent. Partial submissions require manual inspection; no retries,
deletion, resumption of the old controller or adoption of an existing science root.

### Scoped recovery from the v1 portability failure

The b356267787a1ce543db6d327781008315608b12c helper failed before sending any
signal because the installed Python lacks one or both native pidfd wrappers.
Preserve `ordered_followup_v1` in place. A separately authorized v2 attempt may
proceed only when that directory contains exactly its valid plan, authorization,
the exact missing-pidfd failure and an empty claim; controller identity and all
original gate/armed bindings must still match. Any retirement, submission,
unexpected files, different failure or changed process blocks this recovery.

V2 probes kernel pidfd capability during dry review, before any permanent claim.
It prefers Python's wrappers and otherwise uses the native Linux x86_64 LP64
syscalls (pidfd_open 434, pidfd_send_signal 424), with explicit ABI checks and
errno handling. It never falls back to numeric-PID signaling. The dry probe
opens its own process descriptor and checks signal 0 without delivering a signal.
Source for syscall identifiers: the
[Linux v5.14 x86_64 syscall table](https://raw.githubusercontent.com/torvalds/linux/v5.14/arch/x86/entry/syscalls/syscall_64.tbl).
This changes only handover portability and records, not the training environment
or original scientific contracts. No environment upgrade or GPU preflight rerun.

The newly committed/pushed helper worktree stays separate from the measured
worktree. It explicitly bootstraps scientific imports from the latter. Dry review
signals no process and submits no jobs. Require explicit order-plan hash and
authorization before retiring the controller and arming automatic submission.

Donor: the unchanged `luka_fullstat_embedding_sweep/{campaign,contracts,followup,
submission}.py`, `luka_fullstat/submission.py` and `scouting/hcwdl_exact_dag_submission.py`
at fd687181 above. Local failure-injection and exact-DAG tests are required;
original real Oscar preflight acceptance remains the actual resource gate.
