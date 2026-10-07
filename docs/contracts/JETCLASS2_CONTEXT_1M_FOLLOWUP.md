# CONTEXT_V1 1M conditional science submission, v1

Authority: `../plans/JETCLASS2_CONTEXT_1M_FOLLOWUP_PLAN.md`. The operational
contract is `JC2_CONTEXT_1M_FOLLOWUP/v1`, schema_version=1, with canonical
content_hash, parent hashes, final_test_accessed=false and immutable atomic
publication. Kinds: authorization, review, submitted. Scientific artifact
versions and the original gate/source hashes do not change.

The caller binds gate path/hash, original scientific source commit and exact
preflight ID. The original pinned code authenticates GATE_SPEC/v11 and its
three-job live ledger/journal. The controller independently checks Oscar
`slurmctld`, `jc2ctxmg_` job names, current owner and `default` account.
Missing accounting is not success. Failed/cancelled/unknown terminal status,
COMPLETED with a nonzero exit, metadata drift or unavailable accounting stops
without submitting science. Waiting has a 14-day bound.

A separate executor checkout is source-validated and recorded; controller code
lives outside `cms_proxy_ladder/`. Scientific imports and CLI execution happen
in fresh subprocesses using the gate's original checkout and Oscar Python.
Workers retain the existing absolute environment helper, PYTHONNOUSERSITE,
Conda library path and original source enforcement. The launcher sanitizes
inherited Slurm submission variables in subprocesses via the reusable helper.

Without --execute the controller authenticates and prints its binding/policy,
without writing or waiting. With --execute it takes an exclusive Linux lock at
STUDY/context1m_followup/, publishes advance authorization binding both sources,
gate ledger/jobs and the bounded science policy, then prints ARMED. No claim of
being armed is made merely because a background process was started.

After all gate jobs pass, revalidate sources and gate evidence. The original
CLI must admit the real gate_complete/runtime profile, create or validate the
sibling science campaign, and materialize its canonical plan/dry ledger.
Require all 16 jobs, 9 fits/5 reducers, exactly 1M/250k rows, DIRECT/COARSE only,
Oscar gpu/default/norm-gpu, six CPUs/180000 MiB/one L40S for GPU jobs and the
measured time bounds. Review evidence binds the exact plan hash before the
original CLI's authorized live submission and site feasibility checks.
Verify the completed live ledger and per-job journal under the original code
before publishing submitted evidence. Submitted does not mean trained.

Any existing science ledger, live claim or nonempty submission journal stops
the controller before another live attempt. Its own incomplete live submission
is never retried; preserve all artifacts for exact-ID inspection. Exclusive
locks prevent concurrent controllers. No cancellation, source-worktree update,
artifact deletion, test evaluation, metric-based stopping or gate rerun.

This is an explicit opt-in continuation: after arming, no manual MODE=science
submission should be launched alongside it. The controller is a detached
process, not a Slurm allocation, and thus is not itself listed in squeue.
