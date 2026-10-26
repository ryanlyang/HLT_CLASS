# FullStat finalizer runtime-path retry v1

Extends the [finalization contract](LUKA_FULLSTAT_SPLITHEAD_FINALIZE.md).
The scientific objective, populations, selected checkpoints, native readback,
TRAIN exports, resource limits and twenty-epoch DE000 recipe are unchanged.

## Diagnosed integration error

User-supplied job 7963448 log at execution 420d22f8feb9f2237ee2dc2cbf3000d881ecd6a6:
original-source authentication succeeded, but worker.execution called the legacy
site allocator, which required environments/envs/atlas_kd_oscar rather than the
verified runtime_repair_v1/atlas_kd_oscar. FAILED/1:0 after 3m55s, before attempt
directory creation, validation readback, target joins, training or exports.
The initial local GPU worker tests mocked execution and missed this integration.

## Scoped runtime binding

New finalize_execution.execution applies ONLY to FINALIZE_SPEC and
FINALIZE_RETRY_SPEC. It independently retains the original exact site hash,
Oscar/L40S restriction, numeric Slurm allocation, scheduler account/partition/
QOS/cluster/node/task/CPU checks, resource bounds, exact requested CPU/memory,
architecture, CONDA_PREFIX, sys.prefix, no-user-site, one BF16 GPU, accepted
GPU identity, CUBLAS/cuDNN setting and installed-environment fingerprint.
The only substituted requirement is the explicitly bound replacement prefix.
Never spoof the old prefix, patch the global site profile or weaken original
campaign checks. Both finalizer and subsequent DE000 dispatch through this
same check. Old scientific/recovery workers retain the original allocator.

Donors: jetclass2_delphes/execution.py allocation and
luka_fullstat_attention/worker.py execution at 420d22f8. Those source files are
not modified. Extend the finalizer's 028c3314 source whitelist with the new
execution/retry modules and thin retry helper only.

## Explicit retry authorization and evidence

New LUKA_FULLSTAT_SPLITHEAD_FINALIZE_RETRY_{SPEC,EVIDENCE,PLAN,CLAIM}/v1.
Not an automatic retry, in-place resubmission or deletion of failed claims.
Authenticate the failed 420d22f8 spec, original-source checks, existing original
claim and exact accepted-job journal in its unmodified pinned worktree.
Require finalizer 7963448 FAILED/1:0, and its exact registered DE000 descendant
CANCELLED/0:0 with zero elapsed time. Never infer the descendant ID by adjacency.
Require the specific prefix-check traceback and no attempt/work/output dirs.
Bind the original 0c4a71a5 spec and DE033 evidence, failed spec/log/submission
files/claim and scheduler states into the successor. Active, ambiguous,
started or partial attempts fail closed without signalling jobs.

One immutable new root, with exactly two jobs finalize_DE033 -> train_DE000.
Explicit dry review/hash and AUTHORIZE_FULLSTAT_SPLITHEAD_FINALIZE_RETRY_V1.
Exclusive retry_after_7963448_v1/intent.json under the failed finalizer root
serializes successors across roots; preserve every old claim and ledger.
New receipts keep their existing FINALIZE_TASK/READBACK identities, bound to
the new retry spec and unchanged original fit evidence. No reused fit is
retrained, no target is regenerated, no test role is accessed.

The replacement remains scratch storage. This correction is not evidence
that repeated file disappearance cannot recur. Queue only clean pushed source;
original native/data checks still run before any successful publication.
