# CORR_MID automatic tier3 follow-up: 2026-10-07 operational amendment

The user explicitly authorizes automatic full science submission after the
existing debug gate succeeds, with science on **tier3**, not debug. This active
amendment supersedes the debug-only execution choice in the [original plan](JETCLASS2_CORRELATED_TRACKING_PRODUCTION_PLAN.md),
not the scientific comparison. That old plan/contract is source-hashed by the
running gate and must remain byte-identical; this separate amendment resolves
the conflict explicitly. Existing jobs 21820095/21820096/21820097 and their source
`062713ead3e00721d9f3cb7073e88aef83ebd325` stay unchanged. Runtime state is checked
against their authenticated ledger/journal; reported IDs alone are not proof.

A separately pinned controller waits for all three exact gate tasks to be
COMPLETED with exit 0:0, then authenticates gate completion, installed-Weaver
parity, full selected-population CE/KD acceptance and measured resources. Missing
accounting waits; failed/cancelled/ambiguous jobs or corrupt artifacts stop it.
No gate cancellation, rerun, preflight bypass, or unmeasured walltime is allowed.

Create the original v5 scientific specification as read-only measurement lineage
without submitting it. A separate `science_tier3` v6 campaign explicitly imports
that foundation, population, scientific plan, recipe, seeds and reports. The
debug profile remains unchanged; a versioned transferred execution profile names
both the original measurement and tier3. The same A100 identity/environment,
6 CPUs, 90000 MiB, workers, cache limits and measured times remain mandatory.
No extra allowance above the old 24h bound is introduced.

All measured source files, including the old plan/contract, must match the new
executor byte-for-byte, except precisely the three-line v6 validation dispatch
in production.py. The new tier3 adapter is the only added scientific-snapshot
module. Source pins and an authenticated transfer record make this narrow
execution-only exception explicit; runtime code patching or merely editing
sbatch partition is forbidden. No data/model/generator changes are authorized.

The user authorizes a bounded deferred dry review: after profiling, construct and
save the complete exact 11-job plan and canonical dry ledger, validate it against
the preauthorized 100k/50k DIRECT+COARSE tier3 policy, run read-only site probes,
and submit that same hash once with durable claims/journals. A Linux exclusive
controller lock prevents concurrent controllers; ambiguous partial submission is
never automatically retried. All internal science edges retain afterok. The
controller survives SSH loss but is not itself a queued training job. No promise
that tier3 will start faster is made. Test evaluation stays sealed.

See the [transfer contract](../contracts/JETCLASS2_CORRELATED_TIER3_FOLLOWUP.md)
and [arming runbook](../JETCLASS2_CORRELATED_TIER3_FOLLOWUP_RUNBOOK.md).
