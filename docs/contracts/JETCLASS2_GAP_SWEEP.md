# Baseline-gap screen contracts

[Scientific authority](../plans/JETCLASS2_GAP_SWEEP_PLAN.md).

New namespace `JC2_GAP_SWEEP_{KIND}/v1`: RECIPE, SOURCE, SPEC, PLAN, PREFLIGHT,
INPUT, DIAGNOSTICS, FIT, SUMMARY, CLAIM. No existing schema or source-pinned implementation is
modified. References include bytes/SHA-256; all payloads have content and
parent hashes and atomic publication. Source transfer demands identical bytes
for every parent scientific source file; new code lives in its own package.

SPEC pins exact historical campaign, OFFLINE/M0HLT completed-task reports,
population foundation, complete CE recipe, candidates, selection rule and
source. Revalidation rejects corrupt/stale parents even on reuse. Test roles
are rejected before reading particle metadata. Labels are only consumed by
ordinary-role CE training/evaluation, never generation. No native HLT arrays.

RAM caches retain ordered identities and labels. Candidate foundation identity
binds the parent foundation plus new recipe/name. Stream scheduling, source
chunking, serial/process generation and readback cannot alter per-jet response.
Diagnostics are train-only and marked development-tuned. No physical output
dataset is claimed; exact frozen candidate replay still needs parent particles.

One technical GPU preflight is dry-reviewed/live-submitted first. Its successful
receipt is required to create the four-job science plan (3 independent CE fits,
summary afterok all 3). Both modes need exact reviewed plan hash, explicit live
authorization, clean pushed source, site checks and exclusive journaled claim.
No cancellation, in-place partition mutation, silent resume or auto-followup.
Actual allocation and installed environment must match measurement. Measured
walltime/GPU/RAM failures block execution, not scientific metric thresholds.

Every required fit/report is authenticated before selection; unavailable fits
are not a license to select from a subset. Selection recomputes from raw accuracy
using the registered open lower/closed upper interval in percentage points;
0.01 means 1 percentage point, not 1% relative change. Selection result is
development-only. All raw reports, no-winner outcomes, and original baselines
are preserved. Final test remains sealed.
