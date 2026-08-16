# HCWDL-MHPE C10P90 Parallel Full-Data Plan

Status: **implementation-authoritative additive recipe ablation**.

Short name: **HCWDL-MHPE-C10P90-FULL**.

This plan authorizes one parallel companion to the immutable
[HCWDL-MHPE-FULL campaign](HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_IMPLEMENTATION_PLAN.md).
It asks whether the primary campaign's repeated 25% cross-entropy contribution
suppresses teacher-horizon information before the same-domain ensembles can
benefit from it.

The companion changes exactly one scientific variable:

```text
primary non-M1 specialists:   0.25 CE + 0.75 KD, T=2
parallel non-M1 specialists:  0.10 CE + 0.90 KD, T=2
```

M1 remains exactly `0.10 CE + 0.90 KD, T=1`. This keeps final compression
fixed and makes endpoint differences attributable to the upstream specialist
recipe. A later M1-only compression sweep may reuse the frozen D000E target;
it is not part of this campaign.

Everything else is exactly paired with the primary campaign:

- the same authenticated all-mapped FULL3 foundation and exact identities;
- imported U000 and M0paired reports, checkpoints, and target lineage;
- exact U050, U100, D066, D033, and D000 coordinates;
- the same 16-fit teacher lattice and four uniform probability ensembles;
- the same coordinate-specific initialization, sampler, dropout, optimizer,
  validation-order, and repair seed aliases;
- fresh initialization, 20 complete passes, every-pass validation, macro-AUC
  checkpoint selection, optimizer schedule, batching, and numerical policy;
- identical ensemble components, uniform weights, T1/T2 target semantics,
  reports, finalist registry, access boundary, and recovery behavior;
- exact-HLT-only D000 specialists, D000E, M1, and sealed finalists;
- no ordinary final-test access and no new smoke or 300k prerequisite.

The campaign receives a separate source-pinned worktree, root, graph/recipe/
campaign identities, command plan, ledger, reports, recovery closure, and
explicit authorization phrases. It never mutates, resumes, or imports outputs
from the concurrently running C25P75 campaign. Job names use the `hcwmhpe90_`
prefix so the two ledgers remain operationally distinguishable.

Versioned semantic identities are:

```text
HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_GRAPH/v2
HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_NODE_SPEC/v2
HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_RECIPE/v2
HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_CAMPAIGN_SPEC/v2
HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_TRAINING_REPORT/v2
HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_OPERATIONAL_EVIDENCE_WAIVER/v2
```

Unchanged reusable probability, reporting, lock, final-evaluation, task-
attestation, submission-ledger, monitor, and recovery structures retain their
existing contracts while binding the distinct v2 graph, recipe, and campaign
hashes.

The primary paired comparison is C10P90 minus C25P75 at every specialist,
ensemble stage, D000E, and M1. Poor finite performance never blocks the graph.
If C10P90 improves horizon disagreement, ensemble gain, D000E, or M1, that is
evidence that repeated CE injection was a bottleneck. If it does not, the next
diagnostics are uniform stale-teacher drag, insufficient diversity, transition
size, optimizer duration, and M1 compression—not a post-hoc rewrite of this
campaign.

Queue readiness requires the same local tests, full suite, compilation,
contract/help/static checks, clean pushed commit, detached Tigris worktree,
authenticated foundation reuse, canonical nonmutating dry run, and explicit
live phrase as the primary campaign. At the user's direction, no additional
Slurm smoke or reduced-row pilot is required.
