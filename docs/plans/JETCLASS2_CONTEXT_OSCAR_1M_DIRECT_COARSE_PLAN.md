# Frozen CONTEXT_V1: Oscar 1M/250k DIRECT + COARSE

Status: implementation; remote admission still required. This plan supersedes
no existing execution. The 100k/50k campaign and any other recipe remain intact.

## Scientific scope

Use the already completed, relocated CONTEXT_V1 dataset, manifest
`2ec2c933a37f50bb85e83577b9c1cf9d31c35a6e0be42089f0088ae62c485efb`.
Consume all 1,000,000 existing train rows and all 250,000 existing validation
rows, with their original identity-to-offline joins and split membership.
No resplit, regeneration, stronger noise, class balancing, or subsampling.
Final-test particles, inference and selection remain sealed.

Retain the 100k experiment's model, input encoding (including native mm),
capacity 512 without truncation, salience assignments, coordinate construction,
initialization/sampler seeds, CE/KD recipe, pass/selection rules and recovery
definition. Retrain all nine models; reuse no old checkpoints or probabilities.
Controls: M0HLT, pure OFFLINE, U000. DIRECT: U000 -> D000. COARSE:
U000 -> U050 -> U100 -> D066 -> D033 -> D000. Five probability reducers and
aggregate/complete give 16 science jobs. No DENSE branch. Deployable D000
consumes proxy inputs only. Intermediate/offline models are declared oracles.

This is a synthetic context benchmark, not genuine or validated CMS HLT.
The expanded validation population differs from the 50k run; compare DIRECT
and COARSE on this campaign's same rows, not as a strictly paired old/new score.
Poor performance is retained and never stops a registered science row.

## Execution and admission

New clean pinned worktree and fresh gate/science roots on Oscar. Use existing
`oscar_l40s` environment/site, six CPU workers and one L40S for GPU tasks.
Foundation/preflight/science request 180000 MiB (below the site's 192000 cap).
Authentication requests four CPUs/32000 MiB/two hours; foundation six CPUs/
180000 MiB/eight hours; GPU preflight six CPUs/180000 MiB/twelve hours.

Full-version cache preparation uses one source file per task, at most six
ordered tasks in flight. Budget resident features at capacity 512, bounded
worker/IPC copies and assignment/index overhead; do not rely on mean particle
counts, spill full views, or silently reduce the population. Old cache behavior
is unchanged. Persist only authenticated release/assignment banks and ordinary
training products, not materialized ladder views.

The real full-population preflight checks installed-Weaver forward/gradient
parity, one CE pass and one KD pass, inference, GPU memory and U000/D050 cache
preparation. These are acceptance passes, not science checkpoints. Derive fit
walltime from measured cache + 100 times the slower pass, with factor 1.75;
cap at 48 hours. Reducers use measured cache/inference, capped at 24 hours.
Wrong population, insufficient RAM, corrupt bytes, stale source, missing gate,
invalid metrics and resource violations fail closed. No optimistic fixed
walltime extrapolation from the 100k run admits science.

Gate and science each require a saved exact dry plan, its hash, pushed source,
Oscar scheduler feasibility checks and live authorization. No implicit
follow-up, replacement, cancellation or duplicate submitter. Interrupted live
submission preserves the exclusive claim/journal for explicit inspection.

## Tests and handoff

Test full-vs-small version isolation, all ordinary rows, deterministic joins,
sealed test, exact D000/OFFLINE input parity, bounded serial/process caches,
unchanged graph/seeds/recipe, real measurement-path wiring, resource/profile
and authorization rejection, and the 3-gate/16-science dry DAGs. Remote native
preflight is required before full science; local synthetic tests are not that
evidence. Queue instructions: `docs/JETCLASS2_CONTEXT_OSCAR_1M_RUNBOOK.md`.
