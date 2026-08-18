# HCWDL-MHPE Full-Data D066 Schedule Screen

Status: implementation-authoritative, validation-only, queue-ready after the
checks recorded in `docs/HANDOFF.md`.

## Scientific question

The existing full-data C25P75 MHPE campaign trains each specialist for 20
passes. The 300k/60-pass evidence suggests that closer teachers may benefit
from later optimization, but a 300k rerun cannot answer whether the full-data
students are schedule-limited. This study therefore trains on every
authenticated mapped full-data training jet.

At the exact D066 input, cross:

- passes: `20, 30, 40, 60, 80`;
- peak AdamW learning rate: `3e-4, 1.5e-4, 1e-4, 5e-5`;
- teacher: `U000`, `U050`, `U100E`.

This is 20 paired schedules and 60 independent fresh C25P75/T=2 fits. All
other optimizer, batching, warmup/cosine, data, repair, initialization,
sampler, and checkpoint-selection semantics come from the authenticated
full-data source. The shared seed alias pairs the three teachers within every
schedule and across schedules.

## Source and readiness

The source must be an all-mapped full-data `C25P75` MHPE campaign whose reuse
lock points to an authenticated `all_mapped_full3` foundation. Source campaign
completion is intentionally not required: the screen authenticates exactly
the immutable products it consumes—U000 and U050 selected reports/checkpoints,
their train-logit manifests, and the U100E/T2 probability bundle. Missing or
mismatched required products fail closed.

The 300k C25P75 source and every C10P90 source are forbidden. Final-test rows
and products are forbidden.

## Validation protocol

All mapped full-data train rows are ordinary-access training data. The full
mapped validation identity set is reconstructed from the authenticated
validation assignment shards without labels. A global SHA-256 identity rank
splits it into:

- the first `floor(N_validation/2)` identities for validation every pass and
  checkpoint selection;
- the remainder for one post-selection schedule score.

The subsets are disjoint, exhaustive, deterministic, and content hashed. The
scoring half never controls checkpoints, job continuation, or artifact
publication. Schedules rank by U100E held-out macro AUC, then CE, local-minus-
distant AUC, fewer passes, lower LR, and lexical ID. Scientific outcomes never
block completion.

## Runtime design

Each GPU fit requests 8 CPUs, 96 GiB, one GH200, and 72 hours. It constructs
the full train cache and checkpoint-validation half together under the
allocation-aware 72-GiB cache ceiling, trains, records the selected checkpoint,
then releases both caches. Only afterward does it construct the scoring-half
cache and evaluate the selected model. This phase separation prevents all
three full-data caches from coexisting.

All 60 fits are independent Slurm jobs and may run in parallel. Aggregate and
completion jobs depend on all fits. Recovery is exact failed/downstream
closure under the same source, graph, recipe, partition, seeds, and resources.

## Versioning and retired attempts

The unsubmitted v1 C10P90/300k and v2 C25P75/300k designs are scientifically
wrong for this question and retired. Executable full-data semantics use the
v3 contract family. Old roots, if any, are never reinterpreted.

## Acceptance

Before live submission: focused and full tests, all CLI help surfaces,
contract-version checks, Python compilation, and `git diff --check` must pass
from the exact pushed source. Creation performs a dry run first. Submission
still requires the explicit creation, operational-evidence-waiver, and live
ledger phrases. No new smoke or 300k prerequisite is introduced.
