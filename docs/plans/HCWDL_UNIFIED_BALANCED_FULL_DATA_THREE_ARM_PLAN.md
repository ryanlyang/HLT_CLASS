# HCWDL Unified-Balanced Full-Data Three-Arm Plan

Status: **implementation-authoritative additive scale-up plan**.

Short name: **HCWDL-UB-FULL3**.

This plan scales the implemented HCWDL unified-root balanced homotopy study to
every mapped row in the authenticated Scouting split. It does not mutate the
running 300k six-arm HCWDL-UB campaign, reuse its selected checkpoints as
teachers, or reinterpret its results.

## 1. Scientific question and exact graph

The scale-up asks whether the most informative 300k loss recipes remain
effective when statistical uncertainty is reduced. It runs exactly three
independent factorized arms:

| arm | CE | parent KD | grandparent KD |
|---|---:|---:|---:|
| `C25P75` | 0.25 | 0.75 | 0.00 |
| `C10P90` | 0.10 | 0.90 | 0.00 |
| `C10P75G15` | 0.10 | 0.75 | 0.15 |

All three bind one freshly trained full-population foundation and run:

```text
shared/U000
  -> U020 -> U040 -> U060 -> U080 -> U100
  -> D80F -> D60F -> D40F -> D20F -> D0F -> M1F
```

Each arrow is logit KD from the immediate predecessor. Grandparent arms also
use the immediately preceding predecessor's predecessor when it exists; on
the first edge the unavailable grandparent allocation transfers to the
parent. Every model is cold-started. A direct `shared/U000 -> D100direct`
control is run in every arm. No joint path, legacy cost-CDF U path, or legacy
confidence-warped D path belongs to this scale-up.

The complete registry is 38 fits: two shared CE roots and twelve fits in each
of three arms. `D0F` and `M1F` consume exact HLT only. Other homotopy models
are privileged training-time models and are not deployable candidates.

## 2. All-mapped population

The source is the exact split manifest authenticated by the completed 300k
HCWDL-UJ preparation template. The 300k row selection is not reused. A new
selection deterministically includes every baseline-mapped train and
validation row. The sealed final-test population is recorded from split
inventory but is not selected, opened, assigned, coupled, cached, or evaluated
during ordinary work.

The specification stores exact integer counts derived from the authenticated
split inventory. Approximate labels such as `2.6M/1M/1M` are descriptive only;
they are never scientific identities. Assignment and coupling manifests must
cover every selected train/validation identity exactly once.

The preparation template contributes only immutable split/data provenance,
matcher resources and the previously authorized Shell endpoint semantics.
Full-population row selection, assignments, residual couplings, balanced
switch sidecars, U000/M0paired checkpoints, and U000 logits are rebuilt under
this campaign's own hashes.

## 3. Training budget

Every fit uses:

```text
20 natural-population passes
validation after every pass
macro-AUC, then CE, then logR50, then earliest-update checkpoint selection
unweighted per-jet CE
the locked AdamW/warmup-cosine/batching policy from the primary HCWDL recipe
homotopy KD temperature 2
M1 temperature 1 and fixed 0.25 CE + 0.75 parent KD
```

Twenty full-data passes are intentional. With roughly eight to nine times the
300k training population they expose the optimizer to substantially more jet
examples than a 60-pass 300k fit while avoiding a prohibitive full-data
60-pass bill. The 20-pass budget is a new recipe contract and may not be
silently substituted for the 300k recipe.

There is no performance early stopping. Poor finite science completes and
does not prune descendants.

## 4. Preparation and persistence

The foundation performs, in order:

1. authenticate the source-pinned preparation template and full split;
2. publish the all-mapped train/validation row selection and full-data recipe;
3. build and audit one assignment shard per source file, then lock manifests;
4. build train-only coupling scales, full train/validation residual bases,
   and deterministic balanced switch sidecars using the corrected linear
   prepared-endpoint implementation;
5. publish coupling, endpoint-equality, and measured-resource locks;
6. train fresh `U000` and `M0paired` roots;
7. publish one compact identity-ordered FP32 U000 train-logit cache;
8. publish the immutable foundation lock.

Durable reconstructed U/D particle datasets are forbidden. A training worker
builds its train and validation student views once in process-local RAM,
builds or loads each teacher target bank once, and replays those caches for all
20 passes. Ragged branches are prepared a fixed number of times per source
chunk, never once per row. Bounded workers emit chunks in canonical order.

## 5. Resources and scheduler layout

Preparation is shared. After its lock exists, the three arm campaigns have
separate specs, roots, ledgers, monitors, cancellation surfaces, and recovery
closures and may run concurrently. Each arm remains sequential along its
causal factorized path; its direct control may run in parallel.

The live submission registers one source-pinned CPU autolaunch job with an
exact `afterok` dependency on the foundation-lock job. That job creates the
three arm specifications and submits their independent ledgers only after the
lock authenticates. Its immutable receipt binds all three specs and ledgers.
This is operational orchestration only; it creates no cross-arm scientific
dependency.

Initial requests are conservative full-data envelopes and remain subject to
the mandatory resource-measurement gate:

```text
GPU training/targets: 8 CPUs, 256 GiB RAM, 24 hours, one GH200
CPU assignment/coupling arrays: 16 CPUs, 192 GiB RAM, 24 hours
CPU reducers/locks/reports: 4 CPUs, 64 GiB RAM, 4 hours
```

The endpoint/resource gate accounts for simultaneous train and validation
view caches, target banks, prepared in-flight chunks, model/optimizer state,
and allocator headroom. It fails before training if the measured projection
exceeds 75% of the locked request. Resource recovery may only increase CPU,
RAM, or walltime and may not change GPU class or scientific identity.

## 6. Lineage, recovery, and test boundary

Every artifact binds the exact source commit, split, all-row selection,
assignment/coupling parents, graph, recipe, coordinates, seeds, producer, and
logical/content hashes. Path existence never authorizes reuse. Workers use a
clean detached full-commit worktree and the absolute `${PROJECT_DIR}` helper
path, receive `USR1` 120 seconds before timeout, and resume from exact rolling
checkpoints including GPU-mapped RNG state.

Recovery is source-pinned and closure-exact. It preserves completed artifacts
and schedules only failed/downstream tasks. Cancellation accepts only exact
IDs from one bound ledger. Scientific source and resource changes require
separately classified recovery artifacts.

Validation aggregates rank the three arms without touching final test. A
separate finalist lock and explicit human execution lock are mandatory before
any final-test row is opened. Sealed evaluation consumes exact HLT only and
test metrics never select a model.

## 7. Acceptance

Implementation is queue-ready only after focused and full local tests, CLI
help and contract checks, shell/static checks, a complete nonmutating dry run,
source-drift/recovery tests, `git diff --check`, and a handoff update. Existing
HCWDL-UB production-worker evidence may establish worker semantics, but the
new all-row resource gate must still complete before its training jobs are
released.

No additional smoke is required for this graph-only thinning/scale-up. The
all-row endpoint/resource gate is the fail-closed production preflight; it is
not bypassed by the absence of a new smoke.

This plan authorizes implementation when requested. It does not authorize a
push, Slurm submission or cancellation, or final-test access.
