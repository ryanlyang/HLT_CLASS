# HCWDL Unified-Balanced Full-Data Coarse Three-Arm Plan

Status: implementation-authoritative additive validation campaign.

This plan freezes the compact follow-up motivated by the fine-grained
unified-balanced pilot. It does not alter, cancel, reinterpret, or overwrite
the existing 300k or FULL3 campaigns. It asks whether repeated CE injection
across many adjacent distillation rungs is limiting retention and whether the
factorized or joint path is preferable when both have the same number of
fresh transitions.

## Scientific question

The imported root is the completed, CE-trained unified projected-offline
`U000` model. Canonical native `TOFF` is not a teacher in this campaign. The
same authenticated full-data foundation supplies all mapped train and
validation rows, assignments, residual coupling, balanced switch placements,
the `U000` and paired `M0` checkpoints, and the compact train-role `U000`
logits. No repaired particle dataset is persisted.

Three loss recipes are tested:

| Arm | CE | Parent KD | Grandparent KD |
|---|---:|---:|---:|
| `C10P75G15` | 0.10 | 0.75 | 0.15 |
| `C10P90` | 0.10 | 0.90 | 0.00 |
| `C25P75` | 0.25 | 0.75 | 0.00 |

At the first edge there is no grandparent, so any registered grandparent
weight is added to the U000 parent weight. All later grandparent supervision
uses the actual model two rungs earlier on its own input domain. Every fresh
model is cold-started, unweighted CE is used, KD temperature is 2, training is
20 full-data passes with validation every pass, and checkpoint selection is
macro-AUC, then CE, then log-R50, then earliest update. Poor metrics never
block descendants.

## Exact graph

Each arm contains two independent six-transition paths:

```text
factorized:
U000 -> U033 -> U067 -> U100 -> D67F -> D33F -> D0F

joint:
U000 -> J017 -> J033 -> J050 -> J067 -> J083 -> J100
```

Each arrow is one fresh KD-trained student. The displayed labels are rounded;
the immutable coordinates are exact rationals:

- `U033 = V(1/3,0)`, `U067 = V(2/3,0)`, `U100 = V(1,0)`;
- `D67F = V(1,1/3)`, `D33F = V(1,2/3)`, `D0F = V(1,1)`;
- `J017 ... J100 = V(t,t)` for
  `t = 1/6, 1/3, 1/2, 2/3, 5/6, 1`.

Thus factorized and joint each have exactly six transitions. Their transition
index shares initialization, sampler, dropout, optimizer, and repair seed
aliases across path and recipe. `U100` is exact D100. `D0F` and `J100` are
byte-exact HLT model inputs and are the deployable endpoints; no M1/self-KD
step is registered. There are 3 × 2 × 6 = 36 fresh fits.

The paths are compute/edge-count matched, not particle-count or wall-clock
identical. The comparison measures complete path geometry: factorized support
then feature change versus simultaneous support/feature change. It does not
isolate simultaneity from every possible resolution effect.

## Reuse and access boundary

The foundation may be reused only through a new authenticated reuse lock. The
lock binds the exact FULL3 foundation lock/spec/recipe/graph, all-mapped role
counts, assignment/coupling/endpoint/balanced hashes, U000/M0 reports and
checkpoints, U000 target manifest, endpoint projection source, and the
byte-identical model/view/training core. The old target artifact is not
mutated; the reuse lock authorizes the six first-edge consumers plus the two
second-edge U000-grandparent consumers in `C10P75G15`.
All views remain RAM-only. Labels do not enter coupling or switch placement.

Ordinary work reads train and validation only. Final-test rows, assignments,
metrics, logits, and labels remain sealed and no final-test task exists.

## Execution and failure semantics

Each arm has an independent ledger. Within an arm, `U033` and `J017` may run
concurrently, each path is sequential, and aggregation waits for all twelve
fits. The three arms may run concurrently. Workers are source-pinned, use the
Tigris environment, request one GH200, checkpoint on `USR1`, and emit task
attestations. Exact failed/downstream closure recovery preserves completed
artifacts and may only replay the same semantic source; a code repair requires
a separately versioned execution-repair contract.

The prior production-worker foundation is the required real operational
evidence. This additive graph introduces no preprocessing or view-builder
semantics and therefore does not require a new smoke. A full nonmutating dry
run, exact pushed source, clean detached worktree, authenticated foundation
reuse lock, and explicit live phrase are mandatory before submission.

## Primary reporting

Within each arm report every rung and the paired `D0F - J100` delta in macro
AUC, CE, accuracy, log-R50, and R50. Across arms, compare the two endpoints
using the paired transition seeds. Recovery relative to imported `M0paired`
and `U000` is descriptive because `U000` is the starting teacher, not an HLT
endpoint. One exploratory seed does not establish a final physics claim.
