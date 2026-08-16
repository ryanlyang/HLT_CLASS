# HCWDL-MHPE dense anchor-50 300k/60-pass campaign

Status: scientific implementation authority for one additive campaign profile.

## 1. Scientific question

This campaign tests whether a denser factorized homotopy and a
predecessor-anchored multi-horizon ensemble preserve more knowledge than the
existing five-stage MHPE graph. It changes the graph and ensemble reduction;
it is not presented as a recipe-only comparison.

The campaign reuses the authenticated unified-balanced 300k foundation:

```text
train       300,000
validation  100,000
final test  100,000 (sealed during ordinary execution)
```

Every fresh fit is cold-started, trains for 60 complete passes, validates
after every pass, uses unweighted `0.10 CE + 0.90 KD`, and selects macro AUC,
then CE, logR50, and earliest update. Specialist KD uses temperature 2. M1
uses temperature 1 with the same `0.10/0.90` coefficients.

## 2. Exact graph

`U000` is imported and is not retrained. The campaign owns 29 fresh fits:

```text
U033:  U000                                                    (1)
U066:  U000, U033                                              (2) -> U066E
U100:  U000, U033, U066E                                       (3) -> U100E
D75:   U000, U033, U066E, U100E                                (4) -> D75E
D50:   U000, U033, U066E, U100E, D75E                          (5) -> D50E
D25:   U000, U033, U066E, U100E, D75E, D50E                    (6) -> D25E
D0:    U000, U033, U066E, U100E, D75E, D50E, D25E              (7) -> D0E
M1:    D0E                                                      (1)
```

Every comma-separated teacher creates a separate specialist at the target
coordinate. Specialists within one stage run in parallel. Stages are
sequential. Exact coordinates are:

```text
U033 = V(1/3, 0)
U066 = V(2/3, 0)
U100 = V(1,   0)
D75  = V(1, 1/4)
D50  = V(1, 1/2)
D25  = V(1, 3/4)
D0   = V(1,   1) = exact HLT
M1   = exact HLT
```

`D0E`, every `D0_from_*` specialist, and M1 are HLT-only. Only HLT-compatible
models may enter sealed final evaluation.

## 3. Anchor-50 ensemble

For every ensemble, exactly half the probability mass belongs to the
specialist taught by the immediately preceding horizon. The other half is
divided equally among all skip-teacher specialists. For `n` components:

```text
local predecessor specialist: 1/2
each of n-1 skip specialists:  1 / (2(n-1))
```

At U066, this equals a uniform two-member ensemble. At D0, the local
`D0_from_D25E` specialist receives `1/2`; each of the six skip specialists
receives `1/12`.

For declared KD temperature `T`, each component publishes
`softmax(logits/T)` in FP32. Components are traversed in lexical node order,
multiplied by their exact rational weights, accumulated in FP64, normalized
once, and published as little-endian FP32 probabilities. Raw logits are never
averaged. Primary child KD consumes this anchor-50 probability target directly.

The reducer also computes a uniform validation ensemble from the same logits
as a diagnostic. Uniform targets are not durable and cannot supervise a child.
Component, anchor-50, uniform, leave-one-out, diversity, local-versus-skip,
and ensemble-minus-best metrics are reported. No validation metric changes a
weight or graph edge.

## 4. Artifacts, resources, and operations

This profile receives new versioned graph, node, recipe, campaign, training
report, target, stage-report, aggregate, finalist, completion, final-evaluation,
reuse-lock, and waiver semantics. Existing v1-v4 MHPE artifacts remain valid
and byte-semantically unchanged.

The completed 300k foundation supplies immutable U000/M0paired checkpoints,
U000 train logits, splits, selections, assignments, balanced couplings, and
the executable 60-pass HCWDL batching/optimizer recipe. U033 publishes one
authenticated train-logit bank for all direct U033 consumers. Every ensemble
publishes authenticated train and validation T1/T2 probability bundles.
Repaired particle views remain RAM-only.

Each GPU fit or reducer requests:

```yaml
CPUs: 8
RAM: 96G
Walltime: 06:00:00
GPU: gpu:gh200:1
```

CPU reporting jobs use 4 CPUs, 32G, and one hour. The ordinary graph has 38
jobs: 29 fits, six reducers, aggregate, finalist lock, and completion. It uses
the existing exact-ID cancellation, monitoring, checkpoint/resume, and
failed/downstream source/resource recovery machinery.

## 5. Acceptance and launch decision

Queue readiness requires exact graph/weight/coordinate tests, numerical
weighted-ensemble tests, target tamper tests, a complete 38-job nonmutating
dry run, recovery closure tests, CLI/help/compile checks, focused tests, the
complete repository suite, Markdown-link checks, and `git diff --check`.

At the user's direction this additive 300k campaign does not require another
standalone smoke. It carries the completed production-worker foundation,
installed-Weaver evidence, prior MHPE reducer/KD execution, and the current
300k campaign evidence. This is an explicit operational waiver, not a claim
that this new graph itself completed a smoke. Live submission still requires
a clean pushed commit, detached Tigris worktree, canonical dry-run ledger, and
the exact profile-specific creation and submission phrases. No job may be
submitted by local implementation work.
