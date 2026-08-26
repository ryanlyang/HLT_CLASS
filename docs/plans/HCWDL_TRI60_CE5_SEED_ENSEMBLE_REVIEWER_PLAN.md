# HCWDL TRI60 CE5 seed-ensemble reviewer study

Status: implementation authority for the additive full-data reviewer control.

## Scientific question

This study tests the alternative explanation that the observed endpoint gains
do not require homotopy-structured teachers and could instead arise from an
ordinary random-seed ensemble followed by knowledge-distillation compression.
It is separate from, and cannot alter, the running TRI60 or dense-extension
campaigns.

## Fixed graph

Five ordinary exact-HLT Particle Transformers are trained independently:

```text
CE5_S01 --\
CE5_S02 ---\
CE5_S03 ----+--> CE5E --> CE5_KD
CE5_S04 ---/
CE5_S05 --/

CE5_CONTROL  (paired CE-only student)
```

`CE5_S01` through `CE5_S05` are fresh, CE-only teachers. `CE5E` is their
uniform class-probability ensemble. `CE5_KD` is a fresh exact-HLT student
trained with C10P90 and temperature one from `CE5E`. `CE5_CONTROL` is a fresh
CE-only student with the exact same initialization, sampler, model, data,
optimizer, schedule, and checkpoint-selection seed alias as `CE5_KD`.

There are exactly seven fresh fits, one reducer, one aggregate, and one
structural completion report. No validation metric selects a seed, weight,
teacher, loss, or descendant.

## Models and training

Every fit uses:

- the ordinary unified 21-channel, 200-token HLT Particle Transformer;
- exact canonical HLT particles only;
- all authenticated mapped train and validation rows from the source TRI60
  foundation;
- fresh initialization;
- 60 natural-population passes and validation after every pass;
- effective batch size 256;
- the source TRI60 AdamW, cosine schedule, warmup, BF16-forward/FP32-loss,
  deterministic-backend, and AUC-first checkpoint-selection semantics;
- no rolling resume or partial checkpoint reuse; and
- selected/final checkpoint publication only after all passes complete.

The five teacher aliases are fixed, distinct members of
`HCWDL-TRI60-CE5-REVIEWER/v1/teacher/{01..05}`. The two student fits both use
`HCWDL-TRI60-CE5-REVIEWER/v1/student_pair`, which makes initialization,
training RNG, and sampler order identical. The teacher ensemble is fixed
before metrics and includes all five seeds with exact weight 1/5.

## Ensemble and distillation semantics

For each teacher and each authenticated train/validation row, the reducer
computes FP32 logits. It applies temperature-one softmax independently,
accumulates the five probability vectors in lexical component order using
FP64, divides by five, and publishes one FP32 probability bank. Raw logits,
particle views, and hidden states are not durable.

`CE5_KD` consumes the train bank through canonical 32-byte identity joins and
uses exactly 10% unweighted CE plus 90% probability KD at temperature one.
`CE5_CONTROL` receives labels only. Both remain ordinary HLT-only models.

## Comparisons

The aggregate reports every teacher, their mean and sample spread, `CE5E`,
`CE5_KD`, and `CE5_CONTROL`. Primary comparisons are:

```text
CE5E - mean individual CE seed
CE5E - best individual CE seed
CE5_KD - CE5_CONTROL
CE5_KD - CE5E
```

Macro AUC is primary. CE, accuracy, macro R50, and all per-class QCD rejection
metrics remain required. The study may later be compared to TRI60 endpoints,
but no cross-campaign validation result changes this graph.

## Interpretation

- If `CE5E` and `CE5_KD` match the structured ensembles and compressed
  endpoints, ordinary seed diversity explains much of the result.
- If `CE5E` remains weaker, the homotopy graph contributes useful structured
  diversity beyond independent seeds.
- If `CE5E` is competitive but `CE5_KD` compresses worse, the structured
  teacher distribution is easier to distill.
- `CE5_KD - CE5_CONTROL` is the paired estimate of the ensemble-KD objective,
  subject to the usual one-study exploratory limitations.

The study cannot claim universal seed or KD effects from one five-member
ensemble. It does not access final test.

## Execution and isolation

The campaign uses a separate source-pinned worktree, root, `hcwce5_` job
namespace, graph, contracts, command plan, dry/live ledger, attestations,
monitor, and recovery root. It has no Slurm dependency on, and no write path
into, TRI60 or the dense extension. It authenticates their source campaign,
foundation, recipe, and operational evidence read-only.

The five teachers and paired CE control launch in parallel after preflight.
The reducer waits only for the five teachers. `CE5_KD` waits only for the
reducer. Aggregate waits for the reducer and both paired students. Poor finite
metrics never fail or prune execution.

GPU fits request one GH200, 72 CPUs, 256 GiB, and three days. The reducer
requests one GH200, 72 CPUs, 192 GiB, and one day. Compact train and validation
probability banks are the only durable teacher targets. A 16 GiB free-space
reserve is enforced. Existing real TRI60 production-worker evidence is reused
because this study introduces no new model, input, cache, loss, probability,
or no-resume execution primitive.

## Acceptance

Queue readiness requires exact graph/seed/loss tests, hand-calculated ensemble
tests, paired-initialization tests, train/validation identity and lineage
checks, campaign DAG and isolation tests, restart-from-zero recovery tests,
CLI help, worker shell syntax, a complete nonmutating dry run, the full local
suite, and `git diff --check`. Live submission still requires an exact pushed
commit and explicit phrase-bound authorization. No reduced scientific smoke
campaign is introduced.
