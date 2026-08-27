# HCWDL TRI60 D000 matched-seed diversity ablation

Status: implementation authority for an additive full-data validation study.

## Scientific question

The completed five-seed CE ensemble (`CE5E`) is stronger than the original
same-seed LOGIT D000 ensemble. This study isolates one plausible explanation:
the original D000 specialists have different KD teachers but share one
coordinate seed domain, while the CE5 teachers have independent
initialization, training, and sampler domains.

The ablation asks whether the original five frozen LOGIT teachers become a
stronger exact-HLT endpoint when their D000 students receive matched stochastic
diversity. It changes seeds only. It does not alter the source ladders, select
new teachers, or reinterpret their completed results.

## Fixed fits and seed matching

The five source teachers remain exactly:

```text
U000         -> SD5_LOGIT_D000_from_U000
LOGIT_U050E  -> SD5_LOGIT_D000_from_U050E
LOGIT_U100E  -> SD5_LOGIT_D000_from_U100E
LOGIT_D066E  -> SD5_LOGIT_D000_from_D066E
LOGIT_D033E  -> SD5_LOGIT_D000_from_D033E
```

Their fresh D000 students inherit the exact five CE5 teacher seed aliases in
that order: `CE5_S01` through `CE5_S05`. Consequently initialization,
training RNG, and sampler order are distinct across the five fits and exactly
matched to the already observed CE5 ensemble's stochastic domains.

Every other choice is fixed to the source LOGIT D000 specialists:

- all authenticated mapped train and validation rows;
- exact canonical HLT D000 particles and the unified 21-channel model;
- fresh initialization and no checkpoint continuation;
- 60 passes with validation after every pass;
- effective batch size 256;
- 25% unweighted CE plus 75% probability KD at temperature two;
- the source optimizer, scheduler, warmup, mixed-precision, deterministic
  backend, and AUC-first checkpoint selection;
- no rolling resume or partial checkpoint reuse; and
- no final-test access.

All five fits launch concurrently after one read-only authentication and
preflight stage.

## Ensemble and comparisons

The selected validation probabilities of the five new students are averaged
uniformly in fixed lexical component order using FP64 accumulation and one
FP32 cast. The resulting validation distribution is `SD5_LOGIT_D000E`.

The primary comparison is:

```text
SD5_LOGIT_D000E - original paired-seed LOGIT_D000E
```

The report also compares `SD5_LOGIT_D000E` with `CE5E` and projected-offline
`U000`, records all five component metrics, leave-one-out ensembles, pairwise
diversity, mean/spread, and exact seed lineage. Macro validation AUC is the
primary metric; accuracy, CE, macro R50, and per-class rejection remain
required. No metric controls completion or changes an ensemble member.

This is one five-fit matched-seed experiment. It can show that seed diversity
matters materially in this graph, but cannot by itself estimate a universal
seed effect or prove that KD-teacher diversity is unimportant.

## Storage and isolation

This is an independent source-pinned campaign with its own root, worktree,
graph, contracts, job namespace, command plan, ledgers, reports, and task
attestations. It has no scheduler dependency on and no write path into the
running TRI60, dense-extension, or CE5 roots. Those artifacts are
authenticated read-only.

The reducer performs validation inference only. It does not publish a train
or validation probability bank, raw logits, hidden states, particle views, or
deployable checkpoint. Durable outputs are the five ordinary selected/final
checkpoints and compact JSON reports. This deliberately avoids another
full-population target bank for a diagnostic that needs only validation
metrics.

The jobs use positive nice value 10000 so the exploratory study remains below
ordinary user work. Poor finite scientific results complete normally.

## Acceptance

Queue readiness requires exact graph and seed-domain tests, uniform reducer
tests, source/CE5 lineage validation, full-data and no-final-test checks,
parallel DAG and scheduler-isolation tests, storage assertions, CLI checks, a
nonmutating dry-run ledger, neighboring TRI60/CE5 tests, the full local suite,
and `git diff --check`. Existing production TRI60 execution evidence is reused
because this study introduces no new input, cache, model, loss, training, or
inference primitive. No separate smoke campaign is required.
