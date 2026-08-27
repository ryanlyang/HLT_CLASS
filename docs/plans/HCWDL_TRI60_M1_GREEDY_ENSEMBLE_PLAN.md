# HCWDL TRI60 M1 greedy ensemble diagnostic

Status: implemented locally; Tigris execution pending.

## Question

The completed M1 compression screen contains 19 newly trained exact-HLT
models plus the imported original `M1_LOGIT`. This validation-only diagnostic
asks how much of the remaining `LOGIT_D000E` compression gap can be recovered
by a small probability ensemble of those single-model compression methods.

This is explicitly exploratory forward selection on validation. It does not
create a deployable single model, select a finalist, access final test, or
modify the completed M1 screen or original TRI60 campaign.

## Candidate and ensemble boundary

The candidate registry is exactly:

```text
SOURCE_M1_LOGIT + the 19 immutable screen fits, in screen graph order
```

`LOGIT_D000E` remains the teacher/reference and is forbidden from the
candidate set. U000 and CE controls are also not candidates. Every ensemble
is the uniform arithmetic mean of member FP32 categorical probabilities,
accumulated in FP64 and cast once to FP32. Members are distinct and the
maximum size is five.

Three deterministic greedy paths are evaluated:

1. maximum macro one-vs-rest AUC;
2. maximum linear macro R50;
3. maximum equal-weight mean of AUC recovery and linear-R50 recovery from
   `SOURCE_M1_LOGIT` to `LOGIT_D000E`.

At each size, every unused candidate is tested as the next member. Exact ties
use AUC, linear R50, CE, and then frozen candidate order. The report retains
sizes one through five and the best size along each path. Scientific metrics
never control job success or launch another job.

## Parallel execution and storage

The 20 selected checkpoints are partitioned into five fixed shards of four.
Five independent GH200 jobs each build the same exact-HLT validation view once
in process RAM, infer four checkpoints sequentially, and discard the view.
They persist only identity digests, labels, and 15 FP32 probabilities. Each
shard is capped at 320 MiB and the complete durable prediction set at 2 GiB.
Particle views, logits, training targets, hidden states, and final-test data
are never durable.

One 72-CPU reducer loads the five aligned shards and uses at most 32 forked
workers over shared read-only parent arrays. It evaluates unique candidate
sets once even when objectives overlap. The immutable source metrics are
recomputed from every singleton prediction and must agree before selection.

The exact DAG is:

```text
authenticate
    -> infer_shard_00 .. infer_shard_04   (parallel)
    -> greedy_reduce
    -> campaign_complete
```

All jobs use nice 10000 and have no scheduler dependency on existing TRI60,
DX, CE5, or M1-screen jobs. The source campaigns are read-only.

## Interpretation boundary

Greedy selection and reporting use the same validation population and will
overestimate generalization if treated as a confirmatory result. The output
is suitable for diagnosing complementarity and choosing a separately
registered follow-up, not for unsealing final test or making a publication
claim by itself.
