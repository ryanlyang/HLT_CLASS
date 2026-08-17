# HCWDL-MHPE Endpoint Teacher-Mixture Add-on

Status: implementation authority for the validation-only 300k endpoint add-on.

## Question

Does the dense exact-HLT endpoint ensemble benefit when its teacher
distribution is mixed with the independently trained CE-only exact-HLT
`M0paired` distribution before the final M1 distillation?

This is an additive downstream experiment. It does not retrain, mutate, or
reinterpret any U/D specialist or dense ensemble.

## Authenticated source

One completed 300k/100k dense MHPE campaign is selected explicitly at campaign
creation. It must use either
`C10P90_DENSE_ANCHOR50_300K60` or
`C25P75_DENSE_ANCHOR50_300K60`, have an authenticated completion report, expose
the complete `D0E/T1` train and validation probability bundle, and bind the
completed unified-balanced foundation containing `M0paired`.

The add-on binds the source campaign spec, completion report, D0E target lock
and manifests, foundation reuse lock, M0paired report and checkpoint, split,
row selection, executable recipe, and exact source commit. Final-test rows are
forbidden.

## Frozen teachers and graph

At temperature one, for every authenticated jet identity, define

```text
q_w = w * q_D0E + (1-w) * softmax(z_M0paired)
```

with exact rational weights and the following four paired students:

```text
M1_D0only  w = 1/1
M1_mix90   w = 9/10
M1_mix75   w = 3/4
M1_mix50   w = 1/2
```

The D0E and M0paired components are joined by canonical jet identity, never by
incidental row order. M0paired softmax uses max-subtracted FP32 arithmetic.
Mixtures are accumulated in lexical component order in FP64, divided by exact
integer rationals, and published once as little-endian FP32. Every row must be
finite, nonnegative, and normalized within the existing probability-target
tolerance.

All four students:

- consume byte-identical exact-HLT inputs;
- are freshly initialized with the same paired initialization seed;
- use identical sampler, dropout, optimizer, batching, and update schedules;
- train for 60 complete passes and validate every pass;
- use unweighted `0.10 CE + 0.90 probability KD`, `T=1`;
- select the checkpoint by macro AUC, CE, logR50, then earliest update;
- differ only in the frozen teacher mixture.

The baseline is retrained inside this add-on; the source campaign's M1 is only
contextual and is not used as the paired baseline.

## Execution graph

```text
build_targets
  -> train_M1_D0only || train_M1_mix90 || train_M1_mix75 || train_M1_mix50
  -> aggregate
  -> campaign_complete
```

This is exactly seven jobs. `build_targets` and the four fits request 8 CPUs,
96G RAM, 06:00:00, and one GH200. Reporting jobs request 4 CPUs, 32G, and one
hour. The target builder performs one M0paired forward pass per role and reuses
the already durable D0E probabilities. No repaired particle datasets or model
inputs are persisted.

## Reporting and claims

The validation aggregate reports all four rows, pairwise deltas from
`M1_D0only`, source `D0E`, source `M1`, and `M0paired` context, plus macro AUC,
CE, accuracy, macro logR50/R50 and per-class rejection when available. Poor
metrics never fail or prune a job. The completion report requires all four
durable reports and states `final_test_accessed=false`.

The primary comparison is paired `M1_mix90 - M1_D0only`. The other weights
measure whether any benefit is monotone or whether increasing M0 contribution
eventually dilutes the endpoint teacher. This validation study does not select
or access final-test rows.

## Operational policy

Creation requires a clean, full source commit and an explicit live-submission
authorization phrase. Submission requires the canonical dry-run ledger and a
second exact phrase. Workers use absolute source-pinned paths, isolated Python,
the Tigris environment, and `USR1@120` checkpoint signaling. Failed/downstream
recovery is exact-ledger and source-pinned; completed siblings are preserved.

No new smoke is required because the worker path is the already accepted
300k MHPE exact-HLT cache/training path plus a bounded probability reducer.
Campaign creation and a complete dry run remain mandatory. No job may be
submitted by implementation work itself.

