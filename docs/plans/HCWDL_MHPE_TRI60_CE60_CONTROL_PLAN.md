# HCWDL-MHPE TRI60 full-data CE60 control plan

## Authority and purpose

This is the implementation-authoritative additive amendment anticipated by
Section 17 of `HCWDL_MHPE_THREE_TRACK_60E_FULL_IMPLEMENTATION_PLAN.md`. It
registers the missing pass-matched HLT-only control without changing the
immutable 32-fit TRI60 graph or any running/recovery ledger.

The scientific question is whether the designated distilled HLT endpoint
outperforms a fresh ordinary HLT Particle Transformer trained for the same
60-pass optimization budget using labels alone.

## Exact model

The sole fresh node is `M0CE60`:

```text
exact canonical HLT particles
  -> ordinary unified 21-channel, 200-token Particle Transformer
  -> unweighted 15-class cross entropy only
```

It has no teacher, KD target, offline view, assignment input, homotopy input,
representation head, or final-test access. It is technically deployable and
contains only the ordinary HLT model.

The training contract is fixed to:

- every authenticated all-mapped train row and validation row from the exact
  source TRI60 foundation;
- fresh initialization;
- 60 natural-population passes and validation after every pass;
- effective batch size 256;
- the source TRI60 AdamW, cosine schedule, warmup, BF16-forward/FP32-loss,
  deterministic-backend, and checkpoint-selection rules;
- no rolling resume and restart from update zero after interruption; and
- selected/final checkpoint publication only after all 60 passes.

`M0CE60` deliberately reuses `M2`'s initialization, training, and sampler seed
alias. Therefore `M2 - M0CE60` is paired for stochastic initialization and
batch order; their intended difference is the registered loss target.

## Isolation and execution

The control is a separate one-job campaign with:

- its own specification, graph, node, command plan, dry/live ledger, task
  attestation, output root, and `hcwce60_` Slurm name;
- no scheduler dependency on the running TRI60 campaign;
- read-only authentication of the immutable source campaign, foundation,
  recipe, endpoint/resource evidence, row population, and replicate seed;
- no writes under the source campaign root;
- no cancellation capability and no reference to source ledger job IDs; and
- 16 CPUs, 256 GiB RAM, one GH200, and a three-day walltime.

The worker rebuilds exact-HLT train/validation views once in process RAM,
replays them for 60 passes, and releases them on exit. Durable output consists
only of selected/final checkpoints, one small training report, ledger/spec
artifacts, and a task attestation. Poor metrics never fail the job.

Live creation and submission use separate exact authorization phrases. A
canonical dry-run ledger is mandatory. Submission is one exact job and is
journaled so rerunning the submitter cannot knowingly duplicate it.

## Reporting and claim limit

Primary reporting is validation macro AUC, accompanied by CE, accuracy,
macro R50, and per-class rejection. The clean comparison is `M2 - M0CE60`.
The imported 20-pass `M0paired` remains contextual only.

One seed remains exploratory. This control separates the registered
60-pass CE-versus-KD objective at the final exact-HLT architecture/input
endpoint; it does not by itself isolate every upstream path or prove universal
KD superiority.
