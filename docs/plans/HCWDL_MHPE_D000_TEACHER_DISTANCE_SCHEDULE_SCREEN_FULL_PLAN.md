# HCWDL-MHPE Full-Data D000 Teacher-Distance Schedule Screen

Status: implementation-authoritative, validation-only.

## Scientific question

The completed 300k/60-pass MHPE evidence suggests that a teacher closer to a
student's input domain can become more useful later in optimization, while the
20-pass full-data evidence can show the opposite ordering. This study tests
whether that reversal is explained by optimization horizon or peak learning
rate at one fixed, deployable input: byte-exact canonical HLT (`D000`).

The experiment crosses four authenticated teacher targets

```text
U000, U100E, D066E, D033E
```

with six peak AdamW learning rates

```text
3e-4, 2e-4, 1.5e-4, 1e-4, 7.5e-5, 5e-5.
```

This creates 24 fresh paired `D000` students. Every fit uses unweighted
`C25P75` (`0.25 CE + 0.75 single-teacher KD`) at temperature 2 and runs one
locked 80-pass warmup/cosine trajectory. Validation occurs after every pass.

## Horizon estimands

For every fit, the training worker preserves four independently authenticated
checkpoints:

```text
H20 = best checkpoint available among passes 1..20
H40 = best checkpoint available among passes 1..40
H60 = best checkpoint available among passes 1..60
H80 = best checkpoint available among passes 1..80
```

The selection key is the existing maximum macro AUC, minimum CE, maximum
logR50, earliest-update policy. The four checkpoints come from one 80-pass
trajectory. They are not claimed to reproduce separate 20/40/60-pass cosine
schedules because those schedules would decay the learning rate earlier.

All four checkpoints are evaluated exactly once on the untouched scoring half
of full-data validation. Therefore the campaign owns 24 expensive fits and 96
held-out evaluations, not 96 fits. The scoring half cannot select a checkpoint,
stop a job, change a teacher, or alter an artifact.

## Pairing and intended contrasts

All 24 fits share the same fresh initialization, train sampler, validation
order, dropout, optimizer, HLT view, repair, and model seed aliases. Only the
registered teacher target and peak learning rate differ. This deliberately
pairs both teacher and LR comparisons.

The primary table contains, for every `(LR, horizon)`, the scoring AUC of all
four teachers and these predeclared contrasts:

```text
D033E - U000
D066E - U000
U100E - U000
D033E - D066E
```

The primary mechanistic question is whether `D033E - U000` becomes more
positive with horizon and whether that behavior depends on LR. CE, accuracy,
balanced accuracy, macro logR50/R50, and per-class metrics are reported without
controlling execution. Ranking is descriptive and cannot prune finite rows.

## Source and immutable reuse

The sole source is one authenticated, all-mapped full-data `C25P75` MHPE
campaign. Whole-campaign completion is not required. Campaign creation fails
unless the full-data foundation/reuse lineage, U000 report/checkpoint/train-
logit manifest, and the temperature-2 `U100E`, `D066E`, and `D033E`
probability bundles all validate. This exact four-teacher readiness predicate
is the only scientific source gate. Historical 300k profiles, C10P90 profiles,
missing teacher products, and path-only products are forbidden.

An additive source-reuse lock authorizes the 24 new consumers. It does not
rewrite the source target locks or source campaign. Each teacher is always
evaluated in its original source campaign; this screen consumes only its
identity-ordered immutable targets. Final-test targets and rows are forbidden.

## Population and validation boundary

Training uses every authenticated mapped full-data training row. The source's
full mapped validation identity set is reconstructed from authenticated
assignment shards and split, without labels, by a frozen global SHA-256 rank:

- first `floor(N/2)` identities: every-pass checkpoint validation;
- remaining identities: one post-training score for each of four horizon
  checkpoints.

The partition is disjoint, exhaustive, identity hashed, and shared by all 24
fits. Final test has ordinary access count zero.

## Runtime and recovery

Each fit requests 8 CPUs, 96 GiB, one GH200, and 72 hours. Train and checkpoint
validation HLT caches coexist only during training. They are released before
one scoring-half HLT cache is built and reused serially across the four horizon
checkpoints. No repaired particle dataset is persisted.

The 24 fits are independent and may run concurrently. One CPU aggregate waits
for all fits; campaign completion waits for the aggregate. GPU workers receive
`USR1@120` and exact resume must preserve the training trajectory, validation
history, and all horizon-best states. Recovery is the exact failed/downstream
closure and may not change source, graph, recipe, partition, seeds, targets,
horizons, LR grid, or GPU class.

No new standalone smoke is required: the campaign carries the authenticated
full-data MHPE worker/view/target/checkpoint evidence and records an explicit
operational-evidence waiver. This waiver does not claim the new 24-fit graph
completed a smoke. A clean pushed source, complete dry run, exact source,
resource request, and explicit creation/waiver/submission phrases remain
mandatory.

## Acceptance

Queue readiness requires focused and full tests, a bounded synthetic horizon
checkpoint/resume exercise, target corruption and source-readiness failures,
exact 24-fit/96-evaluation graph checks, CLI help, contract probes, Python and
shell syntax checks, Markdown-link checks, `git diff --check`, and an updated
`docs/HANDOFF.md`. Scientific performance never controls completion.
