# HCWDL TRI60 standalone M1 compression screen

Status: implemented, source-pinned, ready for local dry-run acceptance.

## Question

The full-data `M1_LOGIT` improves over each individual exact-HLT D000
specialist but preserves substantially less of `LOGIT_D000E` than desired.
This additive screen tests whether the compression loss is caused by
initialization, CE/KD mixture, teacher temperature, peak learning rate, or a
constant high-KD schedule.

The study is diagnostic. Validation results rank the conditions but never
control job completion, automatically launch a follow-up, or access final
test.

## Source boundary

The screen authenticates only the already-complete subset of the original
TRI60 campaign:

- the canonical campaign, foundation, recipe, graph, and endpoint evidence;
- `LOGIT_D000E` probability lock, train/validation manifests, and stage
  report;
- the completed `M1_LOGIT` report and selected checkpoint;
- the completed structural-nearest specialist
  `LOGIT_D000_from_D033E` and selected checkpoint.

The parent campaign's aggregate, finalist, and completion artifacts are not
required. There are no Slurm dependencies on parent job IDs and no parent
files are written, copied, cancelled, held, or reprioritized.

## Fixed training boundary

Every fresh condition uses all authenticated mapped train and validation
rows, exact HLT D000 particles, unified 21-channel ParT, batch 256, 60 passes,
validation after every pass, and the established AUC/CE/logR50/earliest
checkpoint selector. Optimizer state is always fresh. Rolling resume and
partial checkpoints remain disabled. Only selected and terminal checkpoints
are durable.

All conditions retain the original `M1_LOGIT` model/training/sampler seed
alias. Cold fits initialize from scratch. Warm fits load only the selected
model state of `LOGIT_D000_from_D033E`; polish loads only the selected model
state of `M1_LOGIT`. Neither optimizer, schedule, RNG state, nor previous
training history is inherited.

## Exact twenty-condition registry

One cell is imported, not retrained:

- cold `C10P90`, T=1, peak LR `3e-4` (`SOURCE_M1_LOGIT`).

The 15 remaining factorial cells are:

- cold `C10P90` and `C25P75`, T=1 and T=2, peak LR `3e-4` and `1e-4`, with
  the imported cell omitted;
- warm `C10P90` and `C25P75`, T=1 and T=2, peak LR `1e-4` and `5e-5`.

Four targeted cells complete the registry:

- cold `C50P50`, T=2, LR `1e-4`;
- warm `C50P50`, T=2, LR `5e-5`;
- cold T=2/LR `1e-4` ramp: C75P25 through pass 5, linearly reaching
  C10P90 at pass 15, then remaining C10P90;
- 60-pass polish of selected `M1_LOGIT` at C25P75, T=2, LR `5e-5`.

Thus the study has 20 reported conditions and 19 fresh fits.

## Temperature semantics

`LOGIT_D000E` is the teacher. T=1 consumes its authenticated probability
bank directly. T=2 is defined exactly as
`softmax(log(p_LOGIT_D000E) / 2)`. This is temperature scaling of the
categorical ensemble distribution and does not require reconstructing,
persisting, or copying component logits. The transformed probability array is
held in process memory for each fit and discarded when that job exits.

## Scheduler graph

The standalone DAG is:

```text
authenticate -> preflight -> 19 independent GPU fits -> aggregate -> complete
```

It contains 23 jobs. Fits request one GH200, 72 CPUs, 256G RAM, and three
days. CPU locks request 4 CPUs, 32G, and two hours. Every job uses Slurm nice
5000, so this opportunistic screen does not outrank the original TRI60 work.

## Validation and execution decision

No standalone smoke is required or authorized. The unchanged production
cache/training machinery and carried TRI60 installed-Weaver/resource evidence
are reused. Queue readiness requires focused local tests, source-pinned
campaign publication, a complete exact dry-run ledger, and explicit live
authorization. Final test remains sealed and absent.
