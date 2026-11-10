# Shared summary tokens with privileged embedding supervision

Scientific registration v1, 2026-11-09. User-authorized implementation of five
configurations at two matched seeds (ten fresh fits). This separate campaign
uses Hydra RTX A6000 qualification and automatic release. It does not alter
existing campaigns or authorize submission from an uncommitted workspace.

## Fixed matrix

| Cell | Summaries | Classification | U100 / IT predictor inputs |
| --- | ---: | --- | --- |
| A | 1 | CE | none |
| B | 1 | CE + U100-w5 + IT-w5 | general / general |
| C | 3 | CE | none |
| D | 3 | CE + U100-w5 + IT-w5 | general / general |
| E | 3 | CE + U100-w5 + IT-w5 | U100 summary / IT summary |

Queue E,B,D,C,A for seed0, then E,B,D,C,A for seed1. All ten fits are
independent; submission order is a preference, not a scheduling guarantee.
No teacher is retrained. Reuse the original frozen U100 TRAIN bank and the
authenticated completed selective-information IT TRAIN bank/checkpoint.

## Model and scientific meaning

Use the installed native 17-input,11-class ParT: width128, eight particle
blocks, eight heads, two class-attention blocks, original pair embedding and
no constituent truncation. The particle encoder executes exactly once.
A/B use the unmodified installed forward. C/D/E read its final particle
states using three distinct learned query tokens in canonical order general,
U100,IT. Each query reads all unmasked particle states and its own current
token through the SAME two class-attention blocks and final LayerNorm.
Weights are shared; query streams do not attend to one another. No attention
targets, hand-selected particle subsets or imposed diversity loss are used.

Concatenate the three normalized128 summaries and apply Linear(384,11).
Single-summary models retain Linear(128,11). Extra deployment parameters are
exactly two128 query vectors plus256*11 classifier weights =3072. There are
no cloned class-attention blocks, extra normalizers or fusion MLPs. Class
attention runs three times, so latency must be measured despite the small
parameter increase. All summaries remain in deployed C/D/E; only prediction
heads are removed. Deployment consumes HLT features/vectors/masks only.

Every auxiliary predictor is Linear(128,128),GELU,Linear(128,128), using the
original U100/IT predictor seeds. Targets use their existing independent
TRAIN normalization and mean SmoothL1(beta1). B/D/E optimize exactly
CE +5*U100 +5*IT. Ground-truth CE reaches all used readouts and the shared
encoder. Frozen targets receive no gradients. A/C construct no prediction
heads and consume no targets in optimization. D/E have identical parameter
names, shapes, initial values, losses, weights and optimization schedule;
only the predictor input route differs. Independent queries do not eliminate
gradient interference in shared weights.

Native initialization and sampling use the parent's seeds; seed1 adds100003.
Create extra queries with isolated RNG seed2026110901+seed*100003, truncated
normal std0.02. Preserve the native general token and every backbone weight.
The expanded classifier copies the native weights into its first128 columns,
copies the bias, and initializes extra columns to zero. Thus initial logits
match the native factory within FP32 tolerances. Extra queries initially get
no CE gradient until their classifier columns move; E's auxiliary gradients
can reach them immediately. This is shared across C/D/E, not an extra E-only
advantage. Auxiliary and extra-query construction preserve native RNG state.

FP32 numerical equivalence uses rtol2e-5/atol5e-6 for changed GEMM batch/width
shapes, with actual maximum differences recorded by setup. This is distinct
from historical BF16 AUC replay (absolute tolerance5e-5), and neither is a
scientific quality gate.

All runs are fresh, batch128,30 epochs on299234 TRAIN /74895 development
validation jets, with the original optimizer/LR/selection rules. No optimizer
reset, warm start, extra teacher, loss sweep or final-test inference is added.
Original50k overlaps validation and is report-only. Finite poor performance
never rejects a run or blocks later science.

## Comparisons and observability

Report AUC and fixed historical M0HLT/OFFLINE recovery for every completed
fit, plus read-only epoch progress. Primary comparison E-D isolates routing.
Also report C-A (architecture without privilege), D-B (extra summaries under
the existing supervision), E-B (total improvement), B-A and E-C (supervision),
and (E-C)-(B-A). Compare same seeds; missing controls remain null. Show both
seeds and descriptive mean/min/max, without a significance claim.

Record actual deployed parameter/state counts, warmed batch128 latency,
training duration, weighted loss/encoder gradient diagnostics and per-summary
gradient norms/cosines at epochs1,3,10,20,30. Low summary diversity or small
gradients are results, not quality gates. Strict architecture-specific
checkpoint reload must reproduce selected predictions and metrics. A/B
checkpoints retain ordinary native keys; C/D/E require the three-summary
loader and retain their query tokens and expanded classifier.

## Qualification, source and release

Internal donor commit54ff594df4fb7dddc56c0e99ffb13b9448a6a6c9, primarily
luka_capacity_hydra. New runtime must be additive under its own namespace;
all reused artifacts retain original validators. Register exact pushed clean
source, recipe, selective parent, IT bank/checkpoint/receipt and population.
No dependency on unrelated S4/S6 student completion is introduced.

Bootstrap submits GPU setup and CPU release(afterok setup). Setup authenticates
inputs, replays the original native checkpoint, checks installed-factory FP32
parity and the independent-query computation, checks parameter delta3072,
runs actual-target production miniatures for all ten cells, stresses longest
support and a partial batch, then measures warmed representative training and
full-validation cost. Technical weights are discarded. Every real fit repeats
a disposable startup miniature before fresh science initialization.

One A6000,6CPUs,32000MiB per GPU job; setup cap12h and each fit cap48h. Use
measured work with2x safety,900s fixed overhead and15min startup allowance;
host peak must stay below85% of allocation and GPU peak below88% of VRAM.
CPU release/report use compute/default/normal,1 requested CPU,4000MiB,1h;
accept observed one-requested/two-allocated CPU rounding. A completed setup
receipt, exact COMPLETED/0:0 accounting and validated measurements authorize
automatic release of ten independent fits and an afterany CPU report.

Full dry review covers all13 jobs. Only measured fit walltimes may shrink.
Exact-ID journals, exclusive claims and immutable hashed publication forbid
blind duplicate retries. Logging out is safe after both bootstrap IDs are
recorded. No inference or writes are needed for results/progress inspection.

Queue-ready means local checks and queue machinery are complete. Real Brown
installed-Weaver/data/GPU qualification remains the setup job's responsibility;
local synthetic or laptop GPU tests must never be represented as remote proof.
