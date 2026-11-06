# U100 embedding-first curriculum: twenty configurations

Scientific authority v1, 2026-11-06. This implements the user's accepted
twenty distinct configurations (one seed), rather than ten replicated recipes.
It is separate from the seven-candidate and reconstruction campaigns.

## Fixed comparison

All rows use the authenticated 299,234 D000 TRAIN / 74,895 D000 validation
population, native HLT ParT, original independent 128D embedding predictor,
original initialization/sampler seeds, AdamW groups, batch128 and thirty spent
epochs. Frozen regular COARSE_U100 TRAIN targets and their SmoothL1 loss remain
unchanged. Native HLT inference exports no predictor or teacher. E00 reuses
the authenticated same-seed C0 result; nineteen new fits are submitted.

| Row | Pretraining epochs | Classification stage / change from E01 |
|---|---:|---|
| E00 | 0 | Original CE1+embedding5 throughout; reused control |
| E01 | 10 | Main: CE0 to1 linear over5 epochs, then1 |
| E02 | 3 | Main transition,27 subsequent epochs |
| E03 | 5 | Main transition,25 subsequent epochs |
| E04 | 15 | Main transition,15 subsequent epochs |
| E05 | 20 | Main transition,10 subsequent epochs |
| E06 | 10 | Immediate CE1 |
| E07 | 10 | CE0 to1 over2 epochs |
| E08 | 10 | CE0 to1 over10 epochs |
| E09 | 10 | Half-cosine CE0 to1 over5 epochs |
| E10 | 10 | CE0.1 for2 epochs,0.3 for3, then1 |
| E11 | 10 | CE0 to2 over5 epochs, then2 |
| E12 | 10 | Retain optimizer moments and original LR |
| E13 | 10 | Retain optimizer moments; restart LR |
| E14 | 10 | Reset optimizer moments; continue original LR |
| E15 | 10 | After CE ramp, embedding5 to1 over15 epochs |
| E16 | 10 | After CE ramp, embedding5 to0 over10 epochs, then5 CE-only epochs |
| E17 | 10 | CE0.05 during pretraining; ramp0.05 to1 over5 epochs |
| E18 | 10 | Transition from last pretraining epoch instead of best |
| E19 | 0 | CE1+embedding5 throughout; optimizer/LR restart after epoch10 |

Unless overridden, pretraining uses CE0/embedding5; transition resets AdamW
moments and restarts LR. Restart LR warms linearly to3e-4 over2 epochs, then
cosines to1.5e-5 at spent epoch30. Pretraining and continued LR use original
warmup3/cosine30. Loss ramps are continuous at optimizer-update positions;
stepped CE uses the first2/next3 subsequent epochs exactly. No early stopping.
The classifier receives no gradient or AdamW decay when CE is zero; the
predictor receives neither when its weight is zero.

## Selection, clocks and state

Select the earliest minimum unweighted full-validation embedding loss during
the allotted pretraining window, except E18 uses its last epoch. E19 also uses
its current epoch10 state and never rewinds. Restore the full model (including
predictor and buffers), CPU/CUDA RNG and, for carry arms, matching AdamW moments
and step counters. Save the complete selected transition snapshot atomically.
Never pair earlier model weights with later optimizer moments. Reset arms
construct fresh AdamW with original groups. Epoch/data-order and LR clocks
continue at the *spent* transition epoch N, not the selected epoch k.
All N pretraining epochs count even if k<N. Independent fits repeat pretraining
with identical seeds; there are no fit-to-fit dependencies or shared checkpoints.

Final selection uses the original native validation AUC/CE/rejection/earliest
tie rule only after embedding pretraining. E19 and E00 are eligible throughout
because CE is active throughout. Record full embedding validation and native
classification metrics for every epoch; pretraining AUC never selects a state.

## Explicit new validation role

The old TRAIN-only target contract is unchanged. This campaign adds a separate
validation target artifact: frozen selected COARSE_U100, FP32 native pre-fc
embedding, same original128-row extraction, standardized with the original
TRAIN mean/scale. No validation moments are fitted. Existing Brown validation
U100 shards and encoder delta from the particle-teacher lineage are reused;
no new SPORC transfer or teacher fitting is necessary. Exact identities,
labels, preparation, teacher hashes, counts and role must match. Student
training has no access to validation targets; they are read under inference
mode for selection only. No test data are opened. Historical original50k
overlaps validation and remains report-only; repeated validation use is exploratory.

## Execution and acceptance

Use Hydra gpus/default/normal, one RTX A6000,6CPUs,32000MiB;48h maximum fit,
12h setup. Setup authenticates inputs, builds the frozen validation bank,
checks installed-Weaver native/control parity, then runs warmed production
miniatures through every registered transition. Admission uses measured full
and non-singleton partial batches, both validation heads, state restoration,
reload checks, memory and2x timing safety plus900s. Bad finite scientific
metrics never fail admission. All miniature weights are discarded.

One authorized bootstrap queues setup and a durable compute CPU release
(one requested CPU,4GB,1h, afterok exact setup). Release authenticates completed
setup/receipts, dry-reviews the measured19-fit plan and submits independent jobs.
Order: E01,E06,E12,E19,E03,E15,E17,E18,E02,E04,E07,E08,E09,E10,E11,E13,E14,E16,E05.
No lanes, caps, fit dependencies or priority changes. Ambiguous partial
submission stops with its journals preserved. Reporting is read-only on demand.
Clean pushed source, exact immutable specs and genuine Hydra acceptance are
required before science. Local tests alone establish readiness to queue setup,
not measured Hydra throughput or scientific success.
