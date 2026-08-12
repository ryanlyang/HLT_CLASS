# HCWDL Common-Schema Architecture–Input Factorial Plan

Status: implementation-authoritative validation-only supplemental study.

## Question

Canonical native-offline `TOFF` and HLT models differ in both inputs and
architecture. `TOFF` uses bounded 19-channel charged and 7-channel neutral
streams, while the deployable HLT model uses one unified 21-channel stream.
This study separates the effect of a charged/noncharged split architecture
from HLT versus projected-offline particles without pretending that the
native adapters form a literal tensor-level factorial.

## Exact cells

All four fresh models use the ordinary 21-channel Scouting schema:

| ID | particles | architecture | deployable |
|---|---|---|---|
| `H_U` | exact HLT | unified eight-block ParT | yes |
| `H_S` | exact HLT | split charged/noncharged full-depth ParT | yes |
| `O_U` | projected native-offline `P0` | unified eight-block ParT | no |
| `O_S` | projected native-offline `P0` | split charged/noncharged full-depth ParT | no |

The split model has two full eight-block, 21-channel encoders and the same
256-to-128-to-15 fusion topology as canonical native `TOFF`.
Charged means an active token with `isEl`, `isMu`, or `isChargedHad`; every
other active token, including unknown/unclassified identity, enters the
noncharged stream. Stable compaction preserves order within each stream.
Every token appears exactly once, padding is zeroed, and neither stream has an
independent truncation. Thus `H_U/H_S` and `O_U/O_S` receive byte-identical
particle inputs respectively. This makes `O_S` the closest honest common-schema
analogue of `TOFF`: their encoder depth and fusion topology match, while their
feature adapters do not. The split architecture is necessarily larger than
the unified model, so capacity is part of the declared architecture factor;
exact parameter counts and their ratio are mandatory runtime evidence. The
study may not claim that the architecture contrast isolates routing from
capacity.

Canonical native `TOFF` is imported as a fifth contextual reference. It is
not a factorial cell and no equality is claimed between `O_S` and `TOFF`.
Their difference diagnoses the residual native-adapter/feature-basis effect.

## Training and pairing

All cells are fresh, unweighted CE-only fits using the authenticated parent
recipe: 60 passes, validation every pass, macro-AUC-first checkpoint selection,
same rows, batching, sampler, optimizer, LR schedule, and labels. `H_U/O_U`
share an initialization seed alias; `H_S/O_S` share another. All four share
the sampler seed. Smoke mode uses the generic two-update miniature. No KD or
warm initialization is permitted.

## Reporting

For each metric the aggregate reports:

- architecture effect on HLT: `H_S - H_U`;
- architecture effect offline: `O_S - O_U`;
- input effect in unified architecture: `O_U - H_U`;
- input effect in split architecture: `O_S - H_S`;
- interaction: `(O_S - O_U) - (H_S - H_U)`.

Primary interpretation uses validation macro OVR AUC. CE, accuracy, balanced
accuracy, and geometric-mean log-R50 are descriptive companions. A one-seed
result is exploratory.

## Access, artifacts, and execution

The campaign imports an authenticated unweighted HCWDL smoke or 300k pilot's
split, row selection, recipe, and completed `TOFF` report. HLT and P0 views are
built once per fit into process-local RAM and never persisted. Labels enter
only CE/evaluation; no label enters token partitioning. The campaign registers
zero final-test rows and cannot publish a final-test task. Only HLT cells are
deployable.

The exact DAG is architecture check, four parallel fits, aggregate, completion.
The architecture check must exercise finite forward and backward passes for
both models under training's BF16 autocast policy, including split-stream
embedding assembly from FP32 cached inputs, before it releases any fit.
Training requests use 8 CPUs, 96 GiB RAM, and one GH200. Exact-HLT cells use
six hours. Projected-offline P0 cells use sixteen hours: the first 300k pilot
measurement showed that P0 ROOT projection and process-local cache assembly
alone reached the former six-hour limit before the first optimizer update.
A real Tigris smoke using production workers is required before the 300k
pilot. Scientific performance never controls completion; invalid inputs,
lineage drift, nonfinite values, source drift, or token loss fail closed.
