# Joint privileged adapters and final-token supervision on Hydra — v1

This plan implements the user's approved **two teachers plus twelve students**.
It is a separate campaign; frozen-adapter, contrastive and pair-correction runs
are neither modified nor cancelled. Hydra uses `gpus`, not SPORC's `debug`.

## Scientific question

The preceding frozen-M0HLT oracles recovered 11.1% (bias) and 12.0% (edge) on
the supplied development validation. These results motivate testing whether
learning the backbone together with the privileged adapter gives more useful
corrections. Higher oracle AUC does not establish that HLT can predict them.
The second question is whether predicting corrections from final particle
representations is more useful than predicting them at the intervention layer.

## Two teachers

`teacher_bias` and `teacher_edge` start from the original from-scratch M0HLT
initialization seed, **not from its trained checkpoint**. The complete native
backbone, classifier and adapter train jointly with CE1 using the original
batch128, AdamW, 30-epoch student learning-rate schedule and sampler seed.
Normalization buffers update normally. The unused embedding predictor is frozen.

The main token/value input remains D000/HLT. Privileged U100 features enter only
the adapter at particle block index6, the seventh of eight blocks. U100 preserves
HLT slot order/cardinality, upgrades matched slots to offline values and keeps
unmatched HLT particles. No matching flags, source indices or additional offline
particles are deployable inputs. Oracle validation alone may consume validation
U100, to select each teacher by full validation AUC and the existing tie-breaks.

Preserve the original adapter architecture: 32-wide normalized token projection,
rank4 ordered pair interaction, 16-wide geometry MLP, query chunks of32,
`2*tanh(raw/2)` correction bound, and padding masks. Geometry contains the four
native HLT pair features and, in teachers only, the four U100 counterparts,
scaled by10 with diagonal geometry zeroed. The bias correction has eight
channels centered over valid keys and adds to native pair bias. The edge
correction has four coefficients shared across heads; native attention
aggregates them through a learned per-head 4-to-16 basis and native output
projection. Native values, pair bias, Q and K remain present.

Teacher bias uses the existing math-SDPA mask-gradient path at the intervened
block. Student attention kernels remain those of the original adapter controls.
Zero adapter outputs initially recover the corresponding *randomly initialized*
native model. Both teachers have the same initial native weights as each other
and as the registered students. Freeze the selected teacher for student targets.

## Twelve students and ordering

All students are fresh HLT models with **CE1 + U100 embedding5**, no logit KD.
The original U100 embedding targets, normalization and head are reused exactly.

| First six tasks | Correction weight | Mechanism |
| --- | ---: | --- |
| `bias_both_l0p3` | 0.3 | intermediate prediction + bias injection |
| `bias_final_aux_l0p3` | 0.3 | final-token auxiliary bias prediction |
| `edge_both_l0p3` | 0.3 | intermediate prediction + message injection |
| `edge_final_aux_l0p3` | 0.3 | final-token auxiliary message prediction |
| `bias_aux_l0p3` | 0.3 | intermediate auxiliary bias prediction |
| `edge_aux_l0p3` | 0.3 | intermediate auxiliary message prediction |

The next six repeat this order with suffix `l1` and weight1. Submission order
expresses preference; scheduler eligibility/resources determine actual starts.
No student depends on another student and no campaign concurrency cap is imposed.

Intermediate predictors read the normalized HLT context at block6's attention
input. Final auxiliary predictors read **block7's output before class attention
and jet pooling**, retaining one representation per particle. Both also receive
native HLT pair geometry. The final branch predicts the *same block6 teacher
corrections*; it does not feed back into earlier attention or require a second
student forward pass. Its loss backpropagates through all eight particle blocks.

The objective is `CE + 5*embedding_SmoothL1 + lambda(t)*correction_SmoothL1`.
`lambda(t)` ramps linearly from zero to0.3 or1 over three epochs, by update.
Correction loss divides prediction and detached teacher targets by teacher
channel RMS estimated on the first4096 authenticated TRAIN jets (floor0.05).
Average over channels, valid ordered pairs within each jet, then jets. Never
sum unnormalized quadratic pairs or fit statistics on validation.

Injected students have the existing learned eight-by-eight bias-head alignment
or per-head edge-message basis. Supervision applies before alignment; CE trains
the predictor and alignment through injection. Auxiliary branches do not inject.
Both intermediate and final auxiliary predictors are stripped from deployment,
as is the embedding head in all students. Injected predictors remain, increasing
HLT inference cost. Every student validation/deployment accepts only HLT input.

## Comparisons and reporting

Reuse historical M0HLT/OFFLINE recovery anchors, completed U100-w5 and its Hydra
replay. Reuse authenticated existing `bias_ce_e5`/`edge_ce_e5` as injection
architecture controls; do not train them again. Include old frozen teachers and
CE-only `e0` controls as context. Original-source replay authenticates completed
controls from commit4b9630fa; incomplete controls are explicitly pending and do
not stop this campaign. Results may be rerun after those controls complete.

Primary: full74895 validation jets, same populations and AUC-selected checkpoints.
Recovery is `100*(AUC-M0HLT)/(OFFLINE-M0HLT)` using historical fixed anchors.
For injected students also report delta against the corresponding CE adapter
control. Auxiliary deployment matches native U100-w5. Original50k overlaps full
validation and is report-only; this is single-seed exploratory development, with
no independent test. Poor finite quality never cancels a later experiment.

## Execution and artifacts

Reuse Brown's existing bundle, frozen-adapter input delta, Python environment
and completed precision evidence. **No data transfer, target-bank rebuild,
reference retraining or separate preflight.** M0HLT weights are used only for a
setup numerical replay, never teacher/student initialization.

One shared setup authenticates inputs and runs real production miniatures for
all14 tasks, including backward, selected-state publication/reload and measured
time/memory. Automatic science submission waits for valid technical acceptance.
Each student has only `afterok` on its own teacher. Report uses `afterany` for all
fourteen fits. One L40, six CPUs,120000MiB per job; measured fits capped at24h.
Reserve8GiB outputs plus filesystem headroom. No quality-based admission.

Immutable `LUKA_JOINT_ADAPTER_HYDRA_*/v1` artifacts are incompatible with frozen
teacher artifacts. Keep old controls in their original worktree and authenticate
them there. Exclusive claims and exact-ID submission journals prevent duplicate
fits. A `finish` CLI resumes science submission after a completed setup if the
login-node watcher stops, without deleting claims or repeating setup.

Implementation and commands: [runbook](../LUKA_JOINT_ADAPTER_HYDRA_RUNBOOK.md).
