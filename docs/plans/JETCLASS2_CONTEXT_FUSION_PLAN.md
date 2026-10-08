# CONTEXT_V1 100k/50k Oscar adjacent-view fusion

User authorization (2026-10-08): adapt the multi-view fusion ladder to the
existing frozen CONTEXT_V1 dataset on Oscar, retaining the current training
schedule and validation. This plan supersedes the old dzfix fusion population,
validation partition and site only for this new independent campaign. It does
not revise either historical campaign or the generator.

## Population and comparisons

Import the completed CONTEXT_V1 100k/50k campaign at
`f889f359cdddca1beb3d6e817332d2b42a42ddf3`. Authenticate its original clean source,
gate, foundation, nine fit receipts/reports and U000 train probability bank.
Reuse the exact selected identities, assignments, ordered labels, raw paired
particles and context asinh/log1p 17-channel frontend. Capacity remains 512
without truncation. No new selection, matching, generation, strength tuning,
native-Delphes HLT, class balancing or test decoding. Generator ancestry is not
an input. The final-test million is sealed even though stored nearby.

Reuse M0HLT, OFFLINE, U000, DIRECT and five COARSE reports as explicitly labelled
historical controls; do not relabel them as fresh fits. U000's authenticated
training probabilities supervise only the first new fusion. No weights are
imported as initialization. Reusing completed identical-population controls
avoids redundant fits; this remains a single-seed exploratory comparison.

## New graph

Notation: Fusion(context, primary); prediction comes from primary.

U000 -> Fusion(U000,U050) -> Fusion(U050,U100) -> Fusion(U100,D066)
-> Fusion(D066,D033) -> Fusion(D033,D000).
The last teacher has two endings:

- FINAL_DIRECT_D000: directly distill to a fresh ordinary single HLT ParT.
- Fusion(D000,D000) -> FINAL_BRIDGE_D000: a fresh two-encoder HLT-only bridge,
  then a fresh ordinary single HLT ParT.

Eight new fits, six train-bank reducers, aggregate and complete: 16 science
jobs. No old scheduler dependencies; reused parents authenticate as artifacts.
Only true new DAG edges use afterok. Negative/poor scientific results still
complete every registered node.

Reuse the native asymmetric fusion architecture with injections at primary
blocks 2/4/6/8, learned gates, zero-initialized residual projections, independent
context encoder seeds and alpha=1 for both training and validation. Reuse the
layout-preserving pair-saved-tensor CPU offload policy; no context decay,
primary extraction, logit mixture, teacher weight continuation or K2 inputs.
An explicit two-view transport axis is unpacked before the encoders, restoring
each cache's native batch width exactly. HLT/HLT uses the same data twice, not
two measurements. Only D000/None and D000/D000 are deployable.

Use the unchanged `jetclass2_delphes.runner.train_kernel` and `campaign.recipe`:
batch 256, AdamW, BF16 CUDA forward/FP32 loss, C25/P75 KL at T=2 (T² once),
3-pass warmup, hold through 45, decay through 60, 60--100 passes with the current
patience/minimum-delta rules. Do NOT import the older fusion kernel's
patience-clock rule. All 50k validation jets are used for selection and reported
development metrics, exactly as in the current CONTEXT campaign; no 50/25/25
partition. Single-HLT endpoints share initialization/sampler seeds with the
original direct/coarse D000 nodes. No rolling optimizer resume.

## Execution and admission

Oscar gpu/default/norm-gpu, one L40S, six CPUs/workers, initially 180000 MiB.
This is a resource request, not a throughput claim; it may serialize under the
user's memory QoS. CPU summaries request 1 CPU/16000 MiB. No automatic job
migration/cancellation or change of existing holds. New source is a clean
pushed detached checkout. Separate dry/live gate and science submissions require
the exact reviewed hash, durable exclusive claims and exact-ID journals.

One new 12h GPU gate authenticates inherited artifacts and measures real
full-population paired training on Oscar. It requires installed-Weaver single
FP32 output/gradient parity, native mask parity, adjacent and HLT/HLT three-step
FP32/BF16 offload parity (actual saves/restores at all sites), and longest-real-
training-view repeated batch-256 stress. Stress independently repeats each
side's longest view as a technical upper envelope, not scored paired physics.
It runs one nonscientific full-population KD pass for adjacent, HLT/HLT and
single-HLT models, using the original U000 bank solely for timing; weights are
discarded from science. Checks selected-weight and probability-bank roundtrips.

Bound four role/view caches with at most 65% host RAM; measure process-tree RSS
and require 15% CPU/GPU headroom. Project 100 passes with 1.75 margin; maximum
48h fit/24h reducer. Failure must stop admission, not silently reduce batch,
population or model. This active selected-site Oscar acceptance replaces the
generic Tigris miniature requirement for this Oscar-only adaptation. Local
fake-Weaver/tiny-model tests do not qualify as native GPU evidence.

Report all original and new rows, raw accuracy/AUC/classwise rejection and
recovery M0HLT=0%, OFFLINE=100%. Primary comparison: final single-HLT direct,
coarse, fusion-direct and fusion-bridge. Intermediate oracle improvements do
not establish HLT deployment gains. Unequal total compute, larger teacher
capacity, reused development validation and one seed remain explicit caveats.
