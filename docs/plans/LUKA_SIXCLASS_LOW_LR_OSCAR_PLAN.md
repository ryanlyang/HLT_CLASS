# Six-class lower-LR Oscar mini campaign

Status: locally implemented and verified; ready for scoped commit/push and
Oscar queueing, with real resource acceptance inside the science allocations.
This plan is the scientific authority for the new campaign; it does not revise
any completed campaign. See the validation record for the remote boundary.

## Authorized experiment

Run exactly three fresh fits, sequentially: **OFFLINE -> DE000 -> M0HLT**.
Use 20 epochs, batch size 128, AdamW with the original parameter groups,
betas, epsilon and weight decay. Change the learning-rate peak to 0.0001 and
floor to 0.000005, retaining two-epoch linear warmup and cosine decay through
epoch 20. Preserve initialization seeds, draw seeds, model architecture,
particle support and precision. This is a combined LR/objective/teacher
experiment, not an isolated LR ablation.

Reuse the exact population registered by
`luka_sixclass_balanced_oscar_8a25f126_r1`, spec content hash
`a68c689360012272a1703f67e6da160a617a2d7937450dd5d83ac0a473a5c7a4`.
The classes are QCD, X_bb, X_cc, X_ss, X_qq and X_gg. TRAIN contains
1,658,937 eligible jets, with the same number of replacement draws per epoch,
uniform across six classes and twenty 100 GeV bins in
600 <= offline jet pT < 2600 GeV. This gives 12,961 updates per epoch and a
57-jet last batch. Do not resplit, rebalance validation or access final test.

## DE000 supervision

The deployed classifier has true-label CE with coefficient **1** and no direct
logit KD. Four separate embedding and auxiliary-logit heads predict targets
from OFFLINE, CONCAT, UE100 and DE050. Each embedding SmoothL1 (beta 1,
mean reduction) has coefficient 2.5, for a total embedding budget of 10.
Each auxiliary T-squared forward KL (T=2, class sum and batch mean) has
coefficient 0.25, for a total auxiliary KD budget of 1.

OFFLINE targets come exclusively from this new OFFLINE fit: embeddings,
soft logits and contrastive keys. Normalize embeddings with statistics from
unique TRAIN jets. CONCAT, UE100 and DE050 retain their original frozen
selected checkpoints, target banks and ancestry from the completed campaign.
They are not represented as descendants of the new OFFLINE model.

Use the existing pure-OFFLINE contrastive objective, coefficient 0.3 with its
three-epoch ramp and unchanged candidates, occurrence-aware RNG and negative
sampling. No subclass objective is added. OFFLINE and M0HLT use CE alone.
DE000 and M0HLT inference use only HLT inputs and the native classifier.

## Selection and measurements

Primary checkpoint selection remains the original in-range six-class macro
OVR AUC rule, with original CE, macro log R50 and update tie breakers.
The new OFFLINE exports this primary-selected checkpoint. Validation is
natural, not sampled: 328,589 in-range and 237,443 outside-range jets.

Every epoch also predicts the outside-range validation role, preserving
separate role batching. Pool the two disjoint sets (566,032 jets) before
computing Hbb-versus-QCD AUC, R50 and R30. Record the independent full-Hbb
AUC-selected epoch (earliest on ties) as an exploratory diagnostic; it does
not change primary selection, teacher targets or historical results.
Keep all complete epoch states for exact continuation and later review.
Publish primary-selected probabilities for both validation roles.

## Execution and provenance

Use a separate versioned namespace and fresh HOME result root. Reuse raw
data and population artifacts read-only; export only the new OFFLINE TRAIN
bank. Authenticate original frozen teachers in their original clean source
and bind every reused artifact into the new spec. Never substitute historical
OFFLINE targets when the new OFFLINE result is absent.

Request one L40S, six CPUs, 192000 MiB RAM and 48 hours per fit, with explicit
afterok dependencies enforcing the requested order. No separate preflight
job: each fit performs installed-Weaver parity, an integrated two-step
miniature, objective-gradient checks, maximum-support memory checks and a
discarded timing probe before training. Timing estimates include both
validation roles every epoch and OFFLINE export. Exact pushed source,
scheduler dry review, immutable submission journals and explicit launch
authorization are required. Queue readiness does not claim completed Oscar
GPU validation. Poor scientific scores never fail a job.
