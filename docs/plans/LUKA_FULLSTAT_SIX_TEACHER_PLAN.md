# Oscar FullStat six-teacher subclass student

Status: implemented and locally validated; scoped commit/push and Oscar launch
remain. No jobs submitted. This is the active scientific plan for the
separate `luka_fullstat_six_teacher` campaign. It does not change the Hydra plan
or reinterpret any existing Oscar run.

## Authorized experiment

Train one fresh native HLT ParT on the existing authenticated 3,000,000 TRAIN
and 500,000 development-validation jets, for 20 epochs, batch 128, with the
historical initialization, sampling, optimizer and learning-rate schedule.
Use the original subclass campaign (16546c0e, successful job 8761518) as the
historical reference. No fresh control or teacher training is requested.

The deployed classifier receives CE with coefficient 1. Auxiliary heads share
the HLT encoder but neither their predictions nor real teacher inputs enter
the deployed classifier. All auxiliary heads are removed at inference.

| Frozen teacher | Embedding coefficient | Auxiliary logit coefficient |
| --- | ---: | ---: |
| Pure OFFLINE | 10/3 | r/6 |
| CONCAT | 10/3 | r/6 |
| FUSION | 10/3 | r/6 |
| Hybrid UE100 | 5r/3 | r/6 |
| Previous split-head DE066 | 5r/3 | r/6 |
| Previous split-head DE033 | 5r/3 | r/6 |

Here `r=min(update/(3*ceil(3000000/128)),1)`. The total embedding coefficient
reaches 15 and total auxiliary KD reaches 1. Each embedding head is an
independent 128→128 GELU→128 MLP predicting that teacher's native representation
standardized with its own frozen TRAIN mean/scale, using mean SmoothL1(beta=1).
Each logit head is a separate linear 128→11 classifier. KD is forward
KL(teacher || student), T=2 with T² applied exactly once, class sum/batch mean.

Keep pure-OFFLINE contrastive matching at 0.3r, with the original exact sampler,
1024 negatives, temperature 0.1, TRAIN identity exclusions and matching head.
Keep conditional subclass KL at 0.5r, using the exact original OFFLINE-derived
soft targets/prototypes (10 QCD groups, 5 per signal class). Do not recluster.
Do not include original COARSE_U100, M0HLT, DE022 or DE011 as supervision.
The name UE100 specifically means the hybrid teacher, not COARSE_U100.

## Reuse and scientific identity

Authenticate the original subclass completion in its original source. Also
authenticate completed fine-tail 04d1f0f5 in its original source to obtain the
precise hybrid UE100 and previous split-head DE066/DE033 checkpoint and bank
references that were reused by the prior best split-head DE000. The fine-tail
DE022/DE011 banks and student weights are not training inputs to this experiment.
Require identical registered population, OFFLINE checkpoint, environment/site,
ordered identities and labels. Keep every historical manifest and source intact.

Reuse six standardized embedding banks and four existing T=2 logit banks.
CONCAT/FUSION have no saved native logit banks: export them once from their
selected frozen models on TRAIN only, using their real native particle views,
the historical BF16-forward/FP32-softmax path, and batch 128. Never approximate
these logits by inverting standardized embeddings. Atomic payload+manifest
publication, source/checkpoint/view hashes and exact row joins are required.
Include both exports in the allocation, resource timing and output inventory.

## Execution and acceptance

One Oscar L40S GPU, 6 CPUs, 192000 MB RAM, 48 hours, with 10 GiB result allowance
plus 2 GiB filesystem headroom (at least 12 GiB available quota). No separate
preflight/setup jobs. The user requested direct queueing with integrated
acceptance; this is the campaign-specific replacement for the generic Tigris
miniature gate, not a claim that remote validation already happened.

Before scientific updates, run real installed-Weaver parity, exact sampler
parity, worst-support gradients for all heads, native deployment reload parity,
and measured complete updates. Include export/preparation time, full validation,
publication margin and a conservative factor in the 48-hour bound. Publish
diagnostics before rejecting insufficient resources. Scientific quality never
controls whether a finite valid run proceeds.

Preserve the loss and sampler numerics: cached frozen banks, vectorized losses,
three persistent CPU candidate workers, one-batch lookahead, exact chunked GPU
mining, and one scalar readback per update. No batch/particle truncation, reduced
negative count or lowered precision is introduced as an optimization.

Save full optimizer/RNG/model states and receipts every epoch. An explicit
same-source recovery uses a fresh root, verifies unsuccessful terminal accounting,
copies authenticated exports/epochs, and resumes without repeating completed
epochs. Never automatically retry ambiguous submission or erase claims.

Use a detached launcher with a saved dry review and exact pushed clean source.
Prepare the implementation and runbook locally; do not submit jobs now. Final
test access stays sealed. Report AUC-selected accuracy, AUC, R50/R30 and recovery
against historical references. This is a combined strategy comparison; gains
cannot be attributed to one added objective from this single seed alone.
