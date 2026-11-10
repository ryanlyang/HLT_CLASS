# Oscar FullStat soft subclasses, version 1

This user-authorized experiment adds one fresh HLT-only student to the completed
Multi3 + OFFLINE contrastive strategy. It uses 3M TRAIN / 500k development
validation, batch128 and20 epochs, with the original initialization, row order,
optimizer, two-epoch warmup/cosine and AUC checkpoint selection. No teacher fit,
fresh control or separate preflight is registered. Final test remains sealed.

## Objective

Keep CE1, U100/CONCAT/FUSION embedding SmoothL1 weights10/3 each, and pure-OFFLINE
mix instance matching0.3 at temperature0.1. Add conditional soft-subclass
KL(q_teacher || p_student), coefficient0.5. Matching and subclass coefficients
both ramp linearly over the first3 epochs, using one-based updates. All losses
are means across jets. The new Linear(128,60) head reads native h directly;
QCD has10 outputs and each of10 signal classes has5. Only the true class's group
is normalized and supervised. Labels select a loss group, never a forward input.
All auxiliary heads are removed for unchanged native HLT ParT inference.
Initialize the subclass head in an isolated CPU RNG context (seed2026110901),
preserving native and existing auxiliary initial tensors and dropout RNG.

## Frozen OFFLINE targets

Authenticate the same pure-OFFLINE TRAIN bank and selected native classifier.
Reconstruct cached raw embeddings from standardized FP32 targets and their
saved FP64 mean/scale, then apply the frozen linear classifier in FP64. These
are readout scores reconstructed from a quantized cache, not a fresh exact
particle-forward replay. Compute10 margins: other-class logit minus true-class
logit, ordered by canonical class index excluding the true class.

Within each true class, cluster features composed of sqrt(.75/128) times the
existing TRAIN-standardized embedding and sqrt(.25/10) times the margins.
Margin mean/population standard deviation are fitted on that class's TRAIN
rows only, with floor0.001. Squared distance therefore weights mean squared
embedding and margin distances75/25. Use all TRAIN rows, never validation.
Fit sklearn MiniBatchKMeans separately per class: seed2026110902+class index,
batch4096, n_init5, max_iter40, max_no_improvement20, reassignment_ratio0,
one numerical thread. Do not enforce equal occupancy or drop rare subclasses.

Soft targets are softmax(-squared distance/class temperature). Choose each
temperature by48 log-space bisections within[1e-8,1e8] to make the TRAIN median
maximum membership0.8. If ties/degeneracy make that unattainable, choose the
nearest bound and report achieved membership; finite poor clusters never fail
the experiment. Persist prototypes, statistics, temperatures, occupancy,
entropy, distance summaries and compact FP32 padded10-column target rows.
Class labels mask padding; no hard subclass labels or learned online targets.

## Execution and interpretation

Use one Oscar L40S/6CPU/192000MiB/48h allocation. Target fitting runs once,
before the unchanged sampler and actual full-update resource admission. No
teacher encoder is evaluated inside training. Record target preparation time
and memory in admission. Integrated installed-Weaver, deployment and gradient
checks use disposable models. Scientific training starts from scratch afterward.
Save full model/optimizer/RNG each epoch. Explicit same-source recovery copies
the authenticated target bank and epoch prefix into a fresh result root, with
publication-only recovery after20 completed epochs. No automatic retry.

Historical comparisons include M0HLT, OFFLINE, U100-w5, Multi3 and Multi3 +
contrastive. Report accuracy/AUC and R50/R30 at the same selected checkpoint.
This tests added subclass supervision, not guaranteed recovery or discovered
physical truth. Prototype occupancy and achieved softness are diagnostics,
not quality gates. This is a frozen-prototype adaptation inspired by
[Subclass Distillation](https://arxiv.org/abs/2002.03936), not its exact algorithm.

This active Oscar plan supersedes legacy Tigris-only acceptance language for
this campaign. Local synthetic checks do not constitute real Oscar acceptance.
