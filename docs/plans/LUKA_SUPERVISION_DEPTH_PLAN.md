# Luka supervision-depth/diversity screen v1

User-authorized additive implementation: twenty fresh debug DE000 fits, no
standalone preflight, two new OFFLINE teachers. This specific direct route
supersedes generic new-preflight requirements, not integrity/resource checks.
No submission, cancellation, old-runtime edit or historical deletion authorized.

All students inherit the c50 DE000 physical batch 128, AdamW warmup-3/cosine-30,
30 complete epochs, 299234 TRAIN / 74895 validation and native architecture.
Full-validation AUC selects; original50k is overlapping report-only. Historical
test retired: NO independent test. Poor finite science never blocks a row.
Main classifier receives 0.5 CE only. Four separate auxiliary KD heads receive
total 1 (T=2, T-squared once), and jet predictors total 40. OFFLINE plus frozen
c50 UE100/DE066/DE033 each supply 10 embedding weight and 0.25 KD weight.
No relation, warm start, new ladder, input fusion or teacher-classifier readout.
All added heads are training-only and excluded from native selected checkpoints.

## Exact ranked students

Order balances expected usefulness with informative controls; it is not a
prediction of a statistically established winner. Every row keeps CE 0.5,
auxiliary KD 1 and total jet prediction 40. Token coefficient P is additional.

| Order | Arm | Token blocks | Total P | Change from baseline |
| --- | --- | --- | --- | --- |
| 1 | baseline | none | 0 | Exact K1/E40 repeat |
| 2 | set8_p3 | 8 | 3 | Unordered set prediction |
| 3 | set8_p10 | 8 | 10 | Stronger set prediction |
| 4 | set48_p3 | 4, 8 | 3 | Two depths, same total token weight |
| 5 | set48_p10 | 4, 8 | 10 | Two depths, stronger token weight |
| 6 | aligned8_p10 | 8 | 10 | Matched-slot control, no assignment |
| 7 | set248_p3 | 2, 4, 8 | 3 | Three depths |
| 8 | set248_p10 | 2, 4, 8 | 10 | Three depths, stronger token weight |
| 9 | baseline_seed1 | none | 0 | Second student seed |
| 10 | set48_p10_seed1 | 4, 8 | 10 | Paired second-seed candidate |
| 11 | duplicate_offline | none | 0 | Two heads share the original OFFLINE target |
| 12 | ce_diverse | none | 0 | Original + new CE-only OFFLINE target |
| 13 | supcon_diverse | none | 0 | Original + CE/SupCon OFFLINE target |
| 14 | supcon_low | none | 0 | Smaller CE/SupCon target allocation |
| 15 | supcon_replace | none | 0 | Replace only the OFFLINE jet target |
| 16 | supcon_set8 | 8 | 10 | Diversity + one-depth set prediction |
| 17 | supcon_set48 | 4, 8 | 10 | Diversity + two-depth set prediction |
| 18 | set8_p1 | 8 | 1 | Weak token-weight endpoint |
| 19 | set8_p30 | 8 | 30 | Strong token-weight endpoint |
| 20 | set48_p30 | 4, 8 | 30 | Strong two-depth endpoint |

All token arms except `aligned8_p10` use unordered matching. Each second-seed
arm is compared against its matching baseline seed, not used to select a seed.

Token targets come from c50 UE100, NOT older COARSE_U100. One-based native
particle blocks 2/4/8 (before class attention). Student predictors attach to the
same blocks. Per-depth FP64 population moments over real TRAIN tokens, floor
1e-3, raw FP32 targets. Detached exact Hungarian assignment minimizes mean
SmoothL1 cost across registered depths, one shared permutation per jet.
Differentiable loss averages channels, real particles, depths, then jets;
padding excluded and each jet equally weighted. Aligned control uses identity.
P is total token coefficient, never multiplied by depth count. No val targets.

Seed 0 preserves c50 initialization/sampler and original head RNG exactly.
Seed 1 adds 100000 to initialization/sampler/head seeds. New heads use independent
forked CPU RNG; constructing inactive/extra heads cannot perturb backbone RNG.

## Diversity controls

Two new OFFLINE native teachers, same initialization/sampler seeds (original
OFFLINE seeds plus 200000), same 30 epochs/batch 128/optimizer and OFFLINE input
rows. CE sibling uses CE 1; diverse sibling CE 1 + 0.1 supervised contrastive loss,
temperature 0.1, normalized 128-D two-layer projection from native jet h.
One TRAIN batch gives all same-class positives excluding self; anchors without
positives excluded; no-positive batch contributes differentiable zero. No
augmentation, validation positives, momentum encoder or contrastive search.
The projection is RNG-isolated and exists in both siblings; CE sibling does
not optimize it. Select both by native full-validation AUC, export native h
(not projection). Classifier quality is a result, not an admission threshold.

Original OFFLINE allocation 10 splits 5+5 (duplicate, CE sibling, SupCon sibling),
8+2 (supcon_low), or 0+10 (supcon_replace). Replacement preserves the original
head and its initialization, changing only its target. Other allocations stay 10;
original four logit-KD teachers NEVER change. Diversity+token arms use5+5 and
P=10. Separate heads and independent moments; never average incompatible
embedding coordinates. Combinations and second-seed candidate are fixed upfront.

## Execution / ownership

25 jobs: shared targets; two OFFLINE teacher fits (each exports compact TRAIN
targets); 20 ranked student fits; final report; CPU owned-bank cleanup. Student
fits are serial afterok in registered order for unambiguous priority. Teacher
fits are ordered after the tenth student, before diversity students; no other
campaign job priority changes. Technical failure blocks descendants; metrics
never gate them. No controller, automatic retry or separate preflight.

Reuse completed k1_e40_plain report/checkpoint/predictions through original
pinned grid source and exact completed accounting, including authenticated
registered retirement. Reuse c50 teacher provenance from that grid. Old files
remain immutable. Copy reference predictions; compact old teacher targets are
regenerated once in new owned directories (old banks may be retired).
Node-local canonical /tmp disk only for particle banks: each token job exports
only its required depths once, retains manifest/index and removes its owned
scratch payloads after success/failure; no home fallback. Persist compact jet
banks once; delete only those new six payloads after all science/report checks.
Native selected models, prediction files, reports, normalization/index evidence
and task receipts survive. Need 8 GiB personally reserved persistent space;
scratch capacity checked from actual TRAIN token counts before allocation.

Inherited debug A100 environment/site acceptance, 6 CPU / 90000 MiB. GPU jobs use
12h (or larger inherited allowance <=24h); CPU report/cleanup 1 CPU / 16 GiB / 2h.
These are conservative allowances, NOT new measured full-run throughput.
Shared target preparation performs native parity and physical-128 worst-support
disposable baseline/three-depth shape updates (synthetic zero token targets).
Each fit, including the OFFLINE teachers, additionally checks its
own exact objective on worst support, discards probes and resets RNG before
science. No technical epoch or checkpoint is reused for training.

## Acceptance

Source boundary e21aa91463a8f313bf51e36d6bcac97ee7942bfb permits its exact
already-committed, unrelated luka_fullstat_splithead campaign/kd_banks/submission/
worker revisions after grid028c3314. Original grid replay stays isolated in its
own pinned source; these files are not trial runtime dependencies. Every other
parent runtime byte must match. Any additional modification fails closed.

Versioned LUKA_SUPERVISION_DEPTH_* v1, immutable full dry-plan/hash authorization,
exclusive submission claim, exact IDs/dependencies and installed-source checks.
Local contracts/losses/RNG/padding/permutation/supcon/bank corruption tests,
installed-Weaver parity and production synthetic miniature required before
queue-ready handoff. Local tests are NOT real SPORC acceptance; real combined
objective resource feasibility checked inside production preparation/workers.
