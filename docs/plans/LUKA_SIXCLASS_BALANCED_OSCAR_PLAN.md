# Six-class balanced Oscar ladder

Status: implemented and locally validated; ready to commit/push and queue. User
authorized this campaign on 2026-11-16; no live submission was performed. This plan supersedes
neither the eleven-class Hydra campaign nor existing Oscar executions.

## Population and sampling

Use the verified `luka_qcdrun2com_higgs2p_sb_v1` upload on Oscar scratch. Retain
QCD, X_bb, X_cc, X_ss, X_qq, X_gg, in that order (labels 0 through 5). Other
classes are excluded; raw label 206 remains quarantined. All six models have
six outputs. This changes the task, not just the sampling weights.

Replay the existing foundation and its frozen unused-only Hydra extension
(`7f2413152ecb588764c862b4b83bd6f900320a1b915a3b77f95540dd6e283939`).
Preserve every extended role. Select `600 <= offline jet_pt < 2600` GeV;
do not refill or move rows between TRAIN, validation, test, or unused.

The local scalar-metadata audit authenticated all 6,996 foundation membership
files and their registered extended roles. No particle arrays or test predictions
were read. In class order above, the in-range populations are:

| Role | QCD | X_bb | X_cc | X_ss | X_qq | X_gg | Total |
|---|---:|---:|---:|---:|---:|---:|---:|
| TRAIN | 1,181,123 | 99,000 | 95,790 | 93,837 | 93,383 | 95,804 | 1,658,937 |
| Validation | 233,144 | 19,574 | 19,031 | 19,022 | 18,598 | 19,220 | 328,589 |
| Sealed test | 579,833 | 50,046 | 48,593 | 47,809 | 47,393 | 48,639 | 822,313 |
| Unused | 47,542 | 5,691 | 6,029 | 6,153 | 6,386 | 6,233 | 78,034 |

Outside-range counts for these same classes/roles are respectively 1,253,533,
237,443, 610,553 and 103,443. Outside validation includes *all extended*
validation rows outside the new range, including newly excluded low-pT rows.

TRAIN uses replacement sampling uniformly over six classes and then twenty
100-GeV bins and uniformly over rows within a cell. There are 120 nonempty
cells. Use 1,658,937 draws/epoch, batch 128, final batch 57, 12,961 updates/epoch,
20 epochs. All fits use identical deterministic draw orders. No additional loss
class/pT weighting. Sparsest cell: X_qq at 600–700 GeV, 3,413 rows; maximum
expected appearances per row/epoch = 1,658,937 / (120 * 3,413) = 4.050535.

Validation stays natural and unweighted inside the class/range selection.
Select one checkpoint using maximum six-class macro OVR AUC, then minimum CE,
maximum macro log R50 and earliest epoch. Evaluate R50, R30, accuracy, AUC and
per-pT-bin diagnostics at that checkpoint; outside validation cannot select it.
Final-test inference remains sealed, including outside-range test inference.
Fresh same-population M0HLT/OFFLINE are recovery anchors. Eleven-class historical
aggregate metrics are not matched comparisons to this six-class experiment.

## Six fresh fits and supervision

| Fit | Inputs | Main-head predecessor | Auxiliary teachers |
|---|---|---|---|
| OFFLINE | offline | none, CE only | none |
| M0HLT | HLT | none, CE only | none |
| CONCAT | source-blind HLT + offline multiset | none, CE only | none |
| UE100 | U100 on HLT support | OFFLINE | OFFLINE, CONCAT, M0HLT |
| DE050 | half-offline/half-HLT D050 | UE100 | OFFLINE, CONCAT, UE100, M0HLT |
| DE000 | HLT | DE050 | OFFLINE, CONCAT, UE100, DE050 |

Fresh initialization throughout. CONCAT retains every particle from both views
without a source flag, preserving the existing union semantics. U100/D050 use
the frozen matching/view construction. No FUSION, subclasses or retained heads.

Baselines optimize CE with coefficient 1. At each student rung:

`L = .25 CE(main,y) + .75 T^2 KL(predecessor_T || main_T)`
`    + (10/n) sum SmoothL1(predicted_embedding, teacher_embedding)`
`    + (1/n) sum T^2 KL(teacher_T || auxiliary_logits_T)`
`    + contrastive_DE000`, with `T=2` and `n` the stage's teacher count.

Native 128-dimensional teacher embeddings are standardized using unique eligible
TRAIN rows, FP64 population moments and standard-deviation floor .001. Each
teacher has independent embedding and auxiliary six-logit heads. Teacher
probabilities are already softened at T=2; do not soften a second time. CE and
predecessor KD directly train the deployed classifier. Auxiliary heads and all
teacher inputs are removed for inference; M0HLT/DE000 consume HLT only.

Final DE000 additionally matches pure-OFFLINE TRAIN keys: temperature .1, weight
.3 with a three-epoch linear ramp, 256 hard + 256 same-class + 512 global distinct
negatives. Other-class hard proposals are balanced over the other **five**
classes (4096 candidates; select ranks 257–1024), broad proposals over all six.
Occurrence-keyed sampling distinguishes repeated replacement draws. No test or
validation candidates, duplicate negative identities, or new event claims.

Keep AdamW, learning-rate peak 3e-4/floor 1.5e-5, two-epoch warmup and cosine
through epoch 20, BF16 forward/FP32 losses, full constituent support, and existing
native/auxiliary seed domains. This is fewer updates than the old campaign.

## Execution and recovery

New versioned package/artifact namespace; old execution artifacts fail closed as
new scientific inputs. Reuse implementations only with pinned donor provenance.
Raw data on scratch; source, persistent Python runtime and results in HOME.
Particle views are rebuilt into bounded RAM, not duplicated to HOME caches.
Reserve 14 GiB for results; restoring a missing HOME runtime requires the separate
existing 28-GiB combined envelope and exact recorded package builds. Never run
the old campaign as a side effect of runtime restoration.

Six science jobs, zero standalone preflight jobs. Each science allocation first
performs real installed-Weaver six-output forward/backward parity, a discarded
two-epoch miniature, maximum-support gradient/export stress and measured memory/
runtime acceptance. Initial requests: one L40S, six CPUs, 192000 MiB, 48 hours.
Queue dependencies authenticate all predecessor results; poor metrics are valid
results, not cancellation conditions. Epoch optimizer/model/RNG commits support
exact continuation; export-only recovery does not retrain completed epochs.

Detached registration/dry review/submission survives SSH disconnect. Require
exact pushed source, immutable spec, dataset/runtime checks, quota declaration,
site review and journaled single submission. Provide queue/resume/inspection
commands. Production GPU acceptance remains unclaimed until the jobs execute it.
