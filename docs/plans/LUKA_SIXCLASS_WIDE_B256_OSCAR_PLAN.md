# New-data six-class ladder: 200–2600 GeV, LR 1e-4, batch 256

This is the active scientific plan for the user's November 23 campaign. It is
a new six-fit experiment on the new QCD Run-2 combined + Higgs2p dataset.
It does not revise the identities or interpretation of historical executions.
This plan supersedes the 600–2600 GeV range, batch 128, direct-main-KD ladder
and three-fit/external-teacher scope of the earlier six-class plans **only for
this new campaign**. Every deployed classifier here is trained with unscaled
true-label CE. Teacher logits remain separate auxiliary targets.

## Population and sampling

- Dataset: `luka_qcdrun2com_higgs2p_sb_v1`, already transferred to Oscar scratch.
- Outputs, in order: QCD, X_bb, X_cc, X_ss, X_qq, X_gg. The user's five-class
  choice means five signal classes plus QCD: six classifier outputs.
- Preserve the previously registered extended split roles; no new random split,
  refill, movement between roles, or use of final-test particles.
- Apply `200 <= offline reconstructed jet_pt < 2600` GeV. Use 24 bins of 100 GeV.
  This offline scalar defines TRAIN sampling and evaluation subsets, never a
  deployable model feature.
- TRAIN: **2,488,148 eligible unique jets** and **2,488,148 draws per epoch**.
  Uniform independent class/bin-cell selection, followed by uniform replacement
  draw within that cell. All six fits receive the same epoch draw sequence.
  No additional class or pT weighting in any loss.
- Batch 256, no accumulation/drop-last: **9,720 updates/epoch**, final batch 84.
  Duplicate draws are independent training occurrences; contrastive RNG uses
  occurrence positions so repeated copies do not share a forced negative set.

The scalar-only local audit authenticated 6,996 foundation membership files and
replayed the frozen extended roles. Counts are:

| Role | Inside 200–2600 | Outside | Total six-class role |
|---|---:|---:|---:|
| TRAIN | 2,488,148 | 424,322 | 2,912,470 |
| Validation | 497,954 | 68,078 | 566,032 |
| Sealed final test | 1,227,412 | 205,454 | 1,432,866 |
| Unused | 78,270 | 103,207 | 181,477 |

TRAIN counts are QCD 1,962,846; X_bb 109,657; X_cc 105,367; X_ss 102,755;
X_qq 102,146; X_gg 105,377. All 144 class/bin cells are populated. The sparsest
cell is X_qq, 200–300 GeV, 805 jets. Its expected appearances/jet/epoch are
**21.4644**. This is an expectation, not a hard repetition cap. Expected unique
TRAIN jets seen in one replacement epoch: **867,818**, with **65.12%** repeated
draws. Keep this exposure visible; do not silently soften the approved sampler.
Raw label 206 and excluded signal labels stay excluded under the original map.

## Fresh fits and supervision

| Stage | Input view | Auxiliary embedding and logit teachers |
|---|---|---|
| OFFLINE | Pure offline particles | None |
| M0HLT | HLT particles | None |
| CONCAT | Source-blind union of HLT and offline particles | None |
| UE100 | HLT support, matched offline measurements | OFFLINE, CONCAT, M0HLT |
| DE050 | Same hybrid construction at half offline contribution | OFFLINE, CONCAT, UE100, M0HLT |
| DE000 | HLT particles only | OFFLINE, CONCAT, UE100, DE050 |

All six initialize from scratch using the registered native ParT architecture
and original seed domains. No historical checkpoint or target bank is reused.
The first three fits have no teacher dependencies; later fits wait for every
listed teacher to publish authenticated selected TRAIN banks. Scheduler order
between the three independent baselines is not promised. There is no FUSION
teacher, subclass branch, retained predictor or teacher at deployable inference.

For a student with N teachers:

`L = CE(main,y) + (10/N) sum SmoothL1(predicted_embedding, standardized_target)
     + (1/N) sum 4*KL(teacher_T2 || auxiliary_head_T2) + matching`

N=3 for UE100 and N=4 for DE050/DE000. Baselines use CE alone. The main FC
receives **only CE**, coefficient 1, for every stage. Auxiliary losses still
train the shared encoder. Targets and teacher probabilities are detached.
No predecessor KD term is applied to the deployed classifier.

Only DE000 adds pure-OFFLINE contrastive matching: coefficient 0.3 with the
original three-epoch linear ramp, temperature 0.1, positive same-jet OFFLINE
key plus 1,024 negatives (256 hard-other-class, 256 same-class, 512 global).
The six-class proposal pools, native logits and optimized occurrence sampler
retain their historical semantics. Keys are from **fresh selected OFFLINE**.
Duplicate occurrences cannot become their own negatives.

Target banks export every unique eligible TRAIN row once, using native FP32
128-dimensional embeddings and native BF16 probabilities softened at T=2.
Export batch remains 128 (separate from the training batch); standardization
uses unique eligible TRAIN FP64 population means/stds with 0.001 scale floor.

## Optimization and evaluation

Each fit runs all 20 draw epochs. AdamW, betas (0.9,0.999), eps 1e-8, decay
0.01. Peak LR **1e-4**, floor **5e-6**, two-epoch linear warmup and cosine decay
to epoch 20. Schedule position is actual optimizer updates divided by 9,720.
Batch 256 does not cause automatic LR scaling. Forward BF16, losses FP32.

Primary checkpoint selection uses the **natural, unbalanced 497,954 in-range
validation jets**: maximum six-class macro OVR AUC, then minimum CE, maximum
macro log R50, earliest update. No early stopping or performance gate.

Every epoch saves primary metrics, pooled **full-validation macro AUC and
accuracy**, full Hbb-versus-QCD AUC/R50/R30 diagnostics, loss/gradient snapshots,
sampling evidence, optimizer/model/RNG state. Pool predictions to compute full
metrics; never average the two subset AUCs. Hbb-selected epoch is diagnostic
metadata only and does not change the checkpoint used by the ladder.

At the primary selected checkpoint, publish in-range, outside-range, pooled
full-validation and per-bin results. R50/R30 use `p(signal)/(p(signal)+p(QCD))`,
descending signal ranks and inclusive ties. Zero passing QCD is censored.
Recovery uses the fresh M0HLT/OFFLINE selected checkpoints on the identical
population. Nonpositive anchor gaps report n/a. No final-test evaluation is
implemented or authorized by this plan.

## Oscar execution

New owned namespace: `luka_sixclass_wide_b256_oscar`. Six science jobs, each
one L40S, six CPUs, 192000 MiB, 48-hour limit, partition gpu / norm-gpu. No
standalone preflight job. Each science allocation first runs real installed
Weaver parity, a two-epoch production-model miniature, batch-256 maximum-support
stress, export parity and measured time/memory acceptance. Probes are discarded
before fresh initialization. There is no quality threshold.

Raw data stay on scratch; pinned source, persistent Python environment and new
results live in HOME. Require **18 GiB free HOME for outputs** (32 GiB if the
existing persistent environment also needs creating). Five banks consume about
6.7 GiB before checkpoints, row arrays, predictions and operational margin.
No cleanup of historical models/data is part of this campaign.

Every epoch commits complete student/optimizer/RNG state and a completion-last
receipt. Explicit resume after terminal accounting continues committed epochs
and can finish an export without retraining. Journals reject ambiguous or
duplicate submissions; no blanket cancellation or automatic job resubmission.

Queue readiness means local implementation/tests, immutable audit, launchers
and instructions are prepared. Exact pushed source and current Oscar quota
are supplied at launch. Actual L40S batch-256 peak memory and full-data timing
are measured inside the science jobs; local synthetic CUDA evidence is not a
claim of completed Oscar production acceptance.
