# Small OFFLINE SupCon-5 transfer campaign

Status: local validation passed; ready to commit/push and queue a fresh measured SPORC setup.
User requested a small SPORC/debug study, starting at teacher SupCon coefficient
5 and predicting both U100 and the new
teacher embedding. The user explicitly selected two fits only: one SupCon-5
teacher and its student. Reuse completed U100-w5 job21993056. No CE-only teacher
fit, larger strength sweep or fresh baseline is included.

| Fit | Input | Objective |
|---|---|---|
| teacher_supcon5 | pure OFFLINE | CE1 + SupCon5 |
| student_supcon5 | D000 HLT | CE1 + U100 embedding5 + teacher_supcon5 embedding5 |

All fits are fresh native 17-input, 11-class ParT, batch 128, 30 full epochs,
original AdamW/three-epoch warmup/cosine recipe, 299234 TRAIN/74895 validation.
Teacher uses original OFFLINE seed/sampler plus 200000; student uses the
completed U100-w5 initialization/sampler. Head initialization is RNG-isolated.
Teacher projection uses seed 84221 and Linear128-GELU-Linear128; loss consumes
its L2-normalized output at temperature 0.1. Same-class TRAIN examples excluding
self are positives. Average log probability over positives and eligible anchors;
singleton anchors are excluded and an all-singleton batch contributes zero.
No augmentation, class-balanced resampling, momentum bank or teacher loss ramp.
This preserves the earlier teacher loss definition except coefficient 0.1 -> 5.

Each teacher is selected by full-validation native AUC with original tie rules.
Students predict its native 128D preclassifier embedding, not the projection.
Export selected frozen teacher TRAIN embeddings in FP32; per-channel population
mean/std are fitted on TRAIN in FP64 with std floor 1e-3. Separate student heads
predict the unchanged original U100 targets and new OFFLINE targets with mean
SmoothL1(beta 1). The original U100 head retains seed 20261019; the new OFFLINE
head uses seed 2026110601. No coordinate averaging, logit KD or attention modification.
Teacher projection and student prediction heads are removed from native export.

Original50k overlaps validation and is report-only. No independent test exists;
all ALLJETS disclosures propagate. OFFLINE validation is permitted only for
teacher selection/technical inference; it never supplies student targets.
Poor finite teacher/student quality never blocks an experiment. Nonfinite
execution, invalid inputs, stale source and lineage mismatches fail closed.

Report weighted CE/SupCon and CE/U100/OFFLINE losses, first-batch shared-encoder
gradient norms/cosines at epochs 1, 3, 10, 20, 30, teacher oracle metrics, student
recovery and deltas vs saved U100-w5. This single-seed screen tests the combined
addition, not the causal effect of SupCon versus adding a CE-only teacher.

Execution explicitly opts into SPORC debug/reu-aisocial/qos_tier3, one A100,
six CPUs and 90000 MiB per GPU worker, atlas_kd_sporc. This site-specific plan
supersedes generic Tigris defaults. A real shared GPU setup measures warmed
production paths, worst support, installed-Weaver/native parity and reload;
no science is released before successful setup accounting and acceptance.
Setup requests 12h, each measured fit at most 24h. New output allowance 8 GiB plus
1 GiB free headroom. No transfer or deletion of existing data is required.
Setup measures student kernels with technical target arrays; each student also
checks real selected-teacher targets before fresh training. Technical weights
are discarded. The student depends only on the teacher; the report follows
both fits. A detached SPORC dispatcher
handles measured release after setup. Exact journals prevent duplicate retries.

Authoritative remote qualification remains required; local tests alone do not
establish SPORC resource feasibility or scientific benefit. See the
[runbook](../LUKA_SUPCON5_DEBUG_RUNBOOK.md),
[contract](../contracts/LUKA_SUPCON5.md) and
[validation evidence](../LUKA_SUPCON5_VALIDATION.md).
