# Five matched multi-teacher trials, SPORC tier3, v1

User authorization: implement five small-dataset runs and allow submission with
no new preflight. This plan supersedes the generic new-cluster-miniature gate
ONLY for this direct experimental route; old accepted SPORC evidence, exact
source, dry review, finite inputs and authentic artifacts remain mandatory.
The Oscar FullStat and proposed relational documents are donors, not execution
authority for this small-data campaign. No existing campaign is modified.

## Science

Original trial ALLJETS 299234 TRAIN / 74895 validation, natural class ratios,
original50k report-only; historical test retired, NO independent test. Reuse
fixed30 M0HLT seeds, physical batch128, all30 epochs, AdamW and per-update
warmup3/cosine schedule. Fresh students; no logit KD, warm start, attention,
fusion, sampling changes, stopping by quality or altered checkpoint selection.
Select by entire-validation AUC and original tie breaks.

Frozen selected teachers: OFFLINE and COARSE_U100 from original fixed30 source
74aa58c5; LOGIT_D066 and LOGIT_D033 from completed attention comparison94f56025.
Do not substitute ATTN, fusion or Oscar teachers. Authenticate retained native
checkpoints against original completed receipts, reports and predictions even
when historical attention banks have been retired. No teacher retraining.

All students receive ONLY D000 HLT inputs. Four independent 128-GELU-128 heads
read the native pre-classifier128-vector; targets/predictions never enter the
classifier. Each target is its teacher's FP32 eval native vector, standardized
independently using TRAIN FP64 population moments with scale floor1e-3.
Export once, join by exact ordered TRAIN identities and labels, share read-only.
Selected deployment checkpoints exclude every auxiliary head.

| Arm | Per-head target | Total pointwise coefficient | Relation |
| --- | --- | ---: | --- |
| multi_w25 | four distinct teachers | 2.5 | none |
| multi_w5 | four distinct teachers | 5 | none |
| multi_w10 | four distinct teachers | 10 | none |
| u100x4_w5 | four independent heads, same COARSE_U100 target | 5 | none |
| multi_w5_rel | four distinct teachers | 5 | below |

Pointwise loss = CE + total/4 * sum of four mean SmoothL1(beta1) losses;
means run over jets AND128 dimensions. Same four head seeds across all arms;
CPU forked RNG leaves native initialization/dropout/sampling unchanged. U100
head retains original20261019 seed, others +1,+2,+3. Head slots, not target
names, determine seeds. No claim of equal gradients from equal coefficients.

Relation arm: for each teacher separately compare normalized predicted query
and normalized true query against that same teacher's frozen TRAIN anchors.
Forward KL of reference softmaxes, temperature0.2, sum over anchors and mean
over jets, then MEAN the four teacher KLs. Add beta_u times this mean where
beta_u=min(1,u/(2*ceil(N/128))). No extra temperature-squared factor. Teacher
and anchor tensors detached. Never compare unaligned teacher coordinates or
average their vectors. Pointwise coefficient stays5 throughout.

Use2048 shared reference ROWS per update:186 without replacement per class,
plus1 for each of classes(2*s)%11 and(2*s+1)%11, s=zero-based global update.
An independent per-update NumPy PCG64 SeedSequence([314159,epoch,batch_index])
selects rows; original student RNG/order unchanged. Mask exact same-jet row
on both sides before softmax. No hard negatives, query-label conditioning,
validation anchors or reference inference. Require >=187 TRAIN rows/class;
fail rather than silently altering anchor distribution. All relation math FP32.
This changes supervision emphasis, adds no new teacher information, and offers
no performance guarantee. Poor finite metrics always publish.

## Execution and durability

One reviewed six-job tier3 DAG: export_targets, then five independent fits
with afterok and kill-on-invalid-dep. No preflight, technical epoch, controller,
second manual submission or automatic retry. Export does genuine target work,
not an admission benchmark. Source/account/job/dependency checks fail closed.
Existing accepted OFFLINE-w5 SPORC miniature and resource report authenticate
unchanged native inputs/recipe/environment. New multi-head/relation timing is
NOT measured: request6CPU/90000MiB/A100, export6h, fits max(360,2*accepted w5
minutes+60), rejecting >24h. This is a conservative estimate, not a guarantee.
Per-update finiteness and strict native checkpoint readback remain enabled.

Reserve4GiB persistent headroom. One set of four TRAIN banks (~0.57GiB vector
payload), five selected native checkpoints, predictions, full histories and
lineage; no optimizer snapshots, attention matrices, persistent particle cache
or automatic deletion. Deleted OFFLINE TRAIN cache can be reconstructed in RAM
with original fingerprints; missing/partial other views fail explicitly.
Never touch shared teachers/data, old campaigns or Fresh_check.

Reports support partial completion and full validation/original50k AUC,
accuracy, recovery, R30/R50 with QCD passing counts/censoring; every metric uses
the same full-validation-AUC-selected checkpoint. Imported M0HLT/OFFLINE and
OFFLINE-w5 are historical references, not newly trained controls. Optional
completed U100-w5 reference must authenticate original direct-job results.
No fixed pasted metrics. New result includes deltas vs local multi_w5 when
available. No sealed test is opened and no best-on-test claim is possible.

Verification: focused donor regressions, five-arm objectives/gradients/RNG,
TRAIN masks/normalization/corruption, native deployment parity, local real
installed-Weaver production-worker miniature, source/Slurm journals and shell
syntax. Local GPU evidence is not claimed to be real SPORC acceptance.
