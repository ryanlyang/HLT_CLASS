# Frozen LOGIT_D066 embedding prediction, weight 5, debug

Authorized by the user on 2026-10-21: one fresh HLT-only CE + embedding
prediction fit at weight5, replacing the pure-OFFLINE target with the frozen
validation-selected LOGIT_D066 teacher. No logit KD in the new student.

Reuse exactly trial ALLJETS299234 TRAIN/74895 validation, original50k report-only,
original M0HLT seeds, batch128, AdamW and fixed30 schedule. The historical test
was retired; there is NO independent test. Teacher selection and weight choice
are adaptive development decisions, not an unbiased final evaluation.

Teacher is LOGIT_D066 from the original94f56025 attention experiment, NOT
ATTN_D066, COARSE_D066, or a technical probe. Authenticate retained aggregate,
reports/predictions and original source using the existing archival reader;
additionally authenticate the selected checkpoint against its original receipt
and reproduce its validation predictions with the native model. Deleted old
attention banks/preflight payloads are not reused or reconstructed. Missing
checkpoint is a hard blocker, never permission to substitute another teacher.

Only TRAIN D066 inputs generate the final normalized128-vector before mod.fc.
Freeze teacher/eval mode/FP32/no autocast. Compute new TRAIN-only population
means/stds in FP64 with floor1e-3 and store FP32 targets. Do not reuse OFFLINE
normalization. Student is native HLT with the same isolated-RNG small predictor;
CE+5*mean SmoothL1(beta1) updates backbone and predictor. Predictions never
enter the classifier. Save native selected weights without the predictor.
Full-validation AUC-first selection, all30 epochs, no warm start or quality gate.

Primary comparator is the completed OFFLINE-target weight5 fit in the transferred
debug sweep; CE and pure OFFLINE are secondary references. Authenticate exact
recipe/seeds/population and retained predictions, never substitute rounded log
values. New BANK/RECIPE/records use LUKA_D066_EMBEDDING_*/v1. Existing scientific
runtime and all old campaign records remain immutable.

One debug A100 preflight (6CPUs,90000MiB,8h) exports the compact target bank,
checks native/predictor parity, TRAIN and validation maximum-support stress,
one complete technical epoch and native checkpoint readback. A separately
measured debug science job uses the unchanged30-epoch schedule. Full science
requires exact successful preflight journal/accounting and resource admission.
The opt-in followup may submit it automatically under an exact reviewed graph
and24h ceiling, never initialize it from technical weights.

Reserve2GiB durable headroom. Reuse existing D066 TRAIN and D000 TRAIN/validation
caches read-only; no new particle caches or attention banks. Retain compact
TRAIN targets (~146MiB), one selected native state, histories, predictions and
receipts. No automatic deletion of any files.

The user additionally requests this experiment ahead of their remaining debug
sweep. Separate priority authorization is limited to journal-bound w50/w100/
w200/w500/w1000 jobs from the transferred007885f7 sweep. Review identities and
pending states before holding; never cancel/suspend a running job or modify
Fresh_check/other campaigns. A durable controller owns only holds it created,
releases them after the new fit reaches a terminal state (also on preflight
failure), and records every operation. Ambiguous operations require inspection,
not blind retries. Host loss can strand holds: provide exact-ID recovery.
This orders the user's selected pending work, not other users or reservations.

Required verification: focused unchanged-donor baseline; real installed-Weaver
local CUDA miniature through new workers; teacher/role/identity/normalization
guards; same-target parity with weight5 donor; native checkpoint and reporting;
source/resource/Slurm dry/live authentication; hold/release scope and failure
injection. Genuine SPORC preflight remains mandatory. No jobs are changed by
implementation alone.
