# Replacement LOGIT teachers and five-arm screen, SPORC tier3

Authorized 2026-10-23 after the original LOGIT_D066 and LOGIT_D033 selected
checkpoints were missing, no matching copies were found in the searched tree,
and the older D066 embedding bank had no payload. User approved retraining just
these two teachers and automatically continuing the five planned experiments.
This plan supersedes the original trial's prohibition on teacher retraining
ONLY for this separately registered recovery. Old campaign artifacts stay
immutable. It is not a claim to restore bit-identical historical weights.

## Science

Unchanged ALLJETS 299234 TRAIN / 74895 validation, natural class ratios,
original50k report-only, historical test retired; NO independent test.
Authenticate surviving original fixed30 OFFLINE and COARSE_U100, and retained
original attention-campaign metadata/results in the original source. Deleted
selected weights and old technical/attention target arrays are not prerequisites.

Fresh LOGIT_D066 learns from surviving COARSE_U100, then fresh LOGIT_D033 learns
from the newly selected LOGIT_D066. Reuse the original `luka_attention` LOGIT
trainer, coordinate seeds, AdamW, physical batch128, exact30 epochs, fixed30
per-update warmup3/cosine, C25/P75 temperature2 KD, zero attention coefficient,
and full-validation AUC/CE/logR50/update selection. No warm start, changed
sampler, extra run selection or requirement to beat the deleted teacher.
The unchanged trainer still computes its native attention diagnostic; it has
zero weight, no attention teacher targets and no persistent attention bank.
All finite fits complete30 epochs even if quality is worse.

Export original U100 and new D066 TRAIN temperature2 class probabilities in
each consuming teacher job. These compact input banks remain with receipts.
Keep each selected native checkpoint, history, predictions, costs and exact
parent lineage. Never write replacements into the old attention campaign.

Once both receipts validate, publish a separately hashed execution context
binding the two original plus two replacement native checkpoints. Export the
four TRAIN native128-vector banks and run the unchanged five recipes from
[the original plan](LUKA_MULTITEACHER_TRIAL_PLAN.md): total weights2.5/5/10,
four-head same-U100 weight5, four-teacher weight5 plus relation. All students
are fresh HLT-only, CE plus embedding objectives, NO student logit KD.
Native inference has no auxiliary heads or teacher access.

## Execution

One eight-job exact DAG, all submitted at once on tier3:

```text
train_LOGIT_D066 -> train_LOGIT_D033 -> export_targets -> five parallel arms
```

Each teacher job includes its own compact probability export. Afterok plus
exact successful accounting and artifact checks gate each dependent job.
No extra preflight, controller, benchmark epoch or manual follow-up submission.
No automatic retry, cancellation, scheduler priority changes or cleanup.
Partial/ambiguous submissions keep their exclusive claim and journals.

Original single-head resource evidence and archived attention timing metadata
are inherited, not new measurements or revalidation of deleted probes. Exact
environment/GPU equality is required. All jobs use A100/6CPU/90000MiB. Teacher
request=max(360, archived train request + archived export request) minutes;
export6h; student=max(360,2*accepted single-head request+60) minutes. Reject any
request >24h. Full dry review, clean pushed source and explicit eight-job
authorization remain mandatory. This direct route is the user-approved
exception to a new preflight; no claim of new SPORC miniature acceptance.

Reserve5GiB for two compact probability banks, four shared embedding banks,
seven selected models, histories, predictions and metadata. Retain selected
teachers/models; no automatic deletion. Do not duplicate particle caches.
Authenticate required TRAIN D000/U100/D066/D033 and validation D000/D066/D033
blocks and the raw TRAIN sources before creating the campaign. OFFLINE TRAIN
is rebuilt only in process RAM by the original five-arm exporter.
Shared datasets, historical campaigns, pinned worktrees and Fresh_check are
read-only. Missing/corrupt required inputs stop explicitly.

Saved-only partial reports label NEW_LOGIT_D066/D033 as replacement teachers,
and preserve historical CE/OFFLINE/OFFLINE-w5 comparisons. Every model uses its
own same full-validation-selected checkpoint for AUC, accuracy and R30/R50.
The new multi-teacher results are not an exact historical-teacher replication.

Verify unchanged donor recipes, new source/contracts, exact DAG/no duplicate
submit, retired-weight tolerance, all required block checks, parent corruption,
native teacher readback, execution-context binding and local real-Weaver
teacher-to-student worker integration before handing off queue instructions.
