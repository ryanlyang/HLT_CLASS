# Trial four-arm hybrid ladder, debug, v1

User authorized implementation on 2026-10-24, explicitly WITHOUT a new
preflight. This additive trial campaign does not modify the Oscar FullStat
hybrid, old trial experiments, their artifacts or running jobs.

## Science

Retain authenticated trial ALLJETS 299234 TRAIN / 74895 validation, natural
class proportions, original50k report-only, physical128, fixed30, original
AdamW and warmup3/cosine30 schedule. The old test was retired into TRAIN:
there is NO independent test. No new split, label selection or ROOT cache.

Four independent from-scratch chains: c25, c50, c75, anneal. CE/KD coefficients
are respectively .25/.75, .50/.50, .75/.25. Anneal uses an epochwise constant
KD coefficient `alpha(e)=.75*max(0,(25-e)/24)` for e=1..30, CE=1-alpha;
thus epoch1=.75 KD, epoch25=0, epochs26..30=0. Reset this schedule each rung;
do NOT anneal the embedding coefficient or temperature.

| Stage / view | Logit teacher | Embedding teachers |
| --- | --- | --- |
| UE000 / U000 | original OFFLINE | OFFLINE, original M0HLT |
| UE100 / U100 | this arm's UE000 | OFFLINE, original M0HLT |
| DE066 / D066 | this arm's UE100 | OFFLINE, this arm's UE100, M0HLT |
| DE033 / D033 | this arm's DE066 | OFFLINE, this arm's UE100, DE066, M0HLT |
| DE000 / D000 | this arm's DE033 | OFFLINE, this arm's UE100, DE066, DE033 |

Loss: `(1-alpha) CE + alpha * 4 KL(q_T || p_T) + (5/N) sum SmoothL1(g_i(h),z_i)`.
T=2; batchmean forward KL, exactly one T-squared factor. Each beta1 SmoothL1
is mean over batch and128 dimensions. All target tensors are detached. Weight5
is NOT multiplied by the CE/KD mixture. Teacher embedding axes are never averaged.
Each frozen teacher supplies its native128-D fc-input, not auxiliary predictions.
Independent TRAIN-only FP64 moments, floor1e-3; FP32 targets and heads. Native
teacher embedding export FP32/eval, logit export native BF16-CUDA/FP32-softmax.

All four arms use coordinate-paired native initialization and epoch sampler,
independent fixed head streams identical across arms; no teacher weight copying,
warm start, fusion, attention loss, particle-token prediction or extra inputs.
Auxiliary outputs never enter classification and are absent from selected native
checkpoints. Only DE000 is HLT-deployable; intermediate students are oracles.
Select on full-validation AUC, CE/logR50/earliest-update tie rules. Retain all30
epochs including raw/weighted losses, coefficient, timings and selected epoch.
Poor finite performance cannot fail jobs or suppress subsequent rungs.

## Direct execution and resources

Exactly22 debug jobs: one shared frozen-original TRAIN-target export,20 fits
(five per arm), one dependent report/owned-target-retirement job. The shared
export performs NO optimization, stress updates, benchmark or technical epoch.
There is no preflight and no login-node follow-up controller. Every job is
submitted in one hash-reviewed afterok DAG. Branch dependencies never use a
different arm's learned teacher. Completion receipts publish only after selected
state readback and required outgoing exports finish.

Reuse original-source acceptance from the completed d8dba242 token study,
including original fixed30 native/KD acceptance, exact model/data provenance,
installed environment, A100 and successful token jobs. New combination is NOT
remotely measured. This user-approved policy supersedes a new preflight/miniature
requirement for this campaign only; all integrity checks remain mandatory.
6CPU/90000MiB/A100, debug. Fit request max480min, ceil1.5*prior aligned projection
+120min; reject over24h. Export480min/report120min. These conservative requests
are allowances, not measurements or guarantees. No runtime batch/support fallback.

OFFLINE TRAIN and wholly deleted U000 TRAIN/validation caches may be reconstructed
in RAM by unchanged authenticated donor builders and must match original view
fingerprints. A partially missing or corrupt cache fails. Other caches must exist.
Only one input view per role is resident during each fit/export.

Reserve8GiB personally available durable space. Store only compact jet targets,
not particle/attention banks. Shared OFFLINE/M0HLT targets export once. Every
nonfinal selected model exports its T2 TRAIN probabilities; UE100/DE066/DE033
also export embeddings. Final report authenticates all tasks, persists results
and retirement intent, deletes ONLY exact owned target payloads, then publishes
completion. Retain native models, validation predictions, histories, moments,
manifest hashes, source/evidence snapshots and logs. Partial retirement fails
closed and requires explicit recovery, never blind retry. Never delete old data.

## Validation and handoff

New `luka_trial_hybrid` namespace and v1 contracts. Donor baseline/integration
2376559bc9b1d0ff455bbdb242ae5769a54d6168. Adapt fullstat hybrid science and trial
joint/source/submission machinery; no existing runtime changes. Test exact losses,
schedule endpoints, all20 branches, cross-arm/wrong-role/hash failures, native
parity, TRAIN joins, dry/live submission and duplicate claims, all production
workers and retirement/reporting on synthetic installed-Weaver data. Cluster
boundaries must be identified as mocked. No commit/push/submission by implementation.

See [contract](../contracts/LUKA_TRIAL_HYBRID.md) and
[runbook](../LUKA_TRIAL_HYBRID_RUNBOOK.md).
