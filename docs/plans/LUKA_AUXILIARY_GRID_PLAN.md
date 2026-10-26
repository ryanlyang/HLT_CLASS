# Trial DE000 auxiliary-KD x embedding x relation sweep, v1

## 2026-10-26 operational recovery amendment

User authorized repair after original K5 job21962581 failed post-publication.
The saved30-epoch report passes the original validator when its per-teacher
dictionary order is restored; canonical bytes, numerical values and hash are
unchanged. K10 also failed, but is not an imported reference and is not recovered
by this amendment. The cancelled old report is not restarted.

This amendment supersedes the completed-K5-only prerequisite below, not any
scientific rule. New SPEC/REFERENCE/TASK/PLAN/ACQUISITION artifacts are v2;
scientific recipes remain byte-identical v1. New REFERENCE_REPLAY and
REFERENCE_READBACK v1 evidence explicitly preserve original accounting.
Authorization is AUTHORIZE_LUKA_AUXILIARY_GRID_TWENTY_DEBUG_V2.

The original pinned strong source remains untouched. An isolated read-only
adapter can accept only commit234b029ec14a8d4861134e098cafbf939d325cd8, accepted
K5 job21962581 FAILED/1:0, with the precise post-publication traceback, confirmed
JSON-order discrepancy, complete FIT/TASK receipts, payload hashes, costs,
startup/bundle, exact predictions/population/metrics, and original successful
targets ancestor. Restore dictionary order only in memory for the original
report validator; no tolerance, loss value, selected epoch or hash changes.
Unexplained failures, missing/corrupt files and wrong jobs fail closed.

The shared targets job replays this evidence, privately copies intact banks,
and restores the saved native K5 checkpoint on the identical ordinary validation
population. Require byte-identical predictions and identical saved metrics.
Publish REFERENCE_READBACK and the new successful targets receipt before any
new fit is eligible. This is inference-only verification: no optimizer,
retraining, checkpoint reselection, separate preflight or extra job. Retain the
FAILED old status and bind it to the new verification job, never report it as
COMPLETED. The original completed-reference/registered-retirement route remains
supported, with the same checkpoint readback requirement. After new targets
publication, retained new receipts suffice; old target banks need not survive.

No old files, claims, source, jobs or accounting are edited. Queue in a new
commit-named root after exact pushed-source/dry-plan authorization. The failed
be6a6248 create stopped before root creation/submission; preserve any unexpected
existing root rather than overwriting it. All22 jobs,20 fits, budgets, ranked
waves and native-only inference remain unchanged.

## Original scientific registration

User approved the revised twenty-configuration grid on 2026-10-26. This replaces
the earlier proposed fixed-KD/extreme-embedding grid; it does not extend it.
Implement only; no commit, push, cancellation or submission is authorized here.
Debug direct follow-up uses inherited genuine SPORC acceptance and bounded
checks in the shared targets job, not a standalone preflight. This specific
route supersedes the generic new-preflight requirement, not integrity checks.

## Science

Twenty fresh HLT-only DE000 fits. Frozen teachers are OFFLINE and c50 UE100,
DE066, DE033 from original trial hybrid72490a9d. No upstream retraining,
warm start, weight fusion, alternative teacher, or frozen-classifier readout.
Native classifier CE coefficient0.5; four separate trainable128->11 KD heads
and four128-GELU-128 embedding predictors read native pre-classifier features.
No actual or predicted target enters the main classifier. Auxiliary heads are
removed from native selected checkpoints and inference.

Loss: 0.5 CE + K/4 sum_t[T^2 KL(q_t(T)||p_t(T))]
      + E/4 sum_t SmoothL1(pred_t,target_t) + beta_u/4 sum_t relation_KL_t.
T=2 with T-squared exactly once, KD batchmean, SmoothL1 beta1 mean over
queries/dimensions, independent TRAIN FP64 standardization/floor1e-3, FP32
stored targets. K/E are coefficients, not gradient percentages. No rescaling
of the whole loss, adaptive weighting, gradient instrumentation or LR change.

Every wave has two fits: relational then plain (ordering within a wave is not
guaranteed). Ordered (K,E) waves:
1 (5,10), 2 (5,20), 3 (2.5,10), 4 (2.5,20), 5 (1,10), 6 (1,20),
7 (5,40), 8 (2.5,40), 9 (1,40), 10 (10,20).
Thus3x3x2 on K={1,2.5,5}, E={10,20,40}, plus two K10/E20 checks.
No existing K5/E5 fit is repeated; import its retained result as reference.

Relation semantics are unchanged from luka_multiteacher_trial: for each
teacher, normalized predicted/true queries versus that teacher's normalized
TRAIN anchors; forward KL, tau0.2, sum references, mean queries, mean teachers,
no tau-squared. Independent PCG64 SeedSequence([314159,epoch,batch_index]);
2048 shared row references/update,186/class plus rotated two extra classes.
Same-jet rows masked on both sides; >=187 eligible TRAIN rows/class required.
beta=min(1,update/(2*ceil(N/128))) for rel, identically0 for plain. The sampler
does not advance scientific model/dropout/data RNG. No validation references.
Relation adds supervision emphasis, not independent teacher information.

Original c50 DE000 model/head/sampler seeds, physical batch128, AdamW
warmup3/cosine30 and all30 epochs remain fixed. Identical299234 TRAIN/74895
validation; original50k report-only, historical test retired, NO independent
test. Select by full-validation main-head AUC and existing tie rules. Finite
poor performance never cancels/skips a registered fit. This is a single-seed
exploratory twenty-way screen; repeat finalists before generalization claims.

## Execution, reuse and storage

22-job exact debug DAG: targets,20 fits in ten two-job waves, report. Both
jobs in each wave require targets and both previous-wave jobs to succeed.
At most two fits eligible concurrently; no priority changes to existing jobs.
Technical failures block later waves; poor finite metrics do not. Report waits
for all20. No standalone preflight, login controller, retry or automatic requeue.

Authenticate c50 lineage in original pinned source. Import completed K5/E5
from the supplied original strong campaign, with native fit, predictions,
report/receipts and successful accounting. Do not infer completion from logs.
Record an optional bank-reuse source from that same authenticated campaign.
At setup, re-authenticate original targets and retirement state. If intact,
copy four verified manifests/payloads into new owned directories and validate
the complete TRAIN identity/label join. No symlinks/hardlinks or writes to the
old root. If normally retired, export all four once from retained teachers.
Corruption or unregistered/partial loss fails, rather than silently regenerating.
An old cleanup racing a copy may fail setup; preserve evidence, no blind retry.
Never require old bank survival after the new targets receipt is published.

Shared setup includes native FP32/BF16 parity, CE-only main-head gradient,
three disposable physical-batch128 maximum-support updates for plain K5/E10,
relational K5/E40 and relational K10/E20, and maximum-validation forward and
memory checks. All20 exact objectives additionally receive local gradient tests.
Discard probes and recreate science models/RNG. No technical training epoch.

GPU debug/A100/6CPU/90000MiB, at least8h or larger inherited c50 allowance,
reject>24h; CPU report1CPU/16GiB/2h. Combined-objective full throughput is NOT
newly measured remotely. Reserve4GiB durable headroom for shared banks and20
selected models/predictions/histories. After all fits and report validate,
retire only four new owned target payloads. Keep selected models, predictions,
results, manifests/moments, logs and lineage. No unrelated deletion/Fresh_check.

## Implementation and acceptance

Additive luka_auxiliary_grid namespace and v1 artifacts. Donors at
0c4a71a55e8bc163ea3c4805b72f06d18e640cb5: auxiliary_logits_strong package,
wrappers/tests, and multiteacher_trial AnchorSampler/reference_kl. Exact paths
in LEGACY_SOURCE_MAP. Existing contracts/runtime/immutable campaigns unchanged.
Focused contracts,20-arm losses, native parity, RNG, bank joins/corruption,
reuse/retirement, ordered DAG, duplicate rejection, poor science and real local
installed-Weaver worker miniatures required. Local synthetic/mocked scheduler
tests are NOT new SPORC/Tigris certification. Queue requires exact pushed clean
source, quota, full dry plan hash and explicit22-job authorization.
