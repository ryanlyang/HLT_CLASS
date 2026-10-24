# FullStat split-head continuation: DE066 -> DE033 -> DE000

Status: implementation authority for the user-confirmed Oscar experiment,
2026-10-24. This is NOT the smaller-trial c50 auxiliary-logit sweep and does
not inherit its 30 epochs, four alpha arms, population, or selected teachers.

## Scientific question

Can teacher imitation through separate auxiliary classification heads, together
with native embedding prediction, improve the final HLT classifier? The main
classifier receives only label CE. All auxiliary objectives still update the
shared encoder: separate heads do not eliminate gradient conflict or guarantee
improvement. This changes head placement AND the set of logit teachers relative
to the original hybrid ladder; it is not an isolated head-separation ablation.

## Exact continuation

Reuse the completed FullStat hybrid UE100 (and authenticate its completed UE000
ancestry), execution `e129075e9c3eaa9fd2e49460b79ecf5f56ba4be7`.
Do not require the interrupted old DE066 or cancelled old DE033/DE000.
Do not reuse teachers from the unrelated trial's completed 50/50 ladder.

| New stage | Input coordinate | Teachers for BOTH auxiliary objectives |
| --- | --- | --- |
| DE066 | D066 | OFFLINE, existing hybrid UE100, M0HLT |
| DE033 | D033 | OFFLINE, existing hybrid UE100, NEW DE066, M0HLT |
| DE000 | D000 | OFFLINE, existing hybrid UE100, NEW DE066, NEW DE033 |

All three models train from scratch with coordinate-paired initialization and
sampling seeds. There is no checkpoint warm start. UE100 is not retrained.
Only DE000 is HLT-deployable; intermediate rungs are richer-input oracles.

For N registered teachers, T=2:

\[
 L=0.5\,CE(y,p_{main})+
 \frac{0.5}{N}\sum_{t=1}^{N}T^2 KL(q_t^T\Vert p_{aux,t}^T)
 +\frac{5}{N}\sum_{t=1}^{N}SmoothL1(g_t(h),z_t).
\]

- Main head: native classifier; CE only, coefficient 0.5.
- Separate auxiliary logit head per teacher: Linear(128,11), FP32 head/loss.
- Separate embedding head per teacher: Linear(128,128), GELU, Linear(128,128).
- Forward teacher-to-student KL, batchmean, T-squared exactly once.
- Embedding loss: mean over samples/dimensions, SmoothL1 beta=1.
- Total alpha=0.5, total embedding weight=5; divide both independently by N.
- No attention loss, relational loss, target averaging, fusion, or alpha sweep.
- Head RNG is isolated from backbone/dropout RNG; per-teacher seeds are bound.
- Neither actual nor predicted teacher outputs enter the main head.
- All auxiliary heads are removed from selected/inference checkpoints.

## Data, targets and selection

Same authenticated FullStat views: 3,000,000 TRAIN; 500,000 development
validation. Existing class/QCD-bin-stratified row split keeps each HLT/offline
pair together; event-level independence remains unverified. The 1,500,000
final-test rows remain sealed; this campaign has no test operation. The existing
105,325 unused rows remain unused.

Targets come only from selected, frozen native teachers on TRAIN rows. Native
128-dimensional fc-input embeddings are standardized independently per teacher
using TRAIN population FP64 moments, scale floor 1e-3, FP32 stored targets.
Never standardize from validation/test, average latent axes, or use predicted
embeddings/auxiliary logits as targets. Native T=2 probability exports follow
the accepted FullStat BF16 CUDA forward / FP32 softmax path.

Reuse byte-identical OFFLINE/M0HLT/UE100 embedding banks and OFFLINE/UE100
probability banks. First DE066 job exports the missing M0HLT native probability
bank. After fitting, new DE066 and DE033 export their selected native embeddings
and probabilities for descendants. Final DE000 exports neither outgoing bank.
Exact TRAIN ordered identities, labels, roles, coordinates, preparation identity,
teacher checkpoint/report hashes and manifests must match before consumption.

Training stays fixed at 20 full epochs, batch 128, no accumulation, unchanged
FullStat AdamW parameter grouping/settings and warmup2/cosine-to20 schedule.
Choose each model by full-validation main-head macro OVR AUC, with the existing
deterministic selection tie breaks. R30/R50 use those same checkpoints. Finite
poor performance is a scientific result and never blocks a later registered rung.

## Execution and scope

Exactly three Oscar L40S scientific jobs with afterok dependencies; integrated
target export and publication, no standalone preflight/setup/reducer/aggregator,
no login controller. Explicit dry review and exact pushed source are mandatory.
Reuse authenticated original Oscar acceptance, environment, native support and
resource envelope. Each job runs only three disposable maximum-TRAIN-support
updates, a maximum-validation-support forward and FP32/BF16 native parity/CE-only
main-head gradient checks. Discard that model and reset scientific initialization.
These checks are technical, not a metric threshold or scientific epoch.

Request 6 CPUs, 180000 MiB, exact accepted L40S. Allow an extra 60 minutes above
the inherited hybrid request per rung, 120 for first DE066's M0HLT export;
reject >48h, never silently clamp. This combined three/four-head runtime has
not been measured on Oscar; inherited evidence is not a new performance promise.
No fallback to fewer epochs, smaller batches or truncated support.

Existing campaigns/artifacts stay immutable. Cancellation is a separate explicit
scheduler action, never hidden in the queue helper. Preserve partial failures;
no auto retry or duplicate submission. Technical failure blocks descendants.

## Interpretation

Compare DE000 against historical M0HLT, original direct/logit D000, U100-w5 and
four-teacher embedding references. Report the new intermediate teachers separately.
These are adaptive, single-seed development-validation comparisons, not independent
test evidence. Do not infer significance from small differences or claim that
better intermediate AUC guarantees a better final student.
