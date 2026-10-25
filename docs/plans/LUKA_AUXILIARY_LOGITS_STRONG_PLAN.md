# Higher auxiliary-logit weights: k5 and k10, direct debug

User authorization 2026-10-24: implement exactly two additional fresh DE000
fits in debug, with no standalone preflight. This additive plan overrides the
generic new-preflight requirement only for these two follow-ups. No submission,
commit/push, old-job cancellation or alteration is authorized by implementation.

## Scientific registration

Extend the [original split-head experiment](LUKA_AUXILIARY_LOGITS_PLAN.md) with
total auxiliary-logit weights `k5=5` and `k10=10`. Original k0/k025/k05/k1
contracts and runtime bytes remain unchanged. This is NOT the proposed
frozen-teacher-classifier-on-predicted-embedding experiment.

For each arm, keep exactly:

`0.5 CE(main,y) + alpha/4 sum_t [4 KL(q_t(T2)||p_t(T2))]
 + 5/4 sum_t SmoothL1(g_t(h),z_t)`.

Four frozen teachers: OFFLINE, **c50** UE100, DE066, DE033 from the completed
trial hybrid execution at `72490a9d6de4acabe9cbea518c5b100d893e6866`.
No alternate branch, model substitution, preceding-student initialization or
upstream rung retraining. Current auxiliary-logit jobs need not finish and
their disposable banks are never dependencies of this campaign.

Native main head receives CE only, coefficient 0.5. Four independent trainable
128->11 logit heads receive 1.25 each at k5 or 2.5 each at k10. Four separate
128->GELU->128 embedding heads retain coefficient 1.25 each. T=2; apply T-squared
exactly once. No overall loss renormalization, gradient surgery, adaptive loss
weights, LR changes or new gradient instrumentation. Keep original raw/weighted
per-teacher loss logging. These coefficients are not gradient percentages.

Same detached T2 teacher distributions and native pre-classifier 128-D teacher
embeddings, independently TRAIN standardized with FP64 moments / 1e-3 scale
floor and FP32 targets. No predicted embeddings feed either the main head or
auxiliary-logit heads. Deployment removes both auxiliary head families.

Fresh D000 HLT-only students; exact original c50 DE000 encoder/sampler seeds,
isolated original k1 logit/embedding-head RNG streams, physical batch128,
AdamW warmup3/cosine30 and exactly30 epochs. Same 299,234 TRAIN / 74,895
validation; original50k report-only; NO independent test. Main-head full-
validation AUC selects checkpoints using existing CE/logR50/update tie rules.
Poor finite performance completes every registered epoch and arm.

## Execution, ownership and reporting

Four exact hash-reviewed afterok jobs: shared targets, k5, k10, CPU report.
Both fits depend only on targets; report depends on targets and both fits.
Submit k5 before k10, without forcing serialization or promising Slurm start
order. No standalone preflight, login controller, automatic requeue or retry.

New root `luka_auxlogit_strong_<commit8>_debug_r1`, job prefix `lukaas_`.
Re-authenticate completed c50 ancestry and actual accepted-job success in its
original pinned worktree. Reuse its genuine resource/environment acceptance;
new high-weight throughput is NOT remotely measured. GPU requests: debug A100,
6CPU,90000MiB,8h minimum or larger inherited allowance, reject above24h. Report:
1CPU,16GiB,2h. Require exact owner/account/source/env/GPU/allocation and journals.

Export four independently owned compact TRAIN banks once from retained c50
teachers, even if old campaigns have retired their banks. Export contains
bounded disposable checks: installed-native FP32/BF16 output parity, CE-only
main-head gradients, three batch128 maximum-TRAIN-support updates at BOTH high
weights, and maximum-validation forward/memory headroom. No technical epoch or
warm start. Checks must pass before the export receipt releases either fit.

Reserve3GiB personally available durable headroom. After both fits AND the
saved report validate, retire only the four exact newly owned TRAIN target
payloads with path/hash-bound intent and completion. Keep native checkpoints,
predictions, histories, moments, manifests, logs and source. Partial retirement
fails closed. Never delete old campaigns, shared data, models or Fresh_check.

Retained results compare M0HLT, OFFLINE, OFFLINE-w5, U100-w5, original c50 DE000,
k5 and k10 with full-validation/original50k metrics and recovery. Original k1
is a separate existing campaign, not a required completed job or newly fitted
control. This implementation must not infer its completion from queue absence.

## Implementation and verification

Additive `luka_auxiliary_logits_strong/`; new
`LUKA_AUXILIARY_LOGITS_STRONG_<kind>/v1`. Copied/adapted donor runtime and tests
from `a7efcef6a8399cdcde80b2941e1a21910ba1bf67`; original native target BANK/v1
semantics are unchanged. Exact donor paths in LEGACY_SOURCE_MAP.

Tests cover old contract non-reinterpretation, k1-identical initialization,
KD gradient scaling5x/10x with unchanged CE/embedding gradients, detached targets,
native parity, all30 epochs for both full-architecture workers on miniature
populations, exact four-job graph, invalid data/source/lineage, old-parent target
retirement and new-owned cleanup. Local mocked scheduler tests do not certify a
real SPORC/Tigris run. Queue readiness requires the exact pushed source and a
reviewed plan; no new real-cluster validation is claimed locally.
