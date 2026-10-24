# FullStat split-head contracts v1

Authority: [scientific plan](../plans/LUKA_FULLSTAT_SPLITHEAD_PLAN.md).
Namespace: `LUKA_FULLSTAT_SPLITHEAD_<kind>/v1`, schema_version=1.
Old hybrid and trial contracts are not relabeled or modified.

Kinds: TARGET_RECIPE, KD_RECIPE, RECIPE, REUSE, SPEC, PLAN, SUBMISSION_CLAIM,
BANK, KD_BANK, BUNDLE, STARTUP, TRAINING_REPORT, FIT, TASK, RESULTS.
All carry canonical content hashes, explicit parent hashes and unchanged FullStat
sealed-role flags. Artifacts publish immutably/atomically; path existence is never
reuse evidence.

- REUSE: exact parent spec and original commit; original-source replay verifies
  completed UE000/UE100 task receipts, accepted job identity/account/owner and
  COMPLETED/0:0 accounting, selected models, saved validation, outgoing native TRAIN
  banks and transitive original Oscar acceptance. Old D rungs are not consulted.
  Runtime donor bytes must match current execution before replay.
- SPEC: source snapshot, immutable reuse, exact recipes/three tasks, resource
  policy, external reference and bank manifests. New output cannot overlap old
  campaigns, data or pinned worktrees. At least 16 GiB personal durable headroom
  must be declared; filesystem-wide free space is an additional, not quota, check.
- PLAN: three jobs, no preflight/setup; exact afterok ancestors, no-requeue,
  kill-on-invalid-dep, named spec hash/task, pinned source, Oscar account/QOS/GRES.
  Live submission requires saved dry plan hash and
  `AUTHORIZE_FULLSTAT_SPLITHEAD_V1`. Durable live claim and exact-DAG journal reject
  duplicate/ambiguous replay; do not blindly rerun after connection failure.
- Imported banks retain their ORIGINAL contracts. OFFLINE, M0HLT and UE100
  embeddings and OFFLINE/UE100 probabilities use original bank validators; exact
  refs are authenticated in the completed parent. They are never copied into a
  misleading new semantic identity.
- New BANK/KD_BANK: selected native checkpoint/report refs, TRAIN prepared hash,
  teacher identity/coordinate, width128 or 11 classes/T=2, ordered uint8[N,32]
  identities and int64 labels. FP32 finite targets; FP64 per-dimension moments;
  scales >=1e-3. Probabilities nonnegative, normalized, FP32[N,11]. Payload refs
  include size/hash and exact filename. Validation/test targets forbidden.
- BUNDLE: exact per-stage teacher order; separate embedding and logit records and
  content hashes; spec and ALL new ancestor TASK hashes. Exact cross-view ordered
  TRAIN join. Every teacher has both target types. No old DE066/DE033 substitution.
- STARTUP: exact environment/GPU/source/support maxima; batch128, three disposable
  updates, no benchmark or full validation; native FP32/BF16 parity and main-head
  gradient equal to 0.5 CE gradient. Finite connected auxiliary/backbone gradients,
  measured memory headroom, fresh scientific initialization afterward.
- TRAINING_REPORT: exactly20 full passes, coordinate seed, correct per-update LR
  schedule, 500k validation population, raw and weighted per-teacher losses and
  coefficients, timings, deterministic main-head AUC checkpoint selection. A
  finite poor AUC never causes technical rejection.
- FIT: native-only selected.pt, report, validation probabilities; exact saved
  population and recomputed metrics, restored native checkpoint readback parity.
  No auxiliary keys allowed in selected state.
- TASK: published only after fit and required outgoing targets; exact accepted
  job/source, all ancestor receipts, complete output inventory. Dependents require
  this receipt AND successful exact scheduler accounting, not an epoch20 log.
- RESULTS: authenticated saved predictions only, selected AUC/accuracy, historical
  M0HLT-to-OFFLINE recovery and per-class R30/R50 with the same selected checkpoint.
  Rank/ties follow existing reporting; zero-passing QCD is censored, not infinity.
  No inference/test access; historical rather than fresh matched CE control.

Source compatibility baseline: `a4596cb915252672ab42c7a8724d6a105a0bc56e`.
Only this new namespace and its three runtime wrappers may be added beyond that
baseline; required plan/contract/runbook/tests must be tracked. Execution must be
clean, exact-commit pinned, and reachable from fetched origin/main. A different
runtime, parent, resource envelope or scientific recipe requires explicit review
and a new semantic registration, not weakening checks on the old run.
