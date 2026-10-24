# LUKA_AUXILIARY_LOGITS v1

Authority: [active plan](../plans/LUKA_AUXILIARY_LOGITS_PLAN.md).
Canonical hashes, schema 1, explicit parents and exploratory trial flags.
Existing hybrid, data, native model and target-bank contracts are unchanged.

- RECIPE binds CE weight 0.5, total embedding weight 5, and total
  auxiliary-logit weight 0 / 0.25 / 0.5 / 1; T=2, four separate linear heads,
  detached targets, zero-control no head decay, exact 30-epoch / batch-128
  schedule and isolated paired initialization streams.
- REUSE binds the completed original 72490a9d c50 branch, original-source replay,
  exact successful jobs, teacher checkpoint/report pairs, same validation
  comparisons and inherited resource acceptance. Other arms may be pending.
  Original target retirement is validated by the parent implementation; no
  obsolete bank is required or silently reconstructed in an old directory.
- SPEC binds new clean source, immutable REUSE, unchanged population and
  teacher records, six-task debug graph, 3 GiB quota and conservative requests.
  Fit submission order is k1, k05, k025, k0 (weights 1, 0.5, 0.25, 0);
  fits retain independent afterok:targets dependencies. Order changes the
  spec/plan content hashes, not scientific recipes or artifact schemas.
- Shared target payloads reuse unchanged LUKA_TRIAL_HYBRID_BANK/v1 semantics
  and exact selected TEACHER descriptors. New owner/task references identify
  new export locations. Ordered uint8 Nx32 TRAIN IDs/int64 labels must match
  every consumer exactly; finite FP32 Nx128 standardized embeddings and Nx11
  T=2 probabilities, FP64 moments / floor 1e-3. No validation target fitting.
- BUNDLE binds all four banks in fixed order and new spec; both target types
  are required even for alpha 0 diagnostics. No privileged inference input.
- STARTUP binds three disposable worst-TRAIN-support updates per arm, 128
  physical batch, maximum-validation forward, installed-native FP32/BF16
  parity, CE-only main-head gradient and zero-control disconnection. Native
  scientific initialization is recreated afterward. No technical epoch.
- TRAINING_REPORT/FIT binds exactly 30 main-head validation passes and selected
  native state, all raw/weighted per-head losses, bundle, node, recipe, exact
  schedule and native probability/checkpoint roundtrip. Auxiliary head state
  is never in the deployable checkpoint. Weak finite science is valid.
- TASK completion publishes last, binding exact accepted job/spec, dependency
  hashes, runtime and exhaustive outputs. A failed export/check cannot release
  scientific fits. Submission PLAN/CLAIM and accepted-job journal prevent
  duplicate or ambiguous retry and require exact dry-run hash authorization.
- RESULTS uses retained predictions and receipt-authenticated metadata, not
  live models/ROOT data. All partial arms explicit. Full-validation AUC,
  recovery and delta against original c50; original50k report-only. No test.
- RETIREMENT/COMPLETE binds four exact newly owned canonical target leaves,
  all consumers and saved report before deletion. Interrupted cleanup fails
  closed; nothing is auto-retried. Models, reports, moments, predictions and
  old campaigns are retained. Retained results remain verifiable afterward.
