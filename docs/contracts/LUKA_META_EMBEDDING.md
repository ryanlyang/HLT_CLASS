# LUKA_META_EMBEDDING contracts v1

Authority: [active plan](../plans/LUKA_META_EMBEDDING_PLAN.md).

All new JSON records use `LUKA_META_EMBEDDING_<KIND>/v1`, schema_version1,
canonical content hash, explicit parent hashes and inherited ALLJETS flags.
Kinds: RECIPE, SPEC, INITIALIZER, MEASUREMENT, CHECKS, REPORT, FIT, TASK, PLAN,
CLAIM, REFERENCE, RESULTS. Source is a clean pinned additive Git snapshot;
existing scientific runtime bytes remain those of830a2afb2b3350c14ae147d915b8d3072701269c.

Original teacher/CE imports use original fixed30 replay. Original w5 import
replays only w5's fit/receipt in855c4f8d, authenticating output files and node,
without requiring retired unused target arrays or other sweep arms. It never
edits old records or treats mere checkpoint existence as evidence.

The compact OFFLINE bank retains `LUKA_EMBEDDING_AUX_BANK/v1`; exact original
teacher SHA, prepared hash, ordered TRAIN identity/label join, shape/dtype,
statistics and finite-value validation are required. New split membership is
deterministic SHA256(identity + registered salt), not physics labels/features.
No source indices or targets become deployable inputs.

TASK receipts bind spec, dependency receipt hashes, accepted exact job ID and
complete relative-path output inventory. Setup measurements bind installed
environment, GPU, real/virtual maximum-support witnesses, whole-population
timings, calibrated initializer and target bank. Fits bind those artifacts and
save exact selection history with loss/schedule/meta-update evidence.

Native `selected.pt` contains ONLY original model keys. `training_state.pt`
contains the selected full student, adapter, student/adapter optimizer states,
epoch and RNG state for audit/future explicit recovery (not automatic resume).
No target/prediction head is needed for inference. Results authenticate saved
predictions, ordered populations and selected metrics without opening models.

No performance gates. No final-test use. No automatic destructive cleanup or
ambiguous retry. Remote inline A100 acceptance is required even if local tests
pass; queue-ready is not a claim that remote measurements already succeeded.
