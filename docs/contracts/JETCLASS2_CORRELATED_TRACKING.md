# Correlated tracking pilot contract

Authority: [implementation plan](../plans/JETCLASS2_CORRELATED_TRACKING_PILOT_PLAN.md).
New `JC2_CORRELATED_TRACKING_{KIND}/v1`, schema 1, kinds RECIPE, SPEC, PLAN,
FILE_REPORT, REPORT, RECEIPT, LEDGER. No old artifact is relabelled.

SPEC binds clean exact pushed source, completed original literature pilot
spec/receipt, authenticated original population/physical records, frozen recipe
and resource envelope. Only training OFFLINE is transformed. All access flags
for native HLT, validation, final test and production qualification are false.

Blocks contain `identity`, shared `offsets`, and `SIDE_FIELD` for OFFLINE and
six frozen sides, FIELD in p4/charge/category/tracking/valid. Particle rows keep
the original order and one-to-one correspondence; identity/offsets are metadata,
never features. No latent vertex shift, RNG state, labels or ancestry features.
Masks and fields not modified by the recipe must remain exactly equal.

RECEIPT seals the exact report/CSV/two-PDF set, per-file reports and banks, with
content and byte hashes and explicit parents. Existing results authenticate
before reuse. Nonfinite inputs, changed source/lineage, wrong coverage or corrupt
bytes fail closed. Disappointing scientific metrics never fail a job.

PLAN authorizes one CPU-only Tigris job, dry by default. Live submission needs
the exact reviewed plan hash and
`AUTHORIZE JC2 CORRELATED TRACKING PILOT EXACT PLAN`. Failed/ambiguous submission
retains its lock; do not retry blindly. Completed ledgers are idempotent. No
other jobs are cancelled or changed, no training launches automatically.
