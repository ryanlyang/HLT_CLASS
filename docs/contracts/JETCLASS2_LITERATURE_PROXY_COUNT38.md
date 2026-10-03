# Count-targeted literature proxy contract v2

Authority: [v2 plan](../plans/JETCLASS2_LITERATURE_PROXY_COUNT38_PILOT_PLAN.md).
Prefix `JC2_LITERATURE_PROXY_COUNT38_`, schema 2: RECIPE, SPEC, PLAN,
CALIBRATION, FILE_REPORT, REPORT, RECEIPT, LEDGER. Never interchangeable with v1.

SPEC pins exact clean pushed source, unchanged completed v1 parent SPEC/RECEIPT,
train population and all frozen strengths/eligibility/target rules. Input reuse
authenticates parent metadata, manifest byte hashes, identities, integer offsets,
physical arrays and diagnostic ancestry. It never evaluates raw ROOT HLT/labels.
Calibration parents bind the spec; rates are recomputed from saved integer
counts and the registered target formulas. They must be finite and in [0,1].

V2 blocks use identity plus COUNT38_V2_offsets/p4/charge/category/tracking/valid,
with the same shapes and units as v1 (GeV, mm). OFFLINE/NOMINAL live in the pinned
parent blocks, paired by identical row identity and order. Do not concatenate
offsets across files without rebasing. The new root is a diagnostic pilot, not
a production split export. Separate ancestry JSON is diagnostic-only, forbidden
for deployable inputs or implicit oracle matching supervision.

All v2 reports bind both spec and calibration. The receipt hashes all outputs,
including calibration, per-file reports, physical blocks, lineage, CSV, PDF and
examples. Inspection validates the complete manifest. Every reused file is
authenticated; existence is insufficient. Parent artifacts remain immutable.

Submission requires `AUTHORIZE JC2 LITERATURE PROXY COUNT38 EXACT PLAN` plus
the exact saved PLAN content hash. Dry-only default; clean pushed source;
resource test-only before sbatch; exclusive locks; validated idempotent ledger.
Ambiguous failed submissions retain their lock for scheduler inspection. No
native HLT, validation or final-test access; no production qualification. An
unmet mean-count target is a result, not an execution failure.
