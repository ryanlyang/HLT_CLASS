# JC2 literature noise-only v3 contract

Scientific authority: `../plans/JETCLASS2_LITERATURE_PROXY_NOISE_PILOT_PLAN.md`.

Artifacts use `JC2_LITERATURE_PROXY_NOISE_{KIND}/v3`, schema_version 3,
canonical content hashes and byte-hashed references. They are NOT interchangeable
with v1 or COUNT38 v2. The copied calibration retains its original v2 schema,
content hash and v2 spec parent; it is never falsely reparented to v3.

SPEC pins current source, completed v2 spec and receipt, original calibration,
population, per-file original and saved-control inputs, recipe and resources.
The v1/v2 ancestors and every sealed output must authenticate before reuse.
REPORT links v3 SPEC and original calibration; preserves all v2 mechanism
counts, reports exact topology equality and serial/process replay, and declares
no native HLT/validation/final-test access or production qualification.

Physical blocks contain identity plus NOISE_V3 offsets, p4 (px,py,pz,E in GeV),
charge, category, tracking (d0,dz,d0err,dzerr in mm), and validity masks.
PID order is charged hadron, neutral hadron, photon, electron, muon, unknown.
Offsets and identities must cover exactly the authenticated parent rows.
Invalid tracking entries are zero; applicable errors must be positive.
Ancestry/keys live in separately sealed diagnostic files, never model inputs.

RECEIPT seals all per-file reports, physical/lineage blocks, calibration,
aggregate report, CSV, inclusive/paired PDFs and examples. Missing, extra,
corrupt, nonfinite or mis-parented artifacts fail closed. Scientific performance
or broad distributions do not. Fresh output roots cannot overlap parent data
or execution worktrees. Submission is one CPU-only job, with exact reviewed
PLAN authorization and an immutable ledger; no automatic production followup.
