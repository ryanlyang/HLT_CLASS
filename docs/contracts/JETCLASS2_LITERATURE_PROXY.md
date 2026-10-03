# JC2 literature proxy contracts v1

Authority: [pilot plan](../plans/JETCLASS2_LITERATURE_PROXY_PILOT_PLAN.md).
Artifact prefix `JC2_LITERATURE_PROXY_`, schema version 1. Kinds: RECIPE,
POPULATION, SPEC, PLAN, LEDGER, FILE_REPORT, REPORT, RECEIPT. Each contains a
canonical SHA-256 content hash and explicit parent identities. Files referenced
outside an artifact carry byte hashes. Hashes authenticate identity/integrity,
not scientific quality. Old CMS proxy artifacts are never interchangeable.

Only registered training membership is admitted. Code authenticates the source
inventory/profile, recomputes the selected population, reads only the declared
offline branch allowlist, and authenticates each used ROOT file before/after.
The recipe is reconstructed and compared exactly, not accepted merely because
a modified recipe has a valid self-hash. SPEC fixes no test/validation/native
HLT particle access, all three strengths, resource shape, and source commit.

Each NPZ has `identity` (one SHA-256 string per jet); for offline and each variant,
`<side>_offsets` (N+1), `<side>_p4` (P,4), `<side>_charge` (P),
`<side>_category` (P), `<side>_tracking` (P,4), `<side>_valid` (P,4).
Tracking order d0,dz,d0err,dzerr, in mm; p4 px,py,pz,E in GeV. PID order charged
hadron, neutral hadron, photon, electron, muon, unknown (0..5). Missing tracking
is zero with false mask; zero displacement may be valid. Derived significance
requires both valid value and positive valid uncertainty. No normalized network
features or labels are written. Separate lineage maps output particles to native
offline indices for diagnostics only. Never expose it to a deployable model.
It is also not an implicit particle-matching or KD target for a future ladder:
such use would require a separately declared oracle/supervision experiment.

Output files are atomically published without replacing different bytes. A
completion receipt is published last and authenticates the complete manifest.
Reports explicitly say synthetic benchmark, training-only, no test access and
no production qualification. Results inspection verifies the receipt and every
referenced file. Statistical disagreement cannot make a successful run fail;
invalid inputs, source drift, replay failure or corrupted artifacts must fail.

Submission requires exact PLAN content hash and authorization phrase
`AUTHORIZE JC2 LITERATURE PROXY PILOT EXACT PLAN`. Dry run creates no Slurm job.
Submission is serialized and records the single returned job ID; existing ledger
is checked, not blindly overwritten or submitted twice. An ambiguous interrupted
submission lock requires read-only scheduler inspection, never automatic retry.
