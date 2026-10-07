# CORR_HIGH_TOPO contracts

Authority: [plan](../plans/JETCLASS2_CORR_HIGH_TOPO_PLAN.md).

Producer namespace JC2_CORR_HIGH_TOPO_RECIPE/v1. Shared ladder tooling versions:
RELEASE_REQUEST/RELEASE/FOUNDATION/VIEWS v7, GATE_SPEC/RUNTIME_PROFILE v12,
SCIENTIFIC_PLAN/CAMPAIGN_SPEC v9. Existing versions retain their meaning.
INPUTS v3 remains the unclipped correlated asinh/log1p 17-field interface.

The request binds original v4 release and exact NOISE_V3 pilot specification,
its sealed calibration reference, physical source and frozen recipe. The new
release retains source_file/entry/role/identity arrays exactly and changes only
proxy_block/proxy_row references to new lossless physical banks. No labels are
persisted in these banks. Label access during paired reading is declared;
selection and degradation are label-independent. Ordinary roles only.

Physical banks contain offsets, jet_identity, p4, charge, category, tracking,
valid, using the existing lossless physical codec. Mapping is proxy-slot to
original representative index; exact full construction replay verifies merges
before publication and foundation matching. It is metadata, never features.
Historical Tigris MID versus SPORC replay admits only the recipe-pinned
rtol=1e-12/atol=1e-12 mm tracking roundoff tolerance; structure and ineligible
tracking remain exact. Aggregate drift is recorded in `historical_mid_replay`.
This is distinct from exact same-platform NEW output replay/readback.
All artifacts bind hashes/parents and are atomically published in fresh roots.
Failed partial roots/claims are preserved for inspection, not silently retried.

Intermediate views require paired offline at training/oracle-validation time.
Only D000 and its input transform are deployable. Final-test inference and
materialization are absent. Readback/replay/canonical population checks are
mandatory; classifier quality is never an acceptance condition.
