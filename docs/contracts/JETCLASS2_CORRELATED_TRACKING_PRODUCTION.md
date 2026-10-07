# JC2_CORRELATED_TRACKING_PRODUCTION/v1

Authority: [production plan](../plans/JETCLASS2_CORRELATED_TRACKING_PRODUCTION_PLAN.md).

New namespace JC2_CORRELATED_TRACKING_PRODUCTION_{KIND}/v1, schema 1.
All records have content hashes, explicit parent hashes and immutable atomic
publication. BUNDLE binds the approved pilot job/commit, authenticated original
spec/receipt/report/source, selected CORR_MID recipe and execution environment.
STUDY binds original OFFLINE source, TRAIN_1M profile, donor profile/inventory,
exact population/shards, storage attestation and production source.

RECIPE wraps the unchanged historical pilot RECIPE and selects CORR_MID at 1.
INPUTS specifies the common 17-input asinh/log1p frontend. It does not apply any
CONTEXT_V1 transform. Historical pilot no-training flags remain unchanged;
downstream training is a new separately authorized experiment, not pilot work.

Banks contain exactly offsets (int64 N+1), jet_identity (uint8 N by 32), p4
(float64 M by 4, px/py/pz/E GeV), charge/category (int8 M), tracking (float64
M by 4, d0/dz/d0err/dzerr mm), valid (bool M by 4). Invalid placeholders and
masks remain unchanged. Category order: charged hadron, neutral hadron, photon,
electron, muon, unknown. Identities/offsets are metadata, never neural features.

SHARD verifies exact ordered membership, block hashes/schema/readback, unchanged
structure for every input jet, total particle count and producer attempt. No
inverse-transform diagnostic is applicable or fabricated. Missing structure
evidence, altered source, corrupt bytes or wrong membership fail closed.

ROLE_MANIFEST publishes train/validation separately. MANIFEST requires all
1M/250k/1M rows and identifies CORR_MID plus its input contract. Both bind all
relevant receipt hashes. Public read_role refuses final_test. Generation/test
materialization flags are truthful; final_test_evaluated always remains false.
TEST_BUILD_LOCK grants materialization only. It is not an evaluation lock.

One exact reviewed PLAN queues preflight, array and finalizer. Dependencies are
afterok for generation and afterany for finalization (which fails closed on
incomplete receipts). Live authorization is
`AUTHORIZE JC2 CORRELATED MID DATASET INITIAL EXACT PLAN`.
Durable per-call submission intents/responses and exclusive claims prevent
duplicate submission. Recovery never replaces committed physical blocks.

Read paired data by canonical jet identity using study.population metadata and
population.iterate for original OFFLINE. Verify exact row-wise p4/category/
charge/masks before using one-to-one training pairs. Original keys must not be
inferred from exported ordinal keys for regeneration: replay reads raw source.
