# CMS2JC2 reduced frozen confirmation / v1

Authority: `docs/plans/CMS2JC2_REDUCED_CONFIRMATION_PLAN.md`.

New response kinds: FROZEN_REDUCED_STAGE, FROZEN_REDUCED_RETIREMENT,
FROZEN_REDUCED_REPORT, each `CMS2JC2_RESPONSE_<KIND>/v1`.
Original FROZEN_STAGE/REPORT keep their all-shards semantics.

The new stage pins a source-bound DEV_STUDY, the original stage and ledger,
the exact 36 receipt/result hashes for indices 0..35, and coverage computed
from the original 57-shard membership. Only the excluded 21 evaluations and
original final report may be retired, under explicit exact-plan authority.
All 36 included outputs are validated before aggregation. No raw ROOT reads.

Reduced report: frozen JOINT plus non-selecting B_DZ, original three dependent
replicas, unchanged checks and paired file bootstrap. `decision` describes
only the included sample; `full_population_status` is always
`inconclusive_incomplete_population`; `all_registered_jets_included=false`,
`production_qualified=false`, `transfer_authorized=false`,
`final_test_accessed=false`. Count file/source gaps explicitly; a prefix of
file-ordered shards is not a random sample of all CMS confirmation jets.

Production requires explicit reduced-evidence acknowledgement and uses
`CMS2JC2_PROXY_STUDY_REDUCED_CONFIRMATION/v1` and
`CMS2JC2_PROXY_DATASET_REDUCED_CONFIRMATION/v1`. Both preserve scope/coverage
and carry `physics_production_qualified=false`. Other production invariants
and final-test materialization/evaluation separation remain unchanged.
