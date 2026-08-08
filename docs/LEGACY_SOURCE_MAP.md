# Legacy Donor-Source Map

The new repository is standalone. Donor code may be migrated only through an
explicit entry here and may never become a runtime import from `Fresh_check`.

## Inspected donor snapshot

```text
repository:
C:\Users\22rya\ComputerScience\CERN\Fresh_check

inspected commit:
bfcda5bd6fd48037ab198bcca1eadf6f4d87e131

worktree:
dirty at Block-1 inspection
```

Because the donor worktree was dirty, the commit alone is insufficient for
uncommitted donor files. Each migration block must additionally record the
exact donor path and source content hash used.

## Migrated code

Transfer Block 1 contained new scaffolding only. Transfer Block 2 extracted and
rewrote the following semantics without donor runtime imports:

| Donor path and SHA-256 | New paths | Retained semantics | Intentional changes and evidence |
|---|---|---|---|
| `jetclass_fresh/jetclass_data.py` — `30bbb947f83aeab9b190af640842071af075a558e1e12f7018f6ae96edf248ea` | `src/hlt_classification/data/{schema,identity,root_reader,splits}.py` | class order, longest-prefix mapping, 14 channels, 128 cap, transformations, bounded retries, chunk ordering, per-role sampling RNG, global shuffle | canonical recursive parent root; one root-relative namespace; label-bound identity plus label-free leakage key; strict branches/lengths/finiteness/PID/one-hot checks; stronger capacity/balance/overlap audit; versioned self-hashing immutable atomic manifest; streaming iterator. Covered by `tests/test_data_foundation.py`. |
| `scripts/build_jetclass_splits.py` — `87293fff44a413c5e6e833a2bfcddf422fe3e2017a077b6bbaade9ecba99b222` | `scripts/build_splits.py` | explicit split sizes and report | rewritten standalone CLI; single recursive root; production fail-closed; authenticated atomic output. |
| `scripts/preflight_relational_part_data.py` — `c15cb8c30100809f6b6804ccaea425b545e3598c41e877c0b7d0d7682c8ea780` | `scripts/preflight_data.py` | class capacity preflight | generic standalone CLI; full branch schema validation; diagnostic skips can never be production-eligible. |

The donor test reference was
`tests/test_jetclass_data.py` at
`80095b50141e34249b4561c6293f710e87559fb3a13362bf030e6808042d2c7d`.
The new suite broadens its assertions and uses actual synthetic ROOT fixtures.
No third-party source was copied, so this block adds no new source-license
obligation beyond the declared Python dependencies.

Transfer Block 3 migrated the registered HLT-v3 v1 scientific kernel:

| Donor path and SHA-256 | New path | Retained semantics | Intentional changes and evidence |
|---|---|---|---|
| `teacher_logit_reco/relation_expert_token_bridge/hlt_v3.py` — `3f8509486818469fc63990eccb9d09182df460ae593efef474f4a6a3b9311037` | `src/hlt_classification/data/hlt_v3.py` | profile parameters, degradation profiles, type multipliers, PID transitions, operation order, substream IDs, threshold/loss/merge/response algorithms, mass preservation, stable sorting, validity states, and diagnostics | removed RETB/cache imports; strengthened invalid-type and finite validation; assigned serialized metadata a repository-specific contract name; retained donor random-domain bytes for registered-v1 output parity. Golden array and diagnostic hashes plus focused mechanism tests are in `tests/test_hlt_v3.py`. |
| `teacher_logit_reco/relation_expert_token_bridge/replicas.py` — `fe3013ac4543d048287527bb326c6ea943251b9448170c4f2fbc1dcc6181ffa8` | `src/hlt_classification/data/replicas.py` | policies, donor role seeds, random multipliers, identity cycle, event seed derivation, and evaluation replica zero | removed RETB contract dependency; added canonical `model_val=3054` and `stack_train=3055` domains required by the new five-role split while preserving every donor-existing mapping; added a repository-specific self-authenticating manifest contract, strict input validation, and detached manifest payloads. |
| `jetclass_fixed_hlt.py` — `c222a9233b1a5284a57bbcf2e253c83d3d7679ff26f5e825c854e8cece3bb2e4` | `src/hlt_classification/data/hlt_v3.py` | phi wrapping, local density, v2 efficiency base terms, and v2 kinematic base terms only | excluded legacy v1/v2 builders, CLI, caches, and unrelated profiles; inlined only the inherited formulas needed by registered HLT-v3. |

The focused donor test reference is
`tests/test_relation_expert_token_bridge_step2.py`. Block 3 does not copy
third-party source and introduces no new dependency or license obligation.

Transfer Block 4 inspected both legacy cache surfaces and rewrote them rather
than copying either monolithic format:

| Donor path and SHA-256 | New paths | Retained semantics | Intentional changes and evidence |
|---|---|---|---|
| `jetclass_fresh/hlt_cache.py` — `5631ac115a4f352739492e0db06c5e9c3111fa52d522d273243504e0731823c4` | `src/hlt_classification/data/{cache_contracts,offline_cache,dataset}.py` | bounded cached raw views, stable identity order, reusable HLT-facing dataset access | split offline and degraded caches into distinct versioned contracts; deterministic NPZ bytes; per-array and per-file hashes; atomic immutable sidecars and manifests; bounded cross-shard reads; exact source/split/schema/generator lineage. |
| `teacher_logit_reco/relation_expert_token_bridge/hlt_cache.py` — `57d0dde4e4aad5aa622fc62c2020954ae69a0108cf89178d7304dabd2f6345e4` | `src/hlt_classification/data/{cache_contracts,hlt_cache,dataset}.py` | deterministic degradation from authenticated offline parents, replica/profile binding, resumable shards, aggregate diagnostics | removed RETB campaign coupling and construction-index exposure; added source-snapshot and exact parent-range hashes; made processing-batch and physical-shard layouts semantically invariant; added standalone build/audit CLIs. |

Block-4 failure injection and byte-level resume/layout evidence is in
`tests/test_cache_pipeline.py`. No donor runtime import or third-party source
was copied.

Transfer Block 5 migrated the canonical model-input and adapter semantics:

| Donor path and SHA-256 | New path | Retained semantics | Intentional changes and evidence |
|---|---|---|---|
| `jetclass_fresh/part_inputs.py` — `6e237ba0eec33ef9a2b78cde579896b02c5b739ccd8126de0512774eef30166a` | `src/hlt_classification/data/part_inputs.py` | two point coordinates, exact ordered 17-feature transform, four-vectors, supplied-view axis reconstruction, padding, and all-empty safety | removed donor package and candidate-weight coupling; accepts the standalone 14-token `JetView`; adds strict dtype, finite, padding, identity, label, and explicit source-view validation plus a repository-specific contract. Covered by `tests/test_part_inputs.py`. |
| `jetclass_fresh/hlt_baseline.py` — `982c1696967b809fd534add1766f9b780fb858297f334036a7d9aa84f078a418` | `src/hlt_classification/models/particle_transformer.py` | lazy Weaver construction, exact standard-four canonical architecture, input delegation, and class-token no-weight-decay name | excluded donor dataset/training code; mixed precision is disabled by the authoritative FP32 runtime rather than passed as a noncanonical model constructor argument; adds logits/input-gradient/parameter-gradient/mask/state-dictionary attestation. Covered locally with an interface-faithful test double; authoritative real-Weaver execution remains pending. |

Both donor files were inspected at commit
`bfcda5bd6fd48037ab198bcca1eadf6f4d87e131` and were clean at their recorded
paths. No Weaver source was copied. `weaver-core` and PyTorch are optional
model dependencies and retain their upstream licenses.

Transfer Block 8 copied and contract-rebased the reviewed HOSD scientific
plan:

| Donor path and SHA-256 | New path | Retained semantics | Intentional changes and evidence |
|---|---|---|---|
| `teacher_logit_reco/HLT_OFFLINE_STRUCTURE_DISTILLATION_IMPLEMENTATION_PLAN.md` — `0d4eaab71c13dd64a4af326d085f35259490b90eca084c60d6c6c3a24325a937` | `docs/plans/HLT_OFFLINE_STRUCTURE_DISTILLATION_IMPLEMENTATION_PLAN.md` | all scientific questions, target definitions, controls, split counts, access rules, stages, selectors, metrics, statistics, locks, and acceptance criteria | replaced old repository/runtime paths with `src/hlt_classification/hosd` and local reusable contracts; retained optional donor comparators; required local ports plus donor-parity tests for missing RPT/RETB semantics; reset donor-only implementation statuses. No executable donor code was copied by this document migration. |

## PRAD implementation provenance

PRAD was implemented from the user-supplied repository-root documents
`prad_implementation_agent_prompt.md` and
`privileged_relational_attention_proposal(1).md`. No executable donor file,
checkpoint, cache, metric result, or historical campaign artifact was migrated
for PRAD. The implementation extends the already mapped canonical Weaver
wrapper and standalone data/campaign contracts; consequently there is no new
donor commit or donor-file hash to register for this campaign.

## PMARD implementation provenance

PMARD was implemented independently from the active repository plan and the
read-only operational guide
`C:\Users\22rya\ComputerScience\FCV\SCOUTING_AK8_USAGE_GUIDE.md` (SHA-256
`c80b3f3459848c4f634f8f3a861fc83da0bfa530f71e314826123d8dc6774ac9`).
The adjacent `scouting_ak8_tools` directory has no Git metadata, so no
executable file from it was copied or imported at runtime; without an exact
donor commit it is an operational reference only.

The repository vendors the data-only CMSSW V00 preprocessing transcription
under `configs/scouting/`. Its authoritative upstream source is
`cms-data/RecoBTag-Combined` commit
`37a104b51c8458bcda97a95ba52d304a5041e922`; the documented transcription hash
is `2e346c5da8491046205f29cba21b0f6e434e273eb89dae068aa5dd40e2ca91f7`.
The independently authored `TOFF/v1` JSON records the semantics and the
documented reference-YAML hash
`32da128ad0e82c6933a36cf409ae766523a116ae66df8d43153c671bc89f7c1d` plus
Weaver reference commit `6a084ec341f5af2e06f6d884e6a4ee4dab8bb4e4`. No Weaver source is copied.
The local JSON byte hash is also recorded by the config validator; it differs
from the guide's reference-file byte hash only by terminal newline encoding.
Scientific validation compares the parsed, ordered preprocessing semantics and
retains both hashes rather than falsely claiming byte identity.

### Fitted-strict matcher port

The matching study directory has no Git metadata, so this port is bound to
exact source hashes rather than inventing a donor commit:

| Donor path and SHA-256 | New paths | Retained semantics | Intentional changes and evidence |
|---|---|---|---|
| `HLTvsOfflineComparisons/matching/run_selective_matcher_campaign.py` — `68e2bbf1098506c37dbd5af7a1265db89bfbece19b7f7a6a00ab99605e2544d0` | `src/hlt_classification/scouting/fitted_strict.py` | fitted-strict edge features/signs, empirical LLR lookup, broad and strict gates, rectangular Hungarian private dummies, assignment diagnostics, confidence calibration, exact operating point | refactored inference only into a stable typed API; no implicit fitting; native offline indices are restored after lost-track exclusion; fail-closed artifact validation; numerical golden parity in `tests/test_fitted_strict_matcher.py`. |
| `HLTvsOfflineComparisons/matching/FITTED_STRICT_MATCHER_IMPLEMENTATION_GUIDE.md` — `d3b2adc32df67b0ee971a9383fadbae00540b4022be64a921b7d2d90c361733e` | `docs/contracts/FITTED_STRICT_MATCHER.md` | algorithm, population, selective-mask, and PMARD preservation contract | condensed into a repository-local versioned contract and explicitly blocks the incompatible legacy complete endpoint. |
| canonical `fitted_edge_model.json` — donor byte SHA-256 `2cfd51b209435164263247252a35cf83708fa71fc2580301c35bb5d905d41142`; `confidence_models.json` — `dc1ce2f3f889bef3e3c6cdc46da70bc14e79dff518d6d4b1d759fac51f093735`; `independent_validation.json` — `af1d2b009e7c9186b454acd461e4cf40335261ae4b491f1c82dd1957e941687e` | `src/hlt_classification/scouting/resources/fitted_strict_v1/` | exact parsed model/audit values from the intact 7,500-jet bundle | JSON semantic hashes are authenticated so Windows/Linux checkout line endings cannot change identity; original donor byte hashes remain in the vendored provenance file. No checkpoint, dataset, or race-corrupted result directory was copied. |

The independent donor validator
`validate_selective_matcher_results.py` was inspected at SHA-256
`f83b7e42bdd9d6b28292727fa31c341e22a7f1a4668b0eb0df52cad84a628ae3`.
No runtime import reaches the external FCV tree, and no new third-party source
or license obligation was introduced.

The subsequent selective campaign plumbing—`selective_assignment.py`, the two
builder CLIs, selective 21-field repair, assignment authorization, pilot DAG,
and oracle evaluation—is original repository-local work with no donor file or
commit. It composes the authenticated matcher above with existing Scouting
contracts; tests cover sparse joins, exact unmatched preservation, endpoint
replacement, and campaign registration.

## HCWDL high-coverage matcher port

The HCWDL completion-shell matcher was ported from the independent, clean
repository `high_coverage_matcher_research` at exact commit
`64be1a82f11f42949fdffa639a869ccea2528bfa`. The portable implementation
guide `docs/OUTSIDE_AGENT_IMPLEMENTATION_HANDOFF.md` was read in full. The new
runtime has no import, path lookup, or other dependency on that repository.

| Donor path and SHA-256 | New path | Retained semantics | Repository integration |
|---|---|---|---|
| `src/highcov/assignment.py` — `44dee1847dbc5287ee97c738468abf6be7a551831180130ace5ae8563166bba7` | `src/hlt_classification/scouting/highcov_assignment.py` | cardinality-first lexicographic assignment, private dustbins, consensus assignment, and 18 diagnostics | package-relative imports; immutable assignment shards and parity tests |
| `src/highcov/calibration.py` — `6dcbfefa0222ddd9a06a5cc1fdb7bfc5f137a00cdcef56c1f1c6878d034aa832` | `src/hlt_classification/scouting/highcov_calibration.py` | frozen isotonic confidence calibration | strict packaged-resource validation and uint16 persistence |
| `src/highcov/data.py` — `7e4b2ce99abab15b7abc06a1aae223ba58b5d81c5e16ad59c58f998cf18f8676` | `src/hlt_classification/scouting/highcov_data.py` | validated particle container, kinematics, categories, charge, measurements, and native indices | Scouting particle adapter; lost-track exclusion before matching |
| `src/highcov/features.py` — `db2a1206686697003bb6b8a2803b340424a0f56f667c08ac29286c559cffe597` | `src/hlt_classification/scouting/highcov_features.py` | frozen candidate gate, edge matrices, ranks, and feature order | donor-parity fixture and fail-closed schema validation |
| `src/highcov/scorers.py` — `77debf7a892c6e16149027563f91fb68433f58638ae965b3ed2a20e1c829d97c` | `src/hlt_classification/scouting/highcov_scorers.py` | empirical and independent consensus scores | role/fold scorer selection is enforced by `highcov_matcher.py` |
| `src/highcov/final_matcher.py` — `78a758ef5d3333d67d1e2aa193af2eed06edcb7d04b483f2f788e4722ddbc897` | `src/hlt_classification/scouting/highcov_matcher.py` | selected global matcher and post-assignment confidence | train cross-fitting, validation/audit model policy, native-index output |
| `src/highcov/hashing.py` — `fb602d4048bb8f8497dc899a219e55679272046b424dccd259a2e40c4bdec00a` | `src/hlt_classification/scouting/highcov_hashing.py` | canonical semantic hashing | composed with repository byte/content hashes and atomic publication |
| `configs/selected_matcher.json` — `eab6c945c058a5447063066eca79c931434d06d63e3729d1dfeeeab354100a87` | `src/hlt_classification/scouting/resources/highcov_v1/selected_matcher.json` | exact selected algorithm/configuration | packaged runtime resource; parsed semantic hash is also checked |
| `artifacts/models/empirical_models.json` — `6a9f8d594b0bfadae03a80187917fac2c8261cb0345965daa2f33af4e6836389` | `src/hlt_classification/scouting/resources/highcov_v1/empirical_models.json` | full and four holdout empirical scorers | exact fold selection and train-leakage rejection |
| `artifacts/models/final_confidence_calibration.json` — `40a9c4499244916e58a1f8c74a52d05aac0418a21220b27d03afcd97a8454c31` | `src/hlt_classification/scouting/resources/highcov_v1/final_confidence_calibration.json` | final confidence calibrator | exact semantic/resource validation and quantization tests |

The donor's research-only completion/contextual/inference/synthetic surfaces,
generated results, and datasets were deliberately excluded. The 21-field
Shell Exact/Soft/HC Exact integration is repository-local work in
`scouting/repair.py`; no donor `repair.py` runtime was copied. The port adds no
new third-party dependency or license obligation.

## Approved transfer surfaces

| Transfer block | Donor surface | Intended retained meaning | Migration policy |
|---|---|---|---|
| 2 | `jetclass_fresh/jetclass_data.py` | schema, identity, split, chunked ROOT logic | extract and test; do not copy package imports |
| 2 | `scripts/build_jetclass_splits.py` | CLI behavior | rewrite thin local CLI |
| 2 | `scripts/preflight_relational_part_data.py` | data-capacity preflight | rewrite generic local CLI |
| 3 | `teacher_logit_reco/relation_expert_token_bridge/hlt_v3.py` | HLT-v3 scientific kernel | migrate with semantic parity |
| 3 | `teacher_logit_reco/relation_expert_token_bridge/replicas.py` | deterministic replica seeding | migrate with parity |
| 3 | `jetclass_fixed_hlt.py` | v2 base-term helpers used by HLT-v3 | extract only required helpers |
| 4 | `jetclass_fresh/hlt_cache.py` | bounded cached view and dataset semantics | inspect and rewrite as authenticated shards |
| 4 | `teacher_logit_reco/relation_expert_token_bridge/hlt_cache.py` | degraded-cache lineage and resume | inspect and decouple from RETB |
| 5 | `jetclass_fresh/part_inputs.py` | canonical ParT input transforms | migrate with exact tests |
| 5 | `jetclass_fresh/hlt_baseline.py` | Weaver wrapper and baseline configuration | extract model surface; rewrite trainer |

For each actual migration add:

```text
donor commit
donor path
donor file SHA-256
new path
retained semantics
intentional changes
parity tests and results
third-party license/attribution impact
```

## Deliberately excluded

Do not migrate checkpoints, caches, logs, run registries, historical campaign
IDs, unrelated campaign packages, old handoffs, or broad script collections.
