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

## HCWDL representation-KD source integration

The full-data three-track 60-pass campaign integrates the reviewed
representation-KD implementation from clean commit
`acecf9f74dab3d4ac675d8160cfb5decf83ba680`. The development worktree was used
only to locate and review the files; it is not a runtime import, artifact
parent, or campaign path. The campaign integration lock additionally checks
the exact committed Git blob IDs of the twelve byte-preserved scientific
modules before live creation.

| Donor path and SHA-256 | New path | Retained semantics and integration |
|---|---|---|
| `src/hlt_classification/models/scouting_particle_transformer.py` — `c6128610d4d249984a8d7af874737e1dd2e00337466ebdd91a119424dbf765cc` | same path | Merged one-forward token/jet/relation surface capture into the current unified model while preserving ordinary-logit parity. |
| `src/hlt_classification/models/hcwdl_representation.py` — `49d754e3cae1abfbb1c332c90f2db15245303ffbe5ccf2a649e647a25fcd5219` | same path | Exact representation heads, student wrapper, and head-free deployable extraction. |
| `src/hlt_classification/models/hcwdl_surfaces.py` — `90d8449e862c10e8fe8a73c305d338b7411a4e74c734b353c9be1ccdfa7027ae` | same path | Exact typed one-forward surface container and validation. |
| `src/hlt_classification/scouting/hcwdl_representation_graph.py` — `5205b158aabc2f31587c5554f4bccadcdf3a878e2b1fa12423481a77522ccb0e` | same path | RSET/RREL strategy identities and paired seed semantics. |
| `src/hlt_classification/scouting/hcwdl_representation_artifacts.py` — `45cea8226a6e752857b407031d330453f10e999130a1daa9d4aeb33a2a7a5fb7` | same path | Exact representation artifact-envelope and lineage helpers. |
| `src/hlt_classification/scouting/hcwdl_representation_contracts.py` — `808ca40ed06ecf2ac0df22b2a51c1fc0854c8be1b870aa9beab733654e9c37c1` | same path | Exact versioned representation contract validators. |
| `src/hlt_classification/scouting/hcwdl_representation_data.py` — `0fa2f085e5646d2410135f61968ab139048b212cf1dacb740467c865689d9bbd` | same path | Exact representation-aware input and batch containers. |
| `src/hlt_classification/scouting/hcwdl_representation_recipe.py` — `0cc302ce6aa862506db32112038913cb4b9bd58bd64294d50c7193153188c1ed` | same path | Corrected representation recipe v5 schedules, kernels, and calibration. |
| `src/hlt_classification/scouting/hcwdl_representation_kernels.py` — `c236e668f481d6724f50eca59b693b376000e42788376d483206ed1f7668a173` | same path | Spectral resources, finite kernels, weighted means, and reference oracles. |
| `src/hlt_classification/scouting/hcwdl_representation_losses.py` — `413bcaebf645dce0811d23c1d3093b2255512f34c6c1b5b2c44560d0edf9d2e3` | same path | Jet, set, relation, topology, scheduling, and support-aware losses. |
| `src/hlt_classification/scouting/hcwdl_representation_calibration.py` — `c9a94361e09e27eca999e3e93fcf691f5e4d026c8f644b44294d800f311ae9c1` | same path | Train-only deterministic calibration with state/RNG restoration. |
| `src/hlt_classification/scouting/hcwdl_representation_targets.py` — `97324a3b492ab21785d0fe6be3184afbf91cc7521c93b1881de877965de65b0a` | same path | Target schemas, identity hashes, validation, and support metadata. |
| `src/hlt_classification/scouting/hcwdl_representation_target_runtime.py` — `450a7843293fe69a73306decc086385c650487c36b63cfc17e0c045d4e5a8b2b` | same path | One-forward in-memory carrier target construction and runtime audits. |
| `src/hlt_classification/scouting/hcwdl_representation_training.py` — `16f344be23a7ce40ed7c4a45222a8ca769953ffcfbfc06d82a08c82135817f1b` | same path | Representation model initialization, identity joins, normalized components, diagnostics, and extraction. |
| `src/hlt_classification/scouting/hcwdl_homotopy_stream.py` — `de49beb816ab4f80e83d0159e3143744d3b935cca8428eea1935382a4affcd04` | same path | Merged fixed-work-per-source-chunk endpoint streaming into the current homotopy stream. |
| `src/hlt_classification/scouting/hcwdl_homotopy_representation_training.py` — `bbf468f1e9bcb199dc5ebce790391f47ec20fbcca79eaf440f56d73d538ae29d` | same path | Retained as the numerical wiring reference for carrier-view target generation and training. |
| `src/hlt_classification/scouting/hcwdl_homotopy_representation_targets.py` — `e2e10e74c6009f5cecf1fb19fa1ff90045bd76f505caeed053b6821e45ff11d9` | same path | Retained only as a durable-bank numerical oracle; TRI60 never calls its publisher. |

The following exact same-commit support modules were also migrated so the
representation package and its regression tests remain standalone. They do
not authorize TRI60 to use a historical campaign graph, rolling-resume
publisher, or durable representation-target publisher.

| Donor path and SHA-256 | New path | Campaign role |
|---|---|---|
| `src/hlt_classification/scouting/hcwdl_direct_offline_kd_graph.py` — `16fd891eb2358f17af4e1a82a4628722ba96aad15304e73a13a0f5248c6bf312` | same path | Donor graph validation/reference only. |
| `src/hlt_classification/scouting/hcwdl_homotopy_representation_contracts.py` — `593c5ac9215490abab388f315fd1d81b226ca9d0a2bfee5d07bce874219e613a` | same path | Donor contract validation/reference only. |
| `src/hlt_classification/scouting/hcwdl_homotopy_representation_graph.py` — `d0841a608d1dccf71fa5d474f2ba1e6f19a489f31578079687903bf65d8670c6` | same path | Donor graph/reference fixtures only. |
| `src/hlt_classification/scouting/hcwdl_homotopy_representation_recipe.py` — `84ad20239df5b1b7c7ffab60caccaab36c8ac83e6265189742335e1ce18c4357` | same path | Donor recipe/reference fixtures only. |
| `src/hlt_classification/scouting/hcwdl_numerical_acceptance.py` — `3aa8f02919bbfb964fb7225486c929185744c7e7d128c703290f4295f4687596` | same path | Numerical acceptance helpers reused by regression tests. |
| `src/hlt_classification/scouting/hcwdl_paired_bootstrap.py` — `79d8f5f3e206b3137163208e0780226513e6cc08bb5a6fc954daf12fe049fb1e` | same path | Paired reporting/bootstrap reference. |
| `src/hlt_classification/scouting/hcwdl_parent_loss.py` — `879c8fc49263006375226398659ad8739cb729222f320f7c20fbff8c429bb2bf` | same path | Parent-loss reference and validation. |
| `src/hlt_classification/scouting/hcwdl_representation_graph_registry.py` — `63881ffc2e3554b15a9aefab709b12b6d697fb7ad6354066ebc63d2ca0163049` | same path | Donor strategy registry/reference. |
| `src/hlt_classification/scouting/hcwdl_representation_reporting.py` — `ea6bf42fed13384bdcca13381eb280f0b696dd913ac215640a34950bdf725b6d` | same path | Representation diagnostics/reference reporting. |
| `src/hlt_classification/scouting/hcwdl_representation_resources.py` — `511e17de3a88e6d9c5670af117ce2095680109383be31a07307ad124ab2a611e` | same path | Resource-envelope reference. |
| `src/hlt_classification/scouting/hcwdl_representation_resume.py` — `9333574c5edd3b8913df9f8f6039340817f8ec6fc037c9082434773ae213d223` | same path | Numerical no-resume equivalence oracle only; TRI60 never invokes its publisher. |

The donor regression references were
`tests/test_hcwdl_representation_math.py` (`83a1592346ef0d94f8cff7d313d1298cf98ae78e6f425508e55504a7cbe8ef25`),
`tests/test_hcwdl_representation_model.py` (`7ef207e1d916265a89f83d57c1709f6a5a8ea4969d8fa458c9a87dd4fe2e5d6b`),
and `tests/test_hcwdl_representation_training.py`
(`749f1d76c87c720200f4a0c18e829e2e2143ea24553755df893ff8995bdb7485`).
The math and training assertions were retained; the model fixture was adapted
to the current direct model-surface API. No runtime import from the donor
worktree remains.

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
