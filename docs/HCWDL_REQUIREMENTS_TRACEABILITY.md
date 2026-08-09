# HCWDL Requirements Traceability

Status: locally implemented and verified ledger for
[`HIGH_COVERAGE_COLD_WARM_DISTILLATION_LADDER.md`](plans/HIGH_COVERAGE_COLD_WARM_DISTILLATION_LADDER.md).

`Local passed` means the named implementation and local verification exist.
Tigris execution is outside the authorization for this work and is not implied
by any status in this ledger.

## Block 1: contracts and packaged resources

| Requirement | Implementation | Contract | Verification | Status |
|---|---|---|---|---|
| Package selected donor config and fitted resources without donor runtime dependency | `src/hlt_classification/scouting/resources/highcov_v1/` | `HIGHCOV_MATCHER_RESOURCES/v1` | `tests/test_highcov_matcher.py` resource-load test | Local passed |
| Strict semantic/byte hash, schema, feature-order, coefficient, bin, scale, and isotonic validation | `highcov_resources.py` | `HIGHCOV_MATCHER_RESOURCES/v1` | tampering parametrization in `test_highcov_matcher.py` | Local passed |
| Record every donor file and commit `64be1a8` | `docs/LEGACY_SOURCE_MAP.md` | provenance rules | scaffold/link audit and manual map check | Local passed |
| Version all new artifacts without broadening PMARD contracts | `hcwdl_contracts.py` | contract constants in that module | `tests/test_hcwdl_contracts.py` | Local passed |

## Block 2: matcher and dense assignment cache

| Requirement | Implementation | Contract | Verification | Status |
|---|---|---|---|---|
| Exact particle container, p4/category/charge/track/native-index validation | `highcov_data.py` | matcher input v1 | donor parity and invalid-input tests | Local passed |
| Frozen edge features, broad gate, empirical and independent scores | `highcov_features.py`, `highcov_scorers.py` | matcher resources v1 | donor feature/score parity | Local passed |
| Cardinality-first lexicographic Hungarian with private dustbins | `highcov_assignment.py` | matcher v1 | conflict, ties, rectangular, exclusivity tests | Local passed |
| Eighteen post-assignment diagnostics and calibrated confidence | `highcov_assignment.py`, `highcov_calibration.py`, `highcov_matcher.py` | matcher v1 | donor diagnostics/confidence parity | Local passed |
| File-based train-fold scorer selection and leakage rejection | `highcov_matcher.py`, `highcov_cache.py` | assignment shard v1 | fold-lineage tests | Local passed |
| Native offline indices and lost-track exclusion | `highcov_matcher.py`, existing `particles.py` integration | assignment shard v1 | offset and forbidden-lost-track tests | Local passed |
| Dense int16/uint16 shards, logical/byte hashes, atomic publication | `highcov_cache.py` | `HIGHCOV_DENSE_ASSIGNMENT_SHARD/v2` | round trip/corruption/unclassified-token tests | Local passed |
| Manifest completeness, five-category plus explicit unclassified-token conservation, sampled recomputation, and `<10%` dustbins on complete train and validation roles | `highcov_cache.py` | `HIGHCOV_DENSE_ASSIGNMENT_MANIFEST/v2` and lock | 9-of-10, threshold, recomputation, and real-schema unclassified-token tests | Local passed |
| One-time cache consumption with no training rematch | `highcov_cache.py`, `hcwdl_views.py` | assignment authorization lock | matcher-call-count integration test | Local passed |

## Block 3: high-coverage repair families

| Requirement | Implementation | Contract | Verification | Status |
|---|---|---|---|---|
| Shell Exact warp `gamma(q)=2-1.3q` with explicit alpha endpoints | existing `repair.py` | `HIGHCOV_REPAIR/v1` | hand-calculated warp tests | Local passed |
| Shell Soft and HC Exact diagnostics | existing `repair.py` | `HIGHCOV_REPAIR/v1` | family dispatch/threshold tests | Local passed |
| All 21 fields, wrapped phi, p4, validity/applicability, nested discrete switches | existing `repair.py` | `HIGHCOV_REPAIR/v1` | all-field endpoint and nesting tests | Local passed |
| D0 exact HLT, D100 exact assigned offline, dustbins exact HLT, unchanged skeleton | existing `repair.py` | `HIGHCOV_REPAIR/v1` | endpoint byte-equivalence tests | Local passed |

## Block 4: complete ladder engine

| Requirement | Implementation | Contract | Verification | Status |
|---|---|---|---|---|
| Exact 23-node registry, domains, edges, and deployability | `hcwdl_ladder.py` | `HCWDL_NODE_SPEC/v1` | full registry snapshot/graph tests | Local passed |
| Cold fresh initialization and warm selected-weight continuation with optimizer reset | `hcwdl_training.py` | node spec/training report v1 | state and reset tests | Local passed |
| One- and two-teacher FP32 KD under BF16 forward | `hcwdl_training.py`, existing `engine.py` reuse | recipe and report v1 | analytic loss/autocast tests | Local passed |
| HLT-only deployable inputs and exact teacher domains | `hcwdl_ladder.py`, `hcwdl_views.py` | node spec v1 | forbidden-input and teacher-domain tests | Local passed |
| One-time RAM views and FP32 logit targets with deterministic identity joins | `hcwdl_views.py` | ephemeral cache/target v1 | construction-count, order, missing/duplicate tests | Local passed |
| Sixty pilot/production passes, validation every pass, natural rows, final partial batch | `hcwdl_training.py` | recipe/report v1 | pass-boundary integration tests | Local passed |
| Macro-AUC/CE/log-rejection/earliest checkpoint order and exact resume lineage | `hcwdl_training.py` | selection/report/resume v1 | tie and interrupted-equivalence tests | Local passed |
| Versioned primary/ablation recipe boundary and evidence builder | `hcwdl_recipe.py` | `HCWDL_RECIPE/v4` | complete primary decision, role-aware temperature, authenticated all-ones unweighted loss lineage, profile, placeholder, tamper, and evidence tests | Local passed |

## Block 5: orchestration, locks, reporting, and recovery

| Requirement | Implementation | Contract | Verification | Status |
|---|---|---|---|---|
| Smoke/pilot/midscale500k/midscale1m/midscale2m/production specs, exact registered role counts, and complete dependency DAG | `hcwdl_campaign.py`, `hcwdl_workflow.py` | `HCWDL_CAMPAIGN_SPEC/v7` (v3-v6 readable) | graph/dependency/mode-count tests | Local passed |
| Bootstrap smoke or measured prelaunch candidate and exact non-circular command-plan authorization | `hcwdl_campaign.py`, `hcwdl_authorization.py` | `HCWDL_COMMAND_PLAN/v3`, submission authorization v7 (v3-v6 readable) | first-miniature, automatic-waiver, and changed-lineage rejection tests | Local passed |
| Endpoint qualification infrastructure without validation-selected repair | `hcwdl_qualification.py` | qualification report/lock v1 | bad-performance continuation test | Local passed |
| Confirmation registry, finalist lock, execution lock, and sealed final-test claim | `hcwdl_locks.py` | lock family v1 | pre-lock denial and atomic claim tests | Local passed |
| Metrics, gap recovery, screen/confirmation/final aggregation | `hcwdl_reporting.py` | aggregate report v1 | metric and selection tests | Local passed |
| Exact-ID submission ledger, per-submit journals, monitoring, resume, superseded/replacement history, and cancellation | `hcwdl_campaign.py`, `hcwdl_recovery.py` | submission ledger v2, monitor/event/attestation v1 | mocked scheduler, partial-ledger, corruption, and exact-ID tests | Local passed |
| Thin CLIs, explicit manual/automatic endpoint continuation, dedicated-worktree binding, and absolute-path `exec python` Slurm worker | `scripts/*hcwdl*.py`, `sbatch/run_hcwdl_task.sh` | campaign spec v7 | help/static/dry-run tests | Local passed |
| Corrected-parent non-final scope through `finalist_lock`, with no execution-lock/final task authority and distinct authorization/submission/recovery phrases | `hcwdl_campaign.py`, `hcwdl_authorization.py`, `hcwdl_workflow.py`, `scripts/*hcwdl_campaign.py` | `HCWDL_CAMPAIGN_SPEC/v8`, `HCWDL_COMMAND_PLAN/v4`, submission authorization v8 | prefix closure, legacy-scope injection, phrase, dry-run, monitor, and recovery tests | Local passed |
| No live action without locked recipe, source, assignments, qualification, and explicit execution mode | workflow validators | all parent locks | fail-closed prelaunch tests | Local passed |

## Block 6: local acceptance and readiness

| Requirement | Implementation | Contract | Verification | Status |
|---|---|---|---|---|
| Donor parity for features/scores/assignment/native indices/diagnostics/confidence | ported frozen fixtures | matcher v1 | `tests/test_highcov_matcher.py` | Local passed |
| Complete bounded local behavioral smoke DAG over authenticated synthetic fixtures | `hcwdl_smoke.py`, `scripts/run_hcwdl_local_smoke.py` | smoke report v1 | `tests/test_hcwdl_smoke.py` and recorded report | Local passed |
| Complete 300k/100k/100k pilot dry run, no submission | `scripts/create_hcwdl_campaign.py`, `submit_hcwdl_campaign.py --dry-run` | campaign/ledger v1 | dry-run snapshot test and local command | Local passed |
| CLI help/validation coverage, complete local suite, static scans, and diff check | scripts/tests/docs | testing contract | final verification record | Local passed |
| Handoff evidence and explicit Tigris-only remaining acceptance | `docs/HANDOFF.md` | repository handoff rules | documentation review | Local passed |

## Fixed authorization boundary

This implementation session must not contact Tigris, submit Slurm jobs, push a
remote, or modify ROOT data. The final locally attainable status is:

> Local HCWDL implementation complete and ready for separately authorized
> Tigris validation.

## Local closure evidence

- Complete repository suite: 299 passed with 14 pre-existing
  Matplotlib/Pyparsing deprecation warnings.
- All 51 HCWDL/high-coverage Python surfaces compile; every one of the 25 thin
  CLI surfaces has a passing help test; the Slurm worker invariants pass.
- Standalone local smoke artifact:
  `%TEMP%/hcwdl_local_smoke_final_500de1f8e45f44ba8fb84b0eebca898a/smoke_report.json`;
  deterministic content hash
  `8f3157b434d6b371f3513de83ffa175d13cbb9e5a9d6e9a196849013b13dae0e`;
  all 23 nodes completed,
  each used two real optimizer updates, all five repair alphas were exercised,
  endpoint exactness passed, and final-test access was false.
- `tests/test_hcwdl_cli.py::test_complete_pilot_dry_run_is_nonmutating_and_exact`
  constructs and renders the complete 300,000/100,000/100,000 planning DAG,
  including uncapped source arrays and both manual and preauthorized-automatic endpoint boundaries.
- The only remaining acceptance is separately authorized execution with real
  Weaver/CUDA/data workers on Tigris, followed by measured resource locking.
