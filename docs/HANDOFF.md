# Current Handoff

## Active PMARD implementation (2026-08-05)

The user explicitly activated the PMARD mandate. The active scientific plan is
`docs/plans/SCOUTING_ALPHA_REPAIR_DISTILLATION_OPTIMAL_CAMPAIGN.md`; PRAD is a
separate prior campaign and cannot redefine PMARD.

Implemented locally in the current worktree:

- version-4 Scouting schema with the fitted-strict donor's native
  constituent-count boundary (`n_cpfcands` plus `n_lts`) as the authoritative
  charged/lost-track layout; the auxiliary `cpfcandlt_isLostTrack` model
  feature is deliberately not a matching-projection or boundary requirement;
  plus versioned source, label, row-identity, split, artifact, and
  role-capability contracts;
- version-2 60/20/20 source-file split construction: the source audit streams
  per-file 15-class counts, an exact MILP minimizes the worst class-by-role
  fraction deviation under whole-file disjointness, and validation fails
  closed above a predeclared 0.02 absolute tolerance;
- exact 15-class mapping, 21-channel CMSSW transform, 200-token HLT view, and
  90/60 native-offline no-SV view;
- projected bounded ROOT streaming with rank/worker file partition and sealed
  pre-lock final-test branch access, plus eight-file round-robin train reads
  and deterministic natural-population 32,768-row RAM shuffle buffers;
- v2 sparse candidate graphs with audited node/edge/track-validity features,
  three-round bipartite message passing, positive-unlabeled bootstrap,
  two ultra-conservative pseudo-label rounds, held-out edge and post-assignment
  calibration, deterministic score quantization, exact alternative margins,
  Hungarian/OT controls, and `M0`--`M5`;
- adversarial synthetic loss/fake/split/merge/category-confusion tests,
  genuinely different event-mixed jets, independent HLT/offline perturbation
  stability, confidence Brier and matching-only category gates;
- the canonical `fitted_strict` selective matcher as a standalone production
  module: immutable semantic hashes for the 7,500-jet edge/confidence/audit
  bundle, exact empirical LLR scoring, strict PID/charge and broad kinematic
  gates, rectangular Hungarian private dummies, calibrated ten-diagnostic
  confidence at threshold `0.9828147479721088`, native-index restoration after
  lost-track exclusion, batch inference, and persistent-cache construction;
- deterministic class-proportional row-selection manifests and immutable
  fitted-strict assignment caches computed once per selected jet: compact
  per-source CSR/DEFLATE shards store only entry, uint8 HLT index, uint16
  offline index, and uint16 confidence; consumers use a bounded lazy LRU and
  verify split/selection/matcher/data hashes before reuse;
- exact `P4_ONLY/v1` alpha endpoints and byte-identical `alpha=0`, plus charged
  eligibility, shuffled/corrupted matches, direction/response/wrong/random
  controls, log-angular interpolation, and confidence weighting;
- versioned `SELECTIVE_FULL_PARTICLE_ENDPOINT/v1` mixed-type repair: unique
  accepted assignments, exact all-21-channel plus p4 offline replacement for
  accepted tokens at `alpha=1`, byte-identical unmatched HLT tokens, raw
  continuous interpolation, SHA-256 identity-bound nested discrete/validity
  switches, charged/neutral track-applicability coherence, and unchanged HLT
  count/order/mask without durable repaired datasets;
- a hash-chained selective-assignment authorization boundary binding the
  canonical fitted-strict artifact and threshold, split, deterministic row
  selection, compact train/validation assignment manifest, all five input
  categories, and the exact unmatched-as-HLT policy before any privileged
  teacher or student;
- 21-input/15-output canonical Weaver adapter, native-offline diagnostic model,
  K0--K6 loss semantics including a valid alpha-zero K2 collapse, class
  weights, separate `R4_PAIR`/`R4_GRAM` representation arms, corrected
  per-signal QCD discriminants, and a fixed three-generation registry whose
  representation and anchored-companion paths remain executable when the
  selector legitimately chooses alpha zero via an exact HLT identity alias;
- cross-file full-batch packing, Weaver no-decay optimizer exclusions,
  named RNG domains, FP32 CE/KL/recursive representation and pair/Gram math
  plus FP32 RAM logits under BF16 ParT autocast, eight-check validation cadence,
  durable 50-update interval-mean loss history, validation/preemption-only
  rolling checkpoints, batch-shell-to-Python `exec` signal delivery, exact
  batch-offset/RNG/optimizer/best-state resume, a shared deterministic
  4,096-row smoke selection, one bounded RAM-cache miniature, and HLT-only post-cache streams
  with unused GPU teachers/matchers released for R0 and zero-coefficient
  controls, plus the HLT-only inference signature,
  fixed observer-only stratification and final nested paired statistics;
- hash-chained freeze locks and a minimum-storage Tigris DAG whose smoke graph
  cannot access final test, with exact-ID monitor/resume/cancel, authenticated
  resource/storage/miniature evidence, nonmutating production dry run, and
  explicit production authorization gates; a pilot graph uses exact
  300k/100k/100k row selections, constructs final-test selection/assignments
  only after the execution lock, and reports HLT, native-offline, and
  matched-offline endpoint oracles alongside deployable finalists.

Local verification now passes `208 tests, 14 warnings`, including selective
all-field endpoint, persistent sparse join, pilot-DAG, matcher, KD, resume, and
legacy campaign regressions. CLI help and `git diff --check` also pass; the
remaining notices are repository line-ending notices.

The first genuine pilot DAG, campaign `pmard_pilot_62dd448731cdd932` at pushed
commit `00f03a6b1818ad16f9538e8e9e155082293677cd`, reached the persistent
assignment-cache array (`43465_0`--`43465_42`) after its source, split,
feature-audit, row-selection, Weaver, and matcher-artifact prerequisites
completed. Every assignment shard failed before publication because the port
required `cpfcandlt_isLostTrack` to equal a positional collection boundary.
That check was stricter than the fitted-strict donor, which slices regular
charged candidates from the authenticated scalar counts. Schema v4 corrects
the projection and decoder contract: `n_cpfcands + n_lts` defines the charged
family and the appended lost-track suffix, while the auxiliary model feature
cannot redefine that boundary. No matcher-quality or PMARD result can be
inferred from this infrastructure failure, and the immutable v3 pilot must not
be resumed under v4 source.

The schema-v4 replacement pilot, campaign `pmard_pilot_3f379cb66398fcc8` at
commit `775bed70`, completed its persistent assignment cache, baseline budget
grid, budget selection, and temperature grid. Privileged-teacher array
elements `43733_1`--`43733_5` then failed before launching the trainer because
the workflow appended the authenticated matcher threshold to `argv` as a
Python float after `_script` had stringified the initial arguments;
`subprocess` accepts only text, bytes, or path-like arguments. The same latent
fault existed in the nonzero-alpha student builder. Both builders now
normalize their complete final command vectors to strings, with regressions
covering the exact threshold in teacher and student commands. This is an
execution-only correction and does not change matcher, repair, loss, model, or
selection semantics. The replacement pilot remains immutable and cannot be
continued in place under the corrected source snapshot. A version-1 prefix
import now provides the narrow recovery path: it proves Git/AST compatibility,
hard-links and re-attests only the completed prefix through `training_lock`,
rebuilds every campaign-bound lock under a fresh campaign spec, binds the
import report into the new lock chain, and submits from `teachers` onward.
Failed teacher/descendant artifacts are categorically excluded.

Still required externally before production: commit and push the exact clean
source; authenticate the native 53-file dataset and split; run installed-Weaver
FP32 parity; complete the genuine streamed Tigris smoke graph; capture measured
RAM/GPU/I/O/wall/storage evidence; and inspect the full production dry run.
The smoke DAG ends at `miniature_summary` and structurally contains no
`final_test` task. No PMARD scientific result, real matcher lock, checkpoint,
resource measurement, or final-test output is claimed in this handoff.

Track-only and p4-plus-track repair remain deliberately unavailable until the
real Stage-A units/definition audit creates a compatibility lock; attempting
either fails closed. The old complete full-particle implementation remains
locally verified but is not the selected scientific endpoint. The selected
fitted-strict path is intentionally partial and uses the new selective
all-field contract; it never claims that the unmatched fraction became offline.

## Repository state

Transfer Blocks 1--4, 6, and 7 are complete locally. Transfer Block 5 is
code-complete and its authoritative installed-Weaver FP32 equivalence check
passed on Tigris. A complete real PRAD miniature is still required.

Implemented:

- frozen JetClass schema, canonical identities, bounded ROOT reads, and
  deterministic balanced splits;
- registered `fixed_hlt_v3_track_dominant_proxy/v1` degradation and replica
  contracts with donor-v1 deterministic parity;
- immutable authenticated offline and HLT cache shards;
- atomic no-overwrite publication and exact completed-shard reuse;
- streaming cache range/batch readers that do not materialize the campaign;
- source, split, schema, profile, replica, generator, offline-parent, and
  source-array lineage;
- aggregate HLT diagnostics with construction indices excluded from deployable
  artifacts;
- exact label, identity, role, validity-state, padding, and parent validation;
- canonical 17-feature Particle Transformer input construction;
- frozen from-scratch standard-four Weaver baseline adapter and standalone
  FP32 logits/training-gradient/mask/state-dictionary attestation, including
  exact topology comparison for Weaver's nonfinite auxiliary Lorentz-input
  derivative.
- deterministic shard-local epoch sampling and fixed-update AdamW training;
- exact model, optimizer, schedule, scaler, sampler, replica-cycle, RNG, and
  history checkpoint resume;
- deterministic validation checkpoint selection without early termination;
- ordered label-free prediction shards and authenticated label joins;
- frozen accuracy, CE, efficiency, AUC, Brier, 15-bin top-label ECE,
  QCD-rejection, and paired-bootstrap definitions.
- clean-source snapshots, immutable campaign specifications, storage gates,
  task attestations, per-job durable submission journals, exact-ID Slurm
  ledgers, monitoring, and resume plans;
- thin absolute-path Tigris workers and two-lock final-test authorization;
- a smoke-only deliberate update-one training interruption followed by
  dependent exact-checkpoint resume through the ordinary training worker.
- the active PRAD implementation: proportional 500k/150k/500k split contract,
  paired-view audit/cache, replica-aligned charged/neutral Hungarian matching,
  exclusive C/A K=2/3/4 targets, and train-only moments/positive weights;
- a parity-oriented offline-teacher/HLT-student Weaver extension with symmetric
  contextual relations, bounded centered head bias, zero-initialized gates,
  HLT-only deployment signature, and explicit nondeployable oracle path;
- optional validated float16 dense teacher relation/bias caching and a
  mandatory installed-Weaver CUDA PRAD mechanics attestation before E0;
- configuration-driven E0--E10 and all registered V1--V10 switches, fixed
  staged budgets, frozen teacher, exact resume, deterministic identity-bound
  in-batch shuffled control, and five confirmation seeds;
- PRAD validation selection, bounded relation-fidelity diagnostics, label-free
  inference, validation benchmarks, pre-test five-seed final selection,
  restart-safe finalist/execution claims, paired bootstrap reporting for every
  final five-seed graph, and complete required plot assembly;
- a source-bound PRAD Tigris DAG using `reu-aisocial`/`tigris`, absolute worker
  paths, exact numeric dependencies, durable partial-submit recovery, exact-ID
  monitoring/resume/cancellation, and production requests bound to completed
  smoke `sacct` evidence plus conservative source-bound storage headroom.
- the versioned `prad_minimum_durable_storage_v1` profile: the 15-node DAG
  rebuilds paired views, structural targets, and teacher outputs in Slurm
  job-local temporary storage and deletes them on worker exit; completed runs
  retain optimizer-free best/final checkpoints, incomplete runs retain only
  one rolling exact-resume checkpoint, and prediction shards store logits with
  split-row-bound implicit identities.

Still not implemented or accepted:

- a completed real Tigris minimum-storage miniature.
  The first authentic PRAD smoke submission, source commit
  `8bb20be8cb351f0a0fd71f50dabcc03467790161` and jobs `41126`--`41147`,
  failed closed before data/model execution: sorted JSON role keys exposed an
  order-sensitive split-builder check, and the independent Weaver worker read
  the absent split manifest before dispatch. Both defects now have local
  regressions and fixes. A direct installed-Weaver diagnostic then established
  that every auxiliary Lorentz-input derivative is NaN for all-valid, singly
  padded, and multiply padded fixtures alike, while logits, feature gradients,
  and parameter gradients remain finite. The v3 parity contract compares that
  diagnostic topology exactly and keeps all training-required quantities
  strictly finite. The v3 authoritative rerun passed on Tigris at source
  commit `2b3e34c961067f1d458561d67b51b2ffd780704a`. A disk-backed smoke from
  that source was submitted, but complete monitor/resource evidence has not
  been supplied and it cannot validate the new storage profile;
- PRAD's real `reports/data_audit.md`, exact full split and ephemeral-data
  attestations,
  trained checkpoints, experiment results, sealed 500k test metrics, plots,
  and final scientific recommendation. Those are intentionally not fabricated
  locally and require the clean committed source plus the real miniature first;
- all twelve HOSD implementation steps. The migrated plan is authoritative,
  but `src/hlt_classification/hosd/`, its target extractors/caches/teachers,
  taps/heads/probes, auxiliary and feedback training, combinations, metrics,
  confirmation/scale, final selection/seal, and HOSD production DAG do not
  yet exist.

No file in this repository imports `Fresh_check` at runtime. The repository
has an independent Git history. No remote is configured; direct Tigris access
from the implementation session reached the cluster but was rejected at the
required interactive Duo MFA step.

The exact PRAD execution sequence, including clean-source transfer, smoke dry
run, monitored miniature, reviewed resource requests, and separately reviewed
production dry run, is in `docs/PRAD_RUNBOOK.md`.

## Contracts added

```text
scientific HLT profile:
fixed_hlt_v3_track_dominant_proxy/v1

serialized HLT profile:
hlt_classification_hlt_v3_profile_v1

replica policy:
hlt_classification_hlt_replica_manifest_v1

cache shard:
hlt_classification_cache_shard_v1

offline cache manifest:
hlt_classification_offline_cache_manifest_v1

HLT cache manifest:
hlt_classification_hlt_cache_manifest_v1

canonical ParT inputs:
hlt_classification_part_inputs_v1

Weaver adapter/parity report:
hlt_classification_weaver_part_v1
hlt_classification_weaver_part_v2
hlt_classification_weaver_part_v3

training checkpoint:
hlt_classification_training_checkpoint_v2

training report:
hlt_classification_part_training_report_v2

prediction manifest:
hlt_classification_prediction_manifest_v1

evaluation report:
hlt_classification_evaluation_report_v1

metrics:
hlt_classification_metrics_v1

source snapshot and compact campaign:
hlt_classification_source_snapshot_v1
hlt_classification_baseline_campaign_spec_v1

execution evidence:
hlt_classification_storage_measurement_v1
hlt_classification_slurm_resource_evidence_v1
hlt_classification_runtime_environment_v1
hlt_classification_task_attestation_v1
hlt_classification_submission_job_record_v1
hlt_classification_submission_ledger_v1
hlt_classification_monitor_report_v1
hlt_classification_resume_plan_v1

smoke interruption evidence:
hlt_classification_training_interruption_evidence_v1

final-test locks:
hlt_classification_finalist_lock_v1
hlt_classification_final_test_execution_lock_v1
hlt_classification_final_test_execution_claim_v1

PRAD campaign surfaces:
hlt_classification_prad_split_manifest_v1
hlt_classification_prad_paired_view_v1
hlt_classification_prad_structural_targets_v1
hlt_classification_prad_teacher_outputs_v1
hlt_classification_prad_relation_v1
hlt_classification_prad_checkpoint_v1
hlt_classification_prad_model_checkpoint_v1
hlt_classification_prad_ephemeral_dataset_v1
hlt_classification_prad_training_report_v1
hlt_classification_prad_training_v3
hlt_classification_prad_prediction_manifest_v2
hlt_classification_prad_teacher_prediction_manifest_v1
hlt_classification_prad_evaluation_report_v1
hlt_classification_prad_campaign_spec_v2
hlt_classification_prad_submission_ledger_v1
hlt_classification_prad_task_attestation_v1
hlt_classification_prad_resource_evidence_v1
hlt_classification_prad_final_selection_v1
hlt_classification_prad_runtime_validation_v3
hlt_classification_prad_statistical_report_v2

PMARD/Scouting reviewed surfaces:
hlt_classification_scouting_schema_v4
hlt_classification_fitted_strict_matcher_v1
hlt_classification_scouting_feature_audit_v2
hlt_classification_pmard_row_selection_v1
hlt_classification_pmard_selective_assignment_shard_v1
hlt_classification_pmard_selective_assignment_manifest_v1
hlt_classification_pmard_ephemeral_assignment_v2
hlt_classification_pmard_matcher_report_v2
hlt_classification_pmard_matcher_validation_v2
hlt_classification_pmard_full_role_coverage_v2
hlt_classification_pmard_training_report_v4
hlt_classification_pmard_resume_checkpoint_v4
hlt_classification_pmard_lock_v4
hlt_classification_pmard_campaign_spec_v9
```

Changing registered-v1 event, substream, replica-cycle, degradation, or cache
semantics requires a new applicable scientific or serialized contract version.

## Readiness matrix

| Capability | Status | Evidence |
|---|---|---|
| Package and documentation | Passed locally | `tests/test_scaffold.py` |
| JetClass data foundation | Passed locally | `tests/test_data_foundation.py` |
| HLT-v3 mechanism contracts | Passed locally | `tests/test_hlt_v3.py` |
| Donor-v1 deterministic parity | Passed locally | frozen array/diagnostic hashes |
| Authenticated cache construction/resume | Passed locally | `tests/test_cache_pipeline.py` |
| Cache corruption/cross-profile rejection | Passed locally | `tests/test_cache_pipeline.py` |
| HLT shard/batch layout invariance | Passed locally | bitwise concatenated arrays |
| Canonical ParT transforms | Passed locally | `tests/test_part_inputs.py` |
| Wrapper/parity logic | Passed with interface-faithful test double | `tests/test_particle_transformer.py` |
| Installed-Weaver FP32 parity | Passed on Tigris | v3 report at `2b3e34c` matched logits, masks, model state, training-required gradients, and auxiliary Lorentz-gradient topology |
| Real Tigris minimum-storage miniature | Attempt 6 completed through E4 | core-screen array `42278` reached Stage B in E5--E10 and exposed an optimized-SDPA mixed-freeze backward incompatibility |
| Baseline training/resume | Passed locally | `tests/test_training_engine.py` |
| Inference and metrics | Passed locally | `tests/test_inference.py`, `tests/test_metrics.py` |
| Production DAG dry run | Passed locally | `tests/test_campaign.py` |
| Source-drift/all-reused validation | Passed locally | `tests/test_campaign.py` |
| Failure injection and exact-ID recovery | Passed locally | `tests/test_campaign.py` |
| Worker static contracts | Passed locally | `tests/test_campaign.py` |
| Full production authorization | Not authorized | Transfer Block 8 |

## Donor provenance

Inspected donor commit:

```text
bfcda5bd6fd48037ab198bcca1eadf6f4d87e131
```

Block-4 cache references were inspected, not copied unchanged:

```text
jetclass_fresh/hlt_cache.py
5631ac115a4f352739492e0db06c5e9c3111fa52d522d273243504e0731823c4

teacher_logit_reco/relation_expert_token_bridge/hlt_cache.py
57d0dde4e4aad5aa622fc62c2020954ae69a0108cf89178d7304dabd2f6345e4
```

Exact migration details for all blocks are in `docs/LEGACY_SOURCE_MAP.md`.

## Verification evidence

Validated with bytecode and pytest caches disabled:

```text
python:
C:\Users\22rya\miniconda3\envs\tagging-hlt\python.exe

focused Block 4:
8 passed

focused Block 7:
8 passed

complete discovered suite:
205 passed, 14 warnings

focused PRAD suite:
60 passed

focused PMARD/Scouting suite:
54 passed
```

The Block-4 suite proves byte-identical clean versus interrupted/resumed
offline and HLT artifact trees, fail-closed corruption and lineage drift,
cross-profile rejection, exact parent-role validation, deployable-artifact
hygiene, and bitwise HLT array equivalence across different source shard,
output shard, and processing-batch layouts.

No real Tigris data, GPU, or installed Weaver path was exercised in this
block. The local environment has PyTorch 2.5.1 but no Weaver installation.

The current PMARD/Scouting focused evidence is 54 passing tests. It includes exact
uninterrupted-versus-resumed state equivalence, a deliberately poor model that
still completes, nonfinite failure injection, bitwise batch-invariant
prediction shards, all frozen metric edge cases, full-endpoint authorization
rejection with full-role count-only lineage, recursive FP32 representation KD
under BF16 inputs, durable interval-mean history, batch-shell signal delivery,
bounded cross-file reads, sparse validation/checkpoint cadence, and an
end-to-end zero-alpha selection through representation KD, anchored companion
training, and the next dual-teacher generation student.
The fitted-strict additions authenticate the canonical model bundle, reproduce
reference scores/margins/confidences numerically, exercise both rectangular
assignment directions, private dummies, post-assignment rejection, strict
PID/charge gates, wrapped phi, invalid PID, native lost-track boundaries,
one-to-one integrity, unmatched p4 preservation, artifact tampering, and the
online PMARD-stream dispatch.

The campaign portion includes source drift after an
all-reused graph, cross-campaign lineage rejection, storage insufficiency,
exact dependency IDs, task-byte corruption, dry-run nonmutation,
failed-descendant recovery, stale exact-job cancellation lineage, and
final-test lock authorization. Python compilation, all 21 PRAD script `--help`
paths, all 11 shell syntax checks, `git diff --check`, and the whitespace audit
pass.

The PRAD-focused suite covers exact split constants and proportional/disjoint
construction, Hungarian thresholds and pair masks, C/A targets, symmetric
relations, masked centering/bounds, exact zero-gate parity, class-gradient
flow, replica-aligned target selection, frozen teacher behavior, HLT-only
inference, test locks, exact checkpoint state, frozen primary metrics,
train-only statistics, source-bound campaign guards, numeric dependencies,
partial-submit reuse, smoke-bound production resources, dense teacher-output
shape/storage validation, accumulated training time, semantic-only causal
isolation, teacher semantic validation, exact final-claim recovery, exact
array-element attestations, and synthetic installed-interface runtime
attestation. Minimum-storage coverage additionally proves in-memory
reconstruction does not publish arrays, task-local cache paths cannot fall
inside the campaign/project tree, completed training removes full optimizer
state while retaining loadable best/final model-only checkpoints, and compact
predictions bind identities through authenticated split row ranges. The two
Tigris-derived regressions prove split construction is
invariant to canonical JSON key sorting and independent Weaver dispatch does
not touch a split artifact. A direct installed-Weaver run at corrected commit
`22c15fc2051f103e50361beb2759731cb956dc8a` then reported zero maximum error
for logits and feature/parameter gradients but a nonfinite Lorentz-input
gradient difference. This is consistent with the fixture's multiple identical
padded zero four-vectors. The v3 contract instead authenticates exact nonfinite
topology for that unused derivative while requiring every training surface
finite; its Tigris run at `2b3e34c` passed with exit code zero.

The first minimum-storage miniature at `288aafc` authenticated split, audit,
installed-Weaver parity, CUDA PRAD runtime, and task-local train statistics in
jobs `41820`--`41824`. E0 job `41825` completed training and published compact
model checkpoints, then failed closed when evaluation restored Weaver's
registered tensor `_counter` buffer by assigning a Python integer. The restore
helper now mutates tensor counters in place while preserving legacy integer
counters, with a regression that proves registered-buffer identity and value.

The replacement miniature at `6bcda2d` then completed split, audit, parity,
runtime, statistics, and the entire E0 train/evaluate task in jobs
`41967`--`41972`. E1 job `41973` failed before training because its CLI passed
the intended `model_factory` keyword to an engine signature that still named
the parameter `model`, even though the engine body already invoked
`model_factory()`. The signature is aligned, the CLI now binds its complete
keyword set against the real engine signature in a regression, and a tiny
full-budget teacher test exercises factory construction through compact
best/final checkpoint publication and rolling-checkpoint removal.

The `58ee660b` miniature passed the repaired teacher path, oracle, and E3/E4.
Core-screen array `42015` then failed E5--E10 during their relation-only Stage
A with PyTorch `RuntimeError: LSE is not correctly aligned (strideH)`. Those
are exactly the pair-supervised graphs; E3 begins directly in all-trainable
Stage C and completed. The Stage-A loss had anchored zero-valued hard/KD
placeholders to final logits, unnecessarily asking installed CUDA attention to
backpropagate a zero gradient through frozen later blocks while only its
additive bias path remained differentiable. The zero anchor now uses the
pre-attention relation tensor. A regression requires final logits to receive
no Stage-A gradient, and runtime-validation v2 reproduces the registered
freeze schedule on installed CUDA before E0.

The next `d2237422` miniature completed split, installed-Weaver parity, data
audit, and train statistics. Its runtime-validation subprocess returned exit
code zero in job `42084`, which establishes that the new Stage-A CUDA
backward attestation passed. The task wrapper then rejected that valid report:
it requested the v2 contract name while the reusable content-hash validator
still defaulted to schema version 1. The validator now accepts an explicit
expected schema version while preserving version 1 as its default, the PRAD
runtime worker explicitly requests version 2, and a regression exercises the
real dispatch path with a content-authenticated v2 report.

The `f90cf5fe` miniature then completed its runtime worker, E0 baseline, E1
teacher, E2 oracle, and E3/E4. Core-screen E5--E10 all failed after the same
roughly four-minute interval with
`RuntimeError: LSE is not correctly aligned (strideH)`. Their shared timing
and exclusive pair-supervised stage schedule identify the transition from the
repaired relation-only Stage A to partially unfrozen Stage B as the failure
surface.
Stage B intentionally freezes early ParT blocks while training later blocks
and the differentiable relation bias; this differs from both Stage A and E3's
fully trainable Stage C. Training contract v3 records
`math_for_stage_b_mixed_freeze_v1`: CUDA Stage B selects PyTorch's reliable
math SDPA backend, while Stage A, Stage C, CPU, and inference retain automatic
selection. Runtime-validation v3 now performs a full mixed-freeze Stage-B
forward/backward with the production loss and requires finite nonzero
gradients before E0.

## Historical PRAD next task (superseded as the active conversation task)

Commit the Stage-B backend policy and runtime-validation v3, cancel only the
pending descendants of `smoke_min_f90cf5fe` through its v2 ledger, transfer
the exact fix commit to Tigris, and execute a fresh PRAD real miniature exactly
as `docs/PRAD_RUNBOOK.md` specifies. Runtime-validation v3 must authenticate
the mixed-freeze CUDA backward before E0; the miniature must still complete
training, selection, locked-test, and aggregation.

```bash
export PYTHONNOUSERSITE=1
python -s scripts/validate_weaver_parity.py --device cpu
```

Do not mark PRAD production readiness accepted until every new smoke task
attestation authenticates and measured resource/storage evidence is captured.
A full PRAD campaign remains unauthorized in this handoff.

## Exact active next task: corrected PMARD pilot

Commit and push the prefix-import recovery utility, update the clean Tigris
checkout to that exact commit, create a fresh pilot spec, import the compatible
completed prefix from `pmard_pilot_3f379cb66398fcc8`, inspect the recovery
dry run, and submit from `teachers` onward. Do not reuse failed teacher or
descendant outputs. Cancel only the old campaign's exact dependency-doomed
jobs through its own submission ledger.
