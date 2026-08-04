# Current Handoff

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
hlt_classification_prad_training_v2
hlt_classification_prad_prediction_manifest_v2
hlt_classification_prad_teacher_prediction_manifest_v1
hlt_classification_prad_evaluation_report_v1
hlt_classification_prad_campaign_spec_v2
hlt_classification_prad_submission_ledger_v1
hlt_classification_prad_task_attestation_v1
hlt_classification_prad_resource_evidence_v1
hlt_classification_prad_final_selection_v1
hlt_classification_prad_runtime_validation_v1
hlt_classification_prad_statistical_report_v2
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
| Real Tigris minimum-storage miniature | Pending | older disk-backed attempts do not validate the v2 DAG or ephemeral cleanup |
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
144 passed

focused PRAD suite:
53 passed
```

The Block-4 suite proves byte-identical clean versus interrupted/resumed
offline and HLT artifact trees, fail-closed corruption and lineage drift,
cross-profile rejection, exact parent-role validation, deployable-artifact
hygiene, and bitwise HLT array equivalence across different source shard,
output shard, and processing-batch layouts.

No real Tigris data, GPU, or installed Weaver path was exercised in this
block. The local environment has PyTorch 2.5.1 but no Weaver installation.

The current training/inference/orchestration focused evidence is 24 passing
tests. It includes exact
uninterrupted-versus-resumed state equivalence, a deliberately poor model that
still completes, nonfinite failure injection, bitwise batch-invariant
prediction shards, and all frozen metric edge cases.

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

## Active and next task

The local full-suite audit is complete. Commit the minimum-storage source and
transfer that exact commit to Tigris only after the older `2b3e34c` smoke has
finished or been cancelled by its exact campaign job IDs. Then execute a new
PRAD real miniature exactly as `docs/PRAD_RUNBOOK.md` specifies. It must prove
the 15-node DAG, job-local cleanup, compact checkpoint loading, and measured
durable size.

```bash
export PYTHONNOUSERSITE=1
python -s scripts/validate_weaver_parity.py --device cpu
```

Do not mark PRAD production readiness accepted until every new smoke task
attestation authenticates and measured resource/storage evidence is captured.
A full PRAD campaign remains unauthorized in this handoff.
