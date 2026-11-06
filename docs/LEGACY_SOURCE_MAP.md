# Legacy Donor-Source Map

## Seven-candidate Group B structured targets (2026-11-05)

Internal donor baseline: `342e5d3affba78f9d6add4cbb95d6264fc4b516b`.
Group B extends `luka_next_supervision_hydra`; it does not clone a second
training/submission stack or edit any old pinned execution worktree. No
external source implementation or `Fresh_check` runtime import was introduced.
sklearn MiniBatchKMeans is imported from the existing pinned environment,
not copied. Native/Weaver and original U100 modules remain unchanged.

| Exact donor files at342e5d3a | Group B use |
|---|---|
| `src/hlt_classification/luka_next_supervision_hydra/contracts.py`, `campaign.py`, `reuse.py` | Retain original A recipe/flags and exact source/reference checks; add separate structured recipe/spec/scope and old-A input validation in its own process. |
| `src/hlt_classification/luka_next_supervision_hydra/data.py`, `inputs.py` | Reuse physical TRAIN transfer,17-input HLT views and original U100 bank; add ordered streaming shards, Q geometry and V compact-bank capability. |
| `src/hlt_classification/luka_next_supervision_hydra/parallel.py` | Reuse CPU-only pre-fork transport/PSS accounting; add bounded full-TRAIN question-statistic reduction. |
| `src/hlt_classification/luka_next_supervision_hydra/model.py`, `training.py` | Extend the native128D jet tap and isolated auxiliary RNG with removable Q/V heads and declared BCE/frequency losses; retain CE+U100-w5 and native-only checkpoint replay. |
| `src/hlt_classification/luka_next_supervision_hydra/resources.py`, `worker.py` | Reuse baseline parity, genuine warmed full/partial measurement and immutable receipts; add structured preparation and six-path resource coverage. |
| `src/hlt_classification/luka_next_supervision_hydra/submission.py`, `reporting.py` | Reuse independent exact-ID journals, claims/idempotence, saved results/progress and same-seed deltas; add B-only scope/token/job prefix and realized target review. |
| `scripts/luka_next_supervision_hydra.py`, `scripts/queue_luka_next_supervision_hydra.sh`, `sbatch/run_luka_next_supervision_hydra.sh` | Extend the same CLI/queue interface with create-b and scope-specific authorization; existing absolute-project Slurm worker is reused unchanged. |
| `tests/test_luka_next_supervision_hydra.py`, `tests/test_luka_attention.py` | Reuse compact/native fixture, synthetic caches, old-seed recipes and transport fixtures; add separate Q/V scientific, corruption, worker and submission tests. |

New modules are `structured_contracts.py`, `questions.py`, `vocabulary.py`,
`structured_model.py`, `structured_targets.py` and `group_b.py` in that same
namespace. New STRUCTURED_RECIPE/SPEC/TARGETS, QUESTION_GEOMETRY/CATALOG,
DICTIONARY and FREQUENCY_BANK envelopes are v1. Shared resource/acceptance
envelopes remain v2. The A scientific recipe remains byte-equivalent to342e5d3a
(`d39c6ee09d803622385834756bb57c131ed8a8142659b1d99aec248ec6b6bddc`).

## Group A execution v2 resource repair (2026-11-05)

Internal implementation baseline: `0a6721edf146e2ab3166cdee976377c22240053a`.
Refined existing `luka_next_supervision_hydra/sampling.py`, `data.py`,
`inputs.py`, `training.py`, `resources.py`, `campaign.py` and `contracts.py`,
plus `scripts/queue_luka_next_supervision_hydra.sh` and its existing focused
test suite. No additional external donor or other campaign runtime is changed.
SPEC/RESOURCE_CHECK/ACCEPTANCE are execution v2; raw/scientific recipe v1 stays
compatible. Golden particle/group pools were captured from that exact v1
commit before editing and replay byte-for-byte. The sparse-parent draw is an
equivalent implementation of the original ordered eligible-parent population,
not a new sampling law. Source lineage and failed setup150801 are documented in
the [runbook](LUKA_NEXT_SUPERVISION_GROUP_A_RUNBOOK.md).

## Seven-candidate Group A foundation (2026-11-05)

Additive baseline and all internal donors in the table:
`ddcdd5fa22ce2455b12a62d61a926caa07969159`. New namespace:
`luka_next_supervision_hydra`. No donor runtime was edited, no external source
was copied, and no `Fresh_check` runtime import was introduced. Paths below are
under `src/hlt_classification/` unless prefixed otherwise. New P/G/J objectives,
raw-transfer semantics and independent partial-release policy are v1; old
contracts remain unchanged. See the [contract](contracts/LUKA_NEXT_SUPERVISION_HYDRA.md).

| Exact donor files at ddcdd5fa | Use |
|---|---|
| `luka_embedding_aux/model.py` | Import AuxiliaryParT, native pre-classifier tap and original isolated w5 predictor. |
| `jetclass2_delphes/model.py`, `jetclass2_delphes/runner.py`, `jetclass2_delphes/batch128_30e_recipe.py`, `jetclass2_delphes/reporting.py`, `jetclass2_delphes/execution.py` | Unchanged native model/environment, selection, LR, saved-probability metrics and GPU identity. |
| `luka_u100_embedding/contracts.py` | Original U100 recipe/counts; no replacement target or coefficient. |
| `luka_u100_contrastive/model.py`, `luka_u100_contrastive/sampling.py`, `luka_u100_contrastive/targets.py`, `luka_u100_contrastive/training.py`, `luka_u100_contrastive/contracts.py` | Exact J5 head, original mix candidate streams/binning, frozen-key transforms/summaries, peak reporting and original result validation. |
| `luka_u100_contrastive_hydra/inputs.py`, `luka_u100_contrastive_hydra/transfer.py`, `luka_u100_contrastive_hydra/campaign.py`, `luka_u100_contrastive_hydra/worker.py` | Original OFFLINE key relocation/bindings and authenticated result-only J5 reuse; optional local original-worktree J5 replay. |
| `luka_u100_pair_hydra/inputs.py`, `luka_u100_pair_hydra/site.py`, `luka_u100_pair_hydra/submission.py`, `luka_u100_pair_hydra/portability.py` | Existing allowlisted Bundle/views/U100 bank, pinned environment, exact accounting and tolerant metric replay for old transferred results. |
| `luka_u100_pair_transfer/bundle.py` | Original-source subprocess authentication for the additive SPORC raw export. |
| `luka_fullsim/alljets_data.py`, `luka_fullsim/particles.py`, `cms2jc2_response/bridge.py` | ALLJETS TRAIN membership, selected raw reader, explicit physical conventions/validity and canonical physical particles. |
| `luka_fullsim/foundation.py`, `luka_fullsim/regular_submission.py`, `luka_fullsim/preflight_v2.py` | Clean pushed source, sanitized submission environment and required production backend. |
| `luka_cross_message_ce_hydra/campaign.py` | Authenticate the original completed A6000 parent in its own pinned worktree. |
| `luka_cross_message_hydra/training.py`, `luka_cross_message_hydra/resources.py`, `luka_cross_message_hydra/execution.py`, `luka_cross_message_hydra/inputs.py` | Native BF16 forward/AdamW/predict, miniature Subset, qualified A6000 allocation/site review and original transfer-proof formula. |
| `luka_particle_teacher_hydra/resources.py`, `luka_particle_teacher_hydra/training.py`, `luka_particle_teacher_hydra/submission.py`, `luka_particle_teacher_hydra/worker.py`, `luka_particle_teacher_hydra/campaign.py`, `luka_particle_teacher_hydra/reporting.py` | Adapt warmed full/partial measurement, fit/export/receipt, immutable spec and reporting patterns. Replace teacher-token supervision and dependent release/report with independent P/G/J fits and on-demand reports. Import only the reusable miniature-row helper; local C0/C1 reuse invokes original validators in their original worktree. |
| `data/cache_contracts.py`, `cms_proxy_ladder/contracts.py`, `cms_proxy_ladder/submission.py`, `context_fusion/worker.py`, `provenance.py` | Atomic/hash-bound publication, file references, walltime formatting, native state saving and old-source snapshot validation. |
| `scouting/hcwdl_exact_dag_submission.py`, `scouting/hcwdl_recovery.py` | Exact-ID dry/live command journals and durable ledgers; all new science dependency lists are empty. |
| `tests/test_luka_attention.py`, `tests/test_luka_particle_teacher_hydra.py`, `tests/test_luka_u100_contrastive_hydra.py` | Reuse synthetic cache/seed fixtures and adapt native/worker/submission failure-injection patterns. |
| `scripts/luka_particle_teacher_hydra.py`, `scripts/queue_luka_particle_teacher_hydra.sh`, `sbatch/run_luka_particle_teacher_hydra.sh` | Adapt thin CLI, exact review and absolute-project environment conventions into new entrypoints. |

The installed local Weaver fixture is used as a dependency for tests; no Weaver
implementation is copied into the new package. Real source/population/resource
qualification remains a production Hydra setup step.

## HLT-aligned teacher particle prediction (2026-11-05)

Additive source baseline: `cc55765a5e6a80ed04b660d99dc182a4aaf4b017`.
Original execution/teacher parent: `199928da991f8a4194dcf42df009130cfcf84118`.
New namespace: `luka_particle_teacher_hydra`. No donor runtime is edited and no
`Fresh_check` runtime import is introduced. These are internal adaptations and
imports; no external donor source was added.

| Donor path at cc55765a (under src/hlt_classification unless shown) | Use |
|---|---|
| `luka_shared_message_hydra/{campaign,resources,submission,worker,reporting,training}.py` | Adapt immutable source/parent proofs, warmed miniatures, exact-DAG release/recovery, worker/publication and reporting scaffolds into the new namespace. Replace the joint-message objective with native per-particle prediction. |
| `luka_shared_message_hydra/inputs.py` | Import TRAIN/validation HLT loading, U100 jet-target bank and authenticated frozen m16 teacher. |
| `luka_embedding_aux/model.py` | Subclass native AuxiliaryParT with removable final-token heads; retain its original jet-embedding predictor. |
| `luka_particle_prediction/model.py` | Import `token_forward` only for the frozen native U100 encoder. Teacher targets use the separate privileged oracle forward. |
| `luka_cross_message_hydra/{inputs,model,training,resources,execution}.py` | Import encoder/delta/alignment, authentic teacher model, AdamW/native forward/predict, miniature subset and A6000 allocation semantics. |
| `luka_cross_message_ce_hydra/{campaign,worker}.py` | Import original-worktree parent authentication and largest/full/partial miniature sampling. |
| `jetclass2_delphes/{model,runner,reporting,batch128_30e_recipe,execution}.py` | Native factory/environment, selection, metrics, recipe/schedule and GPU identity. |
| `luka_u100_pair_hydra/{inputs,site}.py`; `luka_u100_embedding/contracts.py` | Bundle/environment authentication and registered population counts. |
| `luka_fullsim/{foundation,regular_submission,preflight_v2,alljets_data}.py` | Source validation, clean subprocess environment, required CUDA setting and scope flags. |
| `scouting/{hcwdl_exact_dag_submission,hcwdl_recovery}.py` | Exact-ID submission journals and ledger assembly. |
| `data/cache_contracts.py`; `cms_proxy_ladder/{contracts,submission}.py`; `context_fusion/worker.py` | Hashes, immutable publication, file references, walltime and checkpoint publication. |
| `luka_u100_contrastive/training.py`; `luka_embedding_weight/worker.py` | Memory peaks and saved-probability validation. |
| `scripts/luka_shared_message_hydra.py`; `scripts/queue_luka_shared_message_hydra.sh`; `sbatch/run_luka_shared_message_hydra.sh` | Adapt thin CLI and absolute-path/environment launch wrappers. |
| `tests/test_luka_attention.py`; `tests/test_luka_shared_message_hydra.py` | Reuse synthetic cache/seed fixtures and adapt submission/CPU-guard regression cases. |

The new CPU worker includes the same separately validated rounding correction
as the existing local shared-message fix: Hydra job146337 requested one CPU and
received two with unchanged4000MiB. Its controller/request checks are copied
into the new namespace, not imported from the dirty donor worker. Contracts
are `LUKA_PARTICLE_TEACHER_HYDRA_*/v1`. Original teacher validation runs in its
own pinned199928da process; prior artifacts keep their original contracts.

## Shared-message CPU allocation fix (2026-11-05)

Modified `luka_shared_message_hydra/worker.py` and its focused tests from commit
`cc55765a5e6a80ed04b660d99dc182a4aaf4b017`. No donor files were copied and no prior
campaign runtime was edited. User-supplied Hydra job146337 accounting supplies
the regression case: one requested CPU, two allocated CPUs, unchanged4000MiB.
Scientific schemas and objectives are unchanged; pinned source identifies the
fixed execution guard. Existing-run recovery uses its original `finish` command.

## Joint shared-message Hydra campaign (2026-11-04)

Additive baseline: `546077aa7fce428bf3f9f136ab2277ac5eed5f85`. Original execution
parent remains `199928da991f8a4194dcf42df009130cfcf84118`. No donor runtime is edited,
and no `Fresh_check` runtime import or donor file copy is introduced.

| Internal donor at baseline546077aa | Use in `luka_shared_message_hydra` |
|---|---|
| `luka_cross_message_hydra/model.py` | Import frozen-token `CrossRead`; adapt block4 injection and deployment into shared-prefix/two-route model; replace HLT MLP with contextual attention using native pair bias. |
| `luka_cross_message_hydra/training.py` | Reuse AdamW, HLT prediction and frozen oracle forward; adapt full fit/publication logic for the new joint objective and HLT-only selection. |
| `luka_cross_message_hydra/inputs.py` | Reuse authenticated U100 encoder/transfer and allocation-local ragged token cache; restrict new campaign extraction to TRAIN. |
| `luka_cross_message_hydra/resources.py` | Reuse `Subset`; adapt per-arm production miniatures and measurements. |
| `luka_cross_message_hydra/execution.py` | Reuse exact A6000 site and allocation validator against original parent. |
| `luka_cross_message_ce_hydra/campaign.py` | Reuse isolated original-worktree parent authentication; adapt additive source/spec/root/storage registration. |
| `luka_cross_message_ce_hydra/worker.py` | Reuse full/partial `miniature_rows`; adapt disposable warmup and fresh measurement workflow. |
| `luka_cross_message_ce_hydra/{submission,reporting}.py` | Adapt exact-plan claims/idempotence and saved epoch/recovery reporting; introduce durable Slurm CPU release. |
| `luka_embedding_aux/model.py` | Reuse HLT jet embedding predictor and installed-Weaver head capture. |
| `jetclass2_delphes/{model,runner,reporting,batch128_30e_recipe,execution}.py` | Reuse authoritative native factory/environment, selection, metrics, schedule and GPU identity. |
| `luka_u100_pair_hydra/{inputs,site}.py` | Reuse verified bundle and environment validation. |
| `luka_fullsim/{foundation,regular_submission,preflight_v2,alljets_data}.py` | Reuse source pin, clean subprocess environment, backend validation and scope flags. |
| `scouting/{hcwdl_exact_dag_submission,hcwdl_recovery}.py` | Reuse exact-ID DAG journal, submission and ledger authentication. |
| `data/cache_contracts.py`, `cms_proxy_ladder/{contracts,submission}.py`, `context_fusion/worker.py` | Reuse atomic/hash artifacts, file references, walltime rendering and checkpoint publication. |
| `luka_u100_contrastive/training.py`, `luka_embedding_weight/worker.py`, `luka_u100_embedding/contracts.py` | Reuse measured memory, prediction validation and registered population counts. |
| `scripts/luka_cross_message_ce_hydra.py`, `scripts/queue_luka_cross_message_ce_hydra.sh`, `sbatch/run_luka_cross_message_ce_hydra.sh` | Adapt thin CLI and absolute-path/environment wrappers into new namespace. |
| `tests/test_luka_attention.py`, cross-message/CE focused tests | Reuse synthetic authenticated cache/seed fixtures and adapt scientific/queue validation cases. |

Original selected teacher authentication executes its own pinned199928da source
in a subprocess. Installed Weaver block, pair embedding and aggregator methods
are called directly; their source hashes remain bound through the installed
environment contract. The new contract family is `LUKA_SHARED_MESSAGE_HYDRA_*/v1`.

## CE-only cold-start timing correction (2026-11-04)

Internal source: `5a3509eb90ff4c561afaf5d0ed73fdba2f05eb58`.
Adapted `luka_cross_message_ce_hydra/{contracts,campaign,worker}.py` and the
focused tests to register execution v2 and authenticate a disposable warmup
before the measured production miniature. No new donor file was copied.
Scientific model/training/recipe and original cross-message donor runtime at
`199928da991f8a4194dcf42df009130cfcf84118` are unchanged.

## Cross-message CE-only architecture controls (2026-11-04)

Internal donor: `199928da991f8a4194dcf42df009130cfcf84118`.
New namespace: `src/hlt_classification/luka_cross_message_ce_hydra/`.
Existing donor runtime is unchanged; no external or Fresh_check source is copied.

| Donor path under src/hlt_classification | New use |
|---|---|
| `luka_cross_message_hydra/model.py` | Subclass the exact HLT message architecture; remove embedding head before optimization, preserve forked head RNG and all deployed parameter values. |
| `luka_cross_message_hydra/training.py` | Reuse forward/predict/optimizer/CE step; adapt fit, selected-checkpoint publication and validators to separate CE-only contracts and forbid target inputs. |
| `luka_cross_message_hydra/{contracts,campaign,submission,worker}.py` | Reuse seeds and adapt immutable source/spec/receipt/exact-job patterns into an independent two-task campaign. Authenticate parent setup in its own source process. |
| `luka_cross_message_hydra/{resources,execution}.py` | Reuse worst-multiplicity subset and A6000 allocation/scheduler validators; adapt resource projection into each new allocation with saved diagnostics. |
| `luka_u100_pair_hydra/inputs.py` | Reuse authenticated bundle/HLT loading; new wrapper allows only D000 TRAIN/validation. No embedding bank or U100 payload is loaded for optimization. |
| `luka_cross_message_hydra/progress.py` and worker reporting | Adapt read-only epoch comparisons and saved-result authentication for CE-only versus CE+embedding controls and historical anchors. |
| `scouting/hcwdl_exact_dag_submission.py`, `scouting/hcwdl_recovery.py` | Reuse exact two-job journals, canonical dry plans and ledger authentication. |
| `jetclass2_delphes/{batch128_30e_recipe,runner,reporting,execution,model}.py`, `context_fusion/worker.py`, `cms_proxy_ladder/contracts.py`, `data/cache_contracts.py`, `luka_fullsim/{foundation,regular_submission,preflight_v2}.py`, `luka_u100_contrastive/training.py` | Unchanged schedule, selection, metrics, GPU/environment identity, atomic save, hashes, source and resource helpers. |

The thin CLI/queue/Slurm wrappers adapt the donor cross-message entrypoints;
tests reuse `test_luka_attention.py` cache fixtures and the compact installed-Weaver
fixture in `test_luka_cross_message_hydra.py`. New artifacts use
`LUKA_CROSS_MESSAGE_CE_HYDRA_*/v1`; no existing contract is relabelled.

## A6000 runtime-envelope correction (2026-11-04)

Internal donor: `c2bdfcc08b6d478165f83ccf59641e4afd34e463`.
Adapted `luka_cross_message_hydra/{campaign,execution,resources}.py` and
`tests/test_luka_cross_message_{hydra,a6000}.py`: version3 separates the48h
A6000 ceiling from the original24h version2, and resource-failure version2
reports explicit measured limits. The time formula, model/training kernels,
32,000MiB envelope and numerical policies are unchanged. No external donor or
Fresh_check migration; execution ancestry remains a7249f49.

## Cross-message A6000 execution (2026-11-04)

Execution baseline: `a7249f49b6f522f7df3b150d27f53613452b7fb0`. Scientific
cross-message donor remains `73abecfd95281246cea8e50bba3887a2737bac82`, whose
original reusable dependencies are listed below at `a40ae47`.
No external source or Fresh_check migration. Existing donor files are unchanged.

| Donor paths under src/hlt_classification at execution baseline | New use |
|---|---|
| `luka_u100_pair_hydra/site.py` | Reuse scheduler/environment validators; adapt allocation and scheduler test-only checks for exact RTX A6000 in new cross-message `execution.py`. |
| `luka_u100_pair_hydra/{precision_campaign,precision,portability}.py` | Reuse inference-only full-control and fixed-batch probes, CPU/GPU numerical qualification and original-source L40 provenance. New `GPU_PORTABILITY/v1` binds A6000-local outputs; no threshold changes. |
| `luka_cross_message_hydra/{campaign,contracts,inputs,submission,worker}.py` and its CLI/queue script | Optional `SPEC/v2` resource profile, bounded cache allowance, site-local replay/reference, and setup-only dispatch with original finish/DAG. Scientific model/training/resource-estimation formula unchanged. |
| `tests/test_luka_cross_message_hydra.py` | Reuse synthetic cache and compact installed-Weaver fixtures for actual local diagnostic kernels and submission tests. |

The old L40 spec remains version1. A6000 needs its own real shared setup;
local GPU identity simulation is only test evidence. See the
[runbook](LUKA_CROSS_MESSAGE_A6000_RUNBOOK.md).

## Oscar FullStat Multi3 contrastive student (2026-11-04)

Internal implementation baseline: `a40ae47ba87ee23020b482c17d467e2a42282e8f`.
All donor paths below refer to that commit unless explicitly noted. New package:
`src/hlt_classification/luka_fullstat_multi3_contrastive/`. Existing runtime is
unchanged; no Fresh_check or external source migration. Completed Oscar parent
is replayed at `ceb2146149d2b1508a36b2cf91b274e0b58a26e6`, whose original
scientific bootstrap uses `a1eb1a746875a2697216d45a7c8ea4b6acdabc8a`.
Execution ancestry now pins `73abecfd95281246cea8e50bba3887a2737bac82`:
the intervening Hydra commit only adds its own runtime files and leaves the
scientific donors above unchanged. The Multi3 runtime remains additive.

| Donor paths under src/hlt_classification unless specified | New use |
|---|---|
| `luka_u100_contrastive/{contracts,sampling,losses,targets}.py` | Unchanged mix/temperature/ramp/RNG/exclusion/normalization semantics; new sampler uses exact chunk8 reductions, stable GPU ranking, persistent CPU workers and bounded lookahead. |
| `luka_u100_multiview/{model,training,teachers}.py` | Existing Multi3 predictors/native tap, validation inference, gradient diagnostics and input forwarding. Independent removable matching head added. |
| `luka_fullstat_multiview/{contracts,training,views,targets,startup,resources}.py` | FullStat20 recipe/seeds, immutable old bank readers, native HLT views, stress/resource conventions; new full-state training/publication. |
| `luka_fullstat_multiview_export/{bootstrap,campaign,worker,submission,launch,execution}.py` | Original-source completion replay; adapted exact-DAG, native publication, scheduler guards and detached launch. |
| `luka_fullstat_{u100_embedding,embedding_sweep}/banks.py` | Unchanged validators for original U100 and pure-OFFLINE TRAIN targets. |
| `luka_fullstat_readout/{source,training,worker}.py`, `luka_fullstat_multiteacher/campaign.py` | Repaired-runtime fingerprint, AdamW/finite gradients/state copying, historical references and protected paths. |
| `luka_embedding_aux/{model,targets}.py`, `luka_fullstat/recipe.py` | Native preclassifier tap, TRAIN standardization and unchanged FullStat LR schedule. |
| `data/cache_contracts.py`, `cms_proxy_ladder/{contracts,submission}.py`, `context_fusion/worker.py` | Canonical hashes, atomic publication, file references, walltime formatting and state serialization. |
| `jetclass2_delphes/{model,runner,reporting,execution,acceptance,dzfix_fusion_model,cache}.py`, `luka_fullsim/{foundation,preflight_v2}.py` | Native model/parity, selection/metrics, bounded process threads, resource/backend/source checks and phase logging. |
| `luka_fullstat/{submission}.py`, `scouting/{hcwdl_exact_dag_submission,hcwdl_recovery}.py`, `luka_fullstat_{retained,fine_tail}/worker.py` | Site dry review, exact submission journal, CUDA synchronization and rejection metrics. |
| `scripts/launch_luka_fullstat_multiview_export.sh`, `sbatch/run_luka_fullstat_multiview_export.sh`, `sbatch/luka_splithead_finalize_common.sh` | Adapted detached one-paste launcher and absolute-path repaired-runtime worker setup. |
| `tests/test_luka_{attention,embedding_aux,u100_contrastive,fullstat_multiview,fullstat_multiview_export}.py` | Reused/adapted synthetic inputs, installed-Weaver parity and failure/recovery test patterns. |

Contracts: `LUKA_FULLSTAT_MULTI3_CONTRASTIVE_*/v1`. See the
[plan](plans/LUKA_FULLSTAT_MULTI3_CONTRASTIVE_PLAN.md) and
[validation](LUKA_FULLSTAT_MULTI3_CONTRASTIVE_VALIDATION.md). No remote validation
or new performance result is inferred from the local tests.

## Cross-attention message Hydra campaign (2026-11-04)

Internal donor/base commit: `a40ae47ba87ee23020b482c17d467e2a42282e8f`.
All paths below are from that commit; old runtime is unchanged. New package:
`src/hlt_classification/luka_cross_message_hydra/`. No external donor,
Fresh_check import, license change or existing-artifact relabelling.

| Donor file(s) | New use |
|---|---|
| `luka_joint_adapter_hydra/training.py` | Adapted shared fit/evaluation/reload; teacher KD, compact message loss/RMS, seed pairs and disabled-message diagnostic. |
| `luka_joint_adapter_hydra/campaign.py`, `resources.py`, `worker.py` | Adapted registration, real production miniatures and exact-job execution/results; encoder delta, frozen-feature cache and fresh controls. |
| `luka_joint_adapter_hydra/submission.py`, `progress.py` | Adapted DAG, measured envelope, claims, watcher recovery and epoch preview. |
| `scripts/luka_joint_adapter_hydra.py`, `scripts/queue_luka_joint_adapter_hydra.sh`, `sbatch/run_luka_joint_adapter_hydra.sh` | Adapted thin entry points and detached queue handoff under new namespace. |
| `tests/test_luka_joint_adapter_hydra.py` | Adapted exact-DAG/recovery, Bash handoff and production setup/worker test patterns. |
| `tests/test_luka_attention.py` | Reused synthetic authenticated cache/node fixtures. |
| `luka_embedding_aux/model.py` | Unchanged native U100-w5 predictor/native initialization semantics. New `model.py` adds cross-attention and HLT compact-message heads. |
| `luka_particle_prediction/model.py::token_forward` | Unchanged native final particle-block tap for frozen U100 encoding. |
| `luka_frozen_adapter_hydra/inputs.py` | Unchanged Brown bundle, validation-U100 delta, view/bank loaders and slot alignment. |
| `luka_u100_pair_transfer/bundle.py` | Unchanged original-source replay, safe file/member checks and canonical metadata validation for new small encoder exporter. |
| `luka_u100_pair_hydra/contracts.py`, `site.py`, `portability.py`, `submission.py` | Existing execution site, input identity, installed environment, completed numerical qualification, accounting and resource-envelope checks. |
| `luka_u100_contrastive_hydra/campaign.py::storage`, `luka_u100_contrastive/training.py::peaks` | Existing fixed-budget disk guard and process/GPU resource measurement. |
| `jetclass2_delphes/model.py`, `execution.py`, `runner.py`, `reporting.py`, `batch128_30e_recipe.py` | Unchanged native model/environment, GPU identity, selection, metrics and schedule. |
| `luka_fullsim/foundation.py`, `alljets_data.py`, `preflight_v2.py`, `regular_submission.py` | Unchanged source pin, exploratory scope, backend requirement and worker environment. |
| `luka_u100_embedding/contracts.py`, `luka_embedding_weight/worker.py` | Fixed population and saved-prediction validation. |
| `data/cache_contracts.py`, `cms_proxy_ladder/contracts.py`, `cms_proxy_ladder/submission.py`, `context_fusion/worker.py` | Canonical hashes/atomic publication, file references, walltime formatting and selected-state storage. |
| `scouting/hcwdl_exact_dag_submission.py`, `scouting/hcwdl_recovery.py` | Existing exact submission journals and ledger assembly. |

`contracts.py`, `model.py`, `inputs.py` and package marker are new, with the
reused semantics listed above. New plan/contract/runbook/validation documents
register `LUKA_CROSS_MESSAGE_HYDRA_*/v1`. See
[validation evidence](LUKA_CROSS_MESSAGE_HYDRA_VALIDATION.md) for completed local
checks and the remaining real Hydra acceptance step.

## FullStat finished-FUSION export recovery (2026-11-03)

Internal scientific donor: `a1eb1a746875a2697216d45a7c8ea4b6acdabc8a`.
Adapter admission baseline: `4b9630fa00f137c4e48b749eb86714719b9d26cf`.
New `luka_fullstat_multiview_export/` operations reuse the original multiview
model/training/views/targets/startup/resources implementations in the original
worktree. Worker fit validation, publication, result inspection and two-job
submission/launch conventions are adapted from the original multiview workers,
recovery and shell helpers. No scientific kernel or historical source is edited.
The [runbook donor inventory](LUKA_FULLSTAT_MULTIVIEW_EXPORT_RUNBOOK.md#validation-and-donor-record)
lists each directly reused/adapted file and its supporting modules. New tests
adapt `tests/test_luka_fullstat_multiview.py`, `_recovery.py`, and `_launch.py`.
No external/Fresh_check migration. New EXPORT execution contracts are v1;
original RECIPE/BANK/scientific semantics remain unchanged. Baseline27 tests
and final36 recovery/parity checks passed locally; real Oscar recovery remains
pending. See [handoff evidence](HANDOFF.md).

## Frozen-adapter export contract repair (2026-11-03)

Donor: `d95975fa256360814225900284f72a1be1d94afc`.
Modified `luka_frozen_adapter_hydra/inputs.py` supplies the existing
`LUKA_FULLSIM_ALLJETS_CACHE/v1` contract to the unchanged validator in
`data/cache_contracts.py`. `tests/test_luka_frozen_adapter_hydra.py` adds actual
export/copy/import and invalid-input regression coverage, using the unchanged
`luka_fullsim/alljets_data.py` artifact producer for synthetic manifests.
No external/Fresh_check migration, schema change or scientific recipe change.
The [repair record](LUKA_FROZEN_ADAPTER_HYDRA_VALIDATION.md) includes reproduction,
71-test evidence and the pending remote export/acceptance step.

## Frozen privileged-adapter Hydra campaign (2026-11-03)

Internal donor baseline: `54473f108c4b1b435f799ca2e41a9c69a4ebe332`.
The new additive `luka_frozen_adapter_hydra/` package reuses the native Weaver
forward, U100 bank/loss conventions, frozen-gate gradient handling, physical
pair features, authenticated Brown relocation and exact-job submission helpers.
It introduces two frozen-M0HLT oracle fits and twelve scratch students; old
gating/KD scientific semantics and existing campaigns remain unchanged.
New artifact contracts are v1. No external/Fresh_check migration or license
change. The [complete donor inventory and validation record](LUKA_FROZEN_ADAPTER_HYDRA_VALIDATION.md)
identifies individual donor files, local evidence, and pending real L40 setup.

## Contrastive Hydra independent-fit adapter (2026-11-03)

Internal donor baseline: `6f7a6cbc587800049c7e5eea6c22cc9af9026d03`.
New execution package: `luka_u100_contrastive_hydra/`; original contrastive
model/loss/sampler/training/resources remain unchanged from successful SPORC
commit `431341db5eced8eaa3e52b37ba047f8cd4beed50`. The serial adapter contributes
only original-source authentication, never its stopping/scheduling policy.
Existing pair-Hydra inputs/site/numerical portability and exact-DAG journals
are reused. New behavior is minimal relocated-key/completed-result import and
all remaining fits queued without student dependencies or a concurrency cap.
No external or Fresh_check migration and no third-party license change.
The complete donor-file inventory, new v1 contracts, test results and remote
limitations are in the [validation record](LUKA_U100_CONTRASTIVE_HYDRA_VALIDATION.md).

## Hydra v2 foreign-checkout source validation repair (2026-11-03)

Donor: `d6f0126dd34dbded4a5be33162765a619edf648c`. Modified in place:
`luka_u100_pair_hydra/portability.py` and its portability tests. Reuse unchanged
`provenance.py:validate_source_snapshot` for the clean foreign snapshot and
`luka_attention_weight/source.py:replay` for the diagnostic's own validation
chain. `foundation._source` remains unchanged and runs in the original process.
No external/Fresh_check migration; v2 contracts and numerical/scientific meaning
are unchanged. Exact reproduction and 81-test evidence are in the
[validation record](LUKA_U100_PAIR_HYDRA_VALIDATION.md).

## Hydra v2 qualification from completed precision evidence (2026-11-03)

Donor: `ad55d3cd32d7826d6e16a475228fece3c6099da4`. New
`luka_u100_pair_hydra/portability.py` reuses unchanged local `precision.py`,
`luka_attention_weight/source.py:replay`, `luka_fullsim/foundation.py:_source`,
data/cache and proxy-ladder file-reference helpers. Existing Hydra
campaign/contracts/submission/worker, CLI and queue helper receive v2 wiring.
All correction model/loss/training/target/resource code and the exact-DAG helper
are imported unchanged. No external or Fresh_check migration. New v2 execution
contracts and test evidence are in the [validation record](LUKA_U100_PAIR_HYDRA_VALIDATION.md).

## Hydra inference-only precision diagnostic (2026-11-02)

Donor: `d7589b312fe5cea7ebb177db0ede70388f0daf45`.
New files `luka_u100_pair_hydra/{precision,precision_campaign}.py` adapt existing
Hydra campaign/submission/worker patterns; the package's `site.py` and `inputs.py`
are reused unchanged. Native `jetclass2_delphes/{model,runner}.py`, contrastive
worker `extended_metrics`, exact-DAG submission journals and proxy-ladder file
references are imported unchanged. The existing Hydra Slurm wrapper is reused.
The CLI and runtime allowlist gain the separate one-job diagnostic route.
No external implementation, Fresh_check code or new scientific model is copied.
New diagnostic-only `PRECISION_*` v1 artifacts cannot replace science acceptance.
See [validation evidence and full donor details](LUKA_U100_PAIR_HYDRA_VALIDATION.md).

## Hydra literal null-cluster export repair (2026-11-02)

Donor: `17a1719ea5d19ae8e2befb1cfc963f31565e2bf6`,
`src/hlt_classification/luka_u100_pair_hydra/site.py` and
`tests/test_luka_u100_pair_hydra.py`. Narrow in-place representation fix based
on user-supplied scheduler evidence from job 131385. No scientific migration,
new external code or Fresh_check dependency. The exact reproduction and
31-test validation are recorded in [the adapter validation record](LUKA_U100_PAIR_HYDRA_VALIDATION.md).

## Hydra allocation compatibility repair (2026-11-02)

Donor commit: `63f31c2ffd3606508a789609e04e2c1f1c5ad38f`.
Modified in place: `src/hlt_classification/luka_u100_pair_hydra/site.py` and
`tests/test_luka_u100_pair_hydra.py`. Scheduler metadata parsing and diagnostic
logging only; no scientific module, training input, model or metric migration.
The Slurm documentation/source links in the execution contract explain metadata
representations; no external source code was copied. The focused validation
record is [LUKA_U100_PAIR_HYDRA_VALIDATION.md](LUKA_U100_PAIR_HYDRA_VALIDATION.md).

## U100 correction Brown CS Hydra adapter (2026-11-02)

Donor base: `65952ed3805f812c06d1ea54117c35b630cc684e`.
New namespace: `luka_u100_pair_hydra`; no Fresh_check import or donor.

- `luka_u100_pair_correction/{worker,submission}.py`: adapted into the new
  namespace for portable references, native-control parity, L40 allocation and
  Hydra exact-DAG dispatch. Original files are unchanged.
- `luka_u100_pair_correction/{targets,campaign,source}.py`: patterns adapted for
  portable TRAIN target assembly, isolated output registration and additive
  source guard. Original pack/statistics/ParticleBank validation/storage reused.
- `luka_u100_pair_correction/{model,geometry,losses,training,resources}.py`:
  imported unchanged, preserving science and real production miniatures.
- `luka_u100_pair_transfer/bundle.py`: imported standard-library archive checks
  and canonical identities; safe extraction is new in the consumer.
- `luka_fullsim/{regular_cache,alljets_data,foundation,regular_submission}.py`:
  unchanged block/foundation validation, source capture and clean environment.
- `luka_u100_embedding/targets.py`, `jetclass2_delphes/{model,runner,reporting}.py`,
  `luka_u100_contrastive/worker.py`, `cms_proxy_ladder/contracts.py` and
  `data/cache_contracts.py`: unchanged bank/model/inference/metrics/artifact APIs.
- `scouting/{hcwdl_exact_dag_submission,hcwdl_recovery}.py`: unchanged exact-job
  journal and ledger primitives. `cms_proxy_ladder/submission.py`: walltime helper.
- `scripts/{luka_u100_pair_correction.py,queue_luka_u100_pair_correction.sh}` and
  `sbatch/run_luka_u100_pair_correction.sh`: CLI and shell orchestration adapted
  for a separate venv and Hydra profile. New isolated environment installer.
- `tests/test_luka_u100_pair_{transfer,correction}.py` and their historical
  synthetic fixtures supply reused test data and failure-injection patterns.

New execution/extraction v1 contracts preserve original scientific contracts.
The active Hydra plan explicitly declares the site and numerical-parity policy.

## U100 particle/pair minimal transfer (2026-11-02)

Donor inspection base: `15bb1abf78aa51e50f036777298919e3dfd552a0`.
New implementation lives in `luka_u100_pair_transfer` and its CLI. It does not
copy or modify the original training, geometry, loss or scheduling code.

- `luka_u100_pair_correction/{source,campaign,targets,worker}.py`: reference
  replay, prepared-input scope and result consumption inform the explicit file
  list. Original registered source performs `campaign.check`, `source.import_control`
  and `targets.metadata` in a read-only subprocess.
- `luka_attention_weight/source.py`: original-source subprocess pattern adapted
  without a runtime import, keeping the verifier standard-library only.
- `data/cache_contracts.py`: canonical JSON/content-hash convention reproduced
  with standard-library JSON/SHA256, avoiding its NumPy dependency on Brown.
- `luka_fullsim/{regular_cache,alljets_data}.py`,
  `luka_u100_multiview/source.py` and `luka_u100_embedding/{source,targets}.py`:
  unchanged metadata schemas and file-reference roles inspected for selection.

The original evidence remains pinned to its recorded SPORC commits. The bundle
records the exporter's module digest and original-source proof. No Fresh_check
donor, raw particle source, or model implementation is migrated by this step.

## Serial contrastive cancellation repair (2026-11-02)

Modified `luka_u100_contrastive_serial/{submission,campaign}.py` from
`f8f61ade` and extended `tests/test_luka_u100_contrastive_serial.py` at that
commit. The fix uses a current controller snapshot, pending-only controller
cancellation, durable diagnostics and final receipt authentication for racing
completions. No training code or donor runtime was copied or modified. Original
scientific donors below and parent replay at `431341db` are unchanged.

## Serial contrastive continuation (2026-11-02)

All donors below are from reviewed commit
`c08f6df603d87a452a4814a9e954421d8731a4a7`. Original scientific runtime is
unchanged; implementation lives only in `luka_u100_contrastive_serial` and its
thin CLI/queue/worker. Parent execution replay stays pinned to431341db.

- `luka_u100_contrastive/training.py`: adapted epoch loop, fit/report/native
  reload logic; unchanged imported `optimizer`, `step`, `diagnostic`, `peaks`,
  `recovery_percent`. New logic adds policy replay and durable best publication.
- `luka_u100_contrastive/{contracts,model,targets,resources}.py`: unchanged
  arm coefficients, seeds, matching ramp, native model, target loaders and
  sampler creation. No new export or resource measurement is run.
- `luka_u100_contrastive/{campaign,source,submission,worker}.py`: patterns for
  hash-bound registration, source guard, exact journals and result joins;
  imported `worker.extended_metrics`. Original-source subprocess invokes its
  own validation functions without changing their meaning.
- `scripts/{luka_u100_contrastive.py,queue_luka_u100_contrastive.sh}` and
  `sbatch/run_luka_u100_contrastive.sh`: adapted CLI, environment and detached
  submission wrappers. New dispatcher adds pending-only exact-ID transition.
- `luka_attention_weight/source.replay`, `luka_fullsim/foundation._source`,
  `luka_fullsim/regular_submission`, `scouting/hcwdl_exact_dag_submission` and
  `scouting/hcwdl_recovery`: unchanged provenance, environment and journal tools.
- `luka_u100_multiview/views.load`, `jetclass2_delphes/{model,runner,reporting,
  execution,batch128_30e_recipe}`, `context_fusion/worker.save_state`,
  `cms_proxy_ladder/contracts` and `data/cache_contracts`: unchanged native
  data/model/selection/LR, artifact publication and authentication primitives.
- Test fixtures from `tests/test_luka_{u100_contrastive,u100_embedding,
  embedding_aux,attention}.py` supply real installed-Weaver miniature models,
  authenticated synthetic banks and original parity oracles.

## Four retained U100 particle/pair corrections (2026-11-02)

Reviewed donor/source-guard baseline:
`431341db5eced8eaa3e52b37ba047f8cd4beed50`. All donors below are from that
commit; existing runtime files remain unchanged. New geometry, correction
heads and their tests live only in `luka_u100_pair_correction`.

- `luka_u100_contrastive/submission.py` is the copied/adapted six-job dry/live
  graph, exact journal and detached dispatcher implementation. Its20-fit grid
  is replaced with exactly four correction fits, retaining two afterany lanes.
- `luka_u100_contrastive/{campaign,source,resources,worker,training,contracts}.py`,
  `scripts/{luka_u100_contrastive.py,queue_luka_u100_contrastive.sh}` and
  `sbatch/run_luka_u100_contrastive.sh` supply patterns for the new registration,
  provenance replay, shared miniature, metrics, worker ownership, CLI and shell
  environment. `resources.Subset/estimated_minutes`,
  `training.optimizer/recovery_percent/peaks` and `worker.extended_metrics`
  are reused unchanged. Original-source replay invokes the completed
  contrastive control's own validators; no other contrastive student is required.
- `luka_u100_embedding/{contracts,targets,training}.py` supplies the original
  U100-w5 recipe, full-payload Bank validator and unchanged CE1/embedding5 loss.
  `luka_embedding_aux/model.py` supplies original native/predictor initialization
  and classifier-input embedding capture. Its native-only checkpoint filter is
  deliberately not used: new deployment retains correction parameters.
- `luka_gating/model.py` supplies the verified native attention pre-hook shape
  pattern. Its gate physics, trainable factors and frozen-QKV backend exception
  are not imported into this experiment.
- `luka_u100_multiview/{source,views,offline_cache}.py` supplies reference
  validation, support lengths and the missing-only byte-identical raw-to-RAM
  replay pattern. New replay supports the registered D000/U100 views and
  preserves prepared matching; present corruption remains fatal.
- `luka_fullsim/{regular_cache,preflight_cache,stage2,particles,foundation,
  regular_submission,preflight_v2,alljets_data}.py` supplies original input
  fingerprints/readers/assignment joins, source guards, site checks and memory
  evidence. `cms_proxy_ladder/{views,contracts,submission}.py` supplies view
  semantics, file references and walltime formatting.
- `jetclass2_delphes/{model,cache,runner,reporting,batch128_30e_recipe,execution}.py`,
  `models/particle_transformer.py`, `context_fusion/worker.py`,
  `data/cache_contracts.py`, `provenance.py`, `luka_embedding_weight/worker.py`,
  `luka_attention_weight/source.py`, and
  `scouting/{hcwdl_exact_dag_submission,hcwdl_recovery}.py` are unchanged model,
  batching, selection, atomic publication, authentication and submission primitives.
- `tests/test_luka_attention.py` supplies physical ragged fixtures;
  `tests/test_luka_u100_contrastive.py` supplies dispatch/failure-test patterns.

The installed Weaver factory is the external runtime dependency, recorded by
the existing full installed-environment hash, not copied into this package.
New artifacts use `LUKA_U100_PAIR_CORRECTION_*/v1`; no historical artifact is
upgraded or mutated. No Fresh_check runtime import or donor copy is introduced.

## Contrastive exclusion filtering repair (2026-11-01)

Runtime/test donor: `810433f83eb6ab4ed78d18f6ab75ce70f5114350`, specifically
`luka_u100_contrastive/sampling.py:draw,balanced` and the existing focused
sampler tests. Their previous scalar logic is retained as independent test
oracles; runtime now groups exclusions and uses vectorized membership with
exact candidate/RNG parity. Existing target-reuse, losses, model, training,
resource limits and submission helpers are reused unchanged. No Fresh_check
code is imported or copied.

The new source guard baseline is current main
`a1eb1a746875a2697216d45a7c8ea4b6acdabc8a`. Its changes after 810433f8 belong
to the independent FullStat recovery and do not change the contrastive
scientific implementation or its imported primitives. That runtime is frozen
by the new guard rather than reverted or included in this repair. No schema
version bump is needed: SPEC/BUNDLE stay v2, other scientific artifacts stay v1.

## FullStat completed-CONCAT / FUSION timing recovery (2026-11-01)

Original execution donor: `8ad0b381547134569818b4258f3cd31b509e9efb`.
Current additive source baseline: `810433f83eb6ab4ed78d18f6ab75ce70f5114350`.

- `luka_fullstat_multiview/{campaign,source,submission,execution,launch,worker,
  startup,resources,contracts}.py` supply the existing source checks, receipts,
  resource formula, three-task production workflow and durable publication.
  New `recovery.py` plus narrow dispatch changes reuse the completed CONCAT
  in original source and schedule only FUSION/student under a72h/24h envelope.
- `scripts/launch_luka_fullstat_multiview.sh` supplies the new recovery launcher's
  exclusive adjacent directory, nohup/setsid detachment, log view and exact-source
  runtime activation. The recovery helper has its own command/authorization.
- `luka_fullstat_multiview/{model,training,fusion,views,targets}.py` are unchanged
  consumers. Scientific recipe and all five file contents were checked against
  the8ad0b381 donor. Existing full-pair synchronous offload is reused unchanged.
- `cms_proxy_ladder/contracts.py` and `scouting/hcwdl_exact_dag_submission.py`
  provide unchanged file authentication, atomic exact-DAG journals and partial
  submission protection. No runtime donor outside multiview is edited.

The only original-campaign write is a new exclusive recovery claim after a
successful dry review. No old artifact is relabeled, copied as a fresh export,
or reparented. No runtime imports of Fresh_check or new scientific semantics.

## Contrastive setup runtime repair and target recovery (2026-10-31)

Donor execution/source: `755f43d874ce90ff50ae9b80fe66d36869601650`.
The existing `luka_u100_contrastive/{sampling,worker,targets,resources,campaign,
contracts,source,submission}.py` and thin CLI/queue helper provide the unchanged
scientific recipe, immutable exports and exact-job journal validation. Only
this campaign's runtime is edited. No Fresh_check or historical model runtime
is changed.

New `luka_u100_contrastive/reuse.py` invokes the original campaign check,
submission journal/accounting and resource validator in that pinned source;
it uses unchanged `luka_attention_weight/source.py:replay`,
`provenance.py:validate_source_snapshot`, `cms_proxy_ladder/contracts.py`
file references and `data/cache_contracts.py` atomic publication/hash helpers
from the same donor commit. Original KEYS/teacher proofs retain their v1
identity. New SPEC/BUNDLE v2 declare external target ownership and new
SETUP_REUSE/REUSED_EXPORTS v1 record import evidence. The indexed sampler is
an execution optimization with candidate parity, not a new sampling recipe.

## U100 plus paired OFFLINE identification (2026-10-31)

Reviewed donor/source commit: `8ad0b381547134569818b4258f3cd31b509e9efb`.
New runtime is exclusively `luka_u100_contrastive` and its thin entrypoints.
Historical runtime is imported unchanged; no Fresh_check donor/runtime import.

| Donor at 8ad0b381 (under src/hlt_classification unless stated) | New use |
| --- | --- |
| `luka_u100_embedding/{contracts,training,targets}.py` | Exact CE1/U1005 recipe; direct original zero-control loop; detached SmoothL1; original teacher-bound TRAIN bank validation/export |
| `luka_embedding_aux/{model,targets}.py` | Native pre-classifier tap, RNG-isolated MLP factory, native state stripping, FP64 TRAIN population moments with scale floor |
| `luka_u100_multiview/{source,views,offline_cache}.py` | Original-source fixed30/direct-U100 result-only authentication and exact historical D000/U100/OFFLINE loading/reconstruction; no multiview objective imported |
| `luka_fullsim/{foundation,regular_submission,preflight_v2}.py` | Exact clean pushed source, site/resource dry checks, sanitized submission environment, required backend and Linux memory evidence |
| `jetclass2_delphes/{model,runner,batch128_30e_recipe,reporting,execution}.py` | Installed-Weaver native graph/environment; original optimizer schedule/selection; saved metric replay; explicit SPORC/debug allocation |
| `data/cache_contracts.py`, `cms_proxy_ladder/{contracts,submission}.py`, `context_fusion/worker.py` | Canonical hashes, atomic JSON/NPZ publication, file references, walltime rendering and native checkpoint serialization |
| `scouting/{hcwdl_exact_dag_submission,hcwdl_recovery}.py` | Existing immutable exact-ID submission journals and ledger replay; new graph has resource-only afterany lanes |
| `tests/test_luka_{attention,embedding_aux,u100_embedding}.py` | Synthetic ragged caches, real installed-Weaver reduced-block fixture and original-bank test fixture |

New schemas are `LUKA_U100_CONTRASTIVE_<KIND>/v1`; old contracts remain intact.
New samplers, fixed-key matching objective, head wiring, diagnostics, resource
estimation, and automatic twenty-fit dispatcher implement the new active plan.
No production run, remote measurement or donor scientific result is claimed by
this mapping.

## FullStat multiview resource repair (2026-10-31)

Reviewed donor/source commit: `2b1f0370993c2f20be3951a6ccafe572ab21600a`.
Only the additive `luka_fullstat_multiview` runtime changes. Historical SPORC,
readout and retained workers remain byte-identical. The original additive
source boundary at 74cba41f is retained.

| Donor at 2b1f0370 (under src/hlt_classification) | New use |
| --- | --- |
| `luka_fullsim/alljets_fusion_sync.py` | `luka_fullstat_multiview/fusion.py`: synchronous, stride-preserving pair saved-tensor host storage; subclass changes only the six-input calling interface; reused three-update storage parity |
| `jetclass2_delphes/{dzfix_fusion_model,salience_learned_model}.py` | Existing compact mask, native six-tensor forward, original graph/state layout, parity backend and native-mask check |
| `luka_u100_multiview/teachers.py` | Identical primary/context initialization and CUDA RNG preservation for the optimized teacher factory |
| `luka_fullstat_retained/worker.py` | `resources.py`: unchanged 90% GPU, 85% host and 8 GiB reserve limits with multiview-specific measured-limit errors |
| `luka_fullstat_multiview/{startup,campaign,launch}.py` | 48h teacher requests, 24h student request, versioned runtime policy, diagnostic-first admission and unchanged timing factor/margin |

SPEC/STARTUP advance to v2; RUNTIME_POLICY and non-accepting STARTUP_DIAGNOSTIC
are v1. Scientific RECIPE, full particle support, CE/embedding objectives,
batch128, seeds, 20 epochs and TRAIN bank meaning are unchanged. Runtime
storage parity is checked with installed Weaver; this is not a claim of
bitwise identical full training across CUDA kernels. No external donor or
license change. See the active plan, contract and HANDOFF for validation.

## FullStat three-teacher multiview adaptation (2026-10-30)

Reviewed implementation donor/additive baseline:
`74cba41f4938df8aedcee4f931ef14bfc79b839d`. New implementation is isolated in
`luka_fullstat_multiview`; no old runtime or artifacts change. Acceptance is
replayed in corrected readout `db9afe8b87cdef6f3489f9b85603f52cf7d2084f`, which
authenticates original U100-w5 at `7ea8b0c693734a26ced3820354e047f4af26c979`.
The four subsequent old-SPORC file differences (campaign/contracts/source/views)
are allowed only with separately pinned original/current source inventories;
new source must remain additive relative to 74cba41f.

| Donor at 74cba41f, paths under src/hlt_classification unless stated | New destination / retained semantics |
| --- | --- |
| `luka_fullstat_retained/{__init__,contracts,source,campaign,execution,submission,launch}.py` | Corresponding new registration, repaired runtime allocation, original-source replay, quota and exact-ID detached submission; three jobs instead of two |
| `luka_fullstat_retained/worker.py` | Resource/synchronization helpers; inline startup, durable epochs and publication structure split across new worker/startup/training |
| `luka_u100_multiview/{model,teachers,contracts,training}.py` | Unchanged native three-predictor Student, CONCAT/FUSION graphs and tap, independent head seeds, prediction and gradient diagnostics; new objective total10, FullStat20 schedule |
| `luka_u100_multiview/views.py` | Source-blind physical multiset union, paired joins and support scan; new FullStat raw adapter with bounded four-worker preparation and no crop |
| `luka_embedding_aux/targets.py`, `luka_fullstat_u100_embedding/{contracts,banks}.py` | Independent TRAIN standardization, original U100 TRAIN-bank reuse and FullStat recipe |
| `luka_fullstat/{data,cache,contracts,recipe,submission}.py` | Exact split/identities, sealed-role raw access, native caches, optimizer schedule, scheduler dry shape validation |
| `luka_fullstat_readout/{campaign,source,contracts,model,training,worker}.py` | Original lineage validation, corrected historical teacher replay, generic optimizer/state helpers and validated reference predictions |
| `luka_fullstat_attention/{contracts,worker}.py`, `luka_fullstat_multiteacher/campaign.py`, `luka_fullstat_fine_tail/worker.py` | FullStat flags/counts, prepared metadata, protected roots and R30/R50 |
| `luka_fullsim/{particles,foundation,preflight_cache,preflight_v2}.py`, `literature_context/transform.py` | Native physical conversion, source locks, ragged blocks, shared feature transform, resource phase logs |
| `jetclass2_delphes/{model,runner,reporting,acceptance,execution,dzfix_fusion_model,cache}.py` | ParT/Weaver parity, selection, metrics, scheduler/environment contracts, strict parity context, RAM arrays |
| `data/cache_contracts.py`, `cms_proxy_ladder/{contracts,submission}.py`, `context_fusion/worker.py`, `scouting/{hcwdl_exact_dag_submission,hcwdl_recovery}.py` | Canonical hashes, atomic publication, file refs, state saving and crash-safe submission journals |
| `scripts/{luka_fullstat_retained.py,queue_luka_fullstat_retained.sh,launch_luka_fullstat_retained.sh}`, `sbatch/run_luka_fullstat_retained.sh` | Adapted thin multiview CLI/queue/launch/worker; reused absolute `luka_splithead_finalize_common.sh` runtime activation |
| `tests/test_luka_fullstat_retained{,_launch}.py`, `tests/test_luka_fullstat.py`, `tests/test_luka_u100_multiview.py`, `tests/test_luka_embedding_aux.py` | Production-worker, launch, real ROOT, native tap and oracle fixtures; new three-job and bank integrity tests |

New `LUKA_FULLSTAT_MULTIVIEW_*/v1` contracts explicitly distinguish this science
from both the retained and SPORC30 campaigns. No external/Fresh_check code or
license changes. See [active plan](plans/LUKA_FULLSTAT_MULTIVIEW_PLAN.md),
[contracts](contracts/LUKA_FULLSTAT_MULTIVIEW.md), and HANDOFF for test evidence.

## Missing native-OFFLINE cache replay for multiview (2026-10-30)

Reviewed donor/source baseline: `ac76bbd9866cb00891bc554dac7ae8b31778c643`.
The original multiview implementation was `de79c7298a54dcb5818b97772635837e0a769914`.
Only this campaign's namespace changes; shared historical readers remain intact.

| Donor at reviewed baseline | New use |
| --- | --- |
| `luka_fullsim/regular_cache.py` | Selected-file raw reader, authenticated present-block read, metadata ordering in `luka_u100_multiview/offline_cache.py` |
| `luka_fullsim/{particles,preflight_cache,stage2}.py` | Native OFFLINE physical decoding, ragged `_block`, original feature frontend and complete `ViewMeter` replay |
| `data/cache_contracts.py` | In-memory deterministic NPZ serialization for exact historical payload byte-count/SHA256 comparison |

SPEC advances to v2 to register the missing-only, RAM-only replay policy.
Scientific RECIPE and bank/fit contracts remain v1 because original inputs must
match exactly. No historical artifact is rewritten, no alternative teacher is
substituted and no donor/Fresh_check runtime is copied or changed. Source guard
pins all other runtime files at the reviewed baseline and permits only the
enumerated multiview files/wrappers to change; see the multiview runbook.

## FullStat retained predictor and matched CE-only control (2026-10-30)

Reviewed donor and additive compatibility baseline:
`db9afe8b87cdef6f3489f9b85603f52cf7d2084f`. Authenticate the completed corrected
Oscar readout in that exact source, then reuse its original FullStat U100-w5
and regular COARSE_U100 lineage (`7ea8b0c693734a26ced3820354e047f4af26c979`).
No existing runtime is edited; no Fresh_check migration or runtime import.

| Donor at db9afe8b (under src/hlt_classification unless stated) | Destination / use |
| --- | --- |
| `luka_u100_multiview/{model,contracts,training}.py` | Reused retained Student/objective, isolated predictor seed, matched CE-only topology and gradient diagnostics |
| `luka_fullstat_readout/{contracts,source,campaign,model,training,worker,execution,submission,launch}.py` | Adapted isolated `luka_fullstat_retained` registration, original-source acceptance replay, repaired Oscar allocation, fit and publication checks, source isolation, quota, detached exact-ID submission; reused corrected historical teacher replay and generic optimizer/state helpers |
| `luka_fullstat_u100_embedding/{contracts,banks,worker}.py`, `luka_fullstat/recipe.py` | Original FullStat20 recipe, TRAIN bank/row guards, HLT and teacher view loading, maximum-support scan |
| `luka_fullstat_attention/{contracts,worker}.py`, `luka_fullstat_multiteacher/campaign.py`, `luka_fullstat_fine_tail/worker.py` | Sealed roles/counts, authenticated prepared metadata, protected roots, rejection semantics |
| `jetclass2_delphes/{model,runner,reporting,acceptance,dzfix_fusion_model,execution}.py`, `luka_fullsim/{foundation,preflight_v2}.py` | Unchanged ParT/Weaver parity, prediction/selection, metrics, environment/site, source and resource witnesses |
| `scouting/{hcwdl_exact_dag_submission,hcwdl_recovery}.py`, `luka_fullstat/submission.py` | Reused exact-ID crash journals, clean scheduler environment/site validation; new afterany dependency only orders two independent fits |
| `scripts/{luka_fullstat_readout.py,queue_luka_fullstat_readout.sh,launch_luka_fullstat_readout.sh}`, `sbatch/run_luka_fullstat_readout.sh` | Adapted retained CLI/queue/detached launch and absolute-project Slurm worker; unchanged `sbatch/luka_splithead_finalize_common.sh` activates repaired runtime |
| `tests/test_luka_fullstat_readout{,_launch}.py` | Adapted allocation/launch/production-worker fixtures and tests; existing U100, meta-embedding and embedding-aux test fixtures are reused |

New versioned namespace: `LUKA_FULLSTAT_RETAINED_*/v1`. Both selected checkpoints
retain predictor and combined classifier; native-only checkpoints are rejected.
Two 20-epoch fits, CE1/E5 then CE-only, same 3M/500k population. The small-data
multiview30 schedule is not migrated. See [active plan](plans/LUKA_FULLSTAT_RETAINED_PLAN.md),
[contract](contracts/LUKA_FULLSTAT_RETAINED.md), [runbook](LUKA_FULLSTAT_RETAINED_RUNBOOK.md).

## FullStat readout teacher replay correction (2026-10-30)

Patch donor: `abe387791386507580b7e826850577c2b4cea6cf`,
`src/hlt_classification/luka_fullstat_readout/{worker,contracts,campaign,execution}.py`
and `tests/test_luka_fullstat_readout{,_launch}.py`. The historical export
semantics come from `luka_fullstat_u100_embedding/banks.py` at
`7ea8b0c693734a26ced3820354e047f4af26c979`: contiguous FP32 TRAIN batches of128,
ordinary inherited backend, no strict parity context. Only the readout namespace
and its tests/docs change; original exporter/model/training files remain intact.
SPEC/STARTUP become v2 and TEACHER_REPLAY is new at v1. Scientific RECIPE remains
v1. No Fresh_check migration or import.

## FullStat U100 frozen-readout CE (2026-10-30)

Reviewed implementation donor and additive compatibility baseline:
`de79c7298a54dcb5818b97772635837e0a769914`. Original completed FullStat U100-w5
artifact/source donor: `7ea8b0c693734a26ced3820354e047f4af26c979` (replayed in its
own immutable source). All new runtime is under `luka_fullstat_readout`; old
runtime bytes and artifacts are unchanged. No Fresh_check migration/import.

| Donor files (under `src/hlt_classification/` unless stated) at de79c729 | Destination / use |
| --- | --- |
| `luka_fullstat_u100_embedding/{contracts,banks,training,worker}.py` | FullStat20 optimizer/schedule, TRAIN bank and role guards, fresh single-predictor loop, validation selection/readback |
| `luka_embedding_aux/model.py` | Reused exact native tap, RNG-isolated predictor and native deployment |
| `luka_u100_multiview/{model,training}.py` | Reused frozen-readout buffers and component gradient diagnostic; CE ramp adapted to 20 epochs, no small-data teacher/bank or schedule reuse |
| `luka_fullstat_meta_embedding/source.py` | Original-source U100-w5 authentication and readonly bank/reference import, adapted to new REUSE identity |
| `luka_fullstat_finetune/{campaign,execution,submission,launch,worker}.py` | Source/quota/isolation guards, actual repaired Oscar allocation, one-job journal/launcher, durable per-epoch full states and saved-result checks |
| `luka_fullstat_multiteacher/{source,campaign}.py`, `luka_fullstat_attention/worker.py`, `luka_fullstat_fine_tail/worker.py` | Reused enumerated unrelated SPORC source exceptions, protected roots, prepared metadata, R30/R50 semantics |
| `scripts/{luka_fullstat_finetune.py,queue_luka_fullstat_finetune.sh,launch_luka_fullstat_finetune.sh}`, `sbatch/run_luka_fullstat_finetune.sh` | Adapted thin readout CLI and detached/manual queue wrappers; reused absolute repaired-runtime helper |
| `tests/test_luka_fullstat_finetune_launch.py`, `tests/test_luka_fullstat_u100_embedding.py`, `tests/test_luka_fullstat_meta_embedding.py`, `tests/test_luka_embedding_aux.py` | Adapted launch/runtime checks and reused synthetic fixtures for complete readout worker tests |

New contracts: `LUKA_FULLSTAT_READOUT_*/v1`. One fresh HLT fit, CE1/E5 with
three-pass-ramped frozen-U100 readout CE; 3M TRAIN/500k validation, 20 epochs,
historical U100-w5 reference only. See [plan](plans/LUKA_FULLSTAT_READOUT_PLAN.md),
[contract](contracts/LUKA_FULLSTAT_READOUT.md) and
[runbook](LUKA_FULLSTAT_READOUT_RUNBOOK.md).

## U100 multiview/readout first screen (2026-10-29)

Reviewed donor commit: `9868f38641d1648e43f519c20e53ec12efe04dc3`.
All destinations are additive under `src/hlt_classification/luka_u100_multiview/`;
existing runtime files are unchanged. No Fresh_check donor/runtime import.

| Donor files at that commit | Destination / use |
| --- | --- |
| `luka_u100_embedding/{contracts,source,targets,training,worker,direct}.py` | New contracts, original-source import, compact TRAIN normalization, fixed30 loop, readback and staged worker patterns |
| `luka_embedding_aux/{model,source}.py` | RNG-isolated predictor/native tap and original fixed30 baseline authentication |
| `jetclass2_delphes/{model,runner,batch128_30e_recipe,reporting}.py` | Imported native factory, selection rule, immutable schedule, metrics |
| `jetclass2_delphes/salience_learned_model.py` | Imported four-depth asymmetric fusion factory; fresh HLT/native-OFFLINE topology, alpha1 |
| `luka_fullsim/{regular_cache,particles,preflight_cache,alljets_data}.py` | Authenticated historical caches, raw selected-file reads, physical conventions and ragged blocks in `views.py` |
| `cms2jc2_response/bridge.py`, `literature_context/transform.py` | Common physical union and exact historical 17-feature frontend; replaces draft's incorrect generic JetClass frontend reference |
| `luka_meta_guard/submission.py`, `scouting/{hcwdl_exact_dag_submission,hcwdl_recovery}.py` | Adapted reviewed submission flow; imported exact-ID journaling, distinct acceptance/science identities |
| `luka_meta_embedding/campaign.py` | Fixed-budget/headroom pattern; never reserve all declared free space |

New semantics: untagged full-support union, frozen-readout CE/KD, retained
predictor plus capacity control, separate debug resource admission, and owned
bank retirement. These use `LUKA_U100_MULTIVIEW_*/v1`; old envelopes are not
relabeled. Historical fixed30 remains pinned to `74aa58c5`; direct U100-w5 to
`ed6d0af4be65f059356543f90583b30185702615`. See HANDOFF for validation evidence.

## Guarded FIXED diagnostic numerical acceptance repair (2026-10-29)

Donor: `691cccf357b7c97e6aa6063469fa9061cb4fba29`, specifically
`src/hlt_classification/luka_meta_guard/{training,contracts}.py`,
`scripts/queue_luka_meta_guard.sh` and `tests/test_luka_meta_guard.py`.
Repaired exact-zero FIXED query-loss validation to an absolute-only `1e-6`
tolerance, preserving exact target identity and raw signed diagnostic evidence.
V3 artifacts/recipe/authorization distinguish this numerical policy from v2;
scientific losses/sampling/guard/training and imported runtime remain unchanged.
No external donor migration. Parent setup/source stays pinned to `7a6dc268`;
guard source compatibility still uses `b9cf63b0`. Failed SPORC `691cccf3` outputs
are preserved, not recovered or rewritten. See HANDOFF for test evidence.

## Guarded SPORC cancellation-safe initializer reuse (2026-10-29)

Revision of `luka_meta_guard` at donor commit
`a65caa78913e5e31a89ff7855b6f184605881d22`. The original setup execution remains
pinned to `7a6dc268800b0788e0e89d7ec56db37583397946`. V2's source-compatibility
baseline is `b9cf63b0bf4a037e11ece91da21db465d2347b96`: the intervening commit
only adds the isolated FullStat fine-tuning namespace. Existing scientific
runtime is byte-unchanged from the v1 `c10d2205` baseline outside the guard.
New changes are allowed only in explicitly listed guard-owned runtime paths.
No external files were migrated and no historical scientific runtime was edited.

Reused original `luka_meta_embedding/worker.py` functions `validate_task`,
`read_shared`, and `load_inputs`, plus `campaign.check` and
`submission.accepted_jobs`, authenticate the completed shared setup, calibrated
full predictor and bank without requiring cancelled FIXED/META fits. Destinations
are guard `source.py`, `campaign.py`, `worker.py`, `contracts.py`, `training.py`,
CLI/queue helper, tests and guard documentation. V2 versions the explicit
initializer choice rather than changing the meaning of existing v1 artifacts.

## Oscar FullStat fine-tail late fine-tuning (2026-10-29)

Donor baseline: `c10d2205677bae4769d45da447c338165bc6724f`.
Additive compatibility baseline: `a65caa78913e5e31a89ff7855b6f184605881d22`.
That intervening commit adds the isolated meta-guard namespace; parent runtime
bytes remain unchanged. Its concurrent uncommitted edits were not touched.
Completed parent execution: `04d1f0f5945ab9af38d83e192eff48e124ad8dc5`.
Destination: `src/hlt_classification/luka_fullstat_finetune/`, four matching
CLI/queue/launch/Slurm entrypoints, two focused test files and FINETUNE docs.
No external migration, Fresh_check runtime import or original runtime edit.

Donor files at the donor baseline commit (fine-tail files are also byte-identical to
the completed parent commit; original-source replay enforces that compatibility):

- `src/hlt_classification/luka_fullstat_fine_tail/{contracts,model,training,banks,kd_banks,worker,source,execution}.py`:
  unchanged seven-teacher objective, native model/heads, AdamW groups, TRAIN
  banks, HLT loading, rejection and repaired-runtime semantics. New training
  replaces the original from-scratch 20-epoch loop with frozen-native head
  calibration and ten low-LR warm-start epochs; it is not exact resume.
- `src/hlt_classification/luka_fullstat_meta_embedding/{submission,launch}.py`,
  `scripts/{queue_luka_fullstat_meta_embedding.sh,launch_luka_fullstat_meta_embedding.sh,luka_fullstat_meta_embedding.py}`,
  `sbatch/run_luka_fullstat_meta_embedding.sh`, and
  `tests/test_luka_fullstat_meta_embedding_launch.py`: adapted to one fit,
  explicit one-job authorization, fresh claims, dry review and detached logging.
- `sbatch/luka_splithead_finalize_common.sh`: imported unchanged replacement
  runtime activation and absolute source path handling.
- `src/hlt_classification/luka_fullstat_multiteacher/campaign.py` and
  `src/hlt_classification/luka_fullstat_attention/{contracts,worker}.py`:
  protected roots, fixed FullStat counts/flags and authenticated cache metadata.
- `src/hlt_classification/luka_embedding_aux/model.py`: native representation tap.
- `src/hlt_classification/jetclass2_delphes/{model,runner,reporting,acceptance,execution,dzfix_fusion_model}.py`:
  native inference, selection/metrics, installed-Weaver parity and runtime guards.
- `src/hlt_classification/luka_fullsim/{foundation,preflight_v2}.py` and
  `src/hlt_classification/luka_fullstat/submission.py`: source authentication,
  phase/resource accounting, scheduler site check and clean submission environment.
- `src/hlt_classification/{data/cache_contracts.py,context_fusion/worker.py}`,
  `src/hlt_classification/cms_proxy_ladder/{contracts,submission}.py`, and
  `src/hlt_classification/scouting/{hcwdl_exact_dag_submission,hcwdl_recovery}.py`:
  immutable publication, file refs, exact-job journaling and no-retry claims.
- `tests/{test_luka_embedding_aux,test_luka_u100_pretrain,test_luka_attention}.py`:
  synthetic HLT cache, teacher/node and small-model fixtures. The complete new
  CUDA worker test uses the full native architecture, not the small-model fixture.

All new identities are `LUKA_FULLSTAT_FINETUNE_*/v1`; old contracts are retained.
See the runbook and HANDOFF for measured local evidence and remote limitations.

## Guarded SPORC FIXED/META continuation (2026-10-29)

Additive compatibility/donor baseline:
`c10d2205677bae4769d45da447c338165bc6724f`. Original FIXED execution pin:
`7a6dc268800b0788e0e89d7ec56db37583397946`.
Destination: `src/hlt_classification/luka_meta_guard/`, matching CLI/queue/Slurm
helpers, `tests/test_luka_meta_guard.py`, and the META_GUARD plan/contract/runbook.
No external donor migration, Fresh_check import, original-runtime modification,
or changes to the separate FullStat/Oscar campaign.

Donor files (relative paths, at the baseline commit above):

- `src/hlt_classification/luka_meta_embedding/{model,meta}.py`: imported unchanged
  student/predictor, bounded adapter, CE+5 embedding loss, full five-step
  differentiable AdamW and virtual-state isolation.
- `src/hlt_classification/luka_meta_embedding/training.py`: imported TRAIN split,
  support/query batches and CPU state copier. Its loop is **not** reused: the new
  version adds three-way TRAIN roles, stationary adapter phases, half-pass blocks,
  bounded rollback and block-zero-inclusive durable selection.
- `src/hlt_classification/luka_meta_embedding/worker.py`: reused borrowed-bank
  loading and maximum-support diagnostic. New worker additionally checks the
  actual lower virtual LR, restores a trained predictor, and does no export or
  recalibration. `{source,campaign,submission,contracts}.py` supply structural
  patterns for original-source replay, storage checks and exact three-job DAG.
- `src/hlt_classification/luka_ce_warmstart/training.py`: matched native AdamW groups.
- `src/hlt_classification/luka_embedding_aux/{targets,preflight}.py`: unchanged
  TRAIN-bank validation and native/auxiliary parity.
- `src/hlt_classification/luka_attention/worker.py`: HLT views and execution binding.
- `src/hlt_classification/luka_attention_weight/source.py`: isolated original-source replay.
- `src/hlt_classification/context_fusion/worker.py`: atomic tensor publication.
- `src/hlt_classification/data/cache_contracts.py`,
  `src/hlt_classification/cms_proxy_ladder/{contracts,submission}.py`, and
  `src/hlt_classification/provenance.py`: hashed artifacts and pinned source.
- `src/hlt_classification/jetclass2_delphes/{model,runner,reporting,acceptance,execution,dzfix_fusion_model}.py`:
  native inference, reporting/selection, installed-Weaver checks and GPU identity.
- `src/hlt_classification/luka_fullsim/{foundation,regular_cache,preflight_v2,science_campaign,regular_submission}.py`:
  source/data authentication, phase/resource accounting and scheduler site checks.
- `src/hlt_classification/scouting/{hcwdl_exact_dag_submission,hcwdl_recovery}.py`:
  exact-ID submission journal and one-shot claims.
- `scripts/{luka_meta_embedding.py,queue_luka_meta_embedding.sh}`,
  `sbatch/run_luka_meta_embedding.sh`: thin entrypoint and detached queue patterns.
- `tests/test_luka_meta_embedding.py`, `tests/test_luka_embedding_aux.py`, and
  `tests/test_luka_attention.py`: regression and synthetic installed-Weaver fixtures.

New scientific authority is `docs/plans/LUKA_META_GUARD_PLAN.md`; new artifacts use
`LUKA_META_GUARD_<KIND>/v1` and do not masquerade as original meta-embedding results.

## Oscar FullStat U100-w5 meta-embedding continuation (2026-10-29)

Internal donor/compatibility baseline:
`7a6dc268800b0788e0e89d7ec56db37583397946`. Original completed U100-w5
artifact/source pin: `7ea8b0c693734a26ced3820354e047f4af26c979`.
Destination: additive `src/hlt_classification/luka_fullstat_meta_embedding/`,
matching CLI, manual/detached queue helpers, Oscar worker, two focused test
files, [plan](plans/LUKA_FULLSTAT_META_EMBEDDING_PLAN.md),
[v1 contracts](contracts/LUKA_FULLSTAT_META_EMBEDDING.md) and
[runbook](LUKA_FULLSTAT_META_EMBEDDING_RUNBOOK.md).
No external code migration, Fresh_check runtime import or historical runtime
edit. Existing unrelated documentation changes are preserved.

Donor files, relative to the repository root:

- `src/hlt_classification/luka_meta_embedding/{model,meta}.py`: imported
  unchanged native student/predictor, bounded target adapter, objective,
  functional five-step AdamW and isolated higher-order query feedback.
  Its `{training,worker,campaign,submission,source,contracts}.py` are adapted,
  not relabelled: original COARSE_U100 bank, FullStat 20-epoch recipe and
  optimizer, exact 3M/500k population, new 95/5 TRAIN partition identity,
  Oscar admission and matched fixed/meta output contracts. The large-population
  hot sampler avoids a repeated full-TRAIN overlap sort at each meta event.
- `src/hlt_classification/luka_fullstat_neighborhood/{source,execution,launch}.py`,
  `scripts/{luka_fullstat_neighborhood.py,launch_luka_fullstat_neighborhood.sh,queue_luka_fullstat_neighborhood.sh}`,
  `sbatch/run_luka_fullstat_neighborhood.sh`: adapted additive source guard,
  runtime/allocation checks, thin CLI and bounded detached dry/live submission.
  The neighborhood loss, teacher roster and fine-tail semantics are NOT reused.
- `src/hlt_classification/luka_fullstat_u100_embedding/{campaign,worker,submission,banks,contracts,source}.py`:
  unchanged original-source completion replay, exact teacher/target bank and
  role-safe HLT loader. The original selected native checkpoint is restored;
  its missing predictor is calibrated once and shared by both arms. The original
  TRAIN target bank is read-only, not exported again or renormalized.
- `src/hlt_classification/luka_fullstat/{recipe,submission}.py`,
  `src/hlt_classification/luka_fullstat_attention/{worker,contracts}.py`,
  `src/hlt_classification/luka_fullstat_multiteacher/{campaign,source}.py`,
  `sbatch/luka_splithead_finalize_common.sh`: unchanged FullStat scheduler,
  counts/flags, cache authentication, protected paths, enumerated unrelated
  SPORC exceptions and repaired Oscar environment setup.
- `src/hlt_classification/luka_embedding_aux/preflight.py`,
  `src/hlt_classification/luka_embedding_weight/worker.py`,
  `src/hlt_classification/jetclass2_delphes/{model,runner,reporting,acceptance,execution,dzfix_fusion_model}.py`,
  `src/hlt_classification/context_fusion/worker.py`: unchanged installed-Weaver
  and auxiliary parity, native factory/prediction/selection/reporting and
  atomic state publication. No new inference-time architecture.
- `src/hlt_classification/luka_fullsim/{foundation,preflight_v2,science_campaign}.py`,
  `src/hlt_classification/scouting/{hcwdl_exact_dag_submission,hcwdl_recovery}.py`,
  `src/hlt_classification/data/cache_contracts.py`,
  `src/hlt_classification/cms_proxy_ladder/{contracts,submission}.py`:
  unchanged source snapshots, measurements, checksums, immutable artifacts
  and exact scheduler journal primitives.
- `tests/test_luka_meta_embedding.py`,
  `tests/test_luka_fullstat_neighborhood_launch.py`,
  `tests/test_luka_fullstat_u100_embedding.py`,
  `tests/test_luka_embedding_aux.py`, `tests/test_luka_attention.py`,
  `tests/test_luka_u100_pretrain.py`: adapted/reused fixtures, production-worker
  miniature, installed-Weaver checks and detached/failure-injection patterns.

## Student-benefit OFFLINE-w5 continuation (2026-10-28)

Internal donor/compatibility pin: `830a2afb2b3350c14ae147d915b8d3072701269c`.
Destination: additive `src/hlt_classification/luka_meta_embedding/`, its CLI,
debug worker/queue helper, focused tests, plan, v1 contracts and runbook.
No existing runtime changed; no Fresh_check import or external code migration.
Functional five-step AdamW and bounded teacher-side adaptation are new code.

Adapted donor files (paths below are relative to repository root):

- `src/hlt_classification/luka_ce_warmstart/{campaign,source,submission,worker,training,contracts}.py`:
  immutable matched-continuation protocol, fresh low-LR AdamW, exact setup DAG,
  resource evidence, selection and native readback. The new initialization is
  OFFLINE-w5, not CE; target is OFFLINE, not U100; new TRAIN-only meta algorithm.
- `scripts/luka_ce_warmstart.py`, `scripts/queue_luka_ce_warmstart.sh`,
  `sbatch/run_luka_ce_warmstart.sh`, `sbatch/common.sh`: thin authenticated interface, environment
  setup and detached authorized submission.
- `src/hlt_classification/luka_embedding_aux/{model,source,targets,preflight,contracts}.py`:
  native128-wide tap/predictor, original fixed30 replay, exact frozen OFFLINE
  TRAIN bank and normalization, input reconstruction, parity and count contracts.
- `src/hlt_classification/luka_embedding_sweep/{debug,worker,contracts,campaign}.py`:
  inspected original-w5 receipt/fit semantics; narrow original-source replay
  does not require retired bank payloads or unrelated sweep results.
- `src/hlt_classification/luka_attention/worker.py`,
  `src/hlt_classification/luka_attention_weight/source.py`,
  `src/hlt_classification/luka_embedding_weight/worker.py`:
  authenticated D000 loader/allocation, original-source subprocess and prediction
  validation. The old misleading2GiB storage message is NOT reused.
- `src/hlt_classification/jetclass2_delphes/{model,runner,reporting,execution,acceptance,dzfix_fusion_model,batch128_30e_recipe}.py`,
  `src/hlt_classification/context_fusion/worker.py`: unchanged native factory,
  prediction/selection/metrics, environment/GPU checks, parity, schedule and
  atomic checkpoint publication.
- `src/hlt_classification/luka_fullsim/{alljets_data,regular_cache,preflight_v2,science_campaign,regular_submission,foundation}.py`,
  `src/hlt_classification/scouting/{hcwdl_exact_dag_submission,hcwdl_recovery}.py`,
  `src/hlt_classification/data/cache_contracts.py`,
  `src/hlt_classification/cms_proxy_ladder/{contracts,submission}.py`,
  `src/hlt_classification/provenance.py`: shared immutable provenance, hashing,
  storage, phase measurement and exact submission primitives (not copied/edited).
- `tests/test_luka_ce_warmstart.py`, `tests/test_luka_embedding_aux.py`,
  `tests/test_luka_embedding_sweep.py`, `tests/test_luka_embedding_sweep_debug.py`,
  `tests/test_luka_u100_pretrain.py`, `tests/test_luka_attention.py`:
  synthetic cache/real-Weaver fixtures and failure-injection patterns.

Original artifact worktree pins remain fixed30
`74aa58c582ff6f497a2c6d91c503f5489fac3e9a` and debug w5
`855c4f8d40b4db7837238c7ac6a3799597466e85`; files are replayed there,
not rewritten to look native to the new campaign.

## FullStat smooth-w10 Oscar port (2026-10-28)

Authority: [active plan](plans/LUKA_FULLSTAT_NEIGHBORHOOD_PLAN.md) and
[v1 contracts](contracts/LUKA_FULLSTAT_NEIGHBORHOOD.md).
Internal donor/compatibility baseline: `62f9773aabd150004be9c319d6ed06c412fcc57a`.
Destination: new `src/hlt_classification/luka_fullstat_neighborhood/`,
matching CLI, detached/manual queue helpers, worker, two focused test files,
plan/contracts/runbook. No existing runtime or external donor is modified;
no Fresh_check runtime import. Original-source ancestry still requires the
completed split-head finalizer donor `420d22f8feb9f2237ee2dc2cbf3000d881ecd6a6`.

This port adapts the fine-tail source/queue/fit/publication scaffolding but
NOT its seven-teacher recipe or D022/D011 views. Smooth descriptors, loss and
native block8 tap retain the earlier neighborhood semantics; only one
FP32 smooth bank is built. FullStat role-safe source extraction and native
D000 replay replace the smaller-trial reader. Listed files include adapted
templates/tests and directly reused kernels/bank validators.

| Donor file at the above commit | SHA256 of inspected bytes |
| --- | --- |
| `src/hlt_classification/luka_neighborhood/targets.py` | `02ea5cd24f257eca9a7007a5990e09f2ee58d871c9628ced2d9f633c470bb5f1` |
| `src/hlt_classification/luka_neighborhood/banks.py` | `2231db21cd999ddb9027b7690b608484fb2514620cd3883251560a4170270fa9` |
| `src/hlt_classification/luka_neighborhood/model.py` | `f34b591bad7799731ae484cfbeabc03b4b25cb597f7216967969c184e6411255` |
| `src/hlt_classification/luka_neighborhood/contracts.py` | `14a1f19d5de1c3bd89bb3592a8618cb1715a10baa071324985d31c7bf7ab0396` |
| `src/hlt_classification/luka_fullstat_fine_tail/contracts.py` | `43e30599de99fd2d46c8d3e4f3c289aaa7a5a487af307cb2adc2173df7c62493` |
| `src/hlt_classification/luka_fullstat_fine_tail/source.py` | `772627ef37eeadaed5ada37c15ab9b517c566a40a8746de6c3e6c156537b473d` |
| `src/hlt_classification/luka_fullstat_fine_tail/campaign.py` | `3c09bff629ed95ee3b51c525b587e0e059bff6272f6ed509835c5824aefa7112` |
| `src/hlt_classification/luka_fullstat_fine_tail/training.py` | `6cd17fd2ad11887bcbe2ff8eb7f2197d7f66199411bac554ec853fe5fb16340e` |
| `src/hlt_classification/luka_fullstat_fine_tail/model.py` | `330fdd23e9ddd4bb53f9d91643b61327e1acf8d2110aaaed814fbf65a092c9a2` |
| `src/hlt_classification/luka_fullstat_fine_tail/worker.py` | `6f6610aa294977e9b70e1107bb6f02d7fa8fa0f7a650efff0f8cd915bcb7ed6a` |
| `src/hlt_classification/luka_fullstat_fine_tail/submission.py` | `c17be4a56c023026ed42be25d4ce16aef45fc7cbfeaa2093ea6785e07dceddfe` |
| `src/hlt_classification/luka_fullstat_fine_tail/execution.py` | `1a7119fbb6daf02fc080dddbf7f2c8800945d8c6a4dc8ce7f20baa857997734c` |
| `src/hlt_classification/luka_fullstat_fine_tail/launch.py` | `22ec9f3a9031899028bce9c849eaaefa3c4d5ab469a7cf6eff9640a7e30ecfff` |
| `src/hlt_classification/luka_fullstat_fine_tail/views.py` | `dae1cd28399fe0185cd57ca68e29590015497a8fc99d7383469af278ebed52ab` |
| `src/hlt_classification/luka_fullstat_fine_tail/banks.py` | `7354c305f63953ac45913478c9e378a94ee2a2bd502626dafd998dd7a91e4373` |
| `src/hlt_classification/luka_fullstat_fine_tail/kd_banks.py` | `975c5801109c4bb73046710ecf66ce0a0fd83b0d9130ed7464c4033a2e4884ab` |
| `scripts/launch_luka_fullstat_fine_tail.sh` | `568aa16832cb48f9688cb93b2bad5b64316dd02056a7068eaabcc5945487d105` |
| `scripts/queue_luka_fullstat_fine_tail.sh` | `c7b12f8800a90611ea394c7a758a9616b68c6757e679ea8fa18972743a93d938` |
| `scripts/luka_fullstat_fine_tail.py` | `76bfe0d6027d2f6899f53f3ffdc267d1136b4cece1f38b627addaf25f876e738` |
| `sbatch/run_luka_fullstat_fine_tail.sh` | `5654a4794581788e932afd4d79a2961fa98e1b6e466d4d516f220b0d2c424fe2` |
| `sbatch/luka_splithead_finalize_common.sh` | `4a32996744bef1dd422b545cf728052bb21c4f252218a32072a486fc90aa8010` |
| `src/hlt_classification/luka_supervision_depth/model.py` | `7a0ef3824d3fcb818a6cfb5cc9f0878f3394cc4232e927b7a9a146ee737dc3ca` |
| `src/hlt_classification/luka_fullstat/cache.py` | `bdc12ca26dd3a9564e098800ac0fabef396a55133ff0d2d9561e08cd0aac9195` |
| `tests/test_luka_neighborhood.py` | `5cc212c757854e39bc8435b3ab73c7ef48b46aa2da95ab9db73b7d6e95bf9651` |
| `tests/test_luka_fullstat_fine_tail.py` | `b5cc327f16ef63669b09942a969bf9ed2a3157f1c98f0586dff1e53de759cc83` |
| `tests/test_luka_fullstat_fine_tail_launch.py` | `be22d36df8047cf3297ed3741976b1d3a9cfa16e6b324ab9900b1993e647a42e` |
| `tests/test_luka_fullstat.py` | `1128e7f3e86962b5a1062a11d4dc08d133420bdf8c7eeeba00af40c0c02df2ac` |
| `tests/test_luka_embedding_aux.py` | `6cdfd74767d85b1be24f61f5327652f284855255b28b3b09fb9fd33acc0df4f1` |


## Luka offline-neighborhood screen (2026-10-28)

Authority: [plan](plans/LUKA_NEIGHBORHOOD_PLAN.md), [v1 contracts](contracts/LUKA_NEIGHBORHOOD.md).
Internal donor commit: `5d55d602549ff47c6301e74b712786650c3bf8f4`.
Destination: `src/hlt_classification/luka_neighborhood/`, corresponding CLI,
queue/Slurm scripts, focused tests and documentation. All historical runtime
bytes remain unchanged; no external donor or Fresh_check runtime import.

Adapted donor files (SHA256 of inspected local bytes):

| Donor | SHA256 | Destination / retained meaning |
| --- | --- | --- |
| `luka_supervision_depth/training.py` | `95f8592e6d1a813325e3147ea303099e5d916c265950ef18defb9b8d1b538bab` | new `training.py`: fixed30, optimizer/sampler/selection, sorted loss accounting |
| `luka_supervision_depth/startup.py` | `f5dc6984847f1f8c157070249a4cff64bddc22363c79e799f6e786a3780b17fa` | new `startup.py`: discarded native/gradient/max-support checks; new population timing admission |
| `luka_supervision_depth/worker.py` | `32b20b5480244d03fe824ca1f9c6c652215a44d9a633963c99e84b837df6c945` | new `worker.py`: exact allocation, reference copying, native publication/readback |
| `luka_depth_recovery/submission.py` | `06947a12822ed4e64c3357795a6b991459f69587f5f6e7435849ec4ae3730cfd` | new `submission.py`: exact-ID journals and two failure-isolated GPU lanes |
| `scripts/queue_luka_depth_recovery.sh` | `0f6f278378bf951195281f34aec29fa5bb34ed11160eb394a134c29fae1b18e3` | new queue helper: pinned source, reviewed plan and detached submitter patterns |
| `sbatch/run_luka_depth_recovery.sh` | `234f78492a8fb199e3f12c942f66a0ed10a179559756f4d94002ee417f889566` | new Slurm wrapper: absolute helpers and isolated conda/runtime environment |

Package donor paths above are relative to `src/hlt_classification/`.
Unchanged runtime API reuse at the same donor commit:

- `luka_supervision_depth/{contracts,model,data}.py` and `training.objective`:
  exact CE0.5/K1/E40 baseline, block8 tap and authenticated D000 population.
- `luka_auxiliary_grid/{model,training}.py`,
  `jetclass2_delphes/{model,runner,reporting,execution,batch128_30e_recipe,dzfix_fusion_model}.py`:
  existing independent jet/KD heads, optimizer, native ParT and parity/metrics.
- `luka_trial_hybrid/{banks,worker}.py`, `luka_attention/{banks,worker}.py`,
  `luka_embedding_weight/worker.py`: borrowed compact targets, native model
  loading, population identity/counts, allocation and storage guards.
- `luka_fullsim/{particles,stage2,regular_cache,alljets_data,preflight_v2,foundation,
  preflight_reuse,regular_submission}.py`, `luka_attention_weight/source.py`,
  `provenance.py`: physical conversion, selected-row replay, source authentication,
  environmental acceptance and bounded memory measurement.
- `data/cache_contracts.py`, `cms_proxy_ladder/{contracts,submission}.py`,
  `context_fusion/worker.py`, `scouting/{hcwdl_exact_dag_submission,hcwdl_recovery}.py`:
  hashes, immutable publication, native checkpoints and exact scheduler ledger.
- `tests/test_luka_supervision_depth.py`, `tests/test_luka_fullsim_stage2.py`:
  reused synthetic science/ROOT fixtures, not research-compute data.

New semantics: three ragged physical neighborhood banks, explicit TRAIN-only
36-channel summaries, removable block8 head, fourteen registered auxiliary
weights, per-channel diagnostics and separately authorized owned-bank retirement.
No new teachers, baseline retraining or standalone preflight. Zero-new-loss
fixed30 parity and all-fourteen-worker native readback are tested locally;
see HANDOFF for final evidence and remote-validation limitations. No third-party
source/license changes; installed Weaver remains an external runtime dependency.

## Luka depth interrupted-fit restart amendment (2026-10-28)

Operational donor: `8a80162f210437fe6c21d157543c57008887eeed`, existing
`luka_depth_recovery/{campaign,contracts,source,submission,worker}.py`, CLI,
queue wrapper and focused tests. Added `interrupted.py` for separately reviewed
set8_p3 archive/restart-zero; recovery contracts are v2. Original scientific
`luka_supervision_depth` runtime and its donor boundary remain byte-identical.
No external donor/Fresh_check import, data, objective, optimizer or seed change.

## Luka depth-screen parallel recovery (2026-10-28)

Authority: [plan](plans/LUKA_DEPTH_RECOVERY_PLAN.md),
[contracts](contracts/LUKA_DEPTH_RECOVERY.md). Donor commit:
`04d1f0f5945ab9af38d83e192eff48e124ad8dc5`. All donor runtime bytes remain unchanged.
New destination: `src/hlt_classification/luka_depth_recovery/`, CLI, shell wrappers,
focused tests and documentation.

- `luka_supervision_depth/{contracts,source,campaign,submission,worker,banks,
  training,startup,data,model}.py`: original recipes, native training/publication,
  target loading/export, per-fit scratch, metrics and validators reused unchanged.
  Scientific worker dispatch/results adapted into separate recovery execution;
  original serial dependency and receipt semantics are not patched/relabelled.
- `scouting/{hcwdl_exact_dag_submission,hcwdl_recovery}.py`: unchanged exact-ID
  submission journal and dry/live ledger; new mixed afterok/afterany command plan.
- `luka_attention_weight/source.py`, `luka_fullsim/{foundation,preflight_reuse,
  regular_submission}.py`, `cms_proxy_ladder/contracts.py`: original-source replay,
  source byte equality, resource checks and file references reused unchanged.
- `luka_joint_prediction/worker.py`: canonical node-local scratch validation;
  additional conservative process-locked reservations are new operational code.

No external donor or Fresh_check import; no scientific objective/seed/split change.

## FullStat K1/E40 plain exact-ninths fine tail (2026-10-27)

Authority: [plan](plans/LUKA_FULLSTAT_FINE_TAIL_PLAN.md),
[v1 contracts](contracts/LUKA_FULLSTAT_FINE_TAIL.md). Repository donors were
inspected at `9e90b7dee50af4f122a60054031c7fcc255e4e08`; none was modified.
New destinations: `src/hlt_classification/luka_fullstat_fine_tail/`, its CLI,
queue/detached-launch/Slurm wrappers, two test modules and plan/contract/runbook.

- `luka_fullstat_relational_ladder/{contracts,source,campaign,banks,kd_banks,
  model,worker,execution,submission,launch}.py`: adapted exact-source replay,
  replacement Oscar runtime checks, independent auxiliary heads, teacher bank
  imports, native publication, timing and detached three-fit submission. New
  v1 identities and five/six/seven teacher sets; relation code is not imported
  or used. `luka_fullstat_splithead/training.py` supplies the plain fixed20
  optimizer/fit/report path, with total embedding40 and KD1.
- `cms_proxy_ladder/views.py`: D-only physical interpolation kernel adapted
  for exact Fraction(2,9)/Fraction(1,9), preserving switch hash/groups and
  validity-aware tracking. Old global coordinate parser remains unchanged.
  `scouting/hcwdl_fullcard_bottleneck_matcher.py` remains the validator of
  existing assignments, not a rerun of particle matching.
- `luka_fullstat/{cache,data,contracts,recipe,submission}.py`,
  `luka_fullsim/{particles,preflight_cache,preflight_v2,foundation}.py`,
  `literature_context/transform.py`, `jetclass2_delphes/{cache,model,runner,
  reporting,execution,dzfix_fusion_model}.py`: original authenticated
  memberships/source paths/role-aware ROOT reads, RamCache, physical inputs,
  native model/parity/selection, resource evidence and schedule. New RAM views
  replay original D000 bytes before acceptance and never request test rows.
- `luka_fullstat_embedding_sweep/banks.py`,
  `luka_fullstat_hybrid_embedding/{banks,kd_banks}.py`,
  `luka_fullstat_splithead/{banks,kd_banks,campaign,worker,submission}.py`:
  original target contracts and completed parent ancestry are reused unchanged.
  `luka_fullstat_attention/worker.py` supplies original cache metadata and
  validation identities. `luka_fullstat_multiteacher/campaign.py` protects
  existing output/source roots from overwrite.
- `luka_embedding_aux/{contracts,model,targets}.py`: independent predictor RNG,
  native jet representation and TRAIN-only normalization. `data/cache_contracts.py`,
  `cms_proxy_ladder/{contracts,campaign,submission}.py`, `context_fusion/worker.py`,
  `scouting/{hcwdl_exact_dag_submission,hcwdl_recovery}.py`: content/file hashing,
  coordinate seeds, atomic native publication and exclusive exact-ID journals.
- `scripts/{luka_fullstat_relational_ladder.py,queue_luka_fullstat_relational_ladder.sh,
  launch_luka_fullstat_relational_ladder.sh}` and
  `sbatch/{run_luka_fullstat_relational_ladder.sh,luka_splithead_finalize_common.sh}`:
  thin additive wrappers and inherited verified environment activation.
- `tests/{test_luka_fullstat_relational_ladder,test_luka_fullstat_relational_launch,
  test_luka_fullstat_splithead,test_luka_fullstat_splithead_finalize,
  test_luka_fullstat,test_luka_embedding_aux}.py`: synthetic physical/ROOT/cache,
  installed-Weaver, finalizer-replay and detached launcher fixtures. Remote
  scheduler/environment are mocked in local miniature tests.

The historical `420d22f8feb9f2237ee2dc2cbf3000d881ecd6a6` ancestry boundary in
source authentication is retained separately from this adaptation donor.
Execution accepts only the completed non-relational finalization/retry parent;
current relational models and trial-screen teachers are not substituted.

## Luka supervision depth / OFFLINE diversity screen (2026-10-27)

Authority: [plan](plans/LUKA_SUPERVISION_DEPTH_PLAN.md) and
[v1 contract](contracts/LUKA_SUPERVISION_DEPTH.md). All repository donors below
were inspected at `e21aa91463a8f313bf51e36d6bcac97ee7942bfb`; none was edited.
New destinations are `src/hlt_classification/luka_supervision_depth/`, its CLI,
queue/worker wrappers, focused tests and documentation.

- `luka_auxiliary_grid/{contracts,model,training,banks,campaign,source,submission,
  startup,worker}.py`: original K1/E40 objective, RNG-isolated auxiliary heads,
  optimizer, fresh-fit selection, compact teacher banks, exact dry-plan/journal
  mechanics and native-checkpoint validation. Adapted under new v1 identities;
  fixed baseline is checked against the donor over a full 30-epoch miniature.
- `luka_joint_prediction/{banks,worker}.py` and
  `luka_particle_prediction/model.py`: joint jet/token objective conventions,
  native particle taps, exact assignment and bounded node-local scratch ownership.
  New multi-depth code shares one Hungarian permutation across depths and retains
  per-depth TRAIN-only moments. It does not reuse old COARSE_U100 token payloads.
- `luka_trial_hybrid/{banks,worker}.py`,
  `luka_embedding_aux/{targets,model}.py`,
  `luka_embedding_prediction/data.py`, and `luka_attention/{banks,worker}.py`:
  unchanged c50 teacher authentication, native jet representations, frozen target
  formats, slot alignment, source/environment checks and validated cache access.
- `jetclass2_delphes/{model,runner,reporting,batch128_30e_recipe,
  dzfix_fusion_model,execution}.py`, `luka_fullsim/{regular_cache,stage2,particles,
  preflight_cache,preflight_v2,foundation,preflight_reuse}.py`,
  `luka_embedding_weight/worker.py`, `luka_attention_weight/source.py`,
  `context_fusion/worker.py`, `data/cache_contracts.py`,
  `cms_proxy_ladder/contracts.py`, and `provenance.py`: reused native model,
  deterministic schedule, metrics, input reconstruction/fingerprints, memory
  evidence, installed-source replay, atomic publication and file/content hashes.
- `luka_fullsim/regular_submission.py`, `cms_proxy_ladder/submission.py`, and
  `scouting/{hcwdl_exact_dag_submission,hcwdl_recovery}.py`: unchanged site checks,
  clean worker environment, time formatting and journaled exact-ID submission.
- `scripts/{luka_auxiliary_grid.py,queue_luka_auxiliary_grid.sh}` and
  `sbatch/{run_luka_auxiliary_grid.sh,common.sh}`: thin entry points, explicit
  approval and absolute-project worker activation. New wrapper queues 25 exact
  debug jobs and no standalone preflight; does not mutate any old campaign.
- `tests/{test_luka_auxiliary_grid,test_luka_joint_prediction,
  test_luka_particle_prediction,test_luka_trial_hybrid,test_luka_u100_pretrain}.py`:
  inspected loss/bank/parity and small-cache fixtures; existing fixture functions
  are reused for the synthetic production-worker miniature, not production data.

The paired CE / CE+0.1 supervised-contrastive OFFLINE controls, registered twenty
student recipes, common multi-depth assignment, exact replacement-head control,
new target/retirement inventories and associated tests are new local work, not
an external code migration. The projection is discarded; native teacher jet
embeddings remain the targets. No Fresh_check runtime import or external donor.

Source compatibility explicitly permits only the already-committed four
`luka_fullstat_splithead/{campaign,kd_banks,submission,worker}.py` changes between
the completed grid's `028c3314394949706787c6d1ef5e632f53ca157c` source and the
e21aa914 boundary. Those files are not trial runtime dependencies; original grid
authentication runs in its own old worktree. All other parent runtime bytes must
match, and additional changes to existing runtime are rejected.

## Relational K2.5/E10 recipe revision (2026-10-26)

The user-selected screening setting changes the pending local relational
ladder's total auxiliary-logit weight from 5 to 2.5 only. No new donor file
was migrated. Existing scientific donors at
028c3314394949706787c6d1ef5e632f53ca157c and execution boundary
420d22f8feb9f2237ee2dc2cbf3000d881ecd6a6 remain as documented below; the
prior local relational implementation was untracked, not a fabricated commit.

Changed local relational contracts/execution/labels, authorization wrappers,
tests and documentation. RECIPE v2 and SPEC/PLAN/SUBMISSION_CLAIM v3 prevent
mixing with K5. Native bank formats, source replay, detached launch mechanics,
teacher graph and all other scientific coefficients/schedules are unchanged.
No external code, Fresh_check import or changes to donor/recovery files.

## Relational ladder execution-v2 and detached launch (2026-10-26)

Authority: [updated plan](plans/LUKA_FULLSTAT_RELATIONAL_LADDER_PLAN.md) and
[contract](contracts/LUKA_FULLSTAT_RELATIONAL_LADDER.md). Scientific code retains
its original028c3314 donor map below. Execution donor/baseline is
420d22f8feb9f2237ee2dc2cbf3000d881ecd6a6.

- jetclass2_delphes/execution.py::allocation and
  luka_fullstat_attention/worker.py::execution -> new relational execution.py:
  original exact Oscar/resource/GPU/backend guards with explicitly registered
  replacement prefix. No edits to donors or global site profiles.
- luka_fullstat_splithead/{finalize,finalize_source}.py and the existing
  relational source.py -> updated original-source replay of completed
  finalization/retry, imported-fit refs and successful new receipts. The
  separately implemented retry's paths/contracts were inspected, not edited or
  copied into this patch. Every parent runtime file must match its pinned bytes.
- sbatch/luka_splithead_finalize_common.sh is reused unchanged by new queue
  and worker paths, activating runtime_repair_v1. Parent/source checks bind it.
- Existing relational submission.py and shared HCWDL exact-DAG journal
  semantics (module scouting/hcwdl_exact_dag_submission.py) -> new launch.py
  orchestration, not a copied/changed submission engine. New shell launcher
  handles exclusive log directory, detached session and independent tail.
- tests/test_luka_fullstat_splithead_finalize.py fixtures at420d22f8 -> new
  finalizer-to-relational replay integration. Existing relational tests are
  extended; new test_luka_fullstat_relational_launch.py covers allocation,
  intent/dry/live boundaries and detached-shell behavior.

REUSE/SPEC/PLAN/SUBMISSION_CLAIM version2; new LAUNCH_* version1. Scientific
recipes/banks/fit artifacts unchanged. No external donor or Fresh_check import.
No finalization/retry files, old source trees, jobs or user artifacts modified.

## FullStat finalizer runtime-path correction and explicit retry (2026-10-26)

Authority: [retry contract](contracts/LUKA_FULLSTAT_SPLITHEAD_FINALIZE_RETRY.md).
Inspected donor commit: `420d22f8feb9f2237ee2dc2cbf3000d881ecd6a6`, the failed
finalizer execution. Scientific source boundary remains the original
`028c3314394949706787c6d1ef5e632f53ca157c`; old recovery source remains pinned
at `0c4a71a55e8bc163ea3c4805b72f06d18e640cb5`.

- `jetclass2_delphes/execution.py::allocation` and
  `luka_fullstat_attention/worker.py::execution` -> new
  `luka_fullstat_splithead/finalize_execution.py`: retain scheduler, site,
  resource, GPU and environment guards with the explicitly registered repaired
  runtime prefix. Original donor modules and global profiles are unchanged.
- `luka_fullstat_splithead/{finalize,finalize_source}.py` original-source
  replay, accounting and evidence patterns -> new `finalize_retry.py` and
  scoped retry branches in those existing files. Failed attempts and consumed
  claims are preserved; no retraining of imported ancestors.
- `scripts/queue_luka_fullstat_splithead_finalize.sh` -> new retry helper;
  existing finalizer CLI gains `create-retry`. Existing worker, campaign,
  submission and bank locator dispatch also recognize the retry contract.
- `tests/test_luka_fullstat_splithead_finalize.py` fixtures -> new retry tests,
  including actual allocation dispatch and an installed-Weaver GPU miniature.
  The old mocked-allocation coverage gap is explicitly tested.

New retry SPEC/EVIDENCE/PLAN/CLAIM v1 only; original fits, banks and scientific
recipe retain their existing semantics and identities. No external donor or
Fresh_check runtime import. Unrelated relational-ladder files are not part
of this source boundary.

## FullStat DE033 post-export finalization (2026-10-26)

Authority: [execution contract](contracts/LUKA_FULLSTAT_SPLITHEAD_FINALIZE.md),
extending the unchanged split-head scientific plan. Inspected donor/source
boundary: `028c3314394949706787c6d1ef5e632f53ca157c`; the failed execution is
pinned at `0c4a71a55e8bc163ea3c4805b72f06d18e640cb5`.

- `luka_fullstat_splithead/recovery.py` -> `finalize.py`: saved-fit readback,
  failed-job accounting, exclusive cross-root claim and explicit new provenance.
  This finalizer verifies EXISTING DE033 exports rather than generating them.
- `recovery_source.py` -> `finalize_source.py`: exact original-source replay,
  allowlisted source boundary, now requiring the verified replacement prefix.
- `scripts/{luka_fullstat_splithead_recovery.py,
  queue_luka_fullstat_splithead_recovery.sh}` and
  `sbatch/run_luka_fullstat_splithead_recovery.sh` -> finalization-named wrappers.
- `sbatch/jetclass2_delphes_common.sh` -> dedicated
  `sbatch/luka_splithead_finalize_common.sh`: environment sanitation/activation,
  with only the new wrappers targeting runtime_repair_v1. Unchanged common.sh
  provides hlt_activate. No old pinned worktree/environment/path edits.
- `tests/test_luka_fullstat_splithead_recovery.py` ->
  `tests/test_luka_fullstat_splithead_finalize.py`: synthetic fixtures,
  corruption/accounting/duplicate guards, installed-Weaver worker integration.
- Existing `campaign.py`, `worker.py`, `submission.py`, `kd_banks.py` receive
  contract-scoped dispatch/locator additions only, plus a scoped output-quota
  minimum. Training/model/recipe remain unchanged; previous plans keep hashes.

New LUKA_FULLSTAT_SPLITHEAD_FINALIZE_* v1 only. Original native fits/banks keep
their existing contracts and hashes. No external donor or Fresh_check import.
Unrelated relational-ladder work in this worktree is preserved, not included
in this finalization source boundary.

## FullStat four-rung K5/E10/relational ladder (2026-10-26)

Authority: [active plan](plans/LUKA_FULLSTAT_RELATIONAL_LADDER_PLAN.md).
Destination `src/hlt_classification/luka_fullstat_relational_ladder/`, additive
LUKA_FULLSTAT_RELATIONAL_LADDER_<kind>/v1. Source boundary and inspected donor
snapshot: `028c3314394949706787c6d1ef5e632f53ca157c`. Existing runtime files,
contracts, pinned worktrees and remote artifacts are unchanged. No Fresh_check
runtime dependency or external donor source.

Copied/adapted from that snapshot:

- `luka_fullstat_splithead/{contracts,source,campaign,banks,kd_banks,model,
  training,worker,submission}.py`: native auxiliary heads, fixed20 loop,
  selected native export/readback, bank joins, completed ancestry, integrated
  checks and exact four-job submission. Expanded with fresh UE100, K5/E10,
  relational loss/ramp, short timing checks and new semantic namespace.
- `scripts/luka_fullstat_splithead.py`,
  `scripts/queue_luka_fullstat_splithead.sh`,
  `sbatch/run_luka_fullstat_splithead.sh`: new thin wrappers, explicit four-fit
  authorization and independent job prefix. No recovery/cancellation operation.
- `tests/test_luka_fullstat_splithead.py`: fixture/replay/DAG/native-worker test
  patterns. Tests reuse its synthetic fixtures and lower-level fixtures; the
  production path never imports test helpers.

Imported unchanged at that snapshot:

- `luka_multiteacher_trial/training.py`: ONLY AnchorSampler/reference_kl pure
  kernels (2048 balanced references, detached FP32 cosine-softmax forward KL).
  Its trial30 training function/recipe is never invoked by this new campaign.
- `luka_fullstat/recipe.py`: batch128 fixed20 optimizer/schedule.
- `luka_fullstat_hybrid_embedding/{banks,kd_banks}.py`: original OFFLINE/M0HLT
  embedding and OFFLINE probability validation; unchanged imported contracts.
- `luka_fullstat_splithead/kd_banks.py`: original M0HLT KD parents/contract
  validation; new adapter explicitly extends consumer coordinates to U100.
- `luka_fullstat_attention/worker.py`: accepted execution/cache metadata and
  ordinary-validation identity readers; `luka_fullstat/cache.py` ordinary views.
- Shared `jetclass2_delphes/{model,runner,reporting,dzfix_fusion_model}.py`,
  `luka_embedding_aux/{model,targets,contracts}.py`,
  `luka_fullsim/preflight_v2.py`, `context_fusion/worker.py`,
  `cms_proxy_ladder/{campaign,contracts,submission}.py`,
  `luka_fullstat/submission.py`, `luka_fullstat_multiteacher/campaign.py`,
  `scouting/{hcwdl_exact_dag_submission,hcwdl_recovery}.py`, and
  `data/cache_contracts.py`: established artifact/execution/kernel utilities.

Completed parent execution is separately pinned to
`0c4a71a55e8bc163ea3c4805b72f06d18e640cb5`, replaying its original split-head
campaign/worker/submission and recovery validators. Native OFFLINE/M0HLT targets
come from authenticated original ancestry; old UE100/DE teachers are historical
references only. No old target is silently relabelled as a new ladder stage.

## Trial auxiliary-grid K5 report recovery (2026-10-26)

User-authorized repair of grid creation at
`be6a6248b09efd6c10b3b0d1ac74a9eb75cbfb22`; inherited scientific source boundary
remains `0c4a71a55e8bc163ea3c4805b72f06d18e640cb5`. No existing strong-KD or
c50 runtime file is modified. No Fresh_check import or external donor code.

New `luka_auxiliary_grid/reference_replay.py` factors and extends the prior
grid `reference.py` REPLAY program at be6a6248. It calls original
`luka_auxiliary_logits_strong/{campaign,worker,training,submission,banks}.py`
from execution commit `234b029ec14a8d4861134e098cafbf939d325cd8`, restoring only
the registered dictionary order in memory for the known failed K5 report.
The original validators, receipts and data remain intact. Exact accounting
and checkpoint-readback design follows the local
`luka_fullstat_splithead/recovery.py` at be6a6248; its source is not copied or
imported. New grid readback reuses `luka_trial_hybrid.worker.native` and the
unchanged `jetclass2_delphes.runner.predict` and metrics APIs.

Modified grid modules: reference, contracts, source, submission, worker;
helper: `scripts/queue_luka_auxiliary_grid.sh`. New recovery regression tests
extend the grid and strong-worker fixture helpers at be6a6248. Operational
SPEC/REFERENCE/TASK/PLAN/ACQUISITION v2 and REFERENCE_REPLAY/READBACK v1 are
documented in the amended grid plan/contract/runbook. All20 scientific
RECIPE/v1 hashes are byte-identical to be6a6248. Old failed accounting and
immutable artifacts are preserved; no recovery of K10 is claimed.

## Trial DE000 twenty-arm auxiliary grid (2026-10-26)

Authority: [twenty-fit plan](plans/LUKA_AUXILIARY_GRID_PLAN.md).
Donor and additive-source boundary: `0c4a71a55e8bc163ea3c4805b72f06d18e640cb5`.
Destination: `src/hlt_classification/luka_auxiliary_grid/`, new
`LUKA_AUXILIARY_GRID_<kind>/v1`. Existing scientific runtime, old contracts and
artifacts remain unchanged. No Fresh_check dependency or external donor code.

Copied/adapted files at that commit (destination has the same basename):

- `src/hlt_classification/luka_auxiliary_logits_strong/__init__.py`.
- `src/hlt_classification/luka_auxiliary_logits_strong/contracts.py`: twenty
  recipes, fixed CE0.5, K/E grid, paired-wave graph, new namespace/authorization.
- `src/hlt_classification/luka_auxiliary_logits_strong/model.py`: unchanged
  native encoder and separate removable heads, paired scientific/head RNG.
- `src/hlt_classification/luka_auxiliary_logits_strong/training.py`: unchanged
  native forward, fixed30 optimizer/sampler/selection; new K/E and optional
  relation objective. Report sums explicitly use registered teacher order.
- `src/hlt_classification/luka_auxiliary_logits_strong/banks.py`: exact native
  BANK/v1 target semantics, owned new paths and BUNDLE lineage.
- `src/hlt_classification/luka_auxiliary_logits_strong/source.py`: unchanged
  original c50 replay; new complete/additive source guard and namespace.
- `src/hlt_classification/luka_auxiliary_logits_strong/campaign.py`: original
  acceptance/resources plus frozen K5/E5 reference, twenty fits and 4GiB quota.
- `src/hlt_classification/luka_auxiliary_logits_strong/submission.py`: exact
  journal/dry-review mechanism, twenty-two-job debug DAG and `lukaax` prefix.
- `src/hlt_classification/luka_auxiliary_logits_strong/startup.py`: bounded
  three-corner disposable checks; physical128 and full2048 reference support.
- `src/hlt_classification/luka_auxiliary_logits_strong/worker.py`: native fits,
  reporting and retirement, extended to all20 consumers and optional bank reuse.
- `scripts/luka_auxiliary_logits_strong.py`,
  `scripts/queue_luka_auxiliary_logits_strong.sh`,
  `sbatch/run_luka_auxiliary_logits_strong.sh`: renamed thin wrappers/helper.
- `tests/test_luka_auxiliary_logits_strong.py`: adapted contract, native parity,
  worker, corrupt artifact, submission and retention tests; added relation and
  original-source target reuse/retirement tests in `test_luka_auxiliary_grid.py`.

Imported without copying: `luka_multiteacher_trial/training.py` AnchorSampler
and reference_kl (2048 class-balanced TRAIN anchors, self mask, tau0.2, forward
KL); `luka_trial_hybrid/banks.py` exact teachers/native TRAIN targets; native
model, fixed30 recipe, immutable hashing/publication and exact Slurm DAG helpers.
These reusable imports also refer to the above repository snapshot. Existing
test fixtures from attention/embedding/pretrain/trial-hybrid packages are reused.

New `reference.py` authenticates retained original strong K5/E5 evidence and
optionally copies its four verified banks into independently owned locations.
It derives the replay pattern from the donor source/worker validators; no
original files are altered. Regeneration requires authenticated normal target
retirement, not missing-file inference. The imported original execution is
`234b029ec14a8d4861134e098cafbf939d325cd8`; scientific c50 parent remains
`72490a9d6de4acabe9cbea518c5b100d893e6866`. This is trial, not FullStat.
Validation evidence and limitations are recorded in HANDOFF.

## FullStat split-head saved-fit recovery (2026-10-25)

Authority: [recovery contract](contracts/LUKA_FULLSTAT_SPLITHEAD_RECOVERY.md).
Integration/source-boundary baseline: `234b029ec14a8d4861134e098cafbf939d325cd8`;
failed execution source: `a7efcef6a8399cdcde80b2941e1a21910ba1bf67`.

In-place donors (same paths at both commits):
`src/hlt_classification/luka_fullstat_splithead/training.py` (validator only:
registered-order loss sum), `worker.py` (recovery dispatch/reference reporting),
`campaign.py` (new-spec dispatch), `submission.py` (three-job recovery plan and
claim), `kd_banks.py` (reuse original M0HLT probability-manifest locator).
The training objective/forward/optimizer/sampler/selection/model/recipe are
unchanged. No old artifacts or pinned source trees are rewritten.

New `recovery.py` and `recovery_source.py` reuse that worker's fit, startup,
bank, readback and original-source verification contracts. CLI/queue/Slurm
wrappers derive from `scripts/luka_fullstat_splithead.py`,
`scripts/queue_luka_fullstat_splithead.sh`, and
`sbatch/run_luka_fullstat_splithead.sh` at integration baseline.
Shared exact-DAG journal, file hashing, source capture and atomic publication
remain imported, not copied. No Fresh_check imports or external donor code.

New `LUKA_FULLSTAT_SPLITHEAD_RECOVERY_<kind>/v1` records distinguish successful
export-only recovery from the failed old job; old FIT/report/bank contracts stay
v1 with unchanged scientific meaning. Local evidence is recorded in HANDOFF.

## Trial auxiliary-logit k5/k10 follow-up (2026-10-24)

Additive destination `src/hlt_classification/luka_auxiliary_logits_strong/`;
new `LUKA_AUXILIARY_LOGITS_STRONG_<kind>/v1`. Authority:
[plan](plans/LUKA_AUXILIARY_LOGITS_STRONG_PLAN.md). Donor/source-guard baseline:
`a7efcef6a8399cdcde80b2941e1a21910ba1bf67`. Existing runtime is unchanged;
no Fresh_check imports, old artifact mutation or teacher substitution.

Copied donor files at that commit (one-to-one destination basenames):

- `src/hlt_classification/luka_auxiliary_logits/__init__.py`
- `src/hlt_classification/luka_auxiliary_logits/contracts.py`: new namespace,
  ONLY k5/k10, fixed CE0.5/embedding5, four-task graph, k5 then k10 ordering.
- `src/hlt_classification/luka_auxiliary_logits/model.py`: identical native
  encoder and paired head initialization; only registered recipe lookup differs.
- `src/hlt_classification/luka_auxiliary_logits/training.py`: identical native
  fixed30 optimizer/selection and detached per-teacher losses; new weights and
  report namespace/log prefix. No schedule or inference change.
- `src/hlt_classification/luka_auxiliary_logits/banks.py`: same original native
  BANK/v1 exports/joins; new BUNDLE namespace and independent owned locations.
- `src/hlt_classification/luka_auxiliary_logits/source.py`: same original c50
  replay, with additive implementation source guard pinned to this donor commit.
- `src/hlt_classification/luka_auxiliary_logits/campaign.py`: two fits, same
  inherited debug allocation,3GiB quota and protected parent paths.
- `src/hlt_classification/luka_auxiliary_logits/submission.py`: four-job exact
  journal/dry review, new authorization and lukaas job prefix; no preflight.
- `src/hlt_classification/luka_auxiliary_logits/startup.py`: both high weights,
  CE-only main-head gradient checks, native parity and disposable stress updates.
- `src/hlt_classification/luka_auxiliary_logits/worker.py`: two consumers,
  retained native results and four own-target payload retirement after reporting.
- `scripts/luka_auxiliary_logits.py`, `scripts/queue_luka_auxiliary_logits.sh`,
  `sbatch/run_luka_auxiliary_logits.sh`: thin wrappers adapted to strong namespace,
  two-fit authorization, new root and four-job dry plan; same absolute helpers.
- `tests/test_luka_auxiliary_logits.py`: adapted focused workflow tests plus
  old-contract rejection and exact k1-to-k5/k10 initialization/gradient scaling.

Existing imported native/data/bank/Slurm kernels retain their donor provenance
in the original auxiliary-logit entry below. The remote scientific parent is
still immutable trial c50 at72490a9d6de4acabe9cbea518c5b100d893e6866, NOT a new
fit in the original auxiliary-logit campaign; pending k0/k025/k05/k1 jobs and
their bank-retirement process are independent of this follow-up.

## FullStat split-head DE066 -> DE033 -> DE000 (2026-10-24)

Additive destination `src/hlt_classification/luka_fullstat_splithead/`, new
`LUKA_FULLSTAT_SPLITHEAD_<kind>/v1` contracts. Authority:
[plan](plans/LUKA_FULLSTAT_SPLITHEAD_PLAN.md). No Fresh_check imports or changes
to existing scientific runtime. Integration/source-guard baseline:
`a4596cb915252672ab42c7a8724d6a105a0bc56e`.

Donors read at that baseline (all paths under `src/hlt_classification/`):

- `luka_fullstat_hybrid_embedding/{contracts,source,campaign,banks,kd_banks,worker,submission}.py`:
  FullStat20 execution, native exports, immutable cross-view joins, provenance,
  three-step safety checks, afterok submission and saved reporting. Rebound to
  three new stages, separate auxiliary logits and authenticated external UE100.
- `luka_auxiliary_logits/{model,training,startup}.py`: independent auxiliary
  linear class heads, paired head RNG, CE-only main-head gradient isolation and
  per-teacher KL/embedding accounting. Copied kernels adapted to fixed20 and the
  new stage-specific teachers. No runtime import of this smaller-trial package;
  its fixed30/alpha-sweep/c50 population semantics are not inherited.
- `luka_fullstat/recipe.py`: reused fixed20 warmup2/cosine schedule and AdamW.
- `luka_embedding_aux/{model,targets,contracts}.py`: reused native differentiable
  fc-input tap, FP64 TRAIN standardization and existing predictor seed.
- `luka_fullstat_attention/{contracts,worker}.py`: unchanged FullStat flags/counts,
  original cache metadata/validation identities and exact Oscar execution checks.
- `luka_fullstat_multiteacher/campaign.py`: unchanged protected output roots.
- `luka_fullsim/preflight_v2.py`, `jetclass2_delphes/{model,runner,reporting,dzfix_fusion_model}.py`,
  `context_fusion/worker.py`, `cms_proxy_ladder/{campaign,contracts,submission}.py`,
  `data/cache_contracts.py`, `luka_fullstat/submission.py`,
  `scouting/{hcwdl_exact_dag_submission,hcwdl_recovery}.py`: existing authenticated native
  kernels, phase logging, artifact publication, resource checks and exact journal.

Wrapper donors at the same commit: `scripts/luka_fullstat_hybrid_embedding.py`,
`scripts/queue_luka_fullstat_hybrid_embedding.sh`,
`sbatch/run_luka_fullstat_hybrid_embedding.sh`. New wrappers use the splithead
namespace, three-job authorization and unchanged absolute common helper.
Test donors: `tests/test_luka_fullstat_hybrid_embedding.py`,
`tests/test_luka_auxiliary_logits.py`, `tests/test_luka_embedding_aux.py`.

Actual remote artifact parent remains the immutable FullStat hybrid execution
`e129075e9c3eaa9fd2e49460b79ecf5f56ba4be7`: only completed UE000/UE100 ancestry
is required. OFFLINE/M0HLT/UE100 target bytes retain original bank contracts;
new DE066/DE033 exports use this namespace. Old D rungs are never reused.

## c50 auxiliary-logit DE000 follow-up, direct debug (2026-10-24)

Donor/integration baseline: `72490a9d6de4acabe9cbea518c5b100d893e6866`.
Additive `luka_auxiliary_logits/`; no prior scientific runtime changes or
Fresh_check imports. Authority: [plan](plans/LUKA_AUXILIARY_LOGITS_PLAN.md),
new `LUKA_AUXILIARY_LOGITS_<kind>/v1` contracts. Existing native target BANK/v1
semantics remain unchanged; new owner/spec/receipt identifies the new exports.

All module paths below are relative to `src/hlt_classification/` at that commit:

- `contracts.py` adapts `luka_trial_hybrid/contracts.py`'s c50 DE000 recipe,
  paired seeds, nodes and trial flags; registers four final-rung loss variants.
- `model.py` adapts `luka_trial_hybrid/model.py`'s RNG-isolated auxiliary heads,
  reusing `luka_embedding_aux/model.py::head_input` and
  `jetclass2_delphes/model.py::DelphesParticleTransformer`. Adds four independent
  linear logit heads; native classifier and inference remain unchanged.
- `training.py` adapts `luka_trial_hybrid/training.py`'s from-scratch fixed30
  loop, target detachment, loss logging and selected native-state extraction;
  separates logit supervision from main-head CE. Reuses
  `jetclass2_delphes/batch128_30e_recipe.py`, `runner.py` and `reporting.py`.
- `banks.py` delegates unchanged native TRAIN export, normalization, exact
  joins and manifests to `luka_trial_hybrid/banks.py`. Newly owned four-teacher
  bundle never depends on the parent's disposable target payloads.
- `source.py` uses `luka_attention_weight/source.py::replay`,
  `luka_fullsim/foundation.py::_source`, `luka_fullsim/preflight_reuse.py::_files`
  and `provenance.py`. Original-source authentication uses the parent
  `luka_trial_hybrid/{campaign,worker,banks,submission}.py`, only the c50
  dependency closure, and existing retained checkpoint/report descriptors.
- `campaign.py`, `submission.py`, `worker.py` adapt the corresponding
  `luka_trial_hybrid` modules. Reuse `luka_fullsim/regular_cache.py`,
  `luka_trial_hybrid/worker.py::{data,native,release}`,
  `luka_embedding_weight/worker.py` storage/prediction checks,
  `luka_attention/worker.py::execution`, `context_fusion/worker.py::save_state`,
  `cms_proxy_ladder/contracts.py`, `data/cache_contracts.py`, and
  `luka_fullsim/preflight_v2.py` phase/memory evidence. Exact DAG/journal/site
  helpers remain `scouting/hcwdl_exact_dag_submission.py`,
  `scouting/hcwdl_recovery.py`, `luka_fullsim/regular_submission.py`,
  `cms_proxy_ladder/submission.py` and `jetclass2_delphes/execution.py`.
- `startup.py` composes new-objective gradient checks with native forward,
  `jetclass2_delphes/dzfix_fusion_model.py::parity_backend` and
  `luka_fullsim/preflight_v2.py::{memory_snapshot,check_headroom}`. This runs
  inside target export, not as a standalone preflight or science epoch.

CLI, queue helper and Slurm wrapper adapt `scripts/luka_trial_hybrid.py`,
`scripts/queue_luka_trial_hybrid.sh`, `sbatch/run_luka_trial_hybrid.sh` and
absolute `sbatch/common.sh`. Tests reuse fixtures from
`tests/test_luka_trial_hybrid.py`, `tests/test_luka_embedding_aux.py` and
`tests/test_luka_u100_pretrain.py`; no local fixture constitutes remote acceptance.

## Four trial hybrid ladders, direct debug (2026-10-24)

Donor/integration baseline: `2376559bc9b1d0ff455bbdb242ae5769a54d6168`.
Additive `luka_trial_hybrid/`; all previous scientific runtime is unchanged.
No external source copies, Fresh_check imports or historical artifact edits.
Authority: [trial hybrid plan](plans/LUKA_TRIAL_HYBRID_PLAN.md), new
`LUKA_TRIAL_HYBRID_<kind>/v1` contracts. FullStat artifacts, populations and
20-epoch training are NOT relabeled as trial artifacts.

- `contracts.py` adapts the stage/teacher/head-seed graph from
  `luka_fullstat_hybrid_embedding/contracts.py`; adds four independent arms,
  explicit per-epoch mixtures and trial flags from `luka_fullsim/alljets_data.py`.
- `model.py` reuses `luka_fullstat_hybrid_embedding/model.py::HybridParT` for
  the native fc-input hook and RNG-isolated removable prediction heads only.
  `training.py` reuses the detached loss components from that package's
  `training.py`, but applies the new mixtures and the original trial schedule
  from `jetclass2_delphes/batch128_30e_recipe.py`. Native model, forward,
  prediction, selection and metrics remain in `jetclass2_delphes/model.py`,
  `runner.py` and `reporting.py`.
- `banks.py` adapts native target export from `luka_embedding_aux/targets.py`,
  `model.py` and `luka_fullstat_hybrid_embedding/kd_banks.py`; new compact banks
  bind exact branch, model and ordered TRAIN identities. No token/attention bank.
- `source.py` reuses `luka_joint_prediction/source.py`'s original-source replay
  of the completed d8dba242 token study and `luka_fullsim/foundation.py::_source`.
  New guard binds all sixteen additive files and rejects old-runtime changes.
- `campaign.py`, `submission.py`, `worker.py` adapt the corresponding joint and
  `luka_embedding_prediction` modules. Reuse `luka_embedding_prediction/data.py`,
  `luka_embedding_aux/targets.py::offline_train`, `luka_fullsim/regular_cache.py`,
  `luka_fullsim/preflight_v2.py` phase/memory evidence, `luka_attention/worker.py`
  execution checks, `luka_embedding_weight/worker.py` storage/prediction checks,
  `luka_joint_prediction/worker.py::snapshot_references`, and
  `context_fusion/worker.py::save_state`. The exact DAG uses unchanged
  `scouting/hcwdl_exact_dag_submission.py` and `scouting/hcwdl_recovery.py`;
  site checks/environment use `luka_fullsim/regular_submission.py`.
- Thin CLI/queue/Slurm files adapt the joint counterparts and absolute
  `sbatch/common.sh`. New plan, contract and runbook describe the direct 22-job
  graph and strictly owned target retirement. No new preflight training.
- `tests/test_luka_trial_hybrid.py` reuses synthetic fixtures from
  `test_luka_u100_pretrain.py`, `test_luka_embedding_aux.py`, and
  `test_luka_attention.py`; exact donor objective comparisons and actual
  installed-Weaver CPU/CUDA parity supplement full 20-worker local execution.
  Cluster/source/accounting boundaries are mocked, not remote validation.

## Joint jet4/token3 U100 prediction, direct debug (2026-10-24)

Donor/integration baseline: `d8dba242c20c65a9c5155f93a68a5eaf12ba5063`.
Additive `luka_joint_prediction/`; no old runtime, source worktree, payload,
contract or job edited. No external donor or Fresh_check import.
Authority: [joint plan](plans/LUKA_JOINT_PREDICTION_PLAN.md), new
`LUKA_JOINT_PREDICTION_<kind>/v1` family, NOT token-only artifact reuse.

- `contracts.py`, `training.py` adapt `luka_particle_prediction/contracts.py`
  and `training.py`: same fixed30/batch128/native sampler/optimizer/selection,
  two heads and CE+4*jet+3*tokens instead of CE+5*tokens; separate losses logged.
- `model.py` combines unchanged `luka_particle_prediction/model.py::token_forward`
  and `feature_loss` with the native fc-input hook/RNG-isolated predictor pattern
  from `luka_embedding_aux/model.py`. One native forward, two independent heads.
- `banks.py` adapts `luka_particle_prediction/banks.py` to export both surfaces
  in one pass; reuses `luka_embedding_aux/targets.py::standardize`, independent
  TRAIN jet/token moments, unchanged `luka_attention/banks.py::alignment/counts`
  and `jetclass2_delphes/dzfix_fusion_model.py::parity_backend`.
- `source.py` adapts `luka_particle_prediction/source.py`, original-source replay
  from `luka_attention_weight/source.py` and source snapshots from
  `luka_fullsim/foundation.py`. Original `luka_particle_prediction/worker.py`
  validates completed token studies and passed acceptance without new compute.
- `campaign.py`, `submission.py`, `worker.py` adapt same-named token modules:
  new direct two-job registration, source-bound prior acceptance/resource margin,
  zero new preflights; unchanged `luka_attention/worker.py::data/execution`,
  `luka_u100_embedding/worker.py::teacher_readback/validate_predictions`,
  `luka_fullsim/regular_cache.py`, `preflight_v2.py`, `context_fusion/worker.py`,
  native model/runner/reporting/fixed30 schedule and exact-DAG journal helpers.
- CLI, queue helper, Slurm wrapper and `tests/test_luka_joint_prediction.py`
  adapt their `luka_particle_prediction` counterparts. Fixture helpers reuse
  `test_luka_particle_prediction.py`, `test_luka_u100_pretrain.py`, and
  `test_luka_attention.py`; production-worker miniature uses real installed
  full-architecture Weaver. Plan/contract/runbook are new.

## Trial particle-feature prediction, aligned versus set (2026-10-23)

Donor/integration baseline: `01d45d6057e799b4e262f8c590b51706803ebfd1`.
Additive `luka_particle_prediction/`; no existing scientific runtime changes,
external donor copies, Fresh_check imports, or historical artifact rewrites.
New `LUKA_PARTICLE_PREDICTION_<kind>/v1` registration, documented in
[the active plan](plans/LUKA_PARTICLE_PREDICTION_PLAN.md).

- `model.py` adapts `luka_embedding_aux/model.py`'s removable, RNG-isolated
  predictor and native-only checkpoint convention. The block-hook pattern
  derives from `luka_attention/model.py`, but the new target is block-8 OUTPUT,
  not attention probabilities. New exact SmoothL1 Hungarian kernel calls the
  already-installed SciPy implementation; no third-party source copied.
- `banks.py` adapts `luka_u100_embedding/targets.py`'s FP64 TRAIN moments and
  teacher binding, and reuses `luka_attention/banks.py::alignment/counts`.
  New ragged FP32 token mmap lives only in owned scratch. The existing
  `jetclass2_delphes/dzfix_fusion_model.py::parity_backend` context is reused
  for deterministic teacher export and native parity, then restored.
- `contracts.py`, `training.py` adapt `luka_u100_embedding/contracts.py` and
  `training.py`: identical M0HLT seeds, fixed30 batch128 AdamW schedule and
  validation selection, with new token prediction losses instead of jet loss.
  Reuse unchanged `jetclass2_delphes/runner.py`, `reporting.py`, and
  `batch128_30e_recipe.py`.
- `source.py` reuses `luka_embedding_aux/source.py` and
  `luka_u100_embedding/source.py` for original fixed30 controls/COARSE_U100;
  `luka_attention_weight/source.py::replay` authenticates completed whole-jet
  U100-w5 using `luka_u100_embedding/direct.py::results` in its original source.
- `worker.py` adapts `luka_u100_embedding/worker.py` readback/report patterns,
  uses `luka_attention/worker.py::data/execution`, `luka_fullsim/regular_cache.py`
  and `preflight_v2.py::PhaseLog/memory_snapshot`. Node-local mount checks adapt
  `luka_attention_weight/retention.py`, excluding tmpfs for these token banks.
  Only unique new temporary directories are retired; shared datasets, models,
  campaigns and scratch roots are never deleted.
- `campaign.py`, `submission.py`, thin CLI/queue/Slurm files adapt U100 trial
  orchestration and `luka_multiteacher_trial/submission.py`'s exact DAG journal.
  New reviewed graph: debug preflight -> two independent afterok fits, with
  successful accounting AND artifact admission, no automatic retry/controller.
- Focused tests reuse synthetic `test_luka_u100_pretrain.py::cache` and
  `test_luka_attention.py::parent_nodes`; new tests cover exact matching,
  production Weaver parity, complete fixed30 worker paths, scratch retirement,
  target drift, source/lineage failures and exact three-job submission.

## FullStat five-stage hybrid KD / embedding ladder (2026-10-23)

Implementation donor: `d98359f76abe3a46c70f948aa4b99d51c82787e9` (the committed
FullStat cumulative embedding ladder). Integration baseline:
`6b76a6777952c8faa2b155084ef560518c6fa55a`. Runtime acceptance anchor:
`b21c2d4e01211a31e57d879c5a86ca90aab2ca0b` (completed four-teacher Oscar study).
New additive `luka_fullstat_hybrid_embedding` namespace and
`LUKA_FULLSTAT_HYBRID_EMBEDDING_<kind>/v1` contracts. No old runtime changes,
external donor files, Fresh_check imports or historical artifact rewrites.

- `contracts.py`: adapts cumulative contracts to five explicit fresh stages,
  independent immediate-logit and embedding teacher maps, T2/.75KD/.25CE,
  total auxiliary5 and an isolated M0HLT head seed. Reuses unchanged FullStat20
  optimizer/schedule and cms_proxy_ladder coordinate-paired seeds.
- `model.py`, `training.py`: adapt cumulative native-tap/removable-head model
  and20-epoch loop. KD formula matches unchanged
  `jetclass2_delphes/model.py::distillation_loss` (forward KL, one T-squared).
  Native CUDA/BF16 logits, FP32 heads/losses, separate per-term logging; no
  online teachers, warm starts, attention or relational objective.
- `banks.py`: adapts cumulative selected-native TRAIN export, original OFFLINE
  bank adapter, independent FP64 moments, exact row/label joins and bundle
  ancestry. Adds original M0HLT target and KD-only UE000 provenance.
- `kd_banks.py`: new compact authenticated TRAIN probability-bank adapter;
  directly reuses `jetclass2_delphes/runner.py::predict` at temperature2,
  batch128, same convention as `luka_fullstat/worker.py::fit_or_reduce`.
  Binds selected native checkpoint/report and exact readback; no legacy
  probability artifact is relabelled as a hybrid embedding bank.
- `campaign.py`, `source.py`, `submission.py`, `worker.py`: adapt cumulative
  source replay, protected outputs, measured-evidence reuse, short integrated
  startup checks, exact journaled DAG, completion-last receipts and saved-only
  reporting. Five jobs, first-job original targets, conditional outgoing KD
  and embedding exports; all ancestors checked, including KD-only UE000.
  Add authenticated original DIRECT_D000/COARSE_D000 references. Shared
  FullStat cache/environment/Slurm/metric helpers remain unchanged.
  An explicit unimported parallel particle-prediction file list may be added
  alongside this package; existing runtime edits are never exempted, and all
  committed companion bytes remain source-snapshot bound.
- `scripts/luka_fullstat_hybrid_embedding.py`,
  `scripts/queue_luka_fullstat_hybrid_embedding.sh`,
  `sbatch/run_luka_fullstat_hybrid_embedding.sh`: adapt corresponding cumulative
  thin entrypoints, new names/authorization, explicit5-job dry review. No
  preflight/controller and no mutations to old studies/jobs.
- `tests/test_luka_fullstat_hybrid_embedding.py`: adapts cumulative/four-teacher
  fixtures and failure-injection/parity/DAG coverage; adds exact hybrid-loss
  gradients/update checks, temperature-two bank integrity, separate target
  maps and all-five-production-worker100-epoch synthetic CUDA miniature.

Authoritative semantics and evidence are in the new plan/contract/runbook and
the dated HANDOFF section; local miniature evidence is not new Oscar timing.

## Two replacement LOGIT teachers plus five trials (2026-10-23)

Integration/donor commit: `01d45d6057e799b4e262f8c590b51706803ebfd1`.
Additive `luka_multiteacher_recovery` package; no existing runtime changes,
external donor copies, Fresh_check imports or historical artifact rewrites.
New `LUKA_MULTITEACHER_RECOVERY_<kind>/v1` owns replacement execution identity.

- `source.py`: adapts `luka_multiteacher_trial/source.py` source/ref guards,
  reuses `luka_embedding_aux/source.py` and `luka_u100_embedding/source.py`
  for surviving controls/teachers, and original-source
  `luka_attention_weight/reference_archive.py` for retained results. Original
  deleted LOGIT weight descriptors are historical only, never substituted.
- `campaign.py`, `submission.py`, CLI/queue/Slurm scripts: adapt the original
  six-job multi-teacher orchestration to eight jobs with two explicitly new
  fits, per-task afterok/accounting and5GiB retained storage. Reuse exact-DAG
  journals and single-claim behavior, no new preflight or retry controller.
- `worker.py`: directly calls unchanged `luka_attention/worker.py::fit`,
  `training.py`, `model.py`, `banks.py` for original C25/P75 T2 LOGIT training,
  zero attention weight, full30 selection/readback. Only compact probability
  exports; no attention target payloads. Directly reuses
  `luka_multiteacher_trial/worker.py::export_targets/fit/validate_fit`,
  `banks.py`, `model.py`, `training.py`, and recipe definitions for all five
  student objectives. A new derived CONTEXT binds actual new checkpoints to
  the approved recovery SPEC; old schemas/specs/receipts remain immutable.
- Results adapt the original multi-teacher saved-only metric/report surface;
  label NEW_LOGIT_D066/D033 explicitly and preserve historical references.
  Tests extend original archive, multi-teacher, native-Weaver and Slurm
  fixtures; production kernels remain unchanged.

## Five smaller-data multi-teacher trials, SPORC tier3 (2026-10-23)

Integration/donor checkout: `d98359f76abe3a46c70f948aa4b99d51c82787e9`.
New additive namespace `luka_multiteacher_trial`; no old runtime edits, external
migration or Fresh_check runtime imports. New identities
`LUKA_MULTITEACHER_TRIAL_<kind>/v1`; old artifact contracts/bytes stay unchanged.

- `model.py` adapts `luka_fullstat_multiteacher/model.py`: four independent
  removable prediction heads, same fixed slot seeds, unchanged native model/RNG.
- `contracts.py`, `training.py` adapt the FullStat four-teacher objective and
  `luka_u100_embedding/training.py` fixed30 path. Reuse unchanged
  `jetclass2_delphes/batch128_30e_recipe.py`, `runner.py`, `reporting.py`,
  `luka_embedding_aux/contracts.py`, `model.py`. Five registered recipes rather
  than a CLI hyperparameter switch; the new relational kernel/sampler adapts the
  documented reference-bank idea in `LUKA_RELATIONAL_EMBEDDING_STRATEGY_HANDOFF.md`
  to four separate teacher spaces and the SPORC 30-epoch schedule.
- `source.py` reuses `luka_embedding_aux/source.py` for fixed30 CE/OFFLINE,
  `luka_u100_embedding/source.py` for original regular U100, and
  `luka_attention_weight/source.py` plus `reference_archive.py` for original
  retained-result replay. The new two-teacher replay adapts
  `luka_d066_embedding/source.py`: actual selected LOGIT_D066 AND LOGIT_D033
  checkpoint payloads are authenticated against their archived receipts.
  It does not require discarded attention banks. Reuses D066 reference import
  and U100 direct resource validation without relabelling old admission.
- `banks.py` adapts FullStat multi-teacher banks and U100 target checks; shares
  unchanged `luka_embedding_aux/targets.py` TRAIN FP64 normalization/OFFLINE RAM
  reconstruction and `luka_attention/worker.py` authenticated prepared-view loads.
- `campaign.py`, `submission.py`, `worker.py` adapt U100 direct and FullStat
  multi-teacher orchestration. Reuse unchanged file/hash/atomic-publish contracts,
  exact-DAG journals/recovery, SPORC allocation/site checks, phase logging and
  native checkpoint saving. Six jobs: real target export plus five fits; zero
  new preflight jobs. New R30 helper follows
  `luka_fullstat_cumulative_embedding/worker.py`'s tie-inclusive rank definition;
  R50/recovery retain the original reporting kernel.
- CLI, shell queue/worker scripts, tests and runbook adapt corresponding U100
  direct/FullStat multi-teacher entrypoints. Tests reuse existing small native
  cache fixtures, retained-attention archive fixture and inherited-resource
  fixture; real models/gradients/exports/fits/reporting execute locally.

Original execution lineage remains distinct: fixed30 parent
`74aa58c582ff6f497a2c6d91c503f5489fac3e9a`; LOGIT ladder
`94f560259dd04253c0aa47b31dec3aedc105ac14`; completed OFFLINE-w5 debug-transfer
reference `855c4f8d40b4db7837238c7ac6a3799597466e85`. FullStat/Oscar is only a
code donor, never a substituted sample, teacher or resource certificate.

## FullStat cumulative representation ladder, Oscar (2026-10-23)

Donor/execution: `b21c2d4e01211a31e57d879c5a86ca90aab2ca0b` (completed
FullStat four-teacher campaign). Integration baseline:
`7e8dd105af1ead66b7ec120d7a391ca8310b5401`. A regression test confirms that
no pre-existing src/scripts/sbatch runtime changed between these commits.
No external migration, Fresh_check import, old-runtime edit or old-artifact
relabelling. New namespace: `luka_fullstat_cumulative_embedding`.

- `contracts.py` adapts `luka_fullstat_multiteacher/contracts.py`: four fresh
  coordinate-paired nodes, cumulative native targets, total weight 5 divided
  equally over one to four independent heads. New v1 scientific identities.
  Imports unchanged `cms_proxy_ladder/campaign.py` coordinate/paired seeds,
  `luka_embedding_aux/contracts.py` head seed, `luka_fullstat/recipe.py` schedule
  and `luka_fullstat_attention/contracts.py` sealed-role/count definitions.
- `model.py` and `training.py` adapt the corresponding four-teacher files:
  dynamic registered head set, unchanged native forward, optimizer, sampling,
  20-epoch schedule and AUC-first selection. No KD, warm start, benchmark or
  technical-epoch mode. Selected checkpoints contain native weights only.
- `banks.py` adapts four-teacher target export/join code, adding new selected
  ancestor checkpoint/report and completed-task bindings. Each new native
  embedding has independent TRAIN moments. Reuses the original sweep OFFLINE
  BANK validator/bytes, with an explicit new cross-view consumer adapter rather
  than disguising coordinates. No old COARSE target bank substitutes for a
  newly trained ancestor.
- `source.py`, `campaign.py`, `submission.py` adapt the corresponding
  four-teacher files. Original-source replay now authenticates completed
  four-teacher/U100 references, original OFFLINE bank and native/multi-head
  resource acceptance. All pre-existing donor runtime bytes are compared.
  Exact reviewed DAG has four scientific jobs only, with afterok dependencies
  on every earlier rung. No separate setup/preflight/controller/automatic retry.
- `worker.py` adapts four-teacher fit/native-readback/completion/report code,
  plus three-step stress mechanics. Each worker loads one ordinary view,
  authenticates all ancestor receipts, checks its exact head/view combination,
  trains 20 epochs, then exports its selected native TRAIN bank (first three
  stages). Completion receipt publishes last. R30/R50 reporting uses existing
  pair-score/rank/tie semantics and saved AUC-selected probabilities.
- Thin entrypoints adapt `scripts/luka_fullstat_multiteacher.py`,
  `scripts/queue_luka_fullstat_multiteacher.sh` and
  `sbatch/run_luka_fullstat_multiteacher.sh`. The absolute-project environment
  helpers `sbatch/jetclass2_delphes_common.sh` and `sbatch/common.sh` are unchanged.
- Tests adapt `tests/test_luka_fullstat_multiteacher.py`, importing its fixtures
  and U100/sweep/FullStat/auxiliary fixtures. Added four-stage full-architecture
  installed-Weaver/CUDA production-worker miniature, four-target loss/gradient
  and optimizer-step parity, exact new ancestry, selected native target export,
  independent normalization, corrupt/forbidden joins, failed-parent blocking,
  poor-score continuation, four-job dry/live journals and shell/doc checks.
- Shared unchanged dependencies: `data/cache_contracts.py`,
  `cms_proxy_ladder/{contracts,submission}.py`, `context_fusion/worker.py`,
  `jetclass2_delphes/{model,runner,reporting}.py`,
  `luka_embedding_aux/{model,targets}.py`, `luka_fullstat/{cache,submission}.py`,
  `luka_fullstat_attention/worker.py`, `luka_fullsim/{foundation,preflight_v2}.py`,
  and `scouting/{hcwdl_exact_dag_submission,hcwdl_recovery}.py`.

Plan: [cumulative FullStat plan](plans/LUKA_FULLSTAT_CUMULATIVE_EMBEDDING_PLAN.md).
Contract: [LUKA_FULLSTAT_CUMULATIVE_EMBEDDING](contracts/LUKA_FULLSTAT_CUMULATIVE_EMBEDDING.md).
Queue: [cumulative Oscar runbook](LUKA_FULLSTAT_CUMULATIVE_EMBEDDING_RUNBOOK.md).

## Immediate-predecessor native embedding prediction ladder, debug (2026-10-22)

Donor baseline: `118aa375eea423e5e99338cd9a6eda198225ef21`.
Reviewed integration baseline: `b21c2d4e01211a31e57d879c5a86ca90aab2ca0b`.
Its only src changes are nine additions under the separate
`luka_fullstat_multiteacher` package; all existing scientific donors remain
byte-identical. The new source guard is pinned to that integrated tree, with a
regression test checking the exact additive-only integration boundary.
New isolated `src/hlt_classification/luka_embedding_prediction`; no old runtime
edits or external/Fresh_check migrations. Original U000 and historical controls
continue to authenticate in their unchanged pinned74aa58c5 source.

- `contracts.py` adapts `luka_embedding_ladder/contracts.py` and
  `luka_u100_embedding/contracts.py`: immediate native targets U000 -> U100 ->
  D066 -> D033 -> D000, no logit KD, weight5, fresh fixed30, distinct v1 family.
- `source.py`, `campaign.py`, `submission.py` adapt corresponding
  `luka_ce_warmstart` files. Initializer/reference imports are removed; the
  original `luka_embedding_ladder/source.py::authenticate` is imported unchanged
  with its original ANCHOR validator. Exact submission becomes five sequential
  jobs; enumeration guard protects existing src relative to the baseline.
- `data.py` adapts `luka_embedding_ladder/data.py`: request one coordinate at a
  time; permit ONLY fully missing U000 reconstruction using saved assignments,
  exact fingerprints and RAM. Remove U050/fixed-OFFLINE-target behavior.
- `targets.py` adapts `luka_u100_embedding/targets.py`. New selected native
  TEACHER/BANK bindings identify the immediate predecessor and technical status;
  reject auxiliary-head checkpoints. Same TRAIN normalization/128-D tap.
- `training.py` adapts `luka_u100_embedding/training.py` with unchanged numerical
  CE+5 objective, AdamW, fixed30 schedule, sampling, validation and native-only
  selection. Generalize authenticated coordinate/target relationships and use
  new v1 records; no CE-warmstart or logit target behavior is inherited.
- `worker.py` adapts the setup/fit/readback/report structure of
  `luka_ce_warmstart/worker.py`, plus `luka_embedding_ladder/worker.py`'s staged
  handoff concept. Separates four technical probes from four fresh science fits,
  exports selected native TRAIN embeddings after each non-final stage, keeps
  compact artifacts and supports authenticated partial reporting without models.
- Thin CLI, helper and wrapper adapt `scripts/luka_ce_warmstart.py`,
  `scripts/queue_luka_ce_warmstart.sh`, `sbatch/run_luka_ce_warmstart.sh`.
  `sbatch/common.sh` remains unchanged and is sourced by absolute project path.
- Tests adapt `tests/test_luka_ce_warmstart.py`, `test_luka_embedding_ladder.py`,
  `test_luka_u100_embedding.py`; fixtures import `test_luka_u100_pretrain.py`,
  `test_luka_attention.py`, `test_luka_embedding_aux.py`,
  `test_luka_fullsim_stage2.py` and `test_luka_fullsim_alljets.py`.
- Unchanged direct dependencies: `data/cache_contracts.py`,
  `cms_proxy_ladder/{contracts,submission}.py`, `context_fusion/worker.py`,
  `jetclass2_delphes/{model,cache,runner,reporting,execution,acceptance,
  dzfix_fusion_model,batch128_30e_recipe}.py`,
  `luka_fullsim/{foundation,regular_cache,stage2,preflight_cache,particles,
  alljets_data,preflight_v2,science_campaign,regular_submission}.py`,
  `luka_attention/worker.py`, `luka_embedding_weight/worker.py`,
  `luka_embedding_aux/{model,targets,preflight}.py`,
  `luka_embedding_ladder/{source,contracts}.py`, and
  `scouting/{hcwdl_exact_dag_submission,hcwdl_recovery}.py`.
  These retain their original semantics and contracts; no artifact relabelling.

## FullStat four-teacher embedding prediction, Oscar (2026-10-22)

Integration baseline: `118aa375eea423e5e99338cd9a6eda198225ef21`.
Original authenticated U100 execution/donor:
`7ea8b0c693734a26ced3820354e047f4af26c979`; original OFFLINE sweep bank retains
its `fd687181a60c1557f98d7498f577e26d6dbb35a7` execution lineage.
No external donor migration, Fresh_check import, or existing runtime edit.

- New `luka_fullstat_multiteacher/{contracts,source,campaign,banks,training,
  worker,submission}.py` adapt the corresponding `luka_fullstat_u100_embedding`
  files at the integration baseline (unchanged from the U100 execution).
  They register a distinct four-head objective, independent per-teacher target
  contracts, authenticated two-bank reuse plus two new exports, one new fit,
  measured setup admission and automatic `afterok` submission.
- `model.py` adapts `luka_embedding_aux/model.py`: unchanged native pre-FC tap,
  separate seeded ModuleDict heads, no target/auxiliary input to classifier,
  native-only selected checkpoints. `contracts.py` imports unchanged
  `luka_embedding_aux/contracts.py`, `luka_fullstat/recipe.py` and
  `luka_fullstat_attention/contracts.py` for predictor seed, schedule and roles.
- `banks.py` imports unchanged `luka_embedding_aux/targets.py` standardization
  and the original `luka_fullstat_embedding_sweep/banks.py` and
  `luka_fullstat_u100_embedding/banks.py` validators. Original OFFLINE/U100
  artifacts are not copied or relabelled. D066/D033 receive new BANK/v1 artifacts.
- Shared unchanged runtime: `luka_fullstat/{cache,worker,submission}.py`,
  `luka_fullstat_attention/worker.py`, `luka_fullsim/{foundation,preflight_v2}.py`,
  `jetclass2_delphes/{model,runner,reporting}.py`, `context_fusion/worker.py`,
  `data/cache_contracts.py`, `cms_proxy_ladder/{contracts,submission}.py`, and
  `scouting/{hcwdl_exact_dag_submission,hcwdl_recovery}.py`.
- CLI/helper/Slurm worker adapt `scripts/luka_fullstat_u100_embedding.py`,
  `scripts/queue_luka_fullstat_u100_embedding.sh`, and
  `sbatch/run_luka_fullstat_u100_embedding.sh`; retain the unchanged absolute
  `sbatch/jetclass2_delphes_common.sh` environment wrapper.
- Tests adapt `tests/test_luka_fullstat_u100_embedding.py`, importing its fixtures
  and `tests/test_luka_fullstat_embedding_sweep.py`, `test_luka_fullstat.py`,
  `test_luka_attention.py`, `test_luka_embedding_aux.py` test helpers. New tests
  cover four independent loss/gradient paths, native RNG/inference parity,
  independent normalization, measured admission, exact two-job DAG, corruption
  and an actual installed-Weaver/CUDA full20 production-worker miniature.
- Original-source replay remains in the clean original U100 worktree. Imported
  donor runtime is byte-compared; explicitly excluded are six unrelated changed
  SPORC D066 entrypoints/modules listed in `source.UNRELATED_SPORC`. Tests verify
  those are the only modified pre-existing runtime files between donor/baseline,
  and none is imported by this workflow. No FullStat kernel is excluded.

Plan: `docs/plans/LUKA_FULLSTAT_MULTITEACHER_PLAN.md`; new artifact family:
`LUKA_FULLSTAT_MULTITEACHER_<kind>/v1`; queue/runbook:
`docs/LUKA_FULLSTAT_MULTITEACHER_RUNBOOK.md`.

## Trial CE warm-start matched comparison, debug (2026-10-22)

Integration/donor baseline: `ed6d0af4be65f059356543f90583b30185702615`.
New isolated `src/hlt_classification/luka_ce_warmstart`; no existing runtime
edits, Fresh_check imports, or external donor migrations. Original selected
M0HLT and regular COARSE_U100 evidence still replays at74aa58c5; historical
OFFLINE-w5 reference at855c4f8d remains only contextual.

- `contracts.py` adapts `luka_u100_pretrain/contracts.py` and
  `luka_u100_embedding/contracts.py`, registering two independent native CE
  warm starts, weight0/5, fixed30 and0.1x LR in a distinct v1 family.
- `training.py` adapts `luka_u100_embedding/training.py`'s fixed30 loop and
  `luka_u100_pretrain/training.py`'s report validation. It changes initialization
  to an exact native CE checkpoint, not full predictor pretraining state;
  resets AdamW in both arms, lowers the LR in both, disables target/predictor
  access for the control, and retains native-only selected state.
- `targets.py` imports `luka_u100_embedding/targets.py`'s unchanged TRAIN-only
  export and Bank and its original contracts, adding metadata validation for
  shared setup receipts. No bank or teacher is relabelled as a new artifact.
- `source.py` adapts `luka_u100_embedding/source.py`'s enumerated additive source
  guard. Imports its regular U100 teacher replay/import validation unchanged;
  imports `luka_embedding_aux/source.py` authentication and
  `luka_d066_embedding/source.py` historical-reference authentication. The
  reusable source chain also uses `luka_attention_weight/source.py` replay and
  `luka_fullsim/foundation.py` clean-source checks, unchanged.
- `campaign.py`, `submission.py`, `worker.py` adapt their corresponding
  `luka_u100_pretrain` files. New worker binds/copies/readbacks M0HLT, exports
  TRAIN-only targets, runs independent matched probes/fits, and reports both
  arms. New exact DAG changes serial20->30 to parallel CE/embed after setup.
- CLI, helper and Slurm worker adapt `scripts/luka_u100_pretrain.py`,
  `scripts/queue_luka_u100_pretrain.sh`, `sbatch/run_luka_u100_pretrain.sh`;
  retain unchanged absolute-project `sbatch/common.sh` and environment setup.
- Tests adapt `tests/test_luka_u100_pretrain.py` and
  `tests/test_luka_u100_embedding.py`, importing their cache/bank fixtures,
  `tests/test_luka_attention.py` parent fixtures and
  `tests/test_luka_embedding_aux.py` tiny installed-model configuration. New
  tests cover matched initialization, native CE equivalence, failed-setup
  blocking, both30-epoch workers and compact read-only results.
- Unchanged direct building blocks: `luka_embedding_aux/{model,preflight,
  targets,contracts}.py`, `luka_attention/worker.py`,
  `luka_embedding_weight/worker.py`, `jetclass2_delphes/{model,runner,reporting,
  execution,acceptance,dzfix_fusion_model,batch128_30e_recipe}.py`,
  `luka_fullsim/{regular_cache,alljets_data,preflight_v2,science_campaign,
  regular_submission}.py`, `context_fusion/worker.py`, `data/cache_contracts.py`,
  `cms_proxy_ladder/{contracts,submission}.py`, and
  `scouting/{hcwdl_exact_dag_submission,hcwdl_recovery}.py`.

Module paths above are relative to `src/hlt_classification/` unless prefixed
with scripts/sbatch/tests/docs. New plan, versioned contract and runbook state
the lower-LR matched protocol; historical inference and scientific meaning
remain unchanged. Local regression evidence is in HANDOFF; SPORC acceptance
still requires the newly queued shared setup, not old preflight assumptions.

## Trial normal U100 embedding weight5, direct tier3 (2026-10-22)

Baseline/donor commit: `dd51e8e25770482134809dbb34d4766f8dac74b9`.
New isolated `src/hlt_classification/luka_u100_embedding` and thin wrappers;
no donor edits, external migration or Fresh_check runtime imports.

- `contracts.py`, `targets.py`, `training.py`, `worker.py` adapt the corresponding
  `luka_d066_embedding` files. They preserve the head, loss, fixed30 update math,
  seeds, TRAIN normalization and native-only selection/readback, change the
  teacher to regular COARSE_U100, and use a distinct v1 artifact family.
  Training timers use perf_counter for local small-batch resolution.
- `worker.teacher_readback` adapts `luka_d066_embedding/preflight.py`'s readback
  function ONLY; no preflight runner/stress/technical fit is invoked.
- `direct.py` adapts `luka_d066_embedding/direct.py`'s inherited-resource and
  exact one-job workflow, omitting the D066 retired-job/cancellation requirement.
  There is no job/controller/dependency belonging to another campaign here.
- `source.py` adapts `luka_u100_pretrain/source.py`'s original-source regular
  COARSE_U100 replay, changes the target role to TRAIN-only and pins this new
  additive baseline. It reuses `luka_embedding_aux/source.py` authentication,
  `luka_d066_embedding/source.py` OFFLINE-w5 reference, `luka_attention_weight/
  source.py` replay, and `luka_fullsim/foundation.py` clean-source checks.
  Teacher execution remains original74aa58c5; OFFLINE-w5 reference855c4f8d.
- Direct imports reuse `luka_embedding_aux/{model,targets,contracts}.py`,
  `luka_attention/worker.py`, `luka_embedding_weight/worker.py`,
  `jetclass2_delphes/{execution,model,runner,reporting,batch128_30e_recipe}.py`,
  `luka_fullsim/{regular_cache,alljets_data,preflight_v2,regular_submission}.py`,
  `context_fusion/worker.py`, `data/cache_contracts.py`,
  `cms_proxy_ladder/{contracts,submission}.py`, and
  `scouting/{hcwdl_exact_dag_submission,hcwdl_recovery}.py` unchanged.
- Queue helper and Slurm wrapper adapt `scripts/queue_luka_d066_embedding_direct.sh`
  and `sbatch/run_luka_d066_embedding.sh`; source the absolute-project unchanged
  `sbatch/common.sh`. CLI is a new thin direct-only interface. Tests adapt
  `tests/test_luka_d066_embedding{,_direct}.py` and U100-pretrain fixtures,
  adding U100 lineage and no-pretraining/no-new-preflight checks. Plan,
  contract and runbook explicitly document the new user's direct-run waiver.

All module paths above are relative to `src/hlt_classification/`. No old bank,
receipt or checkpoint is relabelled to masquerade as new U100 training evidence.

## Trial U100 embedding pretraining20 then joint30, debug (2026-10-22)

Integration/source baseline: `e75237a803d3ea6dfb55f110be1a0a899b467bc1`.
New code is isolated in `luka_u100_pretrain`; no existing runtime or scientific
artifact is edited. This is not the separate FullStat U100 implementation.
There is no Fresh_check migration or runtime import.

- `training.py` adapts `luka_d066_embedding/training.py` at the baseline:
  preserve fixed30 CE+weight5 update math, initialization/sampler and native
  selection rule, but add an explicitly separate embedding-only20 stage,
  classifier freeze, full-state selection/restore and fresh phase-2 optimizer.
  A same-input/same-target test checks the phase-2 update against the donor.
- `targets.py` reuses `luka_embedding_aux/model.py`'s native 128-D tap and
  `luka_embedding_aux/targets.py`'s TRAIN-only standardization. The new bank
  contract separates TRAIN and validation targets; only TRAIN fits moments.
- `source.py` reuses `luka_embedding_aux/source.py`'s original fixed30 control
  authentication, `luka_d066_embedding/source.py`'s completed OFFLINE-w5
  reference replay, and `luka_attention_weight/source.py`'s isolated replay.
  A new replay authenticates regular COARSE_U100 from parent
  `74aa58c582ff6f497a2c6d91c503f5489fac3e9a`; it does not import D066 or fusion
  teacher weights. Historical OFFLINE-w5 remains the transferred source
  `855c4f8d40b4db7837238c7ac6a3799597466e85`. Both original worktrees stay pinned.
- `worker.py` composes unchanged `luka_attention/worker.py` data/allocation
  checks, `luka_embedding_weight/worker.py` storage/prediction checks,
  `luka_embedding_aux/preflight.py` predictor parity, native installed parity
  from `jetclass2_delphes/acceptance.py`, native inference/reporting from
  `jetclass2_delphes/{runner,reporting,execution,model}.py`, cache/subset helpers
  from `luka_fullsim/{regular_cache,alljets_data}.py`, accounting/phase helpers
  from `luka_fullsim/{science_campaign,preflight_v2}.py`, and state publication
  from `context_fusion/worker.py`. These reusable functions remain unchanged.
- `submission.py` uses the existing exact journal/ledger machinery in
  `scouting/{hcwdl_exact_dag_submission,hcwdl_recovery}.py`, SPORC dry-resource
  checks/environment from `luka_fullsim/regular_submission.py`, and walltime
  rendering from `cms_proxy_ladder/submission.py`. The three-job debug DAG
  and new `LUKA_U100_PRETRAIN_*/v1` records are additive, not old-artifact reuse.
- CLI, queue helper, worker wrapper, tests, plan, contract and runbook are new.
  Shared content hashing/file references come from
  `data/cache_contracts.py` and `cms_proxy_ladder/contracts.py`; the wrapper
  sources the unchanged absolute-project `sbatch/common.sh`.

All paths above are under `src/hlt_classification/` unless stated otherwise.
The additive-source guard requires A-only changes under the enumerated new
package relative to the exact baseline; unknown additions or donor edits fail.
See HANDOFF for completed local evidence and the remaining genuine SPORC setup.

## D066 source-integration guard correction (2026-10-21)

Failed execution source:9a6163b75893b566d407022861cdaf8d5222e4a4.
Only `luka_d066_embedding/source.py` and its tests/docs change. The existing
baseline4a3124e9a3b190754ab195ca7430c00ceebcea78 is retained. The eight disjoint
`luka_fullstat_u100_embedding` files introduced by
7ea8b0c693734a26ced3820354e047f4af26c979 are allowed only when their complete
Git tree is8aabd738dd9314e459223a09d04a64c9e31ad01d. No code is migrated from
or imported from that package, and no scientific kernel is modified. Source
status must still be A relative to the baseline; shared changes, unknown new
files or a changed parallel tree fail. Real-repository Git and isolated Git
mutation tests supplement the older mocked source tests that missed this case.

## Direct tier3 D066 embedding execution without a new preflight (2026-10-21)

Integration donor: `7ea8b0c693734a26ced3820354e047f4af26c979`, specifically
`luka_d066_embedding/{source,submission,preflight,worker,targets,training}.py`
and the existing CLI/worker wrapper. New `luka_d066_embedding/direct.py` reuses
the frozen target exporter, native checkpoint readback, fit and comparison
functions; it does not call the preflight runner, stress/parity routines or a
technical fit. Existing worker reference helpers gain only an optional folder
argument (legacy paths/default behavior unchanged). Training, model, targets,
normalization and preflight kernel files are unchanged. CLI and a new direct
queue helper expose distinct DIRECT_*/v1 records and a one-task exact journal.

Resource donor is the completed matched OFFLINE-target weight5 debug fit in
`855c4f8d40b4db7837238c7ac6a3799597466e85`, whose accepted original tier3 sweep
is `007885f7360f4cd28e9f91a690e04f0f4ea0afff`. The new original-source replay
uses that existing `luka_embedding_sweep.debug.check_spec/results` path and
successful accounting, never rewrites its receipts or claims new measurements.
Teacher remains94f56025 LOGIT_D066; parent remains74aa58c5 fixed30. No
Fresh_check imports, copies, jobs or artifacts are affected. See the amended
active plan/contract/runbook for the explicit user waiver of a new preflight,
inherited estimate plus setup allowance, and single-job target setup + science.

## D066 embedding explicit tier3 execution (2026-10-21)

Operational donor/integration baseline: `7ea8b0c693734a26ced3820354e047f4af26c979`.
Extended only `luka_d066_embedding/{campaign,followup}.py`, its existing CLI/
queue helper, tests and documentation. Reuse the already canonical
`jetclass2_delphes.execution.execution_site('sporc_a100')` unchanged. No donor
file migration, Fresh_check import, new scientific kernel, target definition,
teacher/checkpoint substitution or data transformation. The explicit site is
hashed in each new gate/plan; old debug source/artifacts remain pinned. Real
tier3 preflight acceptance is still required before the automatic fit. See the
amended D066 plan/contract/runbook and HANDOFF for verification and safe pending-
job retirement; no remote cancellation or submission by this implementation.

## FullStat U100 auxiliary embedding weight 5 (2026-10-21)

Scientific donor: `fd687181a60c1557f98d7498f577e26d6dbb35a7` (accepted Oscar
FullStat embedding sweep). Additive integration baseline:
`5bf9a00d722c867f652588572cb489451bb1a12e`. No pre-existing runtime is changed;
there is no Fresh_check migration or runtime import.

- New `luka_fullstat_u100_embedding/{banks,training,worker}.py` adapt the
  donor `luka_fullstat_embedding_sweep/{banks,training,worker}.py`: same native
  embedding tap, TRAIN standardization, fixed20 CE+auxiliary training kernel,
  matched initialization/sampling, checkpoint/prediction readback and loss
  reporting. New teacher, coefficient, target parents and contract namespace
  are explicit; donor OFFLINE target artifacts are not rewritten.
- New `source.py`, `campaign.py`, `submission.py` reuse the donor sweep's
  original-source authentication, immutable registration, protected-path
  checks and exact DAG machinery. Original `campaign.accepted` revalidates
  actual measured Oscar evidence and successful journal-bound accounting;
  `luka_fullstat/worker.py` authenticates the selected COARSE_U100 task.
  Registration now has one export and one fresh fit, no new full preflight,
  no old-job mutation and no automatic retry. Permanent submission claims
  and `scouting/{hcwdl_exact_dag_submission,hcwdl_recovery}.py` journals are
  reused unchanged. Old fd687181 runtime byte equality is checked transitively.
- Shared donor modules remain unchanged: `luka_embedding_aux/{model,targets,
  contracts}.py` (head tap, predictor, target moments, predictor seed),
  `luka_fullstat/{cache,recipe,submission}.py`,
  `luka_fullstat_attention/{worker,contracts}.py` (data metadata, allocation,
  environment, validation membership and sealed flags),
  `jetclass2_delphes/{model,runner,reporting,execution}.py`,
  `luka_fullsim/{foundation,preflight_v2,science_campaign}.py`,
  `context_fusion/worker.py`, `data/cache_contracts.py`,
  `cms_proxy_ladder/{contracts,submission}.py` and `provenance.py`.
- CLI/queue/worker environment patterns follow the donor sweep helpers and
  unchanged `sbatch/jetclass2_delphes_common.sh`. Focused tests reuse donor
  `test_luka_fullstat_embedding_sweep.py`, `test_luka_embedding_aux.py` and
  `test_luka_fullstat.py` synthetic fixtures. The new local installed-Weaver
  CUDA miniature exercises actual production model/export/training functions;
  only external scheduler/source/allocation boundaries use fixtures.

New `LUKA_FULLSTAT_U100_EMBEDDING_*/v1` records distinguish this target/weight
from previous experiments. See its active plan, contract and runbook, plus
HANDOFF for test evidence and the boundary of accepted-resource reuse.

## Frozen LOGIT_D066 auxiliary embedding comparison (2026-10-21)

Integration donor: `4a3124e9a3b190754ab195ca7430c00ceebcea78`. All pre-existing
scientific runtime files remain unchanged. No Fresh_check migration or import.

- New `luka_d066_embedding/training.py` adapts the unchanged
  `luka_embedding_sweep/training.py` at that donor: same weight5 objective,
  optimizer, RNG, batching, schedule and native selected-state logic, with a
  new teacher-bound BANK/RECIPE namespace. Its `worker.py` adapts the donor
  sweep worker's fit/checkpoint/readback/validation routines and adds the
  explicit D066-versus-OFFLINE-target comparison, not a new training kernel.
- `contracts.py`, `targets.py`, `campaign.py`, `preflight.py` adapt
  `luka_embedding_aux/{contracts,targets,campaign,preflight}.py` and reuse its
  `model.py` predictor/head tap, TRAIN standardization, zero-auxiliary parity
  and resource admission unchanged. Maximum-support stress is reused from
  `luka_embedding_sweep/preflight.py`. Native model/predict/metrics/selection
  and fixed30 LR remain the donor `jetclass2_delphes` implementations.
- `source.py` reuses `luka_embedding_aux/source.py` control authentication,
  `luka_attention_weight/{source,reference_archive}.py` original-source replay
  and retained-result reader. It additionally authenticates selected.pt from
  original attention commit `94f560259dd04253c0aa47b31dec3aedc105ac14` and
  requires real native validation readback; the archived comparison reader
  alone does not authorize model reuse. Matched OFFLINE-target w5 is replayed
  in the transferred debug source `855c4f8d40b4db7837238c7ac6a3799597466e85`.
- `submission.py` adapts `luka_embedding_aux/submission.py`; `followup.py`
  adapts `luka_embedding_sweep/followup.py` to one gate/one fit and owned-hold
  lifetime. Both reuse `scouting/hcwdl_exact_dag_submission.py` and
  `scouting/hcwdl_recovery.py` immutable journals. `priority.py` reuses pure
  command-to-job binding from `luka_embedding_sweep/debug_retirement.py`,
  but implements holds/releases, **not** retirement/cancellation.
- Thin CLI/queue/worker environment setup follows donor embedding helpers.
  Data/allocation helpers remain `luka_attention/worker.py`; prepared metadata,
  comparison membership, timing and environment helpers remain
  `luka_fullsim/{regular_cache,alljets_data,preflight_v2,regular_submission,foundation}.py`.
  Storage/prediction validation remains `luka_embedding_weight/worker.py`;
  native state publication remains `context_fusion/worker.py`; shared artifact
  and source validation remain `data/cache_contracts.py`, `cms_proxy_ladder/
  {contracts,submission}.py` and `provenance.py` (all at the integration donor).
  Focused tests reuse synthetic cache/parent fixtures from `test_luka_attention.py`,
  128-wide tiny-model configuration from `test_luka_embedding_aux.py`, and
  retained archives from `test_luka_attention_reference_archive.py`.
  No old scientific artifact or contract is rewritten.

## FullStat ordered-controller portability recovery (2026-10-21)

Operational donor: `b356267787a1ce543db6d327781008315608b12c`, specifically
`operations/fullstat_embedding_order.py`, its queue wrapper and tests. V2 adds
the missing-wrapper Linux pidfd adapter, early capability check and exact
pre-signal-failure recovery, without copying or altering any scientific kernel.
The scientific donor remains `fd687181a60c1557f98d7498f577e26d6dbb35a7` in its
original worktree. New standalone Linux smoke invokes the same real-kernel
regression test, not a separate implementation. The x86_64 syscall identifiers
were checked against upstream Linux v5.14 `arch/x86/entry/syscalls/syscall_64.tbl`.
No Fresh_check migration or runtime import.

## FullStat embedding scheduling-only handover (2026-10-21)

Scientific donor: `fd687181a60c1557f98d7498f577e26d6dbb35a7`.
New `operations/fullstat_embedding_order.py` explicitly loads the original
worktree's unchanged `luka_fullstat_embedding_sweep/{campaign,contracts,
followup,submission}.py` for gate/source/admission validation and measured
command generation. It reuses unchanged `luka_fullstat/submission.py` site
checks and `scouting/hcwdl_exact_dag_submission.py`/`hcwdl_recovery.py` journal
semantics. No scientific kernel,
weights, cache or Fresh_check code is copied. The new ordered plan retains all
original command arguments except the six explicitly authorized scheduling
dependencies and ordering; original scientific artifacts stay immutable.
The thin CLI and queue helper are new; the original queue helper's environment
setup is reused through the unchanged original `jetclass2_delphes_common.sh`.


## Trial embedding sweep tier3-to-debug transfer (2026-10-21)

Integration baseline: `fd687181a60c1557f98d7498f577e26d6dbb35a7`.
Measured scientific donor: `007885f7360f4cd28e9f91a690e04f0f4ea0afff`.
Existing runtime remains unchanged; no Fresh_check files are imported or moved.

- `luka_embedding_sweep/debug.py` adapts `luka_embedding_ladder/tier3.py`
  (at the measured donor above) for the explicitly registered reverse site
  transfer. Source replay uses unchanged `luka_embedding_ladder/source.py`;
  accepted sweep and exact journal replay use the original source's
  `luka_embedding_sweep/{campaign,submission,followup}.py`.
- Its science/results orchestration adapts `luka_embedding_sweep/worker.py`
  at that donor. Scientific `fit`, `validate_fit`, target loading, comparisons,
  storage, recipe, model and training are reused unchanged. Only allocation
  uses debug; all FIT records still bind the original measured ARM.
- `debug_retirement.py` is new exact-ID cancellation orchestration, reusing
  `scouting/hcwdl_exact_dag_submission.py`, `scouting/hcwdl_recovery.py` and
  the unchanged `luka_alternating_gate/{campaign,submission}.py` at that donor.
  Authorization is limited to the two user-selected original spec hashes.
- New CLI/helper adapt `scripts/luka_embedding_sweep.py` and
  `scripts/queue_luka_embedding_ladder_tier3.sh`; the worker adapts
  `sbatch/run_luka_embedding_sweep.sh`. No old helper/worker is modified.
- New tests adapt `tests/test_luka_embedding_ladder_tier3.py` and sweep tests,
  with unchanged fixtures from `test_luka_embedding_aux` and
  `test_luka_attention`. New plan, DEBUG/v1 contract and runbook document the
  transfer; original SWEEP/v1 and ladder transfer contracts are not broadened.

## FullStat standalone CE embedding sweep on Oscar (2026-10-20)

Donor/integration commit: `007885f7360f4cd28e9f91a690e04f0f4ea0afff`.
New namespace `src/hlt_classification/luka_fullstat_embedding_sweep/` and
`LUKA_FULLSTAT_EMBEDDING_SWEEP_*/v1` contracts. This is the 3M/500k, 20-epoch
Oscar experiment, not the older trial/fixed-30 tier3 sweep. All existing
runtime files remain unchanged; no Fresh_check migration or runtime import.

- `__init__.py`, `contracts.py`, `campaign.py`, `banks.py`, `preflight.py` and
  `worker.py` adapt corresponding `luka_fullstat_embedding/*.py` files.
  New registration has seven matched weights (0/2/10/50/100/500/1000), CE=1,
  no logit KD or D033 targets, TRAIN-only OFFLINE embeddings and selected
  native HLT checkpoint publication. Each weight gets a measured technical
  full epoch before science. RECIPE's original base recipe is used only for
  optimizer/schedule provenance; its legacy KD fields are not loss inputs.
- `source.py` adapts `luka_fullstat_attention/source.py`; original pinned
  FullStat recovery `9bc3cae082915259ad569a46eefa348ddce74ea3` is reauthenticated
  in its own worktree, with only M0HLT/OFFLINE reference metrics and OFFLINE
  teacher weights imported. No D033 checkpoint or bank is used for training.
- `training.py` adapts `luka_embedding_sweep/training.py`; native model,
  matched RNG/sampler and CE/SmoothL1 semantics remain unchanged, while schedule
  and artifact identities are FullStat fixed-20 rather than trial fixed-30.
- `submission.py` adapts `luka_fullstat_replace_gate/submission.py`: exact
  one-gate or seven-independent-fit plus CPU-aggregate plans, new `lukafses_`
  prefix, no requeue, personal-storage declaration and measured Oscar times.
- `followup.py` adapts `luka_embedding_sweep/followup.py`: explicit bounded
  one-preflight/eight-science authorization, detached submit-host polling,
  scheduler identity plus artifact/resource authentication, exact measured
  dry review, permanent claims, and no ambiguous/partial-submission retry.
- CLI/queue/Slurm files adapt `scripts/luka_fullstat_embedding.py`,
  `scripts/queue_luka_fullstat_embedding.sh`, and
  `sbatch/run_luka_fullstat_embedding.sh`, with the opt-in automatic follow-up
  pattern above. New plan/contract/runbook adapt their FullStat embedding
  and trial sweep counterparts; they do not revise either old campaign.
- Tests adapt `tests/test_luka_fullstat_embedding.py` and
  `tests/test_luka_embedding_sweep_followup.py`. Shared fixtures come from
  `test_luka_attention.py`, `test_luka_embedding_aux.py` and
  `test_luka_fullstat.py`. New coverage includes all seven exact losses,
  early-best retention despite poor metrics, real synthetic ROOT preparation,
  local installed-Weaver CUDA preflight/science miniatures and auto-submit
  failure/duplicate/uncertain-journal injection.

Direct unchanged runtime reuse at the same donor commit:

- `luka_embedding_aux/{model,targets,preflight,contracts}.py` for the native
  auxiliary predictor, TRAIN standardization and FP32/BF16 parity;
- `luka_fullstat/{cache,recipe,submission,contracts}.py` for authenticated
  views, twenty-epoch schedule, Oscar site checks and parent contract;
- `luka_fullstat_attention/{worker,contracts}.py` for generic environment/view
  authentication and sealed FullStat flags; `luka_fullstat_attention_fast/
  preflight.py::selected_cache` for technical miniatures only;
- `luka_fullsim/{foundation,preflight_reuse,science_campaign,preflight_v2}.py`
  for source, accounting, PhaseLog and memory measurements;
- `jetclass2_delphes/{model,runner,acceptance,execution,reporting,
  dzfix_fusion_model}.py`, `context_fusion/worker.py::save_state`,
  `cms_proxy_ladder/{contracts,submission}.py`, `data/cache_contracts.py`,
  `provenance.py`, and `scouting/{hcwdl_exact_dag_submission,hcwdl_recovery}.py`
  for native ParT, metrics, parity backend, file/content contracts and journals.
  Reusing these utilities does not instantiate fusion or a logit teacher.

Existing datasets, parent selected checkpoints, split membership, source
worktrees, running campaigns and unrelated dirty documentation are preserved.

## Extended no-KD embedding sweep, eight tier3 weights (2026-10-20)

Donor/integration commit: `bf360343974121397a568fdb635a2c1c1011bc90`.
New namespace `src/hlt_classification/luka_embedding_sweep/`, with separate
`LUKA_EMBEDDING_SWEEP_*/v1` contracts. Original three-weight and auxiliary
runtime files remain unchanged; no Fresh_check migration or import.

- `__init__.py`, `contracts.py`, `source.py`, `campaign.py`, `training.py`,
  `preflight.py`, `submission.py`, and the fit/result publication portions of
  `worker.py` adapt their corresponding `luka_embedding_weight/*.py` files.
  Only the new registration allows 3/5/20/50/100/200/500/1000. Same scientific
  loop, matched seeds, optimizer, schedule, target semantics and checkpoint
  rule. Training logs additionally show best AUC and its epoch.
- `luka_embedding_weight/worker.py::{storage,validate_predictions,
  snapshot_references,comparison,load_bank}` are directly reused unchanged,
  not duplicated. Targets remain the original AUX/v1 bank.
- Direct unchanged reuse: `luka_embedding_aux/model.py` (predictor/native state),
  `targets.py` (authenticated targets), `contracts.py` (base recipe/node),
  `source.py` (original controls), `preflight.py` (parity/resource admission);
  `luka_embedding_ladder/source.py::import_embedding` authenticates the old0.3
  bank in its pinned worktree. Existing data, metric, cache, source, execution,
  PhaseLog and exact-DAG submission primitives remain shared.
- CLI, queue helper and Slurm worker adapt `scripts/luka_embedding_weight.py`,
  `scripts/queue_luka_embedding_weight.sh`, `sbatch/run_luka_embedding_weight.sh`.
  Eight independent science commands, distinct `lukaes_` job prefix and
  authorization. CLI adds AUC recovery display.
- The opt-in `luka_embedding_sweep/followup.py` adapts
  `luka_gating_cold/followup.py` at the same donor/integration commit above;
  `tests/test_luka_embedding_sweep_followup.py` adapts
  `tests/test_luka_gating_followup.py`. Exact claims, scheduler identity,
  bounded wait, original-source admission, measured-plan review and journal
  validation are retained; scope is one full gate plus eight independent tier3
  weights. New `auto` helper/CLI integration uses distinct SWEEP FOLLOWUP/v1
  artifacts and approval. No existing controller or scientific runtime changes.
- Tests adapt `tests/test_luka_embedding_weight.py`, retaining tiny cache/model
  fixtures from `tests/test_luka_attention.py` and `test_luka_embedding_aux.py`.
  New plan/contract and runbook adapt the original sweep's documentation.

No historical artifact, split, selected checkpoint, bank, source worktree or
running job is changed. Existing0.75/1.5/10 results keep their original command.

## Trial ALLJETS alternating privileged/self gates on debug (2026-10-20)

Donor/integration commit: `e95c0d83a60b75cfc9acca4aaf377740fde8930e`.
Additive namespace `src/hlt_classification/luka_alternating_gate/`; existing
scientific/runtime modules are unchanged. New artifacts use the separate
`LUKA_ALTERNATING_GATE_*/v1` namespace.

- `contracts.py`, `campaign.py`, `source.py`, `training.py`, `worker.py`,
  `preflight.py`, `storage.py`, `submission.py` adapt corresponding
  `src/hlt_classification/luka_gating_cold/*.py` donor files. The new graph
  alternates TEACH/SELF at every D rung, registers three matched six-fit arms,
  uses debug rather than tier3, and binds explicit teacher coordinates.
- `banks.py` adapts `luka_gating/banks.py`: new bank identity binds the complete
  teacher node, factor export covers all three GATE-arm TEACH nodes, and factors
  are loaded once into RAM with prefix offsets rather than shuffled mmap opens.
- Direct unchanged runtime reuse: `luka_gating/model.py` (bounded gate and
  centered loss), `luka_gating/preflight.py` (installed parity/stress),
  `luka_gating/contracts.py` (gate geometry/backend policy/input labels),
  `luka_attention/worker.py` (authenticated inputs/allocation),
  `luka_attention/banks.py` (slot alignment), `luka_attention_weight/source.py`
  (original-worktree fixed30 replay), and `luka_attention_weight/worker.py`
  (original50k report membership). Shared batch128/fixed30 optimizer, metrics,
  content/file contracts and exact-DAG journal implementations are reused.
- Thin CLI/queue/Slurm files adapt `scripts/luka_gating_cold.py`,
  `scripts/queue_luka_gating_cold.sh`, `sbatch/run_luka_gating_cold.sh`.
  No automatic follow-up is carried over; gate and science require separate
  reviewed exact-plan authorization. New tests adapt `tests/test_luka_gating_cold.py`
  and reuse the `tests/test_luka_attention.py` tiny cache/model fixtures.

No code or data is migrated from Fresh_check. No original campaign is rewritten.

## FullStat final-rung cold privileged/self-gate comparison (2026-10-20)

Donor/integration commit: `1dff4e5767e66fb5f355cf5d592626ecffcf73ea`.
Additive namespace `src/hlt_classification/luka_fullstat_replace_gate/`;
all pre-existing runtime files remain unchanged.

- `contracts.py` adapts `luka_fullstat_embedding/contracts.py` and
  `luka_gating_cold/contracts.py`: distinct FullStat v1 identities, three cold
  fits, no trial holdout flags or warm/frozen policy.
- `source.py` reuses `luka_fullstat_embedding/source.py::authenticate` and
  `luka_fullstat_attention/source.py` original-worktree replay, plus
  `luka_fullsim/foundation.py::_source` / `provenance.py`.
- `training.py` adapts `luka_gating_cold/training.py` and
  `luka_fullstat_embedding/training.py`; unchanged `luka_gating/model.py`,
  `luka_gating/banks.py::{forward,predict,paired,alignment,counts}`,
  `luka_gating_cold/training.py::{new_student,initial_digest}` supply gate
  math, strict HLT/context boundaries and fresh initialization.
- `banks.py` adapts the numerical layout/export in `luka_gating/banks.py`,
  but publishes NEW FullStat COMPACT_BANK/BANK contracts. Trial scientific
  flags are not reused. Contiguous resident gathering replaces random mmap
  shard lookup. Original D033 probabilities use unchanged
  `jetclass2_delphes/banks.py`.
- `campaign.py`, `worker.py`, `preflight.py`, `submission.py`, `checkpoints.py`
  adapt the corresponding `luka_fullstat_embedding/` files. Reuse
  `luka_fullstat_attention/worker.py::{data,metadata,execution,validation_arrays}`,
  FullStat cache/recipe, `jetclass2_delphes/{model,runner,reporting,acceptance,
  execution,dzfix_fusion_model}.py`, `luka_gating/preflight.py::{parity,stress}`,
  `luka_fullsim/preflight_v2.py` phase/memory measurements,
  `luka_fullstat_attention_fast/checkpoints.py::rng_state`,
  `context_fusion/worker.py::save_state`, cache/file contracts,
  `luka_fullsim/science_campaign.py::preflight_accounting`,
  `luka_fullstat/submission.py` site checks and
  `scouting/{hcwdl_exact_dag_submission,hcwdl_recovery}.py` exact journals.
- CLI/queue/Slurm files adapt `luka_fullstat_embedding` entry points.
  Tests reuse `test_luka_attention.py`, `test_luka_fullstat_embedding.py` and
  `test_luka_fullstat.py` synthetic/real-ROOT fixtures.

The selected original FullStat D033 parent remains source
`9bc3cae082915259ad569a46eefa348ddce74ea3`. No `Fresh_check` runtime imports,
new external libraries or edits to original scientific kernels.
See [plan](plans/LUKA_FULLSTAT_REPLACE_GATE_PLAN.md) and
[queue runbook](LUKA_FULLSTAT_REPLACE_GATE_RUNBOOK.md).

## Debug-accepted embedding ladder -> tier3 science (2026-10-20)

Additive donor/integration commit `0342951d7a11b959c3baa1611f9e581f6004c9f6`.
New file `luka_embedding_ladder/tier3.py` adapts the donor's
`luka_embedding_ladder/{campaign,submission,worker}.py` operational orchestration
without editing them. It reuses their scientific helpers for fit, data,
logit banks, target joins, validation, prediction, comparison and owned
retirement. No optimizer/model/loss/view code changes.

- Original gate authentication uses `luka_embedding_ladder/source.py::replay`
  to call `campaign.py::evidence` in the original pinned worktree, including
  accepted journal and successful Slurm accounting.
- Source/snapshot authentication reuses `luka_fullsim/foundation.py::_source`
  and `provenance.py`; both active source and unchanged measured runtime bind
  the transfer. Only known additive FullStat/weight namespaces may intervene.
- Same-site policy, GPU and allocation checks reuse
  `jetclass2_delphes/execution.py` and `luka_attention/worker.py::execution`.
- Exact submissions reuse `scouting/{hcwdl_exact_dag_submission,hcwdl_recovery}.py`,
  `luka_fullsim/regular_submission.py` and `cms_proxy_ladder/submission.py`.
- New thin CLI, queue helper and Slurm worker adapt their donor
  `luka_embedding_ladder` counterparts. Tests reuse
  `test_luka_embedding_ladder.py` / `test_luka_embedding_aux.py` tiny fixtures.

New TIER3 SPEC/PLAN/SCIENCE/RESULTS contracts are distinct from original ladder
v1 gates/reports; those remain unchanged and are not rewritten.
See [tier3 transfer runbook](LUKA_EMBEDDING_LADDER_TIER3_RUNBOOK.md).

## Trial single-HLT embedding-weight sweep on tier3 (2026-10-20)

Additive donor/base: `5ed50964085e9bf364030764bbbc755179c445ae`.
Tier3 integration source pin: `ed129fa95a885f0f3b884b2dc861b1776327bc82`.
The intervening source change only adds `luka_fullstat_embedding/`; none of
this sweep's reused runtime files changes. The same strict source guard remains.
New namespace: `src/hlt_classification/luka_embedding_weight/`.
Existing runtime, original0.3 experiment and embedding ladder are unchanged.
Original matched CE/OFFLINE parent remains
`74aa58c582ff6f497a2c6d91c503f5489fac3e9a`.

- `contracts.py`, `campaign.py`, `training.py`, `worker.py`, `preflight.py`,
  `submission.py` adapt the corresponding `luka_embedding_aux/` donor files.
  New contracts register three explicit CE+embedding coefficients, one shared
  preflight and three independent fixed30 science tasks, not KD or a ladder.
- Reuse `luka_embedding_aux/model.py` native predictor/tap/RNG isolation,
  `targets.py::Bank`, `preflight.py::{parity,admission}`, and
  `source.py::authenticate` unchanged. `luka_embedding_ladder/source.py::
  import_embedding` authenticates the completed original target bank and0.3
  comparison in its pinned source. No target copy or particle reconstruction
  was added.
- Native model, predict, metrics, selection and LR helpers remain in
  `jetclass2_delphes/{model,runner,reporting,batch128_30e_recipe}.py`;
  technical helpers reuse `acceptance.py::installed_parity` and
  `dzfix_fusion_model.py::parity_backend`, not fusion supervision.
- Authenticated D000 loading and allocation reuse `luka_attention/worker.py::
  {data,execution}`. Resource/phase reporting uses
  `luka_fullsim/preflight_v2.py`; selected checkpoint serialization uses
  `context_fusion/worker.py::save_state`. Cache metadata, original50k mapping
  and accounting use `luka_fullsim/{regular_cache,alljets_data,science_campaign}.py`.
- Submission reuses `cms_proxy_ladder/submission.py::_walltime`,
  `luka_fullsim/regular_submission.py` site/environment checks and
  `scouting/{hcwdl_exact_dag_submission,hcwdl_recovery}.py` exact journals.
  Integrity/atomic publication reuse `data/cache_contracts.py` and
  `cms_proxy_ladder/contracts.py`.
- Thin CLI and shell workers adapt `scripts/luka_embedding_aux.py`,
  `scripts/queue_luka_embedding_aux.sh`, `sbatch/run_luka_embedding_aux.sh`.
  Tests reuse `test_luka_attention.py` and `test_luka_embedding_aux.py` fixtures.
  No `Fresh_check` runtime imports.

See [plan](plans/LUKA_EMBEDDING_WEIGHT_PLAN.md),
[contracts](contracts/LUKA_EMBEDDING_WEIGHT.md) and
[tier3 queue runbook](LUKA_EMBEDDING_WEIGHT_RUNBOOK.md).

## FullStat final-rung embedding prediction on Oscar (2026-10-20)

Additive donor/base: `5ed50964085e9bf364030764bbbc755179c445ae`.
New namespace: `src/hlt_classification/luka_fullstat_embedding/`; existing
runtime files are unchanged. Original FullStat scientific parent remains
`9bc3cae082915259ad569a46eefa348ddce74ea3`, including selected D033 epoch 17
and the pure OFFLINE checkpoint. No running attention artifacts are reused.

- `contracts.py`, `source.py`, `campaign.py`, `submission.py`, `worker.py` and
  `preflight.py` adapt `luka_fullstat_attention/{contracts,source,campaign,
  submission,worker,preflight}.py` at the donor. Original-source authentication,
  environment/allocation validation, cache metadata and validation references
  reuse those helpers unchanged. New v1 contracts distinguish embedding targets
  and final-rung-only science from attention and trial fixed30 artifacts.
- `banks.py` reuses `luka_embedding_aux/model.py::head_input`,
  `luka_embedding_aux/targets.py::standardize` and
  `jetclass2_delphes/banks.py::load_bank`. It exports fixed pure-OFFLINE TRAIN
  vectors and authenticates the completed original D033 probability bank.
- `training.py` adapts `luka_embedding_ladder/training.py`, reusing its
  `objective` and `luka_embedding_aux/model.py::new_model` unchanged. Native
  optimizer/input/predict/selection semantics come from
  `jetclass2_delphes/{model,runner,reporting}.py` and the unchanged FullStat
  `recipe.py`; this namespace uses 20 epochs, not the donor trial's 30.
- `checkpoints.py` adapts `luka_fullstat_attention_fast/checkpoints.py` and
  reuses its RNG serialization. The preflight reuses
  `luka_fullstat_attention_fast/preflight.py::selected_cache`,
  `luka_embedding_aux/preflight.py::parity`,
  `jetclass2_delphes/acceptance.py::installed_parity` and
  `jetclass2_delphes/dzfix_fusion_model.py::parity_backend`.
  These are reusable technical helpers, not attention/fusion supervision.
- Workers also reuse `luka_fullstat/cache.py`,
  `luka_fullsim/preflight_v2.py` phase/resource reporting,
  `context_fusion/worker.py::save_state`, reusable
  `data/cache_contracts.py` / `cms_proxy_ladder/contracts.py` integrity helpers,
  `luka_fullsim/science_campaign.py::preflight_accounting`,
  `luka_fullstat/submission.py` site checks and
  `scouting/{hcwdl_exact_dag_submission,hcwdl_recovery}.py` exact journals.
- Thin CLI/queue/sbatch wrappers adapt
  `scripts/luka_fullstat_attention_fast.py`,
  `scripts/queue_luka_fullstat_attention_fast.sh` and
  `sbatch/run_luka_fullstat_attention_fast.sh`; no resume command was ported.
  New tests reuse `test_luka_attention.py` / `test_luka_embedding_aux.py`
  synthetic native-model fixtures. No `Fresh_check` runtime import was added.

See [plan](plans/LUKA_FULLSTAT_LAST_EMBEDDING_PLAN.md),
[contracts](contracts/LUKA_FULLSTAT_LAST_EMBEDDING.md), and
[Oscar runbook](LUKA_FULLSTAT_LAST_EMBEDDING_RUNBOOK.md).

## Fixed-OFFLINE embedding + logit ladders (2026-10-20)

Additive implementation donor/base:
`991c9d44161552c3ff8ed65eafd59933217f08aa`. New namespace:
`src/hlt_classification/luka_embedding_ladder/`. Original fixed30 scientific
teacher/control source remains `74aa58c582ff6f497a2c6d91c503f5489fac3e9a`;
completed single-stage embedding source is authenticated from its immutable
spec (user's existing run: `cfb3a1587a6739b1949be9e65ac80f7ba9f8ead9`).

- `contracts.py`, `campaign.py`, `source.py`, `preflight.py`, `worker.py`,
  `submission.py` adapt the corresponding `luka_embedding_aux/` modules at
  the donor commit. New v1 contracts bind two separately authorized five-rung
  arms, immediate-teacher logit targets and one fixed OFFLINE embedding bank.
  `source.py` replays original-source validation rather than migrating or
  weakening old receipts; native/input core files must remain byte-identical.
- `training.py` adapts `luka_embedding_aux/training.py` and reuses its
  `model.py::{new_model,AuxiliaryParT}` unchanged. Native KD and selection are
  `jetclass2_delphes/{model.py,runner.py,batch128_30e_recipe.py}`. No donor
  runtime edits, direct feature alignment, attention targets or warm starts.
- `data.py` reuses `luka_fullsim/regular_cache.py`, `stage2.py`,
  `preflight_cache.py` and `luka_embedding_aux/targets.py::Bank`. Entirely
  deleted U000/U050 caches are reconstructed in bounded RAM with original
  assignments and fingerprints, never written back. `banks.py` is a new
  compact RAM-only probability descriptor around the original native predictor.
- `worker.py` reuses `context_fusion/worker.py::save_state`,
  `luka_embedding_aux/worker.py` storage/reference/prediction validators,
  `luka_attention/worker.py::execution`, and
  `luka_attention_weight/retention.py` exact owned-file deletion/receipts.
  Those reusable ownership receipts retain their existing versioned namespace;
  their owner hash is the new gate. No old payload is retired by this experiment.
- Thin CLI/queue/sbatch wrappers adapt `scripts/luka_embedding_aux.py`,
  `scripts/queue_luka_embedding_aux.sh`, and `sbatch/run_luka_embedding_aux.sh`.
  New tests reuse synthetic native-model fixtures from `test_luka_attention.py`,
  `test_luka_embedding_aux.py` and real ROOT fixtures from
  `test_luka_fullsim_stage2.py` / `test_luka_fullsim_alljets.py`.

See [plan](plans/LUKA_EMBEDDING_LADDER_PLAN.md),
[contracts](contracts/LUKA_EMBEDDING_LADDER.md),
[runbook](LUKA_EMBEDDING_LADDER_RUNBOOK.md). All old runtime and schemas remain
unchanged; no runtime Fresh_check imports or remote submission during development.

## Luka whole-particle replacement ladders (2026-10-19)

Additive implementation donor/base:
`901535f026d32868e19c159a073947dfdbbbdd3c`. Scientific teacher/control parent
remains fixed30 `74aa58c582ff6f497a2c6d91c503f5489fac3e9a`; interpolation
comparison source remains `94f560259dd04253c0aa47b31dec3aedc105ac14`.
New namespace: `src/hlt_classification/luka_particle_swap/`.

- `contracts.py`, `campaign.py`, `source.py`, `training.py`, `worker.py`,
  `retention.py`, `preflight.py`, `submission.py` adapt the corresponding
  `luka_attention_weight/` files, under separate `LUKA_PARTICLE_SWAP_* /v1`
  contracts. Registration changes to logit0/attention0.1 on debug; new input
  identity binds atomic swap views. Fixed30 training numerics are preserved.
- New `views.py` implements the balanced nested whole-record replacement;
  exact U100 delegates to `cms_proxy_ladder/views.py::build_view` and native
  D000 returns its HLT particles. Reuses `validate_pairing` and
  `cms2jc2_response/bridge.py::Particles`; no donor matching algorithm changes.
- New `data.py` reuses `luka_fullsim/regular_cache.py::{metadata,selected_file_rows}`,
  `stage2.py::{load_assignment,ViewMeter,build_inputs}`, `particles.py::physical`,
  `preflight_cache.py::_block` and `alljets_data.py::comparison_indices`.
  No persistent particle-cache migration and no runtime Fresh_check import.
- `banks.py` subclasses unchanged `luka_attention/banks.py::Bank`, keeping
  its authenticated ragged FP32 schema. It adapts the resident-copy pattern
  from `luka_fullstat_attention_fast/banks.py` at the donor commit, but does
  not import FullStat data, artifacts, prefetching or its scientific recipe.
- Unchanged native model/readout/loss/parity remain in `luka_attention/` and
  `jetclass2_delphes/`. Imported comparison archive proof is the existing
  `luka_attention_weight/reference_archive.py` (comparison-only; no acceptance
  of discarded banks). Original-source anchor replay uses
  `luka_gating/source.py::PARENT_CHECK`.
- Thin `scripts/luka_particle_swap.py`, `scripts/queue_luka_particle_swap.sh`
  and `sbatch/run_luka_particle_swap.sh` adapt the weight-sweep wrappers.
  `tests/test_luka_particle_swap.py` adapts lifecycle/guard fixtures from
  `tests/test_luka_attention_weight.py`; new view tests reuse the real ROOT
  fixtures from stage2/ALLJETS tests. No old tests or runtime files are edited.

See [plan](plans/LUKA_PARTICLE_SWAP_PLAN.md),
[contract](contracts/LUKA_PARTICLE_SWAP.md),
[runbook](LUKA_PARTICLE_SWAP_RUNBOOK.md). Exact old source is frozen at the
donor base; old schemas and executions are not relabelled or resumed.

## FullStat science-identical attention execution recovery (2026-10-19)

Additive donor:`cfb3a1587a6739b1949be9e65ac80f7ba9f8ead9`. Bank/export execution
parent remains`62c9b2df91eb3bda4fd4df4edc75c5f810a48c55`, with the original
FullStat scientific parent`9bc3cae082915259ad569a46eefa348ddce74ea3`.

- `luka_fullstat_attention_fast/banks.py` reuses every validation from
  `luka_fullstat_attention/banks.py::Bank`, then copies authenticated FP32
  shards into resident ragged storage. Adds ordered, bounded CPU lookahead.
- `training.py` adapts `luka_fullstat_attention/training.py` without modifying
  its original forward/loss/finite guards/optimizer/sampler/selection math.
  Adds diagnostic-only synchronization reduction, progress and checkpoint hooks.
- `checkpoints.py` uses `context_fusion/worker.py::save_state` and existing
  immutable-publication/file-reference helpers; adds same-execution epoch/RNG
  resumption rather than importing a different trained model.
- `campaign.py` adapts original-source replay from
  `luka_fullstat_attention/source.py`; freezes all old source and authenticates
  only the completed export, not the cancelled duplicate logit fit.
- `preflight.py` reuses installed native/readout parity/stress from
  `luka_attention/preflight.py`, authentic FullStat caches, and old training as
  the equality reference. New measurement uses the full exported bank and
  globally shuffled batches, plus complete validation, not a tiny-bank scaling.
- `worker.py`, `submission.py` and thin CLI/Slurm/helpers adapt the original
  FullStat attention publication, exact-DAG journals and environment setup;
  new contracts identify the attention-only operational recovery explicitly.

No donor files were moved or changed. No `Fresh_check` runtime dependency.

## Luka HLT auxiliary offline-embedding prediction (2026-10-19)

Additive `luka_embedding_aux/` uses in-repository donor
`62c9b2df91eb3bda4fd4df4edc75c5f810a48c55`. Its scientific parent is the
completed trial ALLJETS fixed30 campaign at
`74aa58c582ff6f497a2c6d91c503f5489fac3e9a`, not Oscar FullStat.

- `model.py` adapts the native `mod.fc` input-hook/small-head pattern from
  `jetclass2_delphes/offline_aux/model.py`. The new target is the frozen
  OFFLINE jet embedding, not that donor's handcrafted structures. Head-only
  initialization seeds the CPU generator inside a restored RNG context without
  changing CUDA dropout RNG. Native state keys/inference remain unchanged.
- `training.py` adapts `jetclass2_delphes/runner.py::train_kernel` with the
  unchanged `batch128_30e_recipe.py` optimizer/schedule/selection and adds only
  CE plus the registered auxiliary prediction loss. Zero-coefficient technical
  parity is tested against the original native CE kernel.
- `targets.py` reuses `luka_fullsim/regular_cache.py::selected_file_rows`,
  `particles.py::physical`, `stage2.py::ViewMeter/build_inputs`, and
  `preflight_cache.py::_block` for a TRAIN-only OFFLINE reconstruction. No
  matching or durable cache rebuild. New FP32 embedding banks have their own
  role/identity/normalization contracts, not probability/attention-bank schemas.
- `source.py` adapts original-worktree replay from `luka_gating/source.py`,
  restricted to M0HLT/OFFLINE, and source guards from `luka_gating_cold/source.py`.
  `campaign.py` and `submission.py` adapt the cold-gate lifecycle and exact-DAG
  journal pattern to one preflight and one scientific fit.
- `preflight.py` adapts native parity/stress logic from
  `luka_attention/preflight.py`; `worker.py` reuses its unchanged
  `data/execution` helpers, native prediction/metrics, immutable publication,
  `context_fusion/worker.py::save_state`, and fullsim PhaseLog/memory sampling.
  No attention surfaces or banks are read or produced.
- CLI and queue/worker shells adapt the cold-gate equivalents. Tests reuse
  synthetic cache/ROOT fixtures from `test_luka_attention.py`,
  `test_luka_fullsim_stage2.py`, and `test_luka_fullsim_alljets.py`.

New contracts: `LUKA_EMBEDDING_AUX_*/v1`. No existing scientific runtime edits,
external donor, Fresh_check import, parent mutation, deletion or remote job
submission is introduced. See the new plan, contract and runbook.

## FullStat final-rung attention comparison (2026-10-19)

The additive `luka_fullstat_attention/` package uses in-repository donor
`a1d3a2c6caa20871ce2d0b3c1525fd347fd27df6`. Scientific parent is the completed
Oscar FullStat recovery at `9bc3cae082915259ad569a46eefa348ddce74ea3`, not the
trial ALLJETS/fixed30 attention campaigns.

- `banks.py` adapts `luka_attention/banks.py`: a distinct artifact namespace,
  exclusively frozen D033 TRAIN export and D000 consumption, one retained
  shared bank. Ragged FP32 shards and bounded read-only mmap LRU are unchanged.
- `training.py` adapts `luka_attention/training.py`, substituting the unchanged
  `luka_fullstat/recipe.py` schedule (20 epochs, batch128), and restricting fits
  to the paired LOGIT/ATTN D000 arms. No optimizer/backbone/loss kernel change.
- `worker.py` adapts `luka_attention/worker.py::fit` and completion/result
  validation; it uses existing FullStat prepared caches, keeps both selected
  checkpoints and predictions, and never imports trial retention semantics.
- `source.py` adapts original-worktree replay from `luka_attention/source.py`
  with FullStat-only parents, complete parent-DAG authentication, unchanged
  parent source bytes and epoch17 D033 anchoring. Existing runtime is frozen
  against the donor; all new runtime additions are explicitly allowlisted.
- `campaign.py`, `submission.py`, CLI and queue/worker shells adapt the
  corresponding attention modules with Oscar site validation, a new gate and
  four-task science DAG. Submission/journal helpers are reused unchanged.
- `preflight.py` reuses `luka_attention/preflight.py::{parity,stress}` and
  `luka_fullsim/preflight_v2.py::{PhaseLog,memory_snapshot}`. New checks cover
  all required ordinary-cache payloads, ordered D033/D000 slots, distinct
  TRAIN/validation support, full-bank size, native zero-loss training parity,
  both measured one-pass fits and conservative full-population projections.
- Tests reuse fixtures from `tests/test_luka_attention.py` and
  `tests/test_luka_fullstat.py`; synthetic ROOT/cache bytes and local CUDA
  production-worker smoke runs are not a real Oscar acceptance claim.

Unchanged reusable code includes `luka_attention/model.py`, native ParT,
training/reporting/selection kernels, FullStat source/cache validation,
`context_fusion/worker.py::save_state`, exact-DAG submission/journals, and
`luka_fullstat/submission.py::{check_site,environment}`. No external donor,
Fresh_check runtime import, existing scientific-module edit, final-test
inference, parent modification or automatic deletion is introduced.
New contracts are `LUKA_FULLSTAT_LAST_ATTENTION_*/v1`; see the new plan,
contract and runbook rather than interpreting old trial artifacts differently.

## Cold privileged-gate tier3 ladders and ephemeral targets (2026-10-18)

New `luka_gating_cold/` sources are an additive implementation pinned to this
repository at `84fcbf25529091a2f32daf96bf8b910dec50dc3f`:

- `training.py` adapts `luka_gating/training.py`, replacing warm-state input
  with fresh seeded construction and a state digest. Optimizer, fixed30
  schedule, batch128, losses and full-validation selection are unchanged.
- `contracts.py`, `campaign.py`, `submission.py`, `followup.py`, thin CLI,
  queue helper and worker shell adapt their `luka_gating` counterparts.
  Contracts are distinct `LUKA_COLD_GATE_* /v1`; the graph has12 cold fits,
  no staged freeze and no persistent export jobs.
- `worker.py` adapts selected-state and prediction checks from
  `luka_gating/worker.py`; it imports unchanged `luka_attention/worker.py`
  data/execution and `luka_attention_weight/worker.py::original_indices`.
- `storage.py` adapts `luka_attention_weight/retention.py` to a new ownership
  namespace and per-fit target lifetime. Selected checkpoints are NOT retired.
- `preflight.py` reuses `luka_gating/preflight.py::{parity,stress}` and
  `luka_fullsim/preflight_v2.py::{PhaseLog,memory_snapshot}`. It adds full cold
  graph rehearsal, seed witnesses and separately measured fit budgets.
- `source.py` wraps unchanged `luka_attention_weight/source.py::authenticate`
  (original `luka_gating/source.py::PARENT_CHECK`) with the new anchor schema.
  Original scientific parent remains fixed30
  `74aa58c582ff6f497a2c6d91c503f5489fac3e9a`.
- Tests adapt `tests/test_luka_gating{,_followup}.py`, reusing synthetic cache
  and tiny-model fixtures from `tests/test_luka_attention.py`.

Unchanged runtime dependencies include `luka_gating/{model,banks,contracts}`,
`jetclass2_delphes/{model,runner,reporting,execution,batch128_30e_recipe}`,
`context_fusion/worker.py::save_state`, source/cache contracts, and the existing
exact-DAG submission/journal and SPORC site-check APIs. No external donor,
Fresh_check runtime import, old scientific-module edit or retrospective
cleanup is introduced. Detailed plan/contract/runbook are new files.

## Attention-weight tier3 execution registration (2026-10-18)

Modified the existing `luka_attention_weight/{contracts,campaign,preflight,
submission}.py`, its CLI/queue helper, focused tests and plan/contract/runbook
from repository commit `6b6870ad2b40bb036f8dca83391eb1ed44cfd640`.
GATE/v2 routes both coefficients and both phases to the unchanged
`jetclass2_delphes/execution.py::execution_site('sporc_a100')` profile.
GATE/v1 retains its original routing; acceptance cannot transfer between gates.
No external donor copied, training/model/retirement kernel change, Fresh_check
dependency, or shared execution-source modification. Only operational
registration is versioned; scientific artifacts remain v1.

## Attention-weight archival comparison importer (2026-10-18)

New `luka_attention_weight/reference_archive.py` adapts the metadata/reporting
logic of `luka_attention/{campaign,worker}.py` at
`94f560259dd04253c0aa47b31dec3aedc105ac14`. It runs under that original pinned
runtime via explicit-path `runpy`, reusing unchanged `worker.report_valid`,
`jetclass2_delphes.reporting` metrics/recovery and `luka_fullsim.alljets_data`
validation-population/subset joins. It does not call the legacy full-artifact
`check_spec`, `evidence`, or `completed` validators. Those APIs are unmodified.
`source.py` and `campaign.py` in the new sweep namespace gain an explicit
comparison-only REFERENCE_ARCHIVE/v1 proof; tests and docs describe excluded
payloads. Live fixed30 anchor authentication is unchanged. No external source
copy, Fresh_check import, dataset transformation or scientific-kernel edit.

## Attention-weight sweep and scratch retirement (2026-10-18)

New `src/hlt_classification/luka_attention_weight/` is an isolated adaptation
of internal sources at `4e4a38a1a50bb2e227ace9f65657a8c20c585bb5`:

- `training.py` derives from `luka_attention/training.py`; only coefficient
  registration and new artifact namespace change. A source-body regression
  test protects the optimizer, sampler, loss normalization and selection loop.
- `worker.py` derives its checkpoint/prediction readback from
  `luka_attention/worker.py`; the new sequential lifecycle keeps one rolling
  bank on scratch and publishes explicit retirement evidence.
- `preflight.py::stress` adapts `luka_attention/preflight.py::stress` to the
  registered coefficient. Native parity reuses that module's unchanged API.
- `campaign.py`, `source.py`, `submission.py`, the thin CLI and both shell
  helpers adapt the corresponding original attention entry points.
  Original-source anchor replay reuses `luka_gating/source.py::PARENT_CHECK`.
- Unchanged dependencies include `luka_attention/{model,banks,contracts}.py`,
  `jetclass2_delphes/{runner,model,reporting,execution,batch128_30e_recipe}.py`,
  `luka_fullsim/{regular_cache,alljets_data,regular_submission,science_campaign,
  preflight_v2}.py`, `context_fusion/worker.py::save_state`, provenance/cache
  contracts, and the existing exact-DAG submission/journal APIs.

`retention.py` and the focused sweep tests are new. No external Weaver or
PyTorch source is copied or modified, and no Fresh_check runtime import is
introduced. The parent source remains fixed30 commit
`74aa58c582ff6f497a2c6d91c503f5489fac3e9a`; comparison snapshots come from the
completed attention experiment at `94f560259dd04253c0aa47b31dec3aedc105ac14`.
All old scientific and storage contracts remain unchanged. The new
`LUKA_ATTENTION_WEIGHT_* /v1` namespace declares owned-payload retirement;
missing legacy payloads are not made reusable by this policy.

## Privileged-gate frozen-backbone backend fix (2026-10-18)

In-place scoped revision of `luka_gating/{model,preflight,contracts,campaign}.py`
from commit `1be6c1c040e98402da1844797ebd469453eb1d2d`, not a new donor migration.
Uses the public installed PyTorch `torch.nn.attention.sdpa_kernel` API; no
Weaver or PyTorch implementation is copied or patched. The failure mechanism
was checked against PyTorch v2.5.1 `aten/src/ATen/native/transformers/attention.cpp`
(`should_compute_logsumexp` checks Q/K/V, not a trainable bias) and
`cuda/mem_eff_attention/kernel_backward.h` (LSE stride check).
The existing `luka_gating/training.py::epoch_mode` is reused in stress tests
instead of a separate trainability policy. Scientific recipe and original
fixed30/attention donor pins are unchanged. New ATTENTION_EXECUTION/v1 and
PREFLIGHT/v2 bind the numerical policy and required frozen-mode evidence.

## Privileged-gate automatic follow-up (2026-10-18)

New `luka_gating/followup.py` adapts the submit-host wait/authenticate/review/
submit pattern from `src/hlt_classification/context_full_followup.py`, donor
commit `eb69f6af3d9aacd8effc1125e9ecf273d0089510` (unchanged at current workspace
HEAD `94f560259dd04253c0aa47b31dec3aedc105ac14`). It does not import the Oscar
controller or reuse its site, policy, source identity, or artifacts. The new
SPORC controller adds an upfront bounded exact-graph plan and permanent
exclusive claim, using existing `luka_gating/{campaign,submission}.py` APIs.
`scouting/{hcwdl_exact_dag_submission,hcwdl_recovery}.py` and
`data/cache_contracts.py` remain unchanged dependencies for journal replay and
immutable publication. New operational FOLLOWUP_*/v1 artifacts do not change
scientific recipes. The thin queue helper/CLI, focused tests and runbook are
extended locally; no legacy files migrated or changed at runtime.

## Privileged residual attention-gate ladders (2026-10-18)

Internal donors only; no Fresh_check runtime imports or external-code copy.
New implementation is isolated under `src/hlt_classification/luka_gating/`.
The scientific parent remains fixed30 commit
`74aa58c582ff6f497a2c6d91c503f5489fac3e9a`. The immediate internal donor and
reviewed workspace commit is `94f560259dd04253c0aa47b31dec3aedc105ac14`:

- `luka_attention/{source,campaign,submission}.py` and its CLI/queue/worker
  shell: adapted original-source replay, isolated registration and exact-plan
  Slurm DAG; new namespace, contracts, graph and resource accounting.
- `luka_attention/{banks,worker,preflight}.py`: imported unchanged alignment,
  retained D-view loading, execution checks and installed attention witness.
  New gate banks store linear-size per-particle factors, not attention targets.
- `jetclass2_delphes/{model,runner,reporting,batch128_30e_recipe,execution}.py`:
  installed Weaver factory, KD, checkpoint selection, metrics, fixed30 schedule,
  and allocation identity. Native attention forward is not copied or replaced;
  the new adapter adds a bounded bias through two registered pre-hooks.
- `luka_fullsim/{regular_campaign,regular_cache,alljets_data,preflight_v2,
  foundation,preflight_reuse,science_campaign}.py`: node seeds, authenticated
  persistent caches, original50k reporting, phase/resource measurements and
  source/accounting validation. `alljets_fusion_{campaign,worker}.py` run in the
  original fixed30 worktree for read-only selected-parent authentication only.
- `jetclass2_delphes/dzfix_fusion_model.py`: deterministic parity context only.
  `context_fusion/worker.py`: atomic checkpoint save only. Neither introduces a
  second encoder or fusion training into this experiment.
- `cms_proxy_ladder/{contracts,submission}.py`, `data/cache_contracts.py`,
  `luka_fullsim/regular_submission.py`, and
  `scouting/{hcwdl_exact_dag_submission,hcwdl_recovery}.py`: safe paths, hashes,
  atomic publication, site checks, clean environment and submission journal.
- `tests/test_luka_attention.py`: tiny synthetic persistent-slot fixture and
  miniature architecture helpers; not production data or runtime dependencies.

The ten existing `luka_attention` modules are pinned to `94f56025`; the four
concurrent FullStat modules remain pinned to
`9bc3cae082915259ad569a46eefa348ddce74ea3`. Other old scientific source must be
byte-identical to the parent. No existing schemas, checkpoints or campaigns
are rewritten. New reusable artifacts are `LUKA_PRIVILEGED_GATE_*/v1`.
See HANDOFF and the privileged-gate plan/contract/runbook for verification and
the remaining genuine SPORC A100 acceptance boundary.

## Aligned particle-attention D ladder (2026-10-17)

Internal donors only; no external or Fresh_check runtime migration. Scientific
anchor/donor commit `74aa58c582ff6f497a2c6d91c503f5489fac3e9a`; reviewed workspace
parent `9bc3cae082915259ad569a46eefa348ddce74ea3`. New code is isolated under
`src/hlt_classification/luka_attention/`; existing scientific source is unchanged.

- `jetclass2_delphes/{model,runner,batch128_30e_recipe,reporting,execution}.py`:
  installed ParT factory, exact optimizer/KD/sampler/schedule/selection,
  prediction/metrics and site/allocation checks. `training.py` adapts the
  existing train kernel only to add the declared auxiliary loss/diagnostics.
  `model.py` adds a hook-based Q/K probability readout; it does not copy, replace
  or monkeypatch the installed Weaver forward. Production pins installed bytes.
- `luka_fullsim/{regular_cache,alljets_data,alljets_fusion_campaign,
  alljets_fusion_worker,preflight_reuse,foundation,science_campaign,preflight_v2}.py`:
  immutable cached inputs, original50k reporting membership, original-source
  parent validation, source pinning, accounting and phase/resource measurements.
  `jetclass2_delphes/dzfix_fusion_model.py`: deterministic parity-backend context
  only, not a fusion model or fusion training path.
- `cms_proxy_ladder/{contracts,submission}.py`, `data/cache_contracts.py`,
  `context_fusion/worker.py` (`save_state`): path/payload authentication, walltime
  formatting and atomic JSON/array/checkpoint publication. New attention banks
  have their own contracts; no old bank is reinterpreted.
- `luka_fullsim/regular_submission.py`,
  `scouting/{hcwdl_exact_dag_submission,hcwdl_recovery}.py`: SPORC test-only
  resource review, clean submission environment, exact DAG/journal and claims.
  Existing fixed30 queue/worker shell conventions inform the new thin CLI,
  helper and absolute-project Slurm worker.

The four concurrent FullStat modules (`campaign`, `submission`, `worker`,
`recovery`) are explicitly allowed only at reviewed `9bc3cae` bytes during
parent-source comparison. They are not used as data, teachers or scientific
recipes. Source drift elsewhere fails. See the new plan/contract/runbook and
HANDOFF for acceptance evidence and remaining real-site requirements.

## FullStat sparse recovery and automatic science (2026-10-16)

Internal donors only; no external or Fresh_check runtime migration. Original
FullStat producer `592a6dace8adc662903d3da0d409bcab5fefb9a7`; reviewed current
parent `bf04bb89a95d928faf008d899481ace05a69a5f3`.

- `luka_fullstat/{campaign,submission,worker,data,cache,preflight,recipe}.py`:
  preserve preparation/training semantics and normal admission; new explicit
  recovery gate/spec dispatch retains original cache/foundation parent hashes.
- `scouting/{hcwdl_exact_dag_submission,hcwdl_recovery}.py` and
  `data/cache_contracts.py`: immutable exact-DAG journal, claims, hashes and
  atomic publication, reused without changing their semantics.
- `literature_ladder_followup.py` (`lock`): existing Linux advisory controller
  lock. `context_full_followup.py`: design donor for bounded gate waiting and
  explicit follow-up authorization; no old scientific campaign is imported.
- `scripts/queue_luka_fullstat_oscar.sh`,
  `sbatch/{run_luka_fullstat_oscar,jetclass2_delphes_common,common}.sh`:
  absolute pinned-project activation and isolated Oscar runtime. New thin
  recovery launcher adds reviewed nohup execution, not a new training worker.

New contracts and acceptance evidence are documented in the FullStat plan,
contracts, recovery runbook and HANDOFF. No old ledger or receipt is rewritten.

## Matched trial ALLJETS fixed30 (2026-10-16)

Internal donors: original measured batch128 source
`922bd972ef9ec52079e9e750e6e7896e857aba8b`, recovery source
`ea7cfdfc0222768bdffd386058a807f6fac84de6`, and current opt-in FullStat runner
`592a6dace8adc662903d3da0d409bcab5fefb9a7`. No external/Fresh_check migration.

- `jetclass2_delphes/{batch128_recipe,campaign,runner}.py`: unchanged optimizer,
  KD, seed/order/selection kernels; NEW batch128_30e_recipe and exact runner
  dispatch change only schedule/budget. FullStat's recipe.py is a pattern donor
  for fixed-budget opt-in, not an imported scientific recipe or dataset.
- `luka_fullsim/{alljets_fusion_campaign,alljets_fusion_worker,
  alljets_fusion_submission}.py`: same matched17-node/30-job graph, local TRAIN
  banks, paired models, full/secondary validation and exact submission journal;
  add only v6 dispatch and fixed30 reporting.
- `luka_fullsim/{alljets_report_recovery,alljets_report_replay}.py`: original
  pinned subprocess authentication, exact failed-job sacct records, twelve
  logged probe receipts and conservative memory evidence reused read-only.
  New alljets_30epoch.py keeps old100 candidate distinct from target admission.
- `luka_fullsim/alljets_fusion_preflight.py`: split full technical validation
  from OLD-schedule admission; old public validator retains all prior checks.
  No mutation of the original candidate, producer or original recipe.
- `luka_fullsim/{alljets_batch128,preflight_reuse,foundation}.py`: protected
  path/source guards reused. New source guard explicitly registers all nine
  concurrent luka_fullstat modules, pinned to592a6dac; they are not trial data.
- `scripts/{luka_fullsim_alljets_fusion.py,queue_luka_batch128_report_recovery.sh}`
  and `sbatch/run_luka_fullsim_alljets_fusion.sh`: thin versioned CLI/helper,
  absolute project activation, no-user-site/LD setup and CPU/GPU dispatch reused.
- Existing batch128/recovery/isolated/fusion tests supply synthetic fixtures
  and complete17-fit CPU workflow; new tests exercise30 epochs and resource
  projection without claiming synthetic fixtures are remote A100 acceptance.

New authority: LUKA_ALLJETS_30E_PLAN, LUKA_ALLJETS_30E contracts/runbook.

## FullStat Oscar20 regular ladder (2026-10-16)

Internal donor: `922bd972ef9ec52079e9e750e6e7896e857aba8b`. New implementation
under `src/hlt_classification/luka_fullstat/`; no external/Fresh_check migration,
old snapshot, selected checkpoint or technical teacher is imported. This work
is independent of the concurrent batch128 report-recovery repair.

- `luka_fullsim/regular_campaign.py`: nine-node pure-encoder graph, paired
  initialization/sampler seeds and sixteen-task dependencies reused unchanged.
- `luka_fullsim/{contracts,particles,stage2,preflight_cache}.py`: scalar fields,
  raw-to-physical parsing, provisional mm/error-only zero-error semantics,
  input/view contracts and ragged block construction reused. The new namespace
  implements merged-file chunking, new source categories and row memberships;
  the trial's file split and extra pT cut are NOT reused.
- `cms_proxy_ladder/{contracts,views}.py` and
  `literature_context/transform.py`: safe references, assignment/salience view
  kernels and seventeen-feature construction reused unchanged for genuine data.
- `jetclass2_delphes/{schema,inventory,cache,model,acceptance,banks,reporting,
  campaign,execution}.py`: labels/schema, latest tree, ragged cache, actual
  Weaver model/parity, probability-bank/metric kernels, AdamW recipe defaults
  and existing Oscar site reused. `runner.py` gains only an exact-contract
  FullStat20 schedule dispatch; historical and matched128 recipes are unchanged.
- `context_fusion/{inputs,worker}.py`: the **context=None single-encoder** model
  factory and state serialization/cleanup helpers reused; NO fusion model used.
- `luka_fullsim/preflight_v2.py`: runtime configuration, phase heartbeat and
  memory sampling helpers reused. No old preflight receipt accepted.
- `scouting/{hcwdl_exact_dag_submission,hcwdl_recovery}.py`,
  `data/cache_contracts.py`, `provenance.py`, and
  `sbatch/{jetclass2_delphes_common,common}.sh`: exact journal, immutable payloads,
  source snapshots and absolute-path site activation reused.

New scientific and artifact meanings are versioned separately in
LUKA_FULLSTAT_OSCAR20_PLAN / LUKA_FULLSTAT_OSCAR20. Actual source-byte snapshots
bind the eventual pushed deployment commit, not merely this donor description.

## Batch128 sampler repair and report-only recovery (2026-10-16)

Follow-up internal donor: `ea7cfdfc0222768bdffd386058a807f6fac84de6`,
`luka_fullsim/alljets_report_replay.py` and its focused recovery tests. Correct
only the memory-evidence validator for the two original no-CUDA single-encoder
parity phases. The old `alljets_fusion_isolated_preflight.py` producer remains
unchanged. No new external migration, model/data/recipe or artifact version.

Internal donor: `922bd972ef9ec52079e9e750e6e7896e857aba8b`. No external or
Fresh_check code. `cms2jc2_response/measurement.py` retains its sampling method
and gains narrowly scoped ENOENT/ESRCH child/thread-exit handling.

- `luka_fullsim/alljets_fusion_isolated_preflight.py`: aggregate construction,
  child validation and canonical sequence reused by new `alljets_report_replay.py`.
- `luka_fullsim/alljets_fusion_preflight.py`: unchanged resource projections and
  strict evidence validator invoked from the original donor worktree at runtime.
- `luka_fullsim/alljets_batch128.py` and `preflight_reuse.py`: root/source
  compatibility patterns reused by new `alljets_report_recovery.py`.
- `scripts/queue_luka_alljets_batch128.sh`: exact reviewed dry/live workflow
  adapted for the new CPU-only report helper and explicit later science plan.
- Fusion campaign/worker/submission and thin CLI gain execution-v5 dispatch;
  all model, data, split, cache, recipe and shared training-runner bytes unchanged.

New authority: LUKA_BATCH128_REPORT_RECOVERY_PLAN and contract. The old FAILED
job and original source remain immutable; the new CPU receipt binds recovered
GPU evidence separately. No scientific target/checkpoint is imported.

## Matched ALLJETS batch128 (2026-10-15)

Internal donor: `5cab18304d991e46457aa0d9318f8731ae3b7338`.
No Fresh_check or external code. New `alljets_batch128.py` adapts original-source
authentication from `alljets_fusion_isolated_recovery.py` and true128 stress from
`context_fusion/worker.py` (same longest-row construction/three updates).
New `jetclass2_delphes/batch128_recipe.py` derives a separately named recipe
from `jetclass2_delphes/campaign.py`; the existing runner gains an exact opt-in,
including128 inference. The original recipe and default execution remain256.
Fusion campaign/worker/submission/preflight/isolated_preflight gain v4 dispatch
for17 cold fits/11 local reducers, four technical families and no old targets.
CLI and new queue helper reuse the donor's exact-hash dry/live workflow.
The synchronous model, physics, cache/split implementation and parity tolerances
remain unchanged. Active batch128 plan supersedes prior byte-identical-source
and imported-comparison clauses only for v4; old source policies are retained.

## Luka ALLJETS fusion isolated recovery (2026-10-15)

Donor commit: `f979171c40c0280ec307c4ed585889328b86732a`.
Repository-local donor files (no Fresh_check runtime import):

- `src/hlt_classification/luka_fullsim/alljets_fusion_preflight.py` -> new
  `alljets_fusion_isolated_preflight.py`: unchanged model/parity/stress/kernel/
  bank calls, reorganized into fresh child processes, D000-first stress ordering,
  authenticated child outputs and a CUDA-safe failure reporter.
- `src/hlt_classification/luka_fullsim/alljets_fusion_recovery.py` -> new
  `alljets_fusion_isolated_recovery.py`: original-source validation and exact
  submission/failed-accounting reuse, now bound to failed gate21898311.
- `scripts/queue_luka_fusion_sync_recovery.sh` -> new
  `scripts/queue_luka_fusion_isolated_preflight.sh`: same scoped dry/live workflow,
  with a new failed-gate input and v3 contract.

No donor model, data/cache/scientific kernel or training recipe is modified.
This records an execution-lifecycle change, not a proven A100 kernel fix.

## 2026-10-15: ALLJETS synchronous fusion production recovery

Internal donor `658cafa5f4f59b7be907cbfed245a2c37cb0dcd6` (diagnostic);
original failed fusion gate source `e9b0b560e60b60bf4e779cd69756e1e3163a7fb6`.
New `luka_fullsim/alljets_fusion_{sync,recovery}.py`, recovery queue helper and
tests. Existing ALLJETS fusion orchestration/CLI dispatches explicitly by v1/v2.
No external/Fresh_check code or shared model/data/kernel changes.

| Donor paths under src/hlt_classification | Reuse/change |
| --- | --- |
| `jetclass2_delphes/dzfix_fusion_model.py` | Pair-hook body adapted into isolated subclass; identical pack/layout/stats, synchronous H2D restore. Reuse strict parity harness with actual new constructor, no relaxed tolerances |
| `context_fusion/inputs.py` | Inherit FusionModel input transport; same cold seeds, state keys, single-HLT model and paired identity joins |
| `luka_fullsim/alljets_fusion_diagnostic.py` | Promote passing candidate semantics; authenticate original report and source read-only, diagnostic itself unchanged |
| `luka_fullsim/alljets_fusion_{campaign,source,worker,preflight,submission}.py` | Versioned factory/policy/artifact dispatch; preserve original graph, caches and resource admission |
| `luka_fullsim/{regular_recovery,preflight_reuse}.py` | Named-source compatibility, nonoverlap and terminal failed-job checks; use original validator subprocesses |

Contracts above become v2 only for new repaired executions. The diagnostic v1
remains non-admitting. Cluster evidence21896886 established candidate parity and
stress, not a full repaired-production pass or a debug24h runtime admission.

## 2026-10-15: isolated ALLJETS fusion CUDA failure diagnostic

Internal donor: `e9b0b560e60b60bf4e779cd69756e1e3163a7fb6`. New module
`luka_fullsim/alljets_fusion_diagnostic.py`, thin CLI/helper/worker and tests.
No copied external implementation or Fresh_check dependency. Current scientific
models/workers remain byte-identical; only the named source allowlist expands.

| Donor files (under src/hlt_classification unless stated) | Reuse / change |
| --- | --- |
| `luka_fullsim/alljets_fusion_{source,campaign,worker,submission}.py` | Exact source/parent/cache/Slurm contracts; original gate validated in its own worktree; separate non-admitting one-job diagnostic |
| `context_fusion/{inputs,worker}.py` | Unchanged paired cache, cold model factory and longest-TRAIN batch256/three-step stress |
| `jetclass2_delphes/dzfix_fusion_model.py` | Existing strict FP32/BF16 three-update parity; candidate changes only H2D saved-tensor restore to blocking in an isolated child, with identical shape/stride/dtype/storage span |
| `luka_fullsim/{regular_submission,preflight_reuse}.py`, `scouting/hcwdl_exact_dag_submission.py` | SPORC dry resource checks, exact review/journals and byte-compatibility guards |
| `tests/test_jetclass2_dzfix_fusion_{chain,offload}.py`, `tests/test_luka_fullsim_{stage2,alljets,alljets_fusion}.py` (repository-relative) | Synthetic model/cache fixtures and physical input witnesses; additionally run the installed Weaver0.5.3 CUDA parity, distinct from cluster acceptance |

New `LUKA_ALLJETS_FUSION_CUDA_DIAGNOSTIC_* /v1` contracts explicitly forbid
science admission. A convolution-tiling experiment failed local optimizer and
gradient parity and was removed; no production tiling/microbatch change was
retained. Synchronous restoration is a diagnostic candidate, not a proven
repair of the reported A100 illegal memory access.

## 2026-10-15: all-eligible Luka fusion continuation, both final routes

Internal donor commit: `9e591c7a73e1fcf24d2e2032be441dd3ee965052`.
Five additive `luka_fullsim/alljets_fusion_{source,campaign,worker,preflight,submission}.py`
modules. No old scientific source edits, external donor, or Fresh_check runtime
dependency. Authenticate original results in their original pinned worktree;
import only completed comparison rows and the scientific U000 TRAIN bank.

| Donor files (under `src/hlt_classification/` unless noted) | Reuse / explicit change |
| --- | --- |
| `luka_fullsim/{alljets_campaign,alljets_data,alljets_reporting,regular_cache,regular_worker}.py` | Unchanged full-population split, cached physical views, natural proportions, scientific schedule and old50k diagnostics; no repeated ROOT preprocessing or control fits |
| `luka_fullsim/{science_campaign,preflight_v2,preflight_reuse,foundation}.py` | Last eight fusion nodes and paired seeds, phase/headroom checks, old/new source-byte compatibility; new admission family measures full all-eligible cached pairs |
| `context_fusion/{inputs,worker}.py`, `jetclass2_delphes/dzfix_fusion_model.py` | Independent identity-joined views, context-to-primary two-encoder model, same-input alias, batch256 longest-TRAIN stress, lossless CPU-offload parity and selected-state readback |
| `jetclass2_delphes/{runner,campaign,banks,model,reporting,execution,acceptance,contracts}.py` | Unchanged scientific kernel, CE/KD objective and schedule, identity-bound TRAIN banks, metrics, environment and native/installed parity |
| `data/cache_contracts.py`, `cms_proxy_ladder/{contracts,submission}.py`, `provenance.py`, `scouting/{hcwdl_exact_dag_submission,hcwdl_recovery}.py` | Immutable canonical artifacts and exact durable submission journals; new namespace and exclusive claim, no parent mutation |
| `luka_fullsim/regular_submission.py` | Same SPORC cluster/resource test-only checks and sanitized Slurm environment |
| `tests/test_luka_fullsim_{stage2,alljets,science}.py` (repository-relative) | Synthetic ROOT, full-view cache, tiny CPU model and explicit mocked parity fixtures; not real-GPU acceptance |

The new thin CLI/helper/worker follow the existing regular and ALLJETS wrappers.
New contracts are `LUKA_ALLJETS_FUSION_*/v1`; old contracts are not reinterpreted.
The only external supervision edge is the authenticated U000 TRAIN bank; old
weights and technical-probe banks cannot initialize or supervise new science.

## 2026-10-14: all-eligible Luka regular development experiment

Internal donor commit: `adb2e939fd26b0fa22b84369e283f71a3944cee8`.
No external source or `Fresh_check` runtime dependency was added. New modules
are `luka_fullsim/alljets_{data,campaign,reporting}.py` (paths below relative to
`src/hlt_classification/`).

| Donor files | Reuse / explicit change |
| --- | --- |
| `luka_fullsim/{inventory,splits,contracts,particles}.py` | Unchanged original inventory/split replay and selected-range reader; new separately validated split explicitly retires former test files into TRAIN |
| `luka_fullsim/{regular_campaign,regular_cache,regular_preflight,regular_submission,regular_worker}.py` | Explicit artifact-family dispatch; identical default regular semantics, new streaming-preflight policy and fixed-old-validation diagnostics |
| `luka_fullsim/{stage2,preflight_cache,particle_audit,preflight_v2}.py` | Same conventions, physical inputs, per-file ragged blocks, integrated audit, runtime/headroom checks |
| `cms_proxy_ladder/views.py`, `context_fusion/{inputs,worker}.py` | Unchanged matching and views; single-encoder model constructor, selected-state readback and stress only; no fusion instantiated |
| `jetclass2_delphes/{cache,runner,campaign,reporting,banks,acceptance,execution,model}.py` | Same batch256 CE/KD schedule/seeds, metrics, identity-bound TRAIN banks, installed parity and A100 acceptance |
| `data/cache_contracts.py`, `provenance.py`, `cms_proxy_ladder/contracts.py` | Canonical hashes, deterministic predictions, immutable publication, source and file bindings |
| `scripts/{luka_fullsim_regular.py,queue_luka_fullsim_regular.sh}`, `sbatch/run_luka_fullsim_regular.sh` (repository-relative) | Thin all-jets CLI/helper/worker variants with separate namespace and authorization |

Tests reuse the synthetic ROOT and tiny CPU model fixtures in
`tests/test_luka_fullsim_{stage2,regular,science}.py` from the same donor.
New semantics are versioned under `LUKA_FULLSIM_ALLJETS_*`; legacy split,
feature and model schemas are not silently changed. No weights or old GPU
preflight acceptance evidence are imported. The prior test is not independent.

## 2026-10-13: regular preflight v2 and cache-preserving recovery

Internal donor commit: `a54c60e5354f61b4c065f9322d89dbb4f6cf8dc3`.
Adapted `luka_fullsim/regular_{campaign,preflight,submission}.py` and the thin
regular CLI; added `regular_recovery.py` and a queue wrapper. Reuse source-hash
enumeration from `luka_fullsim/preflight_reuse.py` and snapshot validation from
`provenance.py`. Source/metadata contracts remain canonical and completion-last.

The existing `context_fusion.worker.stress` still supplies the TRAIN-only
backward stress, unchanged. A separate label-free validation stress reuses
`jetclass2_delphes.runner._forward` and the original cached batch interface in
eval/inference mode; it creates no scientific checkpoint or teacher. No fusion
model is instantiated. All physical input/cache/scientific-worker/model modules
and the campaign's scientific declarations remain identical to the donor.

Operational evidence is versioned as REGULAR_PREFLIGHT/v2, with new
REGULAR_RECOVERY_GATE/v1 and REGULAR_SOURCE_REUSE/v1 bindings. Tests cover longer
validation support, no gradients/weight updates, source drift, original
accounting/journals and single-job recovery followed by the unchanged science DAG.

## 2026-10-13: restored Luka regular DIRECT/COARSE with durable inputs

Internal donor commit: `6c4cf9d12a6f354f3175103836d4e85d835a9a46`.
No external donor or runtime Fresh_check import. Five additive regular modules
implement new campaign/cache/admission semantics, rather than changing the
historical fusion/final-test contracts.

- `luka_fullsim/{foundation,inventory,splits,contracts}.py`: original 890-file
  authentication, selection, identity and exact 100k/50k file-disjoint split.
- `luka_fullsim/{particles,stage2,preflight_cache,stage2_cache}.py`:
  selected-contiguous ROOT ranges, physical conventions, salience matching,
  seven-view/input construction, assignment arrays, ragged blocks and ordered
  input fingerprints. The new file worker transcribes the selected-range reader
  after authenticating the complete split once per process. Serial donor replay
  and serial/parallel byte-parity tests guard this adaptation.
- `particle_audit.py`: the only existing scientific module with an additive
  API change, an optional internal reader factory. Its default iterator and
  diagnostic calculations are unchanged. The adapter audits the same decoded
  rows during preparation; file-local moments merge in fixed order, permitting
  diagnostic floating-point roundoff but never differences in model-input bytes.
- `luka_fullsim/{science_campaign,science_worker,science_submission}.py`:
  single-coordinate node fields/paired seeds, receipt-bound training/checkpoint/
  TRAIN-bank logic, partial results/recoveries and exact dry/live submission.
  Worker/submission code is adapted into the separate `REGULAR_*` namespace:
  nine fits/five reducers, durable single-view loader, new gate evidence and no
  historical-worktree dependency. Old workers and admission remain unchanged.
- `luka_fullsim/preflight_v2.py`, `context_fusion/{inputs,worker}.py`,
  `jetclass2_delphes/{cache,runner,banks,acceptance,model,reporting,execution,
  campaign}.py`: phase logging, single-model factory,
  full-support stress, save/readback, original batch masks, training/KD recipe,
  installed single-encoder parity, environment, metrics and fresh-reference recovery.
  No fusion model is instantiated; each node's context is explicitly None.
- `data/cache_contracts.py`, `cms_proxy_ladder/contracts.py`,
  `scouting/{hcwdl_exact_dag_submission,hcwdl_recovery}.py`, `provenance.py`:
  deterministic atomic payload publication, source/content/parent hashes,
  exclusive claims and exact per-job submission journals.
- `tests/test_luka_fullsim_{stage2,science}.py`: synthetic ROOT/prepared fixtures
  and a tiny CPU-only model to test the full regular scientific graph. This is
  not installed-Weaver or SPORC acceptance evidence.

New behavior is authorized by `plans/LUKA_FULLSIM_REGULAR_PLAN.md` and versioned
in `contracts/LUKA_FULLSIM_REGULAR.md`. The prior holdout is recorded consumed;
the new campaign has no test-evaluation capability.

## 2026-10-12: Luka locked final-test comparison

Internal donor commit: `01a738328b267da1ecc13bbf4263c2e4a2d9e4e2`.
Four additive `luka_fullsim/final_test_{campaign,data,worker,submission}.py`
modules; every old tracked `src/` byte remains unchanged and is checked against
that exact parent at admission. No external source or Fresh_check import.

- `luka_fullsim/{science_campaign,science_worker,science_submission}.py`:
  registered twelve-node graph, original selected-checkpoint/report/complete
  validators, measured allocation/environment checks and exact scheduler checks.
  Original source authentication executes in the original pinned worktree;
  its restricted additive-source contract is not weakened.
- `luka_fullsim/{stage2,particles,contracts,preflight_cache,preflight_reuse,
  preflight_v2}.py`: physical conventions, original matching/input kernels,
  selected contiguous ROOT ranges, row identities, ragged blocks, source-byte
  comparison and phase/peak-memory logging. The new reader is explicitly
  final-test-capable behind locks/claim/TRAIN-smoke; old readers remain sealed.
- `jetclass2_delphes/{cache,runner,reporting,acceptance,dzfix_fusion_model}.py`
  and `context_fusion/inputs.py`: unchanged single/paired tensor batching,
  checkpoint model factory, T=1 prediction, validation/test metric definitions,
  recovery convention and installed/native-mask parity. No training kernel,
  teacher-bank creation, optimizer or checkpoint selection is invoked.
- `cms_proxy_ladder/{contracts,submission}.py`,
  `scouting/{hcwdl_exact_dag_submission,hcwdl_recovery}.py`, `provenance.py`,
  `data/cache_contracts.py`, `sbatch/common.sh`: immutable file references,
  source hashes, publication and exact dry/live DAG submission semantics.
- `tests/test_luka_fullsim_stage2.py`: synthetic ROOT, preparation and source
  fixtures. New tests exercise all twelve tiny frozen CPU checkpoints and
  compare tie-correct weighted AUC to installed sklearn; not real test data.

Intentional new semantics: separate v1 finalist/execution/access/result
contracts, exactly the existing 149553-jet holdout, all-model frozen comparison,
and paired source-file bootstrap contrasts. No dataset re-splitting or ordinary
unused rows. Selected historical science weights are explicitly authorized
inference parents, never initialization for new fits. No third-party license
change. Local evidence and remaining genuine GPU execution are in HANDOFF.

## 2026-10-09: Luka stage-3 fusion science campaign

Internal donor commit: `bcccd498ee66fdee6fd3731b823d09ac2bb5df8b`.
Five new `luka_fullsim/science_*.py` modules; no existing `src/` file changes,
no external code, Fresh_check import, imported fit or probe-weight reuse.

- `context_fusion/{campaign,worker,inputs}.py`: graph/receipt design, asymmetric
  native fusion, paired cache identity joins, new-model factory and checkpoint
  writer. Controls and U000 are now fresh Luka fits, not imported CONTEXT fits.
- `luka_fullsim/{stage2,stage2_cache,preflight_cache,preflight_v2,preflight_reuse,
  particles,contracts}.py`: unchanged physical/view/input semantics, exact saved
  fingerprints, one-pass cache design, phase logging and explicit source reuse.
  New cache wrapper supports all seven already prepared coarse coordinates.
- `jetclass2_delphes/{runner,campaign,banks,reporting,model,execution,
  dzfix_fusion_model}.py`: current 60--100 schedule, fit/bank/metric kernels,
  environment, native offload and parity contracts; no old-overview schedule.
- `cms_proxy_ladder/{campaign,contracts,submission}.py`: current paired seeds,
  safe file references and walltime format; not dataset/schema impersonation.
- `scouting/{hcwdl_exact_dag_submission,hcwdl_recovery}.py`: immutable exact-DAG
  dry/live ledgers and per-submission journal. New exclusive Luka live claim.
- `provenance.py`, `data/cache_contracts.py`, `sbatch/common.sh`: unchanged
  clean-source checks, canonical hashes, atomic publication and shell helpers.
- `tests/test_luka_fullsim_{stage2,preflight_v2}.py`: synthetic ROOT/source
  fixtures. `tests/test_context_fusion.py`: miniature/failure-injection design.

New Luka science v1 contracts distinguish admission/source/spec/plan/task/results
from historical artifacts. Accepted genuine SPORC preflight 21835630 is reused
only with exact report/source/output authentication and unchanged kernel bytes.
The new orchestration's local tests do not assert a new remote campaign run.

## 2026-10-09: Luka preflight-v2 operational recovery

Internal donor commit: `1256c75633399ec177d07308eb17d88a78e1fb85`.
New modules: `luka_fullsim/{preflight_reuse,preflight_cache,preflight_v2}.py`.
All original `src/` files stay byte-identical; no external code or Fresh_check.

- `luka_fullsim/preflight.py`: same four technical cases, full-population pass,
  checkpoints/TRAIN banks and resource projections; execution order changes.
- `luka_fullsim/{stage2_cache,particles,stage2}.py`: physical conversion,
  prepared assignment joins, feature kernel and input fingerprints.
- `context_fusion/{inputs,worker}.py`: paired caches, native model, unchanged
  longest-TRAIN batch256/three-step stress, cleanup and checkpoint writer.
- `jetclass2_delphes/{cache,acceptance,dzfix_fusion_model,runner,banks,execution,
  campaign,model}.py`: RAM batches, parity, training, probability joins, seeds,
  recipe, and resource authentication, all unchanged.
- `provenance.py`, `data/cache_contracts.py`, `cms2jc2_response/measurement.py`:
  ordinary source checks, immutable outputs and Linux measurements. Dual-source
  compatibility is additive, not a source-verifier bypass.

No imported scientific weights/data/teachers. Execution/reuse contracts and the
new unmeasured allocation are documented in the stage-2 plan/contract. Local
tests do not establish installed-Weaver or genuine SPORC GPU acceptance.

## 2026-10-08: Luka FullSim metadata foundation (stage 1)

Internal donor commit: `97ee3434a93de2fc7b0f09d506926cd50c5cde46`.
New package `luka_fullsim`, CLI, CPU worker, tests and independent plan/contract.
No external donor code or `Fresh_check` imports; old scientific files unchanged.

| Donor paths (under src/hlt_classification) | Reuse/adaptation |
| --- | --- |
| `data/cache_contracts.py` | Imported canonical hashes, full-file SHA256, immutable atomic JSON publication and loading |
| `provenance.py` | Imported clean tracked-source snapshots and before-publication reauthentication |
| `jetclass2_delphes/{schema,contracts}.py` | Imported eleven-class numeric mapping, branch-header validation and safe relative-path utility only; no Delphes assumptions, particle reader, selection, inputs or artifact contracts are reused |
| `jetclass2_delphes/inventory.py` | Imported latest-tree-cycle helper; adapted bounded scalar scan and before/after content authentication into the new FullSim schema family |
| `jetclass2_delphes/{splits,split_registry}.py` | Design donor for file coverage/deficit grouping and SHA256 within-class subset ranking; new exact-rational 40/20/40 reservoirs and global Hamilton quotas, not the donor's 60/20/20 or reserve-one quota semantics |

New semantics explicitly bind strict offline pT >200, source-restricted QCD,
natural global class proportions, exact 100k/50k and unused ordinary rows
separate from the file-disjoint sealed test reserve. New namespace
`LUKA_FULLSIM_{INVENTORY,SPLITS,FOUNDATION}/v1`. No particle data, teacher bank,
model weights, existing acceptance, or historical split is imported.

## 2026-10-08: external S3 one-time science continuation

Internal operational donor commit: `8d936b9b72ab2a10d2a9d7bb8b4e440e6fe59be5`.
Original S3 scientific implementation remains in its own checkout (introduced
at `06803a8ba106cd8695186140ddf08ca9ec6a6bae`). No existing scientific or
operational Python file is edited; no Fresh_check import or external donor.

| Donor paths | New reuse |
| --- | --- |
| `src/hlt_classification/literature_ladder_followup.py` | Reused canonical metadata reader/hash, scrubbed subprocess environment, SPORC advisory lock and active-state vocabulary; adapted pinned-interpreter/one-time continuation pattern |
| `src/hlt_classification/s3_ladder/campaign.py`; `scripts/jetclass2_s3_ladder.py` | Invoked exclusively through the original worktree: validate unchanged gate, genuine preflight, measured seven-job plan and exact dry/live submitter |
| `src/hlt_classification/scouting/{hcwdl_exact_dag_submission,hcwdl_recovery}.py` | Original-source bridge authenticates resolved dependency commands, all live IDs and submission journal; no automatic partial retry |
| `scripts/start_jetclass2_literature_ladder_followup.sh`; `tests/test_literature_ladder_followup.py` | Separate-checkout/nohup launcher, accounting failure and duplicate-controller tests adapted for the single S3 preflight |

New `s3_ladder_followup.py`, thin CLI/launcher, tests, operational plan/contract
and runbook. `JC2_S3_FOLLOWUP/v1` records separate authorization/review/submission
evidence. Historical S3 SPEC/PLAN v1 `automatic_followup=false` fields stay
unchanged; the user-requested operational continuation is authorized externally.

## 2026-10-08: CONTEXT_V1 100k/50k Oscar fusion adaptation

Internal implementation donor commit: `06803a8ba106cd8695186140ddf08ca9ec6a6bae`.
The separately authenticated historical comparison/teacher campaign is pinned
to `f889f359cdddca1beb3d6e817332d2b42a42ddf3`. No external donor or Fresh_check
runtime import; no historical scientific implementation is modified.

| Donor paths (under src/hlt_classification unless noted) | New reuse |
| --- | --- |
| `cms_proxy_ladder/{context,production,campaign,cache,data,inputs,context_inputs,views,contracts,gate,submission}.py` | Authenticate the completed CONTEXT parent, reuse its selected identities/matching/frontend/U000 bank and results; original paired seeds, bounded RAM views, Oscar site checks and durable submission claims |
| `literature_context_consumer/` | Existing frozen CONTEXT_V1 relocation/readers inherited through the parent foundation; no regeneration |
| `jetclass2_delphes/{dzfix_fusion_model,salience_learned_model,model}.py` | Native adjacent-view asymmetric fusion, compact masks, layout-preserving pair-saved-tensor CPU offload, independent encoder initialization and ordinary single-HLT endpoints |
| `jetclass2_delphes/{runner,campaign,cache,banks,acceptance,execution,reporting}.py` | Current single-view training schedule and validation, identity-bound T=2 banks, selected weights, installed parity, resource identity and recovery metrics; not the older fusion patience/validation partition |
| `cms2jc2_response/measurement.py`; `data/cache_contracts.py` | Process-tree sampled RSS, canonical hashes and atomic non-overwriting publication |
| `scouting/{hcwdl_exact_dag_submission,hcwdl_recovery}.py` | Reviewed exact-plan dry/live ledgers and dependency journals |
| `scripts/queue_jetclass2_gap_sweep.sh`; `sbatch/{common,jetclass2_delphes_common}.sh` | Thin pinned-source helper pattern and absolute Oscar environment activation |
| `tests/test_context_ladder.py`, `tests/test_jetclass2_dzfix_fusion_chain.py`, their producer/fake-Weaver fixtures | Real synthetic CONTEXT loader/paired tensor parity and local miniature fixtures; not remote hardware acceptance |

New implementation: `context_fusion/{__init__,contracts,inputs,campaign,worker}.py`,
thin CLI/queue helper, focused tests and CONTEXT_FUSION plan/contract/runbook.
New `JC2_CONTEXT_FUSION_*/v1` namespace; unchanged original training/bank
contracts. Eight cold-start fits and six reducers import supervision, never
teacher weights as student initialization. Final-test inference stays sealed.

## 2026-10-07: frozen S3 direct/coarse comparison

Internal donor and required completed screen commit:
`8abdc943c02d9844f98c91028f68e82e02b62fd4`. Its inherited CORR_HIGH_TOPO
baseline source remains `8c5b47029ed15f88aa852b6f38d06e98bbbfcd9b`.
No external donor, Fresh_check runtime import or old-file modification.

| Donor paths (under src/hlt_classification unless noted) | New reuse |
| --- | --- |
| `gap_sweep/{campaign,cache,kernel,contracts,worker}.py` | Frozen selection, completed controls, exact endpoint generator/cache, source-transfer and immutable worker pattern; no change to old source |
| `correlated_topology/views.py`; `cms_proxy_ladder/{data,release,cache,cache_full,contracts}.py` | Original representative mapping, surviving-tracking interpolation, 17-input frontend, bounded spawn RAM views and ordinary identity joins |
| `cms_proxy_ladder/{production,gate,correlated,submission}.py` | Original OFFLINE reducer authentication, execution environment checks, source pinning, exact dry/live submission claims and walltime formatting |
| `jetclass2_delphes/{runner,model,banks,cache,campaign,acceptance,execution,reporting}.py` | Unchanged fresh CE25/KL75 training, selected checkpoints, T=2 probability banks, installed parity, paired seeds (inherited via old graph) and recovery metrics |
| `data/cache_contracts.py`; `scouting/hcwdl_exact_dag_submission.py` | Atomic content/file authentication and exact resolved dependency journals |
| `scripts/jetclass2_gap_sweep.py`, `scripts/queue_jetclass2_gap_sweep.sh`; `sbatch/jetclass2_delphes_common.sh`, `sbatch/common.sh` | Thin CLI/helper pattern and unchanged absolute-path SPORC environment activation |
| `tests/test_gap_sweep.py`, `tests/test_correlated_topology.py`, their inherited producer fixtures, `tests/test_jetclass2_delphes_training.py` | Synthetic saved-source/readers, real spawn and tiny-model training tests; explicit mock hardware does not claim real acceptance |

New files: `s3_ladder/{__init__,contracts,views,cache,campaign,worker}.py`,
`scripts/jetclass2_s3_ladder.py`, queue helper, `tests/test_s3_ladder.py`, and
`JETCLASS2_S3_LADDER` plan/contract/runbook. New scientific namespace
`JC2_S3_LADDER_*/v1`; original training/bank/submission contracts reused as-is.
Original OFFLINE weights are not student initializations; only authenticated
training probabilities are imported as supervision. No final-test artifacts.

## 2026-10-07: bounded CORR_HIGH_TOPO baseline-gap screen

Internal donor and required baseline source:
`8c5b47029ed15f88aa852b6f38d06e98bbbfcd9b`. No external donor,
Fresh_check runtime import, imported model initialization, or credentials.
Historical scientific files remain byte-identical; new implementation is
isolated in `src/hlt_classification/gap_sweep/`.

| Donor path (under src/hlt_classification unless noted) | Reuse |
| --- | --- |
| `cms_proxy_ladder/correlated_topology.py`, `production.py`, `gate.py`, `contracts.py`, `correlated.py`, `submission.py` | Authenticate the exact parent/source and two completed CE controls; same execution gate, immutable references, dry/live site checks, exclusive journaled submission and walltime formatting |
| `cms_proxy_ladder/data.py`, `release.py`, `cache.py`, `cache_full.py`; `correlated_topology/views.py` | Authenticated ordinary-only identity pairing, bounded spawn preparation, same existing 17-field transform; no new matching or dataset export |
| `cms2jc2_response/bridge.py`, `data/cache_contracts.py` | Physical particle invariants, canonical hash streams and atomic publication |
| `jetclass2_delphes/cache.py`, `campaign.py`, `runner.py`, `model.py`, `acceptance.py`, `execution.py` | Same RAM batches, full CE recipe, seeds, selected weights, installed-Weaver parity, GPU/environment checks |
| `scouting/hcwdl_exact_dag_submission.py`; `sbatch/jetclass2_delphes_common.sh`, `sbatch/common.sh` | Exact dependency IDs and durable journals; existing isolated SPORC environment |
| `tests/test_correlated_topology.py`, `test_correlated_tracking_ladder.py`, `test_correlated_tracking_production.py`, `test_literature_proxy.py`, `test_jetclass2_delphes_training.py` (under tests) | Synthetic producer/reader fixtures, physical invariants and tiny-model CPU training tests; hardware substitutes do not claim genuine acceptance |

New files: `gap_sweep/{contracts,kernel,cache,campaign,worker}.py` and package
initializer; `scripts/jetclass2_gap_sweep.py`, queue helper, focused tests,
plan/contract/runbook `JETCLASS2_GAP_SWEEP*`. Scientific artifacts use a new
`JC2_GAP_SWEEP_*/v1` namespace; inherited submission journals retain their
existing exact-DAG contract rather than impersonating new artifacts.

## 2026-10-07: CORR_HIGH_TOPO ordinary dataset and direct/coarse ladder

Internal donor snapshot: `eb69f6af3d9aacd8effc1125e9ecf273d0089510`.
No external donor, legacy runtime import, copied credentials, or model import.

| Donor path (under src/hlt_classification unless noted) | Reuse |
| --- | --- |
| `correlated_tracking/kernel.py` | Unchanged HIGH endpoint and tightly bounded historical ARM/x86 MID tracking replay (structure exact) |
| `literature_proxy/kernel.py`, `literature_proxy_v2/kernel.py`, `literature_proxy_v3/worker.py` | Identity-keyed streams, same eligibility/disjoint-pair pattern, authenticated frozen drop/merge calibration; deliberately omit PID and p4 noise |
| `literature_proxy/diagnostics.py`, `literature_proxy_production/codec.py` | Fixed descriptive histograms and lossless physical bank publication/readback |
| `cms_proxy_ladder/correlated.py`, `correlated_views.py`, `release.py`, `data.py`, `cache.py`, `gate.py`, `production.py`, `submission.py`, `cache_full.py` | Versioned successor release/representative views; unchanged old defaults; bounded process work, measured admission, six-fit direct/coarse plan, exact journaled submission |
| `jetclass2_delphes/runner.py`, `model.py`, `acceptance.py`; `sbatch/jetclass2_delphes_common.sh` | Existing 17-field ParT training, CE/KD acceptance, installed-Weaver checks and SPORC environment |
| `tests/test_correlated_tracking_ladder.py`, `test_correlated_tracking_production.py`, `test_literature_proxy_v3_campaign.py`, `test_literature_proxy.py` (under tests) | Synthetic ROOT/real codec/CPU training and saved pilot fixtures |

New code: `correlated_topology/{kernel,views,dataset}.py`, ladder adapter,
`scripts/jetclass2_correlated_topology.py`, queue helper and focused tests.
New plan/contract/runbook: `JETCLASS2_CORR_HIGH_TOPO*`.
The historical tier3 dispatch test now checks its actual pinned executor
`779382740cdd44a5120449f02fde4a7719fed298`, not arbitrary future main;
an additional rejection test proves newer scientific code cannot inherit that
old source-transfer approval. Runtime source-transfer rules are unchanged.

## 2026-10-07: Oscar CONTEXT_V1 1M conditional science followup

Internal donor commit: `9cebdbe133da20f00cc14ec4155bd627accb7e85` (also the
immutable scientific gate source). No external donor or Fresh_check import.

| Donor path | New use |
| --- | --- |
| `src/hlt_classification/literature_ladder_followup.py` | Reuse hash/read, sanitized subprocess environment, exclusive Linux lock and authenticated pinned gate inspector; adapt polling, source binding and immutable publication for the new controller |
| `src/hlt_classification/correlated_ladder_followup.py` | Separate executor-source binding and explicit measured-resource policy pattern |
| `scripts/start_jetclass2_correlated_tier3_followup.sh`, `scripts/jetclass2_literature_ladder_followup.py` | Oscar dry/live detached launcher and thin CLI |
| `tests/test_literature_ladder_followup.py` | Fake scheduler, source-bound orchestration, state machine and real journal-authentication test patterns |
| `src/hlt_classification/cms_proxy_ladder/context_full.py`, `submission.py` | Unmodified original 1M scientific admission, dry/live CLI and exact journaled submission called as subprocesses |
| `src/hlt_classification/scouting/hcwdl_authorization.py` | Unchanged exact clean pushed source validator |

New paths: `src/hlt_classification/context_full_followup.py`, corresponding
CLI/launcher/test and `JETCLASS2_CONTEXT_1M_FOLLOWUP` plan/contract/runbook.
Old source-hashed gate/plan files are not edited. Concurrent V2 changes remain
unrelated to this execution amendment.

## 2026-10-07: CONTEXT_V2 frozen re-encoding and automatic Oscar workflow

Internal donor snapshot: `779382740cdd44a5120449f02fde4a7719fed298`.
The executed V1 classifier parent remains the separate immutable
`f889f359cdddca1beb3d6e817332d2b42a42ddf3` campaign. No external donor or
Fresh_check runtime imports; no migration of credentials, model weights or data.

| Donor paths | New use |
| --- | --- |
| `literature_context/transform.py` | `context_v2/kernel.py` retains the triangular forward/inverse structure, visible-p4 context, phases and scale mapping; only shear amplitude becomes 1.0. Frozen V1 code is imported unchanged for inversion. |
| `literature_context_consumer.py`, `literature_context_production/{contracts,codec,output,population}.py` | Reuse authenticated relocated source, canonical identities and lossless bank schema. New producer validates V1 test-build authority separately and gives V2 its own receipts/manifests and ordinary-only consumer. |
| `cms_proxy_ladder/{context,release,data,cache,views,gate,production,submission,campaign}.py` | New explicit v5/v10/v7 dispatch, byte-identical old selection index, fresh matching and unchanged CE/KD kernel. Private adapter parameters preserve old V1 defaults. |
| `scouting/hcwdl_exact_dag_submission.py`, `cms_proxy_ladder/context.py` | Reuse exact accepted-ID journal, canonical dry ledger and exclusive live-submission claim. New initial DAG includes a preauthorized bounded 16-job science launcher. |
| `sbatch/jetclass2_delphes_common.sh`, `jetclass2_delphes/{execution,model,runner}.py` | Existing absolute-path Oscar environment activation, L40S runtime acceptance and training/reduction code reused. |
| `tests/test_context_ladder.py`, `tests/test_literature_context_consumer.py`, `tests/test_literature_context.py`, `tests/test_literature_proxy.py` | Synthetic ROOT/relocation fixtures and particle fixtures reused for new production, pairing, inverse, process, cache and submission tests. |

All source paths above are under `src/hlt_classification/` except explicitly
named scripts/tests. New plan and contract:
`JETCLASS2_CONTEXT_V2_OSCAR_WORKFLOW_PLAN.md` and
`JETCLASS2_CONTEXT_V2_OSCAR_WORKFLOW.md`. Existing unrelated full-population
CONTEXT_V1 work is a separate experiment and is preserved.

## 2026-10-07: unchanged CONTEXT_V1, full 1M/250k Oscar campaign

In-repository donor commit: `779382740cdd44a5120449f02fde4a7719fed298`.
No external donor or `Fresh_check` runtime import. This does not migrate any
CONTEXT_V2 changes present concurrently in the shared workspace.

| Donor path | New use |
| --- | --- |
| `src/hlt_classification/cms_proxy_ladder/context.py` | `context_full.py`: additive population, resource and version variant; reader/completion/claim semantics retained |
| `scripts/jetclass2_context_ladder.py` | Full CLI, unchanged result/queue workflow |
| `scripts/queue_jetclass2_context_ladder.sh` | Full helper with separate authorization/root |
| `tests/test_context_ladder.py` | Synthetic fixture/profile helpers and regression baseline |
| `tests/test_literature_context_consumer.py` | Actual relocated synthetic ROOT/proxy fixture |
| `src/hlt_classification/cms_proxy_ladder/release.py` | v6 all-ordinary-row release dispatch |
| `src/hlt_classification/cms_proxy_ladder/data.py` | v6 joins, bounded assignment tasks, authenticated slot total |
| `src/hlt_classification/cms_proxy_ladder/cache.py` | v6 file-sized batches; conservative bounds factored into `cache_full.py` |
| `src/hlt_classification/cms_proxy_ladder/gate.py` | v11 source/admission/preflight dispatch |
| `src/hlt_classification/cms_proxy_ladder/production.py` | v8 campaign dispatch |
| `src/hlt_classification/cms_proxy_ladder/submission.py` | Separate prefixes/authorization; same durable exact DAG submission |

Unmodified scientific dependencies include `campaign.py`, `views.py`,
`context_inputs.py`, the JetClass2 model/kernel/recipe and relocated reader.
Source locking binds these dependencies and the new plan/contract/CLI/tests.

## 2026-10-07: CORR_MID deferred tier3 science

Internal donor commit `062713ead3e00721d9f3cb7073e88aef83ebd325`.
No external code or Fresh_check imports. Old gate worktrees/receipts stay immutable.

| Donor paths | Reuse/adaptation |
| --- | --- |
| `src/hlt_classification/literature_ladder_followup.py` | Reuse read/hash, sanitized subprocess, lock and original-pin inspection bridge; new correlated controller binds v9, exact CORR_MID manifest, 11-job tier3 policy and a separate executor |
| `src/hlt_classification/cms_proxy_ladder/{correlated,production,submission,gate}.py` | Retain measured-science validation, kernel, command builder and durable claims. New `correlated_tier3.py` explicitly authenticates source transfer and v6 campaign; sole old-source edit is the three-line validation dispatch |
| `src/hlt_classification/jetclass2_delphes/execution.py` | Import existing same-A100 debug-to-tier3 policy/site definitions unchanged; actual worker allocation/GPU/software checks retained |
| `scripts/jetclass2_literature_ladder_followup.py`, `scripts/start_jetclass2_literature_ladder_followup.sh` | New thin correlated tier3 CLI and dry-first detached launcher |
| `tests/test_literature_ladder_followup.py`, `tests/test_correlated_tracking_ladder.py` | Fake accounting/claim/inspection failure tests, synthetic real gate/foundation integration reused for tier3 |

The [new contract](contracts/JETCLASS2_CORRELATED_TIER3_FOLLOWUP.md) records
versioned execution-only artifacts; science and dataset semantics are unchanged.

## 2026-10-07: frozen CORR_MID production and SPORC direct/coarse

Internal donor commit `0667355d8c4d72e48e32c03f2d1540595fe654b7`.
No external code copied or `Fresh_check` runtime import. Historical pilot kernel,
recipe, frontend and artifacts are unchanged; new versioned adapters select MID.

| Donor paths | Reuse/adaptation |
| --- | --- |
| `literature_context_production/{campaign,contracts,population,storage,submission,engine,output,worker,codec}.py` | New `correlated_tracking_production/`: 2.25M membership, bounded process output, reservations, exact array identity, full DAG and missing-only recovery; new namespace and saved CORR_MID replay |
| `correlated_tracking/{campaign,contracts,kernel,worker}.py` | Historical pilot evidence and unchanged physical generator; new evidence module authenticates known job 228972, source and all output hashes |
| `literature_proxy_v2/inputs.py`, `literature_proxy/population.py`, `cms2jc2_production/population.py` | Original completed-parent evidence, exact raw particle keys and existing registry/split membership; no CMS-fit generation |
| `literature_context/transform.py`; `literature_proxy_production/codec.py` | Import unchanged common asinh/log1p frontend and lossless physical codec, never apply context degradation |
| `cms_proxy_ladder/{literature,context,views}.py` | New `correlated.py` original-storage SPORC adapter and `correlated_views.py` verified identity matching/noise-amplitude bridge; six fits and three reducers |
| `cms_proxy_ladder/{release,data,cache,gate,production,submission}.py` | Explicit new-schema dispatch while preserving old campaign semantics; shared CE/KD/cache/report kernels retained |
| `scripts/jetclass2_literature_context_dataset.py`, `scripts/queue_jetclass2_literature_context_dataset.sh`, `sbatch/run_jetclass2_literature_context_dataset.sh` | Thin production CLI/environment/queue wrappers under `correlated_tracking_dataset` names |
| `scripts/jetclass2_context_ladder.py`, `scripts/queue_jetclass2_literature_proxy_ladder.sh` | Thin CORR_MID SPORC CLI and separately reviewed gate/science helper |
| `tests/test_literature_context_production.py`, `tests/test_context_ladder.py`, `tests/test_literature_proxy_v2_campaign.py`, `tests/test_jetclass2_delphes.py`, `tests/test_jetclass2_delphes_training.py` | Synthetic ROOT/parent/production/cache, exact-plan, teacher-bank and CPU CE/KD integration tests |

Package paths are under `src/hlt_classification/`. New production contracts use
`JC2_CORRELATED_TRACKING_PRODUCTION_*/v1`; shared ladder versions are recorded in
the [new contract](contracts/JETCLASS2_CORRELATED_TRACKING_LADDER.md). The active
plan supersedes the proposed CE-only strength screen through explicit user MID
selection, not through a claim that the desired classifier gap has been measured.

## 2026-10-07: shared-versus-independent tracking pilot

Internal donor commit `f889f359cdddca1beb3d6e817332d2b42a42ddf3`.
New `correlated_tracking/` package; no old generator, dataset or classifier
edited. No external code copied and no `Fresh_check` runtime import.

| Donor paths | Reuse/adaptation |
| --- | --- |
| `literature_proxy/campaign.py`, `contracts.py` | Clean/pushed source, dry/live authorization, durable ambiguous-submit lock, immutable artifacts; new namespace and complete Python source pin |
| `literature_proxy_v2/inputs.py` | Import unchanged completed-v1 pilot authentication and original OFFLINE reader; old NOMINAL endpoint is ignored, no v2 rates used |
| `literature_context/campaign.py`, `worker.py` | Per-file spawn pilot, physical banks, serial/process replay, receipt-last publication patterns |
| `literature_proxy/diagnostics.py`, `worker.py`; `data/cache_contracts.py` | Descriptive histograms/moments/CSV/PDF, physical arrays/RSS, deterministic NPZ serialization |
| `cms2jc2_response/bridge.py`; `jetclass2_delphes/contracts.py` | Physical schema/applicability validation and safe relative output paths |
| `scripts/jetclass2_literature_context.py`, `scripts/queue_jetclass2_literature_context.sh`, `sbatch/run_jetclass2_literature_context.sh` | Thin CLI and dry-first Tigris environment wrappers |
| `tests/test_literature_proxy.py`, `tests/test_literature_proxy_v2_campaign.py`, `tests/test_literature_context_campaign.py` | Particle/synthetic completed-parent fixtures and submission/replay failure checks |

Package paths in the table are under `src/hlt_classification/`. Reference-noise
kernel, conditional residual and analytic pair-covariance diagnostics are new
synthetic equations specified in the versioned plan, not a migrated CMS fit.

## 2026-10-06: CONTEXT_V1 relocated Oscar 100k/50k ladder

Internal donor commit `4f473c3e97ee3270cec520914477dfe9a38728ab`.
No external source or `Fresh_check` runtime imports. SHA-256 values below are
local donor bytes at inspection (before line-ending conversion in another checkout).

| Donor file | SHA-256 | New surface / retained meaning |
| --- | --- | --- |
| `src/hlt_classification/literature_proxy_consumer.py` | `1bd2152aff3febb94c04bf5d0473a0f8da11af8551bb0885391d7c68a1ac692d` | `literature_context_consumer.py`: external-root relocation, identity joins and sealed ordinary reader; new context manifest/recipe/input contract |
| `src/hlt_classification/cms_proxy_ladder/literature.py` | `316734ea5dd44d3fb738c00e81c3cfb88aeaa41c6456177bb7124ebc1a9d7fd0` | `cms_proxy_ladder/context.py`: source-pinned gate, exact dry/live DAG, fresh fits; Oscar 100k/50k and CE/KD preflight |
| `scripts/jetclass2_literature_proxy_ladder.py` | `464d417be8c482fe717c25b972a9745e2b12225c5e2f1b6fcb8e9d3e9230c579` | `scripts/jetclass2_context_ladder.py`: thin CLI and saved validation results |
| `scripts/queue_jetclass2_literature_proxy_ladder.sh` | `1fac627bcdb773cc8cd745c20a7745e59ae3dd9dafe085fa3ad1db1688014c92` | `scripts/queue_jetclass2_context_ladder.sh`: two explicit reviewed phases; Oscar copied roots and clean pushed source |
| `tests/test_literature_proxy_consumer.py` | `f106089c15dd1e34edff8e9ae1f41393b0bdd790ad6391d3ee37be48a30408cc` | Context relocation, corrupt bytes, swapped IDs, no native-HLT/test reads |
| `tests/test_literature_proxy_ladder.py` | `d6a717cd3c2afbb0ed0f65b4bf75c0c77a409b530ba62b9b04dd786d14dfaaeb` | Context fresh foundation/cache/DAG and durable submission tests |
| `src/hlt_classification/literature_context/transform.py` | `5917d13717776200681107adf8415d52828ff20e0d8320c95b8cb78aea26cc59` | Imported **unchanged** through `context_inputs.py`; no generator/inverse retuning |

Shared ladder release/data/cache/gate/production/submission/views dispatch through
explicit new versions, preserving existing versions. Matching, model, loss,
schedule and teacher publication code are reused without new scientific equations.
No third-party license change. Tests and remote-admission status are recorded in
`docs/HANDOFF.md`; local synthetic acceptance is not a real Oscar GPU run.

## 2026-10-06: frozen CONTEXT_V1 full dataset production

Internal donor checkout **`6cb5f32ab6e8d179f846f0cfeb4511ef1a0b6ab4`**.
Separate `literature_context_production/` package and wrappers; no old response
kernel, dataset, input transform or classifier modified. No external donor code
or `Fresh_check` runtime imports. New `JC2_CONTEXT_PRODUCTION_*/v1` contracts
retain the old pilot and NOISE_V3 namespaces without relabelling artifacts.

| Donor paths | Reuse/adaptation |
| --- | --- |
| `literature_proxy_production/campaign.py`, `contracts.py`, `storage.py` | Frozen evidence, source, quota, immutable attempts and reservations; new namespace and exact approved context pilot binding |
| `literature_proxy_production/population.py` | Same metadata population, file-local shards, offline ROOT reader and original identity/RNG keys |
| `literature_proxy_production/engine.py`, `worker.py` | Bounded ordered processes, actual production-reader/writer preflight; explicitly compose unchanged low-noise and context kernels and check inverse integrity |
| `literature_proxy_production/output.py`, `submission.py` | Role manifests, exact array IDs, durable submit journals, full DAG and same-source missing-shard recovery; require context recipe/input encoding |
| `literature_proxy_production/codec.py` | Imported unchanged lossless physical-bank codec; no old scientific artifact identity reused |
| `literature_context/kernel.py`, `transform.py`, `campaign.py`, `worker.py` | Import frozen equations, input encoding, approved pilot authentication and saved CONTEXT endpoints without edits |
| `literature_proxy_v3/inputs.py`, `contracts.py`; v1/v2 parent modules | Completed pilot ancestry, original full-precision COUNT38 rates and physical evidence |
| `cms2jc2_production/population.py`, `contracts.py`; `literature_proxy/population.py` | Existing TRAIN_1M/250k-validation/1M-test metadata selection and raw offline conversion, not CMS-fitted response generation |
| `cms2jc2_response/bridge.py`, `readers.py`, `measurement.py`; `jetclass2_delphes/inventory.py`, `split_registry.py`; `data/cache_contracts.py` | Physical validation, authenticated source reads, measured resources, tree/entry identities and artifact hashing |

Package paths above are under `src/hlt_classification/`. Thin wrappers adapt
`scripts/jetclass2_literature_proxy_dataset.py`,
`scripts/queue_jetclass2_literature_proxy_dataset.sh`, and
`sbatch/run_jetclass2_literature_proxy_dataset.sh` into the corresponding
`literature_context_dataset` files. `tests/test_literature_context_production.py`
adapts `tests/test_literature_proxy_production.py`, reuses the original completed
v1/v2 synthetic ROOT pilot fixtures, and adds context-specific approval,
composition, representation, inverse and namespace checks.

## 2026-10-06: lower noise and context coupled tracking pilot

Internal donor checkout **`0a1013e9fa0571333242871b7cc34b2dfe0ba464`**.
New `literature_context/`, CLI/queue helper and CPU worker are additive.
No third-party source or `Fresh_check` import. Old NOISE_V3, generator, input
transforms and datasets remain unchanged; no fitted CMS response is reused.

| Donor paths under `src/hlt_classification/` | Use or adaptation |
| --- | --- |
| `literature_proxy_v3/kernel.py` | Adapt independent smearing implementation with tracking amplitude 1 instead of 4 and kinematic amplitude 1 instead of 2; original function remains the noisy control |
| `literature_proxy_v2/kernel.py`; `literature_proxy/kernel.py` | Exact frozen topology, PID/drops/merges, full-precision calibration validation and original keyed random streams |
| `literature_proxy_v3/inputs.py`; v1/v2 ancestor input and campaign modules | Authenticate completed training-only parent, physical NPZ endpoints and separate diagnostic ancestry, without reading ROOT |
| `literature_proxy_v3/campaign.py`, `contracts.py`, `worker.py`, `diagnostics.py` | Adapt single-job dry/live safeguards, source closure, atomic artifact patterns and per-file spawn execution; reuse count reduction and descriptive statistics |
| `literature_proxy/diagnostics.py`, `worker.py`, `contracts.py`; `data/cache_contracts.py` | Metrics, PDF/CSV, array serialization, sampled process RSS, file/content hashes and immutable publication |
| `cms_proxy_ladder/inputs.py`, `contracts.py`; `jetclass2_delphes/inputs.py` | Reuse unchanged 17-input physical frontend for saturation audit; optional new four-channel tracking encoding is separately versioned and never installed in an old campaign |
| `cms2jc2_response/bridge.py`; `jetclass2_delphes/contracts.py` | Physical validity, p4/wrapped angles and safe relative file paths |

Adapted thin wrappers from `scripts/jetclass2_literature_proxy_noise.py`,
`scripts/queue_jetclass2_literature_proxy_noise.sh` and
`sbatch/run_jetclass2_literature_proxy_noise.sh`. Tests reuse original synthetic
particle fixtures and completed v1/v2 pilot fixtures. Context equations and
inverse are new synthetic benchmark choices, not fitted/literature-derived
detector parameters. New `JC2_LITERATURE_CONTEXT_*/v1` artifacts preserve all
original donor schemas and calibration parents.

## 2026-10-05: restore original-memory K2 continuation without another native gate

Scientific donor: **`c891da0d45dd3251dea9ea72df975bb96bae3570`**, unchanged.
Accepted execution donor: **`71c1bde2d759b11cddb1f5cfbc1d199847bd6eef`**.
Implementation base and unused retirement target:
**`c640942f8a4a91086016fa4bd151ae5d3bc9c31a`**. No third-party code copied.

| Donor paths | Use/adaptation |
| --- | --- |
| `k2_segmented/memory_migration.py` (c640942f) | Existing exact checkpoint/prefix authentication and independent byte-copy machinery; narrow v3 dispatch to original-memory restoration |
| `k2_segmented/campaign.py`, `runtime.py` (c640942f) | V3 omits native gate/launcher and completed D025 segments, restores v1 resources, retains exact-ID retirement/submission and hardware/software checks |
| `k2_segmented/training.py` (71c1bde2) | Unchanged scientific full-state kernel; byte equality required against the accepted donor |
| `scripts/jetclass2_k2_segmented.py`, `queue_jetclass2_k2_128g.sh` (c640942f) | Add explicit restoration/CPU preparation modes and separate dry-first restoration helper; worker and original scientific bootstrap unchanged |
| `tests/test_k2_segmented_memory.py` (c640942f) | Reuse tiny real CPU state/resume fixtures for original-vs-restored final state and downstream teacher tests |

Package paths above are under `src/hlt_classification/`. Added
`k2_segmented/restoration.py`, `scripts/queue_jetclass2_k2_restored.sh`,
`tests/test_k2_segmented_restoration.py`, restoration contract and plan amendment.
New spec v3 and `RESTORE_RESUME_IMPORT`, `ABANDONED_EXECUTION`,
`REUSED_NATIVE_GATE` artifacts under `K2_SEGMENTED_*/v1`. The actual original
native acceptance retains its original identity; no new measurement is claimed.
Other campaigns, prior specs, completed outputs and final-test seal stay intact.

## 2026-10-05: 128-GiB SPORC K2 full-state continuation

Execution donor: **`71c1bde2d759b11cddb1f5cfbc1d199847bd6eef`**. Scientific donor
remains **`c891da0d45dd3251dea9ea72df975bb96bae3570`**. Developed against local
HEAD `984ec4560cce1c3b6610f69e928dd248218e67cf`. No third-party files copied.

| Donor paths | Use/adaptation |
| --- | --- |
| `src/hlt_classification/k2_segmented/campaign.py`, `runtime.py` (71c1bde2) | Backward-compatible v1 plus v2 graph/resource routing; authenticate the existing segmented donor and import its part2 endpoint into a fresh root |
| `src/hlt_classification/k2_segmented/training.py` (71c1bde2) | Unchanged full-state kernel/serialization; require byte parity with the donor at registration and runtime |
| `scripts/jetclass2_k2_segmented.py`, `queue_jetclass2_k2_segmented.sh`; `sbatch/run_jetclass2_k2_segmented.sh` (71c1bde2) | New `create-128g` CLI mode and separate dry-first helper; existing scientific bootstrap and Slurm worker unchanged |
| `tests/test_k2_segmented.py` (71c1bde2) | Reuse tiny seeded CPU model/cache fixtures; test exact old-vs-migrated continuation and downstream banks |
| `jetclass2_delphes/concat_k2_runtime.py`, `concat_k2_campaign.py`, `salience_learned_training.py`, `banks.py`, `production.py`, `execution.py`; `data/cache_contracts.py`; `scouting/hcwdl_exact_dag_submission.py`, `hcwdl_recovery.py` (under `src/hlt_classification/`, c891da0d) | Existing original scientific semantics, source/ledger authentication, byte hashes, immutable publication and exact-ID retirement/submission; no scientific donor modules edited |

Added `k2_segmented/memory_migration.py`, `scripts/queue_jetclass2_k2_128g.sh`,
`tests/test_k2_segmented_memory.py`, the 128-GiB contract and active-plan
amendment. New spec `K2_SEGMENTED_CAMPAIGN_SPEC/v2`; new import/copy artifacts
`K2_SEGMENTED_MEMORY_RESUME_IMPORT/v1` and `K2_SEGMENTED_RESUME_COPY/v1`.
Existing checkpoint/payload/task schemas retain their identities. The D025
payload keeps its donor binding; v2 execution receipts explicitly record the
migration. No `Fresh_check`, other campaign, final-test, or environment changes.

## 2026-10-04: additive OSCAR NOISE-K2 targeted recovery

Donor commit: **`0ffb4ba54b02cea6558db5ea8a44995c76ef22d7`**.
New `noise_k2/recovery.py` and `recovery_runtime.py` add an isolated recovery
namespace and output resolver; all original scientific Python modules and
environment helpers must remain byte-identical before native acceptance reuse.

| Donor | Recovery reuse/adaptation |
| --- | --- |
| `src/hlt_classification/noise_k2/campaign.py`, `contracts.py` | Original immutable graph, completed receipts, resource commands, safe paths and byte checks; adapt exact-ID replacement planning into a sibling root |
| `src/hlt_classification/noise_k2/runtime.py`, `data.py` | Same caches, seeded model factory, kernels, inference, selected state and preparation; fit/reduce I/O adapted to resolve original or replacement teachers without changing global source state |
| `src/hlt_classification/jetclass2_delphes/salience_learned_training.py`, `banks.py`, `reporting.py`, `model.py`, `execution.py` | Unmodified scientific kernels, ordered T2 banks, recovery metrics, environment and allocation checks |
| `src/hlt_classification/data/cache_contracts.py`; `src/hlt_classification/scouting/hcwdl_exact_dag_submission.py`, `hcwdl_recovery.py` | Atomic immutable publication, content/file hashes, exact submission journals and ledgers |
| `scripts/noise_k2.py`, `scripts/queue_noise_k2.sh`, `sbatch/run_noise_k2.sh`, `sbatch/jetclass2_delphes_common.sh` | Separate recovery CLI/dry helper/worker; unchanged absolute OSCAR environment helper |
| `tests/test_noise_k2.py`, `tests/test_jetclass2_dzfix_fusion_chain.py` | Local graph/cache fixtures and seeded tiny model tests; synthetic adapter parity is not native OSCAR evidence |

New schema family: `NOISE_V3_K2_RECOVERY_*/v1`. Existing scientific/report
artifacts retain their original contracts, hashes and source/job identities.
No donor files modified, external copies, `Fresh_check` imports, producer
changes, or final-test capabilities introduced.

## 2026-10-04: NOISE_V3 K2 100k/50k consumer on OSCAR

Internal donor checkout: **`71c1bde2d759b11cddb1f5cfbc1d199847bd6eef`**.
Additive package `src/hlt_classification/noise_k2/`, CLI, queue helper and worker;
old native Delphes/SPORC K2 and producer source are unchanged. No third-party
copy, `Fresh_check`, detector generator, old assignment or trained weight import.

| Donor under `src/hlt_classification/` | Reuse/adaptation |
| --- | --- |
| `jetclass2_delphes/concat_k2_views.py`; `scouting/hcwdl_fullcard_salience_matcher.py`, `hcwdl_fullcard_salience_contracts.py`, `hcwdl_fullcard_bottleneck_matcher.py` | Same pre-retention endpoint salience, exact capacity-two integer objective and independent exhaustive solver; physical endpoints replace native Delphes arrays |
| `cms2jc2_response/bridge.py`, `readers.py`; `cms_proxy_ladder/inputs.py`, `views.py` | Physical GeV/mm/validity schema, authenticated ROOT context, 17-feature adapter and salience projection; adapt interpolation with atomic validity groups |
| `literature_proxy_production/population.py`, `output.py`; `literature_proxy/population.py`; `jetclass2_delphes/selection.py`, `split_registry.py` | Frozen shard/raw-entry identity joins, physical bank schema, offline-only ROOT fields, scalar labels and natural Hamilton quotas |
| `jetclass2_delphes/concat_k2_campaign.py`, `concat_k2_data.py`, `cache.py`, `salience_learned_data.py` | Full ten-node recipe/seeds, stratified validation partition and ragged RAM-only/indexed caches |
| `jetclass2_delphes/concat_k2_runtime.py`, `concat_k2_model.py`, `salience_learned_training.py`, `salience_learned_graph.py`, `model.py`, `acceptance.py` | Reuse native model/lossless pair storage, scoped FP32/BF16 parity, full original CE/KD kernel, longest-real-batch stress and representative miniature selection; adapt fresh OSCAR admission |
| `jetclass2_delphes/banks.py`, `reporting.py`, `production.py`, `execution.py` | Ordered T2 banks, metrics/recovery, clean pushed source, existing OSCAR site/allocation |
| `data/cache_contracts.py`; `scouting/hcwdl_exact_dag_submission.py`, `hcwdl_recovery.py` | Immutable publication, canonical/file hashes, exact-ID submission journal and dry/live ledgers |

Also imports the independently developed, previously untracked
`src/hlt_classification/literature_proxy_consumer.py` without editing it.
That working-tree donor has no published commit asserted here; its SHA256 at
integration is `1bd2152aff3febb94c04bf5d0473a0f8da11af8551bb0885391d7c68a1ac692d`.
Include that reader and its handoff/contract/tests in the eventual pushed source.
The source dataset's immutable manifest and the original K2 formula's copied
JSON/raw SHA256 are independent provenance anchors, not this development HEAD.

The new Slurm worker sources the existing absolute-path
`sbatch/jetclass2_delphes_common.sh` OSCAR environment. Tests reuse physical
fixtures and real synthetic relocated ROOT/NPZ fixtures from
`test_literature_proxy_consumer.py`/`test_literature_proxy_production.py` and the
ragged cache fixture in `test_jetclass2_dzfix_fusion_chain.py`.
New scientific/execution namespace: `NOISE_V3_K2_*/v1`.

## 2026-10-03: relocated literature NOISE_V3 consumer for Oscar

Internal donor checkout **`71c1bde2d759b11cddb1f5cfbc1d199847bd6eef`**.
New root module `src/hlt_classification/literature_proxy_consumer.py` and thin
inspection/smoke CLI reuse these implementations without modifying original
producer packages or their source-bound science. No third-party code copied,
no `Fresh_check` import, no generator rerun, and no producer metadata rewrite.

| Donor path under `src/hlt_classification/` | Reuse/adaptation |
| --- | --- |
| `data/cache_contracts.py` | Canonical content hashes, file SHA256 and JSON reads |
| `literature_proxy_production/contracts.py`, `output.py`, `codec.py` | Original schema/parent validation, safe local relative paths, physical bank schema and `Particles` decoding |
| `literature_proxy_production/population.py`; `cms2jc2_production/population.py`, `contracts.py` | Exact population/shard metadata, original-entry mask decoding, canonical jet IDs; no split rebuilding |
| `literature_proxy/population.py` | Offline-only branch set and raw native-mm physical conversion; adapt bounded ROOT read loop for paired consumption |
| `literature_proxy_v2/kernel.py`; `literature_proxy_v3/kernel.py`, `contracts.py` | Validate frozen count calibration, NOISE_V3 recipe and direct pilot evidence; no fitting/generation |
| `cms2jc2_response/bridge.py`, `readers.py` | Physical `Particles` validity and ROOT byte authentication before/after access, including early close |
| `jetclass2_delphes/inventory.py`, `split_registry.py`, `selection.py`, `contracts.py` | Original inventory/profile contracts, tree-cycle selection, mapped labels and canonical identity semantics |

Tests reuse existing synthetic production fixtures from
`tests/test_literature_proxy_production.py` (including its v1/v2/v3 pilot fixture
chain), copy their actual ROOT/banks to a new layout and forbid original paths.
New contract `JC2_LITERATURE_RELOCATED_READER/v1` describes read-only runtime
inspection; scientific production/pilot/split contracts remain unchanged.
Dataset producer lineage continues to come from the original, independently
anchored manifest/study, not from this reader-development donor commit.

## 2026-10-03: K2 epoch-boundary segmented continuation

Scientific donor commit **`c891da0d45dd3251dea9ea72df975bb96bae3570`**;
executor developed against local HEAD `98095b680cb968a06bd766cf114007883ce053cc`.
No third-party code copied and no `Fresh_check` import. New package
`src/hlt_classification/k2_segmented/`, CLI/queue wrapper and Slurm worker are
additive; they do not modify the donor checkout or any old scientific module.
The CLI binds every scientific import to the original donor checkout and only
the new executor package to the new pushed checkout.

| Donor path (under `src/hlt_classification/`) | Use |
| --- | --- |
| `jetclass2_delphes/salience_learned_training.py` | Adapt original epoch loop; reuse original AdamW, CE/KD loss, BF16, inference, metrics and selection key; add full-state save/resume without changing schedule or patience |
| `jetclass2_delphes/salience_learned_graph.py` | Frozen training constants and absolute-epoch LR function |
| `jetclass2_delphes/concat_k2_runtime.py`, `concat_k2_campaign.py` | Read-only source registration, completed-prefix acceptance, native models/caches, source teacher bank, metrics/reporting/export primitives |
| `jetclass2_delphes/concat_k2_submit.py` | Original canonical science plan and exact walltime parsing |
| `jetclass2_delphes/concat_k2_model.py`, `model.py` | Original lossless pair storage and scoped deterministic parity backend; installed environment identity and model contract |
| `jetclass2_delphes/salience_learned_data.py`, `banks.py`, `contracts.py` | Indexed miniature cache, identity-joined T2 banks and safe relative paths |
| `jetclass2_delphes/production.py`, `execution.py`, `submission.py` | Clean pushed source, SPORC allocation checks, submission-intent protection |
| `data/cache_contracts.py`; `scouting/hcwdl_exact_dag_submission.py`, `hcwdl_recovery.py` | Immutable publication, content hashes, exact-ID journals and ledger validation |
| `sbatch/jetclass2_delphes_common.sh` | Existing absolute-path SPORC environment setup; inherited from the new pinned executor checkout |

New contracts are `K2_SEGMENTED_*/v1`, `K2_SEGMENT_CHECKPOINT/v1`, and
`K2_FULL_TRAINING_STATE/v1`. Authority and evidence limits are documented in
the K2 plan amendment, segmented contract and HANDOFF. Existing full-size and
pilot schemas are not rewritten.

## 2026-10-03: genuine CMS common-interface direct/coarse comparison

Internal donor checkout: **`b7fd9c63d5f747983b4ab603b86a315d68bcc24a`**.
New implementation under `src/hlt_classification/cms_feature_ladder/`; no
third-party source copied and no runtime `Fresh_check` import. Older scientific
modules/specs are unchanged. New authority is CMS_FULLSIM_FEATURE_LADDER/v1.

| Donor (relative to `src/hlt_classification/`) | Reuse/adaptation |
| --- | --- |
| `scouting/schema.py`, `splits.py`, `selective_assignment.py`, `identity.py` | Authenticated native file roles, 15 labels, seed1337 proportional identity-rank subsets; new200k/50k budget |
| `scouting/hcwdl_homotopy.py`, `repair.py`, `inputs.py` | Raw endpoint projection with validity, retaining lost tracks; native21 transforms for opt-in control, no inversion of clipped inputs |
| `cms_proxy_ladder/views.py`, `inputs.py`, `campaign.py` | PT_LINEAR matching, persistent support, rational U/D coordinates, keyed interpolation, shared17 normalization, paired coordinate seeds; raw binary PID flags explicitly preserved for CMS |
| `cms2jc2_response/bridge.py` | Validated physical `Particles` container only; **not** `from_cms`, whose lost-track filtering would change this population |
| `cms_salience_learned/data.py`, `storage.py`, `training.py` | Bounded raw ROOT projection/process map, RAM batch interface, immutable byte IO; tensor/autocast/AdamW utilities, no learned-fusion scientific graph |
| `jetclass2_delphes/campaign.py`, `runner.py`, `acceptance.py`, `model.py`, `execution.py` | Exact literature schedule, adapted15-class kernel/parity, installed environment hashing and SPORC allocation checks |
| `models/particle_transformer.py`, `scouting/evaluation.py` | Installed Weaver factory/backbone; CMS metrics with explicit censored R50 representation |
| `scouting/hcwdl_authorization.py`, `hcwdl_exact_dag_submission.py`; `data/cache_contracts.py` | Pushed clean source validation, exact DAG/journal, content hashes/immutable publication |
| `cms2jc2_response/measurement.py`; `sbatch/jetclass2_delphes_common.sh` | Process-tree RSS measurement and absolute SPORC environment setup |

The new runner is single-view, cold-start, immediate-parent C25/P75, not the
native learned-fusion acquisition/withdrawal donor. The opt-in CMS21 control
shares support/p4/identity changes with SHARED17 and holds trim=False. This
compares representations, not an assertion of an exact old21-feature result.
Tests/evidence and the remaining real-GPU gate are recorded in HANDOFF.

## 2026-10-03: additive literature gate-to-science follow-up

In-repository donor commit **`99a1e231e38fe5a07ce5a31a4f4a74bb9ec40539`**.
No external code copied, license change or `Fresh_check` import. New reusable
execution module is `src/hlt_classification/literature_ladder_followup.py`.

| Donor | SHA-256 at inspection | Retained use |
| --- | --- | --- |
| `cms_proxy_ladder/literature.py` | `316734ea5dd44d3fb738c00e81c3cfb88aeaa41c6456177bb7124ebc1a9d7fd0` | Original pinned gate/profile/science validation and exclusive submit claim; no changes |
| `cms_proxy_ladder/submission.py` | `a6e969c62c049cb2f3721afee06447fe6ba53109e7e86a81a2266eb1f9a47d02` | Exact recomputed plans, canonical dry ledger and site checks, delegated to old CLI |
| `scouting/hcwdl_exact_dag_submission.py` | `65c031db95630d034d19112d8d0ff76af5189fc9a43e7ad7e9da1430b5df4a6b` | Authenticate old gate journal; original idempotent science DAG submission |
| `cms2jc2_production/recovery_controller.py` | `eb0cb1d862e3903ef9799408f97e4b6f23adda7726778ed7ae8c0c915cd785d5` | Pattern only: kernel controller lock and bounded exact-job polling; no imports |

Donor paths above are relative to `src/hlt_classification/`. The embedded
read-only inspection runs in a fresh interpreter importing the original gate
checkout, not the controller checkout. This preserves already-running source
locks. Execution amendment and operational receipt v1 are documented in
`docs/contracts/JETCLASS2_LITERATURE_LADDER_FOLLOWUP.md`; scientific schema
versions remain unchanged. Tests: `tests/test_literature_ladder_followup.py`;
local/remote evidence and remaining launch step are recorded in `HANDOFF.md`.

## 2026-10-03: literature V3 200k/50k SPORC debug DIRECT + COARSE

Repository-local donor commit **`ebd5bc1ae4eb6d6c11bfba5e1a6cdc5fd487653a`**.
No third-party donor or runtime `Fresh_check` import. The dataset generator
and its recipe/calibration are reused unchanged; new classifier versions
explicitly identify the literature rather than CMS-fitted endpoint.

| Donor files (under `src/hlt_classification/` unless noted) | Reuse/adaptation |
| --- | --- |
| `cms_proxy_ladder/release.py`, `data.py`, `views.py`, `inputs.py`, `cache.py` | Versioned 200k/50k literature release, native-unit paired reader, fresh salience matching, persistent skeleton/offline tail, exact D000 and existing RAM cache |
| `cms_proxy_ladder/campaign.py`, `gate.py`, `production.py`, `submission.py`, `contracts.py` | Existing direct/coarse coordinates, seeds, C25/P75 training, reports and exact DAG; new literature source lock, full-population debug-only gate/profile and nine-fit science variant |
| `literature_proxy_production/campaign.py`, `population.py`, `output.py`, `contracts.py`; `literature_proxy_v3/contracts.py`; `literature_proxy/population.py` | Validate frozen source/bundle/population, ordinary role manifests and physical blocks; convert original offline fields without CMS unit/sign conversion; no generator execution |
| `cms2jc2_production/output.py`; `cms2jc2_response/bridge.py`, `generation_benchmark_data.py` | Reuse physical-bank decoder/schema and offline branch allowlist only, not learned response or CMS bridge conversion |
| `jetclass2_delphes/{campaign,model,runner,acceptance,execution,reporting,inventory,selection,contracts}.py` | Reuse model/recipe, installed-Weaver parity, single-GPU kernels, per-class R50, SPORC site and inventory-bound identities |
| `scouting/hcwdl_exact_dag_submission.py`, `hcwdl_recovery.py` | Existing exact-ID journal/ledger; optional explicit environment for sanitized literature submissions, plus adapter-owned exclusive submission claim |
| `scripts/jetclass2_cms_proxy_ladder.py`, `run_jetclass2_cms_proxy_ladder_task.py`; `sbatch/jetclass2_delphes_common.sh` | Thin new CLI/helper; retain existing worker and absolute environment setup |
| `tests/test_literature_proxy_production.py`, `test_cms_proxy_ladder.py` | Synthetic producer fixtures and classifier contract patterns; new reader/gate/DAG/claim tests |

New authority: `docs/plans/JETCLASS2_LITERATURE_V3_200K_DIRECT_COARSE_PLAN.md`
and `docs/contracts/JETCLASS2_LITERATURE_V3_LADDER.md`. No historical artifact
is relabeled or overwritten; no final-test particle capability is added.

## 2026-10-03: frozen literature v3 full production dataset

Repository-local donor commit **`d505e807500b7e0f4149f940413c542b63b308d2`**.
New code: `src/hlt_classification/literature_proxy_production/`; old scientific
modules remain unchanged. No third-party donor or `Fresh_check` dependency.

| Donor files (under `src/hlt_classification/`) | Production reuse/adaptation |
| --- | --- |
| `literature_proxy_v3/kernel.py`, `inputs.py`, `worker.py`, `campaign.py`, `contracts.py` | Reuse frozen generator and recipe verbatim, completed v3/v2/v1 authentication, saved-v3 replay, exact source/hash chain |
| `literature_proxy_v2/kernel.py` | Validate inherited full-precision calibration and original parent lineage; never recalibrate |
| `literature_proxy/population.py`, `campaign.py` | Reuse exact raw OFFLINE conversion, padded particle keys and branch allowlist, clean/pushed source and exclusive failed-attempt locks |
| `cms2jc2_production/population.py` | Reuse metadata-only TRAIN_1M population builder including unchanged validation subset domain; new sharding and raw reader do not call its CMS-compatible physical converter |
| `cms2jc2_production/contracts.py`, `storage.py` | Adapt truthful flags/atomic references and path-safe quota reservations into a separate versioned namespace |
| `cms2jc2_production/campaign.py` | Adapt fresh persistent-root non-overlap checks only; no learned campaign/gate dependency |
| `cms2jc2_production/submission.py` | Adapt exact array-element identity, scheduler/environment checks, journals and terminal-only recovery; replace staged pilot/bulk submissions with one frozen preflight/array/finalizer DAG |
| `cms2jc2_production/output.py` | Adapt physical-bank validation, receipt binding and manifest checks; add train/validation-only independent releases |
| `cms2jc2_response/generation_benchmark_engine.py` | Copy pure physical digest, pack, lossless codec/readback; adapt bounded ordered process/writer design with ONLY literature v3 generation |
| `cms2jc2_response/readers.py`, `measurement.py`, `bridge.py` | Reuse byte-authenticated ROOT open, Linux process-tree measurement and physical schema; no CMS reading/learning or units conversion |
| `data/cache_contracts.py`, `jetclass2_delphes/inventory.py`, `contracts.py`, `split_registry.py` | Atomic bytes, identities, tree/inventory authentication and canonical split masks |

The new CLI/queue/Slurm wrapper adapt
`scripts/jetclass2_literature_proxy_noise.py`,
`scripts/queue_jetclass2_literature_proxy_noise.sh` and
`sbatch/run_jetclass2_literature_proxy_noise.sh`. Tests reuse synthetic ROOT
and sealed-pilot fixtures from `tests/test_literature_proxy_v3_campaign.py`,
`tests/test_literature_proxy_v2_campaign.py`, `tests/test_jetclass2_delphes.py`
and `tests/test_jetclass2_delphes_split_registry.py`, with scheduler/codec
test patterns from `tests/test_cms2jc2_proxy_production.py`. Production role
counts are not configurable; only isolated tests patch miniature counts.

## 2026-10-02: final noise-only literature proxy v3

Repository-local donor commit `d4b5d5fda19a6401442cb6b605ad22c2b7013e5e`.
New package: `src/hlt_classification/literature_proxy_v3/`. No external source,
learned CMS response or `Fresh_check` runtime dependency.

| Donor files (relative to `src/hlt_classification/`) | v3 reuse/adaptation |
| --- | --- |
| `literature_proxy_v2/kernel.py` | Reuse topology and probability validation unchanged; copy generation equations with only tracking amplitude 3 to 4 and kinematic amplitude 1.5 to 2 |
| `literature_proxy_v2/inputs.py`, `worker.py` | Authenticate original OFFLINE and saved count38 blocks and complete receipts; replace calibration passes with exact inherited calibration, add every-jet structure/count equality |
| `literature_proxy_v2/campaign.py`, `contracts.py` | One-job source pinning, immutable publication, submission protections; new v3 namespace and pinned v2 ancestors |
| `literature_proxy/kernel.py`, `worker.py`, `diagnostics.py` | Unchanged random streams, physical arrays/RSS, inclusive moments/conditional statistics/exports; add separate v3 paired-response diagnostics |
| `literature_proxy/contracts.py`, `cms2jc2_response/bridge.py`, `data/cache_contracts.py`, `jetclass2_delphes/contracts.py` | Hash/reference primitives, validated particles, p4/wrapped phi, deterministic NPZ and confined artifact paths |

The new noise CLI, queue helper and Slurm wrapper adapt the donor's
`scripts/jetclass2_literature_proxy_count38.py`,
`scripts/queue_jetclass2_literature_proxy_count38.sh` and
`sbatch/run_jetclass2_literature_proxy_count38.sh`. Tests reuse the donor's
`tests/test_literature_proxy.py` and `tests/test_literature_proxy_v2_campaign.py`
fixtures (including their JetClass2 synthetic ROOT/split fixtures). All v1/v2
source remains unchanged. No third-party source/license changes. Verification
and genuine Tigris status are recorded in `HANDOFF.md`.

## 2026-10-02: count-targeted literature proxy v2

Repository-local donor commit `08cbf06309ff7bb2bdeccf6da2874e21de43da87`.
New code lives under `src/hlt_classification/literature_proxy_v2/`. No external
donor, learned CMS response, or `Fresh_check` dependency was introduced.

| Donor files (under `src/hlt_classification/`) | Reuse/adaptation |
| --- | --- |
| `literature_proxy/kernel.py` | Physical response equations, crowding, common per-operation random streams and Response container; v2 separately scales PID/tracking/kinematics and changes loss topology |
| `literature_proxy/campaign.py`, `contracts.py` | Clean/pushed source guard, exclusive lock and immutable/hash primitives; new v2 campaign and artifact namespace |
| `literature_proxy/population.py` | Recompute exactly the same authenticated training selection; no new ROOT reader |
| `literature_proxy/worker.py` | Authenticate completed parent receipt, physical-array extraction, per-process RSS conventions; new two-pass calibration/generation worker |
| `literature_proxy/diagnostics.py` | Existing moments, masks, fixed-bin/conditional statistics and PDF/CSV; optional bins/sides preserve v1 defaults |
| `cms2jc2_response/bridge.py`, `data/cache_contracts.py`, `jetclass2_delphes/contracts.py` | Physical particle constraints, deterministic NPZ, byte/content hashes and confined artifact paths |

New CLI/helper/Slurm wrapper adapt the donor's
`scripts/jetclass2_literature_proxy.py`,
`scripts/queue_jetclass2_literature_proxy_pilot.sh`, and
`sbatch/run_jetclass2_literature_proxy_pilot.sh`. Synthetic fixtures reuse
`tests/test_literature_proxy.py`, `tests/test_jetclass2_delphes.py`, and
`tests/test_jetclass2_delphes_split_registry.py` from the same commit. No
original scientific v1 kernel, population or saved result was replaced.
The new count calibration is explicitly a training-only benchmark choice,
not a CMS/literature-measured rate or a classification-performance fit.

## 2026-09-30: proxy dataset consumer documentation

The consumer handoff references existing production, recovery, bridge and
JetClass2 label/identity APIs at repository commit
`3655955f6ef7184cc6fdb3b0e68bdc6d089e9fe2`; it migrates no implementation or
external donor files. Its example test reuses synthetic ROOT/split fixtures
from `tests/test_jetclass2_delphes.py` and
`tests/test_jetclass2_delphes_split_registry.py` at that commit. The separately
tested production reader remains authoritative for physical bank validation.
No new schema, response semantics or final-test permissions are introduced.

## 2026-09-30: committed proxy training audit

Repository-local donor commit `314e9faa680cad677aca9c9cd158d83cf973ea0b`:

- `cms2jc2_production/{campaign,contracts,population,output,recovery}.py`:
  import immutable provenance, safe paths, exact offline-only source membership,
  physical bank schema/identity checks and repaired receipt validation unchanged.
- `cms2jc2_response/{dev_diagnostics,metrics,features,bridge}.py`: reuse canonical
  observables, validity masks, offline-only cohorts, transforms and bridge units.
  The independent audit compares saved populations; it does not reuse a paired
  same-population score for unpaired CMS-vs-JetClass2 samples.
- `cms2jc2_response/{dev_data,provenance,generation_benchmark_engine}.py`:
  reuse authenticated metadata references, environment/source evidence and
  physical digests. The audit never calls the generation kernel.
- `tests/test_cms2jc2_proxy_production.py`,
  `test_cms2jc2_generation_benchmark.py`, and JetClass2 split-registry fixtures:
  reuse synthetic physical schemas and ROOT data for the new audit tests.

New logic lives in `cms2jc2_proxy_audit/`; existing frozen producer/response
modules are unchanged. New plan and `CMS2JC2_PROXY_AUDIT_* /v1` contracts prevent
diagnostic completion from masquerading as dataset completion or qualification.
No external donor files, Fresh_check imports or remote changes.

## 2026-09-30: source-pinned production execution recovery

Repository-local donor commit `b6f88defce6f357969b2e6ded80585db439540a3`,
plus the local array-element authentication fix documented immediately below:

- `cms2jc2_production/{campaign,submission,worker,output}.py`: retain original
  admission, exact submission intents, resource guards, worker generation
  and bank validation. Enumerated operational entry points add explicit
  repair lineage; the generation kernel is unchanged.
- `cms2jc2_production/{contracts,storage,population}.py`: reuse immutable
  content/parent hashes, safe paths, claims, reservations and frozen identities
  without changes. New recovery modules use these rather than rebuilding data.
- Existing production CLI/queue helpers: reuse Tigris environment activation,
  absolute source paths, single-thread numerical libraries and detached log
  handling in new `cms2jc2_proxy_recovery.py` and its queue shell helper.
- `tests/test_cms2jc2_proxy_production.py`: reuse synthetic bank, scheduler and
  source fixtures for `test_cms2jc2_proxy_recovery.py`; local mocked timings
  do not replace the required real retained/retried pilot measurements.

New execution-repair plan/contract version the repaired attempt, shard and
manifest kinds instead of changing legacy artifact meaning. No external
donor migration, Fresh_check runtime import, scientific refit or remote action.
Exact local verification and remaining remote step are recorded in HANDOFF.

## 2026-09-30: exact production Slurm array-element lookup

Repository-local donor commit `b6f88defce6f357969b2e6ded80585db439540a3`:
`src/hlt_classification/cms2jc2_production/submission.py` and
`tests/test_cms2jc2_proxy_production.py`. Correct raw-parent ambiguity in
`worker_identity`/`fields`, retaining original submission and resource guards.
No donor file migration or Fresh_check import; no response kernel, physical
writer, population, random-key or scientific-schema change. Production plan
and contract clarify the exact selector and unchanged source-pinning boundary.
Test and user-supplied remote evidence are recorded in HANDOFF.

## 2026-09-29: explicit 36-of-57 frozen confirmation amendment

Repository-local donor commit `f3db2ce8ff0f8e5716aab19df4cba9677f72c06c`:

- `cms2jc2_response/frozen_joint_campaign.py`, `frozen_joint_metrics.py`,
  `bdz_worker.py`, `bdz_audit_metrics.py`: unchanged frozen protocol, additive
  diagnostic merging, file-bootstrap checks and significance diagnostics.
  Called by new `cms2jc2_response/reduced_confirmation.py`; original complete
  membership/report validation remains untouched. Only the new contract
  authorizes the fixed prefix, with explicit missing-source/file coverage.
- `cms2jc2_response/dev_submission.py`, `dev_campaign.py`, `dev_worker.py`,
  `dev_data.py`: reuse exact-intent/ledger scheduler identity, immutable outputs,
  source/environment authentication and publication; new dispatch/authorization
  for a single report task. Retirement is separately authorized and preserves
  completed excluded artifacts and all included artifacts.
- `cms2jc2_production/{campaign,output}.py`: new explicit reduced-evidence
  study/manifest kinds. Original full-evidence route and physical generation
  semantics remain unchanged. No legacy donor/Fresh_check runtime import,
  license change, data migration, raw-particle read or remote action here.
- Existing synthetic frozen diagnostics and production bank fixtures support
  new retirement/refusal, report aggregation, opt-in, provenance and sealed
  reader tests. Test evidence is recorded in HANDOFF, not remote qualification.

## 2026-09-29: 2.25M frozen proxy dataset production

Repository-local donor commit `b88a35123e6f7658a4b1679cd797b7f157c4602b`:

- `cms2jc2_response/generation_benchmark_engine.py`: unchanged JOINT replica-0
  generation, ordered spawn pool, physical/key digests, float64 bank packing,
  deterministic lossless encoding and full schema/readback checks. Called
  directly; old benchmark writer/caps are not weakened or reused for bulk.
- `cms2jc2_response/generation_benchmark_data.py`, `readers.py`, `bridge.py`:
  original offline-only branch capability, bounded windows, authenticated ROOT
  open, stable row identity, unit/validity conversion and physical schema.
  Production adapts the window reader to its new versioned population; original
  reader semantics remain unchanged.
- `cms2jc2_response/generation_direct_campaign.py`,
  `frozen_joint_campaign.py`, `dev_campaign.py`, `dev_data.py`, `provenance.py`:
  completed gate/confirmation and immutable donor authentication, existing
  scientific source matching and native-library/environment pinning.
- `cms2jc2_response/measurement.py`: real Linux process-tree resource sampling.
- `jetclass2_delphes/{inventory,split_registry,contracts}.py` and
  `data/cache_contracts.py`: input/profile validation, packed masks, existing
  original identities, canonical hashes and atomic no-overwrite publication.
- Existing direct/benchmark tests, `test_cms2jc2_bounded.LocalMeasurement`, and
  JetClass2 registered synthetic ROOT fixtures: fixture inputs and local-only
  measurements. Synthetic timing is explicitly forbidden as bulk admission.
- Existing response queue/wrapper scripts: activation, absolute project paths,
  single-thread child environments and disconnect-safe heartbeat patterns.

New package `cms2jc2_production`, separate CLI/worker/helper, plan and
`CMS2JC2_PROXY_* /v1` contracts add durable production storage, new authorized
test materialization semantics, pilot admission, arrays, recovery, manifests
and the train/validation physical-bank reader. No fitted response, tracking
map, random key algorithm or old test-access boundary is modified.

## 2026-09-29: direct Tigris frozen-JOINT generation

Repository-local donor commit `9215fdcf940c116d5d89675a42bd8c82bdec05ea`:

- `cms2jc2_response/generation_benchmark_{campaign,data,engine,worker}.py`:
  authenticated GEN lineage/bundle/TRAIN membership, unchanged bounded
  offline-only ROOT reader, bounded once-per-process generation, lossless
  NPZ/readback, physical/key/identity digests and filesystem measurements.
- `generation_portable_{campaign,data,worker}.py`: registered Tigris CPU site,
  resource projection, exact discrete/tolerant floating comparison, source
  and environment separation. Reused through calls; PORT semantics unchanged.
- `dev_campaign.py`, `dev_submission.py`, `dev_worker.py`, `dev_restart.py`,
  `storage.py`, `measurement.py`, `provenance.py`: claims, immutable submission
  and output receipts, pre-existing execution-only source allowlist, native
  environment pinning, cap/free-space checks. Dispatcher/CLI/source-list and
  scheduler-worker recognition additions are execution-only.
- `scripts/queue_cms2jc2_portable_benchmark.sh` and
  `sbatch/run_cms2jc2_response_portable_cpu.sh`: heartbeat/stdin/exit and site
  activation patterns. New direct helper adapts them; original files unchanged.
- GEN/PORT response fixtures and `LocalMeasurement` tests: deterministic
  synthetic data, real Generator/spawn replay and mocked scheduler lifecycle.

New `generation_direct_{campaign,worker}.py` provide separate TG_* v1
contracts, reuse the existing 64-jet reference and read the same 10k TRAIN
ROOT population on Tigris. No new external donor, fitting, scientific kernel,
key-domain, unit/sign convention or Fresh_check dependency. Full 10k SPORC
parity is not claimed; only the existing 64-row reference is available.

## 2026-09-29: explicit dzfix partial-snapshot benchmark location

Repository-local donor commit `292e38896d7ae8bf9e973e9fd1c1efed8dc71548`:
`cms2jc2_response/generation_benchmark_{data,campaign}.py` supplies the existing
root guard and authenticated TRAIN membership/reader; the corresponding
benchmark tests supply synthetic ROOT, inventory/profile and campaign fixtures.
The guard now also accepts the exact documented `_partial_v1` directory name.
`docs/JETCLASS2_DZFIX_SALIENCE_MATCHING_HANDOFF.md` and
`docs/plans/JETCLASS2_DZFIX_FIXED_SLOT_CONCAT_K2_LADDER_PLAN.md` at that commit
record the actual SPORC snapshot and explain its noninterchangeable inventory.
No new external donor, particle transform, response fit, key-domain change or
Fresh_check dependency. GEN v1 membership and PORT replay semantics unchanged.

## 2026-09-29: separate Tigris portable JOINT benchmark

Repository-local committed donors remain
`455557980168a131e22b6990c902e1c52815a52f`: `cms2jc2_response/{dev_campaign,
dev_submission,dev_worker,storage,measurement,provenance,bridge,readers,
bdz_joint_maps,response}.py`. Science kernels, fitted maps and key domains are
unchanged. Exact scheduler identity is extended only for PORT_STAGE's new worker
and Tigris's site-assigned (not guessed SPORC) QoS; old contracts retain their
checks. New CLI, helper and source-snapshot registrations are execution-only.

The preceding, still-uncommitted 2026-09-29 GEN benchmark supplies
`generation_benchmark_{campaign,data,engine,worker}.py`: membership, frozen
bundle, bounded offline reader, once-per-process initialization, deterministic
NPZ encoding/readback and measurements. These are co-developed in this worktree,
NOT falsely attributed to the earlier committed donor. The engine additionally
admits 36/72/144 workers for new PORT task registries; old GEN tasks stay frozen.
The new `generation_portable_{campaign,data,worker}.py` adds source-pinned packet
export/import and explicitly separate cross-architecture tolerance contracts;
it never weakens the GEN donor-environment validator. The native CPU wrapper and
dry-default heartbeat helper adapt existing queue patterns. No external donor or
Fresh_check import is added. New tests reuse existing synthetic physical/model
fixtures and supplement them with relocatable packet and lifecycle checks.

## 2026-09-29: generation-only JOINT engineering benchmark

Repository-local donors at `455557980168a131e22b6990c902e1c52815a52f`:

- `cms2jc2_response/bdz_joint_campaign.py`, `bdz_joint_worker.py`,
  `bdz_joint_maps.py`, `response.py`: authenticate and reuse the selected JOINT
  runtime unchanged, one label-independent realization per jet. No numerical
  implementation, fitted parameters, or key domains are edited.
- `frozen_joint_campaign.py`, `frozen_joint_worker.py`, `dev_parallel.py`:
  separate gate/next-stage pattern, bounded spawn queue, native-thread limiting,
  conservative resource estimates and frozen-replay checks, adapted for a
  distinct TRAIN generation capability with no confirmation/test access.
- `bridge.py`, `readers.py`, `dev_data.py` plus
  `jetclass2_delphes/{inventory,splits,split_registry,contracts}.py`: authenticated
  metadata, stable row identities, offline-only physical bridge and file-byte
  verification. A new bounded-window reader is separate from old readers.
- `dev_campaign.py`, `dev_submission.py`, `dev_worker.py`, `storage.py`,
  `measurement.py`: immutable records, receipts, exact CPU submission journals,
  allocation claims and storage accounting. Changes to existing modules are
  additive execution routing/source snapshots/CLI only; no old scientific
  semantics or no-JC2 boundary is relaxed.
- `scripts/queue_cms2jc2_frozen_joint.sh`: adapted heartbeat/stdin/exit handling
  for the new dry-default queue helper, with an explicitly supplied reviewed
  plan hash. `frozen_joint_queue.py`: read-only resource-probe pattern, without
  interpreting a single-job estimate as multi-job completion evidence.
- Existing response-science, JOINT map, synthetic ROOT/split-registry and
  LocalMeasurement test fixtures are reused; mocked allocation evidence cannot
  authorize real production. New NPZ encoding/readback/digest implementation
  stays in `generation_benchmark_engine.py`; no external donor or Fresh_check
  runtime dependency is added.

## 2026-09-28: frozen JOINT collection-response confirmation

Repository-local donors at `7b57a32de74e4bcff4fa83e1ffcde67b5fcd0872`:

- `cms2jc2_response/bdz_joint_campaign.py`, `bdz_joint_worker.py`,
  `bdz_joint_maps.py`: authenticate the completed selected JOINT response and
  replay unchanged maps/random keys. The new `frozen_joint_worker.py` adapts
  ordered, bounded spawn execution to exactly JOINT and B_DZ. Acceptance compares
  both against the original five-candidate kernel; there is no new fitting API.
- `bounded_data.py`, `dev_data.py`, `readers.py`, `splits.py`: authenticated
  metadata, identities, streaming and access-lock pattern. New
  `frozen_joint_data.py` admits ALL outer-confirmation rows only through its
  separately claimed capability; the old fit-only reader is unchanged.
- `dev_diagnostics.py`, `bdz_audit_metrics.py`, `bdz_worker.py`: unchanged bounded
  additive histograms, tracking/PID diagnostics and canonical merges. New
  `frozen_joint_metrics.py` adds predeclared thresholds and paired file-bootstrap
  supported/rejected/inconclusive evidence, not a new development selection.
- `dev_campaign.py`, `dev_submission.py`, `storage.py`, `measurement.py`: existing
  source/receipt/submission/allocation/storage checks. Only execution dispatch,
  source extras and CLI are extended. `frozen_joint_queue.py` probes actual CPU
  shapes with test-only requests; it does not mutate existing jobs.
- `scripts/queue_cms2jc2_bdz_joint.sh`: heartbeat/stdin/exit-status pattern adapted
  into `queue_cms2jc2_frozen_joint.sh`, with separate gate/confirmation and no
  cancellation or automatic follow-up.
- JOINT/B_DZ/audit/bounded synthetic ROOT test ancestry reused. The
  `test_cms2jc2_frozen_c.py` fixture adds its synthetic authenticated `cms_root`
  field; no donor scientific implementation or real dataset is modified.

Intentional new semantics are under additive FROZEN_* v1 contracts and the
frozen JOINT confirmation plan. No external code, model, particle bank or
Fresh_check runtime import was copied. This finite collection-response study
does not replace the original production qualification or authorize JC2 transfer.

## 2026-09-27: bounded PID-safe / joint tracking repair

Repository-local donors at `b1d59811609020b40c65f9f4063128be1faf526b`:

- `cms2jc2_response/bdz_maps.py`: unchanged historical controls/quantile knots;
  `bdz_joint_maps.py` adds coordinate-local PID identity fallback and joint
  log-error / error-rank-conditional asinh-significance transport. No old map
  or model is rewritten; new semantics use new BDZ_JOINT_* contracts.
- `bdz_worker.py`, `bdz_audit_worker.py`: bounded ordered spawn orchestration
  adapted into `bdz_joint_worker.py`; original aggregation called directly for
  exact historical replay. New calibration reads residual jets only.
- `bdz_audit_metrics.py`, `bdz_metrics.py`, `dev_diagnostics.py`: unchanged
  diagnostic accumulators reused; new PID-aware score, guard and rendering
  live separately in `bdz_joint_metrics.py` and `bdz_joint_campaign.py`.
- `bdz_campaign.py`, `bdz_audit_campaign.py`, `bdz_audit_debug.py`: completed
  donor/receipt/source checks reused, including the completed debug audit.
- `scripts/queue_cms2jc2_bdz_audit_debug.sh`: phase heartbeat/stdin/exit-status
  pattern adapted into the new joint helper WITHOUT retirement or cancellation.
- `tests/test_cms2jc2_bdz_tuning.py`, `test_cms2jc2_bdz_audit.py` and their
  bounded/B/C synthetic ROOT fixture ancestry supply regression patterns.

Only the existing execution-only `dev_campaign.py`, `dev_worker.py` and CLI
are extended for source/stage/worker dispatch. No donor scientific file is
edited; no external code or Fresh_check runtime dependency is introduced.

## 2026-09-26: significance audit pending-job debug replacement

Repository-local donors at `a61e4ea1d09d0cca80251c7b8fc7e70e72577fb5`:

- `cms2jc2_response/c_topology_debug.py`: exact-ledger pending-only retirement,
  scheduler identity/account checks, admission-before-cancel, immutable evidence
  and fresh debug study adapted to `bdz_audit_debug.py`. Audit migration also
  refuses original claims and refreshes accounting after slow authentication.
- `bdz_audit_campaign.py`, `bdz_audit_worker.py`, `bdz_audit_metrics.py`:
  unchanged frozen protocol, acceptance, generation, replay and reporting
  primitives. `bdz_audit_debug_worker.py` is the external-acceptance adapter;
  it publishes new-stage results and references the original acceptance hash.
- `dev_campaign.py`, `dev_submission.py`, `dev_worker.py`, and the development
  CLI: additive contract/source/retirement/dispatch routing on the existing
  execution-only allowlist. No enlargement of that scientific-source exception.
- `scripts/queue_cms2jc2_bdz_audit.sh` and `queue_cms2jc2_bdz_tier3.sh`:
  dry/live source-pinned helper pattern, with phase/heartbeat feedback added in
  the new debug helper only. Neither donor helper is modified.
- Audit and C-debug tests plus their B_DZ/bounded/B/C synthetic ROOT ancestry:
  reused fixtures and lifecycle tests; no external donor or Fresh_check import.

## 2026-09-26: frozen B_DZ significance replay

Repository-local donors at `6b9a4bc1b3a9172d8b3ca63deb8c2c9e49c387d8`:

- `cms2jc2_response/bdz_worker.py`: unchanged historical accumulator invoked
  directly; bounded ordered spawn-worker orchestration adapted into
  `bdz_audit_worker.py`. No response/quantile-map refit is introduced.
- `bdz_campaign.py`, `bdz_tier3.py`, `bdz_maps.py`, `bdz_metrics.py`: completed
  donor/calibration lineage, numerical maps and historical score validation
  reused unchanged. New PID/value-error diagnostics live in separate modules.
- `dev_campaign.py`, `dev_worker.py`, `scripts/cms2jc2_response_dev.py`:
  additive stage/source/worker/CLI dispatch; immutable submission/claim and
  receipt machinery reused without changing old scientific semantics.
- `scripts/queue_cms2jc2_bdz_tier3.sh`: source/environment and dry/live wrapper
  pattern adapted to `queue_cms2jc2_bdz_audit.sh`, without any retirement or
  cancellation path.
- `tests/test_cms2jc2_bdz_tuning.py` and its bounded/B/C synthetic ROOT fixture
  ancestry: donor replay, immutable receipts and CPU lifecycle test patterns.

The new plan/contracts explicitly permit bounded hashed forensic examples,
not durable full particle banks. No external donor or Fresh_check import.

## 2026-09-25: B_DZ tier3 execution replacement

Repository-local donors at `1cdf6b890ce0b88de174948916593c106ddc8a06`:

- `src/hlt_classification/cms2jc2_response/c_topology_debug.py`: exact-ID,
  pending-only retirement, fresh-root reuse, immutable retirement evidence and
  scheduler race checks adapted into `bdz_tier3.py` (now also refuses claims).
- `bdz_campaign.py`, `bdz_worker.py`, `bdz_maps.py`, `bdz_metrics.py`: reused
  byte-for-byte for protocol, gate/map ancestry and all numerical evaluation;
  selection-display validation adapted in the new module, not the donor.
- `dev_campaign.py`, `dev_submission.py`, `dev_worker.py` and
  `scripts/cms2jc2_response_dev.py`: additive execution routing only.
- `tests/test_cms2jc2_c_topology_debug.py`, `test_cms2jc2_bdz_tuning.py` and
  their synthetic ROOT ancestors: fixture/lifecycle test patterns reused.

New plan/contracts and queue wrapper change only execution placement. No
external model or Fresh_check runtime dependency is introduced.

## 2026-09-25: B_DZ tracking-calibration screen

Repository-local donors at `b79e76be2ad7fb8884f6bb5128ae5b22e7aaa0a1`:

- `src/hlt_classification/cms2jc2_response/bounded_campaign.py`,
  `bounded_models.py`, `bounded_metrics.py`: authenticated original B_DZ
  selection/recipe and conditional TV semantics reused without modifying donors.
  New `bdz_campaign.py` handles only the known guard-list order normalization.
- `bounded_worker.py`, `dev_parallel.py`, `c_diagnostic_worker.py`: bounded
  spawn-process orchestration, identity digest and tight historical replay checks
  adapted/reused by `bdz_worker.py`.
- `dev_diagnostics.py`, `bridge.py`: physical/conditional histograms, plotting,
  immutable particles and validity semantics reused directly.
- `dev_campaign.py`, `dev_worker.py`, `scripts/cms2jc2_response_dev.py`: additive
  routing/source-plan registration; existing submission and claim machinery reused.
- `tests/test_cms2jc2_bounded.py`, `test_cms2jc2_b_tracking.py`,
  `test_cms2jc2_frozen_c.py`: real synthetic ROOT/receipt fixture ancestry reused
  in `tests/test_cms2jc2_bdz_tuning.py`.

The marginal tracking maps, tail-balanced metrics and BDZ_* contracts are new;
no Fresh_check runtime dependency or external model donor is introduced.

## 2026-09-25: bounded final B / B_DZ / BC comparison

Repository-local donors at `7d56425f3bb88b6ea36bbc9bd80e81fb166e5bb2`,
under `src/hlt_classification/cms2jc2_response/`:

- `b_tracking.py`, `c_diagnostic.py`: completed B/C fit/report/receipt import
  reused directly by `bounded_campaign.py`; scientific source equality remains.
- `b_tracking_worker.py`, `dev_parallel.py`: bounded spawned-process chunking,
  identity checking, histogram aggregation and progress patterns adapted in
  `bounded_worker.py`. New worker merges in bounded deterministic order.
- `dev_data.py:iter_sample`, `_sample`: reader/hashed membership pattern adapted
  into a separately claimed, 20k outer-confirmation capability in
  `bounded_data.py`; the original development reader is not broadened.
- `response.py`, `residuals.py`: unchanged runtime generator and joint residual
  backend reused by new portable recipe wrappers in `bounded_models.py`.
- `dev_diagnostics.py`, `c_diagnostic_worker.py`: unchanged fit-only histograms,
  plotting and historical replay helpers reused; new explicitly development-only
  selection/uncertainty rules live in `bounded_metrics.py`.
- `dev_campaign.py`, `dev_worker.py`, `scripts/cms2jc2_response_dev.py` gain
  additive source/stage/CLI dispatch; no existing scientific worker changes.
- `tests/test_cms2jc2_frozen_c.py` and `tests/test_cms2jc2_b_tracking.py` supply
  the tiny ROOT and actual B/C fitting fixtures for the new staged tests.

No Fresh_check imports, external donor code, model weights or raw-data copies.
No old execution is edited or adopted as a newly fitted model. New artifacts
use BOUNDED_* v1, with an explicit recipe and old-model parents. No new license
obligation. Independent confirmation is separately authorized, not production
qualification or automatic JC2 transfer.

## 2026-09-25: frozen C topology debug replacement

Repository-local donor: `0bed4cbb598f0476b1c00d08f43859c7e46120fa`.
New `cms2jc2_response/c_topology_debug.py` adapts `b_tracking_debug.py`'s
authenticated acceptance reuse, fresh five-job registration and exact pending
retirement lifecycle. C-specific acceptance fields, original six-view worker,
evaluation association recomputation and DEV_C_REUSE lineage are retained.
CLI/source/stage/submission/worker dispatch is additive; original B and C
scientific files remain byte-identical. New tests adapt the B migration fixture
to real tiny ROOT C acceptance and evaluation. No external code, dataset,
weights or output artifacts are migrated; no new license obligation.

Donor checkout-byte SHA-256 (Git may normalize line endings), under
`src/hlt_classification/cms2jc2_response/`:

| Reused donor | SHA-256 |
| --- | --- |
| `b_tracking_debug.py` | `aced8c3dab0430015e65d9bfd281f2de7001fbd6d57bfa0dcd977425072d45ad` |
| `c_topology.py` | `462e5b3bee90b9da0adc6be679e2f692b4f2be782ab63ce6e7c722ba87bd22d8` |
| `c_topology_worker.py` | `65103052a33ab245981de253f8e2cf9ab54b2912ddbfc596edd677984cff0d74` |

Ten focused migration tests pass, including exact worker replay/population,
old-file and unrelated-B preservation, corrupt donor rejection, admission before
cancellation, state races and accounting lag. Full evidence is in HANDOFF.

## 2026-09-25: frozen B debug replacement

Repository-local donor: `56ab24cba04b81456021f10e45caa5379e244256`.
New `cms2jc2_response/b_tracking_debug.py` composes the existing frozen-B
authentication, execution-only source whitelist and acknowledged scheduler
journals. It registers a distinct five-job debug stage, reuses completed B
acceptance, and adds explicitly authorized pending-only retirement evidence.
The CLI and existing campaign/submission/worker dispatch change only for this
new stage. Original B generation, metrics, worker and result reader are reused
unchanged. No external code, data, learned weights or original output artifact
was migrated; no license/attribution changes.

Donor checkout-byte SHA-256 (Git may normalize line endings), under
`src/hlt_classification/cms2jc2_response/`:

| Reused donor | SHA-256 |
| --- | --- |
| `b_tracking.py` | `46f6cf6a896c77f15bb9d9e708daa9462fd2452a2dd919947d281eb797ac7cf3` |
| `dev_restart.py` | `0798dbd1887d4fcf081b27cb03bca221443390ac127602482164826b7ed9d39b` |
| `b_tracking_worker.py` | `6a7deb61ccccdd4f5243a4f0ce576d215a5909b76571d509caa9d7729bfc7adb` |
| `b_tracking_results.py` | `3878d5cfa244c9b7410566378e1e67566c0e897aa9de4d180ea2ab416e052cdb` |

Regression evidence: `tests/test_cms2jc2_b_tracking_debug.py` exercises the
unchanged worker on tiny ROOT inputs, donor byte preservation, exact mocked
five-job submission, cancellation races and authenticated acceptance reuse.
Complete response regression results are recorded in HANDOFF.

## 2026-09-25: frozen B tracking audit

Repository-local donor: `5c18760f06fb5cb727668ce5482c2b6c35a73223`.
New `b_tracking.py` adapts the C frozen-reuse registration to B_L with a distinct
DEV_B_REUSE/v1 contract. `b_tracking_generation.py` copies the diagnostic
traversal and reuses the unchanged residual draw and physical emission codec;
only selected tracking residuals are zeroed. New event fields capture central
predictions, increments, limits and selected cells. `b_tracking_metrics.py`
extends original counters with validity-filtered tracking accounting.
`b_tracking_worker.py` adapts the bounded process/evaluation lifecycle from
`c_topology_worker.py`, removing association/oracle work. New result display
reuses authenticated products and historical replay checks. No old scientific
module, external code, dataset, trained weights or donor artifact is changed.

Donor checkout-byte SHA-256 (Git may normalize line endings):

| Donor under `src/hlt_classification/cms2jc2_response/` | SHA-256 |
| --- | --- |
| `c_diagnostic.py` | `7f02b36c4dbb2692c08a82b9d544f93c22cd4651bde97969c030f67a594fbb29` |
| `c_diagnostic_generation.py` | `59a8ca4069e6e03511421778bb724b077fdc1b9f38297632907352d0754c0050` |
| `c_diagnostic_metrics.py` | `aa467d44b02c977c70d000ddb08654e7290826cecafc0966b59838a662a46b3f` |
| `c_topology_worker.py` | `65103052a33ab245981de253f8e2cf9ab54b2912ddbfc596edd677984cff0d74` |

Fourteen focused tests pass, including actual B fitting, exact FULL replay,
unchanged kinematics/state, selective masks, validity/clipping accounting,
process parity and tiny ROOT -> immutable reports/plots. No new licensing
impact. Combined response regression: 181 passed, one unchanged long 33-fit
production integration deselected. The first registered real SPORC acceptance
is still required.

## 2026-09-25: observed-topology C diagnostic

Repository-local donor: `36a01900a73447680cdc3e28bb0b9170c265a544`.
New `cms2jc2_response/c_topology_generation.py` copies the frozen continuous
emission equations from `c_diagnostic_generation.DiagnosticGenerator.emit`;
learned topology/state draws are intentionally replaced with observed-association
state for a privileged diagnostic. `topology.calibration_records` and its codec
are reused unchanged to authenticate the registry/round trip. The new worker
reuses frozen-C inputs, metrics/accounting, plotting and CPU bounded scheduling.
Only execution dispatch/source registration and the CLI are extended. No donor
scientific kernel, raw dataset, external code, model or weights are copied.

Donor byte SHA-256 values (local checkout bytes; Git may normalize line endings):

| Donor under `src/hlt_classification/cms2jc2_response/` | SHA-256 |
| --- | --- |
| `c_diagnostic_generation.py` | `59a8ca4069e6e03511421778bb724b077fdc1b9f38297632907352d0754c0050` |
| `topology.py` | `f7c9f19a74830e7321068ccd6dad8af1c948c3fa0832a48c5a4924e483fa578a` |
| `c_diagnostic_worker.py` | `2e3e70722a42b21a7e302fd01efe74e673e9b7ee159577497202de2d308dde00` |
| `dev_parallel.py` | `7185b56220a78491c6339924ff22ac50e21dadd5f66390407a19d75de9c9c099` |
| `c_accounting_audit.py` | `891b3b7b038ed60a59e741cd07f61530ac086708299854391966b141dff23f0f` |

Parity tests cover frozen continuous generation given identical free states,
no observed-continuous-value leakage into fixed prediction, canonical codec
closure, real process execution, ordinary full-population historical replay,
and tiny ROOT data through immutable reports. The new observed-topology plan
and contract declare all intentional differences; no new licensing impact.
Local validation: 13 new focused tests passed; combined response regression
167 passed, with the unrelated long 33-fit production integration deselected.
Real SPORC acceptance remains a separately queued requirement.

## 2026-09-25: read-only frozen-C momentum accounting

Repository-local donor: `0c3682bbb9a4c57b0f632574f878c43e863faa5c`.
New `cms2jc2_response/c_accounting_audit.py` consumes the existing
`dev_diagnostics.Histograms` payload and `c_diagnostic_metrics.Counters`
schemas, reuses development receipt/registration authentication and the
historical floating-moment replay comparison. No generator traversal or
scientific kernel is copied or modified. The dev CLI adds `c-audit` only;
existing frozen-C integration fixtures exercise read-only accounting.
New pure accounting tests cover ratios, empty jets, unknown charge bounds,
split/fallback denominators, inventory-not-occupancy and corrupt sums.
Additive stdout-only DEV_C_ACCOUNTING_AUDIT/v1 does not modify the donor reports.
No external data, models, weights or new training artifacts are copied.

## 2026-09-24: frozen CMS2JC2 C diagnostic

Repository-local donor: `5f8fecc8013a2439fb725198487dc5899246ab94`.
`cms2jc2_response/c_diagnostic_generation.py` copies the traversal from
`response.Generator` and reuses `residuals.sample`, RNG, features and topology.
Production `response.py` and `residuals.py` are unchanged. FULL replay is checked
against the production generator on every evaluated jet-replica. Other variants
are explicitly diagnostic interventions, not version-compatible fitted models.
`c_diagnostic.py` reuses development campaign/restart authentication patterns;
`c_diagnostic_worker.py` reuses frozen development readers, histograms, plotting,
publication and receipts. `c_diagnostic_metrics.py` and `c_diagnostic_results.py`
add bounded counters/display. The existing dev CLI, worker and campaign dispatch
are extended without changing old task definitions. Existing submission and
Slurm worker files are reused unchanged. Tests reuse science particles and tiny
ROOT fixtures. No external donor, weights or raw files are copied into Git.
DEV_C_DIAGNOSTIC/REUSE/ACCEPTANCE/SHARD/REPORT/SUMMARY v1 contracts bind the
fresh tier3 study; completed original de9890b2 C_L remains a read-only donor.

## 2026-09-24: CMS2JC2 CPU36 tier3 execution profile

Repository-local donor: `0560c6db42a43fc5ddca22ef7bbf5407b64fa933`.
Reuse `cms2jc2_response/dev_parallel.py`, `dev_restart.py`, `dev_campaign.py`,
`dev_worker.py` and `scripts/cms2jc2_response_dev.py` from the CPU64 implementation.
Generalize only registration/dispatch to a separate 36-CPU tier3 profile;
retain original CPU64 artifact definitions and the legacy 16-CPU path.
Tests extend `test_cms2jc2_response_cpu64.py`, including its original development
and science fixtures. No external donor, raw data, models or weights copied.
New DEV_STAGE36/DEV_CPU36_EXECUTION/DEV_CPU36_REUSE v1 contracts bind fresh
execution. Exact read-only confirmation reuse remains from the original
`de9890b22c7506c58172f20a4ab71e1b85ba5a18` development study; no scientific
matching, sampling, response or diagnostic definitions changed.

## 2026-09-24: CMS2JC2 CPU64 chunked preparation

Repository-local implementation baseline:
`a43c6ef7a00a3d0cc470a4f2be66ea1f5ce3f6f9`. Original development donor pin:
`de9890b22c7506c58172f20a4ab71e1b85ba5a18`.
New `cms2jc2_response/dev_parallel.py` reuses `dev_data.sample_stream`,
`response.collect`, `records.Reservoir`, and the process-pool pattern from
`dev_worker.py`; `combine_records` is extracted from that worker without
changing record semantics. New `dev_restart.py` reuses `dev_campaign.py`
source/import/receipt validation and its existing task/submission machinery.
No external donor code, models, or weights are copied. The matcher, native
readers, features, sampling rules, models and diagnostic definitions remain
unchanged; all non-execution source hashes are checked on confirmation reuse.

Tests reuse the development population fixture and science particle fixtures,
plus the existing real-ROOT end-to-end fit/evaluation test. Additive
DEV_STAGE64, DEV_CPU64_EXECUTION and DEV_CPU64_REUSE v1 contracts distinguish
the opt-in 64-CPU comparison; legacy DEV_STAGE v1 remains unchanged. The old
SEARCH confirmation and preparation products are read-only metadata donors,
not imported fit results or authority to modify their Slurm jobs.

## 2026-09-23: separate K2 100k/50k short-ladder debug pilot

Repository-local baseline/donor: `c891da0d45dd3251dea9ea72df975bb96bae3570`.
Reuse K2 campaign/source/submission/execution/runtime/parity and thin queue
interfaces. New `concat_k2_pilot.py` owns the opt-in recipe; new
`concat_k2_pilot_data.py` adapts `concat_k2_data.py` RAM construction and
`split_registry.py` quota/mask containers to authenticated parent-row subsets.
Matching/data/view producer files themselves are unchanged. Original compact
matching donor remains `1f9306504dfd040c9c22e0a89829d277d1ff2194`.
`salience_learned_training.py` gains an exact opt-in pilot schedule with no
changes to default callers. Fixtures reuse `test_jetclass2_delphes.py`,
`test_jetclass2_concat_k2.py`, `test_jetclass2_concat_k2_source.py`,
`test_jetclass2_concat_k2_memory.py`, `test_jetclass2_concat_k2_reuse.py`,
`test_jetclass2_dzfix_fusion_chain.py`.
No external donor code or checkpoints copied. Pilot launch/campaign v8,
training report v3, GPU acceptance v7, population v1 isolate the smaller
study. Existing v7 full-size registration and other campaigns remain unchanged.

## 2026-09-23: CMS2JC2 isolated CPU development study

Repository-local baseline: `c891da0d45dd3251dea9ea72df975bb96bae3570`.
No external donor code copied. New `cms2jc2_response/dev_{data,campaign,
submission,worker,diagnostics}.py` reuse the namespace's `audit.py`, `splits.py`,
`readers.py`, `bridge.py`, `association.py`, `features.py`, `records.py`,
`parallel.py`, `families.py`, `response.py`, `metrics.py`, `evaluation.py`,
`measurement.py`, `provenance.py`, `storage.py`, `contracts.py`, `assumptions.py`
and underlying keyed RNG, topology and residual modules. Submission journaling
adapts `submission.py`; the CPU worker adapts
`sbatch/run_cms2jc2_response_cpu.sh`. Tiny ROOT test fixtures are reused from
`tests/test_cms2jc2_response_end_to_end.py` and particle fixtures from
`tests/test_cms2jc2_response_science.py`.

Read-only RC metadata donor is the completed preparation at
`f966dd804ed9ca8ca93c9c3227d1a7f036ef4434`, not its models or old JC2 inputs.
CMS audit/split/schema/label semantic hashes must still match before reuse.
New DEV_* v1 contracts isolate smaller exploratory samples/diagnostics from
production gates. No classifier, matching, response-family or old queue code
is changed. Existing jobs remain source-pinned to their original worktrees.

## 2026-09-23: K2 tier3 long-walltime execution policy

Repository-local baseline: `20358f5568a2a5d42c8849ca99439bc435122f26`.
Execution evidence: batch-128 source `11ff05dff72224ff365e3873c2731d79a802780d`,
SPORC preflight 21770762 (memory probes passed; 72.91h projection exceeded
23h guard). Reuse `concat_k2_campaign.py`, `concat_k2_execution.py`,
`concat_k2_runtime.py`, `concat_k2_submit.py`, and the existing CLI/queue helper;
no external source is copied. Adapt existing K2 registration/source/reuse tests
and add `tests/test_jetclass2_concat_k2_walltime.py` for 96h-tier3 admission,
95h acceptance, unchanged short-job portability and exact TimeLimit rejection.

Original preparation donor remains
`1f9306504dfd040c9c22e0a89829d277d1ff2194`; the importer additionally recognizes
fresh v7 donors, without accepting nested imports or historical GPU gates.
Matching/data/view producers and their source fingerprints are not changed.
Launch/campaign v7, GPU acceptance v6 and execution policy v2 distinguish the
new resource envelope. Model/training code, batch 128, scientific seeds and
other campaigns are unchanged. New real SPORC acceptance remains required.

## 2026-09-23: dzfix fusion tier3 long-fit registration

Repository-local baseline/donor: `20358f5568a2a5d42c8849ca99439bc435122f26`,
the 21770778 debug preflight. Adapted `dzfix_fusion_chain.py`,
`dzfix_fusion_runtime.py`, `dzfix_fusion_submit.py`, the fusion CLI/queue/worker
and existing fusion contract tests. Added tier3 scheduler/runtime regression
tests. Reused `execution.execution_site("sporc_a100")` without modifying shared
site code. No model, training kernel, external code or weights copied.
Launch/campaign v6 and acceptance v4 separate the tier3 72-hour registration
from debug; source-import v3 and all matching producer hashes are unchanged.
K2 and salience campaigns are not edited. See HANDOFF for test/remote evidence.

## 2026-09-23: dzfix fusion strict parity and layout-preserving offload

Failed execution baseline: `b35fbda64d2d823a9eb9c5592017074db58d6ac8`
(SPORC preflight 21765886). Repository-local donor:
`91be01940f814e1ea7a8a460b9694f0148f0d024`, specifically
`src/hlt_classification/jetclass2_delphes/concat_k2_model.py` (scoped strict
backend and physical-span saved-tensor storage), `concat_k2_parity.py`
(bounded early real-train diagnostic pattern), and
`tests/test_jetclass2_concat_k2_parity.py` (failure-before-cache tests).
Adapted into `dzfix_fusion_model.py`, new `dzfix_fusion_parity.py`, and fusion
tests. The sampler uses salience `load_assignments` / `build_view`, not K2
maps, views or batch policy. No K2/shared preparation files are changed.
No external source or scientific checkpoints were copied.

Execution contracts: fusion launch/campaign v5, acceptance v3, early parity
v1; matching source-import stays v3. Production remains batch 256. Local
installed Weaver 0.5.3 is an isolated test dependency, not vendored code.

## 2026-09-23: explicitly registered K2 physical batch 128

Repository-local baseline/donor: `91be01940f814e1ea7a8a460b9694f0148f0d024`,
the user-reported 21768860 preflight. Adapt `concat_k2_campaign.py`,
`concat_k2_model.py` probe policy, `concat_k2_runtime.py` and the K2 queue
helper; add opt-in batch overrides to `salience_learned_training.py` while
preserving legacy defaults/report layout. No training loop, Weaver source,
model weight or external code is copied. K2-only versions distinguish the
new recipe/acceptance; preparation import additionally admits fresh v6 donors.

Original matching donor remains `1f9306504dfd040c9c22e0a89829d277d1ff2194`.
Matching/data/view producer files and hashes are untouched by this block.
Tests adapt existing campaign/memory/reuse fixtures and add 300-row physical
CE/KD batch-boundary, teacher-join, final-partial-batch, legacy-default and
batch-evidence rejection tests in `test_jetclass2_concat_k2_batch128.py`.
Real SPORC full batch-128 acceptance remains required before science release.

## 2026-09-22: K2 deterministic parity and layout-preserving saved tensors

Repository-local donor/baseline: `49516092a646cfe21f7bb1377b07a142161eb3d2`,
the failed 21767292 execution. Modified `concat_k2_model.py` storage hooks and
parity harness; `concat_k2_runtime.py` composes a new early gate from existing
`acceptance.installed_parity`, `concat_k2_data.load_assignment`, `DatasetReader`,
`build_view`, `build_inputs` and `RamCache`. New `concat_k2_parity.py` reuses
those APIs without copying or editing their matching/data producer files.
The K2 worker sets preflight-only cuBLAS configuration; registration binds new
execution policy versions. Native model, optimizer, matching and science
recipe are unchanged. Original completed-map donor remains
`1f9306504dfd040c9c22e0a89829d277d1ff2194`, subject to the full reuse checks.

Tests reuse repository synthetic ROOT, completed-donor and model fixtures;
new tests cover stride/value preservation, deterministic-state restoration,
bounded authentic training-row sampling and early failure. Weaver 0.5.3 was
installed into an isolated scratch directory for additional local validation;
no Weaver/PyTorch source was vendored, no external weights imported, and no
third-party license/attribution changes were introduced. Results and the
remaining SPORC acceptance boundary are recorded in `docs/HANDOFF.md`.

## 2026-10-05: exact final-direct node-failure recovery adapter

Scientific donor: `fd1c05786287c57b080b2d229dd64664f830a571`.
New adapter: `src/hlt_classification/jetclass2_delphes/dzfix_fusion_recovery.py`,
with thin CLI/worker and queue helper. No scientific donor files are copied or
edited. The new CLI imports these original checkout modules at execution:

- `jetclass2_delphes/dzfix_fusion_chain.py`, `dzfix_fusion_runtime.py`, and
  `dzfix_fusion_submit.py`: original campaign, gate, training/publication and
  exact allocation checks;
- `jetclass2_delphes/contracts.py`, `jetclass2_delphes/submission.py`,
  `data/cache_contracts.py`, `scouting/hcwdl_exact_dag_submission.py`, and
  `scouting/hcwdl_recovery.py`: original hashing, immutable publication and
  guarded exact-DAG submission;
- `sbatch/jetclass2_delphes_common.sh`: original SPORC environment setup;
- `tests/test_jetclass2_dzfix_fusion_chain.py`: reusable local cache fixtures
  for the recovery tests (not a production import).

The new versioned recovery artifacts govern only a three-task restart-zero
execution exception; scientific contracts and original pinned source remain
unchanged. See `docs/contracts/JETCLASS2_DZFIX_FUSION_RECOVERY.md`.

## 2026-09-22: authenticated K2 compact-assignment reuse

Baseline: `df29abcce078b2be80bd465b22c28066ec32d8f6`. Original reusable K2
producer: `1f9306504dfd040c9c22e0a89829d277d1ff2194` (only after its completed
foundation is authenticated). New repository-local
`jetclass2_delphes/concat_k2_preparation_import.py` composes the existing
`concat_k2_data.load_assignment/foundation_lock/producer`,
`concat_k2_runtime.completed`, and `concat_k2_source.validate_import`
validators. Those matching/data producer files and their byte identities are
unchanged by this block. No external donor code or trained weights are copied.
Campaign/source/runtime and thin CLI/queue helper wire the optional import;
launch/campaign become v4, PREPARATION_IMPORT is v1, GPU acceptance remains v3.

## 2026-09-22: K2 pair saved-tensor storage and 128/256 probes

Repository-local donor/baseline: `b35fbda64d2d823a9eb9c5592017074db58d6ac8`.
The failed original K2 execution was `1f9306504dfd040c9c22e0a89829d277d1ff2194`.
`concat_k2_model.py` adapts the saved-tensor CPU-storage pattern from committed
`jetclass2_delphes/dzfix_fusion_model.py` to the single native K2 encoder; it
does not import or modify that fusion adapter. Native base is `model.py`;
training/optimizer are reused from `salience_learned_training.py` unchanged.
`concat_k2_runtime.py` adds ordered probes and evidence validation;
`concat_k2_campaign.py` registers the storage/probe policies and v3 execution
artifacts. No external code or scientific model weights are copied.

`tests/test_jetclass2_concat_k2_memory.py` reuses committed `PairBN`/physical
input fixtures from `test_jetclass2_dzfix_fusion_offload.py`, and existing
`make_cache`/`fake_native` fixtures ultimately based on
`test_hcwdl_offline_hlt_fusion.py::_FakeWeaver`. They test a K2-only adapter,
not a fusion architecture. PyTorch public `save_on_cpu`/`saved_tensors_hooks`
are composed without vendoring. Existing partition-portability changes are
retained; no other campaign runtime is changed by this repair.

## 2026-09-22: dzfix fusion pair saved-tensor memory repair

Repository-local baseline/donor commit:
`6172f5be459f0a0f7b6c3ed99ee3c6f8ac7e542c`; the failed remote execution was
`2f0afe5c438cbab5d66e047c4d86534aab02032b` (preflight 21757056).
Modified `jetclass2_delphes/dzfix_fusion_{model,chain,runtime}.py` and added a
native-default pair-embedding hook to `salience_learned_model.py` at that
baseline. `tests/test_jetclass2_dzfix_fusion_offload.py` reuses the repository's
existing fusion-chain RAM fixtures and `_FakeWeaver` from
`tests/test_hcwdl_offline_hlt_fusion.py`, adding a trainable full-pair BN test
double. No external source is copied. PyTorch's public `save_on_cpu` and
`saved_tensors_hooks` APIs are composed, not vendored.

Scope: storage of tensors saved inside the three pair embeddings only; full
pair populations, BN computation, dtype, batch, loss, model keys and data stay
unchanged. Tests cover three optimizer updates in CPU/FP32 CUDA/BF16 CUDA,
actual pinned copies/retrieval, BN tampering, checkpoint transparency,
inference bypass and hook cleanup. LAUNCH_SPEC/CAMPAIGN_SPEC become v4,
ACCEPTANCE v2; SOURCE_IMPORT remains v3 so completed matching is reused via
the same authenticated read-only boundary. Real Weaver/A100 acceptance is
still required; local test doubles cannot certify it.

## 2026-09-22: K2 tier3/debug execution portability

Repository-local donor commit: `1f9306504dfd040c9c22e0a89829d277d1ff2194`.
Adapted existing `jetclass2_delphes/concat_k2_{campaign,source,submit,runtime}.py`,
`scripts/jetclass2_concat_k2.py`, `scripts/queue_jetclass2_concat_k2.sh`,
`sbatch/run_jetclass2_concat_k2.sh` and their two test files. New K2-scoped
`concat_k2_execution.py` composes the existing `execution.execution_site`
profiles and strict `allocation`/`gpu_identity` validators without altering
those shared files or other campaigns. No external source or scientific
kernel is copied. New policy defaults to tier3 and accepts only partition-only
pending-job moves to/from debug with identical resources/hardware/software.
Launch/campaign/acceptance are versioned v2; view/matcher/seed identities are
unchanged. Tests add actual/requested-site and cross-partition gate coverage.

## 2026-09-21: isolated dzfix K2 concatenation ladder

Repository-local donor commit: `2f0afe5c438cbab5d66e047c4d86534aab02032b`.
No external repository, data, weights, generated campaign or Fresh_check
runtime import is copied. The K2 matcher/view semantics and graph are new.
Local donor-byte SHA-256 values below precede the LF normalization used while
adapting new files; the Git commit identifies the canonical source revision.

| Donor file | Local donor SHA-256 | New adapted surface |
|---|---|---|
| `jetclass2_delphes/dzfix_fusion_source.py` | `d3a888c251d45ffc053178c4d2124437b8ebfa63209ad88ff3ee6313a665ca94` | `concat_k2_source.py` |
| `jetclass2_delphes/dzfix_fusion_submit.py` | `da30bff454ce8914e3436d05d7b31001566e11a7c6973988747d11d2d3fc2fcf` | `concat_k2_submit.py` |
| `jetclass2_delphes/dzfix_fusion_runtime.py` | `5d9ccefc7faff6132e49e2b30689ad263a9e2cc9476419ef0584c87d0f4c8603` | `concat_k2_runtime.py` |
| `jetclass2_delphes/dzfix_fusion_data.py` | `dc00aad07a7a2d6a2b309caa9f13ce6c9bafa6bc1a5713fc172acb4921cca94c` | bounded-cache/partition patterns in `concat_k2_data.py` |
| `scripts/jetclass2_dzfix_fusion_chain.py` | `6966f734f2bd9f935680a9962d221eea5cf38a7dec9a46cea064a4b63809e9b2` | `scripts/jetclass2_concat_k2.py` |
| `scripts/queue_jetclass2_dzfix_fusion_chain.sh` | `92add0f6e4f241d38b93d73a8befd8592b32f34bdd7d19c5944cf23c33989107` | `scripts/queue_jetclass2_concat_k2.sh` |
| `sbatch/run_jetclass2_dzfix_fusion_chain.sh` | `c0a4a42a3b0342393d32cb43ac7d4913046c7a8ad5f1361ae135e47d5f8a0c8b` | `sbatch/run_jetclass2_concat_k2.sh` |
| `tests/test_jetclass2_dzfix_fusion_chain.py` | `718920b0fc25adcb1fdade60b5e9bf027465045d0d9f132eae15f643f3381bb1` | adapted source tests and reused synthetic fixtures |

Package paths in this table are under `src/hlt_classification/`. Retained:
read-only screen/ledger authentication, immutable task receipts, staged exact
submission, validation-role partitioning, ordinary train-bank KD and selected
checkpoint reporting. Changed: no dual-view model or weight/map reuse; new
capacity-two preparation, fixed 3*N_HLT support, K2-only schemas/namespaces,
ten-fit D-only graph and expanded-input GPU acceptance. All new jobs use debug.

Composition (not file migration) additionally uses the same revision's
`salience_views.pairing_matrices`, `scouting/hcwdl_fullcard_salience_matcher`
exact solver/reference, its versioned salience contracts, and the common
`reader`, `inputs`, `cache`, `model`, `banks`, `reporting`, `execution`,
`salience_learned_training`, `salience_learned_graph`, `salience_learned_data`,
`acceptance`, `submission`, provenance/cache-contract and exact-DAG helpers.
The full-native salience is frozen before new retention/duplication; the old
one-to-one artifact is never reused. Runtime source fingerprints include
actual preparation bytes, so unrelated dirty worktree edits are not falsely
claimed as donor-commit evidence or silently accepted as old artifacts.

Tests: `test_jetclass2_concat_k2.py` (exhaustive matching, typed views, real
synthetic ROOT preparation, CPU test-double DAG, deployment, gates) and
`test_jetclass2_concat_k2_source.py` (screen imports, dry/live plan isolation,
receipts, bad-parent failure injection). No new third-party dependency or
license obligation. Local CPU tests do not certify installed Weaver/SPORC.

## 2026-09-21: isolated JetClass2 dz-fix fusion-to-fusion debug chain

Internal code donor: `f1753bd1f26dd2bffe36d21fc937aeb663cab00d`.
New `jetclass2_delphes/dzfix_fusion_{chain,source,data,model,runtime,submit}.py`
uses the donor's `salience_learned_{graph,training,model,data,cache}.py`,
`cache.py`, `banks.py`, `execution.py`, `reporting.py`, `model.py`,
`salience_foundation.py`, `salience_screen.py`, `salience_production.py`,
`dzfix_salience_continuation.py`, `submission.py` and the shared exact-DAG
submission/authorization/cache-contract helpers. No shared graph or loss is
redefined. A default-no-op injection-bias hook in `salience_learned_model.py`
lets the new adapter apply `cms_salience_learned/model.py`'s temporary-mask
optimization while preserving historical adapters and full pair-BN semantics.
The native pure-OFFLINE adapter uses `cache._prepare_file` at its exact native
offline endpoint; persistent U000 remains a different view.

The scientific graph specializes the prior local CMS fusion-chain plan; that
prior uncommitted CMS extension is not falsely attributed to the donor commit.
No CMS model, acceptance, metric, split or matching artifact is imported.
Data/preparation selection donor: the source-pinned replacement v2 debug
screen at `0d25a4a53aafb1348c8279d86bac7dbac82c8841`. Its unchanged compact
foundations retain their original producer fingerprints. The updated consumer
reuses this commit's `salience_screen.py` v2 validator/command plan/task reports
and `salience_production.py::_screen_artifacts`; winner and exact foundation
paths/hashes are resolved from the screen, never named in advance. The canceled
46f10c7 continuation/production-preview route is no longer a parent.
The consumer orchestration baseline is `cde3ed8` (the committed initial
dz-fix fusion chain); only its source adapter, submission boundary, CLI/helper,
input-contract versions and tests are changed. Scientific kernels are not.

Thin CLI/worker/queue helper derive environment handling from
`sbatch/jetclass2_delphes_common.sh` and staged exact submission from
`salience_learned_autolaunch.py`, but all new jobs are explicitly debug.
Tests reuse only the synthetic `_FakeWeaver` from
`tests/test_hcwdl_offline_hlt_fusion.py` and particle fixtures from
`tests/test_jetclass2_delphes.py` (same code donor).
Capacity-fix baseline: `e3b02d6f6f0f3b6433230603a2438767b69e25fd`.
The consumer adapter now verifies the same inventory-maximum/round-up-to-16
capacity calculation as that commit's `jetclass2_delphes/foundation.py` and
`salience_foundation.py`, and uses `inputs.py::input_contract` unchanged.
Current dzfix capacity is 320, not the stale 240 inherited in the handoff.
Those shared producer files and all assignment payloads remain unmodified.
Tests explicitly preserve >240-particle native inputs, use 320 in pending and
completed source imports, and exercise capacity-driven cache/stress behavior.
Family: `JETCLASS2_DELPHES_DZFIX_FUSION_CHAIN_*`; LAUNCH_SPEC, SOURCE_IMPORT,
CAMPAIGN_SPEC advance to v3 for the inventory-derived no-truncation policy
(v2 introduced direct-screen provenance); other kinds stay v1.
No external source code copied.

## 2026-09-20: isolated CMS adjacent fusion-to-fusion KD chain

Internal implementation donor `f1753bd1f26dd2bffe36d21fc937aeb663cab00d`:
`src/hlt_classification/cms_salience_learned/{contracts,campaign,production,
preparation_import,shared_import,preflight_reuse,coarse_submission}.py` and
`scripts/cms_salience_learned.py` are extended for a new fusion-chain consumer.
New `fusion_chain.py` derives the shared-reference/first-acquisition nodes
from the donor's coarse graph, authenticates completed acquisition outputs,
and provides the new graph/import/report orchestration. Scientific model,
training, cache, publication/reduction and worker kernels remain unchanged.
New `scripts/queue_cms_fusion_chain.sh` derives its environment/staged workflow
from `scripts/queue_cms_direct_fusion.sh`, without cancellation. New tests
reuse `test_cms_direct_fusion.py`, `test_cms_salience_learned.py` and
`test_hcwdl_offline_hlt_fusion.py` fixtures from the same donor.
No external source code was copied.

Scientific execution donor remains accepted dense
`7bb171382b7206013bc5d9308a4c22b2929bc7f4` (genuine SPORC preflight 21720795),
with original preparation `f2e8a374f522a39c7f3a6331f0ec77ae12cabaea`.
First acquisition data donor is the explicit completed coarse-v5 campaign
at `7cf690301d66b1d02d98b450d20aababf296429a`; its spec, live ledger, exact
model/teacher identities, receipts and artifact bytes are verified at reuse.
CAMPAIGN_SPEC/v7, GRAPH/v4, PREPARATION_IMPORT/v4, SHARED_SOURCE/v3,
ACCEPTANCE_IMPORT/v3 and ACCEPTANCE_REUSE/v3 bind this separate consumer.
ACQUISITION_SOURCE/v1 and FUSION_CHAIN_AGGREGATE/v1 add explicit import and
two-endpoint reporting semantics. Old campaign graphs/artifacts stay unchanged.

## 2026-09-19: isolated CMS direct fusion/withdrawal comparison

Internal implementation donor `7cf690301d66b1d02d98b450d20aababf296429a`:
native CMS `contracts.py`, `campaign.py`, `preparation_import.py`,
`shared_import.py`, `preflight_reuse.py`, `coarse_submission.py` and
`scripts/cms_salience_learned.py` are extended for a separate direct consumer.
New `scripts/queue_cms_direct_fusion.sh` uses the environment setup from
`scripts/switch_cms_salience_coarse.sh` at that donor, but not its cancellation
workflow. New regression tests reuse synthetic fixtures from the native CMS,
coarse and fusion tests at the same donor. No external code is copied.

Scientific execution donor remains accepted dense commit
`7bb171382b7206013bc5d9308a4c22b2929bc7f4` (SPORC GPU job 21720795), with
original preparation `f2e8a374f522a39c7f3a6331f0ec77ae12cabaea`.
Native model/training/data/production kernels and the worker shell are reused
unchanged, not migrated to a new runtime. CAMPAIGN_SPEC/v6 and GRAPH/v3 declare
the one-arrow U000/D000 study. New preparation/shared/acceptance import
versions bind this consumer without changing coarse replacement artifacts.

## 2026-09-19: CMS reuse AST encoding compatibility fix

Internal baseline `4f862c11045943f1237d1cef166d32a85c341ed3`;
original preparation producer `f2e8a374f522a39c7f3a6331f0ec77ae12cabaea`,
accepted runtime donor `7bb171382b7206013bc5d9308a4c22b2929bc7f4` and coarse
registry donor `48ab8609ee87ba72ab9868dfc951a36a1c9d851e`.
Extended only the exact reviewed function-hash pairs in native
`preparation_import.py` and `preflight_reuse.py` to cover Python 3.10's
empty-list-emitting AST dump in addition to Python 3.13's original encoding.
Both are derived from the same immutable donor functions; no external source
copied, no stored fingerprints rewritten, no scientific contract version changed.
Regression changes are in the existing coarse and preflight-reuse test files.

## 2026-09-19: explicit dense-to-coarse execution evidence reuse

Internal baseline `48ab8609ee87ba72ab9868dfc951a36a1c9d851e`; execution donor
`7bb171382b7206013bc5d9308a4c22b2929bc7f4` (accepted SPORC preflight 21720795).
Added `cms_salience_learned/preflight_reuse.py` and focused tests; extended
native contracts/campaign/dispatch and the two operator CLIs. No external
code was migrated and no training/model/cache kernel changed.
New CAMPAIGN_SPEC/v5 plus ACCEPTANCE_IMPORT/v1 and ACCEPTANCE_REUSE/v1 record
explicit user-authorized reuse of the old measured envelope. Source evidence
is never rewritten as a new measurement. Git identities cover the same native
reference kernels, shared models/data, worker shell, allocation adapter and
preflight probes; the sole allowed driver AST migration is its schema-version
predicate, previously introduced in coarse commit 48ab8609. V4 fresh-gate and
all older scientific/operational meanings remain unchanged.

## 2026-09-19: native CMS coarse graph and authenticated shared references

Internal baseline: `7bb171382b7206013bc5d9308a4c22b2929bc7f4`, accepted dense
SPORC preflight 21720795. Extended native `cms_salience_learned/contracts.py`,
`campaign.py`, `production.py`, `preparation_import.py`, the native CLI and
tests. Added `shared_import.py`, `coarse_submission.py`, coarse tests and the
staged operator shell helper. No external code was copied. The exact journal
pattern and ledger/event schemas reuse internal
`scouting/hcwdl_exact_dag_submission.py` and `scouting/hcwdl_recovery.py` at the
same donor commit; those shared modules are not changed.

New native CAMPAIGN_SPEC/v4, GRAPH/v2, PREPARATION_IMPORT/v2 and
SHARED_SOURCE/v1 distinguish the coarse graph and reviewed imports. Retaining
the original family seed namespace is intentional. Preparation Git objects
from original producer `f2e8a374f522a39c7f3a6331f0ec77ae12cabaea` are confirmed
identical to the accepted dense source; only the closed addition of coarse
coordinate names gets an explicit AST migration. Reference training/model/
cache modules and scientific worker/seed/schedule ASTs remain unchanged.
Neither imported source files nor the previous GPU gate are rewritten.

## 2026-09-18: native CMS-LFH compact shared cross-attention bias

Internal baseline: `77c9d2b1fe17a9fb321f1f85ebf9cd803c0f5ec1`, source
of SPORC debug preflight 21720511. Extended native
`cms_salience_learned/model.py` and shared
`models/hcwdl_offline_hlt_fusion_transformer.py`; added
`tests/test_cms_fusion_temporary_memory.py`. No external donor was copied.
The shared base exposes a legacy-preserving bias-preparation hook; only the
CMS subclass opts into merging padding once and sharing its compact rectangle
across the four injections. Weaver's full pair construction is not altered.
Other model adapters retain the old path. Scientific/state-dictionary and
execution contracts are unchanged; exact new source still requires its own
installed-Weaver checks and genuine A100 acceptance. Preparation-code Git
objects, graph, matching, loss, batch, schedules and seeds are untouched by
this patch, permitting authenticated read-only original preparation reuse.

## 2026-09-18: native CMS-LFH versioned 90% CUDA acceptance

Internal donor baseline: `9af094b08cdc112a8a5048374fee4f135b704087`, source
of SPORC debug preflight 21719837. Extended native
`cms_salience_learned/contracts.py`, `campaign.py`, `production.py` and
`tests/test_cms_salience_learned.py`; no external code was migrated.
Campaign execution v3 and execution acceptance v2 register the user-authorized
90% CUDA / unchanged 85% CPU policy and consecutive worst-length, batch-256
withdrawal probes. Legacy campaign v1/v2 keeps its 85% CUDA policy. Scientific
graph/artifacts remain v1. Training kernel, preparation modules, coordinate
semantics, matching, model, loss, batch size and schedule are unchanged.
Original completed preparation remains reusable read-only; no prior failed
gate or source campaign is modified or treated as newly accepted.

## 2026-09-18: native CMS-LFH preflight lifetime and memory diagnostics

Internal donor baseline: `85fd0214178227026625dd3a5d2b9c79a5fbe2a7`, source
of SPORC preflight 21719606. Refactored that revision's native
`cms_salience_learned/production.py` miniature loop into per-route function
scope and extended `tests/test_cms_salience_learned.py`. No external code
was migrated. The production training kernel, preparation code, matcher,
models, batch size, losses, schedules and registered 85% memory thresholds
are unchanged. Added console diagnostics, not a new scientific artifact
contract. A weak-reference regression reproduces the old acquisition
optimizer retaining parameters into the next route; failure injection checks
that CPU/GPU headroom failures remain fail-closed and publish no acceptance.

## 2026-09-18: native CMS-LFH paired-cache extraction inference repair

Internal baseline: `08c3647981ad6dac54d3cba30212a2515f839915`, the pinned
source of failed SPORC preflight 21719164. Extended native
`src/hlt_classification/cms_salience_learned/training.py` and
`tests/test_cms_salience_learned.py`; no external donor migration. Ordinary
inference now requests the primary-only batch even when a cache retains context.
The existing `data.Cache.batch_primary` API is reused unchanged. Models, loss,
matching, schedules, graph and byte-exact extraction semantics are unchanged;
no scientific contract/version change. Local tests cover the actual preflight
call with a paired cache and extend the optional installed-Weaver parity check.

## 2026-09-18: native CMS-LFH partition routing and preparation reuse

Internal donor checkout: `f966dd804ed9ca8ca93c9c3227d1a7f036ef4434`.
Extended `src/hlt_classification/cms_salience_learned/campaign.py`,
`contracts.py`, `production.py`, `scripts/cms_salience_learned.py`, and
`tests/test_cms_salience_learned.py`; the new `preparation_import.py` is local
implementation, not an external copied donor. Reused scheduler/environment
checks from `src/hlt_classification/jetclass2_delphes/execution.py` unchanged,
including its already registered `sporc_a100_debug` allocation profile; no
Delphes data/model semantics or profiling-to-production transfer are imported.
Publication continues to use native `storage.py` and `data/cache_contracts.py`.

The completed producer motivating reuse is native CMS commit
`f2e8a374f522a39c7f3a6331f0ec77ae12cabaea`. Import validates the selected
producer's actual source commit, not this documentary example: native
`data.py`, `storage.py`, the full scouting/data Git trees, and the coordinate
function AST must match the consuming commit. Reusing preparation preserves
its original hashes and does not execute code from the old worktree.
`CAMPAIGN_SPEC` advances to v2 and `PREPARATION_IMPORT/v1` is added;
scientific graph, models, seeds, losses and remaining v1 artifacts are unchanged.

## 2026-09-17: CMS2JC2 staged production tooling and diagnostics

Same internal donor commit: `2b4c2531c39118edebc8c7890d312f2285ca8eab`;
the exact reused files remain enumerated below and in `provenance.DONORS`.
`campaign`, `submission`, `orchestration`, `measurement`, `tasks`,
`synthetic_acceptance`, `diagnostics` and `plots` are new repository code, not
copied donor implementations. Publication still uses internal cache contracts.
Vector example plots are generated directly, with no external assets. Installed
NumPy/SciPy/Awkward/Uproot/scikit-learn/threadpoolctl/matplotlib and native numeric
library bytes are recorded by acceptance; no environment was installed or changed.

## 2026-09-17: CMS2JC2 provisional integrated response development

Same internal donor baseline: `2b4c2531c39118edebc8c7890d312f2285ca8eab`.
The response continues using the internal sources listed below. The new storage
guard additionally reuses `data/cache_contracts.py::atomic_publish_bytes`;
held-out non-selecting class diagnostics reuse the existing native label mapping.
Assumption contracts, record sampling/parallel quotas, topology/generation,
conditional summaries, paired selection, support/transfer, graph and quota
logic are new code, not migrated sibling-worktree or external implementation.
No old matcher, classifier, active campaign, external environment or raw dataset
was changed. Portable fitted previews are local development evidence only;
this extension is not yet a full-science queue-ready campaign.

## 2026-09-17: CMS2JC2 response preparation and calibration primitives

Internal donor baseline: `2b4c2531c39118edebc8c7890d312f2285ca8eab`.
The new isolated namespace is `src/hlt_classification/cms2jc2_response/`.
No external source, sibling-worktree module, fitted classifier, or previous
matcher/calibration artifact is imported. Existing dirty scientific kernels
were preserved, not copied into this response.

- `data/cache_contracts.py`: canonical hashes, immutable JSON publication and
  validation, file checksums.
- `scouting/{schema,labels,splits}.py`: native branch names, exact historical
  baseline/mapped label selection, and historical manifest authentication.
  Labels authenticate frozen populations; they are not response predictors.
- `jetclass2_delphes/{inventory,contracts,schema,splits,split_registry}.py`:
  source/checksum validation, canonical row identities and frozen profile
  membership masks. The paired `reader.py` was inspected but deliberately not
  reused: the new offline-only reader must work without native HLT branches.

Association search, counter RNG, role allocation, conditional-model cores,
residual backend and guarded preparation submission are new implementation.
Numerical libraries are imported, not vendored. The optional `response` extra
declares scikit-learn/threadpoolctl without GPU/Weaver dependencies. Production
versions still need a recorded SPORC environment/resource acceptance. This is
partial implementation, not full-science readiness.

## 2026-09-16: bounded JetClass2 raw jet-pair audit

Internal donor snapshot: `9640bae8147c8e78ae14e165cb969b066a23baea`.
`jetclass2_delphes/pairing_audit.py` reuses `reader.py::_particles`,
`inputs.py::{eta_phi,wrap_phi}`, inventory checksum/schema validation,
selection policy, split-profile masks, canonical identities, source-byte
provenance and immutable JSON publication. It adds bounded diagnostic sampling
and class/source/kinematics-preserving shuffle controls. No external code is
copied, no constituent matcher or training kernel is changed, and no model
performance or physical-pairing certification is implied by completion.

## 2026-09-16: JetClass2 native concatenation, one-job oracle

Internal donor snapshot: `f2e8a374f522a39c7f3a6331f0ec77ae12cabaea`.
New adapters are `jetclass2_delphes/native_concat{,_data,_model}.py`.

- `jetclass2_delphes/{reader,inputs,cache}.py`: native paired reads, canonical
  per-reconstruction 17-feature transforms, packed blocks and bounded ordered
  process preprocessing. New batches add a discrete source transport channel
  and retain the sum of both native particle counts, without any matcher.
- `jetclass2_delphes/model.py` and installed Weaver: unchanged canonical
  backbone; only the numerical embedding is wrapped to add a learned source
  embedding. No Weaver source is copied. The earlier conceptual donor
  `models/hcwdl_tagged_concat_transformer.py` is not imported (wrong dataset).
- `jetclass2_delphes/salience_learned_{graph,training,data,production}.py`:
  unchanged CE schedule, matched control seeds, checkpoint selection, held-out
  validation partition and immutable completed-reference authentication.
- `jetclass2_delphes/{execution,submission}.py` and
  `scouting/{hcwdl_exact_dag_submission,hcwdl_recovery}.py`: explicit SPORC
  site, guarded exact-ID intent/receipt ledger and canonical dry-run behavior.

New scientific exception is scoped to `JETCLASS2_DELPHES_NATIVE_CONCAT_*/v1`:
native offline+HLT source tags are authorized only for this privileged oracle.
No existing campaign, matching strategy, input schema, or training kernel was
edited. Real installed-Weaver 0.5.3 CPU parity used an existing isolated Temp
installation; no Conda environment was modified. Actual SPORC/A100 acceptance
is required inside the new single job before its full scientific fit.

## 2026-09-15: CMS learned-dense synthetic four-vector parity repair

Test-only reuse of the positive-energy unit-mass construction already in
`models/scouting_particle_transformer.py::validate_scouting_weaver_fp32_parity`,
inspected at `f33b8bff611786129ff941aa09aeca2d9856af37`, for
`tests/test_cms_salience_learned.py::make_cache`. No production kernel or
scientific contract changed; no external Weaver source was copied into the
repository. The mock-independent fixture invariant and real-Weaver privileged
path checks are new tests.

## 2026-09-15: native-CMS learned dense ladder, 500k/250k/250k

New isolated family: `src/hlt_classification/cms_salience_learned/`.
In-repository donor baseline inspected at
`d878beec498bba1f089f4ea1041ed17794e8bf68`; no external donor checkout,
`Fresh_check`, sibling worktree, old checkpoint, or old execution artifact is
imported at runtime.

- Native inputs/labels/splits: `scouting/{schema,labels,identity,splits,
  selective_assignment,inputs,particles,highcov_matcher,highcov_data,matching}.py`.
  The last four already had user-owned uncommitted category-count additions;
  they were not changed or staged by this implementation.
- Exact matcher: `scouting/hcwdl_fullcard_salience_{matcher,contracts}.py`.
- View/coupling kernels: `scouting/hcwdl_{homotopy,upper_coupling,
  unified_balanced}.py` and `repair.py`. These are reused, not replaced with
  Delphes preprocessing. New preparation fits bounded train-only scales.
- Architecture/loss: `models/{scouting_particle_transformer,
  hcwdl_adjacent_fusion_transformer,hcwdl_offline_hlt_fusion_transformer}.py`
  and `scouting/hcwdl_offline_hlt_withdrawal.py`, unchanged. A native wrapper
  supplies the separate-view call interface and exact extraction.
- Training-loop design donor: `jetclass2_delphes/salience_learned_training.py`
  and `salience_learned_graph.py`. The new kernel has native CMS metrics,
  fifteen-class probabilities and its own recipe/artifact family; it does not
  import their eleven-class trainer, graph, models, data or artifacts.
- Shared operational helpers: `data/cache_contracts.py`,
  `scouting/hcwdl_{authorization,exact_dag_submission,recovery}.py`, and only
  site/allocation authentication from `jetclass2_delphes/execution.py`.

## 2026-09-13: auxiliary debug profile continuation

No external or sibling-worktree donor code was copied. Reused unchanged native
`jetclass2_delphes/execution.py` site/allocation helpers from
`bee8bc48a1e634d0858f689b6b733ee3b2232265` to authenticate debug and tier3
separately. The matching campaign's `profile_attempt.py` was inspected as an
operational reference, not imported as auxiliary acceptance or migrated source.
The new `offline_aux/preparation_import.py` binds original auxiliary CPU task
receipts and compares the registered preparation-kernel Git blobs between
producer/execution commits. Scientific targets/model/training kernels and the
existing source/Slurm journal primitives are reused, not rewritten.

## 2026-09-12: isolated Delphes offline auxiliary-supervision study

New code lives exclusively under `jetclass2_delphes/offline_aux/`, with its own
CLI, worker and tests. No external source or sibling-worktree code was copied.
Reusable in-repository donors inspected and unchanged relative to
`82032e35177f83436741d7fa1b9d38fbc4b3efc7`:

- `jetclass2_delphes/contracts.py`, `schema.py`, `selection.py`, `inventory.py`,
  `split_registry.py`, `reader.py`: identities, selection, file verification,
  exact Hamilton allocation and native particle decoding. The auxiliary reader
  adds explicit TRAIN/SELECT/REPORT capabilities; no generic validation bypass.
- `jetclass2_delphes/inputs.py`, `model.py`: unchanged native HLT transform and
  installed Weaver factory/configuration. The new wrapper taps the final class
  vector without copying or replacing Weaver's forward implementation.
- `jetclass2_delphes/cache.py`: RamBlock, the pure array batching operation and
  process-thread limiter only. New cache construction does not call constituent
  matching, view construction or require a matching-foundation artifact.
- `jetclass2_delphes/reporting.py`, `execution.py`: native eleven-class metrics
  and exact SPORC site/allocation checks. New weighted cluster-bootstrap code is
  tested against the existing metric evaluator on explicitly repeated rows.
- `data/cache_contracts.py`, `scouting/hcwdl_authorization.py`: canonical hashes,
  deterministic compact NPZ, immutable publication and exact clean/pushed source.
- `sbatch/jetclass2_delphes_common.sh` (and its unchanged `sbatch/common.sh`
  helper): isolated SPORC activation and thread/import-path hygiene.

New submission/recovery code follows the repository's exact-ID intent/receipt
pattern but owns a separate namespace and never calls cancellation/priority APIs.
New artifact families are all `JETCLASS2_DELPHES_OFFLINE_AUX_*/v1`; no existing
dataset, split, matching, scientific recipe or runtime-profile family was
silently redefined. The donor paths above have no dirty semantic-byte delta
against the cited HEAD. Unrelated dirty Scouting files were not modified.

Actual production artifacts require their own later clean pushed source pin;
this donor baseline is not an assertion that the working tree is clean.

## 2026-09-12: site-bound Delphes SPORC readiness

New execution/readiness modules and thin CLI/shell helper extend the preceding
uncommitted Delphes migration. No external donor was copied. Hash/immutable
publication and exact-DAG receipt helpers still come from repository baseline
`fd1ed1d01d54bf2ad4d42ffa6311432263a14770`, specifically
`data/cache_contracts.py`, `scouting/hcwdl_authorization.py`,
`scouting/hcwdl_exact_dag_submission.py` and `scouting/hcwdl_recovery.py`.
The new readiness submitter reuses Delphes `submission.py`'s guarded exact
submission, with an independent identity, roots and authorization phrase.
`sbatch/jetclass2_delphes_common.sh` explicitly sets the selected isolated
environment before calling the unchanged `sbatch/common.sh` activation helper.

RUNTIME_PROFILE and INSTALLED_ENVIRONMENT are v2; CAMPAIGN_SPEC is v3.
EXECUTION_SITE, READINESS_SPEC and RESOURCE_MEASUREMENTS are new v1 contracts.
The cache budget calculation is shared with the existing Delphes preparation
bound; science, membership, field policy and model/loss/schedule are unchanged.
No pre-existing dirty Scouting donor file was edited or staged by this task.

## 2026-09-11: reusable Delphes training-size split registry

New `jetclass2_delphes/split_registry.py` and
`scripts/create_jetclass2_delphes_split_registry.py` implement metadata-only,
class-stratified nested selection and compact per-file entry masks. No donor
split algorithm was copied. Runtime hash/immutable-JSON helpers are reused from
`data/cache_contracts.py` at repository donor baseline
`fd1ed1d01d54bf2ad4d42ffa6311432263a14770`; inventory/selection/schema/reader and
foundation/production integration extend the preceding uncommitted local
Delphes migration, not an invented published donor commit. Registry artifacts
record the actual implementation-file hashes separately from Git HEAD.
No raw ROOT files, legacy campaign source, checkpoints or remote jobs were
modified. New families are SPLIT_DESIGN/ROLE_MEMBERSHIP/SPLIT_REGISTRY/
SPLIT_PROFILE v1 and subset FOUNDATION_SPEC/CAMPAIGN_PLAN/CAMPAIGN_SPEC v2.

## 2026-09-11: isolated JetClass2 Delphes local migration foundation

New implementation lives under `src/hlt_classification/jetclass2_delphes/`.
Repository donor baseline: `fd1ed1d01d54bf2ad4d42ffa6311432263a14770`.
No `Fresh_check` imports, sibling-worktree imports, FullSim weights, old dataset
artifacts, or previous runtime evidence are reused.

| Donor file | Use and explicit adaptation |
| --- | --- |
| `data/cache_contracts.py` | Runtime reuse of hashing, deterministic NPZ and atomic publication |
| `data/part_inputs.py` | Adapted analytic 17-feature mathematics; new p4 reader, class map, capacity and contracts |
| `models/particle_transformer.py` | Runtime Weaver factory/config reuse; new 17/11 wrapper, sequence trimming disabled |
| `scouting/hcwdl_fullcard_bottleneck_matcher.py`, `hcwdl_fullcard_bottleneck_contracts.py` | Runtime matrix-level exact solver and quantization; no old schema projection |
| `scouting/hcwdl_homotopy.py`, `repair.py` | Adapted exact endpoint/support and atomic applicability principles into new raw-field layout |
| `scouting/hcwdl_upper_coupling.py`, `hcwdl_unified_balanced.py` | Adapted mass-normalized insertion/removal and balanced circular switch mathematics; no residual substitutions under full cardinality |
| `scouting/hcwdl_tri100_spine4_graph.py` | Copied branch/rational-coordinate/schedule definitions into independently versioned fresh-reference graph |
| `scouting/hcwdl_mhpe_tri60_training.py` | Adapted CE+forward-KL/T², schedule, patience/selection and no-resume principles; not a runtime import of its 15-class trainer |
| `scouting/hcwdl_authorization.py` | Runtime exact-clean-pushed-checkout validation |
| `scouting/hcwdl_exact_dag_submission.py`, `hcwdl_recovery.py` | Runtime canonical dry-ledger validation, dependency resolution and submission receipts; new wrapper adds a campaign-wide submitter claim, pre-sbatch durable intent and cross-recovery active-job checks |
| `sbatch/common.sh` | Existing absolute-path Conda/Tigris activation helper; new worker scripts supply explicit new data/spec paths |

The matrix solver's inspected SHA256 was
`9c92616446e77c27a00e1871dfdbdb2d46b2634665b80a79c68fe25365d5f190`.
Its module-level transitive data imports include pre-existing dirty work:
`highcov_data.py` SHA256
`0803a1a813d043ebb30581ecaa622dd7d60bdb65cdbd09ae59895f15badd90c5`,
`particles.py` SHA256
`73d5890a2a5f4b35e0f2fd7e0fe7cd232a7b5afc9e2ea4dd4814e232435e4f38`,
and `highcov_features.py` SHA256
`1190f99e917e186a16e66dea5a865f41616cbc6255595dd623e3194015ef376b`.
Those files were not edited or migrated in this task. The new adapter passes
numeric matrices directly and does not use their old raw-particle adapters.

The provisional label table references upstream
`jet-universe/jetclass2_generation`, commit
`3a7a1355f4230b5790669286466080d7fa3b6794`,
`delphes_analyzers/FatJetMatching.h`. This is a code reference, **not** a claim
that it is Luka's exact producer revision. Source inventory records actual
implementation bytes as well as Git HEAD because new local code is uncommitted.

This block also implements source-pinned production orchestration and synthetic
end-to-end tests. It is not a claim of genuine Weaver/Tigris acceptance.

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

## Historical full-cardinality U100 compatibility adapter

The adjacent output-handoff `/v2` family consumes the immutable
non-persistent full-cardinality `SP4_COARSE_U100_from_U050` artifact produced
by Tigris job `98318`. The exact completed source family was implemented at
clean same-repository commit
`175cbcd6bf749f0014486a59a8c103802708abd1`; the earlier `/v2` snapshot at
`965bdad6aea2fdf81356275dfcdca698717a471f` lacked the corrected offline
lost-track endpoint identity semantics. Its
module names were later reused by a separately versioned persistent-HLT
experiment. The adapter therefore preserves the historical contract and view
construction locally instead of importing the current persistent-HLT module.
The source checkpoint and report remain read-only artifact parents.

The following `/v3` donor blobs supersede the three `/v2` contract, graph, and
campaign rows retained below as historical implementation context:

| Historical path and Git blob | Current destination | Retained semantics |
|---|---|---|
| `src/hlt_classification/scouting/hcwdl_tri100_spine4_bottleneck_contracts.py` -- `dd0109ab40bf2b1f89717afab952aa7d129f927d` | `hcwdl_adjacent_output_handoff_source.py` | Exact non-persistent full-cardinality `/v3` source contract identities. |
| `src/hlt_classification/scouting/hcwdl_tri100_spine4_bottleneck_graph.py` -- `a2ad38187006e43e6aa28c19386fd56f5b4115f9` | `hcwdl_adjacent_output_handoff_source.py` | Exact graph and recipe including both unclassified-particle endpoint policies. |
| `src/hlt_classification/scouting/hcwdl_tri100_spine4_bottleneck_campaign.py` -- `751d02b0b40f24358e86e3acd3704cac2bb90961` | `hcwdl_adjacent_output_handoff_source.py` | Immutable `/v3` campaign-parent, endpoint-policy, and population checks. |

| Historical path and Git blob | Current destination | Retained semantics |
|---|---|---|
| `src/hlt_classification/scouting/hcwdl_tri100_spine4_bottleneck_contracts.py` — `6e218fecb50c54b2a2f718e958e456eb206d55c7` | `hcwdl_adjacent_output_handoff_source.py` | Exact non-persistent full-cardinality `/v2` source contract identities. |
| `src/hlt_classification/scouting/hcwdl_tri100_spine4_bottleneck_graph.py` — `058f0a4729137f9e0973b43e68e20769e921e297` | `hcwdl_adjacent_output_handoff_source.py` | Campaign, recipe, pairing, and exact coarse-U100 graph authentication. |
| `src/hlt_classification/scouting/hcwdl_tri100_spine4_bottleneck_campaign.py` — `0602fc5bfbfff0e3bbbc5d74bfc8390f53cdb598` | `hcwdl_adjacent_output_handoff_source.py` | Immutable campaign-parent and all-mapped-population checks. |
| `src/hlt_classification/scouting/hcwdl_tri100_spine4_bottleneck_runner.py` — `4002c92c62e4d7ed02ad370db0dfa7fcb3124dac` | `hcwdl_adjacent_output_handoff_runner.py` | Exact `replace_source_with_target_v1` RAM-only student-view reconstruction and historical training authority. |
| `src/hlt_classification/scouting/hcwdl_tri100_spine4_graph.py` — `3de1bbe6886ac98df76fea9db6d6adc21856bc7e` | reused current module (byte-identical at the compatibility boundary) | Exact `SP4_COARSE_U100_from_U050` node payload and seed alias. |

No runtime import from a detached historical worktree is used.

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

## Strategy-B adjacent learned-fusion composition

The `HCWDL_ADJACENT_LEARNED_FUSION_HANDOFF_*/v1` implementation is original
repository-local work; no external or legacy source file was copied. It
composes the already versioned adjacent-output source adapter and
full-cardinality cache construction at commit
`7acc62313c5cd47dc0902d025a31716448e7daf1` with the repository-local anchored
fusion/exact-withdrawal primitives. The latter primitives were developed in
this repository for the immediately preceding offline/HLT fusion campaign and
are imported as reusable package APIs, not migrated donor files. Strategy B
changes their content semantics from offline/HLT to adjacent richer/lower
views, freezes unused cross-pair parameters, adds the exact rational morph
schedule and physical primary extraction, and tests these changes in
`tests/test_hcwdl_adjacent_learned_handoff.py`. There is no new third-party
code, dependency, or attribution obligation.

## JetClass2 salience Strategy-B composition

The `JETCLASS2_DELPHES_SALIENCE_LEARNED_HANDOFF_*/v1` implementation is
repository-local composition rather than an external migration. It reuses the
versioned Strategy-B withdrawal loss from
`src/hlt_classification/scouting/hcwdl_offline_hlt_withdrawal.py` and the
JetClass2 salience foundation/view APIs already recorded by their own plans and
contracts. The new 17-input/11-output asymmetric fusion model, three-spine
54-fit graph, U000-to-D000 morph control, RAM cache adapters, staged SPORC DAG,
and tests were authored in this repository. No historical checkpoint, target
bank, dataset, generated campaign result, or external source file was copied,
and runtime imports do not reach an old worktree or repository.

The 2026-09-14 loss-interface repair adds read-only `hlt_states` / `hlt_mask`
aliases to the JetClass2 `FusionOutput` for compatibility with that shared
withdrawal objective. Both names refer to the original lower/primary tensors,
not necessarily native-HLT content on intermediate U/D coordinates. The
reference API is the unchanged repository-local
`scouting/hcwdl_offline_hlt_withdrawal.py` at failed-execution commit
`15094633f9aa3a0e3f9e418704ac3c0a46dac11d`. No donor file or loss implementation
was copied; the coefficients, masks, directed gradients and exact-zero route
are unchanged. Tensor-level regression tests now exercise the interface even
when installed Weaver is unavailable.

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

## 2026-10-08: Luka FullSim stage-2 internal reuse

Donor commit: `97ee3434a93de2fc7b0f09d506926cd50c5cde46`. No external code or
`Fresh_check` was copied/imported. New code is under
`src/hlt_classification/luka_fullsim/{particles,particle_audit,stage2,stage2_cache,preflight}.py`.
The stage-1 files from the same local implementation are parents, not new donor
data. No historical training weights, split membership or fitted proxy is reused.

| Donor paths relative to `src/hlt_classification/` | Retained semantics |
| --- | --- |
| `data/cache_contracts.py`, `provenance.py` | Canonical hashes, deterministic NPZ, immutable publication, source locks |
| `cms2jc2_response/bridge.py` | Physical particles, charge/applicability and p4 validation only |
| `cms2jc2_response/measurement.py` | Linux process-tree resource measurements |
| `cms_proxy_ladder/views.py` | Full-cardinality salience matching, persistent shell, mass-balanced tail and exact deterministic switch domain |
| `literature_context/transform.py` (`build_inputs`, `input_contract` only), `cms_proxy_ladder/inputs.py` | Existing 17-feature asinh/log1p frontend; NO synthetic context transformation called |
| `jetclass2_delphes/cache.py` (`RamBlock`, `RamCache`) | Ragged RAM storage, real batch lengths, metadata outside features |
| `jetclass2_delphes/{model,campaign,runner,reporting,banks,acceptance,execution,dzfix_fusion_model}.py` | Eleven-class model, unchanged 60--100-pass science recipe, marked one-pass probes, metrics, probability joins, installed parity and SPORC checks |
| `context_fusion/inputs.py` (`PairedCache`, `new_model`), `context_fusion/worker.py` (`clear`, `save_state`, `stress`) | Paired identity joins, native independent encoders, checkpoint and longest-TRAIN-batch stress |

Intentional changes: real Luka native endpoints and selected source-row joins;
new Luka contracts instead of proxy provenance; stored-unit audit before explicit
unit/sentinel declaration; all selected particle/view validation; no final-test
capability; temporary same-dataset probe teacher rather than imported U000 bank.
Current tests/evidence: `tests/test_luka_fullsim_stage2.py` and HANDOFF.
No third-party source/license changes. Installed Weaver remains an authenticated
runtime dependency, and local tests do not establish remote acceptance.

## Deliberately excluded

Do not migrate checkpoints, caches, logs, run registries, historical campaign
IDs, unrelated campaign packages, old handoffs, or broad script collections.

## 2026-10-02: literature-inspired offline-only pilot

No external donor or `Fresh_check` runtime dependency. New implementation lives
under `src/hlt_classification/literature_proxy/`; it does not reuse any learned
CMS response, calibration map, or generated proxy particles. In-repository
interfaces reused from donor commit `41a59b644f1fa4398fc022e06ff88ab16f5fb5d9`:

| Donor file | Local SHA-256 at inspection | Retained semantics |
| --- | --- | --- |
| `cms2jc2_response/bridge.py` | `c3a3e9e810996899894390f8733c5c6ac547599fecaf323a3415c137d7db9dae` | Physical `Particles`, wrapped phi, physical p4 reconstruction; no fitted response |
| `data/cache_contracts.py` | `004aef8e8bbdf6ce513479024d4a119f92277fca3e359f977a5f4022d8e0d7dc` | Canonical hashes, deterministic NPZ, immutable atomic publication |
| `jetclass2_delphes/split_registry.py` | `fd2246185b042de3e255c5073ccd75c47d687a9428139d22cb6fac32aeace911` | Existing authenticated train membership, without resplitting |
| `jetclass2_delphes/inventory.py` | `998323e064d10916268f0ab2e4ea833d2cd7a568eafde259611d92ce6b98e605` | Per-source ROOT byte/schema/cycle authentication |

Paths above are relative to `src/hlt_classification/`. The new offline adapter
retains the bridge's explicit unknown PID and unavailable-error policy, but
uses native JetClass2 GeV/mm/sign/PV semantics directly instead of importing a
CMS/JC2 fitted compatibility review. The new reader deliberately does not use
`DatasetReader`, which also reads HLT arrays. No third-party source was copied;
literature citations motivate mechanisms, not measured parameter values.
Focused replay, mask, physics-invariant, native ROOT, and artifact tests reside
in `tests/test_literature_proxy*.py`; execution evidence is in `docs/HANDOFF.md`.


## 2026-11-03 ? joint privileged-adapter Hydra follow-up

No Fresh_check runtime import or external donor. In-repository donor commit:
`4b9630fa00f137c4e48b749eb86714719b9d26cf` (frozen-adapter implementation),
unchanged at the new source baseline `ceb2146149d2b1508a36b2cf91b274e0b58a26e6`.

| Donor | New use |
| --- | --- |
| `luka_frozen_adapter_hydra/model.py` | Copied to `luka_joint_adapter_hydra/model.py`; jointly train teacher backbone, prohibit warm starts, add final-token auxiliary tap. |
| `luka_frozen_adapter_hydra/training.py` | Copied; uniform full30 schedule, correction weights0.3/1, final auxiliary deployment, new report semantics. |
| `luka_frozen_adapter_hydra/contracts.py`, `campaign.py` | Copied; new v1 namespace, ordered grid, source baseline and original control reference. |
| `luka_frozen_adapter_hydra/worker.py`, `resources.py` | Copied; same authenticated data/real miniatures, fresh teachers, new control comparison reporting. |
| `luka_frozen_adapter_hydra/submission.py` | Copied; new job namespace plus authenticated idempotent post-setup `finish`. |
| `luka_frozen_adapter_hydra/__init__.py` | New package marker. |
| `luka_frozen_adapter_hydra/inputs.py` | Imported unchanged through a thin wrapper; no new export or raw-data preparation. |
| `scripts/luka_frozen_adapter_hydra.py`, `scripts/queue_luka_frozen_adapter_hydra.sh`, `sbatch/run_luka_frozen_adapter_hydra.sh` | New joint-adapter CLI/launcher/worker; existing-input discovery, detached session, progress and resume commands. |
| `tests/test_luka_frozen_adapter_hydra.py` | Adapted synthetic Weaver/gradient/publication/DAG coverage, expanded final-layer and recovery tests. |

`references.py` and `progress.py` are new. Original-source replay uses unchanged
`luka_attention_weight/source.py::replay`; Hydra numerical qualification/site,
portable bundle, selected-model storage, native Weaver factory, exact-DAG journal
and fixed30 recipe remain imports of existing contracts. Donor packages and
existing campaign artifacts are untouched. New science contracts are
`LUKA_JOINT_ADAPTER_HYDRA_*/v1`; original frozen controls retain their own namespace.
