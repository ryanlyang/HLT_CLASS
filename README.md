# HLT Classification

A clean, standalone research repository for HLT-only JetClass classification
and training-time use of paired offline information.

## Current status

The [next supervision ideas draft](docs/plans/LUKA_NEXT_SUPERVISION_CAMPAIGN_IDEAS.md)
collects seven proposed additions to native CE+U100-w5: conditional distributions
of richer jet embeddings, conditional denoising of pure-OFFLINE particle sets,
a learned HLT/OFFLINE compatibility judge, structured OFFLINE question prediction,
particle-soup membership, intact pair/triplet membership and learned offline
pattern-frequency prediction. Particle soup is the user's current priority, with
broad, matched and mined distractor comparisons. The draft describes preferred
designs, controls and an expandable grid at fixed deployed architecture.
The [three-group implementation proposal](docs/plans/LUKA_SEVEN_CANDIDATE_IMPLEMENTATION_PLAN.md)
selects40 comparison rows with an interleaved submission order: the first seven
science submissions cover all seven preferred setups. Every science fit is
independent, with no fit-to-fit dependencies or family lanes; partial results
must be reportable as fits finish. Group A now has an additive
[implementation and Hydra runbook](docs/LUKA_NEXT_SUPERVISION_GROUP_A_RUNBOOK.md):
22 particle-soup, intact-group and learned-judge configurations with authenticated
reference reuse. It keeps native CE+U100-w5 and adds removable heads; independent
science release follows real measured setup. Groups B/C remain unimplemented.

The [HLT-aligned particle-teacher campaign](docs/LUKA_PARTICLE_TEACHER_HYDRA_RUNBOOK.md)
adds twelve native HLT fits on Hydra RTX A6000. Every fit keeps CE + U100-w5.
Ten add removable heads predicting full block6/block8 particle embeddings from
the completed privileged `teacher_m16`; two are same-seed native U100-w5 controls.
Student attention and deployed architecture remain native. The grid tests target
depth, weight, centered supervision and a second seed. Targets stay in allocation
RAM; a durable CPU release queues independent fits after real measured setup.
See the [scientific plan](docs/plans/LUKA_PARTICLE_TEACHER_HYDRA_PLAN.md),
[v1 contract](docs/contracts/LUKA_PARTICLE_TEACHER_HYDRA.md) and
[local validation](docs/LUKA_PARTICLE_TEACHER_HYDRA_VALIDATION.md).

The [shared-message Hydra campaign](docs/LUKA_SHARED_MESSAGE_HYDRA_RUNBOOK.md)
adds twelve fresh RTX A6000 fits. Every HLT model retains U100 embedding weight5.
HLT and privileged message producers share the downstream classifier; agreement
encourages messages the HLT branch can reproduce. Matched attention-predictor
controls, two seeds, weighting and fade ablations separate the effects. A durable
Slurm release job submits independent fits after warmed production miniatures.
See the [plan](docs/plans/LUKA_SHARED_MESSAGE_HYDRA_PLAN.md) for scientific scope;
the existing campaigns and their outputs remain separate.

The [CE-only cross-message controls](docs/LUKA_CROSS_MESSAGE_CE_HYDRA_RUNBOOK.md)
add two independent Hydra A6000 fits alongside the running cross-message study.
They retain the width16/32 HLT message architectures and matched seed0 but remove
the embedding head and use CE alone, with no privileged optimization targets.
The existing CE+U100-w5 controls remain separate. Parent setup evidence is reused;
each new allocation warms its CE-only kernels, measures a fresh miniature, then
starts full training from the matched initial weights. Execution v2 corrects
the cold-start runtime projections that stopped jobs145718/145719.

The [Oscar FullStat Multi3 contrastive campaign](docs/LUKA_FULLSTAT_MULTI3_CONTRASTIVE_RUNBOOK.md)
adds pure-OFFLINE jet identification to the completed U100/CONCAT/FUSION
embedding-prediction recipe: one fresh HLT student,3M/500k,20 epochs, mix0.3
with a three-epoch ramp. It reuses all frozen teachers and has no separate
preflight job. Exact sampler optimizations, full epoch recovery, detached
submission and integrated real-allocation checks are included. See the
[plan](docs/plans/LUKA_FULLSTAT_MULTI3_CONTRASTIVE_PLAN.md) and
[local evidence](docs/LUKA_FULLSTAT_MULTI3_CONTRASTIVE_VALIDATION.md).

The [cross-attention message Hydra campaign](docs/LUKA_CROSS_MESSAGE_HYDRA_RUNBOOK.md)
adds two jointly trained privileged teachers and twelve fresh CE+U100-w5 students.
Teachers read frozen U100 particle features and inject compact32/16-wide messages
after HLT block4. Students predict those messages from HLT, with injection,
auxiliary-only and matched architecture controls at weights0.1/0.3/1 plus a
second-seed pair. Existing Brown data are reused; a small authenticated U100
encoder export is required. There is one shared measured setup, no separate
preflight or scheduling lanes. See the [plan](docs/plans/LUKA_CROSS_MESSAGE_HYDRA_PLAN.md)
and [local validation](docs/LUKA_CROSS_MESSAGE_HYDRA_VALIDATION.md).

The [joint privileged-adapter Hydra follow-up](docs/LUKA_JOINT_ADAPTER_HYDRA_RUNBOOK.md)
adds two teachers trained end to end from scratch and twelve fresh HLT students,
all with U100 embedding weight5. It compares intermediate injected corrections,
intermediate auxiliary supervision and final-particle auxiliary supervision at
weights0.3/1. Existing Brown inputs and completed architecture controls are
reused. Shared setup measures the new training paths; there are no student lanes
or new reference fits. See the [plan](docs/plans/LUKA_JOINT_ADAPTER_HYDRA_PLAN.md)
and [validation evidence](docs/LUKA_JOINT_ADAPTER_HYDRA_VALIDATION.md).

The [frozen privileged-adapter Hydra campaign](docs/LUKA_FROZEN_ADAPTER_HYDRA_RUNBOOK.md)
adds two oracle adapters on a frozen M0HLT checkpoint and twelve fresh HLT
students: attention-bias or edge-message correction, auxiliary-only/CE-only/
combined use, with and without U100-w5. It has no scheduling lanes; only genuine
selected-teacher dependencies. Shared setup measures the new kernels before
science submission. See the [plan](docs/plans/LUKA_FROZEN_ADAPTER_HYDRA_PLAN.md)
and [validation evidence](docs/LUKA_FROZEN_ADAPTER_HYDRA_VALIDATION.md).

The [U100 particle/pair correction plan](docs/plans/LUKA_U100_PARTICLE_PAIR_CORRECTION_PLAN.md)
defines **four new SPORC/debug fits, each on top of CE + 5 * U100
jet-embedding prediction**: direct pair correction, particle correction,
particle correction with derived-pair supervision, and independent particle/pair
predictors. The question is whether each addition improves AUC and recovery
beyond U100-w5. **Reuse the completed U100-w5 reference (u100_w5_s0,
job 21993056); do not retrain or queue another reference fit.** Authenticate its
saved results and reuse compatible existing U100 targets. The additive
implementation is under `src/hlt_classification/luka_u100_pair_correction/`.
The graph has one shared setup, four student fits and one report, automatically
queued in two debug lanes after measured setup acceptance. See the
[queue/runbook](docs/LUKA_U100_PAIR_CORRECTION_RUNBOOK.md) and
[v1 contract](docs/contracts/LUKA_U100_PAIR_CORRECTION.md). New-model SPORC
resource acceptance is performed by shared setup before any scientific fit.

For the proposed Brown CS Hydra move, the [minimal transfer runbook](docs/LUKA_U100_PAIR_HYDRA_TRANSFER_RUNBOOK.md)
provides a campaign-only exporter and plain-Python checksum verifier. The user
has verified that compact bundle on Brown. The additive
[Hydra execution adapter](docs/LUKA_U100_PAIR_HYDRA_RUNBOOK.md) provides portable
inputs, a separate pinned environment, control portability checks and the same
four fits in two L40 lanes on `gpus`. The environment and precision diagnostic
131484 have completed; the [v2 campaign runbook](docs/LUKA_U100_PAIR_HYDRA_V2_RUNBOOK.md)
reuses that numerical evidence and the saved Hydra control. Genuine shared
setup miniatures/resource acceptance remain required before science release.
Existing SPORC workers and jobs are unchanged.

The [contrastive Hydra runbook](docs/LUKA_U100_CONTRASTIVE_HYDRA_RUNBOOK.md)
adds a separate execution of the original twenty-row U100-w5/OFFLINE jet
identification campaign. Completed SPORC fits are reused; all remaining
students queue together after shared setup with no student dependencies or
concurrency cap. Only the report waits. It reuses Brown's existing inputs and
environment and transfers only OFFLINE TRAIN keys plus completed results.
Original thirty-epoch scientific code is unchanged. Local validation and
required real L40 resource acceptance are documented in the runbook.

For a different Oscar split or epoch schedule, the
[U100 contrastive porting handoff](docs/LUKA_U100_CONTRASTIVE_OSCAR_HANDOFF.md)
explains the successful mixed/kinematic sampling candidates, exact losses,
code dependencies, and how to adapt targets, controls and scheduling together.
It documents a future port; it does not add an executable Oscar campaign.

The isolated [CMS-calibrated JetClass2 response comparison](docs/plans/CMS_CALIBRATED_JETCLASS2_HLT_RESPONSE_THREE_FAMILY_IMPLEMENTATION_PLAN.md)
has staged CPU queue tooling and local end-to-end tests. It compares tables,
neighbourhood splines and shallow trees on CMS training-only roles, then audits
frozen transfer to JetClass2 **offline only**. Producer conventions are explicitly
provisional under the user's authorization, not verified physics. Real SPORC
20k/100k acceptance, measured resource locks, clean pushed source and separate
stage authorization remain mandatory before full science. Existing ladders are
untouched. See the [response contract](docs/contracts/CMS2JC2_RESPONSE_PREPARATION.md).

The separate [JetClass2 Delphes migration](docs/plans/JETCLASS2_DELPHES_DATASET_MIGRATION_IMPLEMENTATION_PLAN.md)
now has a locally tested inventory/reader, frozen file splits, reduced 17-input
interface, full-cardinality U/D views, RAM cache, probability banks and CE/KD
components under `src/hlt_classification/jetclass2_delphes/`. Its reusable split
registry defines nested 500k/1M/1.5M/2M training profiles with identical 1M
validation and 1M sealed test jets; the first intended run is TRAIN_500K,
not the full-data reservoir. Source-pinned
preparation, measured acceptance, checkpoint publication, Slurm dry/live plans,
monitoring and restart-zero recovery are implemented. The user has now supplied
successful genuine new-data Weaver/A100 profile evidence (job 21619139), then
a passed 59-task dry run and live submission of the 500k matching campaign.
Its profile-only debug measurement is explicitly bound to tier3 production
using `atlas_kd_sporc`; readiness does not auto-launch science. See the detailed
[SPORC and new-dataset agent handoff](docs/JETCLASS2_DELPHES_SPORC_AGENT_HANDOFF.md)
for exact paths, measured resources, current branch-cancellation intent, safe
commands, code map, and the limits of this evidence. The
[new dataset contract](docs/contracts/JETCLASS2_DELPHES.md) remains authoritative;
another study needs its own applicable acceptance. Existing campaigns below
are not migrated or mutated by that implementation.

PMARD is the active campaign. Its repository-local Scouting schema, label and
file-split contracts, bounded ROOT streaming, 21-channel HLT and native-offline
inputs, the canonical fitted-strict selective matcher, persistent compact
per-source sparse assignments, retained `P4_ONLY/v1`, and primary mixed-type
`SELECTIVE_FULL_PARTICLE_ENDPOINT/v1` alpha repair,
15-output Scouting ParT adapters, KD losses/metrics, hash-chained locks, and
minimum-storage campaign DAG are under `src/hlt_classification/scouting/`.
The selective endpoint replaces all 21 fields and p4 for accepted matches while
leaving unmatched tokens byte-identical to HLT. Its authorization lock binds
the fixed matcher artifact, exact threshold, row selection, and assignment
manifest; training uses
deterministic cross-file RAM shuffling and FP32 KD/representation losses under
BF16 model autocast. Campaign-spec v10 additionally computes each selected
nonzero-alpha repaired view once into an authenticated process-local FP32 RAM
cache and exactly replays the existing epoch sampler; immutable v9 executions
retain their original streamed behavior.
Local synthetic verification does not authorize production: installed-Weaver
parity and the complete real streamed Tigris miniature remain mandatory.

Transfer Blocks 1--4, 6, and 7 are complete locally. Transfer Block 5 is code-complete:
the repository contains the canonical 17-feature input transform, frozen
from-scratch Weaver Particle Transformer adapter, and authoritative FP32
parity command. Its final acceptance still requires running that command
against real installed Weaver on Tigris. Deterministic baseline training,
checkpoint resume, label-free inference, and frozen metrics are implemented;
the source-bound compact campaign, production workers, storage gate,
monitoring, and exact-ID recovery workflow are implemented. Real Tigris
acceptance remains.

The earlier PRAD campaign remains implemented locally as a separate authenticated
three-role campaign: fixed split construction, real-data audit, Hungarian
association, C/A targets, offline teacher, HLT-only relation-biased student,
E0--E10 and V1--V10 configurations, exact resume, locked evaluation,
five-seed reporting, and measured-resource Slurm orchestration are present.
This is implementation readiness, not a scientific result: installed-Weaver
parity, a real Tigris miniature, the 500k/150k/500k artifacts, training, and
the sealed final evaluation remain to be executed.

Do not interpret the presence of plans or directory placeholders as an
executable scientific pipeline. See [the current handoff](docs/HANDOFF.md).

## Scientific setting

The deployment constraint is:

```text
degraded HLT constituent view -> model -> jet-class prediction
```

Paired offline information may be used during training only through declared
targets, teachers, or controls. It is not available to a deployable inference
graph.

The initial synthetic input domain will use
`fixed_hlt_v3_track_dominant_proxy`, explicitly described as a controlled
HLT-like stress proxy rather than real ATLAS or CMS HLT reconstruction.

## Repository locations

```text
Windows:
C:\Users\22rya\ComputerScience\CERN\HLT_Classification

Tigris:
/home/ryreu/atlas/HLT_Classification

JetClass data on Tigris:
/home/ryreu/atlas/PracticeTagging/data
```

## Documentation

- [Agent instructions](AGENTS.md)
- [Repository transfer plan](REPOSITORY_TRANSFER_PLAN.md)
- [Current handoff](docs/HANDOFF.md)
- [Research-compute runbook](docs/RESEARCH_COMPUTE_RUNBOOK.md)
- [PRAD research-compute runbook](docs/PRAD_RUNBOOK.md)
- [PRAD requirements traceability](docs/PRAD_REQUIREMENTS_TRACEABILITY.md)
- [PMARD active implementation plan](docs/plans/SCOUTING_ALPHA_REPAIR_DISTILLATION_OPTIMAL_CAMPAIGN.md)
- [PMARD pilot preliminary results and implementation map](docs/PMARD_PILOT_PRELIMINARY_RESULTS.md)
- [PMARD T100 KD sweep contract](docs/contracts/PMARD_T100_KD_SWEEP.md)
- [PMARD paired KD schedule follow-up contract](docs/contracts/PMARD_KD_SCHEDULE_FOLLOWUP.md)
- [PMARD all-model exploratory test comparison contract](docs/contracts/PMARD_EXPLORATORY_TEST_COMPARISON.md)
- [PMARD research-compute runbook](docs/PMARD_RUNBOOK.md)
- [HCWDL primary cold/warm ladder plan](docs/plans/HIGH_COVERAGE_COLD_WARM_DISTILLATION_LADDER.md)
- [HCWDL dense cold 300k runbook](docs/HCWDL_DENSE_COLD_RUNBOOK.md)
- [HCWDL dense cold 300k contract](docs/contracts/HCWDL_DENSE_COLD_PILOT.md)
- [HCWDL five-point dense cold 300k runbook](docs/HCWDL_DENSE5_COLD_RUNBOOK.md)
- [HCWDL five-point dense cold 300k contract](docs/contracts/HCWDL_DENSE5_COLD_PILOT.md)
- [HCWDL dense failed-closure recovery contract](docs/contracts/HCWDL_DENSE_RECOVERY.md)
- [HCWDL primary failed-closure recovery](docs/HCWDL_CAMPAIGN_RECOVERY_RUNBOOK.md)
- [HCWDL primary recovery contract](docs/contracts/HCWDL_CAMPAIGN_RECOVERY.md)
- [HCWDL structural-feature homotopy implementation plan](docs/plans/HCWDL_STRUCTURAL_FEATURE_HOMOTOPY_IMPLEMENTATION_PLAN.md)
- [HCWDL structural-feature homotopy contract](docs/contracts/HCWDL_STRUCTURAL_FEATURE_HOMOTOPY.md)
- [HCWDL structural-feature homotopy runbook](docs/HCWDL_STRUCTURAL_FEATURE_HOMOTOPY_RUNBOOK.md)
- [HCWDL unified-root balanced homotopy successor plan](docs/plans/HCWDL_UNIFIED_BALANCED_HOMOTOPY_IMPLEMENTATION_PLAN.md)
- [HCWDL unified-root balanced homotopy contract](docs/contracts/HCWDL_UNIFIED_BALANCED_HOMOTOPY.md)
- [HCWDL unified-root balanced homotopy runbook](docs/HCWDL_UNIFIED_BALANCED_HOMOTOPY_RUNBOOK.md)
- [HCWDL unified-balanced full-data three-arm plan](docs/plans/HCWDL_UNIFIED_BALANCED_FULL_DATA_THREE_ARM_PLAN.md)
- [HCWDL unified-balanced full-data runbook](docs/HCWDL_UNIFIED_BALANCED_FULL_DATA_RUNBOOK.md)
- [HCWDL unified-balanced full-data contract](docs/contracts/HCWDL_UNIFIED_BALANCED_FULL_DATA.md)
- [HCWDL unified-balanced coarse full-data plan](docs/plans/HCWDL_UNIFIED_BALANCED_FULL_DATA_COARSE_THREE_ARM_PLAN.md)
- [HCWDL unified-balanced coarse full-data contract](docs/contracts/HCWDL_UNIFIED_BALANCED_FULL_DATA_COARSE.md)
- [HCWDL unified-balanced coarse full-data runbook](docs/HCWDL_UNIFIED_BALANCED_COARSE_RUNBOOK.md)
- [HCWDL multi-horizon projection-ensemble full-data plan](docs/plans/HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_IMPLEMENTATION_PLAN.md)
- [HCWDL full-data three-track 60-pass ensemble-compression plan](docs/plans/HCWDL_MHPE_THREE_TRACK_60E_FULL_IMPLEMENTATION_PLAN.md)
- [HCWDL TRI60 scientific strategy explainer](docs/HCWDL_MHPE_TRI60_STRATEGY_EXPLAINER.md)
- [HCWDL full-data three-track 60-pass contracts](docs/contracts/HCWDL_MHPE_THREE_TRACK_60E_FULL.md)
- [HCWDL full-data three-track 60-pass runbook](docs/HCWDL_MHPE_THREE_TRACK_60E_FULL_RUNBOOK.md)
- [HCWDL TRI100 four-spine LOGIT plan](docs/plans/HCWDL_TRI100_FOUR_SPINE_LOGIT_IMPLEMENTATION_PLAN.md)
- [HCWDL TRI100 four-spine LOGIT contract](docs/contracts/HCWDL_TRI100_FOUR_SPINE_LOGIT.md)
- [HCWDL TRI100 four-spine LOGIT runbook](docs/HCWDL_TRI100_FOUR_SPINE_RUNBOOK.md)
- [HCWDL TRI100 full-cardinality bottleneck-matching plan](docs/plans/HCWDL_TRI100_FOUR_SPINE_FULL_CARDINALITY_BOTTLENECK_MATCHING_IMPLEMENTATION_PLAN.md)
- [HCWDL TRI100 full-cardinality salience-matching plan](docs/plans/HCWDL_TRI100_FOUR_SPINE_FULL_CARDINALITY_SALIENCE_MATCHING_IMPLEMENTATION_PLAN.md)
- [HCWDL full-cardinality salience-matching contracts](docs/contracts/HCWDL_FULLCARD_SALIENCE_MATCHING.md)
- [HCWDL full-cardinality salience-matching scientific and code guide](docs/HCWDL_FULLCARD_SALIENCE_MATCHING_CODE_GUIDE.md)
- [JetClass2 Delphes 500k salience persistent-HLT three-spine plan](docs/plans/JETCLASS2_DELPHES_FULLCARD_SALIENCE_PERSISTENT_500K_PLAN.md)
- [JetClass2 CMS-proxy Oscar dual-slot recovery plan](docs/plans/JETCLASS2_CMS_PROXY_OSCAR_DUAL_SLOT_RECOVERY_PLAN.md)
- [JetClass2 CMS-proxy Oscar 100k DIRECT+COARSE plan](docs/plans/JETCLASS2_CMS_PROXY_OSCAR_100K_DIRECT_COARSE_PLAN.md)
- [JetClass2 Delphes salience persistent-HLT contracts](docs/contracts/JETCLASS2_DELPHES_FULLCARD_SALIENCE_PERSISTENT.md)
- [JetClass2 Delphes 500k salience learned-fusion handoff plan](docs/plans/JETCLASS2_DELPHES_SALIENCE_LEARNED_HANDOFF_500K_PLAN.md)
- [JetClass2 Delphes salience learned-fusion handoff contracts](docs/contracts/JETCLASS2_DELPHES_SALIENCE_LEARNED_HANDOFF.md)
- [JetClass2 Delphes 500k salience MT20 three-spine plan](docs/plans/JETCLASS2_DELPHES_SALIENCE_MT20_500K_PLAN.md)
- [JetClass2 Delphes salience MT20 contracts](docs/contracts/JETCLASS2_DELPHES_SALIENCE_MT20.md)
- [JetClass2 D033-only C25/P75 endpoint-ablation plan](docs/plans/JETCLASS2_DELPHES_D000_D033_ONLY_C25P75_500K_PLAN.md)
- [JetClass2 D033-only C25/P75 endpoint-ablation contract](docs/contracts/JETCLASS2_DELPHES_D000_D033_ONLY_C25P75.md)
- [HCWDL adjacent-view fusion-handoff ladder plan](docs/plans/HCWDL_ADJACENT_VIEW_FUSION_HANDOFF_LADDERS_PLAN.md)
- [HCWDL adjacent output-fusion handoff contract](docs/contracts/HCWDL_ADJACENT_OUTPUT_FUSION_HANDOFF.md)
- [HCWDL TRI100 full-cardinality bottleneck-matching runbook](docs/HCWDL_TRI100_FOUR_SPINE_FULLCARD_BOTTLENECK_RUNBOOK.md)
- [HCWDL TRI60 M1 greedy-ensemble diagnostic plan](docs/plans/HCWDL_TRI60_M1_GREEDY_ENSEMBLE_PLAN.md)
- [HCWDL TRI60 M1 greedy-ensemble contracts](docs/contracts/HCWDL_TRI60_M1_GREEDY_ENSEMBLE.md)
- [HCWDL TRI60 M1 greedy-ensemble runbook](docs/HCWDL_TRI60_M1_GREEDY_ENSEMBLE_RUNBOOK.md)
- [HCWDL TRI60 full-data CE60 control plan](docs/plans/HCWDL_MHPE_TRI60_CE60_CONTROL_PLAN.md)
- [HCWDL TRI60 full-data CE60 control contract](docs/contracts/HCWDL_MHPE_TRI60_CE60_CONTROL.md)
- [HCWDL TRI60 full-data CE60 control runbook](docs/HCWDL_MHPE_TRI60_CE60_CONTROL_RUNBOOK.md)
- [HCWDL-MHPE C10P90 parallel plan](docs/plans/HCWDL_MHPE_C10P90_PARALLEL_PLAN.md)
- [HCWDL multi-horizon projection-ensemble contracts](docs/contracts/HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE.md)
- [HCWDL multi-horizon projection-ensemble runbook](docs/HCWDL_MHPE_RUNBOOK.md)
- [HCWDL-MHPE endpoint teacher-mixture plan](docs/plans/HCWDL_MHPE_ENDPOINT_MIX_300K60_PLAN.md)
- [HCWDL-MHPE endpoint teacher-mixture contract](docs/contracts/HCWDL_MHPE_ENDPOINT_MIX.md)
- [HCWDL-MHPE endpoint teacher-mixture runbook](docs/HCWDL_MHPE_ENDPOINT_MIX_RUNBOOK.md)
- [HCWDL-MHPE refined-continuation plan](docs/plans/HCWDL_MHPE_REFINED_CONTINUATION_300K60_PLAN.md)
- [HCWDL-MHPE refined-continuation contract](docs/contracts/HCWDL_MHPE_REFINED_CONTINUATION.md)
- [HCWDL-MHPE refined-continuation runbook](docs/HCWDL_MHPE_REFINED_CONTINUATION_RUNBOOK.md)
- [HCWDL-MHPE C10P90 parallel runbook](docs/HCWDL_MHPE_C10P90_RUNBOOK.md)
- [HCWDL-MHPE paired 300k/60-pass plan](docs/plans/HCWDL_MHPE_300K_60E_PAIRED_PLAN.md)
- [HCWDL-MHPE paired 300k/60-pass runbook](docs/HCWDL_MHPE_300K_60E_RUNBOOK.md)
- [HCWDL-MHPE D066 full-data schedule-screen plan](docs/plans/HCWDL_MHPE_D066_SCHEDULE_SCREEN_FULL_PLAN.md)
- [HCWDL-MHPE D066 schedule-screen contract](docs/contracts/HCWDL_MHPE_D066_SCHEDULE_SCREEN.md)
- [HCWDL-MHPE D066 full-data schedule-screen runbook](docs/HCWDL_MHPE_D066_SCHEDULE_SCREEN_FULL_RUNBOOK.md)
- [HCWDL-MHPE D000 teacher-distance screen plan](docs/plans/HCWDL_MHPE_D000_TEACHER_DISTANCE_SCHEDULE_SCREEN_FULL_PLAN.md)
- [HCWDL-MHPE D000 teacher-distance screen contract](docs/contracts/HCWDL_MHPE_D000_TEACHER_DISTANCE_SCHEDULE_SCREEN.md)
- [HCWDL-MHPE D000 teacher-distance screen runbook](docs/HCWDL_MHPE_D000_TEACHER_DISTANCE_SCREEN_RUNBOOK.md)
- [HCWDL architecture–input factorial plan](docs/plans/HCWDL_ARCHITECTURE_INPUT_FACTORIAL_PLAN.md)
- [HCWDL architecture–input factorial contract](docs/contracts/HCWDL_ARCHITECTURE_INPUT_FACTORIAL.md)
- [HCWDL interrupted final-evaluation recovery](docs/HCWDL_FINAL_RECOVERY_RUNBOOK.md)
- [HCWDL final-recovery contract](docs/contracts/HCWDL_FINAL_EVALUATION_RECOVERY.md)
- [HCWDL matching-free representation-KD ascent plan](docs/plans/HCWDL_MATCHING_FREE_REPRESENTATION_KD_ASCENTS.md)
- [Tigris miniature acceptance procedure](docs/TIGRIS_ACCEPTANCE.md)
- [JetClass and HLT data contract](docs/DATA_CONTRACT.md)
- [Cross-campaign experiment contract](docs/EXPERIMENT_CONTRACT.md)
- [Testing and acceptance ladder](docs/TESTING.md)
- [Legacy donor-source map](docs/LEGACY_SOURCE_MAP.md)
- [Architecture decisions](docs/decisions/README.md)
- [Campaign implementation plans](docs/plans/README.md)

## Repository layout

```text
configs/                  versioned site and experiment configuration
docs/                     contracts, handoff, plans, and decisions
scripts/                  thin Python command-line entry points
sbatch/                   thin Tigris workers and submitters
src/hlt_classification/   reusable scientific and operational package
tests/                    local contracts and scientific regressions
```

## Local bootstrap

Python 3.10 or newer is required. The data and HLT-v3 layers use NumPy,
Awkward, and Uproot. The model extra adds PyTorch and Weaver.

```powershell
python -m pip install -e .
python -m pytest -q
```

For model work, use `python -m pip install -e ".[model]"` or the established
Tigris environment.

The existing Windows scientific environment may also be used explicitly:

```powershell
C:\Users\22rya\miniconda3\envs\tagging-hlt\python.exe -m pytest -q
```

## Tigris activation

```bash
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
source /home/ryreu/miniforge3-aarch64/etc/profile.d/conda.sh
conda activate atlas_kd_tigris
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
cd /home/ryreu/atlas/HLT_Classification
python -s -c 'import sys; assert sys.version_info[:2] == (3, 10); print(sys.executable)'
```

Create a clean-source smoke specification and inspect or simulate it without
Slurm mutation:

```bash
python -s scripts/create_campaign.py \
  --mode smoke \
  --output checkpoints/specs/hlt_part_smoke.json
bash sbatch/submit_baseline.sh \
  --campaign-spec checkpoints/specs/hlt_part_smoke.json \
  --dry-run
bash sbatch/submit_baseline.sh \
  --campaign-spec checkpoints/specs/hlt_part_smoke.json \
  --smoke-simulate \
  --simulate-failure-task hlt_cache
```

Use `--smoke-submit` only for the real miniature. Full production is separately
gated by an explicitly authorized production specification and an
authenticated storage measurement.

The repository currently has no Git remote. Transfer an exact commit to
Tigris (or configure a remote) before creating a campaign specification; the
source contract rejects dirty or nonidentical worktrees.

The additive [dense anchor-50 MHPE plan](docs/plans/HCWDL_MHPE_DENSE_ANCHOR50_300K60_PLAN.md)
and [runbook](docs/HCWDL_MHPE_DENSE_ANCHOR50_300K60_RUNBOOK.md) define the
queue-ready 300k/60-pass, 29-fit predecessor-anchored ensemble campaign.
The paired [dense C25P75 plan](docs/plans/HCWDL_MHPE_DENSE_C25P75_300K60_PLAN.md)
and [runbook](docs/HCWDL_MHPE_DENSE_C25P75_300K60_RUNBOOK.md) hold that graph
fixed while changing specialist supervision from 10/90 to 25/75 CE/KD.

The available data commands include:

```bash
python -s scripts/preflight_data.py
python -s scripts/build_splits.py --output artifacts/data/split_manifest.json.gz
python -s scripts/build_offline_cache.py --help
python -s scripts/build_hlt_cache.py --help
python -s scripts/audit_hlt_cache.py --help
```

They default to the canonical Tigris data root. `build_splits.py` always fails
closed on unreadable or schema-incomplete ROOT files.

## Attribution

The baseline uses the reference Particle Transformer implementation supplied
by Weaver through its public Python interface; no Weaver source is copied into
this repository. JetClass, Particle Transformer, and Weaver citations and
licenses must be included in any scientific release.
