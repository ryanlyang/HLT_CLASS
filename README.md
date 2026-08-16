# HLT Classification

A clean, standalone research repository for HLT-only JetClass classification
and training-time use of paired offline information.

## Current status

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
- [HCWDL-MHPE C10P90 parallel plan](docs/plans/HCWDL_MHPE_C10P90_PARALLEL_PLAN.md)
- [HCWDL multi-horizon projection-ensemble contracts](docs/contracts/HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE.md)
- [HCWDL multi-horizon projection-ensemble runbook](docs/HCWDL_MHPE_RUNBOOK.md)
- [HCWDL-MHPE C10P90 parallel runbook](docs/HCWDL_MHPE_C10P90_RUNBOOK.md)
- [HCWDL-MHPE paired 300k/60-pass plan](docs/plans/HCWDL_MHPE_300K_60E_PAIRED_PLAN.md)
- [HCWDL-MHPE paired 300k/60-pass runbook](docs/HCWDL_MHPE_300K_60E_RUNBOOK.md)
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
