# HLT Classification Repository Transfer Plan

## 1. Purpose

Build a clean, standalone repository at:

```text
local:   C:\Users\22rya\ComputerScience\CERN\HLT_Classification
Tigris:  /home/ryreu/atlas/HLT_Classification
```

The new repository is the common foundation for HLT-only JetClass
classification research. Its first major campaign will be HLT Offline
Structure Distillation (HOSD), but the foundation must remain useful for other
HLT classifiers and training-time offline supervision.

The new repository must not import `Fresh_check` at runtime. Selected,
well-tested scientific logic may be migrated with recorded provenance; old
campaign orchestration and historical clutter must not be copied.

## 2. Permanent scientific and operational rules

- Deployable models consume HLT inputs only.
- Offline information is permitted only through declared training targets,
  teachers, controls, or oracle diagnostics.
- `fixed_hlt_v3_track_dominant_proxy` is a controlled HLT-like proxy, not real
  detector HLT.
- All compared models use the same authenticated HLT views.
- Final-test inference remains sealed until the campaign's finalist and
  execution locks exist.
- Poor scientific performance never fails a job, cancels later experiments,
  or skips registered rows.
- Invalid inputs, nonfinite execution, forbidden data access, stale source,
  corrupt artifacts, and lineage mismatches fail closed.
- Production artifacts are content authenticated and atomically published.
  Path existence alone never authorizes reuse.
- Production runs use clean, committed source and must pass a genuine Tigris
  miniature first.

## 3. Transfer blocks

Implement these blocks in order. Each block ends with focused tests and an
updated `docs/HANDOFF.md`.

### Block 1 — Repository scaffold and agent documentation

Create:

```text
AGENTS.md
README.md
pyproject.toml
.gitignore
configs/
docs/
scripts/
sbatch/
src/hlt_classification/
tests/
```

Write:

```text
docs/HANDOFF.md
docs/RESEARCH_COMPUTE_RUNBOOK.md
docs/DATA_CONTRACT.md
docs/EXPERIMENT_CONTRACT.md
docs/TESTING.md
docs/LEGACY_SOURCE_MAP.md
docs/decisions/
docs/plans/
```

Freeze the documentation authority order:

1. active scientific implementation plan;
2. versioned reusable contracts;
3. immutable campaign specification;
4. `AGENTS.md`;
5. living status in `docs/HANDOFF.md`.

Done when the package imports, documentation links resolve, local test
discovery works, and an isolated agent can identify the mission, active task,
test commands, and Tigris rules without reading `Fresh_check`.

### Block 2 — JetClass data foundation

Migrate and cleanly separate the proven logic from:

```text
jetclass_fresh/jetclass_data.py
scripts/build_jetclass_splits.py
scripts/preflight_relational_part_data.py
```

Implement:

```text
src/hlt_classification/data/schema.py
src/hlt_classification/data/identity.py
src/hlt_classification/data/root_reader.py
src/hlt_classification/data/splits.py
scripts/preflight_data.py
scripts/build_splits.py
```

Freeze:

- recursive data root `/home/ryreu/atlas/PracticeTagging/data`;
- ROOT tree `tree`;
- ten-class order and longest-prefix filename mapping;
- exact 14-channel raw constituent schema;
- maximum 128 constituents;
- canonical jet identity;
- balanced sampling without replacement;
- deterministic global row shuffle after classwise sampling;
- split disjointness and capacity audits;
- bounded ROOT-open retries and fail-closed unreadable production files.

Done when synthetic ROOT fixtures and manifest tests prove schema, identity,
balance, deterministic ordering, disjointness, hashing, and chunked reads.

### Block 3 — HLT-v3 degradation kernel

Migrate the scientific kernel from:

```text
teacher_logit_reco/relation_expert_token_bridge/hlt_v3.py
teacher_logit_reco/relation_expert_token_bridge/replicas.py
jetclass_fixed_hlt.py
```

Implement it under:

```text
src/hlt_classification/data/hlt_v3.py
src/hlt_classification/data/replicas.py
```

Preserve the registered `fixed_hlt_v3_track_dominant_proxy/v1` semantics:

- exact strength-zero identity;
- deterministic identity/role/replica random streams;
- type-aware constituent loss and neutral merging;
- mild type-dependent kinematic response;
- PID and charge consistency;
- track-measurement loss;
- correlated `d0`/`dz` response, uncertainty inflation, and tails;
- mass-preserving energy recomputation;
- stable degraded-`pT` ordering;
- explicit measurement-validity states.

Record the donor commit, original paths, semantic changes, and parity evidence
in `docs/LEGACY_SOURCE_MAP.md`.

Done when focused tests prove hand-calculated scaling, identity, determinism,
replica behavior, merge rules, mass preservation, track/PID/charge validity,
stable ties, field-isolation profiles, and finite outputs.

### Block 4 — Sharded authenticated offline and HLT caches

Do not copy either legacy monolithic cache implementation unchanged.

Implement:

```text
src/hlt_classification/data/cache_contracts.py
src/hlt_classification/data/offline_cache.py
src/hlt_classification/data/hlt_cache.py
src/hlt_classification/data/dataset.py
scripts/build_offline_cache.py
scripts/build_hlt_cache.py
scripts/audit_hlt_cache.py
```

Requirements:

- bounded shards with ordered identities;
- streaming readers rather than whole-campaign RAM loading;
- per-shard SHA-256 and immutable manifest;
- source, split, schema, degradation, replica, and generator lineage;
- atomic shard publication;
- exact validation before reuse;
- interrupted-run resume from authenticated completed shards;
- no degradation construction indices in deployable datasets;
- useful progress output and diagnostics.

Done when an interrupted synthetic build resumes byte-identically, corrupted
or cross-profile shards are rejected, and batch/shard layout does not change
HLT output bytes.

### Block 5 — Canonical ParT inputs and real-Weaver baseline

Migrate and simplify the proven surfaces from:

```text
jetclass_fresh/part_inputs.py
jetclass_fresh/hlt_baseline.py
```

Implement:

```text
src/hlt_classification/data/part_inputs.py
src/hlt_classification/models/particle_transformer.py
scripts/validate_weaver_parity.py
```

Freeze the canonical baseline:

```text
17 particle features
Weaver standard-four pair inputs
embed_dims = [128, 512, 128]
pair_embed_dims = [64, 64, 64]
8 attention heads
8 particle blocks
2 class-attention blocks
GELU
10 output classes
trim enabled
from-scratch initialization unless explicitly declared otherwise
```

All axes and derived kinematics must be rebuilt from the supplied view only.
Retain deterministic safe handling for all-empty degraded jets.

Done when the installed-Weaver authoritative FP32 test establishes logits,
input gradients, parameter gradients, masks, and state-dictionary parity with
mixed precision disabled.

### Block 6 — Baseline training, inference, and metrics

Implement:

```text
src/hlt_classification/training/engine.py
src/hlt_classification/training/checkpoints.py
src/hlt_classification/evaluation/inference.py
src/hlt_classification/evaluation/metrics.py
scripts/train_part.py
scripts/evaluate_part.py
```

Requirements:

- sharded streaming and deterministic epoch ordering;
- fixed configured training budget;
- exact resume of model, optimizer, scheduler, scaler, sampler, replica cycle,
  epoch/update count, and RNG state;
- `last.pt` plus deterministically selected validation checkpoint;
- checkpoint/cache/config/source hash binding;
- no silent skipping of nonfinite training batches;
- no performance-driven cancellation of other runs;
- standalone inference artifacts with ordered identities and logits;
- accuracy, CE, per-class efficiency, OVR AUC, Brier score, exact 15-bin
  top-label ECE, and locked QCD-versus-signal rejection metrics;
- deterministic paired statistics where comparisons require them.

Done when uninterrupted and resumed miniature training are equivalent and an
intentionally poor model still completes, evaluates, and exits successfully.

### Block 7 — Compact campaign, provenance, and Tigris execution layer

Implement:

```text
src/hlt_classification/contracts/
src/hlt_classification/provenance.py
src/hlt_classification/campaign.py
scripts/measure_storage.py
scripts/monitor_campaign.py
scripts/resume_campaign.py
sbatch/common.sh
sbatch/run_preflight.sh
sbatch/run_build_splits.sh
sbatch/run_build_hlt_cache.sh
sbatch/run_train_part.sh
sbatch/run_evaluate_part.sh
sbatch/submit_baseline.sh
```

Central Tigris defaults:

```text
PROJECT_DIR=/home/ryreu/atlas/HLT_Classification
DATA_DIR=/home/ryreu/atlas/PracticeTagging/data
OUTPUT_ROOT=/home/ryreu/atlas/HLT_Classification/checkpoints
CONDA_BASE=/home/ryreu/miniforge3-aarch64
CONDA_ENV=atlas_kd_tigris
SBATCH_ACCOUNT=reu-aisocial
SBATCH_PARTITION=tigris
GPU_GRES=gpu:gh200:1
```

Every worker must:

```bash
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
source "${PROJECT_DIR}/sbatch/common.sh"
```

Never locate helpers relative to Slurm's spool copy. Use `sbatch --parsable`,
exact dependency IDs, bounded arrays, campaign-local logs and ledgers, measured
resources, authenticated reuse, and explicit resume plans. Do not repair a
failed graph with broad `scontrol release` or ambiguous job-name cancellation.

Expose:

```text
--dry-run
--smoke-simulate
--smoke-submit
full production submission
```

Done when shell/static tests, storage preflight, source-drift rejection,
all-reused restart validation, failure injection, and dry-run nonmutation pass.

### Block 8 — Real miniature acceptance and HOSD handoff

Run a genuine Tigris miniature through:

```text
data preflight
split construction
offline/HLT shard construction and audit
real-Weaver parity
GPU forward/backward
baseline ParT training
resume/restart
inference
metrics and report
```

Then:

- record the campaign root, commit, environment, job IDs, resource use, and
  results in `docs/HANDOFF.md`;
- copy the HOSD implementation plan into `docs/plans/`;
- revise its old repository paths and RETB parent references to the new local
  contracts without changing scientific meaning;
- explicitly list which HOSD components remain unimplemented.

Done when the miniature completes from production workers with no manually
injected artifacts, the full baseline graph resolves in dry-run mode, and the
next agent has one unambiguous HOSD implementation task.

## 4. Items deliberately excluded

Do not transfer:

- checkpoints, caches, predictions, plots, logs, or Slurm ledgers;
- old run registries and campaign IDs;
- unrelated particle-view, residual-field, fusion, RPT, RETB, or adaptive
  campaign packages;
- hundreds of experiment-specific `scripts/` and `sbatch/` files;
- stale handoffs or chat histories;
- old Git metadata or commit history as the new repository's history;
- code whose only justification is that another old module imports it.

## 5. Per-block handoff format

After every block, report:

- files added or changed;
- donor files and recorded donor commit;
- contracts added or versioned;
- local tests and exact results;
- real-Weaver or Tigris validation, if applicable;
- remaining blockers for the next block.

Do not mark a block complete because its files exist. Mark it complete only
when its stated completion evidence passes.
