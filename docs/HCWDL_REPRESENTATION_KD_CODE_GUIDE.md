# HCWDL Representation-KD: Code, Design, and Execution Guide

This is the reviewer map for the current HCWDL representation-distillation
implementation. It is intentionally detailed. A new engineer should be able
to use it to locate the scientific code, understand both losses, follow one
training example end to end, inspect a generated campaign, and run the setup
without first reconstructing the repository's history.

The two representation-KD methods are:

- **RSET**: `HCWDL_REP_SET/v1`, which distills jet and token-set structure;
- **RREL**: `HCWDL_REP_REL/v1`, which distills jet and token-set structure plus
  stratified token-relation geometry.

They run as four downward tracks:

```text
TOFF -> RSET_D100 -> RSET_D95c -> RSET_D90c -> ... -> RSET_D5c -> RSET_D0c -> RSET_M1c
                  -> RSET_D95w -> RSET_D90w -> ... -> RSET_D5w -> RSET_D0w -> RSET_M1w

TOFF -> RREL_D100 -> RREL_D95c -> RREL_D90c -> ... -> RREL_D5c -> RREL_D0c -> RREL_M1c
                  -> RREL_D95w -> RREL_D90w -> ... -> RREL_D5w -> RREL_D0w -> RREL_M1w
```

There are two separately trained D100 roots because RSET and RREL optimize
different objectives. Each root splits into a cold and warm track at D95.
Every child uses the immediate richer predecessor as its only source of
distillation logits and representation targets. All four tracks stop at M1.

D100 through D5 are nondeployable transfer intermediates. D0 and M1 are
HLT-only deployable models. The four M1 models are the primary results.

## 1. Authority and historical naming

Before changing the code, read:

1. [`docs/plans/HCWDL_MATCHING_FREE_REPRESENTATION_KD_ASCENTS.md`](plans/HCWDL_MATCHING_FREE_REPRESENTATION_KD_ASCENTS.md)
   â€” the active scientific plan. Its controlling dense-descent amendment near
   the beginning supersedes retained historical ascent sections later in the
   same document.
2. [`src/hlt_classification/scouting/hcwdl_representation_contracts.py`](../src/hlt_classification/scouting/hcwdl_representation_contracts.py)
   â€” the central durable contract/schema registry.
3. [`src/hlt_classification/scouting/hcwdl_representation_layout.py`](../src/hlt_classification/scouting/hcwdl_representation_layout.py)
   â€” the canonical route and validator for every artifact.
4. The immutable `campaign_spec.json`, runtime binding, command plan, and
   submission ledger for the particular execution being reviewed.
5. [`docs/HANDOFF.md`](HANDOFF.md) â€” current implementation/execution status.
6. [`docs/HCWDL_RKD_RUNBOOK.md`](HCWDL_RKD_RUNBOOK.md) and
   [`scripts/README.md`](../scripts/README.md) â€” operator workflows.

Several compatibility names still say `ascent` even though the active graph
is a descent. In particular, the graph artifact is still published at
`graph/ascent_graph.json`, and the validators are still named
`validate_ascent_graph*`. The contract and content define a dense descent.

Do not confuse this implementation with:

- the superseded M1-to-M6 representation ascent;
- the retired M5 controls;
- `RepresentationScoutingParticleTransformer` in
  `models/scouting_particle_transformer.py`, an older R1-R5 experiment;
- the bounded non-final D0/M1 acceptance-action machinery;
- shared-final prediction/label-access code.

Those surfaces remain in the repository for history or other campaigns. The
four-track graph described here uses the dense graph, dense preparation, and
dense campaign paths named below.

## 2. Naming and mental model

### 2.1 Node names

Most node names have this form:

```text
<strategy>_<view><initialization>
```

Examples:

- `RSET_D75c`: set loss, D75 view, cold student;
- `RREL_D20w`: relation-aware loss, D20 view, warm student;
- `RSET_M1c`: set loss, HLT-only M1 terminal, cold student.

The two roots, `RSET_D100` and `RREL_D100`, have no suffix because each is one
shared fresh root. The `c` and `w` branches begin at D95.

### 2.2 Cold and warm change initialization, not supervision

Cold means:

- create a fresh deployable HLT backbone at the rung;
- load no predecessor model state.

Warm means:

- load only the exact immediate predecessor's authenticated deployable model
  state.

Both receive the same kind of supervision from their own immediate richer
predecessor. Warm does **not** carry optimizer, scheduler, gradient scaler,
representation heads, or old target-bank state. The optimizer/scheduler and
representation heads are reset at every node.

For example, `RREL_D40w` uses `RREL_D45w` for:

- the compact target bank;
- teacher logits;
- representation targets;
- warm deployable-model initialization.

`RREL_D40c` uses `RREL_D45c` for the first three but initializes a fresh model.

### 2.3 Teacher, target bank, and checkpoint are distinct

- The **teacher model** performs the source forward pass.
- The **target bank** is the compact, immutable output of those forward
  passes over authenticated training rows.
- The **selected checkpoint** is a trained student's selected full state.
- The **deployable extraction** is the HLT-only model state used for warm
  initialization and eventual inference.

Keeping these identities separate is a central lineage rule.

## 3. Exact graph implementation

The graph lives in
[`hcwdl_representation_graph.py`](../src/hlt_classification/scouting/hcwdl_representation_graph.py).

Important symbols:

- `RSET_STRATEGY = "HCWDL_REP_SET/v1"`;
- `RREL_STRATEGY = "HCWDL_REP_REL/v1"`;
- `TRACKS = ("cold", "warm")`;
- `DESCENT_LEVELS = (100, 95, ..., 0)`;
- `TRACKED_LEVELS = (95, 90, ..., 0)`;
- `RepresentationNodeSpec` â€” one complete node description;
- `_build_node_registry()` â€” constructs the legal nodes/edges;
- `NODE_REGISTRY` â€” the frozen 86-node graph;
- `CONTROL_REGISTRY` â€” empty for the dense graph;
- `validate_ascent_graph()` â€” in-memory validation;
- `ascent_graph_artifact()` / `validate_ascent_graph_artifact()` â€” durable
  `HCWDL_REPRESENTATION_DENSE_DESCENT_GRAPH/v1` boundaries.

The node count is exact:

```text
2 strategy-specific D100 roots
+ 2 strategies * 2 tracks * (20 D95...D0 nodes + 1 M1 node)
= 86 nodes
```

The registry freezes, rather than infers from strings:

- strategy and step;
- view identity;
- initialization policy and parent;
- sole logit/representation teacher;
- target-bank identity;
- deployable/HLT-only status;
- explicit graph stage. The campaign derives the four terminal M1 endpoints
  from `stage == "terminal_m1"`.

An unregistered node, skipped rung, cross-strategy parent, extra teacher, or
wrong warm source fails validation.

## 4. Common training objective

Both strategies use:

```text
L_total = L_base + rho * (L_representation + lambda_orth * L_orthogonality)

rho         = 0.10
lambda_orth = 0.001 while the jet/set schedule is active
```

The base objective is:

```text
L_base = 0.25 * cross_entropy(labels)
       + 0.75 * KD(immediate_predecessor_logits)
```

There is no simultaneous second predecessor and no privileged-logit term.
The TOFF bank teaches the two D100 roots. Every later bank is generated by the
immediate richer representation student.

The main implementation split is:

- [`hcwdl_representation_losses.py`](../src/hlt_classification/scouting/hcwdl_representation_losses.py)
  â€” tensor mathematics;
- [`hcwdl_representation_kernels.py`](../src/hlt_classification/scouting/hcwdl_representation_kernels.py)
  â€” fixed random spectral features and MMD;
- [`hcwdl_representation_recipe.py`](../src/hlt_classification/scouting/hcwdl_representation_recipe.py)
  â€” frozen coefficients, schedules, taps, and lifecycle;
- [`hcwdl_representation_training.py`](../src/hlt_classification/scouting/hcwdl_representation_training.py)
  â€” base-loss assembly, representation-loss assembly, and training loop.

## 5. RSET: jet and set representation KD

RSET's full-strength representation mixture is:

```text
L_RSET = 0.40 * L_jet + 0.60 * L_set
```

The whole mixture is multiplied by `rho = 0.10` before being added to the base
loss.

### 5.1 Jet loss

`jet_representation_loss()` in
[`hcwdl_representation_losses.py`](../src/hlt_classification/scouting/hcwdl_representation_losses.py)
uses:

- a projected, layer-normalized direct cosine term; and
- a within-batch Gram-geometry mismatch on raw jet states.

The combination is `0.75 * direct + 0.25 * gram`. The Gram term preserves
batch geometry rather than aligning each jet independently only.

### 5.2 Token weights

`classify_hlt_token_families()` derives explicit charged/neutral family state
from HLT charge and PID information. Contradictions and malformed states are
not silently repaired.

`token_weights()` gives visible tokens:

```text
0.5 / N + 0.5 * sqrt(pT_i) / sum_j sqrt(pT_j)
```

so every visible token receives mass while higher-pT tokens receive softened
additional emphasis.

### 5.3 Matching-free set loss

The important functions are:

- `build_ordinary_token_targets()`;
- `ordinary_set_representation_loss()`;
- `build_native_offline_token_targets()`;
- `native_offline_set_representation_loss()`.

The loss does not match particle index `i` between views. It summarizes each
weighted contextual-token set with deterministic finite random features and
compares the fixed-dimensional mean embeddings. The token bandwidths are
`0.10`, `0.25`, `0.50`, and `1.00`, with 256 features each (1,024 total).

Ordinary predecessor models use one HLT token-family path. The native-offline
TOFF root target uses separate charged and neutral heads and an equal
active-family reduction.

`resolve_node_execution()` in `hcwdl_representation_training.py` activates
only `jet` and `set` for RSET nodes.

## 6. RREL: jet, set, and relation KD

RREL uses the RSET components plus relation sketches. At full strength:

```text
L_RREL = 0.30 * L_jet + 0.45 * L_set + 0.25 * L_relation
```

This is a reallocation, not `0.40 + 0.60 + 0.25`. With jet/set ramp `r_js`
and relation ramp `r_rel`:

```text
common_weight = r_js - 0.25 * r_rel
jet_weight    = 0.40 * common_weight
set_weight    = 0.60 * common_weight
relation      = 0.25 * r_rel
```

### 6.1 Relation construction

The main functions are:

- `build_student_relation_sketches()`;
- `build_teacher_relation_targets()`;
- `relation_representation_loss()`.

For every jet, the code:

1. independently ranks visible tokens by descending pT with canonical-ID
   tie-breaking;
2. keeps the top 32;
3. forms unordered, off-diagonal pairs;
4. divides pairs into own-view physical deltaR strata: `<0.05`,
   `[0.05, 0.20)`, and `>=0.20`;
5. computes cosine geometry from raw normalized block-2 contextual states;
6. weights pairs by token-weight products;
7. requires at least four pairs and FP64 effective sample size at least three;
8. maps the statistic through four bandwidths with 64 features each (256 per
   stratum);
9. compares equal-weight active strata/families.

It remains matching-free: teacher and student build their summaries
independently. No cross-view particle-index map is constructed.

### 6.2 Raw-state relation contract

The active plan, implementation, and tests use cosine geometry from **raw
FP32 L2-normalized block-2 states**. `HCWDL_REPRESENTATION_RECIPE/v5` now binds
that choice explicitly through `latent_state_source`,
`latent_projection_applied`, `latent_normalization`, and `latent_statistic`.
The set projection is not part of the relation objective. The superseded v4
recipe metadata said projected normalized states and is intentionally rejected;
v4 artifacts may not be relabeled as v5.

## 7. Ramps, calibration, and numerical behavior

The recipe freezes:

- jet/set zero through pass 2, full at pass 6;
- relation zero through pass 4, full at pass 8;
- 4,096 train-only calibration rows in 16 batches of 256;
- jet/set calibration after pass 2;
- RREL relation calibration after pass 4;
- gradient-RMS scale bounds `[1e-4, 1e4]`;
- BF16 student forward and FP32 teacher/loss boundaries;
- 60 scientific passes and validation after every complete pass;
- no performance-based early stopping.

[`hcwdl_representation_calibration.py`](../src/hlt_classification/scouting/hcwdl_representation_calibration.py)
implements deterministic selection, early-backbone parameter selection,
gradient-RMS measurement, state/RNG restoration, and the no-optimizer-step
calibration boundary.

`projection_orthogonality()` and projection diagnostics are in
`hcwdl_representation_losses.py`. Required nonfinite values fail. Poor but
finite conditioning/metrics are reported as scientific results rather than
used to cancel later registered experiments.

## 8. Model surfaces and deployable extraction

### 8.1 Particle Transformer taps

[`models/scouting_particle_transformer.py`](../src/hlt_classification/models/scouting_particle_transformer.py)
contains:

- `HCWDLScoutingSurfaces`;
- `HCWDLNativeOfflineSurfaces`;
- `_forward_hcwdl_weaver_surfaces()`;
- `ScoutingParticleTransformer.forward_hcwdl_surfaces()`;
- the native-offline charged/neutral surface path.

One forward captures contextual particle states after particle block 2 and the
pooled penultimate jet state. Integer identity/family metadata are used only
for trimming/classification and do not enter the learned embeddings.

[`models/hcwdl_surfaces.py`](../src/hlt_classification/models/hcwdl_surfaces.py)
owns tap schemas, runtime signatures, installed-Weaver parity, checkpoint
audits, and architecture attestation.

### 8.2 Training-only representation heads

[`models/hcwdl_representation.py`](../src/hlt_classification/models/hcwdl_representation.py)
defines:

- `HCWDLRepresentationHeads`: identity-initialized, bias-free 128x128 jet and
  token projections, with separate native charged/neutral token projections;
- `HCWDLRepresentationStudent`: the canonical HLT model plus training heads;
- `publish_hcwdl_deployable_extraction()`;
- the strict deployable checkpoint loader.

RREL has no separately deployed relation projection. It compares raw
contextual-token cosine geometry during training.

Deployable extraction strips every training-only head, reloads the canonical
HLT model strictly, and verifies logit parity. Offline inputs, target banks,
degradation identities, and projection heads cannot become deployable inputs.

## 9. Compact target banks

The target subsystem separates an expensive teacher forward from student
training.

### 9.1 Main files

- [`hcwdl_representation_targets.py`](../src/hlt_classification/scouting/hcwdl_representation_targets.py):
  logical banks, consumer registries, target-forward specs, build intents,
  crash-safe generations, immutable shards/manifests, bank loading, identity
  joins, and cleanup.
- [`hcwdl_representation_target_runtime.py`](../src/hlt_classification/scouting/hcwdl_representation_target_runtime.py):
  real teacher forwards and compact target construction.
- [`hcwdl_representation_target_planning.py`](../src/hlt_classification/scouting/hcwdl_representation_target_planning.py):
  pre-campaign freezing of every bank, generation, consumer, and forward spec.
- [`hcwdl_representation_target_recovery.py`](../src/hlt_classification/scouting/hcwdl_representation_target_recovery.py):
  cleanup/recovery rules.

### 9.2 Bank contents

An ordinary bank records compact, authenticated values such as:

- row identity and class;
- teacher logits (15 classes);
- jet representation (128);
- token spectral mean (1,024);
- relation means (three strata by 256);
- eligibility, token/pair counts, pT mass, ESS, and explicit inactive reasons.

The native TOFF bank has separate charged and neutral token/relation targets.
It does not keep the raw particle batch in the committed target bank.

`RepresentationTargetBank` validates the generation and joins training rows
by authenticated identity, not by incidental batch position. RSET does not
load relation arrays into its loss path. RREL does.

### 9.3 Generation flow

The one TOFF generation teaches both D100 roots. Each D100 result creates a
bank used by the cold and warm D95 children of the same strategy. Thereafter,
each branch has one consumer per bank.

There are 83 screen generations:

```text
1 TOFF bank
+ 2 strategy-specific D100 banks
+ 4 tracks * 20 later predecessor banks
= 83
```

The campaign serializes target waves. Each wave creates a generation, runs all
registered consumers, then publishes authenticated cleanup. This keeps at
most the planned live target footprint while preserving the full lineage.

## 10. Training path, step by step

The core is
[`hcwdl_representation_training.py`](../src/hlt_classification/scouting/hcwdl_representation_training.py).

### 10.1 Resolve the node

`NodeExecution` and `resolve_node_execution()` map a registered node to its
strategy, active loss components, teacher, target bank, initialization rule,
and deployment policy. The training path does not authorize a node because its
name happens to look valid.

### 10.2 Derive configuration

`representation_training_configuration()` reopens the exact authorized parent
`HCWDL_RECIPE/v4` and inherits batch size, optimizer, learning-rate schedule,
AMP choice, and other base training settings.

- `scientific` means exactly 60 complete passes;
- `smoke` means the production path bounded to exactly two optimizer updates;
- `synthetic_test` is a local, nonauthorizing fixture mode.

The learning-rate role is cold or warm child. The dense route never selects
an obsolete dual-teacher rate.

### 10.3 Initialize the student

`initialize_representation_student()`:

1. derives paired RNG streams;
2. constructs fresh deployable weights for cold nodes, or strictly reloads the
   immediate predecessor's deployable extraction for warm nodes;
3. creates fresh identity-initialized representation heads;
4. rejects warm state on a cold node or missing/wrong warm state on a warm
   node.

Paired RNG streams make the strategy/initialization comparisons deliberate.

### 10.4 Compute one training loss

At a high level, `compute_node_loss()` executes:

```text
student HLT forward
-> authenticated row-identity join to the one target bank
-> 0.25 label CE + 0.75 target-bank-logit KD
-> obtain student jet and token taps
-> compute jet/set losses
-> compute relation loss only for RREL
-> apply pass ramps and calibrated component scales
-> add rho-scaled representation and projection orthogonality
-> verify required values are finite
```

The target bank's logits pass through a historically named HLT-teacher field.
That does not introduce another teacher. A non-null second predecessor-logit
input is rejected for this graph.

### 10.5 Run the node

`train_hcwdl_representation_node()` owns:

- authenticated train/validation view caches;
- optimizer and scheduler;
- BF16/FP32 boundaries;
- calibration barriers;
- validation;
- immutable checkpoint and resume generations;
- exact preemption state;
- checkpoint selection;
- training report;
- HLT-only extraction.

Scientific selection orders candidates by validation AUC, CE, QCD rejection,
earliest update, then lexical checkpoint identity. Poor finite metrics remain
valid results. Invalid lineage, inputs, required numerics, or output bytes fail
closed.

`validate_representation_training_report()` is the main deep report boundary.

## 11. Real data views and production adapters

[`hcwdl_representation_data.py`](../src/hlt_classification/scouting/hcwdl_representation_data.py)
owns row identities, parent-batch conversion, and iteration surfaces.

[`hcwdl_representation_production.py`](../src/hlt_classification/scouting/hcwdl_representation_production.py)
is the bridge from immutable runtime assemblies into the scientific functions.
Review these entrypoints closely:

- `target_build_adapter()` authenticates teacher source, row selection,
  assignment, target-forward spec, kernel/tap resources, storage bounds, and
  output ownership before building a bank;
- `_teacher_surface_forward()` runs the teacher surface in FP32;
- `_build_training_view_caches()` builds D5-D100 Shell-Exact alpha views and
  HLT-only D0/M1 views from au…1324 tokens truncated…ing spec
  campaign_spec.json                   # authorized live spec, when created
  command_plan.json
  submission_ledger.json

  architecture/
    tap.json
    surface_parity.json

  import/
    dense_teacher_import.json
    dense_training_disposition.json

  graph/
    ascent_graph.json                  # compatibility filename; descent graph
  controls/
    registry.json                      # empty for this experiment
  recipes/
    representation_recipe.json
    kernel_resources/committed/...

  operator/
    runtime_facts.json
    runtime_signatures.json
    static_inputs.json
    artifact_content_hashes.json
    target_generations.json

  runtime/
    runtime_prerequisites.json
    task_rows.json
    runtime_binding.json
    dry_run_audit.json

  targets/<bank>/
    logical_bank.json
    generations/<generation_id>/
      consumer_registry.json
      target_forward_spec.json
      build_intent.json
      shards/...
      manifest.json
      generation.json

  training/<scope>/<node>/<execution>/
    calibration/...
    resume/...
    checkpoints/selected/committed/...
    checkpoints/final/committed/...
    training_report.json
    checkpoint_selection.json
    deployable_extraction.json

  cleanup/<bank>/<generation>/
    authorization.json
    completion.json

  reports/
    screen_aggregate.json
    dense_training_aggregate.json

  confirmation/                        # pilot only
    registry.json
    runs/...
    aggregate.json

  submission_events/...
  monitoring/reports/...
  recovery/...
```

File existence is never enough for reuse. Load the artifact, call its typed
validator, and verify expected parent/content/source bindings.

[`hcwdl_representation_artifacts.py`](../src/hlt_classification/scouting/hcwdl_representation_artifacts.py)
implements common immutable JSON/binary publication. Semantic contract version
(`/vN`) and common envelope `schema_version` are separate fields; do not assume
they always have the same number.

## 15. Reporting and endpoint selection

[`hcwdl_representation_reporting.py`](../src/hlt_classification/scouting/hcwdl_representation_reporting.py)
owns:

- `build_screen_aggregate()`;
- `build_dense_training_aggregate()`;
- dense disposition validation;
- the four-M1 confirmation registry, runs, and aggregate.

The screen aggregate must account for all registered graph nodes. The dense
aggregate checks the exact four-track inventory and endpoint set. It does not
drop a weak rung or branch. Poor performance is still a scientific result.

[`hcwdl_representation_dense_acceptance.py`](../src/hlt_classification/scouting/hcwdl_representation_dense_acceptance.py)
builds/validates the completed dense-smoke acceptance artifact after the exact
campaign output audit. It is not part of training itself.

## 16. Inspecting the graph from Python

With `PYTHONPATH=<checkout>/src`:

```python
from hlt_classification.scouting.hcwdl_representation_graph import NODE_REGISTRY
from hlt_classification.scouting.hcwdl_representation_campaign import (
    primary_node_ids,
    terminal_node_ids,
)

assert len(NODE_REGISTRY) == 86
print(len(primary_node_ids()))         # all 86 active non-control graph nodes
print(terminal_node_ids())             # the four M1 endpoints

for node_id, node in NODE_REGISTRY.items():
    print(
        node_id,
        node.strategy,
        node.initialization,
        node.initialization_parent,
        node.representation_logit_teacher,
        node.target_bank_identity,
        node.deployable,
        node.stage,
    )
```

Use registry fields instead of parsing names in production code.

Inspect the smoke task graph without scheduling anything:

```python
from hlt_classification.scouting.hcwdl_representation_campaign import (
    DENSE_TRAINING_DISPOSITION,
    build_task_registry,
)

tasks = build_task_registry(
    disposition=DENSE_TRAINING_DISPOSITION,
    final_source_partitions=0,
    combined_finalist_count=0,
    mode="smoke",
)
assert len(tasks) == 261
for task in tasks:
    print(task.task_key, task.kind, task.dependencies, task.graph_node)
```

## 17. Local review and testing

### 17.1 Environment

PowerShell:

```powershell
$env:PYTHONPATH = "$PWD/src"
$env:PYTHONNOUSERSITE = "1"
$env:PYTHONDONTWRITEBYTECODE = "1"
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = "1"
```

Tigris:

```bash
source /home/ryreu/miniforge3-aarch64/etc/profile.d/conda.sh
conda activate /home/ryreu/miniforge3-aarch64/envs/atlas_kd_tigris

export PROJECT_DIR=/home/ryreu/atlas/HLT_Classification_hcwdl_dense
export PYTHONPATH="${PROJECT_DIR}/src"
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONDONTWRITEBYTECODE=1
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
cd "${PROJECT_DIR}"
```

### 17.2 Scientific core

```bash
python -s -m pytest -q -p no:cacheprovider --assert=plain \
  tests/test_hcwdl_representation_graph.py \
  tests/test_hcwdl_representation_math.py \
  tests/test_hcwdl_representation_model.py \
  tests/test_hcwdl_representation_recipe.py \
  tests/test_hcwdl_representation_calibration.py \
  tests/test_hcwdl_representation_targets.py \
  tests/test_hcwdl_representation_training.py \
  tests/test_hcwdl_representation_production.py
```

These cover graph inventory/edges; RSET/RREL weights and gradients; model taps
and extraction; recipe values; no-step calibration; bank lifecycle; cold/warm
initialization; real loss assembly; and adapter lineage.

### 17.3 Runtime and orchestration

```bash
python -s -m pytest -q -p no:cacheprovider --assert=plain \
  tests/test_hcwdl_representation_campaign.py \
  tests/test_hcwdl_representation_campaign_artifacts.py \
  tests/test_hcwdl_representation_runtime_rows.py \
  tests/test_hcwdl_representation_runtime_provenance.py \
  tests/test_hcwdl_representation_task_runtime.py \
  tests/test_hcwdl_representation_worker_runtime.py \
  tests/test_hcwdl_representation_reporting.py \
  tests/test_hcwdl_representation_recovery.py \
  tests/test_hcwdl_representation_contracts.py \
  tests/test_hcwdl_representation_layout.py \
  tests/test_hcwdl_representation_cli.py
```

### 17.4 Dense preparation and evidence

```bash
python -s -m pytest -q -p no:cacheprovider --assert=plain \
  tests/test_hcwdl_representation_dense_teacher.py \
  tests/test_hcwdl_representation_dense_preparation.py \
  tests/test_hcwdl_representation_candidate.py \
  tests/test_hcwdl_representation_resource_probe.py \
  tests/test_hcwdl_representation_nonfinal_resources.py \
  tests/test_hcwdl_representation_evidence.py
```

An independent review of the code at source commit `63139ca` ran a focused
scientific/runtime subset with **132 passed and 1 platform skip**. That is code
evidence, not a replacement for a fresh production-worker Tigris smoke.

## 18. Preparing a complete dense smoke candidate

The preferred entrypoint is
[`scripts/prepare_hcwdl_representation_dense_smoke_candidate.py`](../scripts/prepare_hcwdl_representation_dense_smoke_candidate.py).
Do not hand-assemble runtime rows for a real run.

### 18.1 Required immutable inputs

Candidate preparation expects:

- an isolated, clean, pushed source checkout;
- source and split manifests;
- train row selection and train/validation assignment manifests;
- the historical native-offline teacher campaign, recipe, report/checkpoint,
  and its clean historical source checkout;
- `dense_teacher_import.json` from
  `prepare_hcwdl_representation_dense_teacher_import.py`;
- tap and installed-Weaver parity from
  `prepare_hcwdl_representation_dense_surface_evidence.py`;
- descent graph, empty controls, kernel resources, v4 representation recipe,
  and dense disposition from
  `prepare_hcwdl_representation_recipe_assets.py`;
- measured or explicitly compatible resource profile;
- storage template/estimate and sufficient free space;
- worker runtime measurements;
- read-only data root.

Every input is reopened and cross-bound. Copying a JSON file into a newer
directory does not make it current.

### 18.2 Candidate CLI

```bash
python -s scripts/prepare_hcwdl_representation_dense_smoke_candidate.py \
  --representation-root "${DENSE_ROOT}" \
  --project-dir "${PROJECT_DIR}" \
  --compatible-resource-profile "${PROFILE}" \
  --storage-estimate "${STORAGE_ESTIMATE}" \
  --storage-template "${STORAGE_TEMPLATE}" \
  --dense-teacher-import "${DENSE_TEACHER}" \
  --representation-graph "${GRAPH}" \
  --representation-recipe "${RECIPE}" \
  --dense-disposition "${DISPOSITION}" \
  --tap-schema "${TAP}" \
  --surface-parity "${PARITY}" \
  --source-manifest "${SOURCE_MANIFEST}" \
  --split-manifest "${SPLIT_MANIFEST}" \
  --train-row-selection "${ROW_SELECTION}" \
  --train-assignment-manifest "${TRAIN_ASSIGNMENT}" \
  --validation-assignment-manifest "${VALIDATION_ASSIGNMENT}" \
  --historical-campaign-spec "${HISTORICAL_SPEC}" \
  --historical-recipe "${HISTORICAL_RECIPE}" \
  --historical-project-dir "${HISTORICAL_PROJECT}" \
  --runtime-measurement-root "${RUNTIME_MEASUREMENTS}" \
  --data-root "${DATA_ROOT}"
```

It publishes, without scheduler mutation:

- all target planning assets;
- runtime facts/signatures;
- runtime prerequisites and task rows;
- runtime binding;
- planning campaign spec;
- exact command plan;
- runtime dry-run audit;
- executable-candidate audit.

The summary must show 261 tasks, zero final rows, and
`scheduler_mutated: false`.

Use fresh immutable output paths. If an attempt partially publishes artifacts,
preserve it for diagnosis and use a new root after a semantic/source change;
do not overwrite it.

## 19. Review, live conversion, submission, and operations

### 19.1 Review and live conversion

1. Reopen and validate the planning spec, runtime binding, command plan, and
   executable-candidate audit.
2. Inspect the exact source, candidate, recipe, graph, resource, storage,
   worker, task-count, and final-role identities.
3. Build the candidate-bound authorization with
   `build_hcwdl_representation_submission_authorization.py`.
4. Create the authorized live spec with
   `create_hcwdl_representation_campaign.py --authorized-executable`.
5. Revalidate the live spec and exact command plan.

The current code enforces this authorization split. This guide describes the
actual callable interface; it does not add an extra policy layer.

### 19.2 Submit

The only production submitter is:

```bash
python -s scripts/submit_hcwdl_representation_campaign.py \
  --campaign-spec "${DENSE_ROOT}/campaign_spec.json" \
  --command-plan "${DENSE_ROOT}/command_plan.json" \
  --output "${DENSE_ROOT}/submission_ledger.json" \
  --execute
```

This mutates Slurm. It writes append-only submission events and the exact job
ledger. A planning spec, changed command plan, or authorization for a different
candidate is rejected.

### 19.3 Monitor exact IDs

```bash
python -s scripts/monitor_hcwdl_representation_campaign.py \
  --campaign-root "${DENSE_ROOT}" \
  --submission-ledger "${DENSE_ROOT}/submission_ledger.json" \
  --query-scheduler
```

The monitor publishes an append-only report and validates expected artifacts.
Do not infer campaign completion from a broad account-level `sacct` listing.

Use `audit_hcwdl_representation_campaign_outputs.py` for the complete output
audit and `build_hcwdl_representation_dense_smoke_acceptance.py` only after the
ledger and all outputs authenticate.

### 19.4 Recover

`resume_hcwdl_representation_campaign.py` first creates an exact nonexecuting
recovery plan. With `--execute`, it submits only authenticated absent outputs
and writes a chained recovery ledger. It is not a generic retry-all command.

### 19.5 Cancel

`cancel_hcwdl_representation_campaign.py` prints exact ledger-bound `scancel`
commands. It executes them only with `--execute`. Never cancel this campaign by
a broad job-name match.

## 20. Current execution-status warning

At the snapshot reviewed for this guide, the first 261-task smoke built from
source `526a620...` failed at the root `tap_schema` worker before registered
scientific inputs were opened. Runtime facts had bound the outer source-
snapshot artifact hash instead of the inner clean-checkout identity.

Commit `63139ca` fixes that source identity and has focused regression
coverage. The failed campaign is not scientific evidence and cannot be
repaired into evidence in place; a fresh candidate is required at the fixed
source. Consult `docs/HANDOFF.md` for newer execution status rather than
treating this paragraph as a permanent run ledger.

## 21. Appropriate reuse levels

### Pure mathematical work

Import tensor helpers from `hcwdl_representation_losses.py` and
`hcwdl_representation_kernels.py`. This is appropriate for unit tests and loss
experiments, not campaign evidence.

### Local nonauthorizing training work

Use `representation_training_configuration(..., mode="synthetic_test")` and
explicit fixtures. Synthetic artifacts cannot be relabeled as smoke,
scientific, or Tigris results.

### Production work

Build the campaign/runtime artifacts and run the registered worker. Do not
call `training_adapter()` with a hand-built dictionary in a research run. The
runtime binding exists to prove that reviewed inputsâ€”not merely plausible
Python valuesâ€”reached the scientific function.

## 22. Reviewer checklist

### Scientific semantics

- RSET activates only jet/set; RREL activates jet/set/relation.
- Full weights are RSET 0.40/0.60 and RREL 0.30/0.45/0.25.
- `rho` is 0.10; projection orthogonality coefficient is 0.001.
- Base loss is 0.25 label CE plus 0.75 one-predecessor KD.
- Set/relation comparisons remain matching-free.
- Relation selection is top-32 with exact strata and ESS eligibility.
- Calibration is train-only, restores state/RNG, and performs no step.

### Graph semantics

- Exactly 86 nodes and two separate D100 roots.
- Four five-point descents stop at M1.
- Every child uses the immediate richer predecessor only.
- Cold is fresh; warm loads only predecessor deployable weights.
- D100-D5 are nondeployable; D0/M1 are HLT-only deployable.
- Only four M1 nodes are primary endpoints.
- No M2-M6 or old M5 controls are registered.

### Target/data semantics

- TOFF is training teacher only.
- Source/split/selection/assignment bytes are authenticated.
- Joins use row identity rather than order.
- No offline/degradation input appears in deployable extraction.
- Every target generation has exact consumers and authenticated cleanup.

### Runtime semantics

- Every task/index has one typed runtime row.
- Adapter names come from the closed registry.
- Teacher/model/warm-checkpoint/generation/output-owner lineage is cross-bound.
- Live checkout/runtime checks precede scientific input access.
- Source commit and snapshot identify the same clean pushed checkout.
- Publications use canonical immutable routes.

### Campaign semantics

- Smoke reconstructs to 261 tasks; pilot to 275.
- Dense role counts contain zero final rows/partitions/finalists.
- No shared-final/final task is registered.
- The command plan uses exact `hcwdlr_` workers and reviewed resources.
- Submission uses one exact live spec and append-only events.
- Monitor/recovery/cancel operate on ledger IDs only.

## 23. Common misunderstandings

**â€œCold means no teacher.â€** No. It means fresh initialization; predecessor
logits and representation targets are still used.

**â€œWarm resumes the predecessor optimizer.â€** No. Only the deployable model
state is loaded; optimizer, scheduler, scaler, and representation heads reset.

**â€œRREL is RSET plus 0.25.â€** No. Relation strength reallocates jet/set mass,
ending at 0.30/0.45/0.25.

**â€œ`ascent_graph.json` means M1-to-M6.â€** No. That path/name is retained for
compatibility; the active contract/registry is the dense descent.

**â€œD100-D5 should be deployable because they are models.â€** No. They are
training intermediates whose purpose is to reduce the teacher/student gap.

**â€œThe smoke tests only a few nodes.â€** No. It exercises all 86 nodes and 261
tasks, bounded to two updates and 512/256/0 rows.

**â€œCOMPLETED in Slurm means reusable.â€** No. Reuse needs typed artifact,
content, parent, source, and output-owner validation.

## 24. Fast file index

| Concern | Primary file |
|---|---|
| Exact 86-node graph | `src/hlt_classification/scouting/hcwdl_representation_graph.py` |
| RSET/RREL math | `src/hlt_classification/scouting/hcwdl_representation_losses.py` |
| Spectral kernels | `src/hlt_classification/scouting/hcwdl_representation_kernels.py` |
| Frozen recipe | `src/hlt_classification/scouting/hcwdl_representation_recipe.py` |
| Particle Transformer taps | `src/hlt_classification/models/scouting_particle_transformer.py` |
| Tap/Weaver parity | `src/hlt_classification/models/hcwdl_surfaces.py` |
| Heads/deployable extraction | `src/hlt_classification/models/hcwdl_representation.py` |
| Calibration | `src/hlt_classification/scouting/hcwdl_representation_calibration.py` |
| Training loop/report/checkpoints | `src/hlt_classification/scouting/hcwdl_representation_training.py` |
| Target contracts/lifecycle | `src/hlt_classification/scouting/hcwdl_representation_targets.py` |
| Teacher target forward | `src/hlt_classification/scouting/hcwdl_representation_target_runtime.py` |
| Target planning | `src/hlt_classification/scouting/hcwdl_representation_target_planning.py` |
| Production adapters/data caches | `src/hlt_classification/scouting/hcwdl_representation_production.py` |
| Historical TOFF import | `src/hlt_classification/scouting/hcwdl_representation_dense_teacher.py` |
| 261/275-task campaign | `src/hlt_classification/scouting/hcwdl_representation_campaign.py` |
| Runtime prerequisites/rows | `src/hlt_classification/scouting/hcwdl_representation_runtime_rows.py` |
| Runtime binding | `src/hlt_classification/scouting/hcwdl_representation_runtime_binding.py` |
| Closed adapters | `src/hlt_classification/scouting/hcwdl_representation_runtime_adapters.py` |
| Task dispatch | `src/hlt_classification/scouting/hcwdl_representation_task_runtime.py` |
| Live environment check | `src/hlt_classification/scouting/hcwdl_representation_worker_runtime.py` |
| Reporting | `src/hlt_classification/scouting/hcwdl_representation_reporting.py` |
| Contract/schema registry | `src/hlt_classification/scouting/hcwdl_representation_contracts.py` |
| Canonical routes | `src/hlt_classification/scouting/hcwdl_representation_layout.py` |
| Dense candidate composition | `src/hlt_classification/scouting/hcwdl_representation_dense_preparation.py` |
| Candidate audit | `src/hlt_classification/scouting/hcwdl_representation_candidate.py` |
| Generic task CLI | `scripts/run_hcwdl_representation_task.py` |
| Ordinary worker | `sbatch/run_hcwdl_representation_task.sh` |
| Deterministic worker | `sbatch/run_hcwdl_representation_deterministic_task.sh` |
| Operator inventory | `scripts/README.md` |
| Compute runbook | `docs/HCWDL_RKD_RUNBOOK.md` |

## 25. Bottom line

This is one production system with two scientific loss families. RSET transfers
jet and token-set structure. RREL transfers the same information plus
stratified pair-relation geometry. Each strategy has its own D100 root and
cold/warm five-point descents. Every rung is taught by its immediate richer
predecessor, while cold versus warm changes initialization only. The graph
stops at four M1 endpoints and exports only HLT-only deployable state.

For a concrete review, follow `TOFF -> RREL_D100 -> RREL_D95w` through:

1. `NODE_REGISTRY`;
2. TOFF and D100 target generation;
3. RREL loss assembly;
4. warm deployable-checkpoint loading;
5. runtime row and binding;
6. campaign bank wave;
7. worker dispatch;
8. training report, selection, extraction, and cleanup.

That one path crosses almost every important scientific and operational
boundary and makes teacher, bank, checkpoint, strategy, and initialization
roles unambiguous.

