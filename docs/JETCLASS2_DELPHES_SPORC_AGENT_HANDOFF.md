# JetClass2 Delphes on SPORC: detailed handoff for another chat

This is the starting document for an assistant joining Ryan's **new JetClass2
Delphes dataset / SPORC A100** work. It explains both the scientific interface
and the practical execution machinery. It is intentionally more detailed than
a queue command: several old campaigns in this repository use incompatible
data, environments, and meanings for similar-looking model names.

Execution snapshot: **2026-09-12**, after the successful debug A100 profile and
the user's live submission of the 500k four-spine campaign. Job states and free
space below are historical observations, not a live connection to Slurm.

## 1. The short version: read this before doing anything

- Log into **`ryreu@sporcsubmit.rc.rit.edu`**. Science uses cluster `sporc`,
  partition `tier3`, account `reu-aisocial`, QoS `qos_tier3`, and one A100.
- Use **`/home/ryreu/miniconda3/envs/atlas_kd_sporc`**, not the ARM Tigris
  environment. Source the new dataset's absolute-path worker helper.
- The new raw dataset is already on shared RC storage. **Do not upload it
  again, copy it into checkpoints, or point a new run at the old CMS dataset.**
- The current profile is **500,000 training jets, 1,000,000 validation jets,
  and 1,000,000 sealed final-test jets**, not all rows in the directory.
- Reuse the immutable split registry. The 1M/1.5M/2M training profiles are
  nested extensions with exactly the same validation and test identities.
- The study progression is **500k first, then 1M, then 1.5M, then 2M**.
  This split-sharing rule applies to **all experiments on this benchmark**,
  not only the four spines: at a given size use the same exact training jets;
  across all sizes and methods keep the same exact 1M validation and 1M test
  jets. Equal counts, class proportions, or split seeds alone are not enough.
- This benchmark has **17 model features and 11 classes**. It is not the old
  21-feature, 15-class Scouting benchmark. Train fresh references and teachers.
- Use `hlt_matched=True`, scalar `jet_label`, and the frozen source-selection
  policy. Signal-production files also contain QCD. Filenames are not labels.
- Particle views/cache arrays stay in RAM. Compact matching maps, probability
  banks, selected model weights, and reports persist. **No rolling resumes.**
- The matching campaign has real installed-Weaver/A100 acceptance evidence.
  That does **not** certify a different loss, architecture, profile, execution
  environment, or another chat's separate study.
- Do not change a running source worktree, immutable spec, split, environment,
  or dependency chain casually. Use exact campaign-bound job IDs.
- The user most recently requested **dropping ULTRADENSE, retaining DIRECT,
  COARSE, and DENSE**. A cancellation command was supplied, but its execution
  has not been confirmed in the evidence available here. Check current state.
  The original spec still describes four branches; do not silently rewrite it.

The user prefers explicit copy-paste commands in chat when asking how to run
something. Use this document to construct those commands; do not respond only
with “read the runbook.” This document itself does not authorize submission,
cancellation, final-test access, or modifications to other projects.

## 2. Authority, scope, and how to resolve older documentation

Read the repository's [README](../README.md), [HANDOFF](HANDOFF.md),
[transfer plan](../REPOSITORY_TRANSFER_PLAN.md), and
[agent instructions](../AGENTS.md), then the
[active Delphes migration plan](plans/JETCLASS2_DELPHES_DATASET_MIGRATION_IMPLEMENTATION_PLAN.md)
and [versioned Delphes contracts](contracts/JETCLASS2_DELPHES.md).

Scientific authority remains: active plan, reusable contract, immutable
execution spec, agent instructions, then living status. This guide explains
those sources; it does not redefine their scientific meaning.

There are three important historical distinctions:

1. Old repository defaults say Tigris/GH200 and `atlas_kd_tigris`. The active
   migration plan explicitly selects SPORC/A100 **for this benchmark**. Do not
   change global defaults and thereby move unrelated campaigns.
2. Older status paragraphs say new-data A100 evidence is still pending. The
   later user-supplied result for job **21619139** establishes that the
   matching campaign's registered profile passed. The underlying parity,
   resource, and final-test requirements were not waived.
3. “Four-spine” describes the immutable submitted graph. A later request to
   cancel one branch is an execution change, not evidence that its registered
   fits completed. The whole-graph aggregate cannot truthfully report success
   after that branch is abandoned.

The newer, separate
[offline auxiliary-supervision study](plans/JETCLASS2_DELPHES_OFFLINE_AUXILIARY_SUPERVISION_500K_PLAN.md)
has its own [contract](contracts/JETCLASS2_DELPHES_OFFLINE_AUXILIARY_SUPERVISION.md),
roles, targets, and acceptance gate. At this handoff it is separate local work,
not part of commit `82032e35`'s queued matching campaign. Reuse compatible
dataset infrastructure, but do not transfer the matching campaign's GPU
certification to that study or assume it needs the matching foundation.

## 3. Locations and the currently evidenced execution

### 3.1 Filesystem map

| Purpose | Location |
| --- | --- |
| Windows repository | `C:\Users\22rya\ComputerScience\CERN\HLT_Classification` |
| Windows raw data | `C:\Users\22rya\ComputerScience\CERN\data\jetclass2_10M_20260910\jetclass2` |
| Windows split bundle | `C:\Users\22rya\ComputerScience\CERN\HLT_Classification\artifacts\jetclass2_delphes_scaling_splits_20260911_v1` |
| RC main repository | `/home/ryreu/atlas/HLT_Classification` |
| RC dataset parent | `/home/ryreu/atlas/datasets/jetclass2_10M_20260910` |
| RC raw data root | `/home/ryreu/atlas/datasets/jetclass2_10M_20260910/jetclass2` |
| RC frozen inventory | dataset parent + `/manifests/inventory.json` |
| RC outer file split | dataset parent + `/manifests/splits.json` |
| RC split bundle | dataset parent + `/jetclass2_delphes_scaling_splits_20260911_v1` |
| Production Conda prefix | `/home/ryreu/miniconda3/envs/atlas_kd_sporc` |

The RC home/data storage is shared with the user's other RC work. Raw ROOT
files were uploaded and checksum-verified already. The local Windows dataset
and the RC copy have different physical roots but identical scientific row
identities. Relative paths, latest ROOT tree cycles, entry indices, and the
inventory identity determine rows; machine-specific absolute paths do not.

Keep these three kinds of directory separate:

```text
read-only dataset parent/
    jetclass2/                         raw ROOT files
    manifests/                       frozen inventory and outer split
    jetclass2_delphes_scaling_splits_20260911_v1/
        registry.json
        profiles/TRAIN_500K.json
        profiles/TRAIN_1M.json
        profiles/TRAIN_1P5M.json
        profiles/TRAIN_2M.json

clean detached source worktree/       code used by specific jobs

main repository/checkpoints/
    readiness root/foundation/        compact matching preparation
    profile attempt root/evidence/   genuine acceptance/resource evidence
    scientific campaign root/        specs, journals, attempts, selected outputs
```

Do not delete a foundation or source worktree just because it is outside the
scientific campaign root. Specs may reference it directly. Audit references
before cleanup, and preserve backups and any data used by another project.

### 3.2 Exact current paths

These are the evidenced paths, not names to recreate on every invocation:

```text
Pushed production/profile source:
82032e35177f83436741d7fa1b9d38fbc4b3efc7

Pinned profile/science worktree:
/home/ryreu/atlas/HLT_Classification_jc2_debug_82032e35

Original readiness root:
/home/ryreu/atlas/HLT_Classification/checkpoints/jc2_sporc_ready_500k_275b984d_r1

Reusable matching foundation:
/home/ryreu/atlas/HLT_Classification/checkpoints/jc2_sporc_ready_500k_275b984d_r1/foundation

Successful profile-only attempt:
/home/ryreu/atlas/HLT_Classification/checkpoints/jc2_debug_profile_82032e35_r1

Scientific campaign:
/home/ryreu/atlas/HLT_Classification/checkpoints/jc2_sporc_four_spine_500k_82032e35_r1
```

The word `debug` in a worktree's directory name does not determine the Slurm
partition. Its campaign's explicit execution-site contract does. The scientific
jobs created using this worktree request `tier3`.

### 3.3 Evidence received from RC

The user supplied the following results; this documentation task did not log
into RC and independently rerun them.

| Stage | Job(s) | Evidence |
| --- | --- | --- |
| Bounded sample | `21612443` | Readiness chain advanced past it |
| Matching assignment | `21612444`, array `0-257%16` | Completed; 258 elements, not 256 |
| Foundation lock | `21612445` | COMPLETED, 3 minutes 2 seconds |
| Original tier3 profile | `21612446` | Cancelled while pending; zero runtime |
| Replacement debug profile | `21619139` | COMPLETED, exit `0:0`, 1 hour 4 minutes 50 seconds |
| Scientific dry run | root above | Audit passed: 31 train + 26 reduce + aggregate + complete |
| Scientific live submission | root above | User supplied a populated queue; exact task/ID mapping is its live ledger |

The profile summary explicitly reported `passed=True`, miniature passed,
full-population probe passed, RAM-only views, no rolling resumes, and
`final_test_accessed=False`. The live scientific submission followed the
successful dry run. No completed scientific fit from this new campaign was
provided before this guide was written.

## 4. SPORC is not Tigris: environment and resource rules

### 4.1 Correct site settings

| Setting | New Delphes production | Historical Tigris runs |
| --- | --- | --- |
| Login endpoint | `sporcsubmit.rc.rit.edu` | `tigris.rc.rit.edu` |
| Slurm cluster | `sporc` | `tigris` |
| Partition | `tier3` | `tigris` |
| Account | `reu-aisocial` | `reu-aisocial` |
| QoS for this new campaign | `qos_tier3` | Do not import automatically |
| GPU request | `--gres=gpu:a100:1` | GH200 allocation |
| Architecture/environment | x86-64, `atlas_kd_sporc` | ARM, `atlas_kd_tigris` |
| Conda installation | `/home/ryreu/miniconda3` | `/home/ryreu/miniforge3-aarch64` |
| Current GPU job allocation | 1 node, 1 task, 1 GPU, 8 CPUs, 72 GiB host RAM | Old allocations are not defaults here |

The measured GPU was **NVIDIA A100-PCIE-40GB**. Production checks the measured
GPU identity, including memory/capability information, not just whether its
name contains “A100.” An A100 with a different memory configuration or an
H100 is not automatically a valid substitute for this profile.

The site definitions live in
[execution.py](../src/hlt_classification/jetclass2_delphes/execution.py).
They include conservative SPORC bounds of 36 CPUs and 340,000 MiB host RAM,
but those are not instructions to request the maximum. Use measured resources
and inspect current scheduler configuration when proposing another allocation.

Do not upgrade or reinstall packages in an environment that active jobs use.
The installed-environment artifact fingerprints Python, architecture,
Torch/CUDA/cuDNN, Weaver version and Python source bytes, NumPy, Uproot, Awkward,
SciPy, scikit-learn, and threadpoolctl. “Imports work” is weaker evidence than
“the profiled environment matches.” If a dependency must change, use a
separate environment and the appropriate new acceptance/profile workflow.

### 4.2 Safe activation and read-only status for the current campaign

Run this in a SPORC SSH session. The subshell confines `set -e` and environment
changes: a failed check exits the block rather than closing the SSH login.
This block submits, cancels, and edits **nothing**.

```bash
(
set -euo pipefail
export MAIN_REPO=/home/ryreu/atlas/HLT_Classification
export JC2_ROOT="${MAIN_REPO}/checkpoints/jc2_sporc_four_spine_500k_82032e35_r1"
test -f "${JC2_ROOT}/campaign_spec.json"

export PROJECT_DIR="$(python3 - "${JC2_ROOT}/campaign_spec.json" <<'PY'
import json
import sys
from pathlib import Path
spec = json.loads(Path(sys.argv[1]).read_text())
print(spec["project_dir"])
PY
)"
export JC2_SITE=sporc_a100
source "${PROJECT_DIR}/sbatch/jetclass2_delphes_common.sh"

python -c 'import sys, platform; print(sys.executable); print(platform.machine())'
git -C "${PROJECT_DIR}" rev-parse HEAD
git -C "${PROJECT_DIR}" status --short

python -s "${PROJECT_DIR}/scripts/jetclass2_delphes_production.py" results \
  --spec "${JC2_ROOT}/campaign_spec.json"

python -s "${PROJECT_DIR}/scripts/jetclass2_delphes_production.py" monitor \
  --spec "${JC2_ROOT}/campaign_spec.json" \
  --ledger "${JC2_ROOT}/submission_ledger.json"
)
```

[jetclass2_delphes_common.sh](../sbatch/jetclass2_delphes_common.sh) selects the
Conda installation using explicit `JC2_SITE`, clears inherited Python/loader
paths from other projects, and sources the absolute
`${PROJECT_DIR}/sbatch/common.sh`. It sets `PYTHONNOUSERSITE=1`,
`PYTHONDONTWRITEBYTECODE=1`, unbuffered output, the pinned `src` import path,
and one numerical-library thread per worker. The shared activation helper
prepends `${CONDA_PREFIX}/lib` to `LD_LIBRARY_PATH`.

Always use this helper from the pinned worktree. Do not source a worker helper
relative to Slurm's spool copy of a batch script. Do not inherit an unrelated
project's `PYTHONPATH` or a Tigris Conda prefix through `--export=ALL`.

### 4.3 Login-node work versus compute-node work

Git operations, reading JSON reports, dry-plan creation, and scheduler queries
belong on the submit host. Genuine training, GPU parity, resource profiling,
and full particle-cache construction belong in the registered Slurm workers.
A missing CUDA device on the login host is not evidence that the A100 compute
environment is broken. Do not invoke production `run`/`profile` directly there
to “test it quickly.”

Helpful read-only scheduler commands are:

```bash
squeue --me -o '%.18i %.14P %.60j %.2t %.10M %R'
squeue --me --start
scontrol show partition tier3
scontrol show partition debug
sinfo -h -N -p tier3,debug -o '%N|%t|%c|%m|%G|%l'
```

These show other projects too. Displaying them does not put them in scope for
cancellation. In particular, the user's `bsmr_*` jobs belong to another project.
Estimated start times and idle-node displays are not guarantees of an
immediate allocation. Lowering CPU/RAM or changing partition on an existing
profile-bound science job can make its execution gate reject the allocation.

## 5. What the new dataset actually contains

### 5.1 Provenance and publication context

The dataset was supplied by Luka Lambrecht as paired offline/HLT-like Delphes
jets. Its source directory was:

```text
/eos/user/l/llambrec/jetclass/output_jetclass2_10M_20260910/jetclass2
```

The preceding CMS dataset came with collaboration publication constraints.
The new production is intended to support a publicly reproducible route, but
that intent is **not** a completed dataset release/license or permission
audit. Exact generation/Delphes cards, production revision, citations, and
release arrangements still need to be recorded before publication claims.
Luka's August 27 slides are useful context, not proof of the final September
10 card settings.

This is a **new benchmark**, not the old benchmark with only matching changed.
Simulation, class set, available particle fields, selection, and data population
differ. Compare methods within this dataset using fresh references. Do not use
old FullSim AUC or R50 as recovery denominators, import old weights, or claim
that a difference between the two datasets isolates one algorithmic effect.

### 5.2 Frozen snapshot and selection counts

The uploaded snapshot contains **333 ROOT files**, **25,430,379,343 bytes**
(23.68 GiB). The user verified all 333 uploaded file checksums against the
local inventory. The directory name `10M` is a production label, not an exact
row-count contract.

| Quantity from the audited latest tree cycles | Count |
| --- | ---: |
| All stored rows | 12,216,863 |
| Rows with a matched HLT jet | 11,443,876 |
| Rows selected by the default benchmark policy | 8,996,860 |
| Selected signals | 1,987,133 |
| Selected QCD-production QCD rows | 7,009,727 |
| Matched QCD rows in signal production excluded by default | 2,447,016 |

Relative file paths are under:

```text
train_higgs2p/onlyFatJet+onlyFatJetHLT/ntuple_*.root   (133 files)
train_qcd/onlyFatJet+onlyFatJetHLT/ntuple_*.root       (200 files)
```

Do not identify files by basename alone; both categories have `ntuple_*.root`.
Use the inventory's complete relative path. Read the inventory's **latest
ROOT tree cycle** rather than concatenating all cycles, which can duplicate
entries. The schema and selected row counts are authenticated per file.

### 5.3 Selection is not inferred from filenames

The frozen default policy does the following:

1. Require `hlt_matched=True`. Rows without a matched HLT jet have dummy HLT
   branches and are not sensible deployable examples.
2. Determine class from scalar `jet_label`, using the registered numeric map.
3. Collapse the 27 QCD labels into one model class.
4. Exclude QCD-labelled rows in `train_higgs2p` under the default source policy.
   Including them is possible only as a differently hashed selection and split,
   not an invisible reader convenience.
5. Do not silently discard a supported signal label merely because it is in
   a QCD-production file. Label and source policy are separate concepts.

Unknown labels, invalid retained-particle values, or shape/count mismatches
fail closed. No old tight-ID, jet-rank, pT/mass selection, or minimum
16-particle jet cut is inherited. The number 16 below is a **padding minimum**,
not a sample-selection threshold. Retained jets must have nonempty sides.

The implementation is
[selection.py](../src/hlt_classification/jetclass2_delphes/selection.py) plus
[schema.py](../src/hlt_classification/jetclass2_delphes/schema.py). Do not write
a second slightly different label/filter implementation in a new training
script.

### 5.4 Exact classifier label map

| Raw `jet_label` | Output index | Output name |
| --- | ---: | --- |
| 161 through 187 | 0 | QCD |
| 0 | 1 | X_bb |
| 1 | 2 | X_cc |
| 2 | 3 | X_ss |
| 3 | 4 | X_qq |
| 9 | 5 | X_gg |
| 10 | 6 | X_ee |
| 11 | 7 | X_mm |
| 12 | 8 | X_tauhtaue |
| 13 | 9 | X_tauhtaum |
| 14 | 10 | X_tauhtauh |

There are no fabricated outputs for absent charged/BSM modes. The five broad
QCD groups, if needed for diagnostics, are 161–169 (`bb`), 170–178 (`b`),
179–181 (`cc`), 182–184 (`c`), and 185–187 (light). They are **not** five
separate outputs in the current classifier.

Luka confirmed that the label enumeration was unchanged from upstream
JetClass2. The migration plan records the inspected supplied-source reference
`69c58b243476412b7489ddd80fd7d1897ec27b8a`. The original label-map artifact was
created under a provisional policy and cites upstream reference
`3a7a1355f4230b5790669286466080d7fa3b6794`. Neither is asserted to be the exact
dataset production commit. Preserve existing artifact hashes and append
confirmation evidence; do not edit old inventories just to change a
`producer_revision_confirmed` or `provisional` field.

### 5.5 Available fields, missing fields, and zero errors

Each raw particle is represented by 14 values:

```text
px, py, pz, energy,
charge,
isChargedHadron, isNeutralHadron, isPhoton, isElectron, isMuon,
d0val, d0err, dzval, dzerr
```

Offline branches use `part_`; HLT branches use `hlt_part_`. Jagged lengths
must agree with `jet_nparticles` or `hlt_jet_nparticles`. PID flags are
exclusive one-hot, charge is in {-1, 0, 1}, and PID/charge applicability is
validated. Particle energy and transverse momentum must be positive; required
values must be finite and uncertainties nonnegative.

The old setup had additional quality, track-fit, lost-hit, and track-origin
fields that are not present here. They are **omitted**, not filled with guessed
values. In particular, do not import an old adapter that expects those fields
and make it run by silently inserting zeros.

The declared zero-error policy is important. A zero `d0err`/`dzerr` is treated
as unavailable/not-applicable uncertainty, **not infinite precision**. Retain
a finite stored displacement even if its error is zero. Do not divide it by
epsilon to manufacture a significance, drop the particle, or erase its value.
The feature transform uses displacement `tanh` and directly clipped errors;
it does not compute displacement significance.

Luka expected neutral-particle errors to be dummy values and asked for charged
zero-error fractions and associated displacements. Their precise physical
interpretation remains unresolved. Stored-mm interpretation and other producer
details remain declared assumptions, not confirmed measurement semantics.
The user authorized proceeding under these explicit assumptions while waiting
for clarification. A later policy correction requires versioned artifacts
and an honest comparison boundary, not hot-patching existing runs.

### 5.6 Model inputs are exactly 17 channels

The feature order in
[inputs.py](../src/hlt_classification/jetclass2_delphes/inputs.py) is:

```text
pt_log, e_log, logptrel, logerel, deltaR, charge,
isChargedHadron, isNeutralHadron, isPhoton, isElectron, isMuon,
d0, d0err, dz, dzerr, deta, dphi
```

The first five use the canonical analytic log offsets/scales and clipping,
not a fitted normalizer from the old dataset. Relative angles and fractions
are computed from the **visible constituent-sum axis of the supplied view**.
Eta is reflected by jet-eta sign and phi is wrapped. Stored producer-relative
angles are not substituted into an intermediate repaired view with a different
axis. This is why simply renaming raw branches is insufficient migration.

The model receives only `features`, `vectors`, and `mask`. Matching maps,
source indices, labels, row identities, `hlt_matched`, and
`hlt_jet_dr_offline` are not input channels. Training code may carry identities
alongside batches for target joins, but it must never forward them to ParT.

Capacity is **240** for this snapshot: the pre-split selected-count maximum
rounded up to a multiple of 16. Observed maxima were 176 HLT and 231 offline
particles. Overflow fails instead of truncating. RAM caches pack real tokens;
each batch pads to its own longest row, at least 16. The installed Weaver
model has 17 inputs, 11 outputs, and **`trim=False`**. Re-enabling random
training-time trimming would change the registered no-truncation experiment.

## 6. Reusable splits: exactly what another study must share

### 6.1 Study progression and the cross-experiment consistency rule

The intended research sequence is to run setups at **500k training jets first**,
then repeat the planned comparisons at **1M**, then **1.5M**, then **2M**.
These are four stages of a training-data scaling study, not four independent
random draws and not permission to jump straight to full-data training.
Moving to a later size still requires its applicable readiness checks and
explicit submission authorization; this document does not queue those stages
or require a scheduler dependency on every earlier experiment finishing.

The central comparison rule is **same exact jets, not merely the same number
of jets**. It applies across all experiments using this frozen new-dataset
benchmark: baselines, offline oracles, direct KD, coarse/dense ladders,
matching variants, auxiliary-supervision studies, hyperparameter ablations,
and training-seed repetitions.

- **Within one training-size stage:** every compared experiment uses exactly
  the same registered training-row membership. For example, a 500k CE baseline
  and a 500k KD or auxiliary model must see the identical 500,000 training jets,
  not two different 500k samples from the same files.
- **Across training-size stages:** the smaller training set is contained in
  the larger set. The 1M stage retains every 500k training jet and adds 500k;
  1.5M retains that entire 1M and adds 500k; 2M retains that entire 1.5M and
  adds 500k. Do not replace earlier jets with a newly drawn sample.
- **Validation throughout:** one fixed set of exactly 1,000,000 validation
  jets, unchanged across sizes, methods, campaign roots, and training seeds.
- **Final test throughout:** one fixed set of exactly 1,000,000 test jets,
  likewise unchanged. It stays sealed until the required final evaluation
  authorization and locks; sharing its membership does not permit inspecting
  its predictions during development.

In set notation, with T denoting training membership:

```text
T_500K subset T_1M subset T_1P5M subset T_2M

All methods at size S: the same T_S
All methods and sizes: the same V_1M and the same sealed E_1M
```

This keeps changes in the evaluated population from being confused with gains
from the method or from more training data. Different inputs, teachers,
losses, initialization seeds, and minibatch order can be declared experimental
variables; **split membership is not a new randomization variable for each
experiment**. A fresh training seed must not trigger a fresh data split.

Every new chat should reuse the existing inventory, outer split, registry,
and chosen exported `TRAIN_*` profile. Before treating two runs as a matched
comparison, validate their artifact lineage and check:

1. The inventory and split-registry identities agree.
2. At the same training size, the profile and training-membership hashes agree.
3. Validation-membership and final-test-membership hashes agree across every
   size and method. The complete profile hashes differ between sizes because
   their training memberships differ; that is expected.
4. Actual readers/outputs obey those masks and canonical row identities, with
   no method-specific row dropping, replacement jets, or unregistered filters.

The existing registry encodes these exact sets; do not generate a new registry
per campaign even with the same nominal seed. Store the shared membership
hashes in each experiment's artifacts so consistency is auditable rather than
just asserted in its name.

Some separately registered studies divide the fixed validation set into
internal selection/reporting masks, such as the auxiliary study's 200k/800k
partition. Such masks remain explicit subsets of the **same canonical 1M**;
they do not replace or redraw it. Compare models on the same declared reporting
mask, label subset results clearly, and do not present a 200k or 800k score as
a directly matched full-1M score. Do not silently change those existing study
locks to satisfy a new printer.

A genuinely necessary future change to the snapshot, eligible population, or
split requires explicit user agreement and versioned artifacts. It must be
reported as a changed comparison, not quietly described as using the same
jets. Existing immutable runs are not rewritten.

### 6.2 There are two distinct split layers

The outer file-group split (`SPLITS/v1`, seed 20260910) assigns whole ROOT files
to approximately 60/20/20 reservoirs while preserving class coverage:

| Reservoir | Selected eligible rows |
| --- | ---: |
| train | 5,376,107 |
| validation | 1,810,176 |
| final_test | 1,810,577 |

**Those are not the current fit sizes.** The reusable registry chooses exact
memberships inside these reservoirs, with seed 20260911:

| Profile | Training | Validation | Final test |
| --- | ---: | ---: | ---: |
| TRAIN_500K | 500,000 | 1,000,000 | 1,000,000 |
| TRAIN_1M | 1,000,000 | the same 1,000,000 | the same 1,000,000 |
| TRAIN_1P5M | 1,500,000 | the same 1,000,000 | the same 1,000,000 |
| TRAIN_2M | 2,000,000 | the same 1,000,000 | the same 1,000,000 |

Training memberships are nested. Validation and test membership hashes are
identical across profiles. Unused evaluation rows do not become extra training
capacity. The current production creator rejects the old full-reservoir
foundation when a registered subset is required.

### 6.3 Membership is explicit, not just a seed

[split_registry.py](../src/hlt_classification/jetclass2_delphes/split_registry.py)
implements deterministic classwise SHA256 priorities and proportional integer
quotas. The smallest subset reserves a row per class, then apportions the
remaining capacity using Hamilton remainders; larger sizes extend the same
classwise ordering. The resulting class distribution is naturally proportional,
not a balanced-class sampler.

Each exported profile includes the selected training membership and the shared
evaluation pair. Per-file entry masks are stored as canonical base64 packed
bits, **little bit order**. Bit `i` refers to original latest-cycle ROOT entry
`i`, not the `i`th row after filtering. Reader order is inventory file order
then increasing selected entry; the training sampler shuffles those rows.

Reuse the exported profile or `select_profile()`; do not reconstruct a split
with `train_test_split`, “take the first 500k,” random file globs, or a new RNG.
Equal counts alone do not mean equal splits. Record inventory, registry,
profile, and role-membership hashes in any new study's lineage.

### 6.4 What is and is not proven about leakage

File/content disjointness is validated. The data do not provide a complete
event/generation-group key that proves different files contain independent
generator events. Cross-file event independence is an explicitly authorized
provisional assumption. Do not claim that a file split proves more than this,
and do not make duplicate event-ID columns up from ROOT entry numbers.

Final-test labels/count metadata were used to freeze and authenticate the
predeclared split. Full-file checksums also read raw bytes. Neither activity
authorizes decoding final-test particle arrays, predicting on them, choosing
hyperparameters with them, or making final-test plots. Ordinary readers,
caches, matching tasks, banks, and this campaign's result commands only allow
train/validation. There is no public `allow_test=True` escape hatch. A later
finalist-locked evaluation executor must be separately designed/authorized.

### 6.5 A new data-size comparison needs fresh references

At each profile size, train a fresh HLT CE baseline and offline CE oracle on
that profile's training membership. A full-data teacher imported into the
500k campaign would change the data-efficiency question. Keep validation/test
identities fixed, but report differing update counts and compute across sizes.

The current implementation requires profile-specific matching preparation and
measured acceptance. A 500k resource profile is not automatically a 2M RAM or
walltime certification. Metadata reuse is broader than execution-evidence reuse.

## 7. How to consume the data without bypassing its contracts

Use [DatasetReader](../src/hlt_classification/jetclass2_delphes/reader.py) rather
than a new ad hoc ROOT loader. It authenticates the inventory/split, respects
original-entry membership masks, verifies source files, and checks selected
counts/classes. Its `include_offline=False` path requests **no offline particle
branches**. Selection metadata and labels remain outside model inputs.

Here is a bounded API illustration, not a command to preprocess the whole
dataset on the login node. `foundation` is a loaded, validated
`foundation_spec.json`; `data_root` points to the relocated raw snapshot.

```python
from itertools import islice
from pathlib import Path
from hlt_classification.jetclass2_delphes.reader import DatasetReader
from hlt_classification.jetclass2_delphes.inputs import build_inputs

reader = DatasetReader(
    Path(data_root), foundation["inventory"], foundation["splits"],
    role="train", include_offline=False,
)
for jet in islice(reader, 2):
    assert jet.offline is None
    inputs = build_inputs(jet.hlt, capacity=foundation["inputs"]["capacity"])
    payload = inputs.model_inputs()  # only features, vectors, mask
    print(jet.identity, jet.label, {k: v.shape for k, v in payload.items()})
```

Stopping after two rows does not prove full-population coverage; the complete
production reader/cache traversal performs the final coverage checks. For
large reads, use
[prepare_cache()](../src/hlt_classification/jetclass2_delphes/cache.py) and its
explicit RAM budgets/process workers. Do not accumulate millions of Python
`Jet` objects or allocate capacity-padded arrays for every jet as a shortcut.
For intermediate views, use the authenticated matching foundation and the
existing coordinate/view functions rather than recomputing a different matcher.

## 8. Matching and the meaning of U/D in the new campaign

### 8.1 Matching is full-cardinality, not the historical selective matcher

The new adapter reuses the exact integer solver in
[hcwdl_fullcard_bottleneck_matcher.py](../src/hlt_classification/scouting/hcwdl_fullcard_bottleneck_matcher.py)
with the versioned
[matcher specification](../src/hlt_classification/scouting/hcwdl_fullcard_bottleneck_contracts.py).
The new schema adapter is
[views.py](../src/hlt_classification/jetclass2_delphes/views.py); the old
21-field endpoint adapters are not used.

For each jet, the matching contains exactly `min(n_hlt, n_offline)` one-to-one
pairs. If HLT is the smaller side, every HLT particle is matched. If offline
is smaller, every offline particle is matched and surplus HLT particles cannot
all receive unique partners. “Full cardinality” therefore does not promise
100% aggregate HLT coverage when HLT multiplicity exceeds offline multiplicity.

The primary objective is lexicographic bottleneck minimization of the sorted
worst-first **quantized delta-R** distances: improve the worst distance first,
then the next-worst, and so forth. Secondary response/category/charge/index
tie rules are frozen by the matcher contract. This is not independent nearest
neighbours, an arbitrary maximum matching, or a minimum-total-distance rewrite.
Directions come from global p4, not coordinates relative to two different jet
axes. Poor forced-pair distances are diagnostic scientific outcomes, not a
reason to silently lower cardinality or discard a jet.

Persisted assignment shards contain row digests, offsets, and int32 HLT-to-
offline maps. They are reusable only after parent, semantic source, coverage,
and byte checks. A directory named “foundation” or a Slurm `COMPLETED` state
alone is insufficient proof.

### 8.2 Endpoint and intermediate semantics

| Coordinate | What the model sees |
| --- | --- |
| U000 | Exact native offline particles, in native order; privileged oracle |
| U050, U033, etc. | Structural transition from offline support toward HLT support; paired content is still offline |
| U100 | HLT carrier/support slots, with matched offline content and unavoidable unmatched native HLT particles |
| D066, D033, etc. | Fixed HLT support; matched content progressively restored toward native HLT |
| D000 | Exact original HLT particles, with no offline access needed for its model-input path |

Maximum cardinality leaves unmatched particles on at most one side. During U,
extra offline particles are removed if offline is larger; extra HLT particles
are inserted if HLT is larger. It is not universally “only add offline
particles.” Structural switches use the existing mass-balanced rule, with a
new label/teacher/branch-independent hash domain. No residual substitution
scale calibration is needed for this full-cardinality case.

During D, matched p4 and applicable numerical measurements interpolate.
Charge/PID switch atomically, charged/neutral measurement applicability follows
that identity choice, and a change in uncertainty validity switches the
associated value/error pair atomically. This avoids fractional particle
identities and invented intermediate significances. Unmatched HLT remains HLT.

In code, U has `(u, f) = (structural_fraction, 0)` and D has
`(1, 1 - remaining_offline_fraction)`. `033` and `066` mean exact rational
one-third and two-thirds, not decimal 0.33/0.66. Use
[campaign.coordinate()](../src/hlt_classification/jetclass2_delphes/campaign.py)
instead of parsing names with floating-point division.

Exact native U000/D000 endpoints have explicit bypasses. Intermediate U/D
models are privileged teachers/diagnostics, not deployable HLT-only models.
The structural policy here is `replace_source_with_target_v1`, not a separate
salience, MT20, persistent-support, or representation-KD variant found elsewhere
in the repository.

## 9. The scientific campaign that was queued

The complete original graph is:

```text
fresh M0HLT: CE-only, native HLT baseline
fresh U000: CE-only, native offline oracle shared by the four spines

DIRECT:
U000 -> D000

COARSE:
U000 -> U050 -> U100 -> D066 -> D033 -> D000

DENSE:
U000 -> U033 -> U066 -> U100 -> D080 -> D060 -> D040 -> D020 -> D000

ULTRADENSE (subsequently requested for cancellation):
U000 -> U020 -> U040 -> U060 -> U080 -> U100
     -> D090 -> D080 -> D070 -> D060 -> D050
     -> D040 -> D030 -> D020 -> D010 -> D000
```

Every arrow trains a **fresh student with logit KD from its immediate single
parent**. It is not weight continuation, the earlier ensemble-zoo ladder, or
KD from every higher rung. There are no E-rung ensembles, RSET/RREL losses,
M1 compression, or M2 in this campaign. A `reduce` task publishes reusable
probabilities from one selected teacher; its name does not imply an ensemble.

Repeated coordinates in different spines are distinct fits with different
parent lineages. Initialization and sampler seeds are deliberately paired by
coordinate, omitting teacher/branch from the seed domain. All D000s and M0HLT
share the paired coordinate-seed convention. Do not switch every rung to a
new seed as an unrecorded “improvement.”

The original job count is 31 training fits (2 references + 29 students),
26 reductions (U000 + 25 nonterminal students), aggregate, and completion:
**59 tasks**. Terminal D000s and M0HLT do not need teacher-bank publications.
The user may abandon ULTRADENSE without stopping the other spines' fits;
whole-campaign completion then needs the treatment in section 13.

### 9.1 Frozen training recipe

| Item | Registered value |
| --- | --- |
| Optimizer | AdamW, betas 0.9/0.999, epsilon 1e-8, weight decay 0.01 |
| Batch/execution | 256, one GPU, accumulation 1; no DDP |
| Precision | BF16 model forward, FP32 losses |
| References | CE only; no KD bank |
| Students | C25P75: 0.25 CE + 0.75 forward KL, temperature 2 |
| Temperature scaling | Exactly one T-squared factor on KL |
| Passes 1–3 | Warmup to 3e-4 |
| Passes 4–45 | Hold at 3e-4 |
| Passes 46–60 | Cosine decay to 1.5e-5 |
| Passes 61–100 | Constant 1.5e-5 |
| Early stopping | Minimum 60 passes, maximum 100, patience 15 |
| Meaningful AUC improvement for patience | Greater than 5e-5 |
| Validation | Every pass, on the fixed 1M validation jets |
| Training population | Every selected training row once per pass; no class weighting/oversampling |
| Persistence | Restore best; selected weights only; no rolling resume |

The learning rate is evaluated at update endpoints in continuous completed-pass
coordinates. Early-stopping patience starts with the first validation, **not**
with a fresh clock at pass 60. It cannot stop before pass 60, but it need not
run 15 more passes after pass 60 if patience has already expired.

Checkpoint selection and patience are different rules. Selection maximizes
validation macro AUC, then minimizes CE, maximizes macro log-R50, and prefers
the earliest update. Smaller AUC gains may change the selected checkpoint
without resetting patience. Reports record both passes actually run and the
selected pass. Do not describe every successful job as having trained exactly
100 passes or confuse selected pass with early-stop pass.

## 10. Readiness, debug profiling, and measured resources

### 10.1 What the readiness jobs do

The original readiness chain is:

```text
bounded sample -> per-file assignment array -> foundation lock -> GPU profile
```

The sample checks real selected train/validation rows and endpoint behavior.
The array builds compact assignments only for selected ordinary-role subsets.
The lock authenticates complete expected shard coverage and lineage. The
profile runs installed-Weaver parity, bounded CE/oracle/KD checks, full-profile
cache construction/timing, a nonscientific full offline training+validation
pass, inference, and a worst-capacity GPU batch check. These are resource and
correctness checks, not scientific training results to put in the ladder table.

The actual first readiness had **258 array elements**, at most 16 concurrent.
It did not train 258 models. Its CPU-only tasks requested 1 CPU/4 GiB; sample
and lock requested 30 minutes, assignments 120 minutes each. The initial
unmeasured GPU envelope was 8 CPUs/8 workers/80 GiB/four hours. That old
80-GiB envelope is not the final measured production allocation.

### 10.2 Why debug profiling did not require rerunning matching

The tier3 profile remained pending after all preparation completed. The
implemented exception created a **new one-job profile attempt on debug**,
read-only-authenticated the completed old foundation, and produced separate
evidence under a new source pin. It did not rebuild assignments or change the
old readiness spec. The original pending profile was cancelled separately.

This is implemented in
[profile_attempt.py](../src/hlt_classification/jetclass2_delphes/profile_attempt.py)
and [prepare_jetclass2_delphes_debug_profile.py](../scripts/prepare_jetclass2_delphes_debug_profile.py).
`RUNTIME_PROFILE/v3` explicitly permits only the registered debug-to-tier3
transfer with the same actual A100, environment, CPU/RAM allocation, and
workers. Older same-site `/v2` evidence cannot be relabelled to do this.
Scientific execution on debug is not enabled by this exception.

The supplied debug partition allowed the account/QoS and a 24-hour maximum.
That was a scheduler snapshot, not a promise that debug will always have free
A100s or an exemption from site usage policy. Recheck it for a new attempt.

### 10.3 Successful 500k profile measurements

| Measurement | Observed value |
| --- | ---: |
| Host CPUs / process workers | 8 / 8 |
| Requested host RAM | 72 GiB |
| Peak cache-array accounting | 5.09 GiB |
| Peak Torch GPU allocation | 6.61 GiB |
| One training pass including validation | 4.43 minutes |
| Reducer inference | 2.27 minutes |
| U000 train + validation preprocessing | 16.36 minutes |
| U050 train + validation preprocessing | 18.17 minutes |
| D050 train + validation preprocessing | 19.09 minutes |
| Profile job total | 1 hour 4 minutes 50 seconds |

Cache-array bytes are **not total process MaxRSS**, and Torch allocated GPU
memory is not every CUDA/driver allocation. Do not shrink the request to
5 GiB host RAM or 7 GiB GPU capacity based on those fields. The production
cache budgets account for worker/IPC peaks and reserve host memory for other
runtime use.

At that measured pass rate, 60 passes plus about 19 minutes preparation is
roughly 4.8 hours; 100 passes roughly 7.7 hours. These are estimates from the
probe, not completed timing measurements for every rung. Sequence lengths,
validation work, shared filesystem load, and GPU/node differences can matter.

### 10.4 Actual queued walltimes, not remembered estimates

The successful dry-run audit specified:

- **Training: 808 minutes = 13 hours 28 minutes.**
- **Reduction: 43 minutes.**
- GPU tasks: tier3, 1 A100, 8 CPUs, 72 GiB.
- Aggregate/completion bookkeeping tasks: 1 CPU, 8 GiB, 60 minutes.

Training envelopes are `ceil(1.75 * (worst_cache_seconds + 100 * pass_seconds)
/ 60)` minutes. Reducers use `ceil(2 * (worst_cache_seconds + inference_seconds)
/ 60)`, subject to registered minimums and the planning ceiling. The safety
margin is deliberate; it is not a claim that a 500k model takes 13.5 hours.
Shorter walltimes were discussed, but the evidence here does not establish a
later applied change. Inspect the command plan and `scontrol` for current jobs.

The profile and every GPU task validate matching resource/environment
identities. Increasing CPUs “because a node has more” or lowering RAM to get a
queue slot can violate those checks. For a proposed allocation change, measure
and register it rather than patching the queued contract. With no rolling
resume, a timeout loses that attempt's training progress.

## 11. RAM versus disk: what must persist and why

| Object | Storage policy |
| --- | --- |
| Original ROOT files and frozen metadata | Read-only shared disk; one snapshot copy |
| Matching row IDs, offsets, integer maps | Compact authenticated disk shards |
| Intermediate particle views / training caches | Process-local RAM, built once per job |
| Per-batch padded arrays / optimizer / in-progress best weights | RAM/GPU memory |
| Teacher class probabilities | Compact persistent shards, reusable by children |
| Successful fit | One selected model-weight checkpoint plus small reports |
| Rolling checkpoints / dense representation targets | Not written by this campaign |

Each probability row contains 11 FP32 values plus a 32-byte identity: 76 bytes
of principal array payload. Shards have at most 100,000 rows, around 7.6 MB
before small metadata. Train banks use T=2; validation probabilities T=1.
Temperature, class order, role, parent checkpoint, and ordered row identities
are authenticated. Do not join targets by current shuffled batch position.

For 500k train + 1M validation and 26 publications, principal bank payload is
about **2.76 GiB**, before assignment/checkpoint/report overhead. The live
submission headroom requirement was **7.02 GiB**, with **95.30 GiB available**
in the user's dry-run output. Available space is time-sensitive; recheck it.
The gate budgets twice planned bank/checkpoint payload plus 1 GiB. Neither
RAM-only views nor this estimate means “no disk use.”

Failed attempts have separate directories, and compact partial outputs are
not automatically deleted. Count accumulated attempts when checking space.
Do not delete journals to make an ambiguous submission retry succeed, remove
teacher banks needed by pending children, or remove a selected checkpoint just
because its reduce job completed. Never store full particle caches as `.npy`,
memmaps, temporary ROOT files, or large resumable optimizer checkpoints without
an explicitly different storage contract.

## 12. How another chat should prepare or extend a run

### 12.1 First determine whether this is inspection, reuse, or a new study

| Requested work | Correct starting point |
| --- | --- |
| Inspect current ladder results/progress | Existing canonical spec and live ledger; no new jobs |
| Retry profile only after preparation | Profile-attempt workflow; authenticate/reuse foundation |
| New 1M/1.5M/2M matching campaign | Corresponding existing split profile, new foundation and measured gate |
| New matching/scientific semantics | Versioned plan/contracts, new identities and appropriate preparation |
| New auxiliary-supervision study | Its own plan/roles/acceptance, shared compatible reader/split infrastructure |
| Restart failed science under identical code | Guarded same-source recovery, only when all old attempts are terminal |
| Resume while deliberately omitting ULTRADENSE | Requires an explicit scoped continuation/partial-report design; do not use whole-graph recovery blindly |

### 12.2 Pin source without disturbing active jobs

Inspect local changes before staging. This workspace has unrelated salience,
auxiliary-supervision, scratch, and documentation work. Do not `git add .`,
`git add -A`, reset, clean, or commit everything to get one study pushed.
The old scoped staging helper was written for an earlier diff; review its
allowlist before using it on a workspace containing new files in the same
namespace. Use explicit files/hunks and check the staged diff.

On Windows, commit/push only the intended changes and record the full hash.
On RC, fetch the appropriate branch and create a **fresh detached worktree**
at that exact pushed hash. For an existing worktree, verify both HEAD and
clean status; never update it in place to make new jobs use a fix. A current
`origin/main` that has advanced is not a reason to change a recorded source pin.

For new work, the general pattern is below. Replace placeholders deliberately;
this is not a request to create another copy of the already-running campaign.

```bash
(
set -euo pipefail
MAIN_REPO=/home/ryreu/atlas/HLT_Classification
JC2_COMMIT=PASTE_EXACT_PUSHED_40_CHARACTER_COMMIT
export PROJECT_DIR="/home/ryreu/atlas/HLT_Classification_jc2_NEW_${JC2_COMMIT:0:8}"

git -C "${MAIN_REPO}" fetch origin main
git -C "${MAIN_REPO}" merge-base --is-ancestor "${JC2_COMMIT}" origin/main
if [ -e "${PROJECT_DIR}" ]; then
  test "$(git -C "${PROJECT_DIR}" rev-parse HEAD)" = "${JC2_COMMIT}"
  test -z "$(git -C "${PROJECT_DIR}" status --porcelain)"
else
  git -C "${MAIN_REPO}" worktree add --detach "${PROJECT_DIR}" "${JC2_COMMIT}"
fi
export JC2_SITE=sporc_a100
source "${PROJECT_DIR}/sbatch/jetclass2_delphes_common.sh"
python -s "${PROJECT_DIR}/scripts/jetclass2_delphes_production.py" --help
)
```

Use the appropriate fetched remote branch if the fix was intentionally pushed
to a branch other than main. Local cleanliness and an available commit alone
are not substitutes for a pushed, reviewed source pin and matching evidence.

### 12.3 Preparation and evidence: exact entry points

The detailed initial procedure is retained in
[JETCLASS2_DELPHES_SPORC_READINESS.md](JETCLASS2_DELPHES_SPORC_READINESS.md).
Treat its initial resource guesses and initial staging commands as historical
instructions, not a reason to repeat completed work. Current CLI roles are:

| CLI | Purpose |
| --- | --- |
| [audit_jetclass2_delphes.py](../scripts/audit_jetclass2_delphes.py) | Inventory/schema/selection audit and source verification |
| [create_jetclass2_delphes_split_registry.py](../scripts/create_jetclass2_delphes_split_registry.py) | Build, inspect, or verify the reusable exact split bundle |
| [create_jetclass2_delphes_foundation.py](../scripts/create_jetclass2_delphes_foundation.py) | Create a foundation intent from an explicit subset profile/registry |
| [prepare_jetclass2_delphes_sporc.py](../scripts/prepare_jetclass2_delphes_sporc.py) | Create/dry-run/explicitly submit readiness-only sample, array, lock, profile |
| [prepare_jetclass2_delphes_debug_profile.py](../scripts/prepare_jetclass2_delphes_debug_profile.py) | New profile-only attempt reusing authenticated preparation |
| [jetclass2_delphes_production.py](../scripts/jetclass2_delphes_production.py) | Profile/create/dry/live/run/monitor/recover/results for production |
| [create_jetclass2_delphes_spine4_campaign.py](../scripts/create_jetclass2_delphes_spine4_campaign.py) | Historical scientific preview; **not the executable production submitter** |

Run each CLI's `--help` and its chosen subcommand's `--help` from the pinned
worktree before constructing a new command. The creation functions verify
raw-file hashes and semantic prerequisites; do not remove those checks because
333-file verification takes time.

A completed foundation can survive a new Git commit **only if** reuse
authentication finds unchanged relevant assignment semantics and valid shard
bytes/coverage. `authenticate_preparation()` checks that. The debug-only
attempt was designed for this case. A different matcher, input policy, split,
or class selection must not impersonate the old foundation by copying its
lock or changing its source hash.

### 12.4 Scientific creation, dry run, and explicit live authorization

Only after the chosen source/profile passes its gate, creation uses the
following interface. All variables must name the **new intended execution**;
`NEW_CAMPAIGN_ROOT` must not be the current campaign root.

```bash
python -s "${PROJECT_DIR}/scripts/jetclass2_delphes_production.py" create \
  --foundation-root "${FOUNDATION_ROOT}" \
  --data-root "${DATA_ROOT}" \
  --output-root "${NEW_CAMPAIGN_ROOT}" \
  --source-commit "${JC2_COMMIT}" \
  --runtime-profile "${PROFILE_ROOT}/evidence/runtime_profile.json"

python -s "${PROJECT_DIR}/scripts/jetclass2_delphes_production.py" submit \
  --spec "${NEW_CAMPAIGN_ROOT}/campaign_spec.json"
```

The second command is a **dry run** because `--execute` is absent. Inspect
`campaign_spec.json`, `command_plan.json`, and
`dry_run_submission_ledger.json`: counts, source pin, site, foundation/profile
parents, split memberships, CPU/RAM/GPU/time requests, dependencies, output
paths, and disk headroom. Do not invent JSON keys from a related old campaign;
the new spec embeds `foundation`, `scientific_plan`, and `runtime_profile`.

After user authorization for that exact audited campaign, the separate live
command is:

```bash
python -s "${PROJECT_DIR}/scripts/jetclass2_delphes_production.py" submit \
  --spec "${NEW_CAMPAIGN_ROOT}/campaign_spec.json" \
  --execute \
  --authorization-phrase 'AUTHORIZE EXACT JETCLASS2 DELPHES FOUR SPINE CAMPAIGN'
```

There are different phrases for different scopes:

```text
Readiness only:
AUTHORIZE JETCLASS2 DELPHES SPORC READINESS ONLY

Profile-only debug attempt:
AUTHORIZE JETCLASS2 DELPHES DEBUG PROFILE ONLY

Scientific four-spine execution:
AUTHORIZE EXACT JETCLASS2 DELPHES FOUR SPINE CAMPAIGN
```

Knowing a phrase is not standing authorization. No readiness job automatically
launches science. Do not rerun `create` on an existing output directory because
a later shell audit failed: inspect its already-published spec and dry/live
ledgers and continue from the appropriate stage.

Submission writes durable intent/receipt journals and a campaign-wide claim.
A network/SSH error can occur **after Slurm accepted a job**. An unmatched
submission intent must be reconciled against exact scheduler records; do not
delete the journal or blindly submit again. `submission_ledger.json` is the
authoritative task-to-job map once fully published, not a guessed contiguous
ID range or a truncated `squeue` name.

## 13. Cancellations, dependencies, and restart-zero recovery

### 13.1 The current ULTRADENSE request

The user requested cancelling ULTRADENSE only and retaining the other three
spines. The original graph has 15 ULTRADENSE fits and 14 associated reductions.
Their tasks contain the exact branch-qualified `JC2_ULTRADENSE_` node names.
Shared M0HLT/U000, U000 publication, DIRECT, COARSE, and DENSE are not targets.

Before doing anything, read the current campaign's authenticated live ledger
and live/accounting states. The record available to this guide confirms the
request and supplied cancellation command, **not successful cancellation**.
Do not recancel unrelated IDs from an old transcript or submit a replacement
four-spine graph just to drop one branch.

The three retained spines' fits do not depend on ULTRADENSE outputs. However,
the global **aggregate** depends on all registered fits, and **complete**
depends on aggregate. If ULTRADENSE is cancelled, these final bookkeeping jobs
may remain blocked. This is expected and does not prevent the retained fits
from training. Removing Slurm dependencies alone is not a fix: the aggregate
implementation itself requires every registered training report.

Use partial validation reporting while the retained spines run. A later clean
three-spine completion report should explicitly record omitted/cancelled tasks
under a separately designed, authorized reporting/continuation artifact. Do
not modify the original spec, fabricate missing reports, or label the original
59-task campaign complete.

### 13.2 What same-source recovery actually promises

[submission.prepare_recovery()](../src/hlt_classification/jetclass2_delphes/submission.py)
validates completed outputs and restarts unfinished tasks **from zero**. It
requires all jobs from old attempts to be terminal; active/unknown jobs in
another recovery also block it. It does not hot-patch source, cancel jobs,
resume optimizer state, or safely infer the user's intent to abandon a branch.

In particular, generic recovery selects **all unfinished registered tasks**.
After deliberate ULTRADENSE cancellation it can bring that unwanted branch
back. Do not use it blindly, and do not try it while the retained spines are
still running. A selective active-parent/source-changing continuation is a new
engineering task, not an undocumented option to this recovery CLI.

When cancellation is authorized, resolve exact IDs through the specific
campaign's ledger, validate ownership/current state and task membership, print
the target list, and cancel only those IDs. Never use `scancel -u ryreu`, a
job-name substring across projects, or a numerical ID range. Ordinary jobs
can be interleaved with other submissions. Preserve completed reports and
checkpoints; cancellation is not permission to delete their files.

## 14. Progress and results without waiting for aggregate

The `results --spec ...` command in section 4 prints completed validation
rows even before aggregate. It includes selected/actual passes, accuracy,
macro AUC, macro R50, and recovery relative to this campaign's fresh references.
Absent reports appear as `PENDING`; that means **no completed report**, not
proof Slurm still has a queued job. Use `monitor` and accounting to distinguish
pending, running, failed, cancelled, and completed-with-invalid-output states.

Successful tasks publish a canonical pointer at `tasks/TASK_ID.json`. Its
`result` contains the relative paths of that task's selected outputs; those
outputs live under `attempts/TASK_ID/ATTEMPT_ID/`. For a training task,
`result.training_report` points to `training_report.json` and
`result.checkpoint` points to `selected.pt`. Reducers publish `train_bank` and
`validation_bank` directories with manifests and probability shards. Resolve
these paths relative to the campaign root and validate their output inventory.

Do not assume the old `training/NODE/pmard_training_report.json` layout exists
here. `production.completed_task()` validates the pointer and output byte
hashes; `result_rows()` also validates the scientific report's node/foundation
lineage. These functions are the starting points for an additional printer.
An attempt directory or a partial file is not a published completed result.

For one exact job ID obtained from the live ledger:

```bash
# Replace with a current exact campaign-bound ID; this block is read-only.
JOB=PASTE_JOB_ID
sacct -j "${JOB}" -P \
  --format=JobID,JobName%70,State,ExitCode,Elapsed,Timelimit,AllocCPUS,TotalCPU,MaxRSS
scontrol show job -o "${JOB}" || true
sstat -j "${JOB}.batch" -P \
  --format=JobID,AveCPU,MaxRSS,AveDiskRead,AveDiskWrite || true
```

Read the actual `StdOut` path from `scontrol`, or locate `slurm-JOBID.out` under
the specific campaign/attempt root if the controller has aged the job out.
`scontrol: Invalid job id` after completion is not proof a job never existed;
use `sacct` and the ledger. Avoid an unrestricted whole-home recursive search.

Live log patterns come from
[cache.py](../src/hlt_classification/jetclass2_delphes/cache.py) and
[runner.py](../src/hlt_classification/jetclass2_delphes/runner.py):

```text
JC2 phase=cache role=... coordinate=... files=... resident_GiB=... seconds=...
JC2 node=... pass=.../... auc=... train_seconds=... validation_seconds=...
```

Cache time is preprocessing. A completed pass includes training and validation.
Use recent `train_seconds + validation_seconds` to estimate loop speed,
separately from startup. A `pass=.../100` denominator is a maximum, not a promise
early stopping will reach 100. Final selected-checkpoint metrics come from the
immutable report, not a maximum over rounded live-log values.

CPU utilization also depends on phase. Cache building uses bounded process
parallelism, capped by workers and available file tasks; it is not accelerated
by setting BLAS threads to the allocation size. GPU training/inference does not
need every host core continuously busy. `TotalCPU` for a running job may be
incomplete, and parent-job `MaxRSS` may be blank while `.batch` has a value.
When needed, compare two live `sstat` samples over a modest interval instead
of interpreting a zero accounting field as an idle job.

### 14.1 Metric meanings

[reporting.py](../src/hlt_classification/jetclass2_delphes/reporting.py) computes
unweighted accuracy and macro eleven-class one-vs-rest AUC. Per-signal QCD
rejection uses `p_signal / (p_signal + p_QCD)`, restricted to that signal and
QCD. Other signal classes are not its background. The 50%-efficiency threshold
is the descending signal rank `ceil(N_signal/2)`, including ties.

Macro R50 is the geometric mean over the ten signal-specific rejections, not
the arithmetic average. If no empirical QCD event passes a threshold, the
rejection is censored/null with a stated finite-sample resolution, not an
infinite JSON number. Required nonfinite model quantities fail; scientifically
poor finite metrics do not.

For accuracy/AUC or **linear R50**, recovery is:

```text
100 * (model_metric - fresh_M0HLT_metric)
    / (fresh_U000_metric - fresh_M0HLT_metric)
```

There is no clipping at 0% or 100%. Missing/censored/zero-denominator cases are
undefined, not zero. Always show absolute metrics alongside recovery. Do not
use old `M0CE60` or a historical CMS `U000` as the denominator on this dataset.
The exact new JSON field is `macro_r50`; old scripts expecting an old TRI60
metric layout need adaptation rather than silent fallback defaults.

## 15. Code map for implementation work

All core new-data semantics are under `src/hlt_classification/jetclass2_delphes/`.
Do not import a sibling worktree or `Fresh_check` to make them run.

| Code | Responsibility / reason to read it |
| --- | --- |
| [contracts.py](../src/hlt_classification/jetclass2_delphes/contracts.py) | Artifact families, declared assumptions, canonical row identity |
| [provenance.py](../src/hlt_classification/jetclass2_delphes/provenance.py) | Actual implementation-file hashes; HEAD alone is not clean-source proof |
| [inventory.py](../src/hlt_classification/jetclass2_delphes/inventory.py) | Latest cycles, source verification, schema/count inventory |
| [schema.py](../src/hlt_classification/jetclass2_delphes/schema.py) / [selection.py](../src/hlt_classification/jetclass2_delphes/selection.py) | Label order, raw branches, matched-HLT/source selection |
| [splits.py](../src/hlt_classification/jetclass2_delphes/splits.py) / [split_registry.py](../src/hlt_classification/jetclass2_delphes/split_registry.py) | File reservoirs, exact nested memberships and verification |
| [reader.py](../src/hlt_classification/jetclass2_delphes/reader.py) | Bounded paired or HLT-only ROOT reads, sealed test role |
| [inputs.py](../src/hlt_classification/jetclass2_delphes/inputs.py) | 17-feature interface, view-relative axes, no truncation |
| [views.py](../src/hlt_classification/jetclass2_delphes/views.py) | New-schema matching adapter, structural U edits and D interpolation |
| [foundation.py](../src/hlt_classification/jetclass2_delphes/foundation.py) | Compact assignments, sample audit, foundation lock |
| [cache.py](../src/hlt_classification/jetclass2_delphes/cache.py) | RAM packing, ordered process preparation, budgets, batch assembly |
| [model.py](../src/hlt_classification/jetclass2_delphes/model.py) | Installed Weaver adapter, environment fingerprint, CE/KD loss |
| [campaign.py](../src/hlt_classification/jetclass2_delphes/campaign.py) | Four paths, coordinates, paired seeds, optimizer/LR/stop recipe |
| [runner.py](../src/hlt_classification/jetclass2_delphes/runner.py) | Train/predict kernel, selection, early stopping, progress logs |
| [banks.py](../src/hlt_classification/jetclass2_delphes/banks.py) | Small probability shards with role/temperature/identity checks |
| [reporting.py](../src/hlt_classification/jetclass2_delphes/reporting.py) | New eleven-class metrics, censoring, recovery |
| [acceptance.py](../src/hlt_classification/jetclass2_delphes/acceptance.py) | Bounded acceptance and installed-Weaver parity |
| [execution.py](../src/hlt_classification/jetclass2_delphes/execution.py) | Site/resource/real-allocation/GPU validation |
| [readiness.py](../src/hlt_classification/jetclass2_delphes/readiness.py) | Readiness-only preparation/profile graph |
| [profile_attempt.py](../src/hlt_classification/jetclass2_delphes/profile_attempt.py) | Debug measurement reuse without preparation reruns |
| [production.py](../src/hlt_classification/jetclass2_delphes/production.py) | Authentication, runtime profiling, tasks, selected outputs, partial results |
| [submission.py](../src/hlt_classification/jetclass2_delphes/submission.py) | Exact dry/live commands, submission journals, monitor, recovery |

Thin workers are
[run_jetclass2_delphes_preparation.sh](../sbatch/run_jetclass2_delphes_preparation.sh),
[run_jetclass2_delphes_profile_attempt.sh](../sbatch/run_jetclass2_delphes_profile_attempt.sh),
and [run_jetclass2_delphes.sh](../sbatch/run_jetclass2_delphes.sh).
Inspect their actual arguments rather than reusing an old TRI60 wrapper.

The exact matcher, canonical Particle Transformer factory/configuration, and
atomic/hash primitives are deliberate reusable in-repository donors. Their
history is in [LEGACY_SOURCE_MAP.md](LEGACY_SOURCE_MAP.md). A reused mathematical
solver is not permission to reuse its old dataset-specific inputs or fitted
artifacts. New donor migrations must be recorded there.

## 16. Testing and final checklist for the next chat

Before code changes: follow the required reading order, inspect dirty state,
and run focused tests. The relevant original matching-migration suite is:

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
python -m pytest -q `
  tests/test_jetclass2_delphes.py `
  tests/test_jetclass2_delphes_training.py `
  tests/test_jetclass2_delphes_split_registry.py `
  tests/test_jetclass2_delphes_production.py `
  tests/test_jetclass2_delphes_sporc.py `
  tests/test_jetclass2_delphes_debug_profile.py
```

Use the configured scientific Python environment; the current local one is
`C:\Users\22rya\miniconda3\envs\tagging-hlt\python.exe`. There is no local
installed-Weaver/A100 claim. See [TESTING.md](TESTING.md) for the full ladder.
New study-specific tests are additional, not evidence that its real GPU gate
has run. Some existing scratch snapshots cause repository-wide Markdown-link
test failures; report them accurately rather than altering unrelated work.

Before proposing live work, verify all of the following:

- The user wants this action now; inspecting status does not authorize a run.
- Exact dataset snapshot, selection, class order, split/profile and ordinary
  role membership are explicit; test remains sealed.
- All compared methods at this size use the same exact training membership;
  every size shares the same exact validation/test memberships. Training-seed
  changes do not redraw splits. The intended progression is 500k, 1M, 1.5M, 2M.
- The source is pushed, pinned, clean, and not an active worktree being edited.
- The chosen foundation is authenticated and semantically compatible, or this
  study explicitly does not need matching and uses its own preparation.
- Real installed-Weaver/GPU evidence applies to this code, data size, site,
  model/loss and resource allocation; an unrelated miniature is not substituted.
- Outputs are isolated from raw data and other campaigns. Disk quota/headroom,
  RAM peak assumptions, selected checkpoints, and partial-attempt costs are clear.
- Dry commands show the intended account/partition/QoS/GPU/CPU/RAM/walltime,
  exact dependencies, and no unexpected branch or final-test task.
- Existing live jobs and deliberate cancellations are reconciled through the
  ledger. Recovery will not duplicate fits or resurrect an abandoned branch.
- The user receives explicit commands, expected output, and the next evidence
  to paste back, rather than an unsupported “queue ready” assurance.

### Copy-paste context to give another chat

```text
Please read docs/JETCLASS2_DELPHES_SPORC_AGENT_HANDOFF.md and the active
Delphes plan/contracts before working. This is Luka's JetClass2 Delphes
dataset on SPORC tier3 / reu-aisocial / qos_tier3 / one A100, using
/home/ryreu/miniconda3/envs/atlas_kd_sporc. It is NOT the old Tigris dataset.
The fixed registry defines nested 500k/1M/1.5M/2M training subsets with the
same 1M validation and 1M sealed test. Run setups at 500k first, then 1M,
then 1.5M, then 2M. ALL experiments at a given size must use the exact same
training jets; ALL sizes/methods must share the exact same validation/test
jets, not just equal counts. Reuse the frozen membership masks and hashes;
never redraw the data split for a method, campaign, or training seed.
Current matching production is 500k,
17 features, 11 classes, fresh references, single-parent logit KD, RAM views,
compact persistent banks and selected weights, no rolling resumes.
Read the actual campaign spec/ledger for source and job IDs. Real matching
profile job 21619139 passed, but it does not certify another study. The user
requested dropping ULTRADENSE while retaining the other three spines; verify
whether cancellation happened, and do not revive it with generic recovery.
Do not change other jobs, raw data, frozen splits or active source worktrees.
```
