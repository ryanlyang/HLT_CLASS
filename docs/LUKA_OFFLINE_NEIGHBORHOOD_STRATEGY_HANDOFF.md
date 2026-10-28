# Luka: offline-neighborhood prediction on the K1/E40 baseline

Date: 2026-10-28. Status: **implemented locally; remote execution not performed**.

The additive `luka_neighborhood` package implements this fourteen-fit design.
The subsequent user request explicitly chose **debug with no separate
preflight**. The [active implementation plan](plans/LUKA_NEIGHBORHOOD_PLAN.md)
settles that operational amendment; scientific targets and the fixed baseline
are unchanged. See the [runbook](LUKA_NEIGHBORHOOD_RUNBOOK.md) for exact-source,
dry-review and queue instructions, and [HANDOFF](HANDOFF.md) for test evidence.
No running campaign, raw dataset or Fresh_check content was changed.

## 1. Recommendation in one paragraph

Keep the successful from-scratch HLT-only K1/E40/plain student, its four frozen
teachers, and all existing losses. Add a small training-only head to each
particle's final contextual representation. Ask that head to predict the
particle composition and transverse-momentum distribution that OFFLINE
reconstruction sees around that HLT particle's direction, at three angular
scales. My preferred candidate is **smooth neighborhoods, total auxiliary
weight 3** (`smooth_w3`). Smooth distance weights avoid abrupt target changes
at cone boundaries when HLT and offline directions differ slightly. This is a
reasoned prior, **not evidence that it will win**. Annular neighborhoods at
weights 3 and 10 are the essential, simpler alternatives.

The earlier discussion put annular runs first for interpretability. This
handoff prioritizes smooth_w3 as the single best-bet candidate while retaining
the same proposed 14-fit comparison. No new experimental result motivated that
ordering choice.

No new neural teacher is trained. No fusion, Hungarian matching, SupCon,
attention-matrix distillation, or additional ladder is introduced.

## 2. Why this is different from our token-prediction experiments

The existing depth screen predicts frozen c50 UE100 **learned particle vectors**
at blocks 2/4/8. Here we predict explicit **physical reconstruction summaries**
from native OFFLINE particles, not an embedding from any model.

For one HLT particle the question becomes:

> Around this HLT direction, how many charged hadrons, neutral hadrons, photons,
> electrons, muons and uncategorized particles does offline reconstruction see,
> and how much transverse momentum does each category contribute?

This provides 36 meaningful targets per real HLT particle. The targets are
correlated; 36 outputs are not 36 independent pieces of information. The
hypothesis is that local structure supervision complements the existing
task-specific logits and global teacher embeddings. More targets alone are
not a guarantee of a better classifier.

GN2 motivates physics-informed auxiliary tasks, but its track-origin and
common-vertex labels use simulation-level ancestry/vertices. Our proposed
neighborhood summaries are **not** substitutes for truth decay labels, and
this is not a reproduction of GN2. See the
[ATLAS GN2 paper](https://arxiv.org/abs/2505.19689).

## 3. Frozen baseline and interpretation of the reference results

The user supplied these completed-result numbers; they have not been fetched
or independently revalidated remotely during this documentation task:

| Reference | Full-validation AUC | Interpretation |
| --- | ---: | --- |
| M0HLT | 0.96056122 | Historical CE-only HLT reference |
| OFFLINE | 0.96452569 | Historical privileged reference, denominator anchor |
| K1_E40_PLAIN | 0.96315997 | Original grid winner; about 65.6% recovery |
| Repeated `baseline` | 0.96315546 | Current depth baseline; about 65.4% recovery |

Use the **repeated baseline** as the primary delta reference and retain the
original grid winner as a historical reference. Do not round either anchor
before calculating recovery:

```text
recovery_percent = 100 * (AUC - 0.96056122) / (0.96452569 - 0.96056122)
delta_repeat = AUC - 0.96315546
```

Import exact values and selected-checkpoint identities from authenticated saved
artifacts, not these printed decimals, when producing the eventual report.

Keep all of the following unchanged:

- 299,234 TRAIN jets; 74,895 full-validation jets; original 50,000 validation
  subset is overlapping and report-only. The former test was retired into
  TRAIN: **there is no independent test for this Luka trial**.
- Native D000 HLT inputs, full registered support, 17 input features and
  11-class order. Do not import generic ten-class or 128-particle defaults.
- Fresh initialization, original c50 DE000 backbone/head/sampler seeds,
  physical batch 128, all 30 epochs, inherited AdamW and warmup-3/cosine-30.
- Main classification head receives CE coefficient 0.5 only.
- Four independent auxiliary logit heads: total KD coefficient 1, T=2.
- Four independent jet-embedding heads: total coefficient 40, or 10 per
  teacher. Teachers remain original OFFLINE and c50 UE100/DE066/DE033.
- Existing frozen pre-classifier targets and their TRAIN-only normalization.
- Native full-validation main-head AUC selects, with the existing tie rule.
  No early stopping or selection by auxiliary loss.
- No relational loss, new SupCon teacher, token-embedding loss, weight transfer,
  or warm start. Existing auxiliary predictions never enter the classifier.

"Build on the baseline" means preserve its recipe, **not load its weights**.

Authority for these existing semantics is the
[depth scientific plan](plans/LUKA_SUPERVISION_DEPTH_PLAN.md),
[depth contract](contracts/LUKA_SUPERVISION_DEPTH.md), and
[K1/E40 grid plan](plans/LUKA_AUXILIARY_GRID_PLAN.md).
The [recovery plan](plans/LUKA_DEPTH_RECOVERY_PLAN.md) supersedes the depth
plan's serial scheduling only for the existing recovery; it does not change
the scientific baseline. This new document does not rewrite those artifacts.

## 4. Exact target definition proposed for v1

These numerical scales/transforms are explicit proposed design choices, not
measured detector resolutions or optimized values. Freeze them in a versioned
recipe before science; do not tune them by inspecting validation targets.

### 4.1 Inputs, geometry and particle identity

For paired TRAIN jet j, let H_j be its native HLT particles and O_j its native
OFFLINE particles. The query particles are **all real HLT particles**, in the
exact order used by the existing D000 model input. They are not U100 particles.

Use the original authenticated physical conversion and constituent acceptance.
Reconstruct the original D000 inputs and require their fingerprint, order,
offsets and jet identities to match before joining any targets. Do not introduce
new physics cuts, matching, sorting tie rules, particle truncation or HLT changes.
Neighborhood construction does not use the U/D assignment indices.

Compute directions from native four-vectors in a common coordinate system:

```text
pt = hypot(px, py)
eta = asinh(pz / pt)
phi = atan2(py, px)
delta_phi = wrap_to_minus_pi_pi(phi_offline - phi_hlt)
d(i,k)^2 = (eta_offline[k] - eta_hlt[i])^2 + delta_phi(i,k)^2
Q_j = sum(pt_hlt[i] for all real HLT particles i)
```

Use precisely the inherited valid-particle rules; reject an unexpected
nonpositive pt needed by this formula, Q_j <= 0, or nonfinite real input rather
than silently dropping it. Do not compare separately recentered HLT and offline
angles. Do not use the source `deta`/`dphi` branches as interchangeable axes.

Use the existing reconstructed categories, in this order:

```text
charged hadron, neutral hadron, photon, electron, muon, other/unknown
```

The final category is the existing all-zero known-PID-flag case. Unknown PID
is a real category, not padding. Validate flags/charge with the existing
`identity_categories` semantics. These are reconstruction categories, not
truth origin labels.

Include every accepted OFFLINE particle, even if it has no HLT counterpart.
Do not exclude an offline particle because it is near or matched to the query.
For HLT-only controls, include the query particle itself as well; this keeps
the definition consistent and explicitly allows the control to exploit its
own measured properties. Self-exclusion would be a separate experiment.

### 4.2 Preferred smooth kernel B

Use three Gaussian radial weights with widths:

```text
sigma = [0.02, 0.05, 0.10]    # angular delta-R units
K_s(d) = exp(-0.5 * (d / sigma_s)^2)
```

Evaluate the formula over all real context particles; no nearest-k selection,
hard cutoff, per-particle matching, or kernel normalization over neighbors.
Underflow to zero for very distant particles is harmless. These are overlapping
soft neighborhoods, not disjoint rings; their weighted counts can be fractional.

For each HLT query i, scale s and category c:

```text
C[i,s,c] = sum_k K_s(d(i,k)) * indicator(category[k] == c)
F[i,s,c] = sum_k K_s(d(i,k)) * indicator(category[k] == c) * pt[k] / Q_j
```

The common denominator Q_j comes from HLT and is positive. F is a local scalar
pt contribution relative to HLT scalar pt, not a probability distribution;
do not impose a softmax or require that its entries sum to one.

### 4.3 Annular kernel A and the HLT controls

Use disjoint intervals `[0,0.05)`, `[0.05,0.10)`, `[0.10,0.20)`; the kernel is
one inside its interval and zero outside. Boundary equality goes into the
outer interval; d=0.20 is outside. Compute the same C and F quantities.

For `hlt_ann_*`, replace the OFFLINE context set by H_j without changing the
query particles, Q_j, head, transforms or reduction. Thus these controls ask
whether structural supervision alone helps, without per-jet offline targets.

### 4.4 Transforms, normalization and the correction variant C

For A and B, transform quantities before normalization:

```text
x_count = log1p(C)
x_pt = log1p(100 * F)
```

The factor 100 is a fixed numerical scaling choice, not a physical calibration.
Use FP64 geometry/reductions, cast transformed outputs to stored FP32, then
fit deterministic FP64 moments on those stored values. Flatten channels as
18 counts followed by 18 momentum values, with scale-major/category-minor order.

For each of the 36 channels use **equal-jet-weighted TRAIN moments**:

```text
mu_c = mean_j(mean_real_particles_i(x[j,i,c]))
var_c = mean_j(mean_real_particles_i((x[j,i,c] - mu_c)^2))
scale_c = max(sqrt(var_c), 0.1)
z[j,i,c] = (x[j,i,c] - mu_c) / scale_c
```

Handle only roundoff-sized negative variance as numerical zero; reject invalid
statistics. The fixed floor 0.1 is deliberate for these new transformed
neighborhood channels: it limits amplification of almost-always-empty categories.
It **does not change** the existing jet-embedding scale floor of 1e-3. Do not
drop absent/rare channels, infer missing measurements from numerical zero, fit
moments on validation, or clip large finite targets without a versioned change.

Fit separate moments for OFFLINE-annular, OFFLINE-smooth and HLT-annular targets.
The HLT control uses only HLT TRAIN moments. Compare raw/weighted loss scales
as well as performance; normalization does not guarantee identical gradient
magnitudes across target families.

Variant C predicts the annular **transformed-space correction**:

```text
z_correction = (x_offline_annular - x_hlt_annular) / scale_offline_annular
```

Do not log a signed difference, subtract the offline mean again, or restandardize
the residual to unit variance: that could greatly amplify tiny reconstruction
differences. Correction inputs to the head are still just the student's tokens;
neither descriptor is injected into the classifier or the head. This is a
different regression parameterization of related information, not extra truth.

### 4.5 Exclusions

No impact-parameter, vertex, ancestry, subjet-clustering or class-conditioned
targets in v1. The raw reader contains tracking fields, but their validity
conventions and physical interpretation deserve a separate audit before use.
Do not construct labels from a jet's class or use classes to choose radii,
neighbors, query particles, normalization or target weights.

## 5. Student attachment and loss

Tap the output of particle block 8, **before class-attention pooling**, using
the existing native-preserving hook pattern. Each real particle has a 128-D
contextual vector t_i. Add one shared-per-particle MLP:

```text
128 -> Linear(128) -> GELU -> Linear(36)
```

No new attention layer, batch normalization or dropout. Initialize the new
head with the same registered seed in all 14 arms (proposed seed 94271), inside
an isolated CPU RNG context. Construct it without changing backbone, existing
head, dropout or sampler random streams. Existing optimizer grouping must
remain identical for existing parameters; add only the new head's parameters.

Predictions and auxiliary losses run in FP32; the backbone keeps its original
mixed-precision policy. Targets/moments are detached and never learnable.
Training heads may consume HLT-derived states only.

Let L_N be SmoothL1 with beta=1, averaged over the 36 channels, then real HLT
particles within each jet, then jets. Equivalently, give counts and momentum
families one half of the neighborhood loss each. Padding is excluded; genuine
empty-region zeros remain supervised. Never sum the loss over all particles
in the batch, because that would favor high-multiplicity jets.

With four teachers t, the complete objective is:

```text
L = 0.5 * CE(main_logits, labels)
  + (1/4) * sum_t [ 2^2 * KL(q_t(T=2) || softmax(aux_logits_t / 2)) ]
  + (40/4) * sum_t SmoothL1(jet_predictor_t(h), frozen_jet_target_t)
  + lambda_N * L_N
```

Preserve the baseline's exact KD direction, reductions, T-squared applied once,
teacher summation order and embedding normalization. lambda_N is the **total**
neighborhood coefficient, not a coefficient per scale/category/particle.
Coefficients are not percentages of gradient contribution. No whole-loss
renormalization, adaptive weighting, gradient surgery or LR change in v1.

Keep native classification as the only deployed computation. Offline summaries,
predicted summaries, teacher embeddings and auxiliary logits are never fed
back into its forward path. Auxiliary gradients update the shared encoder;
the classifier parameters receive exactly the CE0.5 gradient. Strict native
checkpoint loading must prove that all auxiliary heads were removed.

## 6. Proposed 14-fit screen and priority

All rows use the same seeds and fixed baseline above. A second-seed follow-up
would be a separately approved matched baseline/finalist comparison, not a
silent extra row or selection of the better seed.

| Priority | Arm | Target | lambda_N |
| ---: | --- | --- | ---: |
| 1 | smooth_w3 | B: smooth OFFLINE | 3 |
| 2 | smooth_w10 | B: smooth OFFLINE | 10 |
| 3 | ann_w3 | A: annular OFFLINE | 3 |
| 4 | ann_w10 | A: annular OFFLINE | 10 |
| 5 | hlt_ann_w3 | A: annular HLT control | 3 |
| 6 | hlt_ann_w10 | A: annular HLT control | 10 |
| 7 | correction_w3 | C: annular correction | 3 |
| 8 | correction_w10 | C: annular correction | 10 |
| 9 | smooth_w1 | B: smooth OFFLINE | 1 |
| 10 | ann_w1 | A: annular OFFLINE | 1 |
| 11 | correction_w1 | C: annular correction | 1 |
| 12 | smooth_w30 | B: smooth OFFLINE | 30 |
| 13 | ann_w30 | A: annular OFFLINE | 30 |
| 14 | correction_w30 | C: annular correction | 30 |

This is the discussed 12 OFFLINE-derived variants plus two HLT controls, not
another 20-run campaign. The upper weights are informative stress points, not
the strongest performance prediction. Keep all finite results, including poor
ones. Do not cancel or omit rows because interim AUC is disappointing.

Reuse completed controls and compact teacher targets where authenticated.
Before relying on the reused baseline, require zero-new-loss native forward,
gradient, optimizer-step, RNG and complete synthetic fixed30 parity with the
existing baseline implementation. If parity fails, stop and investigate;
do not quietly treat a changed baseline as matched or queue another control
without revising the plan.

## 7. Data boundaries, target storage and safe retirement

Target generation reads paired **TRAIN only**. Validation uses ordinary HLT
classification inference, without neighborhood targets. Existing saved oracle
reference predictions may be imported for reporting. No test access or split
change. Preserve existing provisional source conventions and report limitations.

Build the three distinct transformed banks once: OFFLINE-annular,
OFFLINE-smooth, HLT-annular. Derive corrections from the first and third bank
at load time; do not store 14 copies. Use canonical ragged offsets and ordered
jet identities, and join to exact D000 particle order. Labels may be checked
for join integrity but must not enter target construction.

For S TRAIN HLT particles, FP32 descriptor payloads alone require:

```text
one bank:       S * 36 * 4 bytes
all 3 banks:    S * 432 bytes
```

For illustration, S=10 million means about 4.02 GiB for all three payloads,
before indexes, atomic-publication overlap, outputs and safety reserve. This
is an example, not a measured population size. Count actual authenticated
particles and budget before allocating. Personal quota is not simply the
filesystem-wide `df` value. Never silently reduce support, quantize to FP16,
skip an arm or use another directory to evade an admission failure.

Stream jets and chunk query/context distances. Do not persist or materialize
a dataset-wide pair matrix. Compute all required target families during one
selected-TRAIN pass; keep no new persistent physical-particle cache. The
neighborhood loss itself is O(number of HLT tokens * 36); target construction
can still require O(n_HLT * n_OFFLINE) work per jet and must be timed.

An immutable target manifest must bind source commit/files, split/prepared
hashes, geometry/categories/kernels/transforms, HLT-input fingerprint, per-jet
offsets/order, FP64 normalization, payload byte counts/SHA256 and role. Publish
atomically and reject partial, corrupted or wrong-role banks. Missing borrowed
payloads require explicit authenticated retirement/rebuild handling, not an
existence-only fallback.

Own all new outputs in a fresh campaign namespace. Borrowed old banks/models,
current recovery outputs, datasets and Fresh_check remain read-only. Do not
delete or alter their claims, worktrees, manifests or jobs.

Register eventual retirement of **only new owned neighborhood payloads** after
every consumer is finished, required results/predictions validate, and no retry
is using them. Retain manifests, normalization, reports, validation predictions,
all epoch histories, exact accounting and deletion receipts. Readers must
explicitly support authenticated retirement; do not make reporting depend on
deleted payloads or fake their continued presence. A failed/partial campaign
must not trigger unconditional cleanup.

Discard training-only head checkpoints and temporary probes. Retain compact
native selected student checkpoints through confirmation/reproducibility review;
any later model retirement needs an exact, dependency-checked plan. No deletion
is authorized by writing this document.

## 8. Debug execution and explicit no-separate-preflight amendment

Use SPORC `debug`, account `reu-aisocial`, and the established x86 environment
`/home/ryreu/miniconda3/envs/atlas_kd_sporc`. The historical Luka A100 allocation
uses `--qos=qos_tier3`; the QoS name does **not** change the requested partition.
Do not use unrelated generic Tigris/ARM defaults from older repository plans.

Suggested starting envelope to measure, not established acceptance:
one A100, six CPUs, 90000 MiB, up to the prior 12-hour allowance per GPU task.
Exact requests must follow current site limits and measured new-objective
memory/throughput, including target I/O. Do not claim a fixed runtime yet.

Require local installed-Weaver parity and production-worker synthetic miniatures.
By the subsequent explicit user authorization, queue one shared preparation,
14 student fits and one report together, without a separate preflight job.
Real-data parity, maximum-support memory and throughput admission occur inside
preparation and each dependent fit before scientific epochs. This replaces the
original proposal's requirement to measure before submission; it does not waive
input, lineage, finite-value or resource checks. Probes are discarded and all
RNG/optimizer state reset before science. Local tests are not genuine SPORC
acceptance or a claim of future AUC improvement.

Maximum-support memory probes are distinct from eight seeded real-population
timing batches. The runtime projection uses the slowest sampled train/validation
batch, a factor of two, thirty training passes, thirty-three validation passes
and a twenty-minute publication reserve, plus elapsed startup time. This is a
conservative estimate, not a guarantee. It must fit the registered allowance.

Run student priorities in two lanes (odd and even rows above). Each fit needs
the shared targets successfully authenticated. Capacity-only edges use
afterany across earlier jobs in the same lane; actual scientific dependencies
use afterok. Do not make one failed independent fit block the whole experiment,
and do not assume submission order guarantees global scheduler priority.
The final report waits for all fits to terminate and labels incomplete rows.

Retain full dry plans, exact pushed source, immutable hashes, exclusive claims,
accepted numeric job IDs and durable partial-submission journals. No automatic
retry, cancellation or modifications to the running depth/Fresh_check jobs.
All workers set PYTHONNOUSERSITE=1, disable bytecode, prepend CONDA_PREFIX/lib
to LD_LIBRARY_PATH and source helpers via the absolute pinned project path.

Use only the new `scripts/queue_luka_neighborhood.sh` from the exact pushed
implementation commit, as documented in the runbook. It reviews all sixteen
jobs and uses the durable exact-ID submitter. It does not rerun the existing
baseline or compact teacher exports.

## 9. Implementation and inherited code map

Inspected local revision: `5d55d602549ff47c6301e74b712786650c3bf8f4`.
This is a provenance anchor for the existing interfaces, not a neighborhood
implementation commit. Actual internal donors are recorded in
[LEGACY_SOURCE_MAP](LEGACY_SOURCE_MAP.md); no external source was migrated.

| Existing surface | Reuse or inspect for |
| --- | --- |
| [depth contracts](../src/hlt_classification/luka_supervision_depth/contracts.py) | Exact baseline recipe, teacher identities, seeds and class count |
| [grid model](../src/hlt_classification/luka_auxiliary_grid/model.py) | Native ParT, jet/KD heads and RNG isolation |
| [depth model](../src/hlt_classification/luka_supervision_depth/model.py) | `tapped_forward` block-8 hook, mask/order checks and native-state extraction |
| [depth training](../src/hlt_classification/luka_supervision_depth/training.py) | Fixed30 loop, loss arithmetic and native validation selection |
| [raw particles](../src/hlt_classification/luka_fullsim/particles.py) | `FIELDS`, `identity_categories`, physical conversion and paired TRAIN reading |
| [regular cache](../src/hlt_classification/luka_fullsim/regular_cache.py) | Authenticated selected-row streaming and D000 identity/order |
| [input transform](../src/hlt_classification/literature_context/transform.py) | Original constituent acceptance, input vectors and ordering |
| [depth data](../src/hlt_classification/luka_supervision_depth/data.py) | Explicit authenticated RAM-only OFFLINE rebuild pattern |
| [depth banks](../src/hlt_classification/luka_supervision_depth/banks.py) | Compact teacher-target validation; do not repurpose token-bank schema |
| [recovery worker](../src/hlt_classification/luka_depth_recovery/worker.py) | Import baseline/targets without requiring unrelated recovery fits to finish |
| [recovery submission](../src/hlt_classification/luka_depth_recovery/submission.py) | Two-lane exact-ID scheduling and partial submission journals |
| [cache contracts](../src/hlt_classification/data/cache_contracts.py) | Hashes and atomic publication |

Known remote references, to authenticate rather than trust by path:

```text
Original depth campaign:
/home/ryreu/atlas/HLT_Classification/checkpoints/luka_depth_9e90b7de_debug_r1
Repeated baseline: baseline/fit/{training_report.json,selected.pt,validation_predictions.npz}
Original baseline receipt: baseline/receipt.json; accepted job 21973954

Current recovery:
/home/ryreu/atlas/HLT_Classification/checkpoints/luka_depth_recovery_5d55d602_debug_r1
Recovery pinned source:
/home/ryreu/atlas/HLT_Classification_depth_recovery_5d55d602
```

Discover older grid/c50/prepared/raw paths through authenticated parent specs,
not globs or the newest timestamp. Import only the completed baseline/targets
dependency closure; unfinished diversity/token fits are not prerequisites.
Do not rewrite old validators to tolerate deletions or different recipes.

New implementation:

```text
src/hlt_classification/luka_neighborhood/
  contracts.py       # versioned scientific recipe and 14 arms
  targets.py         # deterministic physical neighborhood definitions
  banks.py           # streaming build, equal-jet moments, authenticated reads
  model.py           # native-preserving particle head
  training.py        # unchanged baseline plus normalized neighborhood loss
  startup.py         # discarded parity, gradient, memory and timing probes
  source.py          # scoped read-only parent imports
  campaign.py        # immutable spec, quota and acceptance
  submission.py      # exact two-lane dry/live plans
  worker.py          # preparation, fit, native readback and report
  retention.py       # separately authorized exact owned-payload retirement
scripts/luka_neighborhood.py
scripts/queue_luka_neighborhood.sh
sbatch/run_luka_neighborhood.sh
tests/test_luka_neighborhood.py
docs/plans/LUKA_NEIGHBORHOOD_PLAN.md
docs/contracts/LUKA_NEIGHBORHOOD.md
docs/LUKA_NEIGHBORHOOD_RUNBOOK.md
```

New `LUKA_NEIGHBORHOOD_* /v1` contracts bind these artifacts. Old scientific
runtime remains byte-identical. No runtime import from Fresh_check.

## 10. Required tests, diagnostics and acceptance

Before calling the implementation queue-ready:

1. Hand-calculated target tests: categories, unknown PID, one particle, empty
   regions, exact annular boundaries, smooth weights, delta-phi wrap, common
   axis, momentum denominator, unequal HLT/OFFLINE counts and unmatched context.
2. Context permutation invariance and query permutation equivariance; exact
   native HLT ordering/fingerprint; padding cannot enter targets or reductions.
3. TRAIN-only access, label-blind target generation, HLT-only control statistics,
   zero-difference correction when views are identical, deterministic chunk/
   worker replay and exact channel order.
4. Equal-jet moments and loss, rare/absent channels, empty-region supervision,
   floor behavior, finite-required inputs, byte corruption and bad joins.
5. Zero-new-loss baseline RNG/forward/gradients/optimizer/fixed30 parity. Verify
   that the new head has nonzero connected gradients when its loss is active,
   teacher targets have none, and main-classifier gradients equal CE0.5 alone.
6. Strict native checkpoint round trip, original inference logits, no new input
   fields, no auxiliary outputs/heads, and HLT-only validation execution.
7. Synthetic production-worker build/fit/report/retirement replay; source drift,
   race/partial publication, two-lane failure isolation and exact-ID guards.
8. Installed-Weaver CPU/CUDA parity and BF16 finite checks locally. Actual
   batch128 maximum-support A100 memory and real-data timing admission inside
   authorized workers, under the section-8 direct-debug amendment.

Log every completed epoch's main validation AUC/accuracy, selected epoch,
baseline-relative AUC and recovery, raw/weighted loss families, learning rate,
elapsed training/validation time, memory peaks and current epoch batch progress.
Record per-category/scale auxiliary errors as diagnostics, not selection rules.
Gradient-norm diagnostics, if added, belong in disposable probes and must not
advance scientific RNG or silently change optimization.

Report the same selected native checkpoint on full validation and original50k.
Compare against repeated baseline, original K1/E40, CE and OFFLINE, clearly
separating these references. Lower neighborhood loss alone is not a win.
The 14-way comparison is exploratory on heavily reused validation. Original50k
is not independent confirmation; confirm any promising finalist with matched
baseline seeds before claiming a reliable gain. Do not retrofit an independent
test label onto any of these rows.

## 11. Next task and explicit non-goals

Next: review and commit/push the scoped implementation, then use the exact
debug runbook and dry-plan approval. Real target size and new-objective resource
admission remain worker checks; no genuine SPORC measurement is claimed yet.
If data availability, acceptance or quota requires a design change, document it
and get direction before replacing the registered full screen or scientific
semantics.

No changes to old runs, no new teachers, no fusion network, no extra HLT input
features, no pretraining stage, no resplitting and no test evaluation. Local
implementation readiness does not imply remote jobs were submitted or passed.
