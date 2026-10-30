# U100 embedding prediction used by the final classifier

Updated: 2026-10-30. Implemented small-data arm: **`u100_use_prediction`**.
Code inspected at `db9afe8b87cdef6f3489f9b85603f52cf7d2084f`; the multiview
implementation originated in `de79c7298a54dcb5818b97772635837e0a769914`.

The student predicts a frozen U100 teacher's jet embedding, then lets its own
trainable classifier use that prediction alongside its native HLT embedding.
The predictor stays in the deployed model. All deployed inputs remain HLT-only.

The retained variant belongs to the small-data multiview campaign. Its separate
FullStat Oscar adaptation is now implemented under `luka_fullstat_retained`:
see the [3M/500k,20-epoch plan](plans/LUKA_FULLSTAT_RETAINED_PLAN.md) and
[queue runbook](LUKA_FULLSTAT_RETAINED_RUNBOOK.md). The user chose retained first,
then a matched expanded-architecture CE-only control. The remainder of this
handoff documents the originating small-data graph and port requirements;
the FullStat plan governs the new campaign. The older FullStat frozen-readout
job remains a distinct model and cannot enable this variant with a flag.

Authority: [multiview plan, sections 7.7–7.8](plans/LUKA_U100_MULTIVIEW_READOUT_PLAN.md),
[versioned contracts](contracts/LUKA_U100_MULTIVIEW.md), and
[debug runbook](LUKA_U100_MULTIVIEW_RUNBOOK.md). This document does not redefine
any saved campaign or historical result.

## 1. Motivation and evidence

The user reported these selected-checkpoint results from FullStat readout job
8118608, on the same 500,000 validation jets:

| Model | Validation AUC | Historical AUC-gap recovery |
| --- | ---: | ---: |
| M0HLT | 0.96552387 | 0.0% |
| OFFLINE oracle | 0.96980545 | 100.0% |
| U100-w5 | 0.96656917 | 24.4% |
| Frozen-readout CE | 0.96638719 | 20.2% |

Readout CE was lower than U100-w5 by 0.00018198 AUC. These are user-supplied,
single-seed development results; this documentation task did not independently
replay their artifacts. They do not establish why readout CE underperformed.

The retained variant tests a different mechanism: whether supervised predicted
features help when the final classifier can learn how to use them. In the
readout experiment, the deployed classifier only used `h`; the teacher's fixed
classifier supplied an extra training loss. Here, the deployed classifier
learns from `[h, z_hat]` and no frozen-classifier loss is used.

Because `z_hat = g(h)` is deterministic, it adds no new measured information
beyond `h`. It adds a nonlinear feature transformation trained with privileged
supervision. Any improvement could also involve extra model capacity, which is
why the matched CE-only architecture control below is essential. This is a
plausible follow-up, not a prediction that recovery will reach 45–55%.

## 2. Exact model and loss

For paired jet `i`, define:

- `h_i`: the student's 128-dimensional pre-classifier jet embedding from HLT.
- `e_i`: the frozen COARSE_U100 teacher's 128-dimensional pre-classifier jet
  embedding, computed on that jet's U100 view.
- `z_i = (e_i - mean_train) / scale_train`: the standardized teacher target.
- `z_hat_i = g(h_i)`: the student's prediction in that standardized space.

The predictor is `Linear(128,128) -> GELU -> Linear(128,128)`. The final
classifier is a single `Linear(256,11)` operating on the concatenation:

```text
HLT particles -> student ParT -> h -----------------+
                                |                 |
                                +-> g(h) = z_hat -+-> concat -> learned classifier
                                            |
paired frozen U100 TRAIN target z ----------+-> SmoothL1
```

The complete student objective is:

```text
logits = W_h h + W_z z_hat + b
E = SmoothL1(z_hat, stop_gradient(z), beta=1, reduction="mean")
L = CrossEntropy(logits, ground_truth_class) + 5 * E
```

SmoothL1 averages over both jets and embedding coordinates. There is one CE
term, with coefficient 1, and total embedding weight 5. There is no native-head
KD, auxiliary-logit KD, frozen-readout CE/KD, or second classification CE.
The three-pass readout-loss ramp used by sibling arms does not apply here.
The ordinary registered optimizer/learning-rate schedule still applies.

Do not detach `z_hat` before classification. The encoder receives CE and
embedding gradients; the predictor receives embedding gradients and, once the
classifier uses its branch, CE gradients. The combined classifier receives CE
gradients. Teacher weights and exported targets receive none.

The combined classifier takes **standardized predictions directly**. It does
not invert their normalization and does not use the teacher's classifier.

## 3. Initialization and the matched control

Every student is fresh, with the authenticated baseline initialization and
sampler seeds. This is not continuation from a trained U100-w5 checkpoint.

`Student.__init__` initializes the combined classifier as follows:

```text
W_h = matched native classifier's fresh random weight
b   = matched native classifier's fresh random bias
W_z = zeros
```

The native logits are therefore preserved at initialization mathematically;
mixed-precision comparisons use numerical tolerances. Predictor initialization
uses an isolated RNG scope and `HEAD_SEEDS['U100'] = 20261019`, preserving the
native model's random stream. The old native `mod.fc` is frozen and its logits
are not used for the retained arm's loss. The current wrapper still executes
that native forward to obtain the embedding tap, then replaces its logits.

At the first backward pass, `W_z=0` means CE has no gradient into the predictor
through that branch. CE can immediately update `W_z`, while embedding loss
trains the predictor and encoder from the first update. Later CE gradients can
flow through the predictor too.

Use these three comparisons on identical data, seeds and training schedules:

| Arm | Training objective | Deployed classifier input |
| --- | --- | --- |
| `u100_w5` | CE + 5E | Native `h` |
| `u100_use_prediction` | CE + 5E | `[h, g(h)]` |
| `hlt_capacity_control` | CE only | `[h, g(h)]` |

The capacity control has the exact same predictor, combined classifier,
initialization and optimizer rules as the retained arm. Its extra branch is
simply a trainable HLT feature transformation; its training reads no teacher
target bank. The difference between the retained arm and this control tests
the contribution of embedding supervision within that architecture. Comparison
with `u100_w5` tests the value of retaining the supervised predictor.

## 4. Teacher targets and data boundaries

The small-data implementation reuses the **regular COARSE_U100 teacher** from
the fixed30 lineage (`74aa58c582ff6f497a2c6d91c503f5489fac3e9a`), authenticated
through the completed direct U100-w5 experiment
(`ed6d0af4be65f059356543f90583b30185702615`). This is not the hybrid ladder's
UE100 teacher or the native OFFLINE checkpoint.

`targets.export` freezes the teacher in evaluation mode and exports FP32
embeddings for TRAIN only. Standardization uses FP64 TRAIN population moments
and a standard-deviation floor of 0.001. The bank binds source, teacher,
preparation and recipe hashes; ordered jet identities and labels must match the
HLT training population exactly. Target tensors are supervision, never inputs.

The common bank also stores teacher logits and classifier parameters for
readout reconstruction checks and sibling arms. The retained arm's scientific
loss only consumes its standardized embedding targets.

The registered small-data run uses 299,234 TRAIN jets, 74,895 validation jets,
30 epochs and batch size 128. `original50k` overlaps validation and is
report-only. Final-test access remains sealed. A FullStat port must use its
own authenticated teacher and population; the small-data bank is not a
replacement for a 3M-jet bank.

## 5. Code map

Paths below are relative to the repository root.

| File | What to read or reuse |
| --- | --- |
| [multiview/model.py](../src/hlt_classification/luka_u100_multiview/model.py) | `Student`, `new_student`, `objective`, and `Student.deployed`: retained branch, initialization, gradients and complete deployment |
| [multiview/contracts.py](../src/hlt_classification/luka_u100_multiview/contracts.py) | `ARMS`, `HEAD_SEEDS`, `weights`, `node`, `recipe`, `TASKS`: registered identities, objectives and dependencies |
| [embedding_aux/model.py](../src/hlt_classification/luka_embedding_aux/model.py) | `head_input`: captures the input to the native 128-to-11 classifier during one ParT forward |
| [multiview/targets.py](../src/hlt_classification/luka_u100_multiview/targets.py) | `export`, `Bank`: frozen targets, normalization and authenticated row joins |
| [multiview/training.py](../src/hlt_classification/luka_u100_multiview/training.py) | `optimizer`, `loss_for`, `predict`, `fit`, `validate_fit`: schedule, selected graph, readback and costs |
| [multiview/worker.py](../src/hlt_classification/luka_u100_multiview/worker.py) | `run`, `completed`, `results`: task execution and artifact validation |
| [multiview/campaign.py](../src/hlt_classification/luka_u100_multiview/campaign.py) and [submission.py](../src/hlt_classification/luka_u100_multiview/submission.py) | Immutable registration, measured resource plan and campaign-bound jobs |
| [CLI](../scripts/luka_u100_multiview.py), [queue helper](../scripts/queue_luka_u100_multiview.sh), [Slurm worker](../sbatch/run_luka_u100_multiview.sh) | Thin operational entry points |
| [tests](../tests/test_luka_u100_multiview.py) | Retained initialization, objective, full miniature fit and deployed readback checks |

These files depend on the repo's Weaver adapter, prepared-data interfaces and
artifact contracts. Copying `model.py` alone does not create a standalone
training package. For another implementation, port the equations and graph
above into that setup's existing model and training interfaces.

## 6. Checkpoints, inference and results

`training.fit` selects on the **combined classifier's validation AUC**, using
the registered tie-breaks. The selected outputs under `<task>/fit/` include:

- `selected_full.pt`: complete selected student state.
- `selected.pt`: deployed state, including the predictor and combined classifier.
- `validation_predictions.npz`: ordered identities, labels and probabilities.
- `training_report.json`: history, selected metrics, diagnostics and costs.
- `fit_receipt.json`: hashes binding the published fit outputs.

The task directory also contains its task `receipt.json`. For this arm the
report records `deployed_graph = 'retained_predictor'`. Publication checks
strict state readback and selected deployed predictions/metrics.

**Do not call an older `native_state()` helper to strip the predictor.**
`Student.deployed()` retains the full graph. Setting `capture=False` does not
disable the predictor when `retained=True`.

After authenticating the checkpoint and its campaign, an inference sketch is:

```python
import torch
from hlt_classification.luka_u100_multiview.model import Student

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Student("u100_use_prediction")
state = torch.load(selected_path, map_location="cpu", weights_only=True)
model.load_state_dict(state, strict=True)
model.capture = False
model.to(device).eval()

# features, vectors, mask are the registered HLT tensors, already on device.
# Match the production worker's precision when reproducing saved predictions.
with torch.no_grad(), torch.autocast(
    device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
):
    logits = model(features, vectors, mask)
    probabilities = logits.float().softmax(dim=1)
```

No teacher, target bank or normalization moments are needed for deployment.
The predictor and combined classifier do add inference work. The training
report measures deployed parameters/state size, latency and memory; its latency
benchmark is the first validation batch with three warmups and ten forwards,
including transfer. This is not a production HLT latency certification.

Use historical anchors consistently:

```text
recovery_percent = 100 * (student_AUC - M0HLT_AUC) / (OFFLINE_AUC - M0HLT_AUC)
```

Also report the retained arm's AUC difference from the matched capacity control
and native U100-w5. Improved recovery against M0HLT alone cannot distinguish
privileged supervision from the benefit of the extra architecture.

## 7. Running the existing small-data implementation

Follow the [multiview runbook](LUKA_U100_MULTIVIEW_RUNBOOK.md) from the exact
registered, pushed source. Its helper has `accept` and `science` stages and
queues the full multiview study; it does not expose a standalone retained-arm
switch. If an external watcher already owns automatic science submission for
an existing campaign, let that path finish rather than creating duplicates.

`u100_use_prediction` depends on acceptance and `export_U100`, which reuses the
authenticated U100 bank exported during acceptance. It does not require the
new CONCAT/FUSION teachers. `hlt_capacity_control` needs acceptance only.
The worker's `run --task` command verifies the allocation and ledger; it is not
a login-node submission shortcut.

From that campaign's environment, with `PROJECT` and `SPEC` set to its recorded
source directory and `campaign_spec.json`, inspect saved results with:

```bash
python -u -B -s "${PROJECT}/scripts/luka_u100_multiview.py" results --spec "${SPEC}"
```

Look for `u100_use_prediction`, `hlt_capacity_control`, and `u100_w5` rows.
No scheduler or campaign changes are made by the results command.

## 8. Adapting this to FullStat

The implemented high-data comparison uses two fresh matched students (retained
and CE-only) and reuses completed U100-w5 as a historical reference, per the
user's explicit choice. Preserve FullStat's 3M TRAIN / 500k validation
split, 20-epoch schedule, batch size 128, preprocessing, seeds and selection
rules. Keep the final test and unused rows sealed. Reuse the exact completed
FullStat U100 teacher and bank only after source, target recipe, TRAIN moments
and ordered row identities validate; no teacher retraining is inherent here.

Useful FullStat donor interfaces already exist:

- [U100 bank code](../src/hlt_classification/luka_fullstat_u100_embedding/banks.py)
  provides the FullStat target schema and row authentication.
- [readout/source.py](../src/hlt_classification/luka_fullstat_readout/source.py)
  authenticates the original completed U100-w5 lineage.
- [readout/worker.py](../src/hlt_classification/luka_fullstat_readout/worker.py)
  provides `load_inputs`, original teacher replay, startup and artifact handling.
- [readout/training.py](../src/hlt_classification/luka_fullstat_readout/training.py)
  provides the high-data schedule and checkpointing patterns.
- [FullStat readout plan](plans/LUKA_FULLSTAT_READOUT_PLAN.md) and
  [contracts](contracts/LUKA_FULLSTAT_READOUT.md) define that existing experiment.

These donors require the following adaptations, now implemented in the separate
`luka_fullstat_retained` namespace:

1. Replace the old auxiliary-only student with the retained graph and its
   control. The historical U100-w5 reference uses the native graph.
2. Use CE + 5E for the retained arm; omit all frozen-readout terms. Keep the
   control CE-only, with identical expanded architecture and initialization.
3. Adapt prediction access: the multiview model uses `predictions['U100']`,
   which is not an interchangeable API with every older auxiliary wrapper.
4. Replace native-only selection/publication paths. Current FullStat readout
   training calls `model.native_state()` and declares
   `selected_state='native_backbone_only'`; startup and fit validators also
   construct native-only readbacks. All must support the actual retained graph.
5. Save and restore the predictor, combined classifier and relevant optimizer
   state in continuation checkpoints. Validate inference without teacher inputs
   and measure its added latency/memory on the execution hardware.
6. Register new scientific identities/contracts and source lineage. Do not
   relabel old readout artifacts, edit live specs, or substitute small-data
   moments. Preserve historical teacher replay batching/backend checks when
   authenticating the same bank.

If native architecture and inference cost must remain identical, this variant
does not meet that constraint. Plain embedding prediction from a new fusion
teacher is a separate option with its auxiliary head removed at deployment.

## 9. Verification and scope

For this handoff, the focused registered-inventory, retained-initialization and
complete miniature/deployed-readback tests for `u100_use_prediction` and
`hlt_capacity_control` passed: **4 passed, 23 deselected**. These local tests
establish implementation checks, not real-data gains or remote acceptance.

The initial handoff change added documentation and navigation only. The later
FullStat implementation is separately registered in the linked active plan;
its verification evidence is recorded in [HANDOFF](HANDOFF.md), and its donor
files/revisions are in [LEGACY_SOURCE_MAP](LEGACY_SOURCE_MAP.md). No remote
training or queue submission is performed by this implementation task.
