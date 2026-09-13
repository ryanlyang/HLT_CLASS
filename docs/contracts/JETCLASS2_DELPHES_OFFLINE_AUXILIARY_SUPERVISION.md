# JetClass2 Delphes offline auxiliary supervision — executable v1

Scientific authority: [500k four-arm plan](../plans/JETCLASS2_DELPHES_OFFLINE_AUXILIARY_SUPERVISION_500K_PLAN.md).
Implementation: [offline_aux](../../src/hlt_classification/jetclass2_delphes/offline_aux/).
This contract is separate from the matching/four-spine migration. No existing
scientific graph, environment, data, job, or immutable artifact is modified.

## Scientific contract

- Native HLT inputs: the existing 17-feature, own-axis Delphes transform,
  canonical native order, capacity 240, no truncation. Eleven existing classes.
- Frozen TRAIN_500K profile, registry
  `72e8b76555e3707e90aa7ea46d521c93a2b801e46ee94c35ecee978e324d7d01`.
  One million outer validation rows are deterministically split into exactly
  200k VAL_SELECT and 800k VAL_REPORT. Final-test readers do not exist here.
- Offline inputs occur only in target construction. No constituent matcher,
  offline teacher, KD logits/temperature, ensemble, or representation bank.
- Twenty-eight FP32 targets: five log-counts, five species pT fractions, two
  log-transformed normalized structure scalars, eight radial fractions and
  eight pair-angle fractions. Pair masks handle native single-particle jets.
  The exact formulas, numerical tolerances and bins remain in the plan.
- TRAIN-only FP32-to-FP64 two-pass population moments for seven scalar columns;
  minimum standard deviation 0.001. TRAIN-only native HLT pT quintile boundaries.
- CE has coefficient one. COMP and STRUCT each add their registered group
  loss; BOTH adds one half of each, under the same total auxiliary coefficient.
  Lambda is one of 0.1, 0.3 and 1.0, constant throughout the fit.
- Same native HLT ParT/class head in every arm. A temporary `mod.fc` pre-hook
  reads its final normalized 128-vector without copying the Weaver forward.
  Five independent 128→128→output GELU heads are training/diagnostic-only.
  The hook is removed in `finally`. The deployable object is the original
  backbone with an HLT-only forward signature.
- Batch 256, AdamW with explicit non-fused/non-foreach semantics, BF16 forward
  and FP32 losses. Three-pass warmup, hold to 45, cosine to 60, floor thereafter;
  min60/max100/patience15/delta5e-5. Exact best-checkpoint selection and the
  patience reference are separate. Only one final selected weight payload is
  published. No optimizer, last, resume or per-pass checkpoint files.
- Ten discovery fits under one shared replicate; three fresh paired
  confirmation replicates × four arms = twelve fits. Exactly 22 scientific fits.
  The 432 registered initialization/sampler/training/head/per-pass seed values
  are materialized in the study spec and collision-checked.
- TF32 and cuDNN benchmarking are disabled by workers. Kernel-level universal
  determinism is not claimed; the full installed environment is acceptance-bound.

The installed Weaver source must expose the specified final classifier surface.
The [upstream implementation](https://github.com/hqucms/weaver-core/blob/main/weaver/nn/model/ParticleTransformer.py)
is a design reference, not a substitute for the mandatory local-installation
hash/parity evidence. No external donor source was vendored.

## Artifact namespace and integrity

New artifacts use `JETCLASS2_DELPHES_OFFLINE_AUX_<KIND>/v1`, schema version 1,
canonical content hashes and `final_test_accessed=false`. Families implemented:

| Families | Purpose |
| --- | --- |
| STUDY_SPEC, MODEL, RECIPE, METRICS, TARGET_DEFINITION | Frozen science, seed register, source, inputs and site |
| ROLE_SPLIT, SAMPLE_PROFILE | Authenticated row capabilities and measured CPU preparation estimate |
| TARGET_SHARD, TARGET_BANK, NORMALIZATION, PREPARATION_LOCK | Compact ordered targets and TRAIN-only calibration |
| EXECUTION_ACCEPTANCE | Genuine installed-Weaver/A100 checks and measured resource envelope |
| STAGE_SPEC, COMMAND_PLAN | Concrete staged DAG and allocation/dependency arguments |
| SUBMISSION_LEDGER, SUBMISSION_INTENT, SUBMISSION_RECEIPT | Canonical dry run and durable exact-job submission |
| TRAINING_REPORT, SELECTED_WEIGHTS, TRAIN_OUTPUT | Selected validation checkpoint, provenance and whole history |
| CONFIGURATION_LOCK, REPORTING_LOCK | All-ten discovery selection and all-twelve reporting authorization |
| EVALUATION_REPORT, BOOTSTRAP, AGGREGATE, COMPLETION | Registered paired single-model results and uncertainty |
| TASK_RECEIPT, MONITOR, RECOVERY | Checksum-authenticated completion and restart-zero recovery |

Descriptors contain relative paths, byte sizes and SHA256 checksums. Arrays
use deterministic NPZ serialization without pickle or memory mapping. Identities
are full 32-byte native row digests. Target loading checks exact ordered identity
equality, not just counts. Role payloads also bind file indices, entries and labels.
Target payloads and semantic manifests are byte-identical across worker counts;
operational elapsed time goes to stdout, not target scientific identity.

Source validation requires an exact 40-character commit, clean checkout and a
locally known origin ref containing the commit. Every production task repeats
the check. Existing historical source pins are not silently upgraded.

## Access boundary

TRAIN and VAL_SELECT metadata/targets can be prepared before fitting. Neither
models nor target construction may iterate VAL_REPORT without the reporting
lock containing all twelve registered confirmation model identities. Metadata
needed to define the internal split can be read earlier; it does not score or
transform report particles. There is no final-test role, boolean bypass or CLI.

ROOT baskets/chunk arrays can physically contain neighboring entries. The
capability is enforced at selected-row interpretation/yield, transformation and
scoring, not a claim that ROOT decompresses no neighboring basket bytes. The
reader verifies role-array hashes and never yields neighboring rows. Internal
validation slices can share source files; event independence remains provisional.
Prior exploratory use of outer validation is not erased by defining new masks.

All confirmation checkpoints are fixed before any report target or inference
job starts. Offline target banks never enter the classifier forward. Evaluation
loads only the checkpoint named by the reporting lock, checks its payload and
tensor-state hashes, and supplies native HLT features/vectors/masks.

## Concrete stages and queue boundary

The CLI is [jetclass2_delphes_offline_aux.py](../../scripts/jetclass2_delphes_offline_aux.py).
The worker is [run_jetclass2_delphes_offline_aux.sh](../../sbatch/run_jetclass2_delphes_offline_aux.sh).
Stage creation, dry run and live authorization are separate operations:

| Stage | Jobs | What it can do |
| --- | ---: | --- |
| GATE | 1 CPU | Freeze internal roles and time 1,024 TRAIN target rows |
| PREPARE | 7 CPU shards + normalization + 1 GPU profile = 9 | Build TRAIN/SELECT targets, calibrate TRAIN, genuine Weaver/A100 parity and resource pass |
| DISCOVERY | 10 GPU fits + configuration lock = 11 | Select one lambda per auxiliary arm on VAL_SELECT |
| CONFIRMATION | 12 GPU fits + reporting lock + 8 CPU target shards + 12 GPU evaluations + 20 CPU bootstrap shards + aggregate + completion = 55 | Frozen three-seed paired reporting; no ensemble/test |

No stage submits the next stage. In particular, PREPARE does not launch any
scientific fit; its one-pass probe cannot be reused as a scientific checkpoint.
The configuration lock requires every discovery report, including losing ones.
Scientific quality never fails operational acceptance, blocks a registered arm
or changes completion. Invalid/nonfinite required data, stale source, provenance
errors and insufficient resource margins fail closed.

Initial GPU profile: SPORC, tier3, reu-aisocial, qos_tier3, one A100, one task,
8 CPUs/workers, 72 GiB, four hours. CPU sample/target tasks request 8 CPUs/8 GiB;
normalization/locks use one CPU/8 GiB; bootstrap uses one CPU/16 GiB. Workers
source the existing absolute-path, isolated `atlas_kd_sporc` environment helper.
No packages are installed by the study or submission commands.

The GPU profile checks both eval and training-mode FP32 forward/feature-gradient/
shared-parameter parity, all four BF16 loss paths, nonzero auxiliary gradient
reachability, selected-state reload and HLT-only deployable parity. Its stress
fixture fills all 240 slots with repeated valid native particles at batch 256
so masked padding cannot conceal sparse-pair memory cost. These synthetic stress
inputs never replace scientific data. The production kernel then runs one real
500k training pass plus a real 200k validation pass with BOTH active.

Fit walltime is at least `ceil(1.75*(cache_seconds+100*pass_seconds)/60)` minutes,
minimum 60. Evaluation includes conservative 800k preparation/inference scaling,
factor two, minimum 30. Each 50-draw bootstrap task uses a measured full-SELECT
sort/metric draw extrapolated to 800k × twelve models × fifty draws, factor two.
CPU target requests use the measured TRAIN sample, conservatively extrapolated
serially. Nothing is clamped down to fit a queue: any estimate over 2,880 minutes
fails and requires an explicit resource revision. GPU peak must be ≤85%; host
peak and conservative cache/worker bounds leave at least 25% headroom.

### Commands after committing/pushing and creating a clean worktree

These are command templates, not permission to submit or an invented commit.
On SPORC, first activate the already prepared `atlas_kd_sporc` environment and
set `PROJECT_DIR` to the clean worktree, `COMMIT` to its pushed full hash,
`DATA_ROOT` to the verified uploaded `jetclass2` directory, and `AUX_ROOT` to a
fresh study directory outside raw data. `INVENTORY` and `PROFILE` point to the
uploaded frozen inventory and TRAIN_500K profile. Do not use old CMS manifests.

Alternatively, use `create --readiness-spec /absolute/path/readiness_spec.json`
instead of the inventory/profile/data arguments: this imports only the existing
Delphes readiness spec's embedded inventory, selected split and raw location.
It neither waits for its jobs nor reuses its matching or GPU acceptance. This
is useful when the raw upload and metadata are already prepared in the other
chat; no second dataset upload is required. `--data-root` can explicitly relocate
the same checksum-frozen data. Frozen manifests are data artifacts, not assumed
to be present in a clean Git checkout.

```bash
python -s "${PROJECT_DIR}/scripts/jetclass2_delphes_offline_aux.py" create \
  --root "${AUX_ROOT}" --data-root "${DATA_ROOT}" \
  --inventory "${INVENTORY}" --profile "${PROFILE}" \
  --project-dir "${PROJECT_DIR}" --source-commit "${COMMIT}"

python -s "${PROJECT_DIR}/scripts/jetclass2_delphes_offline_aux.py" stage \
  --study "${AUX_ROOT}/study_spec.json" --name GATE
python -s "${PROJECT_DIR}/scripts/jetclass2_delphes_offline_aux.py" dry-run \
  --stage "${AUX_ROOT}/stages/GATE/stage_spec.json"
python -s "${PROJECT_DIR}/scripts/jetclass2_delphes_offline_aux.py" submit \
  --stage "${AUX_ROOT}/stages/GATE/stage_spec.json" \
  --authorization-phrase "AUTHORIZE JC2 OFFLINE AUX GATE EXACT STAGE"
```

Inspect `monitor` or `results` with the same `--stage`. Once GATE succeeds,
create PREPARE and repeat dry-run/submit with PREPARE in the exact phrase.
Only after PREPARE succeeds can DISCOVERY be created. Only after its
configuration lock exists can CONFIRMATION be created. Authorization phrases
are respectively `AUTHORIZE JC2 OFFLINE AUX PREPARE EXACT STAGE`,
`AUTHORIZE JC2 OFFLINE AUX DISCOVERY EXACT STAGE`, and
`AUTHORIZE JC2 OFFLINE AUX CONFIRMATION EXACT STAGE`.

The tool verifies the previously measured locks; copying an old migration
runtime profile does not satisfy this gate. No other campaign needs to finish.

## Reporting and storage

Accuracy, macro one-vs-rest AUC, and ten signal-vs-QCD rejections at 50% signal
efficiency reuse the native Delphes metric convention. Macro R50 is geometric;
censored zero-FPR rejection is null, not infinity. No recovery percentage is
invented without an independently registered new-data offline oracle.

Report three within-replicate deltas, their mean/sample standard deviation and
number improved. Discovery fits are not pooled into confirmation. Six fixed
contrasts are implemented: each auxiliary arm minus CE, BOTH minus each single
group, and STRUCT minus COMP. One thousand paired source-file-cluster draws
reuse the same file multiplicities across models and seeds; sort/tie groups are
cached and integer-weight evaluation is checked against explicit row repetition.
Missing-class draws are not redrawn. Censored endpoints stay null. An interval
with fewer than 900 valid draws is suppressed with its valid count. The file
interval is conditional on the three trained seeds, not independent-seed proof.

Additional outputs: fixed TRAIN-derived HLT-pT bins; active-head scalar MAE/RMSE,
physical count MAE, distribution KL, TRAIN-mean and HLT-observable diagnostic
predictors. HLT pair diagnostics require joint HLT/offline validity. True zero
support in nontrained diagnostic predictors yields null KL with an infinite-row
count rather than introducing pseudocounts. Numeric overflow in the optional
inverse-count diagnostic is likewise reported, not used to suppress the fit.

Particle views, pair matrices, hidden states, optimizer state and best-in-progress
weights stay in RAM. Shared target payloads are 145 bytes/row; shards contain
at most 100k rows. Compact TRAIN pT and report-only HLT summaries support
diagnostics. Retained confirmation probabilities are 44+32 bytes/row/model.
One selected checkpoint includes auxiliary heads for diagnostics, not optimizer.

Each generated file is limited to 64 MiB and the whole study to 4 GiB, including
attempts and logs. Submission requires free space of twice projected artifacts
plus one GiB (conservative four-GiB projection before real measurement). The
profile records the actual selected payload size and projected total. Failed
attempts remain visible; there is no automatic deletion or spill-to-disk fallback.

## Recovery

All output directories are attempt-local and created fresh. Successful tasks
publish a checksum inventory/receipt last. Restart-zero recovery can reuse only
these validated receipts. It inspects exact IDs from all prior attempts of the
same stage; any running, pending, unknown or ambiguous attempt blocks recovery.
It never calls scancel, hold, release or priority-changing commands.

```bash
python -s "${PROJECT_DIR}/scripts/jetclass2_delphes_offline_aux.py" recover \
  --stage "${AUX_ROOT}/stages/PREPARE/stage_spec.json" --attempt retry1
python -s "${PROJECT_DIR}/scripts/jetclass2_delphes_offline_aux.py" dry-run \
  --stage "${AUX_ROOT}/stages/PREPARE/stage_spec.json" --attempt retry1
```

Live submission then uses that same attempt and stage phrase. The recovery
creates a new exact DAG for unfinished tasks and binds omitted completed
dependencies; it does not resume models or rewrite old partial files. Changed
source/science needs a new explicitly reviewed source/spec, not bypassing this
same-source recovery. A submitted intent without an acknowledged job ID blocks
blind retry and needs operator reconciliation; the code will not guess.

## Local versus real readiness

Local tests exercise native synthetic ROOT reads, worker-byte invariance,
mathematics, seed pairing, scalar buffers, toy-model training/restore, staged
locks, weighted bootstrap and exact submission/recovery failure paths. A bounded
read-only test also evaluated 1,024 actual TRAIN targets without report/test
access. Neither substitutes for genuine installed-Weaver/A100 execution.
The first queue-ready artifact is the GATE workflow. Scientific submission
remains deliberately blocked until the actual PREPARE acceptance report exists.
