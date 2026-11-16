# EMA control campaign contracts v1

Authority: [nine-fit plan](../plans/LUKA_EMA_CONTROL_HYDRA_PLAN.md).
Artifacts use `LUKA_EMA_CONTROL_<KIND>/v1`, content and parent hashes and the
inherited exploratory-validation flags. No previous artifact changes identity.

`RECIPE` freezes CE/U/R24, seeds 7/8/9, seed-major nine-fit order, single A6000,
physical 128, 30 epochs, FP32/TF32-off, original optimizer/schedule and R24
hard matching. CE receives no privileged target arrays and has no predictor
optimizer state. U adds weight 5 embedding regression; R24 additionally uses
delay 3 / ramp 3 / max 0.3 complete OFFLINE reconstruction. No quality gate.

`SPEC` authenticates clean pushed implementation, the 3ccc35ae stability parent,
58d0c0ed scientific worker, environment, TRAIN/validation populations, frozen
U100 bank, complete target inventory and parent TRAIN BatchNorm membership.
Parent readers disable future-write allowance checks. New output roots cannot
overlap any parent/source/data/environment path. `STORAGE_INVENTORY` measures
serialized state sizes and includes all nine setup/startup miniatures, four
readouts, final training states, canonical starts, predictions and overhead.
Registration requires the estimated output allowance plus one GiB reserve.

`INITIAL_STATES` authenticates three untrained seeded full states and per-tensor
hashes. Every fit loads its seed's exact native/predictor bytes; R24 also loads
the decoder. `INITIALIZATION` binds the start hashes, full first-epoch TRAIN
permutation and dropout seed. All miniatures and scientific fits must match the
canonical state, and available same-seed pairs must match each other. Setup
trained weights are never reused for science. Frozen initial states are not
pretraining or a selected checkpoint.

`BN_MEMBERSHIP` reproduces the parent's 8192 TRAIN-only identities/order,
batch 128 and disabled dropout. EMA05/EMA10 average only native floating
parameters after position 3; their half-lives use examples processed, not
updates. Both EMA copies and RAW_BN recalibrate only their own BatchNorm buffers.
Readouts restore CPU/CUDA RNG and never change live training buffers.

`QUALIFICATION` and 15 `PARITY_STEP` records compare production updates against
an independent full-graph CE/U/reconstruction calculation at positions3,3.01,
4.5,8.99,9 with dropout disabled only for parity. Required tolerances: scalar
loss 1e-3; gradient, parameter and AdamW moment relative L2 5e-4; named buffer
absolute 1e-4, evaluation probability 1e-5 and exact integer counters. Complete start states remain transient
in RAM; hashes and comparisons are retained. Historical FP32 native replay
must agree within 5e-5 AUC on authenticated validation identities.

`RESOURCES`/`ACCEPTANCE` require nine two-epoch production miniatures with four
native export/reload checks; each objective's timed warm phases and full HLT,
largest OFFLINE, partial 11 and tail batches; full validation and BN readouts;
actual GPU/host memory and storage. Single-A6000 GPU memory fraction 0.88 and
host 0.85 of 32000 MiB are ceilings. Fit requests use 2x timings plus 30 min overhead,
at most 48h. No scientific AUC value admits or rejects a fit.

`EPOCH`/`TRAINING_REPORT`/`DEPLOYMENT`/`FIT` bind per-epoch metrics and separately
selected RAW/RAW_BN/EMA05/EMA10 native-only checkpoints and aligned predictions.
**EMA05 is primary.** The same readout is used on both sides of every paired
delta. No ensemble, auxiliary head, OFFLINE input or inference parameter growth.

`PLAN`/`SUBMIT_CLAIM`/`RUN_CLAIM`/`TASK` and exact DAG journals bind commands
and accepted IDs. Bootstrap is setup_a6000 then release_a6000 afterok. The
CPU release verifies successful setup accounting and technical acceptance,
then submits nine independent fits and report_a6000 afterany. Twelve jobs
total. Review covers the entire 48h science envelope before authorization;
only measured walltime can change. Completed submissions are idempotent;
ambiguous partial submissions stop and preserve evidence. No blind retries.

`RESULTS`/`PROGRESS` read saved evidence without inference or future-write
quota checks. They preserve every readout, three-seed denominators, mean/min/SD,
same-seed U-CE and R24-U, and late 21-30 metrics separate from selected results.
Missing controls produce missing deltas. Seeds were previously used; validation
is exploratory and original 50k overlaps it. Final-test access remains sealed.
