# U100-w5 plus subclass Hydra contract v1

Scientific authority: [active plan](../plans/LUKA_SUBCLASS_HYDRA_PLAN.md).
All artifacts carry `LUKA_SUBCLASS_HYDRA_<KIND>/v1`, schema_version1, content
hash, explicit parents and the existing exploratory-development disclosure
flags. Old FullStat and teacher-sweep artifacts retain their own contracts.

## Registration and source

SPEC binds the exact clean pushed additive commit descended from
5571240c8c42fce1103344dff94c88bf8edf13c5, target/training recipes, teacher-sweep
spec file reference, T00 checkpoint/bank/payload/report/receipt references,
matched C0 proof, original U100 node and target hash, prepared population,
environment, paths and resource envelope. Original T00 authentication runs in
its pinned original project; no S00 student is required. Counts are exactly
299234 TRAIN and 74895 development-validation. All ten fits keep the same
initialization and sampler seeds. No new control or teacher is registered.

## Target bank

SUBCLASS_BANK is TRAIN-only and binds SPEC, TARGET_RECIPE, OFFLINE bank hash,
T00 selected-checkpoint SHA256 and prepared population. Atomic publication
places `manifest.json`, `targets.npz`, `prototypes.npz` in `setup/subclasses`.
Readers require checksums, exact inventories, finite arrays, role, population
and ordered unique identity/label joins. Presence alone is insufficient.

Targets: identities uint8[N,32], labels int64[N], probabilities float32[N,10].
Rows sum to one within2e-6; signal rows have exact zero columns5:10. Prototypes
are float64[11,10,138], padded with zero unused signal centers. Margin means
and scales are float64[11,10]; embedding means/scales float64[128]; class
temperatures float64[11]. Scale floor0.001; temperatures in[1e-8,1e8].
The original pure FullStat clustering kernel is reused, with the new small
population explicitly bound by TARGET_RECIPE. Its legacy coefficient metadata
does not set this campaign's training weights.

One immutable bank serves all ten fits. No validation or final-test row fits
prototypes, moments or temperatures. Finite degenerate clusters/occupancy and
unattainable softness goals are reported, never accuracy gates.

## Training and inference

Each loss is CE +5*original standardized U100 SmoothL1 +lambda*r*conditional
KL(q||p). lambda is one of .03,.1,.3,.5,1,2,5,10,40,100; r is the one-based
update divided by three epochs, capped at one. KL averages over jets without
dividing by subclass count; only the known TRAIN class's group is normalized.
The true label never enters the forward pass. Targets are detached.

The native graph and original U100 predictor initialization are preserved.
An isolated-seed Linear128x60 head branches off the pooled native embedding.
One ordinary backward updates all trainable parameters. Sampled gradient
probes observe CE/U100/subclass contributions and do not project, normalize,
clip or change gradients. Auxiliary metrics never select a checkpoint.

EPOCH and TRAINING_REPORT bind SPEC, RECIPE, U100 and subclass-bank hashes,
full schedule and native validation. All30 epochs are required for scientific
completion; technical miniatures have two epochs and explicit technical=true.
Selected native `mod.*` state must strictly reload and reproduce probabilities.
Neither auxiliary head is included in deployment. FIT/TASK receipts bind every
output by checksum and TASK binds the exact accepted Slurm job and environment.

## Execution

An exact dry review covers GPU setup, CPU afterok release, ten independent GPU
fits and CPU afterany report. One authorization covers that bounded13-job DAG.
No sibling fit dependencies, automatic retries or quality gates. Setup must
pass native factory parity, original U100 model/AdamW zero-subclass parity,
historical native replay, all actual-target miniatures and warmed measurements.
Technical weights are discarded; scientific workers start fresh after their
own disposable startup check. Preserve training memory peaks before deployment
benchmarks reset GPU counters.

Release requires exact COMPLETED/0:0 setup accounting and authenticated
acceptance. Measured commands may reduce walltime only. Existing exclusive
submission claims and immutable exact-ID journals reject partial or ambiguous
retries. Complete resubmission is idempotent. Poor finite recovery, KL,
agreement or teacher quality must never cancel or skip another registered row.

Results/progress are read-only and compute native recovery relative to the
historical M0HLT/OFFLINE anchors, with C0 and SPORC U100 distinguished.
Original50k overlaps validation; neither is an independent test. Final test
remains sealed. Remote readiness is established by the actual A6000 setup,
not by local synthetic CUDA evidence.
