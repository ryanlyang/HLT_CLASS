# Balanced reconstruction on Hydra: v1 contract

Authority: [active scientific plan](../plans/LUKA_BALANCED_RECONSTRUCTION_HYDRA_PLAN.md).
Implementation namespace: `luka_balanced_reconstruction_hydra`. Artifacts use
`LUKA_BALANCED_RECONSTRUCTION_HYDRA_<KIND>/v1`, schema1, canonical content hashes,
explicit parent hashes and inherited development-validation/test-sealed flags.
They are not interchangeable with the old reconstruction or full ladder artifacts.

## Frozen population and capabilities

New verified Run-2 QCD/Higgs samples, parent six-class population
`5092ce94a5e32fac10bdbd5b91b1becddc3fe9123affc41a93a0b312db2ea616`.
Proportional cell quotas and fixed identity ranking define subset
`21175a03adc3d847e62d84a447b04525fc813207685a37d0b0dcb630acafd1f1`.
Its exact counts are 600000 TRAIN/150000 validation. The original frozen roles
are replayed before subset selection; reduced-out rows do not change roles.
Raw label206 stays quarantined. Six classes, 600 <= pT <2600 GeV, 20 bins,
120 supported cells. Draw uniform cell then uniform row with replacement.
Validation is natural, unweighted and identical across fits. Test remains sealed.

Only selected TRAIN contributes normalization, teacher banks or reconstruction
targets. No validation reconstruction targets. HLT-only input construction does
not request OFFLINE branches unless the arm explicitly needs TRAIN supervision.
Bounded contiguous ROOT reads may include reduced-out rows within the original
authorized TRAIN/validation role, discarded before learning/statistics. They
must never cross a test/unused role. Particle counts are complete, never clipped.

## Models, objectives and clocks

Two seed indices10/11; nine arm types and exact coefficients/schedules are
recorded by `contracts.recipe()` and the active plan. Fresh OFFLINE/M0HLT,
single predecessor-KD U100 teacher, U100-w5, R24/D20/R20/R22/R28. A same-seed
selected EMA05 OFFLINE probability bank supervises U100's native main head.
Its selected EMA05 standardized128-dimensional embedding bank supervises all
HLT students. Scientific banks must have exact TRAIN IDs/labels, expected view,
selected native checkpoint/report hashes and all payload checksums.

Four Pascal GPUs, global128, synchronized BN, FP32/TF32-off, explicit
sample-weighted gradient SUM. Final global64 batch is not dropped. 600000 draws
and4688 updates per epoch,20 epochs. AdamW/warmup2/cosine20 and EMA clocks use
actual draws. No accumulation, optimizer interventions, LR scaling or clipping.
The original full-support measured14/mixed22 set loss uses chunk8, exact
detached assignment and the declared decoder/encoder schedules. At zero weight,
explicit decoder zero gradients preserve original AdamW bookkeeping.

Native-only RAW/RAW_BN/EMA05/EMA10 selections are independent. EMA05 is primary.
BN recalibrates copies on fixed8192 balanced TRAIN draws without changing live
buffers or RNG. AUC/CE/logR50/earliest-update ordering selects checkpoints.
R50/R30 and per-pT metrics use those same checkpoints. Recovery anchors must
match this subset, seed and readout. Missing/nonpositive gaps or censored tails
stay unresolved. Primary results include individual seeds and paired statistics;
fixed late16..20 means are separate from selected results.

## Publication, execution and continuation

`SPEC` binds exact pushed clean source, transitive donor pins, installed runtime,
population, backend, resources, protected ladder reservation and output inventory.
Canonical untrained starts bind native/predictor/decoder hashes. Temporary probes
are discarded; scientific starts reload the canonical state. No quality threshold
may suppress a scientific row or fail training.

Bootstrap queues CPU prepare, four-GPU qualification and CPU release. Science
release requires installed-Weaver parity, full-graph versus four-rank parity,
both-seed two-epoch miniatures/readout reloads, full-support stress, measured
complete validation and resource projections within the48h/64GB/11GB envelopes.
Every scientific allocation repeats an integrated miniature using real teacher
banks. A failed qualification releases no scientific fits. Real four-GPU evidence
is a prerequisite for release, not a claim made by local queue-readiness tests.

Slurm allocations must match accepted task/job IDs, spec comments, absolute source,
site, CPU/RAM/time and four physical GTX1080Ti GPUs. Science fits are serialized
with afterany ordering and afterok teacher dependencies; report is afterany.
No job cancellation or mutation outside this campaign is implemented.

Every submission writes intent before `sbatch`, then binds its accepted ID.
Uncertain submissions fail closed; never erase the journal or blindly resubmit.
Explicit continuation requires all prior accepted IDs terminal. It preserves
completed outputs and resumes incomplete fits at a committed epoch boundary.
Full restarts include model, AdamW, native EMA, selected checkpoints and each
rank's CPU/CUDA RNG. A failed final export can finish from epoch20 without updates.

`EPOCH` commits are completion-last and bind task/job, parents, metrics and full
restart state. A new committed epoch authenticates and journals retirement of
the obsolete full state inside that exact fit. `RESULT` binds all four native
checkpoints/predictions/metrics, initialization, report and outgoing teacher bank.
Only successful authentication of published outputs permits retiring the final
restart payload. Scientific native outputs remain. No timestamp is an identity.

Allocation-local `/dev/shm` inputs/targets are mmap-shared, authenticated and
removed only under the exact owned job path. HOME budget8GiB+2GiB reserve is
checked together with the unspent authenticated14GiB existing ladder reservation
only at registration or explicit manual resume. Operator-authorized
`LUKA_BALANCED_RECONSTRUCTION_HYDRA_STORAGE_POLICY/v1` binds this execution rule:
no automatic HOME checks at submission/release, job startup or epoch boundaries;
no runtime output cap. Genuine filesystem write errors propagate. Specs without
this policy fail closed under the revised source; use a fresh study root.
Dataset, other campaigns, existing models and jobs are never cleanup targets.

The revised initial8GiB budget is bound numerically in the STORAGE_POLICY and
SPEC. The full initialized inventory is4,828,964,049 bytes; the protected14GiB
ladder allowance and separate2GiB reserve remain, so the initial ceiling is24GiB.
This estimate does not reserve physical filesystem space or promise write success.

The separate operator-authorized one-file retirement described in the runbook
may remove the unused old `contrast_delta_54473f10/keys/keys.npz` derived TRAIN
bank after manifest/checksum and transitive live-job authentication. It does
not change this experiment's dataset, teacher targets, model or prediction files.
