# Hydra reconstruction distributed diagnostic: v1

Authority: the operator requests diagnosing repeated six-hour failures before
another long attempt. Implementation is standalone
`operations/hydra_reconstruction_gpu_diagnostic.py`; source parent is immutable
`a2b634c7a4406f24f6a11422f24fdb76b5a1c7ca`, root
`runs/balanced_reco_600k_a2b634c7_r1`, failed job218567.
The scientific source/spec, datasets, normalizers and results remain unchanged.

The supplied log shows all four rank mmap validations completed, rank0 completed
installed-Weaver parity, and rank0 began OFFLINE full-weight parity. Ranks1--3
then timed out after1200s in a two-element NCCL object-size broadcast inside
Context.main. The later supplied earlier stack shows rank0 waiting in native
SyncBatchNorm while ranks1--3 remain in the preceding object broadcast.
The ranks reached different collectives; a hardware/P2P fault is not proven.
No arm proof completed.
This is not a quality metric, disk quota or Slurm12h walltime failure.

Run exactly three sequential afterany diagnostic allocations on gpu1905, each
4 GTX1080Ti /12 CPUs/64000MiB /5 minutes, account default/qos normal/partition
gpus. Profiles: original donor Context with default NCCL; Gloo CPU control
object broadcasts/gathers and a host barrier after root callbacks, with
unchanged NCCL tensor reductions; same Gloo
control with NCCL_P2P_DISABLE=1. This last switch disables direct GPU peer
transport as documented by [NVIDIA NCCL2.21.5](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2215/user-guide/docs/env.html#nccl-p2p-disable).
No production fallback or transport choice is authorized by a profile name alone.

Operator result2026-11-21: helper4a19da6c85172343141bbfbc43fa494bad59482439be6edee2f8435c56bf0466
recorded original219050/Gloo219051 failures at the known tensor SUM, and
Gloo/no-P2P219052 saved pass plus COMPLETED/0:0,1m27 on gpu1905. The explicit
campaign choice is now recorded in GPU_TRANSPORT_POLICY/v1 and the v3 launch
contract. This is evidence for the no-P2P configuration, not identification of
a particular hardware/driver defect. Original diagnostic metadata and helper
remain unchanged; all-arm early acceptance and real-input qualification follow.

Each probe authenticates the old spec/source/runtime and accepted job/comment,
then tests the original first Context.main operation, a known NCCL tensor SUM
and broadcast, a CPU-object gather, and original production OFFLINE full-weight
parity with sixteen synthetic jets. Only authenticated canonical untrained
starts are loaded; all temporary updates are discarded. No real jet particle
reads, inference on real jets, final-test access, scientific checkpoint export,
input preparation, scientific fit or full-campaign submission/release.

Collective timeout45s; per-rank nonfatal stack samples40s; Slurm5m is the outer
bound even if imports, I/O or native CUDA calls hang. Zero torchrun restarts.
Original argument separation is preserved. Logs/stacks/errors are separate by
rank, and core dumps are disabled. Worker CWD is its diagnostic output folder.
The five-minute bound is allocation runtime, not a queue-start guarantee.

Artifacts use LUKA_HYDRA_GPU_DIAGNOSTIC_<KIND>/v1, schema1, canonical hashes,
explicit parents, helper SHA256, final_test_accessed=false,
raw_particle_reads=false and scientific_training=false. PLAN -> INTENT before
each sbatch -> SUBMITTED ID, then per-rank RANK/ERROR and parent COMPLETE/FAILED.
Parent completion requires all four correct rank/job/source/helper receipts and
successful torchrun exit. Read Slurm terminal state separately before accepting
a profile as passed. Missing receipts are not success. Uncertain submission
preserves intent and refuses blind duplicate launch. Failed profiles do not
suppress the other independent profiles. No cancellation outside these jobs.

Owned output is `reports/reconstruction_gpu_diagnostic_<helper_sha12>`; old
scientific artifacts are neither overwritten nor relabeled. The PowerShell
launcher uploads and SHA-checks one helper with the existing Brown identity,
then detaches submission. No new scientific commit or dataset upload is needed.
Helper SHA and immutable original source supply the diagnostic identities.

Passing these probes establishes only their declared communication/native
checks. It does not certify all arms, full support, resources or scientific
results. The locally implemented WORKER_LAUNCH_POLICY/v2 repair runs actual
GPU communication/native/auxiliary checks before full input preparation on
every allocated GPU node, using the same Gloo control wrapper. Do not queue
another long attempt until the applicable diagnostic passes. Full real-input
qualification and resource checks remain mandatory afterward. The subsequent
v3 repair pins the passing no-P2P configuration; it supersedes v2 control-only
execution for this campaign.
