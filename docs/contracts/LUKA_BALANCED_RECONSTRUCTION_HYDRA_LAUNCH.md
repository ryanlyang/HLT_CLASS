# Balanced reconstruction worker launch: v3

Authority: [active plan](../plans/LUKA_BALANCED_RECONSTRUCTION_HYDRA_PLAN.md).
Implementation: `luka_balanced_reconstruction_hydra/launch.py`.
Fresh SPEC binds canonical hashed WORKER_LAUNCH_POLICY/v3 with schema3, parents{}
and inherited development-validation/test-sealed flags. Old specs without this
policy fail closed under revised source; no accepted old execution is rewritten.

Operator-reported diagnostics establish saved pass and COMPLETED/0:0 for
Gloo/no-P2P job219052 on gpu1905,1m27. Original219050 and Gloo-only219051 failed
the known tensor SUM on one or more ranks; Gloo alone is insufficient. The
tested helper SHA is4a19da6c85172343141bbfbc43fa494bad59482439be6edee2f8435c56bf0466.
GPU_TRANSPORT_POLICY/v1 records this operator evidence, limited to the original
OFFLINE_s10 synthetic parity and known collectives; it is not all-arm resource
qualification or proof of a particular driver/hardware fault.

Pin NCCL_P2P_DISABLE=1 before the first Python import in the owned common shell.
Allocation/rank guards reject missing/changed values before CUDA initialization.
Early srun uses --export=ALL and both child executions receive the explicit
transport environment. No fallback is chosen during execution. NVIDIA documents
this as disabling the direct GPU peer transport:
[NCCL2.21.5](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2215/user-guide/docs/env.html#nccl-p2p-disable).
Transport may affect throughput and reduction rounding; do not promise bitwise
identity. Scientific objectives, FP32 precision, SyncBN and SUM semantics remain.

Job218567's earlier stacks establish rank0 blocked in native SyncBatchNorm
while ranks1--3 remained in the preceding Context.main NCCL object broadcast.
The ranks had reached different collectives. This motivates separating control
messages and testing that path before full decoding; hardware/P2P failure is
not established by the stacks alone. Preserve the failed a2b634c7 execution.

Use an owned Context subclass with a separate Gloo group for CPU object
broadcasts, gathers and barriers. A root callback broadcasts its error/result,
then all ranks cross a host barrier before proceeding. Native SyncBatchNorm,
FP32 tensor collectives and gradient SUM retain the default NCCL group and
their original mathematics. The frozen donor Context is not edited.

Before population replay or any particle decoding in each GPU allocation,
launch four actual workers in a separate synchronous srun step with the same
four GTX1080Ti/12CPUs/64000MiB, --export=ALL and --time=00:05:00. The step
runs torchrun with zero restarts and --startup-only. Slurm owns the hard
five-minute limit and descendant cleanup; a Python subprocess timeout alone
is not used as a substitute. Queue wait and Slurm KillWait are separate from
the nominal step runtime. See [srun](https://slurm.schedmd.com/srun.html).

The EARLY_GPU_POLICY/v2 nested in the launch policy fixes45s collective
timeouts, Gloo control/NCCL tensors,16 synthetic inputs, canonical untrained
starts and discarded updates. Check control broadcast/barrier, known tensor
SUM/broadcast, installed-Weaver parity and the actual independent full-graph
versus production native/auxiliary update. Qualification covers all9 arms
at full weight and both sides of each delayed reconstruction schedule;
science checks its current arm. No real particle or split access is needed.
The production decoder uses its unchanged full registered output support;
synthetic target sets contain1--4 particles and are not a full-data resource
qualification. The passing no-P2P transport is pinned for this campaign; all
other campaigns and the original diagnostic helper are unchanged.

EARLY_GPU_ARM/v1 and completion-last EARLY_GPU/v1 bind spec, policy, task,
allocation ID, per-rank parity and payload checksums. Persist under
early_gpu/JOB, with separate rank stack/error files. A nonzero step exit,
missing completion or invalid proof stops before population/input work and
cleans only owned scratch. Scientific execution still requires the complete
real-input qualification and its resource/miniature checks. No new batch job
or science DAG node is introduced. The standalone passing profile establishes
the transport candidate; all-arm early GPU and full real-input qualification
remain necessary. Missing or mismatched transport proof is not reusable.

The rank command is Python -> torch.distributed.run -> torchrun flags -> `--`
-> absolute training script -> `rank --spec ... --task ... --node ...`.
The separator precedes the training script, so argparse cannot interpret the
worker's `--node` as a prefix of torchrun's `--node-rank`/`--node_rank`.

Exercise the installed PyTorch get_args_parser on the actual generated command
before registration's data authentication/population construction, and again
inside every GPU allocation before population replay or particle construction.
Require exact training script/argument transport, one node, four processes,
standalone rendezvous and zero restarts. These are CPU-only parse checks, not
new preflight jobs or substitutes for actual GPU acceptance.

Qualification gets720 minutes (12h) instead of360. Job217775 spent5h49 overall
and finished all views before failing at launch; U100 input construction alone
took13547.9s. This evidence leaves insufficient room under6h for all GPU checks.
Twelve hours adds headroom; it is not a measured runtime guarantee. The launch
policy binds this qualification limit. Scheduler requests and allocation guards
use the same constant. Scientific fits retain2880 minutes/48h; four GTX1080Ti,
12 CPUs and64000MiB remain. Existing per-arm memory and48h projections still
gate scientific release. No metric-quality threshold is introduced.

Scientific recipe, seeds, subset/roles, sampler, losses, epochs, input execution
policy, native/EMA readouts, output allowance and HOME recheck policy are unchanged.
No dependency upgrade, persistent particle cache, new target or test capability.
Previous scratch cleanup already retired job217775's temporary arrays; the fixed
fresh execution must reconstruct them. Preserve old source/spec/logs/receipts.

Local regression reproduces the exact original ambiguity with the installed
torch2.5.1 parser and checks the corrected command. A real four-CPU-worker
elastic smoke verifies rank/world and exact `--node` transport, including spaces.
The Windows-only smoke harness selects the real non-libuv TCPStore constructor
because this local wheel lacks libuv; production Linux invocation is unmodified.
This is not four-GPU or Pascal acceptance. Genuine Hydra qualification remains
required before scientific release.
