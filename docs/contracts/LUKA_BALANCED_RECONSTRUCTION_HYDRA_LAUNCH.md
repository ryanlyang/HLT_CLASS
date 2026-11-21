# Balanced reconstruction worker launch: v1

Authority: [active plan](../plans/LUKA_BALANCED_RECONSTRUCTION_HYDRA_PLAN.md).
Implementation: `luka_balanced_reconstruction_hydra/launch.py`.
Fresh SPEC binds canonical hashed WORKER_LAUNCH_POLICY/v1 with schema1, parents{}
and inherited development-validation/test-sealed flags. Old specs without this
policy fail closed under revised source; no accepted old execution is rewritten.

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
