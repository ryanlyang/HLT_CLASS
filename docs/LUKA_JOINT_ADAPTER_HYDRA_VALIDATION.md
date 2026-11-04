# Joint-adapter Hydra validation and provenance

The implementation is additive and locally queue-ready. Real Brown L40
acceptance runs in shared setup before scientific fits are submitted; it has
not been executed by this agent. This is not a claim of scientific improvement.

## Local evidence

- Before changes: frozen-adapter focused baseline **44 passed in45.24s**.
- Joint implementation intermediate suite **42 passed in51.84s**.
- Joint plus frozen-adapter, Hydra site/input and portability regression:
  **158 passed in116.94s**. This includes actual installed-Weaver CPU/CUDA-BF16
  synthetic forward/backward, selected-state publication/reload, all14 setup
  miniatures, full30 teacher/student workers and authenticated result assembly.
- Final focused follow-up after schema, registration and detached-shell checks:
  **51 passed in54.04s**. An initial MSYS-only directory-probing failure in the
  shell test harness was repaired using direct creation of existing-parent
  fixture leaves; the production Linux launcher was not weakened.
- Python AST and CLI help, both Bash workers/launcher and every runbook Bash
  block passed syntax validation. Existing donor runtime is unchanged.

Focused test coverage includes zero-adapter/native identity, from-scratch equal
native initialization, trainable teacher backbone and buffers, frozen selected
teacher targets, valid-pair masking/centering, pair permutation equivariance,
multichunk checkpoint gradients, edge contraction equivalence, correction weights
and ramp, intermediate versus final gradient paths, inference stripping, privileged
input rejection, TRAIN-only statistics, lineage/hash validation, real exact-DAG
journals with mocked Slurm, failed-setup handling, interrupted-watcher recovery,
idempotent complete submission and refusal to retry partial claims.

The final-token branch test isolates correction loss and demonstrates gradients
at block7 and block0; the intermediate auxiliary branch has no block7 gradient.
Full production architecture identity/tap tests also run, beyond the smaller
embedding/pair layers used to keep the synthetic end-to-end suite fast. CPU/CUDA
synthetic rows are not a full-size Brown dataset measurement. The Bash handoff
smoke stubs `setsid` on Windows; it does not prove Linux SSH-disconnect survival.

## Source and contracts

New package `src/hlt_classification/luka_joint_adapter_hydra/`, CLI/launcher,
Slurm worker, focused test file, plan/contract/runbook and this record. README,
HANDOFF and LEGACY_SOURCE_MAP receive additive entries. No old campaign runtime,
source worktree, data, saved result, scheduler job or user hold was changed.

The in-repository donor is frozen-adapter commit
`4b9630fa00f137c4e48b749eb86714719b9d26cf`, unchanged at baseline
`ceb2146149d2b1508a36b2cf91b274e0b58a26e6`. Copied/adapted modules and unchanged
imports are enumerated in [LEGACY_SOURCE_MAP](LEGACY_SOURCE_MAP.md). There is no
Fresh_check runtime dependency. New semantics use
`LUKA_JOINT_ADAPTER_HYDRA_*/v1`; old artifacts retain their original contracts.

## Remaining execution

Commit/push the scoped changes, create the clean Brown worktree, and execute the
[queue runbook](LUKA_JOINT_ADAPTER_HYDRA_RUNBOOK.md). The real shared setup must
pass environment/data authentication, original numerical qualification and
measured production miniatures before releasing the14 fits. Full-size runtime,
real Hydra submission and new scientific outcomes remain unmeasured. No separate
preflight, input transfer or reference retraining is required by this campaign.
