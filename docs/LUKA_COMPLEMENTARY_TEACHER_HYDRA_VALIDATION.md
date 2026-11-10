# Complementary teacher Hydra validation

Local validation completed, 2026-11-09. Ready to queue the real Hydra setup.
This task performed no commit, push, remote submission or scientific recovery
measurement. Real qualification remains mandatory before scientific release.

Environment: Python3.10.19/Torch2.5.1 in
`C:/Users/22rya/miniconda3/envs/tagging-hlt/python.exe`, Weaver0.5.3 source fixture
`tmp/hydra-weaver-fixture-20261102`, local NVIDIA RTX2000 Ada Laptop GPU.
The fixture is a local test aid, never a Brown environment override.

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONPATH='tmp/hydra-weaver-fixture-20261102;src'
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
& C:/Users/22rya/miniconda3/envs/tagging-hlt/python.exe -B -s -m pytest -q -p no:cacheprovider tests/test_luka_complementary_teacher_hydra.py
```

Final focused suite: **50 passed in99.81s**, no skips. Donor baseline:
**138 passed in136.29s** across teacher-sweep and selective-teacher suites with
the local Weaver fixture. These are188 distinct passing tests. Initial attempts
without the source path or Weaver fixture failed on imports; setting the
documented environment fixed those invocation errors. One new failure-injection
test initially supplied the wrong review hash and correctly hit the authorization
guard; correcting the test to use its own reviewed plan exercised the intended
ambiguous-claim rejection. No runtime guard was loosened.

## Evidence covered

- Fixed three final teachers, three OOF helpers and four students; each student
  has total embedding weight5 and no U100 target. Identical native initialization,
  separately seeded predictors, strict native export/reload and inference without
  any target access.
- CPU and CUDA finite second derivatives through both native input streams,
  nonzero gradients to the learned pair embedding, frozen reference parameters,
  unchanged BN statistics and restored mode. Per-jet microbatch/full-batch
  gradient equivalence, particle permutation and padding invariance, and finite
  standardized-coordinate response diagnostics.
- Full production eight-particle/two-class-block CUDA updates at batch128 and
  partial11 for T0, TE, TF, S0 and SEF, followed by native inference parity.
  Synthetic sequences are short; these checks do not estimate Brown memory.
- Deterministic class-stratified identity folds; train/held exclusion, no
  validation argument allowed during fold fitting, fixed-last selection,
  complete OOF assembly and bounded positive class-normalized difficulty.
  Corrupted memberships and substituted training-ID lineage are rejected.
- The complete GPU setup function across all ten profiles: replay, technical
  fold miniatures, full/partial/longest updates, two-epoch fits, bank exports,
  timings and recomputed acceptance. Data and model depth are reduced; Slurm,
  A6000 identity and Linux memory are simulated. Injected setup failure preserves
  diagnostics and publishes no acceptance.
- All ten scientific worker paths run all30 epochs on synthetic data with the
  actual production fit/export/readback functions. The test exercises three
  OOF predictions, actual-parent TE/TF/student startups, all independent target
  combinations, saved difficulty hashes, completed and per-epoch reports and
  afterany report publication. This uses reduced-depth native Weaver and the
  local CUDA device; it is not a remote science run.
- Exact bootstrap and automatic-release simulation, registered dependencies,
  report afterany, completed-submit idempotence, wrong-job/failed-setup and
  ambiguous-claim rejection. Resource/memory/timing/profile failures remain
  technical gates. A deliberately poor finite teacher finishes all30 epochs.
- Original-worktree authentication, source guard accepting an additive sibling
  but rejecting donor edits, registration round-trip, parent substitution,
  TRAIN-only target firewalls and byte-for-byte read-only reporting checks.
- New Python compilation, CLI registry, queue/worker and runbook Bash syntax,
  documentation links and scoped whitespace/EOF checks.

The internal donor is `54ff594df4fb7dddc56c0e99ffb13b9448a6a6c9`; no existing
runtime file was changed. The separate summary-token commit appeared during
this task; its additive runtime remains untouched and is accepted by the new
source guard. See the [donor map](LEGACY_SOURCE_MAP.md).

## Remaining execution boundary

Real Brown setup must still authenticate installed Weaver and original data,
replay the native checkpoint, perform the actual miniatures and measure every
profile within the fixed A6000 envelope before automatic science release. Local
synthetic tests are not remote acceptance or evidence of an HLT gain. The
[active plan](plans/LUKA_COMPLEMENTARY_TEACHER_HYDRA_PLAN.md) explicitly selects
Hydra qualification in place of generic Tigris defaults.
Use the [runbook](LUKA_COMPLEMENTARY_TEACHER_HYDRA_RUNBOOK.md) for the scoped
commit/push and Brown registration/queue steps. It reuses the existing
teacher-sweep native OFFLINE transfer; no new SPORC export is needed.
