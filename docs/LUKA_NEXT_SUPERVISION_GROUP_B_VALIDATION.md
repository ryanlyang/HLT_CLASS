# Group B validation and execution boundary

Internal donor baseline: `342e5d3affba78f9d6add4cbb95d6264fc4b516b`.
Runtime extends `src/hlt_classification/luka_next_supervision_hydra` with
`structured_contracts`, `questions`, `vocabulary`, `structured_model`,
`structured_targets` and `group_b`. There is one common trainer/worker/
submission/reporting system. Group A's scientific recipe and old pinned
worktrees/artifacts remain unchanged. See the
[contract](contracts/LUKA_NEXT_SUPERVISION_GROUP_B.md) and
[runbook](LUKA_NEXT_SUPERVISION_GROUP_B_RUNBOOK.md).

## Local checks

Local synthetic environment: Python3.10.19, Torch2.5.1, installed Weaver-core0.5.3
fixture at `tmp/hydra-weaver-fixture-20261102`, NVIDIA RTX2000 Ada Generation
Laptop GPU. This is not Hydra or real-data throughput qualification.

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONPATH='tmp/hydra-weaver-fixture-20261102;src'
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
$env:OMP_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:LOKY_MAX_CPU_COUNT='1'
& 'C:/Users/22rya/miniconda3/envs/tagging-hlt/python.exe' -B -m pytest -q -p no:cacheprovider tests/test_luka_next_supervision_group_b.py tests/test_luka_next_supervision_hydra.py
```

Pre-edit Group A baseline:66 passed in52.63s. Latest combined A/B run:
**114 passed in95.85s**. The subsequently added B-only read-only results check
passed separately: **1 passed,48 deselected in7.03s**. Together these cover all
115 current focused tests. The B token is bound through the reviewed plan/claim
and cannot authorize an A stage. Native A recipe equality was also checked
against the exact342e5d3a implementation, preserving hash
`d39c6ee09d803622385834756bb57c131ed8a8142659b1d99aec248ec6b6bddc`.
Final static checks passed:23 Python AST/import checks, all30 required files,
741 local documentation links, balanced fences,5 Bash scripts/runbook blocks,
CLI help/create-b/33-row implemented registry and scoped git diff --check.
The full40 release remains unavailable while Group C is unimplemented.

Coverage includes:

- Exactly9 B configurations in filtered40-row order, proper source/weight/
  seed/marginal/count-only controls, no C0/C1 retraining and no full40 release.
- Hand-calculated fraction/count/joint/pair answers, charge relation, distinct
  real particles, wrapping, particle permutation, valid significance and
  shared HLT-axis geometry. Source/target roles and ordered joins fail closed.
- Semantic encoding independent of catalog indices, answer/class/source-
  independent question streams, epoch/replicate/chunk replay, and parallel
  transport equality. Full-TRAIN support retains a single rare positive;
  constant/short catalogs are disclosed without padding or quality failures.
- TRAIN-only dictionary fitting, no class-label use, equal block energy,
  deterministic nearest-center ties/chunks, frozen centers, hand-calculated
  count/pT frequencies and finite normalization/zero-denominator failures.
- Binary/fraction family reductions and frequency sum-code/mean-head
  reductions; detached targets and no dependence of predictions on answers.
- Every B head preserves native/w5 initial state and RNG; meaningful auxiliary
  gradients reach the native encoder. All9 real fit code paths export/reload
  native-only checkpoints on synthetic data. Full30 Q/V fits continue even
  with constant/zero-applicable finite targets.
- Actual streaming target preparation, sklearn fit/assignment and compact-bank
  publication/readback, codebook/source identity and corrupted-byte rejection.
- Warmed six-path production-kernel resource acceptance, full/partial timings,
  native plus largest-source/valid-track target tails, frozen preparation-parent
  authentication and target-parent tampering.
- Additive B registration/parent immutability, exact worker publication,
  scope-specific authorization, no science admission before exact successful
  setup, real submission journals, idempotence and no fit dependencies.
- B-only saved results are read-only, preserve failed/pending rows and never
  substitute seed0 for a missing seed1 reference. Constant-answer redundancy
  diagnostics handle NumPy2.x inverse shapes without emitting nonfinite values.
- CUDA full128-row/partial2-row production-width Q-full, Q-marginal,
  V-both and V-count paths, including synchronized phase timing.

Windows tests emulate Linux process transport; actual fork/PSS and genuine
full-TRAIN runtime/memory remain Hydra setup checks. No synthetic accuracy is
used as a quality admission threshold.

## Local cost observations

A serial synthetic benchmark used256 jets with54 particles each and all8192
question proposals. Geometry/proposal construction took0.187s; the full support
scan took8.601s. The frozen catalogs retained1024 composition,826 count,1024
joint and812 correlation entries. These counts show explicit shortfall
reporting, not real-data support. A128-anchor Q1 oracle batch took0.072s.

On a separate1024-jet/55296-particle synthetic collection, a K128 dictionary
fit on8192 reservoir particles took0.938s; full assignment took0.753s. These
are laptop serial observations, not a Brown walltime request or an admission
certificate. Real source diversity, tails, target I/O and CPU contention can
change costs. The production setup measures them and preserves diagnostics.

## Required remote acceptance

The user reports that Group A is queued/running. No new Group B source was
pushed, job submitted, real-data target prepared or Hydra miniature executed
by this agent. Group B is locally queueable through its explicit setup stage;
science release remains gated on real production acceptance.

After clean commit/push, register a separate B spec using the existing Group A
input spec. Existing raw TRAIN data, original U100 targets and reference proofs
are reused; no SPORC export/transfer or new teacher training is needed. Setup
builds frozen Q geometries/catalog and independent OFFLINE/HLT V dictionaries/
banks, verifies installed-Weaver baseline parity, then measures all six
execution supports with complete candidate universes and actual target paths.

One A6000,6CPUs,32000MiB;12h setup ceiling; full30 fit budgets projected with
the original2x safety factor,900s overhead,48h ceiling and85% host/GPU limits.
One-time preparation time/child memory and actual catalog/dictionary coverage
are reviewed. All9 science fits submit independently only after the exact
setup is COMPLETED/0:0 and its receipt/acceptance authenticates. Old A jobs,
source, claims and outputs remain untouched. Results are exploratory validation;
original50k overlaps validation and there is no independent test.
