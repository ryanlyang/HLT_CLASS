# Group C validation and execution boundary

Internal donor: `3b839fc3aef7e1e24e886b9dd512a6a8114b5473`.
New modules in `luka_next_supervision_hydra`: `probabilistic_contracts`,
`probabilistic_model`, `probabilistic_targets`, `group_c`. They extend the same
trainer, worker, resource measurements, journals and read-only reports.
See the [contract](contracts/LUKA_NEXT_SUPERVISION_GROUP_C.md) and
[queue runbook](LUKA_NEXT_SUPERVISION_GROUP_C_RUNBOOK.md).

Local environment: Python3.10.19, Torch2.5.1, installed Weaver0.5.3 fixture
under `tmp/hydra-weaver-fixture-20261102`, NVIDIA RTX2000 Ada laptop GPU.
Synthetic tests do not establish real-data Hydra throughput or recovery.
Pre-change A/B baseline: **115 passed in115.30s**.

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONPATH='tmp/hydra-weaver-fixture-20261102;src'
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
$env:OMP_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:LOKY_MAX_CPU_COUNT='1'
& 'C:/Users/22rya/miniconda3/envs/tagging-hlt/python.exe' -B -m pytest -q -p no:cacheprovider tests/test_luka_next_supervision_hydra.py tests/test_luka_next_supervision_group_b.py tests/test_luka_next_supervision_group_c.py
```

Final combined A/B/C run: **154 passed in175.54s**. This includes38 new C tests
and the added C parameterization of the shared read-only result audit.
Static checks passed:24 module AST/import checks, all38 required files,
751 local documentation links with balanced fences, CLI create-b/create-c help
and40-method registry, plus2 Bash scripts and3 C-runbook Bash blocks.
Original B scientific recipe source matches the exact donor; A preserves hash
`d39c6ee09d803622385834756bb57c131ed8a8142659b1d99aec248ec6b6bddc`.
B recipe remains `5d6f3d7521e6ef65322c9a2724424deb92196dc85dd9f89ee3a42ea7834c3cee`.
No pinned donor runtime file changed. Scoped tracked/untracked whitespace
checks also pass, including PowerShell parsing of the commit/push block.

Coverage includes exact
joint mixture likelihood, coordinate-mixture counterexample, scale bounds,
detached targets/encoder gradients and negative-NLL full30 execution; raw-field
transforms, signed tracking/valid errors, noise/velocity identities, source
normalization, memory-mapped publication/readback and corrupt-byte rejection;
set permutation/padding/missing-value invariance and full collections above64;
all seven fit/export/reload paths, frozen target parentage, native/w5 RNG parity;
four-path warmed resource acceptance, target-tail miniatures, source/spec
immutability, exact setup admission, C-specific authorization and idempotent
dependency-free submission. CUDA full128/partial2 checks use production native
width on synthetic short HLT inputs, not actual Brown multiplicities.

No live Brown allocation, real-data C miniature, C submission or Tigris run
was performed locally. Before scientific release, the exact clean pushed
source must pass its own genuine A6000 setup: full TRAIN source admission,
installed-Weaver/native-w5 parity, warmed four-path production miniatures,
full target cardinality/measurement support and memory/walltime bounds,
native-only checkpoint replay and authenticated exact setup accounting.
The implementation is queue ready for that technical setup; it is not a claim
of completed remote qualification. Poor finite quality never gates experiments.
