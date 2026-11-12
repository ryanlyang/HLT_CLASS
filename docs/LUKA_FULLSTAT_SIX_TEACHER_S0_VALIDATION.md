# Six-teacher subclass-off validation

Scientific donor: 2fb09f38a3a1da85cdceff9df311963fdb3aa872 (Oscar job 8848995).
Implementation source baseline: 49b838adab43bf20d0440fc79467e2318aaebfdd.
The workspace advanced during this implementation to
27a8341210ca686b74d1c538a3ed258bb7fe947d, an independent bridge-memory change.
Its runtime diff contains only six `luka_embedding_bridge_hydra` files; none
is imported by this experiment. Source admission pins that reviewed descendant
so the user's intervening commit does not block the scoped subclass-off launch.
All 16 source/authentication tests passed again in 9.73s after that pin update.
The original six-teacher runtime remains unchanged. New identities distinguish
the subclass-0 objective from the completed subclass-0.5 reference.

Before implementation, 69 focused donor tests passed in 115.65s in the installed
tagging-hlt environment, with one non-fatal Windows physical-core discovery
warning. The default system Python 3.14 lacks NumPy/package installation; its
initial collection failure was environmental, not a passing test run.

The combined new suite passed all 46 tests in 62.20s, with the same non-fatal
Windows warning. An additional export-recovery-chain regression then passed
in 9.08s: 47 distinct new tests pass. Coverage includes:

- Exact recipe delta, identical initial state/RNG, CPU/CUDA objectives at
  partial/full batch, zero subclass gradient and unchanged other gradients.
- CE-only native classifier, identical deployment graph, full 20-epoch fit,
  exact optimizer/RNG continuation and publication-only recovery.
- Completed-control source/report/receipt authentication, actual Git snapshot
  checks, altered-source and artifact rejection, and exact submission bindings.
- Byte-identical six-bank reuse with teacher forwards, view reconstruction
  and reclustering forbidden; the original exporter identity survives the
  completed control's export-recovery chain.
- Full worker publication, late publication failure and explicit recovery,
  saved prediction/result comparisons, and corruption rejection.
- Exactly one detached submission, saved dry review, ambiguous-response
  rejection, repeat-launch refusal and completion after the foreground tail exits.
- Installed full-depth Weaver on local CUDA: sampler parity, 1,025 candidates,
  batch 128, 64 timed optimizer updates, three CPU candidate workers, native
  classifier/deployment parity and zero-subclass-gradient acceptance.

The startup fixture uses synthetic populations, a fixture parent host-memory
measurement and Windows worker-memory stubs. The shell fixture stubs setsid.
These checks do not certify real Linux SSH, Oscar host memory or FullStat
throughput. Those are checked inside the registered allocation before training.

Static checks passed for 20 Python files, five Bash scripts/documented blocks,
three CLI help paths, new documentation links and scoped whitespace. The
original six-teacher runtime has no working-tree diff.

No remote jobs or real Oscar/Tigris data have been run by this implementation
session. No final-test data were accessed. Queue-ready means ready for a scoped
pushed commit and the registered integrated checks on Oscar; it does not claim
scientific gains or completed remote qualification.

## Reproduce local checks

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONNOUSERSITE='1'
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
$env:PYTHONPATH="$PWD/src;$PWD/tmp/hydra-weaver-fixture-20261102"
& 'C:/Users/22rya/miniconda3/envs/tagging-hlt/python.exe' -B -m pytest -q -p no:cacheprovider tests/test_luka_fullstat_six_teacher_s0.py tests/test_luka_fullstat_six_teacher_s0_source.py tests/test_luka_fullstat_six_teacher_s0_lifecycle.py
```
