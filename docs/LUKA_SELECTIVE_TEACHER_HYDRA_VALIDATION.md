# Selective teacher Hydra validation

Scientific authority: [plan](plans/LUKA_SELECTIVE_TEACHER_HYDRA_PLAN.md).
Internal donor commit: `6fe7e3499767f0a2216586d965efe39a05c24bf7`.
No remote jobs have been submitted by this implementation session.

## Local checks

Baseline teacher-sweep and reconstruction-release tests: **91 passed in90.93s**.
Final combined selective-campaign, teacher-sweep and reconstruction-release
suite: **151 passed in150.47s**. The subsequently added immutable-registration
round-trip check separately passed (**1 passed,60 deselected in6.80s**):
**152 distinct focused tests passed** overall, including61 selective-campaign
tests. CLI help, registered file inventory, Python compilation, Bash syntax
(scripts and runbook blocks), relative documentation links and whitespace
checks passed.

Reproduce the final set from PowerShell:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONPATH='tmp/hydra-weaver-fixture-20261102;src'
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
& 'C:/Users/22rya/miniconda3/envs/tagging-hlt/python.exe' -B -m pytest -q -p no:cacheprovider `
    tests/test_luka_selective_teacher_hydra.py `
    tests/test_luka_teacher_sweep_hydra.py `
    tests/test_luka_reconstruction_release.py
```

Local environment: Windows, Python3.10 in `tagging-hlt`, installed local PyTorch
and the existing `tmp/hydra-weaver-fixture-20261102` Weaver fixture. Local CUDA
device is RTX2000 Ada Laptop8GiB, not Hydra's RTX A6000. Reduced one-block
synthetic full30 tests and separate unchanged eight-block native batch128 tests
have different purposes; neither is authoritative Brown production evidence.

Checks cover field isolation against the original raw-field frontend, support
and fingerprint enforcement, missing validation-cache reconstruction with exact
historical bytes, target joins/normalization/corruption, frozen targets, native
inference removal, all teacher/student objectives and full/partial GPU updates,
two-epoch checkpoint/export/reload, full30 transfer, all12 setup profiles,
actual multi-teacher worker execution/reporting, fixed total weights, original
parent authentication, exact dependencies, automatic release and failed/partial
submission behavior. Weak finite teachers complete; no quality gate exists.

## Remote boundary

The code is locally ready to queue **setup**, after commit/push and input
authentication. Real setup must still prove installed-Weaver parity, original
population fingerprints, native selected-checkpoint replay, warmed throughput,
resource limits and actual A6000 allocation. CPU release checks its exact
COMPLETED0:0 status and hashes before submitting science. Each student performs
another disposable miniature with real completed teacher banks. No remote
throughput, start-time, or recovery claim is made locally.
