# Six-teacher local validation

Baseline:53 focused subclass-C06 tests passed before this implementation and
again as a regression check. All52 distinct new cases passed across focused
runs (32 model/runtime/queue,18 target/source,2 lifecycle cases). The final
target/source/lifecycle plus donor regression run passed73 cases; the last
four export cases were rerun after extending them through the real preparation
entrypoint and also passed. The frozen CPU reference must disable parameter
gradients, just like the export teacher, to exercise the same inference kernels.
One Windows physical-core detection warning from joblib was non-fatal.
Python AST/whitespace, CLI help, Bash syntax and relative documentation links
passed. `git diff --check` passed, with only existing CRLF conversion notices.

Coverage includes:

- CPU and CUDA full/partial-batch six-head objectives, explicit KL arithmetic,
  coefficient ramps, independent seeds, connected heads and CE-only native
  classifier gradients; native deployment parity.
- Original six embedding/KD bank readers, ordered identities, labels, roles,
  checksums and byte-identical frozen subclass targets and OFFLINE keys.
- Actual native CONCAT/FUSION T2 export, full128 and final partial batch,
  CPU/CUDA numerical roundtrip, invalid joins and payload tampering.
- Full20-epoch reduced-architecture fixtures on CPU/CUDA, exact epoch-boundary
  optimizer/RNG continuation, selected native publication, publication-only
  recovery and corrupt-state rejection.
- Full production architecture and installed Weaver on local CUDA, real1025-key
  objective, exact sampler parity,64 measured complete updates and three CPU
  candidate workers. Host memory and data are synthetic fixtures, not Oscar
  resource evidence.
- Original-source replay boundaries, dedicated roots, strict source ancestry,
  scheduler allocation, dry review, ambiguous submission, exclusive claims and
  background launcher completion after the foreground tail exits. The local
  Windows setsid stub does not certify real Linux SSH behavior.

Run with the installed local environment and matching Weaver import path:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONNOUSERSITE='1'
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
$env:PYTHONPATH="$PWD/src;$PWD/tmp/hydra-weaver-fixture-20261102"
& 'C:/Users/22rya/miniconda3/envs/tagging-hlt/python.exe' -B -m pytest -q -p no:cacheprovider tests/test_luka_fullstat_six_teacher.py tests/test_luka_fullstat_six_teacher_targets.py tests/test_luka_fullstat_six_teacher_lifecycle.py
```

No actual Oscar/Tigris data miniature, scheduler submission or scientific result
was produced locally. Queue readiness requires a scoped pushed commit and
clean detached worktree. The registered job performs original-source/data
authentication and genuine allocation admission before scientific updates.
No separate preflight job is submitted; no accuracy gain or runtime speedup is
claimed from synthetic tests. Final-test data were not accessed.
