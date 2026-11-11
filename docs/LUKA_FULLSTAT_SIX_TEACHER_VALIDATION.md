# Six-teacher local validation

## 2026-11-11 memory-counter repair

The user's job8821154 log shows both native TRAIN KD exports completed in
6482 seconds, then a failure during startup: summed host peak180.28GiB against
151.38GiB. The main process peak was48393801728bytes (45.07GiB); no student epoch
started. This is an explicit application MemoryError, not a module import
failure, Slurm timeout or reported OOM kill.

Linux `ru_maxrss` survives exec and can carry the parent's pre-exec peak into
spawned workers. The old sum therefore charged the parent image again for each
worker. The new versioned worker-memory policy reads post-exec `VmHWM`, keeps
the old counters as diagnostics, requires all three worker measurements and
retains the original host/GPU safety bounds. See the
[Linux getrusage documentation](https://man7.org/linux/man-pages/man2/getrusage.2.html)
and [kernel process-memory implementation](https://github.com/torvalds/linux/blob/master/fs/proc/task_mmu.c).

Before edits, all52 six-teacher tests passed. The repair suite covers malformed
and missing worker counters, a reconstruction of the45GiB-times-four false
rejection, rejection of genuinely excessive worker memory, exact original-source
export replay, narrow changed-source admission, no-epoch/failed-startup guards,
byte-identical export copies, preserved exporter lineage across another resume,
and corruption rejection. The production-depth local CUDA/Weaver startup and
existing sampler parity pass with host-memory fixtures on Windows.

The Linux standard-library reproduction in
`tests/fixtures/luka_six_teacher_memory_smoke.py` passed through real WSL Ubuntu
spawned processes: parent peak287264768bytes; three worker lifetime counters
287444992bytes each; actual worker address-space peaks18976768,18952192,
19083264bytes. The patched reader admits the approximately328MiB total under a
512MiB test bound; the old approximately1.07GiB sum falsely rejects it. This is
an OS-accounting regression, not a genuine Oscar data miniature.

Both new contracts (`WORKER_MEMORY_POLICY/v1`, `EXPORT_RECOVERY/v1`) are
operational; recipe, losses, model, optimizer, candidate count, sampler and data
split are unchanged. Recovery from15808976/job8821154 reuses both complete banks
and starts a fresh20-epoch student. Production resource acceptance still runs
inside that job. No remote jobs, source changes or data mutations were performed.

## Original implementation evidence

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
