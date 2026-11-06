# Direct OFFLINE reconstruction: validation and execution boundary

## R3 execution repair after Brown setup153208

The user supplied the setup log and `resource_R1.json` from
`next_R_57789d36_a6000_v2_r1`. It prepared complete OFFLINE/HLT TRAIN banks and
ran warm/measured R1 miniatures, then stopped on time admission only:
2.030168seconds/full batch,4791 projected minutes versus2880 allowed;
host peak5,144,285,184bytes and GPU peak6,891,415,552bytes. Fixed K is298.
The projection includes the unchanged2x timing margin; it is not an observed
80-hour training job. Matching timings were0.843624/0.904818seconds on the full
batches versus decoder-forward0.130157/0.129955seconds. No science release from
this failed setup is claimed. These are user-supplied remote measurements.

R3 adds batched cost/matched-loss execution and six-thread independent exact
SciPy assignments. CPU canonicalization avoids per-jet GPU-to-CPU-to-GPU
target copies; a chunk transfers its detached cost once. The original scalar
objective is retained unchanged. SPEC/v3 records the execution change; scientific
recipe hash, all298 slots, all particles, FP32, chunk8, native batch128,30epochs,
CE+U100-w5, scientific order and48-hour/85%-memory limits remain unchanged.

Pre-edit baseline:43 passed in130.86s. Initial scalar/batched CPU/CUDA loss,
component and prediction-gradient parity plus full encoder/decoder VJP checks:
26 passed in8.06s. These exercise empty/single/full/variable counts, canonical
permutations and duplicate ties, mixed track validity, malformed/nonfinite
targets and predictions, consistency and the six-thread solver path. Error
tolerances are recorded in `tests/test_luka_reconstruction_batched.py`.

Full R/scalar-batched/release verification: **91 passed in97.19s**. Focused
A/B/C registry/submission regression: **12 passed in7.67s**. The CUDA stress
case was then expanded to the observed298-slot support with128jets and mixed
target counts298/87/3, token memory and whole-jet consistency; it passed in8.14s.
Static checks passed changed-file Python ASTs/whitespace,53 required files,
three Bash runbook blocks, the PowerShell commit block and R documentation links.

Synthetic CUDA benchmark on the local RTX2000 Ada laptop, Torch2.5.1: native
batch128,decoder chunk8,K298, identical initialized decoder and HLT conditions,
one warmup and three timed repetitions. These timings cover decoder/loss/
backward only; they exclude native classification, validation, setup and probes.

| Targets per jet | Scalar median | Batched/parallel median | Ratio |
| --- | ---: | ---: | ---: |
|87|1.8602s|0.6210s|3.0x|
|298|4.8878s|1.3484s|3.6x|

Mean losses differed by at most2.3e-7 in this benchmark. Batched GPU allocated
peaks were348,027,392 and383,144,448bytes respectively; these exclude the native
classifier and are not campaign memory estimates. Scratch evidence is
`tmp/reconstruction_r3/final_benchmark.json`. This is local synthetic evidence,
not a measured Hydra speedup or a replacement for resource admission.

The new execution needs its own clean pushed-source A6000 setup and six real
measured paths. No time ceiling was raised; no Hydra/Tigris job was submitted,
canceled or modified locally. Prior R1/R2 evidence below remains historical.

## Earlier scientific implementation and R2 release checks

Internal donor: `505b07364041d4f51c42ac9e6a2f85353258606a`.
See the [source map](LEGACY_SOURCE_MAP.md), [contract](contracts/LUKA_DIRECT_OFFLINE_RECONSTRUCTION.md)
and [Hydra runbook](LUKA_DIRECT_OFFLINE_RECONSTRUCTION_RUNBOOK.md).

The scientific implementation adds six reconstruction modules, a separate twelve-row
scope and focused tests. Original A/B/C recipe identities remain unchanged.
No new dependency is installed: PyTorch and SciPy are in the pinned environment.
No external source implementation was copied.

The subsequent R2 orchestration adds `reconstruction_release.py`, bootstrap
CLI/queue support and `tests/test_luka_reconstruction_release.py`. SPEC/v2 binds
the automatic policy; the scientific v1 recipe hash is unchanged. Its exact
Slurm journal tests simulate scheduler responses; they do not claim a real
Brown CPU allocation. The pre-change focused baseline passed4 tests in10.53s;
the first release/R integration check passed30 tests in12.37s. Coverage includes
one-confirmation bootstrap, measured independent fit requests, exact accounting
lag/failure, authorization/job/source/receipt rejection, safe handling of lost
sbatch responses, idempotence, and observed Hydra CPU rounding. A/B/C retain
their original manual release semantics. The numeric evidence below describes
the earlier scientific implementation checks; final R2 checks are recorded next.

Final R2 regression: **61 passed,162 deselected in106.34s**, running the A/B/C/R
files plus `tests/test_luka_reconstruction_release.py` with selection
`submission or queue or release or registry or registration or create_check or
receipt or source or resource or admission or setup or immutable`. This retains
real local resource/receipt integration while mocking only the Slurm boundary
for submission tests. No model/loss change was made in the R2 step.
Static verification passed34 Python ASTs,51 required files, two shell scripts,
three Bash runbook blocks, the PowerShell commit block,16 R documentation links
and whitespace. Source remains uncommitted locally; no Brown/Tigris execution
or remote qualification is claimed.

Local environment uses Python3.10/Torch2.5.1 and the installed Weaver0.5.3
fixture under `tmp/hydra-weaver-fixture-20261102`, with an RTX2000 Ada laptop GPU.
The pre-change A/B/C baseline passed **154 tests in161.32s**.

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONPATH='tmp/hydra-weaver-fixture-20261102;src'
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
$env:OMP_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:LOKY_MAX_CPU_COUNT='1'
& 'C:/Users/22rya/miniconda3/envs/tagging-hlt/python.exe' -B -m pytest -q -p no:cacheprovider tests/test_luka_next_supervision_hydra.py tests/test_luka_next_supervision_group_b.py tests/test_luka_next_supervision_group_c.py tests/test_luka_direct_reconstruction.py
```

Combined A/B/C/R verification: **197 passed in273.10s**. The final R-only
rerun after freezing exact uniform categorical/count initialization and
presence0.1 passed **43 tests in131.68s**. This includes six CUDA
full128/partial2 production-width paths, a257-slot/full128 bounded-decoder
stress case, and six-path warmed CPU execution/acceptance with synthetic data.
The first focused run after correcting the poor-quality test fixture passed
35 checks in66.85s.
The tiny synthetic classifier had initially learned its test fixture perfectly;
the corrected test uses deliberately mismatched validation labels to exercise
full30 completion despite poor finite quality. No scientific implementation
was changed to accommodate that fixture assertion.

Static checks passed:32 Python ASTs, all49 required files present,745 local
documentation links, fences and tracked/new-file whitespace; two Bash scripts,
three Bash runbook blocks and the PowerShell commit block parse. CLI create-r
help and additive registry/source inventory checks passed. R recipe hash:
`90b582b9f37955f13217ec36f3f724c83ccecd7c88c9d9983ce8bce6bda345cc`.
Original A and B recipe hashes are unchanged; C remains
`99d63190c1ee3d2f6965eb6f7dd4fc005e9054bc50b8c83d09f5e2933816f273`.
Pinned historical runtime donor paths are unchanged. Unrelated local work
was preserved. New source files still need to be committed/pushed before
the immutable-source registration guard will accept them on Brown.

Coverage includes rectangular assignment against exhaustive enumeration;
permutation/duplicate/empty sets; invalid-field masking; count/presence/group
normalization; signed mass and physical inverse; full targets/common K;
decoder conditioning/RNG/native parity; gradient equivalence between full and
chunked decoder backward including both h and T VJPs; all twelve native
fit/export/replay paths; resource acceptance, immutable source/target references,
quota checks, exact setup accounting and independent idempotent submission.

No live Brown allocation, actual TRAIN inventory, real-data R miniature,
scientific submission or Tigris run was performed locally. Synthetic CPU/CUDA
tests do not establish recovery, actual K, memory or runtime on Brown. The
exact clean pushed commit must pass its own genuine A6000 setup, including
all real target/native multiplicity tails and installed environment parity,
before scientific release. Queue readiness means readiness to submit this
technical setup, not completed remote qualification. The implementation is
locally queue ready for that setup.
