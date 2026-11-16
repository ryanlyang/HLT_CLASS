# EMA control Hydra validation

The approved design is implemented in the separate `luka_ema_control_hydra`
package, CLI and Slurm wrappers. It has not been submitted remotely by this
implementation session. Single-A6000/full-population Hydra qualification is
mandatory in setup before automatic release.

The broader local regression passed: **89 passed, 1 skipped in 344.28 seconds**
across the EMA comparison, reconstruction stability and SOFT recovery tests.
The skipped test requires four CUDA devices and NCCL for the older stability
campaign; this environment has one CUDA device. The final focused campaign
suite passed: **31 passed in 116.36 seconds**, with no skips. These suites
overlap; their counts are not additive.

Local checks cover:

- All nine frozen starts, named-tensor authentication and corruption rejection.
- Independent full-graph versus production updates for CE, U100 and R24 at
  five transition positions, using installed Weaver on CUDA.
- Full 298-slot chunked reconstruction on CUDA, including empty and maximum
  constituent targets, and native/RNG/optimizer agreement between U100 and
  R24 while the reconstruction coefficient is zero.
- Production miniatures and all four native export/reload paths for each arm;
  a separate portable 30-epoch triplet checks selection and paired reporting.
- CE target exclusion in both the batch loader and worker, and isolated EMA /
  BatchNorm readouts that preserve live training state and RNG.
- The exact 12-job submission graph, completed-submission idempotence, failed
  setup and ambiguous partial-submission rejection, and resource failure cases.
- Same-seed/readout comparisons, incomplete-seed denominators, TRAIN-only
  calibration membership and read-only reporting when output space is low.

The runtime imports successfully against the exact scientific donor at
58d0c0edf5f41cff2398fb168fc16fc92e169294. Python syntax, Bash wrappers and
runbook snippets, and focused documentation links pass local checks.

The installed-Weaver serialization inventory with 298 output slots estimates
9 GiB of output allowance plus a 1 GiB free-space reserve. Registration and
setup verify the actual inventory; this is not a remote quota measurement.

Focused local command:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONPATH="$PWD/src;$PWD/tmp/r8-followup-weaver"
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
& 'C:/Users/22rya/miniconda3/envs/tagging-hlt/python.exe' -m pytest -q -p no:cacheprovider tests/test_luka_ema_control_hydra.py
```

The local environment has one CUDA device. This does not establish A6000
runtime, memory or production-data acceptance. Setup measures those values,
verifies 15 transition comparisons and nine discarded production miniatures,
checks full support and actual storage, then publishes its acceptance receipt.
Poor finite classification results never constitute technical failure.

Donors: 58d0c0edf5f41cff2398fb168fc16fc92e169294 scientific reconstruction;
3ccc35ae709c63d8833cb0388c618b53dcde5b9e readout/lineage design;
d28c403360f7b4662f679f5564529a048d8db5e2 stability orchestration, adapted to
a distinct nine-fit recipe. See the [source map](LEGACY_SOURCE_MAP.md).
Old campaign workers and artifacts remain unchanged.
