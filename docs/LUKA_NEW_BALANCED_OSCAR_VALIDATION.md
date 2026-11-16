# New-data balanced Oscar main-KD validation

Implementation baseline: `ec541496de9120cd046f5eb351a7bc0f97da31ad`.
Authority: [plan](plans/LUKA_NEW_BALANCED_OSCAR_PLAN.md).
Execution/training contracts: `LUKA_NEW_BALANCED_OSCAR_*/v1`; unchanged shared
population: `LUKA_NEW_BALANCED_HYDRA_POPULATION/v1`.

## Local evidence

Final focused suite: **45 passed, zero skipped**, 157.67 seconds. Bash syntax
checks passed for all four new shell entrypoints/helpers, and CLI help passed.
After the final submission review, ten focused queue/locking/semantic-guard
tests passed in 7.02 seconds. Python syntax, new Markdown links and whitespace
checks also passed.
The donor baseline checks passed 34 cases (31 initially; three CUDA cases rerun
successfully after setting their required CUBLAS environment variable).

The real uploaded dataset's scalar membership metadata was replayed locally
through the production population builder. It reproduced exactly:

```
7f2413152ecb588764c862b4b83bd6f900320a1b915a3b77f95540dd6e283939
TRAIN 2,425,536; validation 500,000; sealed in-range test 1,200,000;
remaining unused in-range 689; all 198 TRAIN cells nonempty (minimum 306).
```

The replay reads scalar membership metadata and writes an audit artifact under
`artifacts/audits/new_balanced_oscar_population_v1`; it performs no particle
inference or final-test evaluation. Parent roles and the unused-only extension
are independently replayed by the donor loader/builder. The exact matching hash
proves the population identity matches Hydra rather than merely matching counts.

Recorded maximum particle support is TRAIN HLT251 / OFFLINE270 / CONCAT521;
in-range validation HLT221 / OFFLINE251 / CONCAT467; outside validation HLT240 /
OFFLINE291 / CONCAT531. These are scalar maxima, not measured full-data GPU
memory. Five outgoing banks plus all 120 full AdamW epoch states and result
payloads fit the 14 GiB budget: native state is 8.179 MiB and the largest auxiliary
model state is 8.831 MiB; tensor payload for 120 model/AdamW states totals about
2.97 GiB, and five TRAIN target banks about 6.73 GiB before metadata/headroom.

All 21 frozen donor file hashes match their canonical blobs at the recorded
commit and current local source. New source snapshots cover the complete clean
pushed checkout at registration. The foundation's own producer bytes are checked
separately and no teacher checkpoints are imported from another campaign.

Focused validation command:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONPATH="$PWD/src;$PWD/tmp/r8-followup-weaver"
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
& 'C:/Users/22rya/miniconda3/envs/tagging-hlt/python.exe' -B -s -m pytest -q -p no:cacheprovider tests/test_luka_new_balanced_oscar.py
```

The local CUDA tests use an NVIDIA RTX 2000 Ada Generation Laptop GPU,
PyTorch2.5.1/CUDA12.1 and the installed Weaver-core0.5.3 distribution under
`tmp/r8-followup-weaver`. They exercise all six production stage architectures,
FP32 installed-Weaver parity, BF16 production forwards, two-epoch full-objective
miniatures/reloads, native deployment parity, classifier-gradient identity,
contrastive serial/optimized parity and real candidate-worker execution on
synthetic rows. Linux RSS collection is replaced by a bounded fixture on Windows;
CUDA forward/backward and peak allocation are real. Synthetic support lengths
do not establish full real-data resource sufficiency.

Other focused coverage includes exact predecessor KL and coefficients at every
rung, unchanged initialization/RNG, no double-softening, all six 20-epoch
production-worker miniature publications, selected native weights, fresh banks
and anchor results, bounded ROOT slices excluding sealed rows, payload corruption,
epoch-boundary continuation, export-only recovery, quota accounting, exact Oscar
allocation, dry review, ambiguous submissions and process-level writer exclusion.
Real Bash/nohup tests verify the detached launcher finishes after its watching
tail exits and that duplicate launch is rejected. This is not a remote SSH test.

## Remaining site validation

No new Oscar jobs have been submitted and the current upload's completion has
not yet been reported by the user. Production queueing requires the verified
scratch transfer, clean pushed commit, healthy explicit runtime and declared
HOME headroom. There is no additional standalone setup/preflight job.

Each science allocation must pass real full-support parity/miniature checks,
64 measured representative training updates, validation/export probes, memory
headroom and conservative remaining-walltime projection before science. Those
actual Oscar measurements are pending. Resource failure preserves diagnostics
and does not silently change batch size, precision, loss weights or jet support.
Tigris validation is not claimed; this user-authorized campaign targets Oscar.
