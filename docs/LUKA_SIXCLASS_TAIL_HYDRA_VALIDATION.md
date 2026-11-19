# Six-class Hydra tail validation

The focused implementation suite covers scientific identity, two-fit L40
resources/DAG, scheduler dry review, submission journals, ambiguous submissions,
binary transfer integrity and retries, archive traversal/links/extras,
original-byte preservation, selected teacher relocation, TRAIN bank row joins,
both 20-epoch miniature workers, export-only recovery, native-only publication,
selected metrics and progress, and actual Bash detachment after the tail exits.

The transferred-teacher fixture uses production workers with a small model and
small synthetic population. It trains the first four source models, exports
and streams their selected closure, makes their original paths unavailable,
and runs DE050 and DE000 from the received artifacts. This is an integration
test, not a measurement of real-data performance or L40 throughput.

Separate local CUDA tests exercise the unchanged production acceptance code for
DE050 and DE000 with installed Weaver 0.5.3 and the actual production model,
including gradients, miniature, BF16 forward, native reload and contrastive
candidate checks. Local hardware is NVIDIA RTX 2000 Ada Generation Laptop;
Linux host RSS is substituted in that test, while CUDA measurements are real.
The local environment uses PyTorch 2.5.1+cu121. This does not certify Hydra L40
walltime, Linux host memory or the remote environment.

Baseline before edits: `tests/test_luka_sixclass_balanced_oscar.py`: **54 passed**.
Combined existing/new suites: **79 passed in 163.24 seconds**, no skips.
After adding a standalone receiver source-path/older-system-Python compatibility
guard, the focused transfer suite passed **15 tests in 6.39 seconds**. That adds
one distinct test, for **80 distinct passing tests** (54 existing and 26 new).
All five new Bash scripts pass `bash -n`; all 66 pinned donor files agree with
their canonical Git blobs. Original scientific source files remain unchanged.

The real production population metadata were also copied and hash-verified at
a new location: **7,000 files, 0.2458 GiB**. Production replay from the existing
6,996 source membership records retained population hash
`5092ce94a5e32fac10bdbd5b91b1becddc3fe9123affc41a93a0b312db2ea616` and exact
counts 1,658,937 TRAIN / 328,589 validation / 822,313 sealed in-range test.
Epoch-1 sampling retained 689,989 unique jets and order SHA256
`16d359b30383047d22318a46abd2fbda6aa3118f83aa0707124f680ccbe1f4ea`.
This read only scalar metadata and took 177.41 seconds. Evidence is saved in
`artifacts/audits/sixclass_tail_relocation_ek4j4gna/evidence.json` (ignored).

No teacher files have been fetched from Oscar or sent to Hydra during local
implementation. No jobs were submitted, cancelled or changed. No final-test
inference ran. Actual import/source/runtime verification and L40 acceptance
remain remote execution steps; they are enforced by the launch and worker.

Reproduce locally from the repository with the installed Weaver location:

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONPATH = "$PWD/src;$PWD/tmp/r8-followup-weaver"
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
& 'C:\Users\22rya\miniconda3\envs\tagging-hlt\python.exe' -B -m pytest -q -p no:cacheprovider `
    tests/test_luka_sixclass_tail_hydra.py tests/test_luka_sixclass_balanced_oscar.py
```
