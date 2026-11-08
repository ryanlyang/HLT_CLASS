# OFFLINE pretext and structure: validation record

Date: 2026-11-08. Internal donor/source baseline:
`a6da7a02b773d2397ff0d301bb4b5b400fc7528c`.

The additive implementation is ready for the documented commit/export/queue
workflow. It has not run scientific fits on Brown, submitted jobs, or measured
recovery. Real-data installed-Weaver and RTX A6000 qualification is performed
by the setup job before the durable CPU release can submit science.

## Local environment and commands

Python3.10 from `C:/Users/22rya/miniconda3/envs/tagging-hlt/python.exe`, PyTorch
with the local NVIDIA RTX2000 Ada Generation Laptop GPU (8GiB). The local
Weaver0.5.3 fixture is `tmp/hydra-weaver-fixture-20261102`. It exercises the
factory path but is not a claim about Brown's installed environment.
Independent FastJet3.5.1.5 was installed only under the temporary test directory;
it is neither a production dependency nor a modification of the active env.

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONPATH='tmp/pretext-fastjet-3.5.1.5;tmp/hydra-weaver-fixture-20261102;src'
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
& C:/Users/22rya/miniconda3/envs/tagging-hlt/python.exe -m pytest -q -p no:cacheprovider --tb=short tests/test_luka_pretext_structure_hydra.py
```

Final consolidated suite: **108 passed in249.36s**, including all six pinned
FastJet cases, the complete setup integration and production worker/export
paths. After adding the descriptive effective-rank and full-metric report
fields, the focused teacher-bank/worker/report checks passed **2 tests** in29.60s.

Donor baseline: **127 passed** in158.73s for
`tests/test_luka_capacity_hydra.py tests/test_luka_teacher_sweep_hydra.py`, using
the same Weaver fixture. Initial attempts without this fixture failed to import
Weaver; the documented environment resolves that local dependency. Donor runtime
files are unchanged by this campaign.

## Scientific and training coverage

- Exact30-fit registry, fixed budgets/ramps, unchanged U100 weight5 and native
  deployment. C_U initialization and one optimizer update match the original
  U100 auxiliary implementation exactly, including its predictor parameters.
- All8 teacher objectives and all22 student paths execute actual two-epoch
  production loops, publish epochs, select/reload checkpoints, and export native
  deployment. Synthetic populations and reduced depth are explicit for these
  exhaustive path tests.
- All8 teachers plus representative C_U/M04/S07/S08 use the full native model
  on genuine local CUDA with batch128 and partial batches. Set teachers retain
  298 output slots. This is not a full real-population maximum-support measurement;
  setup performs that stress separately on actual data.
- Actual30-epoch teacher/student/control workers with native reload, selected
  teacher bank export, actual-bank student startup, immutable task receipts and
  CPU reporting. A finite deliberately poor teacher completes30 epochs, selects
  the earliest tied epoch and exports its targets normally.
- Label-blind teacher validation, detached targets, isolated auxiliary-head RNG,
  class-head exclusion, clean-path/backbone gradients, permutation-stable masks,
  visible-only feature reconstruction, TRAIN-only normalization, constant-channel
  masks, empty-primary/single-particle and missing-tracking cases.
- Exact e2/e3 against independent tuple sums; permutation and global-phi symmetry;
  hand-calculated two-prong branching/soft-drop; complete298-particle support.
  Every cluster and E-scheme p4 agrees with pinned FastJet for N=1,2,3,12,64,298
  equivalent no-beam fixtures. Production tie breaking remains the explicitly
  registered physical-key convention, not an undocumented FastJet tie rule.
  Reference API: [FastJet classic interface](https://fastjet.readthedocs.io/en/latest/pseudojet.html).
- Mixed-set assignment invariance/gradients, source/role/identity firewalls,
  corruption detection, atomic bank publication and pre-allocation storage stop.
  Synthetic ROOT export roundtrip reconstructs the saved HLT/OFFLINE native
  features exactly and rejects a corrupted validation shard.

The C/A implementation caches coordinates and vectorizes distances without
changing the exact tree. One local synthetic all-family calculation took about
0.008s at60 particles and0.185s at298, compared with about0.110s and1.218s before
that optimization. These are illustrative single-host timings, not Hydra runtime
estimates or a real-population preparation benchmark.

## Integration, reporting and scheduler coverage

A full setup-worker test uses1804 synthetic TRAIN jets and22 validation jets,
actual target preparation, all8 technical selected-bank exports, all22 student
miniatures, twelve warmed updates per fit, separate phase diagnostics, full
validation/replay and all30 acceptance entries. It runs genuine local CUDA with
reduced model depth. Slurm allocation, A6000 identity, installed-environment
metadata and Linux peak-RSS values are simulated; this test cannot qualify the
real GPU/site. Production checks require the actual expected values.

Scheduler simulation verifies bootstrap setup/release, failed-setup prevention,
all33 total job submissions, exact teacher-to-student dependencies, afterany
report, completed-ledger idempotence and rejection of partial submission claims.
No remote scheduler was contacted. Read-only progress/results leave file hashes
unchanged, show uncompleted controls as missing, and distinguish teacher task
loss from HLT classifier recovery. CPU reports preserve incomplete rows.

Python runtime compilation, CLI registry, shell syntax, local Markdown links,
fit counts/order and whitespace are checked. The scoped source guard requires
all runtime/docs/tests to be committed, clean exact source and additive changes
relative to the donor baseline before registration.

## Execution boundary and next step

Follow the [runbook](LUKA_PRETEXT_STRUCTURE_HYDRA_RUNBOOK.md): scoped commit/push,
minimal physical validation export on SPORC, transfer to Brown, fresh registration,
then one authorization for measured setup and durable release. Existing physical
TRAIN data and completed teacher-sweep T00/S00 remain required and authenticated.
Allow32GiB for outputs plus2GiB personal headroom after transfers.

Setup must pass genuine full-data/native frontend replay, full/partial and
longest-support backward, actual installed factory parity, complete target/bank
publication,30 measured resource envelopes and concrete storage inventory.
Acceptance has no scientific-quality threshold. A technical failure preserves
its diagnostics and does not release the30 fits. No real-Weaver/Hydra or Tigris
production qualification is claimed by the local evidence above. This is an
additive Hydra campaign, not a change to the original Tigris transfer blocks.
