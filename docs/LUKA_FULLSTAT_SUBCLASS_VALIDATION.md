# FullStat subclass implementation validation

Local evidence,2026-11-10. No Oscar submission, remote data access, model
inference on research data, final-test access, commit or push was performed.

## Validation performed

- Combined new campaign/target and unchanged Multi3 contrastive regression:
  **72 passed in160.89s**.
- Additional complete target-bank →20-epoch worker → simulated publication
  failure → recovery → results test: **1 passed in23.49s**. It proves that
  recovery copies identical targets, retains the target hash in every epoch,
  and performs neither reclustering nor further optimization after20 epochs.
- Four focused recipe/source/bank checks passed after the final source-file
  inventory and prepared-population guard additions.
- Python ASTs, Bash entrypoints, runbook Bash blocks, forbidden-import scan,
  new-file whitespace/BOM checks and local documentation links checked.

The73 distinct passing cases cover CPU/CUDA training, exact epoch-boundary
continuation, native checkpoint stripping/reload, independent auxiliary
initialization, CE-only native-classifier gradients, real conditional KL,
zero gradients outside the selected subclass group, deterministic TRAIN-only
clustering and temperature calibration, all four actual teacher-bank readers,
ordered identity/label joins, finite degenerate prototypes, metadata/payload
corruption, exact scheduler/authorization journals, duplicate/ambiguous launch
rejection and detached Bash orchestration. The unchanged donor sampler tests
verify exact candidates, ties, exclusions and CPU worker/GPU lookahead parity.

The actual installed-Weaver architecture passed the production startup path on
local CUDA with synthetic6006-row TRAIN data, full1025-candidate objectives,
longest-support stress and64 complete measured updates. This includes the new
subclass head and native deployment parity. Synthetic miniature worker tests
use reduced model size and mocked scheduler/allocation boundaries; they do not
certify full-population Oscar resources or a scientific gain.

Environment: Python3.10, PyTorch2.5.1, Weaver0.5.3 from the existing isolated
`tmp/hydra-weaver-fixture-20261102` installation, CPU and local CUDA. Tests set
PYTHONPATH to src plus that installation, PYTHONNOUSERSITE=1,
PYTHONDONTWRITEBYTECODE=1, CUBLAS_WORKSPACE_CONFIG=:4096:8.
The initial donor run had29 passes and9 environment failures because Weaver
was not on PYTHONPATH and the CUDA parity setting was absent; no donor runtime
was changed to address them. The final combined run passed all38 donor cases.
One extra lifecycle test initially lacked its fixture parent directory; that
test-only setup was fixed. Windows emitted a joblib physical-core detection
warning and used logical core count; target fitting still explicitly limits
numerical libraries to one thread.

Focused reproduction (with the environment above):

```powershell
python -m pytest -q -p no:cacheprovider tests/test_luka_fullstat_subclass.py tests/test_luka_fullstat_subclass_targets.py tests/test_luka_fullstat_subclass_lifecycle.py tests/test_luka_fullstat_multi3_contrastive.py
```

## Remote boundary and next step

The implementation is ready for a scoped commit/push and the
[detached Oscar launcher](LUKA_FULLSTAT_SUBCLASS_RUNBOOK.md). Authentication of
job8361672 and its source/checkpoints/banks happens on Oscar. Full3M target
fitting, actual L40S timing, memory, filesystem health and remaining quota are
verified there. No separate preflight job is required; measured acceptance is
inside the one science allocation and precedes fresh scientific initialization.
No recovery gain or end-to-end speedup has been established by these tests.

Scientific donor baseline: `6ab66183a0906183887386d278f6ef202a891365`.
Queue execution ancestry: `d88eb82c6569312b72a3a3bccb1c39f9e357b4a4`.
Historical scientific parent: `a7249f49b6f522f7df3b150d27f53613452b7fb0`.
Donor files and adaptations are listed in [LEGACY_SOURCE_MAP](LEGACY_SOURCE_MAP.md).

Queue follow-up: the user committed a separate complementary-teacher Hydra
package as d88eb82c. The complete src/scripts/sbatch delta from6ab66183 contains
only additions under that package and its three entrypoints; existing Oscar
runtime bytes are unchanged. The subclass source guard now accepts d88eb82c
as execution ancestry while retaining rejection of later donor changes.
Source/registration baseline checks:10 passed in8.45s; after the adjustment,
source/registration/detached-launch checks:15 passed in10.37s. No scientific
recipe, contract version, model, data, runtime environment or resource changed.
