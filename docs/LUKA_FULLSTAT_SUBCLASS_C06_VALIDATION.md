# Contrastive0.6/subclass0.5 local validation

Evidence dated2026-11-10. No Oscar connection, research-data inference,
final-test access, submission, commit or push was performed by this task.

## Checks performed

- Existing subclass baseline before edits: **35 passed in83.93s**.
- New model/recipe/queue/training/startup/recovery suite: **34 passed in63.84s**.
- Frozen-target reuse plus complete worker/publication recovery: **11 passed
  in27.41s**.
- Additional historical-source authentication/report/target corruption guards:
  **8 passed in13.06s**.
- Total new coverage: **53 distinct passing cases**. Python ASTs, Bash scripts
  and runbook blocks, forbidden imports, whitespace and documentation links
  were checked. Historical subclass runtime files were unchanged.

CPU and CUDA gradient comparisons use real student forwards, full1025-key
objectives and full/partial batches. Identical initialization, native and
auxiliary tensors, raw loss components and subclass gradients are preserved.
At matching0.3/subclass0.5 the independent objective exactly reproduces the
previous implementation's loss and gradients. At matching0.6/subclass0.5 its
gradient matches the old objective plus an extra0.3 matching term within the
tested floating-point tolerance. Invalid/NaN coefficients are rejected.
Ramp comparisons retain the original subclass coefficient at every sampled
warmup/end point while doubling only contrastive strength.

The production integrated startup path ran with the actual installed-Weaver
architecture on local CUDA, synthetic6006-row TRAIN input and all1025
candidates, including64 measured complete updates, longest-support stress,
classifier CE-only gradients, auxiliary connectivity and native deployment
parity. This is local synthetic evidence, not real Oscar data acceptance.

Synthetic CPU/CUDA workers execute20 epochs, strict native reload/validation
replay, exact epoch-boundary continuation, and publication-only recovery with
optimization forbidden after20 saved epochs. Frozen-bank tests forbid target
fitting, assert byte-identical original/copied manifests, probabilities and
prototypes, and reject wrong roles, row order, prepared identity, modified
payloads and rewritten target identities. Full four-bank loading preserves the
original OFFLINE normalized keys. Historical authentication tests mock the
remote scheduler/replay boundary; actual historical completion remains checked
in the pinned original Oscar worktree before queueing.

Queue tests cover one-task dry/live plans, bounded authorization, ambiguous
submission, duplicate launch refusal, exact memory/time/GPU allocation guards,
and detached completion after the foreground tail exits. Windows shell stubs
test orchestration; they are not a live SSH-disconnection experiment.

Environment: Python3.10, PyTorch2.5.1, installed Weaver0.5.3 through the existing
`tmp/hydra-weaver-fixture-20261102`, CPU and local CUDA. Tests used
PYTHONNOUSERSITE=1, PYTHONDONTWRITEBYTECODE=1 and CUBLAS_WORKSPACE_CONFIG=:4096:8.
Joblib emitted its existing Windows physical-core detection warning and used
logical core count; target construction still limits numerical threads to one.

Reproduce the new suite with the scientific Python environment:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONNOUSERSITE='1'
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
$env:PYTHONPATH="$PWD/src;$PWD/tmp/hydra-weaver-fixture-20261102"
& 'C:/Users/22rya/miniconda3/envs/tagging-hlt/python.exe' -B -m pytest -q -p no:cacheprovider tests/test_luka_fullstat_subclass_c06.py tests/test_luka_fullstat_subclass_c06_targets.py tests/test_luka_fullstat_subclass_c06_lifecycle.py
```

Next: scoped commit/push and the [Oscar runbook](LUKA_FULLSTAT_SUBCLASS_C06_RUNBOOK.md).
The one scientific allocation repeats real installed-environment, sampler and
resource admission before fresh initialization. The48h limit is a ceiling,
not a runtime prediction; no separate preflight job is registered.
