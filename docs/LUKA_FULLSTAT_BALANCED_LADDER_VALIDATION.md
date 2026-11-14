# Balanced FullStat ladder validation

Date:2026-11-13. Baseline read/adapted:
`1094dea4ccfb36d08e29f740a97d90fc770335ec`.
New runtime is isolated under `luka_fullstat_balanced_ladder`; historical
campaigns and their results are unchanged. No commit, push or remote job was
performed during implementation.

## Scientific and workflow evidence

- Frozen198-cell occupancy and2,425,536 eligible TRAIN rows; exact half-open
  bounds; no role substitution; same stateless draw order across all six fits;
  replacement draws, repeated identity gathering and final partial batch.
- Real synthetic ROOT reads for D050 and CONCAT. Tests guard every requested
  particle slice against final-test/unused roles. Serial and four-process
  construction produce identical bytes and order. D050 replays native HLT;
  CONCAT retains full union support.
- All six full native architectures pass installed-Weaver FP32 parity and
  FP32/BF16 native deployment comparisons on the local CUDA GPU. Auxiliary losses
  connect to the encoder while the main head receives CE gradients only.
- The complete integrated startup probe runs on CUDA for UE100 and DE000 with
  synthetic populations, full native architectures and three actual candidate
  workers. GPU peaks, forward/backward, serial matching parity and timing are
  real; Linux host RSS is substituted on Windows. This is not a FullStat resource
  measurement or Oscar acceptance.
- Serial, optimized, prefetched and two-process contrastive candidates agree,
  including score ties and repeated query identities; positives excluded and
  negatives distinct. These tests use a sufficiently large synthetic bank.
- Six production worker functions complete20 epochs each in a reduced native
  architecture CPU fixture,
  publish selected native checkpoints, five new target banks and all metric
  populations. Scheduler, resource admission and the final small candidate
  population are substituted in this integration fixture; it is not Oscar
  acceptance. Native model/loss/optimizer/export/validation code runs normally.
- A20-epoch training run resumed after epoch10 exactly reproduces subsequent
  weights and selected metrics. Restoring all20 epochs executes zero optimizer
  updates. Injected export failure recovers in a new root without retraining.
- Malformed semantics, corrupt banks/epoch states, identity mismatches, missing
  classes, censored rejection and exact fresh-anchor recovery are covered.
- Dry/live DAG tests preserve topological order after sorted JSON serialization,
  bind all teacher dependencies, reject duplicate submission and retain journals
  on ambiguous scheduler replies.
- Exact D050 arithmetic retains half of each matched endpoint and preserves
  unmatched HLT particles. CONCAT permutation checks preserve the full union.
  Result reuse checks the fit binding against the actual teacher receipts,
  integrated acceptance flags, selected epoch and twenty committed epochs.
- The launcher is tested through dry review, approval and submission, including
  failure records and exact allocation checks. Bash syntax and detached wrapper
  smoke tests pass; Windows uses a setsid stub, so this is not a real Linux SSH
  disconnect test.

## Commands and environment

Healthy local Python:
`C:/Users/22rya/miniconda3/envs/tagging-hlt/python.exe`.
Existing installed-Weaver test tree: `tmp/r8-followup-weaver`.
Local GPU: NVIDIA RTX2000 Ada Generation Laptop GPU; PyTorch2.5.1.

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
$env:PYTHONPATH="$PWD/src;$PWD/tmp/r8-followup-weaver"
& 'C:/Users/22rya/miniconda3/envs/tagging-hlt/python.exe' -m pytest -q -p no:cacheprovider tests/test_luka_fullstat_balanced_ladder.py tests/test_luka_fullstat_balanced_ladder_workflow.py tests/test_luka_fullstat_balanced_ladder_launch.py tests/test_luka_fullstat_pt_audit.py tests/test_luka_fullstat_pt_comparison.py
```

Initial model checks exposed a missing Weaver import path in the
local environment; pointing to the existing test installation resolved it.
CUDA parity additionally requires CUBLAS_WORKSPACE_CONFIG before Python starts.
Neither issue required a package installation or a scientific-code workaround.

Final evidence covers **111 distinct passing tests**, with no skips:

- Aggregate:110 passed and one invalid synthetic D050 mapping rejected in117.54s.
  That fixture accidentally left an HLT slot unmatched despite having more
  offline than HLT particles, violating the donor full-cardinality contract.
- Corrected fixture and final affected production/recovery paths:3 passed,
  17 deselected in36.37s. This includes the D050 case and the six-stage miniature
  and export-recovery reruns after the final lineage checks were added.
- Static checks:24 Python ASTs,24 local Markdown links, Bash syntax/detach smoke
  and whitespace. The initial audit/comparison baseline was61 passing tests.

The focused rerun command was:

```powershell
& 'C:/Users/22rya/miniconda3/envs/tagging-hlt/python.exe' -m pytest -q -p no:cacheprovider tests/test_luka_fullstat_balanced_ladder_workflow.py -k 'd050_exact or six_production_workers or epoch_boundary_failure'
```

## Remote acceptance still required

No real Oscar/Tigris miniature or full-data resource qualification has run in
this local task. The approved Oscar plan performs the real checks inside each
science allocation before fresh initialization.48h is a request ceiling, not
a measured promise. Measurements include full-support stress, candidate CPU
workers, complete balanced updates, validation and outgoing exports, with
explicit time/memory reserves. Refusal preserves diagnostic files and never
changes batch size, teachers, particle support, sample law or loss coefficients.

The runtime on Oscar is fingerprinted against the existing FullStat environment.
The detached wrapper checks concrete imports first, including torch.nn.functional
and Weaver, to catch the scratch-runtime damage seen in earlier attempts.
