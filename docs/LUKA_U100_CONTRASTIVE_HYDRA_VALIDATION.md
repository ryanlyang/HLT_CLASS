# Contrastive Hydra implementation and validation

Date: 2026-11-03. Authority: [independent-fit plan](plans/LUKA_U100_CONTRASTIVE_HYDRA_PLAN.md).
Execution instructions: [runbook](LUKA_U100_CONTRASTIVE_HYDRA_RUNBOOK.md).

## Scope

The new `luka_u100_contrastive_hydra` package imports the original contrastive
model, loss, sampler, training and resource code unchanged. All remaining
students have empty dependency lists; the report alone waits with `afterany`.
The existing twenty recipes, thirty epochs, target weights, population and
native HLT inference remain fixed. Completed SPORC students are reused.

One shared Hydra setup authenticates inputs and runs the original six
production resource miniatures before automatic science submission. There is
no standalone preflight. Numerical qualification reuses completed diagnostic
131484 under the existing v2 policy. It does not repeat control inference.

The additional transfer contains only OFFLINE TRAIN keys and completed
reports, predictions and receipts, plus original-source metadata evidence.
Existing Brown HLT/U100 data, references and environment remain in place.
Existing SPORC jobs and partial outputs are not changed.

## Files and contracts

- `src/hlt_classification/luka_u100_contrastive_hydra/`: contracts, transfer,
  relocated inputs, campaign, submission and workers.
- `scripts/luka_u100_contrastive_hydra.py`: thin CLI.
- `scripts/queue_luka_u100_contrastive_hydra.sh`: pushed-source review and
  detached automatic setup/science submission.
- `sbatch/run_luka_u100_contrastive_hydra.sh`: isolated pinned-environment worker.
- `tests/test_luka_u100_contrastive_hydra.py`: relocation, lineage, creation,
  submission, failure and full-thirty-epoch integration coverage.
- Plan, contract and runbook linked above; living status/source map updated
  without replacing pre-existing documentation changes.

New execution contracts use `LUKA_U100_CONTRASTIVE_HYDRA_*/v1`: TRANSFER, SPEC,
INPUTS, TASK, RUN_CLAIM, REVIEW, PLAN, SUBMIT_CLAIM, DISPATCH_CLAIM and RESULTS.
Original training, targets and measured acceptance retain their versions,
binding the new execution spec as parent. The existing pair-Hydra numerical
PORTABILITY_EVIDENCE/PORTABILITY v2 contracts are reused unchanged.

## Donor and retained meaning

Internal donor baseline: `6f7a6cbc587800049c7e5eea6c22cc9af9026d03`.
No external source, Fresh_check imports or third-party files are migrated.
Paths below are relative to `src/hlt_classification/`.

| Unchanged donor files | Use |
| --- | --- |
| `luka_u100_contrastive/{contracts,model,losses,sampling,targets,training,resources,worker}.py` | Original twenty recipes, sampling, frozen keys, original training loop, native deployment and metrics/resource measurements |
| `luka_u100_contrastive_serial/source.py` | Original-source authentication only; no serial scheduler or patience stopping policy |
| `luka_u100_contrastive_serial/contracts.py` | Validate the source replay record only |
| `luka_u100_pair_hydra/{contracts,inputs,site,portability,submission}.py` | Existing bundle/environment/site, qualified numerical evidence, accounting and measured-envelope comparison |
| `scouting/{hcwdl_exact_dag_submission,hcwdl_recovery}.py` | Exact command review, exclusive journals and accepted job ownership |
| `data/cache_contracts.py`, `cms_proxy_ladder/{contracts,submission}.py`, `provenance.py`, `luka_fullsim/{foundation,regular_submission,preflight_v2}.py` | Hashes, file references, atomic publication, source locks, scheduler environment and backend checks |
| `jetclass2_delphes/{model,execution,reporting}.py`, `luka_embedding_weight/worker.py` | Environment/GPU identity, metrics and prediction-array validation |

The original science package has no Git diff between successful SPORC commit
`431341db5eced8eaa3e52b37ba047f8cd4beed50` and this donor baseline. The transfer
authenticates that original checkout in its own subprocess. It retains the
earlier v1 owner of any reused key bank rather than inventing a new target
parent. Source portability follows the already repaired foreign-checkout path.

## Local evidence

Before editing: 53 original-contrastive and Hydra-portability tests passed in
33.34 seconds. Adapter integration: 10 tests passed in 14.97 seconds.
Final combined suite: **152 passed in 68.42 seconds**, no skips or failures:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONNOUSERSITE='1'
$env:PYTHONPATH='src;tmp/hydra-weaver-fixture-20261102'
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
& C:/Users/22rya/miniconda3/envs/tagging-hlt/python.exe -m pytest -q -p no:cacheprovider `
    tests/test_luka_u100_contrastive_hydra.py `
    tests/test_luka_u100_contrastive.py `
    tests/test_luka_u100_contrastive_serial.py `
    tests/test_luka_u100_pair_hydra.py `
    tests/test_luka_u100_pair_hydra_precision.py `
    tests/test_luka_u100_pair_hydra_portability.py `
    --basetemp=tmp/contrast-hydra-combined-20261103
```

Nine Python files parse successfully; CLI `--help` passes. Git Bash syntax
checks passed for both shell files and all three runbook Bash snippets; the
PowerShell commit snippet parses. Links in all four new documents resolve.
Changed tracked documentation passes `git diff --check`; existing scientific
and pair-Hydra runtime packages have no worktree changes.

The adapter integration uses actual Weaver 0.5.3 sources with a reduced
architecture on synthetic input:
an entire thirty-epoch zero-control worker fit, a separate full-thirty-epoch
original fit, native checkpoint validation, mixed-site reporting, refusal of
changed target lineage, and preservation of all incomplete rows. A relocation
test makes the original directory unavailable and reads the same authenticated
key bank from its new directory. Corrupt keys and incomplete accounting fail.
Zero-control workers do not open the OFFLINE payload.

Scheduler tests mock Slurm but exercise real exact-DAG plans and journals:
all remaining fits have no dependencies, only the report waits; duplicate
submission and failed setup cannot release additional students. Campaign tests
bind the existing input bundle and completed evidence, reject policy drift and
require a fresh execution root. Synthetic/local tests do not qualify real-data
runtime or memory usage on Hydra.

## Remote evidence and next action

The user supplied successful diagnostic 131484 and subsequently running
pair-correction jobs 131589/131590 on Hydra. Those are evidence for the existing
environment/numerical adapter, not completed contrastive acceptance.

No new SPORC/Hydra commands, transfers or submissions were executed in this
implementation session. No real contrastive L40 miniature or Tigris validation
has been performed here. Next: commit/push the scoped additions, export the
small delta on SPORC, copy/verify it on Brown, and invoke the queue helper. The
shared setup must succeed before the detached dispatcher releases the remaining
fits. The report labels reused SPORC controls and Hydra attempts explicitly;
cross-site inference qualification does not assert identical training paths.
