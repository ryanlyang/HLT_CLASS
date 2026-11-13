# FullStat TRAIN class/pT audit validation

Implemented and checked locally on2026-11-12. This is metadata analysis, not
model training or sampler implementation. No actual Oscar files were read,
no jobs were submitted, and no production histogram/bin choice is claimed.

## Evidence

- Baseline: two existing FullStat foundation tests passed before edits.
- Final focused suite: **37 passed,15 deselected**,8.40seconds. This includes
  35 new audit cases and the two unchanged foundation checks.
- Real existing foundation producer exercised on synthetic ROOT files, with
  the audit reading back the exact selected TRAIN rows and matching donor
  labels. This is synthetic compatibility evidence, not a real-data witness.
- Exact hand-calculated histogram endpoints, class-cell counts, tails,
  expected unique rows and exposure factors checked. Empty cells stay in the
  grid and suppress the whole-grid sampling projection.
- Failures: invalid budgets/edges/pT, corrupt hashes/payloads, alien anchors,
  duplicate sources, path escapes and wrong quotas fail before completion.
- Non-TRAIN pT deliberately poisoned in fixtures; it is excluded from the
  summaries. A guarded NPZ reader admits only scalar roles/labels/raw labels/
  pT columns. Input bytes remain unchanged after end-to-end publication.
- NumPy-only import checked in an isolated subprocess that forbids Torch,
  Weaver, Uproot, Awkward, SciPy and scikit-learn imports.
- Actual Python CLI run/show and Bash wrapper/interpreter override passed.
  Report replay verifies the copied summary and CSV hashes; corruption fails.
- Python AST, runbook Bash syntax, relative documentation links and scoped
  whitespace checks passed.
- Frozen lightweight label/policy/hash/file-reference adapters match their
  original implementations. Publication is intentionally fresh-output only.
- Synthetic3M-row in-memory analysis of three candidate grids took0.362seconds
  locally. This excludes archive reads/hashing and is not an Oscar walltime
  prediction. No physics distribution or resource acceptance is inferred.

Command:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONPATH=(Join-Path (Get-Location) 'src')
& 'C:/Users/22rya/miniconda3/envs/tagging-hlt/python.exe' -B -m pytest -q -p no:cacheprovider tests/test_luka_fullstat_pt_audit.py tests/test_luka_fullstat.py -k 'pt_audit or foundation'
```

The default system Python lacked scientific dependencies, so validation used
the existing `tagging-hlt` environment. No packages or environments were changed.
An initial test caught eager heavy imports in the reused package initializers;
the final implementation uses separately versioned lightweight adapters instead.

## Scope and remaining step

New files: audit module and metadata adapter, Python CLI and Bash wrapper,
focused tests, active audit plan, v1 contract, runbook and this evidence record.
HANDOFF and LEGACY_SOURCE_MAP receive additive evidence only; pre-existing user
edits remain intact. Original training/runtime/contracts are untouched.
Donor baseline: `1fc128dca7dc7ef547d242a1e2941e9a74e25ac8`; exact source
attributions are in the runbook and donor map. Contract:
`LUKA_FULLSTAT_PT_AUDIT/v1`.

No installed-Weaver parity or Tigris GPU miniature is applicable to a
NumPy-only scalar audit. The remaining task is to push the audit, run the
[Oscar command](LUKA_FULLSTAT_PT_AUDIT_RUNBOOK.md), and inspect its actual
TRAIN occupancy before deciding on a sampling range or launching fresh fits.
