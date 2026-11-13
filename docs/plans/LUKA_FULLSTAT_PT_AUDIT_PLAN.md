# FullStat TRAIN class and pT occupancy audit

Authorized 2026-11-12: implement the immediate metadata-audit workflow before
choosing a class-balanced, per-class pT-flattened training recipe. The user
explicitly excludes mass, eta and other balancing variables.

This is an additive diagnostic, not a new fit or sampler. The historical
[Oscar20 plan](LUKA_FULLSTAT_OSCAR20_PLAN.md) retains its natural-proportion
membership and once-per-row training semantics. The user's new direction
authorizes investigating a different future sampler; this audit changes no
historical spec, split, teacher, model, loss, source, job or final-test policy.
Final binning, sampling and matched training campaigns require their own plan.

## Inputs and authority

Use the completed six-teacher campaign's authenticated original FullStat
anchor, original spec, cache-manifest JSON, foundation JSON and scalar
membership NPZs. The default parent is
`luka_fullstat_six_teacher_memory_2fb09f38_r1/campaign_spec.json`.
The original FullStat SPEC/RECOVERY_SPEC can also be supplied directly.
No glob/latest-time selection or directory-name inference establishes identity.

The variable is the saved **offline reconstructed `jet_pt` in GeV**, common
to each paired jet. It is neither constituent pT nor the generated QCD source
bin. This definition is explicitly reported and does not certify that it is
the requested HLT-pT definition for any future campaign.

Verify JSON content hashes, anchor file references, original population,
prepared-cache/foundation parent hashes, membership byte hashes, source
uniqueness/order, role counts, TRAIN label mapping and original TRAIN stratum
quotas. The production entry point requires the existing 3M/500k/1.5M/105325
counts. One scalar shard is decoded at a time; only `roles`, `labels`,
`jet_label` and `jet_pt` columns are needed. Shared archives contain other
roles' scalar rows, but only TRAIN labels/pT contribute to diagnostics.
No ROOT particle data, models, predictions, test metrics or cache payloads
are loaded. This is not a fresh split replay or full source-lineage audit.

## Diagnostics

- Per-class count, fraction, min/max and p01/p05/p50/p95/p99 pT.
- Default diagnostic widths 50, 100 and 200 GeV. Each grid covers the complete
  TRAIN extent, rounded outward to multiples of its width with origin zero.
  These are candidate widths, not approved CMS binning or training settings.
- Optional explicit common range, with underflow/overflow counts by class.
  Every requested width must divide that range exactly; no short final bin.
- NumPy histogram boundaries: left inclusive, right exclusive, except the
  final upper endpoint is included. Zero-count cells remain present.
- Empty cells and nonempty cells below an advisory minimum (default100).
  The warning threshold is not a scientific acceptance cut.
- Contiguous bin spans containing every class, both with any population and
  with at least the advisory minimum; gaps remain explicit. No span is
  automatically selected as the training range.
- Conditional exposure projection for 3M independent replacement draws per
  epoch,20 epochs, equal class and equal bin probabilities. Empty cells make
  that candidate infeasible; no renormalization over supported cells occurs.
- For a feasible grid with C classes, B bins and n jets in a cell, each jet
  has probability `p=1/(C*B*n)` per draw. Expected exposures are `D*p`;
  expected distinct jets are summed as `n*(1-(1-p)**D)`. Stable log1p/expm1
  arithmetic avoids cancellation. Repeat fraction is `1-E[distinct]/D`.
  Expected distinct jets across20 epochs use `20*D`, not20 times the
  one-epoch distinct count. These projections are not measurements of a
  deterministic future sampler or evidence of statistical independence.

## Outputs and workflow

Publish a fresh directory with `summary.txt`, `cells.csv` and completion-last
`report.json`. The report binds metadata/membership files, content hashes,
audit-source hashes, package versions, budgets, boundaries and diagnostic
parameters. A portable `show` command checks the copied report and both export
hashes without accessing original Oscar paths. No row-level export is needed.

Runtime is Python3.10+ and NumPy only, CPU, one thread. No GPU, Slurm submission,
environment modification or training preflight. The shell wrapper uses the
existing Oscar environment, or an explicit `AUDIT_PYTHON` interpreter override.
The runbook uses a new source worktree and a separate report directory.

## Verification and next step

Tests cover hand-computed bin edges/exposures, empty/sparse cells, tails,
invalid inputs, stale/corrupt/cross-population metadata, prohibited array
access, immutable inputs, fresh output, portable report replay, real CLI and
NumPy-only imports. Frozen lightweight adapters are checked against the
original contracts. No model parity/GPU miniature is relevant to this audit.
Real Oscar data remain the next step; synthetic tests do not establish their
histograms or a suitable pT range.
