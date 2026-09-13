# JetClass2 Delphes Full-Cardinality Salience Persistent-HLT Contracts

This contract family implements the
[500k plan](../plans/JETCLASS2_DELPHES_FULLCARD_SALIENCE_PERSISTENT_500K_PLAN.md).
All artifacts use `JETCLASS2_DELPHES_SALIENCE_* /v1` identities and content
hashes. They are not interchangeable with the bottleneck `JETCLASS2_DELPHES_*`
foundation or campaign artifacts.

## Invariants

- Population is the authenticated `TRAIN_500K` profile with fixed 1M
  validation and sealed 1M final test.
- Matcher specs are exact `HCWDL_FULLCARD_SALIENCE_MATCHER_SPEC/v1` objects.
- Assignments are HLT-oriented, one-to-one, and contain exactly the smaller
  endpoint cardinality.
- Persistent-HLT U/D semantics are those frozen in the plan.
- Dense matching matrices and particle views are never durable.
- Candidate selection is based only on the fixed validation firewall; final
  test is inaccessible.
- Production has DIRECT, COARSE, and DENSE only.
- Scientific quality cannot fail a task; invalid data, nonfinite quantities,
  stale source, corrupt bytes, or lineage mismatch fail closed.

## Artifact families

The implementation publishes version-1 artifacts for matcher foundations,
assignment shards and locks, screen specification/split/fit/report/selection,
runtime acceptance, production specification and plan, task reports,
aggregate/completion, monitoring, and recovery. Every consumer validates the
complete content hash and relevant parent identities rather than accepting a
path or job state as evidence.

The selected matcher foundation may be reused by a later ladder only when the
inventory, split membership, particle decoding, native-index semantics,
matcher spec, assignment producer bytes, and foundation lock all validate
unchanged. A new split or matcher requires a new foundation.

## Implementation map

- `jetclass2_delphes/salience_views.py`: JetClass2 particle adapter and the
  persistent-HLT U/D construction.
- `jetclass2_delphes/salience_foundation.py`: compact assignment shards,
  candidate-independent pT-weighted diagnostics, raw-row recomputation audit,
  and foundation lock.
- `jetclass2_delphes/salience_cache.py`: bounded process preparation of
  RAM-only ragged views; it has no disk-spill path.
- `jetclass2_delphes/salience_screen.py`: fixed U100 matched-seed screen,
  validation firewall, actual SPORC preflight, and selection lock.
- `jetclass2_delphes/salience_campaign.py` and `salience_production.py`: the
  16-fit/12-reducer three-spine graph, execution, result printing, exact DAG
  submission, monitoring, and restart-zero recovery.
- `scripts/jetclass2_delphes_salience_{foundation,screen,production}.py` and
  their same-named `sbatch/run_*.sh` workers are the only supported CLIs.

The three stages use separate fresh output roots. Candidate preparation never
submits a fit. Screen submission does not launch production. Production cannot
be created until the screen completion and selection locks authenticate.
