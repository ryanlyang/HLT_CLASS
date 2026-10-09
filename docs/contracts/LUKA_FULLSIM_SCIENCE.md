# Luka FullSim science contracts v1

Authority: [stage-3 plan](../plans/LUKA_FULLSIM_SCIENCE_PLAN.md).

All ordinary artifacts have canonical content hashes, schema version 1 and
`final_test_accessed=false`. No test loader, selection or evaluation command is
exposed. No prepared-v1 or preflight-v2 contract is changed.

- `LUKA_FULLSIM_SCIENCE_SOURCE_REUSE/v1`: separately authenticates the accepted
  preflight and new science source snapshots. Every pre-existing tracked `src/`
  file must be byte-identical; exactly `science_{admission,campaign,cache,worker,
  submission}.py` may be added under `luka_fullsim`. CLI/workers/docs/tests are
  bound by the entire new clean source snapshot and must be committed/pushed.
- `LUKA_FULLSIM_SCIENCE_ADMISSION/v1`: binds prepared-v1, exact reviewed
  preflight-v2, its full output byte references and historical preparation reuse,
  source reuse, and all source/artifact locators. Probe weights and banks are
  hashed as evidence only; the scientific import list is empty.
- `LUKA_FULLSIM_SCIENCE_SPEC/v1`: binds admission/source, recorded successful
  preflight accounting, exact 12-fit/7-reducer/21-job graph, seeds, views, recipe,
  full-validation policy, and measured debug resource requests. Raw inputs and
  old artifacts must remain accessible read-only at their authenticated paths.
- `LUKA_FULLSIM_SCIENCE_PLAN/v1`: complete ordered `sbatch` argv and exact
  dependency placeholders; resolved numerical IDs are journalled on submission.
  Scheduler feasibility is checked with `--test-only` for each resource shape.
- `LUKA_FULLSIM_SCIENCE_SUBMISSION_CLAIM/v1`: exclusive live intent; any partial
  or ambiguous failure keeps the claim and journal and requires explicit repair.
  Successful submission is idempotent. The unchanged reusable exact-DAG helper
  publishes its versioned dry/live ledgers and events (not mislabelled as Luka).
- `LUKA_FULLSIM_SCIENCE_TASK/v1`: task/spec/preflight/source/prepared/dependency
  bindings, output paths/bytes/SHA256, result and scheduler ID. Receipts publish
  last. A partial task directory without its valid receipt is not reusable.
- `LUKA_FULLSIM_SCIENCE_RESULTS/v1`: every registered row, recoveries to the
  same-run pure OFFLINE and M0HLT controls, deployability distinctions and
  caveats. Poor metrics do not prevent publication or dependent experiments.

Training reports and probability banks retain their unchanged shared kernel
contracts. Each report must be a genuine 60--100-pass scientific fit, not an
acceptance run, and name the precise node, recipe and prepared population.
Selected weights are written and read back. Reducers load only the completed
scientific parent's selected checkpoint, publish T=2 TRAIN probabilities and
read them back with exact ordered identities. No bank is imported from another
campaign. A probability manifest and every shard must be receipt-bound.

The generalized two-view loader uses the unchanged stage-2 physical conversion,
assignment join, view and input functions. Reproducing **all** prepared input
fingerprints is mandatory, not just a sample check. D000/D000 aliases one cache;
single D000 does not decode offline branches. U000 bounds cardinality, and the
accepted three-view cache reservation bounds each science worker's two views.

GPU workers recheck 6 CPUs / 256000 MiB / one A100, exact measured GPU identity,
installed environment and cuDNN setting; they set cuBLAS workspace before Python.
The source and parent evidence reauthenticate before completion. New wrapper
tests do not claim another genuine GPU run: accepted job 21835630 supplies
unchanged-kernel technical evidence, while stage-3 science remains unrun until
the user commits, reviews and authorizes the full plan.
