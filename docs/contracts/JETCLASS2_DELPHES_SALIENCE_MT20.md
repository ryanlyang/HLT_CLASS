# JetClass2 Delphes salience MT20 reusable contract

This contract covers the isolated `TRAIN_500K` three-spine multi-grandparent
campaign defined by the active implementation plan.

## Scientific identity

The identity binds the frozen JetClass2 inventory and split membership, selected
salience foundation and assignment lock, reduced 17-input/11-output model,
persistent-HLT U/D coordinate definitions, three ordered spine paths, paired
coordinate seeds, C20/P80 T=2 objective, exact rational same-spine ancestor
weights, optimization schedule, and absence of warm starts or ensembles.

All earlier same-spine teachers are required. Their order is nearest first. A
single teacher receives loss weight 4/5. For two or more teachers, the immediate
teacher receives 1/2 and historical teachers divide 3/10 geometrically with
ratio 1/2. Cross-spine banks are invalid.

## Artifact rules

Every JSON artifact has a versioned contract and canonical content hash. Every
model checkpoint and probability shard is inventoried by checksum. Probability
banks bind foundation, producer training report, teacher node, role, temperature,
row count and exact ordered row identities. Paths alone never authorize reuse.

Only train-role T=2 teacher banks are durable. Multi-teacher mixtures are never
durable: components are loaded in registered order, exact-identity joined,
accumulated in float64, normalized, converted to contiguous float32, and released
after training. Particle views are RAM-only. Rolling optimizer/checkpoint state
is forbidden; only the selected restored model checkpoint is retained.

## Execution and access

The ordinary roles are train and validation. Final-test access is absent.
The source lock directly binds and authenticates the completed
`SALIENCE_PT_LINEAR` winner foundation, foundation lock, assignment producer,
matcher, inventory, input, view and split identities, plus the immutable screen
selection/completion/profile evidence. It does not authenticate losing matcher
foundations and must never rebuild assignments or repeat matcher selection.

Scientific tasks require the screened one-A100 SPORC environment and exact
resources. The operational routing policy defaults each task to SPORC `debug`
and permits explicit task-level `tier3` overrides. Both routes retain the same
account, QoS, A100 family, Conda environment, CPU/RAM request and single-task
shape; every request is below debug's 24-hour ceiling. Routing is recorded in
the immutable command plan and does not alter scientific identity. A submitted
pending job is never silently repartitioned: an exact-ID cancellation and
restart-zero recovery is required.

Live submission is staged: authenticate/storage/preflight gates first,
then a separately dry-run and authorized 30-task science DAG. A full live stage
is forbidden. Exact-ID recovery requires all old jobs terminal and restarts
incomplete tasks from zero.

The gate is specific to this intervention. It must execute the selected salience
U000 view, production model, two-teacher mixer, C20/P80 loss, T=2 and finite
backward pass. Evidence for a different matcher or loss cannot substitute.

Poor metrics are valid results and never prevent completion. Invalid lineage,
identity joins, nonfinite quantities, stale source, forbidden files, insufficient
storage, or any final-test capability fail closed.
