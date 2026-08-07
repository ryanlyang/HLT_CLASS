# PMARD T100 KD Weight, Temperature, and Exposure Sweep

Contract family: `v2`

This supplemental pilot study tests whether the small observed T100 transfer
gain is limited by the fixed 15% privileged-teacher coefficient or by using
the ordinary-teacher temperature for both KL terms. It does not alter the
immutable parent pilot, select a deployable finalist, or read final-test jets.

## Registered grid

Every row uses the exact parent-pilot model architecture, batch size, learning
rate, initialization seed, 300,000 train rows, and 100,000 validation rows.
The student input is HLT-only. Class-weighted CE is fixed at `0.25`; the T100
coefficient is one of `0.15, 0.25, 0.35, 0.50`; the T0 coefficient is
`0.75 - T100`; the T100 temperature is one of `1, 2, 4`; and training exposure
is `10, 20, 40` complete natural train-role passes. T0 temperature remains
`1`. This is a complete 4 x 3 x 3 grid of 36 registered rows.

Updates are defined by the existing campaign convention:
`ceil(300000 / effective_batch_size) * training_passes`. With the selected
pilot batch size 256, the registered budgets are therefore 11,720, 23,440,
and 46,880 updates. The parent lock must authenticate the original 10-pass
budget before this supplement can be submitted.

The two KL terms retain the existing `temperature**2` correction, applied
independently using their own temperatures. The training report and resume
contracts are version 5 because separate per-teacher temperatures change the
scientific loss semantics. Existing version-4 reports remain readable only as
bound parent artifacts.

## Efficient execution

One prerequisite GPU job computes frozen T0 and T100 logits once in identical
train-row order. T100 consumes the parent pilot's authenticated persistent
fitted-strict assignment cache and constructs the alpha-one repaired view only
for this single pass. It publishes one compact float32 NPZ plus a manifest;
particle inputs and repaired datasets are never published. All 36 students
then use ordinary HLT-only streaming and identity-join both cached targets in
RAM. The array is intentionally uncapped.

The authorization artifact retains the versioned repair-family identity
`SELECTIVE_FULL_PARTICLE_ENDPOINT/v1`. At execution, the shared repair module
translates that allow-listed contract name to the unversioned tensor-builder
selector at PMARD-stream entry, before native offline branch projection;
arbitrary family strings still fail closed. Target construction runs the T100
repaired pass before the ordinary T0 pass so the constrained path is validated
first.

## Lineage and access

`hlt_classification_pmard_t100_kd_sweep_spec_v2` binds the clean source,
execution worktree, parent campaign, split, feature audit, exact row selection,
assignment manifest, endpoint authorization, training lock, T0, T100, K1, and
the prior alpha-one K2 report. Every target array and training report is
content-hashed. Aggregation requires all 36 rows and reports deltas against
both K1 and the original K2 alpha-one result. Validation selection uses the
existing PMARD utility ordering; this exploratory single-seed result cannot
authorize test access or replace the registered confirmation campaign.

The durable contracts are:

- `hlt_classification_pmard_t100_kd_sweep_spec_v2`;
- `hlt_classification_pmard_t100_kd_targets_v1`;
- `hlt_classification_pmard_training_report_v5`;
- `hlt_classification_pmard_resume_checkpoint_v5`;
- `hlt_classification_pmard_t100_kd_sweep_report_v2`;
- `hlt_classification_pmard_t100_kd_sweep_ledger_v2`.
