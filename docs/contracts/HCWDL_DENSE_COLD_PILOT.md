# HCWDL Dense Cold 300k Supplemental Campaign

## Status and scope

This contract registers one validation-only supplemental campaign over the
completed unweighted 300,000-train/100,000-validation HCWDL pilot. It tests
whether small, cold-started, single-teacher privilege reductions preserve more
offline-derived knowledge than the original 25-point cold/warm ladder.

It does not replace or relabel the original `HCWDL_GRAPH/v1`. It introduces
`HCWDL_DENSE_COLD_GRAPH/v1`, imports the original pilot as immutable evidence,
and never accesses the pilot final-test role.

## Imported parent artifacts

The creation command authenticates and binds:

- the exact executable unweighted `pilot` campaign specification;
- its 300k/100k/100k role counts and split hash;
- its deterministic train/validation row selection;
- its `HCWDL_RECIPE/v4` primary recipe and exact unweighted class vector;
- its complete train and validation high-coverage assignment manifests;
- its assignment and Shell Exact qualification locks;
- the selected M0, label-only D100, and TOFF reports and checkpoint hashes.

Assignments and reconstructed datasets are not copied. Each dense worker reads
the existing compact assignment store, builds its one student view in RAM once,
computes its one teacher-logit target bank once, and replays both for training.

## Frozen graph

```text
TOFF
  -> D100offkd
  -> D90c
  -> D80c
  -> D70c
  -> D60c
  -> D50c
  -> D40c
  -> D30c
  -> D20c
  -> D10c
  -> D0c
  -> M1c
```

`D100offkd` is deliberately distinct from the imported label-only `D100`.
Every new node uses fresh deterministic initialization. There are no warm
nodes and no M2--M6 ascent nodes.

For the top-end ablation, D100offkd aliases the imported D100 node's
initialization seed, sampler trajectory, dropout/training RNG domain, and
optimization budget. Their intended difference is the registered TOFF KD
term. All later dense nodes use their own declared node seed identities.

Every D child uses unweighted `0.25 * CE + 0.75 * KD` with temperature 2.
`D100offkd` receives TOFF logits evaluated on native offline particles while
the student consumes the D100 Shell Exact view. Each later D child receives
the immediately preceding teacher evaluated on that teacher's own view.
`D0c` consumes exact HLT and learns from D10c targets. M1c is a fresh HLT-only
student trained from D0c with `0.25 * CE + 0.75 * KD` at temperature 1.

All nodes train for 60 complete natural-population passes, validate after every
pass, use the existing AdamW/warmup-cosine recipe, and select the checkpoint by
highest validation macro OVR AUC with CE, log-rejection, and earliest-update
tie breakers.

## Repair-coordinate correction

Every D view uses Shell Exact v1 with the same identity-bound discrete repair
seed. Consequently, discrete identity, PID, charge, validity, and quality
switches are nested across alpha: reducing alpha can switch an offline field
back to HLT but cannot independently redraw the particle at each rung.

The original coarse runner derived its repair seed from the domain name. Its
individual views remain valid Shell Exact views, but its discrete choices are
not a strictly nested cross-alpha trajectory. The completed coarse campaign is
therefore retained as an empirical 25-point control, not relabeled as the new
nested trajectory.

## Outputs and interpretation

The immutable aggregate includes imported M0/D100/TOFF controls, all twelve
new node metrics, recovery of the M0-to-D100offkd and M0-to-TOFF validation-AUC
gaps, report/checkpoint hashes, and an explicit `final_test_accessed=false`.

Poor finite performance completes the graph. Missing or corrupt parents,
nonfinite required metrics, source drift, checkpoint/hash mismatch, assignment
lineage mismatch, or a failed Slurm parent stops descendants through `afterok`.

This is a single-seed screening study. It can motivate confirmation seeds but
cannot by itself establish a final performance claim.
