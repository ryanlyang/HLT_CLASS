# HCWDL Five-Point Dense Cold 300k Supplemental Campaign

## Status and scope

This contract registers a second validation-only supplement over the completed
unweighted 300,000-train/100,000-validation HCWDL pilot. It asks whether a
strictly nested sequence of five-percentage-point privilege reductions
preserves more offline-derived knowledge than the separately registered
ten-point dense ladder.

The five-point campaign has its own `HCWDL_DENSE5_*` contracts, graph hash,
campaign root, Slurm job prefix, reports, and authorization phrases. It cannot
reuse or overwrite outputs from the ten-point campaign. It imports the same
authenticated parent assignments and controls read-only and never accesses
the final-test role.

## Frozen graph

```text
TOFF
  -> D100offkd
  -> D95c -> D90c -> D85c -> ... -> D10c -> D5c -> D0c
  -> M1c
```

There are 22 new training nodes: 21 D-domain models from D100 through D0,
inclusive, and one born-again M1. TOFF is an imported teacher and is not
retrained. The Slurm DAG therefore contains 22 sequential GPU jobs and one CPU
aggregate job.

Every node is cold-started. Every D node uses unweighted
`0.25 * CE + 0.75 * sole-teacher KD` at temperature 2. D100offkd consumes the
Shell Exact D100 view and learns from imported native-offline TOFF logits. Each
later D node learns from the immediately preceding D model evaluated on that
teacher's own view. D0 consumes exact HLT and learns from D5. M1 is a fresh
HLT-only model trained from D0 at temperature 1 with the same CE/KD weights.

Every node trains for 60 natural-population passes, validates after every
pass, and selects the checkpoint by highest macro OVR AUC, followed by CE,
log-rejection, and earliest-update tie breakers. Finite poor performance does
not stop the graph.

## Shared repair coordinate and imported evidence

The campaign uses the same identity-bound Shell Exact repair coordinate at
every alpha. Discrete PID, charge, identity, validity, and quality switches
are consequently nested as alpha descends. Alpha values shared with the
ten-point campaign construct the same view under the same parent selection and
replicate seed, although the learned teachers differ because the graphs differ.

The creator authenticates and reuses the exact parent pilot's split, row
selection, unweighted recipe, train/validation assignment manifests,
assignment lock, Shell Exact endpoint lock, and M0/D100/TOFF reports and
checkpoints. It does not rematch, copy assignments, or persist repaired
datasets. Each worker constructs its student view once in RAM and its teacher
logits once, then replays them across all 60 passes.

The aggregate is validation-only, records recovery of the M0-to-D100offkd and
M0-to-TOFF AUC gaps for every rung, and declares
`final_test_accessed=false`. This remains a single-seed screening study.
