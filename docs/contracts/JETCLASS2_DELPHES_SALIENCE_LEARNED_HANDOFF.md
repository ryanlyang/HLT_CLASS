# JetClass2 Delphes Salience Learned-Fusion Handoff Contract

Version: `JETCLASS2_DELPHES_SALIENCE_LEARNED_HANDOFF_*/v1`

This family implements the
[500k learned-handoff plan](../plans/JETCLASS2_DELPHES_SALIENCE_LEARNED_HANDOFF_500K_PLAN.md).
It is not interchangeable with the historical CMS Strategy-B family, the
ordinary JetClass2 logit ladders, or any bottleneck-matching campaign.

## Invariants

- The population is the exact registered TRAIN_500K membership, shared 1M
  validation membership, and sealed 1M final-test membership.
- All views derive from one authenticated selected salience foundation and its
  persistent-HLT semantics.
- The graph has exactly two fresh references, 14 transitions with DIRECT,
  ACQUIRE, and WITHDRAW fits, and one ten-fit control panel: 54 fits total.
- The child/lower branch owns the sole classifier. Context flows one way and is
  absent from alpha-zero dispatch and the extracted checkpoint.
- U-side and D-side transitions use the same transition protocol.
- The global controls are `U000 -> D000`; the morph uses exact denominator-25
  U and D coordinates, reaches D000 at pass 51, and remains there afterward.
- Every next carrier is a physically extracted ordinary single-view model and
  teaches only through compact probabilities.
- Fits use the frozen 100-pass warmup/hold/decay/floor schedule; the patience
  clock begins after pass 60, so ordinary early stopping cannot occur before
  pass 75.
- No matching identity, salience, coordinate, or source metadata is a model
  input. D000 is exact native HLT without offline access.
- Particle views, paired caches, hidden states, optimizer state, and dynamic
  morph coordinates are nondurable. There is no rolling resume.
- Final-test capability is absent from all ordinary tasks.
- Scientific quality cannot fail or prune the graph. Invalid lineage, corrupt
  bytes, nonfinite required values, forbidden data access, and stale source fail
  closed.

## Artifact families

Version-1 artifacts cover the graph, node and recipe specifications; source,
population, seed, validation-partition, storage, execution and resource locks;
training reports and selected/final/extracted checkpoints; compact probability
banks; diagnostic and stage reports; aggregate and completion reports; command
plans, submission journals and ledgers; monitors; and restart-zero recovery.

Every reusable artifact carries a content hash, exact parent identities,
source commit, population identity, and `final_test_accessed: false`. Path
existence or Slurm state alone never authorizes reuse.
