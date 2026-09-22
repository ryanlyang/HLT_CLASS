# JetClass2 dz-fix native-offline CE control

This version-1 standalone control adds one scientific fit to the registered
JetClass2 dz-fix `TRAIN_500K` comparison. It is independent of the running
three-spine scheduler DAG and never mutates that campaign.

The `OFFLINE` view is the native offline constituent collection read directly
from the frozen dataset. It is not persistent-HLT `U000`: it has native
offline cardinality, is not projected onto an HLT skeleton, retains no extra
HLT particles, and never reads matching assignments. Train and validation
identities are exactly those bound by the source campaign's frozen 500k/1M
membership hashes. Final-test particles remain sealed.

The fit uses the same 17-input/11-output Particle Transformer, full CE-only
reference loss, batch size, optimizer, 100-pass schedule, early stopping,
checkpoint selection, and single-A100 tier3 execution profile as the source
campaign. Initialization and sampler seeds are deliberately paired with its
`U000` CE anchor, leaving the particle view as the controlled difference.

The DAG is `authenticate -> train_OFFLINE -> control_complete`. Only the
training task requests a GPU. The authentication task verifies the complete
raw snapshot and all source lineage. No matching job or source-campaign job is
a scheduler dependency because the native offline view does not consume
matching artifacts; the immutable source campaign specification is read-only
provenance.

Implementation:

- `jetclass2_delphes/native_offline.py`: assignment-free RAM cache;
- `jetclass2_delphes/pure_offline_control.py`: specification, execution,
  exact DAG submission, monitoring, and results;
- `scripts/jetclass2_delphes_pure_offline_control.py`: thin CLI;
- `sbatch/run_jetclass2_delphes_pure_offline_control.sh`: pinned worker.
