# JC2_LITERATURE_PRODUCTION/v1

Authority: `docs/plans/JETCLASS2_LITERATURE_NOISE_V3_PRODUCTION_PLAN.md`.

Artifacts use `JC2_LITERATURE_PRODUCTION_<KIND>/v1`, schema version 1,
canonical `content_hash`, parent hashes and truthful test flags. This namespace
is not interchangeable with literature pilot or learned CMS-proxy artifacts.
The reused immutable `CMS2JC2_PROXY_POPULATION/v1` is explicitly a metadata
selection contract only, not the generator identity.

The study binds exact source, inventory, TRAIN_1M profile, pilot profile,
population, shards, storage attestation and frozen bundle. The bundle binds
the authenticated v3 pilot spec/receipt/report, full calibration, recipe,
and execution environment. Production must retain the pilot's scientific
implementation hashes, not merely the same recipe name. Test-build locks
authorize materialization only.

Each shard is a contiguous slice of an authenticated file entry mask, with
role, original relative path, start/stop in that mask and ordered jet-identity
digest. Shard receipts bind study, shard registration and immutable attempt;
list exact physical bank sizes/hashes and ordered identity/physical digests;
and report jet/particle counts. Only complete receipts can be reused.

Physical NPZ banks contain exactly `offsets` (little-endian int64, N+1),
`jet_identity` (uint8, N x 32), `p4` (float64, M x 4; px/py/pz/E GeV),
`charge` and `category` (int8, M), `tracking` (float64, M x 4;
d0/dz/d0err/dzerr in mm), and `valid` (bool, M x 4).
Categories: 0 charged hadron, 1 neutral hadron, 2 photon, 3 electron,
4 muon, 5 unknown. Invalid tracking is zero-filled with explicit masks.
No labels, ancestry, latent noise or original particle keys are exported.

Independent role manifests are allowed only for train/validation, validate
all that role's bytes and identities, and never open final-test banks. The
complete dataset manifest requires exact counts 1M/250k/1M and every shard.
The test-materialized flag is true on test shard receipts and full manifest;
`final_test_evaluated` is always false. Metadata-only registrations/plans do
not count as test particle access. A downstream final-test reader requires
its own scientific finalist/evaluation authorization; none is supplied here.

Default CPU resources are frozen in the study. Execution changes cannot
silently mutate a reviewed plan. Failed science metrics are results, while
corrupt inputs, wrong lineage, unauthorized access and nonfinite required
physical quantities fail closed. The synthetic output is not detector HLT.

## Consumer interface

Use the production source revision pinned in `study_spec.json`. A completed
`releases/train.json` or `releases/validation.json` is required. Those markers
can be created with the CLI's `release --spec ... --role train|validation`
once that role is complete; the scheduled finalizer also creates both.

```python
from hlt_classification.literature_proxy_production.output import read_role

for jet_identity, proxy in read_role(dataset_root, "train"):
    # proxy: p4, charge, category, tracking, valid; no construction features
    ...
```

For offline pairing, use `study["population"]["files"]` and the original entry
masks with the pinned inventory hash. `population.entries_for(population,
shard)` gives original ROOT entries, not positions in compressed banks;
`population.ids(inventory_hash, file_row, entries)` gives their canonical
identities. `population.iterate(data_root, population, shard)` yields offline
physical particles in that same order for training/validation shards. Join
and assert identities. Labels, if needed by the downstream classifier, come
from its separately authorized native source reader and the same entries.
This generator never reads or manufactures labels.

The existing `JETCLASS2_CMS_PROXY_DATASET_CONSUMER_HANDOFF.md` describes the
older **learned CMS proxy**, not this dataset. Never infer the generator from
similar bank layouts or accidentally point a literature experiment at the
older dataset root. Check the study contract, recipe bundle and role release.
