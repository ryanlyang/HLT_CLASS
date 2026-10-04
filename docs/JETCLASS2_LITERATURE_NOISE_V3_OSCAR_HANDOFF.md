# Frozen literature NOISE_V3 on Oscar: dataset and paired-reader handoff

## Read this first

This document describes the **new literature-inspired randomized proxy**, not
the CMS-fitted proxy in `JETCLASS2_CMS_PROXY_DATASET_CONSUMER_HANDOFF.md`, and not
the native Delphes HLT trees. Do not substitute any of those datasets.

The response recipe is frozen **NOISE_V3**. It was not fitted to CMS FullSim.
The preceding count38 pilot did calibrate drop/merge rates on training jets to
a target mean of 38; saying that no parameter was ever calibrated would be
incorrect. V3 increases smearing while preserving that pilot's topology and
rates. Do not recalibrate, regenerate on x86, adjust counts, or change seeds.
This is a controlled synthetic benchmark, not a certified CMS detector model.

User-reported Oscar checks on 2026-10-03 establish:

- complete copy: **1,000,000 train / 250,000 validation / 1,000,000 final test**;
- **2,341** proxy blocks and **186** required offline ROOT files passed their
  frozen checksums; reported directory usage was approximately 5.7 GiB + 17 GiB;
- the inventory, three direct pilot evidence files, and TRAIN_1M profile passed
  their exact SHA256 checks;
- test files were transported/hashed as opaque bytes, not decoded or evaluated.

These are user-supplied remote results, not independent local measurements.
The new relocated Python loader still requires the small Oscar smoke below
after its source is pushed and installed. No Oscar training campaign has been
created or authorized by this handoff.

## Exact storage locations

Persistent personal-home copy (the shared `/oscar/data/nopi` destination failed
with permissions and was **not** used):

```text
BASE=/oscar/home/rlyang/datasets/literature_noise_v3_2250k_ebd5bc1a_r1

Proxy dataset:
${BASE}/jetclass2_literature_noise_v3_2250k_ebd5bc1a_r1

Original offline ROOT source:
${BASE}/jetclass2_10M_20260918_dzfix_partial_v1/jetclass2

Original split profiles:
${BASE}/jetclass2_10M_20260918_dzfix_partial_v1/jetclass2_delphes_scaling_splits_20260918_v1/profiles

Additional provenance:
${BASE}/oscar_provenance_v1
```

Source copies remain at RIT and were not deleted:

```text
/home/ryreu/atlas/datasets/jetclass2_literature_noise_v3_2250k_ebd5bc1a_r1
/home/ryreu/atlas/datasets/jetclass2_10M_20260918_dzfix_partial_v1/jetclass2
```

Use `checkquota` on Oscar for current user allowances, not `df`'s shared free
space. At transfer time home usage was 4.44 GB against a 100 GB soft quota,
before this roughly 23 GB copy. That figure is historical, not current quota.

## Immutable identities and provenance

Complete `dataset_manifest.json` **canonical content hash**:

```text
26c902d04988b3f32be98a8d2ccbed6d48dd8c3f49c71ddc2c660da0f2ec5deb
```

This is not the raw file SHA256 from `sha256sum dataset_manifest.json`.
The loader checks canonical content and all relevant parent hashes. Producer
commit is recorded in `study_spec.json`; the original production root is named
`ebd5bc1a`. Never infer the complete commit solely from that short directory name.

Copied direct references (paths relative to `oscar_provenance_v1`):

| Reference | File | Raw file SHA256 |
| --- | --- | --- |
| Inventory | `jc2_dzfix_partial_inventory_0b3ed523_r1/inventory.json` | `feff98e604e55cdd9bdede6c933a82a5f23b7e8db4fdc2f375e50fbb3ddba0a6` |
| Pilot spec | `jc2_literature_noise_d505e807_r1/study_spec.json` | `8d721ceea726fb2611c3dc8057dfc500f978998ecb7ca3c5a28307beec93d553` |
| Pilot receipt | `jc2_literature_noise_d505e807_r1/receipt.json` | `ca7f55d40b577f4941e16157e49cda3b7cd08a33d16cc37557970a3236745e5e` |
| Pilot report | `jc2_literature_noise_d505e807_r1/report.json` | `d631a7aa5d330a4d12a3bbb17def74347f11df28d802a6a65842cad315e640ad` |

`profile` and `donor_profile` both reference the copied `TRAIN_1M.json`, raw hash
`ad01a6796b1860d2ad311411f5e9e1bdd44ac1d3582366d0d56b8c209c07e103`.
The production population contains the actual 250k validation subset; do not
replace it with the entire profile's outer validation reservoir.

Original JSON still contains RIT absolute paths. **Do not search/replace them**,
recompute their hashes, symlink over other datasets, or modify an old reader's
global state. The new adapter resolves declared paths externally and checks
the original bytes. The copied pilot files are direct provenance, not a full
archive of every pilot block, plot or transitive parent campaign.

## Install source, then run a read-only smoke

Code: `src/hlt_classification/literature_proxy_consumer.py` and
`scripts/jetclass2_literature_proxy_consumer.py`. Use a clean checkout of the
exact pushed reader commit; do not change a worktree used by running jobs.
The source must include this module before the following commands work.
No GPU, Slurm submission, PyTorch or Weaver is needed for this reader.

In a shell on Oscar, set `PROJECT_DIR` to that checkout. The previously recorded
Oscar scientific Python is shown below; check it still exists rather than
installing into or altering another campaign's environment:

```bash
export PROJECT_DIR=/oscar/home/rlyang/hlt_classification
READER_ENV=/oscar/scratch/rlyang/hlt_classification/environments/envs/atlas_kd_oscar
test -x "${READER_ENV}/bin/python"
test -f "${PROJECT_DIR}/scripts/jetclass2_literature_proxy_consumer.py"
export PYTHONPATH="${PROJECT_DIR}/src"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="${READER_ENV}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

"${READER_ENV}/bin/python" -s \
  "${PROJECT_DIR}/scripts/jetclass2_literature_proxy_consumer.py" inspect

"${READER_ENV}/bin/python" -s \
  "${PROJECT_DIR}/scripts/jetclass2_literature_proxy_consumer.py" smoke \
  --role train --jets 8 --labels

"${READER_ENV}/bin/python" -s \
  "${PROJECT_DIR}/scripts/jetclass2_literature_proxy_consumer.py" smoke \
  --role validation --jets 8 --labels
```

Defaults are the exact BASE and manifest hash above. `--root` changes only the
local copy's container location, not the dataset identity. `inspect` opens JSON
only and truthfully reports `physical_banks_verified: false`; it is not a new
full transfer audit. `smoke` checks only banks/source files needed for the
requested ordinary rows, then reports `PAIRED READER SMOKE PASS`. Save that
output as genuine Oscar reader acceptance. Do not confuse it with acceptance
of a new classifier, training schedule, GPU environment or complete dataset.

## Load paired endpoints correctly

```python
from contextlib import closing
from itertools import islice
from hlt_classification.literature_proxy_consumer import RelocatedDataset

dataset = RelocatedDataset.from_oscar_copy()
with closing(dataset.iter_pairs("train", labels=True)) as rows:
    for jet in islice(rows, 8):
        print(jet.identity, jet.source_file, jet.entry, jet.label)
        offline = jet.offline
        hlt = jet.proxy
        # Physical fields: p4, charge, category, tracking, valid.
        # No offline quantities or identity metadata enter an HLT-only model.
```

For a proxy-only inference input stream on an ordinary role:

```python
for identity, hlt in dataset.iter_proxy("validation"):
    # This path never opens offline ROOT or reads labels.
    ...
```

Iterators are file/shard ordered, **not training-shuffled**. The same frozen
order is deterministic, but use the campaign's registered sampler for fitting.
`shard_ids=[...]` can divide an ordinary role among workers without changing
identities; disjoint allocation and aggregate coverage remain the scheduler's
responsibility. A subset must stay entirely within its requested role.
For bounded reads, close the iterator as above so the ROOT stream closes and
its post-read checksum runs even when stopping early.

The exact join is:

```text
original inventory hash + relative ROOT path + tree key/cycle + raw entry
                          = stored proxy jet_identity
```

The production file masks and shard start/stop reconstruct the **raw entries**.
A row's position in a compressed bank is not its ROOT entry. The loader checks
all identities in every loaded proxy block before yielding it. Do not zip two
independently sorted datasets, pair by label, or use particle index as truth
correspondence. Drops/merges make offline/proxy particle counts/order different.
No constituent ancestry was exported; a ladder's particle matcher is a separate
declared algorithm, not something to derive from the proxy's array position.

## Physical schema, labels and feature boundaries

| Field | Meaning |
| --- | --- |
| `p4` | float64 `[particles,4]`: px, py, pz, energy in GeV |
| `charge` | int8 per particle |
| `category` | int8: 0 charged hadron, 1 neutral hadron, 2 photon, 3 electron, 4 muon, 5 unknown |
| `tracking` | float64 `[particles,4]`: d0, dz, d0err, dzerr, all in native Delphes **mm** |
| `valid` | bool `[particles,4]`: explicit applicability/validity, matching tracking columns |

Do not divide tracking by 10: that conversion belonged to the separate
CMS-calibrated response comparison. Invalid measurements are zero-filled with
false masks; a zero value and a missing uncertainty are different situations.
Use valid nonzero errors when forming significance. Derive axes, angles,
relative kinematics and feature normalizations from the supplied view only.

`labels=False` is the default. With `labels=True`, only ordinary-role source
labels and the existing `hlt_matched` eligibility scalar are read. This scalar
does not supply native HLT particles. Returned labels are mapped targets, not
the original ROOT `jet_label` codes. They use the existing frozen 11-class map:

```text
0 QCD; 1 X_bb; 2 X_cc; 3 X_ss; 4 X_qq; 5 X_gg; 6 X_ee;
7 X_mm; 8 X_tauhtaue; 9 X_tauhtaum; 10 X_tauhtauh.
```

The raw source snapshot also contains native Delphes `hlt_part_*` branches;
those are **not this proxy** and the new loader never requests them. Generator
keys, particle keys on the in-memory object, filenames, raw entries, canonical
jet IDs, labels and lineage are metadata/targets, forbidden model features.
An HLT-only deployable model must not accept `jet.offline`.

## Splits and existing experiments

All 2.25M jets, including test materialization, are now copied. Test is no
longer 'still generating'. Both reader routes reject `final_test` before
opening particle data. A downstream test reader requires separately reviewed
finalist/evaluation locks; this module deliberately offers no override.

The existing SPORC experiment used a registered **200k/50k** subset, immediate
parent C25/P75 KD at T=2, controls M0HLT/OFFLINE/U000, DIRECT U000 -> D000 and
COARSE U000 -> U050 -> U100 -> D066 -> D033 -> D000. No dense branch. Its root:

```text
/home/ryreu/atlas/HLT_Classification/checkpoints/jc2_litv3_200k50k_direct_coarse_99a1e231_r1
```

That classifier's source, selected release, matching assignments, checkpoints,
runtime profile and execution receipts were **not** copied with the dataset.
Do not claim the first 200k/50k reader rows reproduce its subset. To reproduce
it, authenticate/import that exact selected release and matching foundation,
or explicitly register a different experiment. Its SPORC-specific gate is not
Oscar execution authorization; the older CMS-proxy Oscar port is a different
dataset adapter. This handoff supplies paired data, not an automatic training
campaign migration.

## Code and authority

- [Frozen production plan](plans/JETCLASS2_LITERATURE_NOISE_V3_PRODUCTION_PLAN.md)
- [Original producer contract](contracts/JETCLASS2_LITERATURE_NOISE_V3_PRODUCTION.md)
- [Relocated consumer contract](contracts/JETCLASS2_LITERATURE_RELOCATED_CONSUMER.md)
- [Literature ladder contract](contracts/JETCLASS2_LITERATURE_V3_LADDER.md)
- `tests/test_literature_proxy_consumer.py`: synthetic copy, hash/identity/role
  rejection, original-path isolation, native mm and bounded paired reads.

The existing datasets and all source manifests remain immutable. Store model
outputs and new experiment artifacts outside these copied source directories.
