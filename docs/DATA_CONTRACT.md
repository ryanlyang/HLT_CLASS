# JetClass and HLT Data Contract

This document freezes the reusable data semantics. The schema, identity,
split-manifest, discovery, bounded chunk reader, HLT-v3 degradation kernel,
and authenticated sharded caches are implemented through Transfer Block 4.

## Source dataset

```text
recursive root  /home/ryreu/atlas/PracticeTagging/data
ROOT tree       tree
max particles   128
```

Canonical class order:

```text
0 QCD
1 Hbb
2 Hcc
3 Hgg
4 H4q
5 Hqql
6 Zqq
7 Wqq
8 Tbqq
9 Tbl
```

Filename prefixes:

```text
ZJetsToNuNu -> QCD
HToBB       -> Hbb
HToCC       -> Hcc
HToGG       -> Hgg
HToWW4Q     -> H4q
HToWW2Q1L   -> Hqql
ZToQQ       -> Zqq
WToQQ       -> Wqq
TTBar       -> Tbqq
TTBarLep    -> Tbl
```

Matching is longest-prefix-first so `TTBarLep` cannot be assigned to `TTBar`.

## Required branches

Particles:

```text
part_px
part_py
part_pz
part_energy
part_charge
part_isChargedHadron
part_isNeutralHadron
part_isPhoton
part_isElectron
part_isMuon
part_d0val
part_d0err
part_dzval
part_dzerr
```

Labels are the ten one-hot branches `label_QCD` through `label_Tbl`.

## Canonical raw particle schema

Raw tensors have shape `[jets, particles, 14]` with this exact order:

```text
pt, eta, phi, energy, charge,
isChargedHadron, isNeutralHadron, isPhoton, isElectron, isMuon,
d0val, d0err, dzval, dzerr
```

Masks have shape `[jets, particles]` and boolean dtype. Padding rows are zero.
Raw tensors use `float32`; HLT-v3 refuses other dtypes or nonzero/nonfinite
padding rather than allowing profile-dependent dtype or padding behavior.

## Identity and split rules

Canonical identity binds normalized source-file identity, ROOT entry, and
class label. Every campaign freezes split sizes and seeds.

Leakage audits separately use normalized source file plus ROOT entry, without
the label, so an inconsistent relabel cannot evade an overlap check.

Standard roles:

```text
model_train
model_val
stack_train
stack_val
final_test
```

Required properties:

- balanced classwise sampling without replacement;
- disjoint canonical identities across roles;
- deterministic global row shuffle after balanced classwise sampling;
- immutable authenticated split manifest;
- explicit capacity and overlap audit.

Jet-level disjointness must not be mislabeled as file-level disjointness.
All 14 jagged source branches must have identical constituent counts per jet.
Particle values must be finite, PID flags exact 0/1, and label branches exact
finite one-hot values. Production reads fail closed on any violation.

## Canonical Particle Transformer inputs

The versioned `hlt_classification_part_inputs_v1` contract defines:

```text
points          [B, 2, N]
features        [B, 17, N]
lorentz_vectors [B, 4, N]
mask            [B, 1, N]
```

Every jet axis, relative feature, and four-vector is derived from the supplied
view only. An HLT model may not reuse offline jet kinematics. All-empty
degraded rows receive one deterministic finite placeholder, whose semantics
and test are versioned with the input contract.

The 17 channels, in order, are transformed log particle `pT`, transformed log
particle energy, transformed log relative `pT`, transformed log relative
energy, transformed axis distance, charge, five PID indicators, transformed
`d0`, clipped `d0` uncertainty, transformed `dz`, clipped `dz` uncertainty,
signed-axis `deta`, and wrapped `dphi`. Points are the last two coordinates;
Lorentz vectors are `(px, py, pz, energy)`. Padded outputs are exactly zero.

## HLT proxy

The initial registered profile is:

```text
fixed_hlt_v3_track_dominant_proxy/v1
```

It is a deterministic controlled proxy, not real HLT. A campaign freezes
profile, parameters, strength, logical role, replica, and seed lineage.
Strength zero is exact identity.

The registered v1 kernel implements:

- deterministic event streams from logical role, replica, and canonical
  identity;
- independent fixed random substreams for every corruption family;
- neutral-only equal-type local merging by four-vector addition;
- type-dependent constituent loss and mild kinematic response;
- explicit PID, charge, and track-measurement validity;
- track loss, uncertainty inflation, correlated `d0`/`dz` core response, and
  independent tails;
- per-constituent mass-preserving energy recomputation;
- stable descending degraded-`pT` ordering with canonical-index tie breaking.

The random domain separators retain their registered donor-v1 bytes. Changing
them would define a new scientific profile version.

The new repository freezes domain seeds `3053` through `3057` for
`model_train`, `model_val`, `stack_train`, `stack_val`, and `final_test`.
Donor/HOSD aliases `scale_train`, `val_stop`, and `val_design` retain their
original shared-domain mappings.

An HLT cache must bind:

- raw schema and split manifest;
- ordered identities;
- source and output array hashes;
- profile/version/parameters/strength;
- role and replica;
- generator source;
- degradation diagnostics.

Degradation construction indices are not deployable inputs. Future genuine
detector-HLT data require a distinct schema and may not masquerade as this
proxy.

## Authenticated cache format

Offline and HLT caches use bounded deterministic NPZ shards, one
self-authenticating JSON sidecar per shard, and one immutable manifest.
Publication is NPZ first and sidecar second; a final manifest is withheld
until every ordered row range validates. Restart reuses only complete shards
whose file, array, identity, role, schema, source, split, generator, profile,
replica, and parent hashes match exactly.

Readers validate the complete manifest and every consumed shard. Logical
batches may cross physical shard boundaries without changing row order or
scientific output. HLT shards additionally bind their exact offline parent
ranges and contain only deployable tokens, masks, labels, identity keys, and
measurement-validity states. Per-event degradation construction indices are
excluded.
