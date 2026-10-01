# JetClass2 CMS-proxy Tigris 200k three-spine implementation plan

Status: active implementation plan

## Scientific question

Measure how much of the gap between a CMS-calibrated proxy-HLT endpoint and
the paired original offline endpoint can be recovered by logit knowledge
distillation along three persistent-HLT interpolation spines.  This is a new
endpoint study; it must not reuse Delphes-native HLT particle assignments or
models.

## Immutable population and endpoints

- Proxy dataset: the authenticated `CMS2JC2_PROXY_*` production study rooted
  at `/home/ryreu/atlas/datasets/jetclass2_cms_proxy_joint_2250k_b6f88def_r1`.
- Offline endpoint: the exact original dz-fixed ROOT rows named by that study,
  under
  `/home/ryreu/atlas/datasets/jetclass2_10M_20260918_dzfix_partial_v1/jetclass2`.
- Ordinary roles only: 200,000 training jets and 100,000 validation jets.
- Final-test particles, predictions, and labels remain sealed.
- Row selection is label blind, deterministic, immutable, and bound to exact
  authenticated proxy shard receipts and exact offline ROOT file hashes.
- Proxy/offline joins use the canonical 32-byte row identity.  Position alone
  is never accepted as identity.

If the complete dataset manifest is not yet present, an explicit ordinary-role
release may use only committed, individually authenticated train/validation
receipts.  Its receipt set and selected identities are frozen in the release;
it cannot masquerade as the eventual full dataset manifest.

## Matching and view semantics

The particle assignment is recomputed for the new proxy endpoint with the
already selected `SALIENCE_PT_LINEAR` full-cardinality matcher.  Old Delphes
assignment banks are incompatible.

The proxy endpoint supplies the persistent skeleton:

- `U000`: every matched proxy slot carries its paired offline particle; an
  unmatched proxy slot retains its proxy particle; unused offline particles
  form the removable offline tail.
- `Uxxx`: the matched core is unchanged from `U000`; a deterministic fraction
  of the offline tail has been removed.
- `U100`: exact proxy cardinality, with matched slots still carrying offline
  particle features and unmatched proxy slots retaining proxy features.
- `Dxxx`: exact proxy cardinality; matched particle attributes interpolate
  from the `U100` offline-carried endpoint toward proxy-HLT.  Unmatched proxy
  slots remain proxy particles.
- `D000`: exact proxy-HLT particles.
- `OFFLINE`: pure original offline particles, independent of the persistent
  proxy skeleton.

All structural switches and categorical endpoint choices are deterministic
functions of row identity, native particle identity, coordinate, and a
versioned domain.  Labels, classifier outputs, and degradation indices are
forbidden model inputs.

The physical proxy/offline banks are adapted to the established 17 Particle
Transformer inputs with explicit validity handling: invalid impact-parameter
values and errors map to zero; no zero error is treated as a measured
uncertainty; unknown category maps to an all-zero PID one-hot vector.  A fixed
capacity is only an admission ceiling.  Any selected jet exceeding it fails
before scientific training; truncation is forbidden.

## Controls and graph

Fresh controls use the same selected row identities and validation population:

- `M0HLT`: CE-only model on the exact proxy endpoint (`D000`).
- `OFFLINE`: CE-only pure-offline oracle control.
- `U000`: CE-only persistent-HLT offline endpoint.

The KD recipe is C25P75, temperature 2, batch size 256, one GPU per fit, BF16
forward with FP32 loss, selected-best checkpoint restoration, no rolling
resume, and the established 100-pass schedule with early stopping.

Branches:

```text
DIRECT: U000 -> D000
COARSE: U000 -> U050 -> U100 -> D066 -> D033 -> D000
DENSE:  U000 -> U033 -> U066 -> U100 -> D080 -> D060 -> D040 -> D020 -> D000
```

Every nonterminal teacher publishes one durable train-role temperature-2
probability bank.  Views remain RAM-only.  Checkpoints, reports, assignment
indices, and compact probability banks may be durable; materialized full-view
caches may not.

## Execution and gates

Execution site is Tigris only:

- account `reu-aisocial`;
- partition `tigris`;
- environment `atlas_kd_tigris` under the aarch64 Miniforge installation;
- one GH200 GPU for GPU tasks;
- `PYTHONNOUSERSITE=1` and `${CONDA_PREFIX}/lib` first in
  `LD_LIBRARY_PATH`.

Queueing is intentionally two-stage.

1. The gate authenticates source/data, freezes the ordinary-role release,
   computes and audits the new assignment bank, proves endpoint/input
   invariants, runs installed-Weaver parity and a genuine Tigris miniature,
   and measures full selected-population cache/training/reducer resources.
2. Only the exact pushed source and eligible runtime profile may materialize
   and submit the full scientific DAG.

The full campaign cannot be submitted merely because the gate job was queued.
The gate's immutable completion lock and runtime profile must exist and pass
validation first.

## Required evidence

- exact source commit and semantic source hashes;
- authenticated proxy study/release and exact offline source hashes;
- 200k/100k identity counts and disjoint roles;
- exact identity joins between proxy, offline, assignments, caches, and banks;
- full-cardinality assignment and matcher-spec validation;
- `U000`, `U100`, `D000`, and `OFFLINE` endpoint audits;
- no selected-row truncation and no final-test access;
- installed-Weaver parity and genuine Tigris allocation evidence;
- measured RAM/GPU/walltime envelopes;
- exact job ledger, dependency graph, output inventories, and immutable
  completion reports.

## Completion boundary

Implementation is queue-ready when focused local tests pass, the source-pinned
gate and science specifications/CLIs produce exact dry-run command ledgers,
and the operator has commands to push the exact source and queue the gate.
Scientific submission remains gated on the real Tigris evidence above.
