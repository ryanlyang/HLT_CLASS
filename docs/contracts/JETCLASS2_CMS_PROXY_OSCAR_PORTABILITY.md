# JetClass2 CMS-proxy Oscar portability contract

## Artifacts

`JETCLASS2_CMS_PROXY_LADDER_PORTABLE_BUNDLE/v1` is root-independent. Its
payload registry records a relative path, byte count, SHA-256 digest, and
category for every copied file. Its parents are the original gate, release,
foundation, and proxy-study content hashes.

The payload is exactly:

- original gate, release, foundation, and proxy-study JSON;
- `release_index.npz` and `assignments.npz`;
- the imported ordinary inventory;
- every offline ROOT file named by the frozen release;
- every proxy block named by the frozen release.

The payload contains no final-test file. Release selection remains label blind.

`JETCLASS2_CMS_PROXY_LADDER_PORTABLE_MATERIALIZATION/v1` is site-local. It
records the bundle parent, a relocated release, and a relocated foundation.
The relocated wrappers have new hashes because their absolute locations have
changed. They retain the original release/foundation hashes in
`relocated_from`; their physical release bank, assignment bank, offline ROOT
files, and proxy blocks are byte-identical to the portable bundle.

## Validation

Consumers must validate the bundle manifest and every payload file before
materialization. Materialization must validate the relocated release bank,
foundation assignment bank, selected identity digests, matcher/view/input
contracts, and parent lineage. Path existence alone is insufficient.

Proxy-study imports are resolved relative to the authenticated relocated study
root. The immutable study artifact itself is not edited.

## Execution

Oscar preflight uses `GATE_SPEC/v4` and produces `RUNTIME_PROFILE/v4`. The
registered site is `oscar_l40s`. A profile measured on Tigris or SPORC cannot
authorize Oscar science. The Oscar gate has one full-population preflight task;
it does not rebuild the release or matching foundation.

Final-test access remains false throughout. Scientific metrics cannot control
completion, and live gate/science submissions retain separate exact phrases.

