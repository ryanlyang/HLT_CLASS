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

`GATE_SPEC/v5` and `RUNTIME_PROFILE/v5` are the Oscar dual-slot operational
replacement. They retain the same `oscar_l40s` execution site and exact
portable materialization but require exactly 6 CPUs/workers, 90000 MiB, one
L40S, and a 12-hour preflight ceiling. Their explicit two-job intent totals
12 CPUs, 180000 MiB, and two GPUs, remaining within the observed user QOS.
The v5 profile is eligible only after its own genuine full-population
preflight; v4 timing evidence cannot be edited or relabelled as v5.

Final-test access remains false throughout. Scientific metrics cannot control
completion, and live gate/science submissions retain separate exact phrases.

`GATE_SPEC/v6` and `RUNTIME_PROFILE/v6` are the 100k-train/50k-validation
DIRECT+COARSE variant. They reuse the exact v4/v5 portable materialization and
assignment bank, bind `POPULATION_SELECTION/v1`, and retain the v5 per-job
shape of 6 CPUs/workers, 90000 MiB, and one L40S. The v6 preflight must measure
all selected train and validation rows under its own exact pushed source.
