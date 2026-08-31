# HCWDL TRI100 Four-Spine Persistent-HLT-Support Implementation Plan

Status: active scientific implementation plan.

## Question

The full-cardinality bottleneck matcher gives every particle on the smaller
endpoint a partner.  When a jet contains more visible HLT particles than
visible offline particles, some HLT particles necessarily have no offline
partner.  The existing full-cardinality control introduces those unmatched
HLT particles as the structural coordinate moves from `U000` to `U100`.

This control asks whether the U path is cleaner when those unavoidable native
HLT particles are present from the beginning.  In that construction U never
adds an HLT-skeleton slot: it only removes offline-only tail particles, or
leaves the support unchanged.

## Frozen comparison

The experiment reuses, read-only:

- the authenticated full-cardinality bottleneck assignment and balanced
  coupling stores;
- the identical train and validation populations;
- the established four paths, coordinates, C25/P75 logit KD at T=2, batch
  size 256, seeds, single-GH200 execution, 100-pass floor-tail schedule, and
  early stopping;
- the existing M0CE60 report and pure-offline U000 report as shared reporting
  references.

It creates an independent campaign root, worktree, contracts, ledger, jobs,
checkpoints, probability banks, aggregate, and completion marker.  It has no
Slurm dependency on, and never mutates, any running campaign.

## Persistent support semantics

Let `P0` be the capped pure-offline support and `H` the visible HLT skeleton.
The frozen assignment divides endpoint particles into common pairs,
offline-only source particles, and HLT-only target slots.  The new policy is:

1. Every HLT target slot is present at every U coordinate.
2. A matched HLT target slot carries its matched offline endpoint features at
   every U coordinate, including `U000` and `U100`.
3. An unmatched HLT target slot carries its native HLT features at every U
   coordinate, including `U000` and `U100`.
4. Every offline-only source particle is also present at `U000` and is removed
   at its authenticated balanced structural switch.
5. A matched offline particle appears exactly once, on its HLT skeleton slot;
   substitutions never create duplicate tail tokens. Therefore `U000` has
   `max(n_offline, n_HLT)` particles under complete smaller-side matching,
   `U100` has exactly `n_HLT`, and support cardinality is non-increasing along
   U.
6. D keeps the HLT skeleton fixed and changes matched slots from offline to
   HLT features exactly as before.  `D000` is byte-identical canonical HLT.

Target slots retain HLT-slot order and source-only tails retain native offline
order.  Any malformed lineage requiring more than 200 tokens fails closed; it
is never silently truncated.  This support policy is legal only with neutral
pairing-validity provenance from the full-cardinality matcher.

Full-cardinality assignments can select legitimate offline lost-track records
from the native charged collection whose stored particle-type flags are finite
binary but zero-hot. Under the versioned validity-only policy
`native_collection_applicability_atomic_raw_endpoint_v1`, native collection
membership determines charged/neutral field applicability, while the raw
offline identity vector is copied unchanged when an endpoint switch fires.
No particle category is invented. One-hot identities must agree with native
collection membership, and nonfinite, nonbinary, incompatible, or
non-discrete required values fail closed. Established confidence-backed paths
remain strictly one-hot. This semantic boundary is part of the campaign,
recipe, graph, and execution-acceptance identities.

## Anchor and training graph

The pure-offline U000 checkpoint is an oracle/reporting reference, not the
training anchor, because its inputs differ.  The campaign trains a fresh
CE-only persistent-support anchor named `SP4P_U000` using the source U000 seed
and source U000 60-pass recipe.  It then materializes a local probability bank
`DIST_SP4P_U000` for the four branch heads.

From that anchor the established four paths are retrained independently:

- direct: `U000 -> D000`;
- coarse: `U000 -> U050 -> U100 -> D066 -> D033 -> D000`;
- dense: `U000 -> U033 -> U066 -> U100 -> D080 -> D060 -> D040 -> D020 -> D000`;
- ultra-dense: `U000 -> U020 -> U040 -> U060 -> U080 -> U100 -> D090 -> ... -> D010 -> D000`.

Every downstream edge uses only the immediately preceding selected model's
T=2 probability bank.  There are no ensembles and no weight continuation.
The campaign has 30 fresh fits and 26 reducers.

## Recovery and reporting

Recovery remains on the shared scientific scale `M0CE60 = 0%` and pure
offline `U000 = 100%`.  `SP4P_U000` is reported as an experimental hybrid
anchor, not mislabeled as an offline oracle.  The aggregate records both its
absolute performance and its recovery on that shared scale before reporting
the four paths.

Poor validation performance never blocks downstream tasks.  Invalid lineage,
forbidden final-test access, nonfinite values, hidden truncation, support
growth along U, stale source, or corrupt artifacts fail closed.

## Evidence required before live submission

1. Focused unit tests prove default U000 semantics are unchanged.
2. Policy tests prove persistent unmatched HLT slots, matched-offline features
   on the HLT skeleton, monotone U support, exact U100 cardinality, exact D000,
   metadata alignment, and fail-closed overflow.
3. Campaign tests prove 30 fits, 26 reducers, independent job namespace,
   correct dependencies, exact recipes, source lineage, and no existing-job
   mutation.
4. Installed-Weaver parity and a genuine Tigris miniature/preflight must pass
   before the full campaign is submitted.
