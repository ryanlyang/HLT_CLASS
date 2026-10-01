# HCWDL Full-Cardinality Salience Matching: Scientific and Code Guide

## Purpose

This document is the reusable guide to the full-cardinality salience matcher.
It explains the scientific objective, the exact assignment rule, the persistent
HLT homotopy that consumes the assignments, the implementation surfaces, and
the steps required to use the strategy safely in another jet-classification
campaign.

The implementation-authoritative scientific specification remains
[the active plan](plans/HCWDL_TRI100_FOUR_SPINE_FULL_CARDINALITY_SALIENCE_MATCHING_IMPLEMENTATION_PLAN.md),
and artifact schemas remain governed by
[the versioned contract](contracts/HCWDL_FULLCARD_SALIENCE_MATCHING.md). This
guide is the practical bridge between those specifications and future code.

## The idea in one paragraph

For each jet, construct exactly `min(n_HLT, n_offline)` one-to-one HLT/offline
particle pairs. Thus every particle on the smaller side is matched. Among all
assignments with that cardinality, choose the global assignment that gives the
best angular matches preferentially to salient particles, where salience is
driven primarily by the particle's share of endpoint jet pT and may include a
small preregistered jet-core bonus. This deliberately permits low-salience
particles to absorb worse matches when that protects high-pT, central
particles. It is a forced training coordinate, not a claim that every selected
pair is a true physical correspondence.

## Why this matcher exists

The earlier full-cardinality bottleneck matcher optimized the worst selected
edge first, then the second worst edge, and so on. That objective is clean but
can spend assignment quality on a soft peripheral particle whose influence on
the classifier is small. The salience matcher instead optimizes the total
useful angular alignment, weighted by particle importance.

It preserves the useful property of full-cardinality matching:

- if `n_HLT <= n_offline`, every HLT particle receives a distinct offline
  partner and the matcher chooses which offline particles to use;
- if `n_HLT > n_offline`, every offline particle is used exactly once and the
  unavoidable extra HLT particles remain unmatched;
- no particle on either side is reused within a jet;
- the output is always oriented on the HLT token skeleton.

The last point is important. An unmatched HLT slot is represented by native
offline index `-1` and `pairing_validity = False`; it is not deleted from the
persistent HLT support used by the homotopy.

## Frozen salience candidates

All arithmetic that controls the assignment is quantized and deterministic.
For endpoint particle `i`, define

```text
x_i       = pT_i / sum_k pT_k
qshare_i  = round_ties_to_even(1,000,000 * x_i)
qaxis_i   = round_ties_to_even(deltaR(i, own endpoint jet axis) / 1e-7)
QCAP      = round_ties_to_even(0.8 / 1e-7)
qcore_i   = max(QCAP - min(qaxis_i, QCAP), 0)
```

The registered candidates are:

| Candidate | Integer particle weight |
| --- | --- |
| `SALIENCE_PT_LINEAR` | `10,000 + qshare_i` |
| `SALIENCE_PT_QUADRATIC` | `10,000 + round(qshare_i^2 / 1,000,000)` |
| `SALIENCE_PT_QUADRATIC_CORE25` | quadratic weight plus `round(weight * qcore_i / (4 * QCAP))` |

The core term can add at most 25% to the quadratic weight. It is intentionally
mild: pT remains the dominant physical prior, while centrality breaks some
otherwise close tradeoffs.

For pair `(i, j)`, define the quantized angular closeness and utility as

```text
qdr_ij       = round_ties_to_even(deltaR(i, j) / 1e-7)
qclose_ij    = max(QCAP - min(qdr_ij, QCAP), 0)
pair_weight  = salience_HLT_i + salience_offline_j
utility_ij   = pair_weight * qclose_ij
```

The `deltaR = 0.8` cap prevents a very bad edge from accumulating increasingly
large negative influence. Beyond the cap, the primary objective regards all
edges as having zero closeness; exact secondary criteria still order them.

## Exact global assignment and tie-breaking

The matcher does not greedily walk particles in descending pT. It solves one
global rectangular assignment problem. The primary objective maximizes total
utility across all selected pairs. The exact secondary hierarchy is:

1. minimize total uncapped quantized delta R;
2. maximize the selected salience on the larger-cardinality side;
3. minimize total absolute log-pT response;
4. minimize raw particle-category mismatch count;
5. minimize valid charge mismatch count;
6. choose the lexicographically smallest native offline-index sequence in HLT
   order, with unmatched slots ordered after real indices.

The production solver encodes this hierarchy with arbitrary-precision
mixed-radix integers and solves it with the integer Hungarian algorithm. It
therefore does not rely on fragile floating-point epsilon weights. For jets
whose largest side has at most eight particles, an independent exhaustive
enumerator checks the production answer exactly.

The canonical implementation is
[`hcwdl_fullcard_salience_matcher.py`](../src/hlt_classification/scouting/hcwdl_fullcard_salience_matcher.py).
The frozen constants, candidate registry, and matcher-spec contracts are in
[`hcwdl_fullcard_salience_contracts.py`](../src/hlt_classification/scouting/hcwdl_fullcard_salience_contracts.py).

## What the pairing means

The pairing is a deterministic coordinate system for training-time view
construction. `pairing_validity = True` means only that an assignment exists
for that HLT slot. It does **not** mean:

- the pair is certainly the same physical particle;
- its delta R is small;
- it passed a correspondence-confidence threshold;
- the classifier may consume an assignment or confidence flag at inference.

Downstream code must never relabel pairing validity as confidence. Match
quality belongs in diagnostics, not deployable features.

## Persistent-HLT homotopy semantics

The assignments feed the persistent-HLT support policy implemented in
[`hcwdl_homotopy.py`](../src/hlt_classification/scouting/hcwdl_homotopy.py).
The complete HLT token skeleton exists throughout the U and D paths.

| View | Matched HLT slots | Unmatched HLT slots | Source-only offline tail |
| --- | --- | --- | --- |
| `U000` | offline endpoint features | native HLT features | all unused offline particles |
| U progression | offline endpoint features | native HLT features | progressively removed |
| `U100` | offline endpoint features | native HLT features | absent |
| D progression | interpolated/degraded offline-to-HLT features | native HLT features | absent |
| `D000` | native HLT features | native HLT features | absent |

Consequently, U changes support cardinality only by removing source-only
offline particles. It never deletes an HLT slot. D changes the content of the
matched slots until the endpoint is exact HLT. This is the distinction future
campaigns must preserve: `U100` has HLT cardinality but is not yet the exact
HLT feature endpoint; `D000` is exact HLT.

## Code map

| Responsibility | Code |
| --- | --- |
| Matcher math, production solver, exhaustive reference | [`hcwdl_fullcard_salience_matcher.py`](../src/hlt_classification/scouting/hcwdl_fullcard_salience_matcher.py) |
| Constants, candidate registry, versioned matcher specs | [`hcwdl_fullcard_salience_contracts.py`](../src/hlt_classification/scouting/hcwdl_fullcard_salience_contracts.py) |
| Compact assignment shards, manifests, lazy lookup, recomputation audit | [`hcwdl_fullcard_salience_cache.py`](../src/hlt_classification/scouting/hcwdl_fullcard_salience_cache.py) |
| All-row matching diagnostics | [`hcwdl_fullcard_salience_diagnostics.py`](../src/hlt_classification/scouting/hcwdl_fullcard_salience_diagnostics.py) |
| Assignment/coupling foundation construction | [`hcwdl_fullcard_salience_foundation.py`](../src/hlt_classification/scouting/hcwdl_fullcard_salience_foundation.py) |
| Immutable foundation DAG | [`hcwdl_fullcard_salience_foundation_campaign.py`](../src/hlt_classification/scouting/hcwdl_fullcard_salience_foundation_campaign.py) and [`hcwdl_fullcard_salience_foundation_workflow.py`](../src/hlt_classification/scouting/hcwdl_fullcard_salience_foundation_workflow.py) |
| Matched four-fit candidate screen and selection lock | [`hcwdl_fullcard_salience_screen.py`](../src/hlt_classification/scouting/hcwdl_fullcard_salience_screen.py), [`hcwdl_fullcard_salience_screen_campaign.py`](../src/hlt_classification/scouting/hcwdl_fullcard_salience_screen_campaign.py), and [`hcwdl_fullcard_salience_screen_workflow.py`](../src/hlt_classification/scouting/hcwdl_fullcard_salience_screen_workflow.py) |
| Persistent-support view assembly | [`hcwdl_homotopy.py`](../src/hlt_classification/scouting/hcwdl_homotopy.py) |
| Production four-spine graph and execution | [`hcwdl_tri100_spine4_salience_graph.py`](../src/hlt_classification/scouting/hcwdl_tri100_spine4_salience_graph.py), [`hcwdl_tri100_spine4_salience_runner.py`](../src/hlt_classification/scouting/hcwdl_tri100_spine4_salience_runner.py), and [`hcwdl_tri100_spine4_salience_workflow.py`](../src/hlt_classification/scouting/hcwdl_tri100_spine4_salience_workflow.py) |
| Production source authentication and contracts | [`hcwdl_tri100_spine4_salience_source.py`](../src/hlt_classification/scouting/hcwdl_tri100_spine4_salience_source.py) and [`hcwdl_tri100_spine4_salience_contracts.py`](../src/hlt_classification/scouting/hcwdl_tri100_spine4_salience_contracts.py) |
| Focused tests | [`test_hcwdl_fullcard_salience.py`](../tests/test_hcwdl_fullcard_salience.py) and [`test_hcwdl_tri100_spine4_salience.py`](../tests/test_hcwdl_tri100_spine4_salience.py) |

## Direct in-memory use

For development or a new foundation worker, decode both endpoint particle sets,
adapt them to the common `Particles` representation, and run one frozen
candidate:

```python
from hlt_classification.scouting.hcwdl_fullcard_salience_contracts import (
    SALIENCE_PT_QUADRATIC_CORE25,
)
from hlt_classification.scouting.hcwdl_fullcard_salience_matcher import (
    FullCardinalitySalienceMatcher,
)
from hlt_classification.scouting.highcov_matcher import from_scouting_particles
from hlt_classification.scouting.particles import decode_particle_sets

hlt_raw, offline_raw, truncated_hlt_count = decode_particle_sets(arrays, row)
# The third result is the number beyond the configured HLT support cap. Bind
# it into diagnostics and enforce the source campaign's declared cap policy.

hlt = from_scouting_particles(hlt_raw, offline=False)
offline = from_scouting_particles(offline_raw, offline=True)
pairing = FullCardinalitySalienceMatcher(
    SALIENCE_PT_QUADRATIC_CORE25
).match(hlt, offline)

native_offline_index = pairing.native_offline_index
pairing_validity = pairing.pairing_validity
```

`native_offline_index` is HLT-oriented and durable. The
`concatenated_offline_index` field is a transient local index into the decoded
offline array and must not be treated as a cross-file identity.

For a toy solver test in which edge matrices are already available, use
`production_pairing_from_matrices`. Use `reference_pairing_from_matrices` only
for exhaustive verification with `max(n_HLT, n_offline) <= 8`.

## Durable assignment use

The durable format intentionally stores only what future view construction
needs:

- `entries` as `int64`;
- ragged `offsets` as `uint64`;
- HLT-oriented `native_offline_index` as `int16`;
- `pairing_validity_u8` as `uint8`.

It does not store dense cost matrices, pairwise delta-R matrices, particle
views, salience vectors, or correspondence confidence. Those large
intermediates remain in memory and are discarded.

Consume a validated manifest through the store rather than opening NPZ files
ad hoc:

```python
from hlt_classification.scouting.hcwdl_fullcard_salience_cache import (
    FullCardinalitySalienceAssignmentStore,
)

store = FullCardinalitySalienceAssignmentStore(assignment_manifest_path)
row = store.get(source_path, entry)

# Batched, padded HLT-oriented arrays for a model/cache builder.
indices, validity = store.join(source_path, entries, max_length=200)
assert store.provenance_kind == "pairing_validity"
```

Foundation writers must use `publish_assignment_shard` and
`publish_assignment_manifest`; consumers must call
`validate_assignment_manifest`. Production foundations also publish a
deterministic `sampled_recomputation_audit` that recomputes selected source
rows and demands exact native indices and validity masks.

## Reusing the selected JetClass2 TRAIN_500K assignments

The JetClass2 Delphes `TRAIN_500K` screen completed on 2026-09-14 and selected
`SALIENCE_PT_LINEAR`. Later ladders on the same authenticated population must
reuse that completed assignment foundation rather than resubmitting the three
matching-foundation DAGs or rerunning the candidate screen.

On SPORC, the selected assignment foundation and its selecting screen are:

```bash
export JC2_SALIENCE_FOUNDATION=/home/ryreu/atlas/HLT_Classification/checkpoints/jc2_salience500k_linear_4f5e520e_r1/foundation
export JC2_SALIENCE_SCREEN=/home/ryreu/atlas/HLT_Classification/checkpoints/jc2_salience500k_screen_4f5e520e_r1
```

The foundation directory is the reusable scientific object. Preserve it
read-only and preserve its path. In particular, retain:

- `foundation_spec.json`;
- `foundation_lock.json`;
- `sample_audit.json` and `assignment_audit.json`;
- every `assignments/NNNN.npz` payload and matching `NNNN.json` report.

The screen directory supplies the authorization for choosing this foundation.
Retain at least `screen_spec.json`, `selection_lock.json`,
`runtime_profile.json`, and `screen_complete.json`. The selection lock records
the four preregistered fit results and names `SALIENCE_PT_LINEAR` as winner.
Do not replace it with an informal note that linear won.

Before reuse, authenticate the complete foundation from the exact source-pinned
checkout. This reads and validates the existing compact maps; it does not
recompute the Hungarian assignments:

```bash
export PROJECT_DIR=/home/ryreu/atlas/HLT_Classification_jc2_salience_4f5e520e
export PYTHONPATH="${PROJECT_DIR}/src"

python -s - "${JC2_SALIENCE_FOUNDATION}" "${JC2_SALIENCE_SCREEN}" <<'PY'
import sys
from pathlib import Path

from hlt_classification.data.cache_contracts import load_json
from hlt_classification.jetclass2_delphes.salience_foundation import (
    authenticate_preparation,
)

foundation_root = Path(sys.argv[1]).resolve()
screen_root = Path(sys.argv[2]).resolve()
foundation = load_json(foundation_root / "foundation_spec.json")
lock = authenticate_preparation(foundation, foundation_root)
selection = load_json(screen_root / "selection_lock.json")

assert foundation["candidate"] == "SALIENCE_PT_LINEAR"
assert selection["winner"] == foundation["candidate"]
assert selection["winner_foundation_sha256"] == foundation["content_hash"]
assert Path(selection["winner_foundation_root"]).resolve() == foundation_root
assert selection["final_test_accessed"] is False

print("Authenticated reusable candidate:", foundation["candidate"])
print("Foundation SHA256:", foundation["content_hash"])
print("Foundation lock SHA256:", lock["content_hash"])
print("Selection lock SHA256:", selection["content_hash"])
PY
```

### Supported reuse in another JetClass2 ladder

For another DIRECT/COARSE/DENSE campaign using the same implementation, point
production creation at the completed screen instead of creating any matching
foundation:

```bash
python -s "${PROJECT_DIR}/scripts/jetclass2_delphes_salience_production.py" create \
  --screen-spec "${JC2_SALIENCE_SCREEN}/screen_spec.json" \
  --data-root /home/ryreu/atlas/datasets/jetclass2_10M_20260910/jetclass2 \
  --output-root "${NEW_CAMPAIGN_ROOT}" \
  --source-commit 4f5e520e0c46d8e24dff420e93c8109434539c84
```

Creation authenticates the frozen screen, selected foundation, compact
assignment bytes, recomputation audit, dataset inventory, and split membership
before writing the new campaign specification. It does not run the matcher.
The new output root must be fresh and separate from the foundation, the screen,
and every existing campaign. This existing creator is source-pinned to the
screen's exact producer commit and recreates its registered three-spine graph;
it is not the interface for a new graph implemented at a later commit.

The current supported creator authenticates the complete original screen
provenance. Therefore, while using that creator, also retain the other two
salience candidate foundation roots and their assignment payloads, the
bottleneck contextual foundation spec/lock, and the measured runtime-profile
artifact referenced by `screen_spec.json`. They are checked as evidence for
the frozen selection decision; they are not used to construct the new ladder's
particle views. A future source contract may bind only the winner foundation
and immutable selection evidence, but it must implement and test that narrower
authentication explicitly before the nonwinning roots are retired.

### Reuse in a new graph implementation

A different ladder graph may consume the same assignments by loading
`foundation_spec.json`, calling `authenticate_preparation`, and passing the
same foundation root to
`jetclass2_delphes.salience_cache.prepare_salience_cache`. Its versioned source
contract must bind at least:

- the foundation-spec and foundation-lock content hashes;
- the exact `SALIENCE_PT_LINEAR` matcher-spec hash;
- inventory, split-registry, role-membership, reader, input-schema, and
  assignment-producer identities;
- the immutable selection-lock hash and winner identity;
- `persistent_hlt_skeleton_offline_tail_v1` view semantics;
- the absence of final-test access.

Changing the optimizer, KD recipe, random seeds, pass budget, or ladder graph
does not by itself require rematching. Changing the dataset snapshot, split
membership, particle decoder, input feature schema, particle-cap/truncation
policy, endpoint identity, matcher constants, matcher candidate, or
persistent-HLT support semantics does require a new foundation. If an
authentication check fails, fail closed; do not silently regenerate maps
inside a training job and do not fall back to path-existence checks.

## How to adopt the matcher in another campaign

### 1. Decide whether assignments may be reused

An existing assignment manifest may be reused only if all of the following are
identical and authenticated:

- source ROOT-file identities and entry identities;
- split and role membership;
- particle decoding, truncation, native-index, phi-wrap, and category/charge
  semantics;
- the complete matcher-spec content hash and selected candidate;
- the assignment-manifest parent lineage.

Path existence is never sufficient. A different dataset, split, truncation,
particle schema, matching constant, or salience formula requires new
assignments. A downstream ladder change alone does not, provided its source
lock authenticates the unchanged foundation and selection lock.

### 2. Use a versioned matcher spec

Call `matcher_spec(candidate)` from the contracts module and bind its content
hash into the foundation, assignment shards, manifest, diagnostics, and every
downstream source lock. Do not copy the numeric formula into a new campaign or
replace the candidate string with an informal label.

If scientific semantics change, add a new named candidate and bump the
relevant contract/schema. Never change an existing candidate in place.

### 3. Build assignments outside the training loop

Run the matcher once per authenticated source entry in foundation jobs. Publish
compact shards and a manifest, then let training jobs lazily join by
`(source_path, entry)`. Do not recompute global assignments every epoch and do
not serialize dense matrices.

### 4. Rebuild every assignment-dependent artifact

Changing a matcher invalidates more than the native-index array. Rebuild:

- assignment shards and manifests;
- assignment diagnostics and recomputation audit;
- coupling configurations and coupling caches;
- balanced sampling/coupling manifests that depend on those caches;
- endpoint-equivalence and source locks;
- every downstream view or probability artifact whose lineage includes the
  old assignment.

Pure-offline artifacts that provably do not depend on matching may be reused
read-only under an explicit parent hash.

### 5. Preserve persistent-HLT endpoint semantics

Future view builders must keep unmatched HLT slots native at every U and D
point. They must not turn `pairing_validity = False` into padding and must not
substitute an offline token into more than one HLT slot. Source-only offline
particles may appear only in the removable U tail.

### 6. Keep deployable inputs clean

Neither native assignment indices, salience weights, source indices,
pairing-validity bits, nor matching diagnostics may enter the deployable HLT
classifier as features. They may control training-time view construction and
lineage only.

### 7. Validate before scientific use

At minimum, a new integration must prove:

- exact selected cardinality `min(n_HLT, n_offline)` on every row;
- no offline-index reuse within a jet;
- exact agreement with exhaustive enumeration on eligible small jets;
- deterministic results under repeat execution;
- compact artifact round-trip and exact sampled recomputation;
- correct unmatched-HLT persistence and exact `D000 == HLT` endpoint;
- correct `U000`, `U100`, and coupling-cache identities;
- no final-test access.

## Candidate selection for a new scientific domain

The repository does not assume one salience formula is universally best. The
current workflow creates one isolated foundation for each of the three fixed
candidates and compares them with the bottleneck matcher in a matched four-fit
CE-only U100 screen.

The validation role is deterministically divided into a 75% checkpoint subset
and a one-shot 25% selection subset, stratified by class and stable identity.
Selection maximizes macro one-vs-rest AUC on the selection subset. Within a
`5e-5` AUC band it chooses macro log R50, then accuracy, then lower pT-weighted
delta R, then registry order. A 2,000-pair bootstrap is descriptive and cannot
change the winner. `validate_screen_report` and `validate_selection_lock`
recompute the decision.

For another dataset or materially different jet domain, rerun this screen
rather than assuming the previously selected candidate transfers. For another
ladder on the same authenticated population, reuse the selection lock and
winner foundation rather than screening again.

## Reference three-stage command flow

These commands show the intended lifecycle. Use a detached worktree pinned to
the exact pushed commit, materialize each dry run, inspect its command plan and
ledger, and obtain explicit authorization before adding `--execute`.

### A. Build one foundation per candidate

```bash
python -s scripts/create_hcwdl_fullcard_salience_foundation.py \
  --source-campaign-spec "${SOURCE_SPEC}" \
  --bottleneck-foundation-spec "${BOTTLENECK_FOUNDATION_SPEC}" \
  --candidate SALIENCE_PT_QUADRATIC_CORE25 \
  --foundation-root "${FOUNDATION_ROOT}" \
  --project-dir "${PROJECT_DIR}" \
  --source-commit "${SOURCE_COMMIT}" \
  --authorize-live-submission \
  --authorization-phrase \
    "AUTHORIZE HCWDL TRI100 FOUR SPINE SALIENCE FOUNDATION EXACT SPEC"

python -s scripts/submit_hcwdl_fullcard_salience_foundation.py \
  --spec "${FOUNDATION_ROOT}/foundation_spec.json" \
  --output "${FOUNDATION_ROOT}/dry_run_submission_ledger.json"
```

Repeat with a separate root for all three registered candidates. Only after
auditing the dry run should the same submit command use `--execute` and
`"SUBMIT HCWDL TRI100 FOUR SPINE SALIENCE FOUNDATION EXACT LEDGER"`.

### B. Run the matched candidate screen

```bash
python -s scripts/create_hcwdl_fullcard_salience_screen_campaign.py \
  --bottleneck-foundation-spec "${BOTTLENECK_FOUNDATION_SPEC}" \
  --salience-foundation-spec "${LINEAR_FOUNDATION_SPEC}" \
  --salience-foundation-spec "${QUADRATIC_FOUNDATION_SPEC}" \
  --salience-foundation-spec "${CORE25_FOUNDATION_SPEC}" \
  --campaign-root "${SCREEN_ROOT}" \
  --project-dir "${PROJECT_DIR}" \
  --source-commit "${SOURCE_COMMIT}" \
  --authorize-live-submission \
  --authorization-phrase \
    "AUTHORIZE HCWDL FULLCARD SALIENCE SCREEN EXACT SPEC"

python -s scripts/submit_hcwdl_fullcard_salience_screen_campaign.py \
  --spec "${SCREEN_ROOT}/campaign_spec.json" \
  --output "${SCREEN_ROOT}/dry_run_submission_ledger.json"
```

The live submission phrase is
`SUBMIT HCWDL FULLCARD SALIENCE SCREEN EXACT LEDGER`.

### C. Bind the winner into a downstream campaign

The existing four-spine control is one example:

```bash
python -s scripts/create_hcwdl_tri100_spine4_salience_campaign.py \
  --foundation-spec "${WINNER_FOUNDATION_SPEC}" \
  --selection-lock "${SCREEN_ROOT}/selection_lock.json" \
  --established-campaign-spec "${ESTABLISHED_CAMPAIGN_SPEC}" \
  --m0ce60-report "${M0CE60_REPORT}" \
  --campaign-root "${CAMPAIGN_ROOT}" \
  --project-dir "${PROJECT_DIR}" \
  --source-commit "${SOURCE_COMMIT}" \
  --authorize-live-submission \
  --authorization-phrase \
    "AUTHORIZE HCWDL TRI100 FOUR SPINE SALIENCE EXACT SPEC"

python -s scripts/submit_hcwdl_tri100_spine4_salience_campaign.py \
  --spec "${CAMPAIGN_ROOT}/campaign_spec.json" \
  --output "${CAMPAIGN_ROOT}/dry_run_submission_ledger.json"
```

The live production phrase is
`SUBMIT HCWDL TRI100 FOUR SPINE SALIENCE EXACT LEDGER`. A future campaign need
not copy this graph; it should copy the authentication pattern in
`hcwdl_tri100_spine4_salience_source.py` and bind the same foundation,
selection-lock, matcher-spec, assignment-manifest, and persistent-support
identities into its own versioned source contract.

## Storage and operational behavior

The strategy is designed for large jet samples without leaving huge matching
artifacts on research compute:

- pairwise matrices and decoded particle views are job-local memory only;
- durable assignment artifacts contain one small native index and one validity
  byte per visible HLT token, plus row offsets and identities;
- probability reducers store only compact class-probability banks;
- production training does not persist rolling optimizer/resume state;
- immutable reports, locks, compact checkpoints, and task attestations remain.

Before deploying on a new population, estimate visible HLT-token count and
verify that `int16` can represent its native per-jet offline indices. The
publisher fails closed if it cannot.

## Common integration mistakes

- **Calling U100 exact HLT.** It has HLT support cardinality, but matched slots
  still carry offline endpoint features. D000 is exact HLT.
- **Dropping unmatched HLT slots.** Those slots are persistent native HLT
  tokens, not padding.
- **Treating validity as confidence.** It records assignment presence only.
- **Using concatenated offline indices durably.** Persist native indices.
- **Reusing an old manifest after a split or decoder change.** Rebuild it.
- **Implementing the objective with floating scalar weights.** Use the exact
  registered solver.
- **Greedily matching high-pT particles.** The intended objective is a global
  assignment; greedy matching can sacrifice total utility.
- **Changing a candidate silently.** Add a new versioned candidate and rerun
  foundations and selection.
- **Writing dense matrices or particle targets to disk.** Keep them in RAM.
- **Letting matching metadata enter deployable inputs.** The final classifier
  remains HLT-only.

## Handoff checklist for a future campaign

Before declaring a new use queue-ready, the implementing agent should be able
to answer yes to every item:

- Is the exact matcher candidate named and content-hashed?
- Is the source population identical to a reusable foundation, or were fresh
  foundations and a candidate screen created?
- Are assignments full-cardinality, unique, HLT-oriented, and recomputation
  audited?
- Does the view path preserve unmatched HLT tokens and reach exact HLT only at
  D000?
- Were all assignment-dependent caches rebuilt?
- Does the source lock authenticate the selection lock and winner foundation?
- Are dense matching intermediates RAM-only and durable outputs compact?
- Are assignment metadata excluded from deployable inputs?
- Do focused solver, cache, endpoint, lineage, and recovery tests pass?
- Was the exact pushed commit dry-run on the intended production workers before
  live submission?

If any answer is no, the new campaign is not yet a scientifically equivalent
use of this matching strategy.
