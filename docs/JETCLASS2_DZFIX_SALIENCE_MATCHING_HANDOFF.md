# JetClass2 dz-fix salience matching handoff

Last updated: 2026-09-21

This document is the practical handoff for reusing the full-cardinality,
salience-selected, persistent-HLT matching setup on the authenticated
JetClass2 `20260918_dzfix` `TRAIN_500K` population. It records the current
execution state, dataset and artifact locations, scientific semantics, code
interfaces, validation boundary, storage requirements, and the safe way for a
different ladder to inherit the selected matching.

The implementation-authoritative documents remain:

- [dz-fix migration plan](plans/JETCLASS2_DELPHES_DZFIX_500K_MIGRATION_PLAN.md);
- [salience persistent-HLT plan](plans/JETCLASS2_DELPHES_FULLCARD_SALIENCE_PERSISTENT_500K_PLAN.md);
- [salience persistent-HLT contract](contracts/JETCLASS2_DELPHES_FULLCARD_SALIENCE_PERSISTENT.md);
- [dz-fix continuation contract](contracts/JETCLASS2_DELPHES_DZFIX_SALIENCE_CONTINUATION.md);
- [generic salience matcher/code guide](HCWDL_FULLCARD_SALIENCE_MATCHING_CODE_GUIDE.md).

This handoff specializes those documents to the new dz-fixed snapshot. It does
not authorize final-test access or live submission of a new scientific
campaign.

## Executive summary

- Dataset: the authenticated partial snapshot of Luka's September 18 dz-fixed
  JetClass2 Delphes production.
- Population: exactly 500,000 TRAIN rows and 1,000,000 validation rows;
  1,000,000 final-test rows remain sealed.
- Matching: exactly `min(n_HLT, n_offline)` unique one-to-one pairs, selected
  by a global pT-salience-weighted angular objective rather than worst-edge
  minimization.
- View support: the complete HLT skeleton persists at every U and D point.
- Candidate choice: linear, quadratic, and quadratic-plus-core25 salience are
  compared in a matched four-fit U100 screen; the old bottleneck matcher is a
  contextual control and cannot win.
- Reuse rule: another ladder on this exact population should authenticate and
  reuse the selected compact assignment foundation. It should not rerun the
  matcher or screen.
- Current queue boundary: a new campaign launcher that needs the selected
  matching should use `afterok:21748725`, the completion job of the
  source-pinned replacement screen. Do not depend on an individual fit.

## Current state on 2026-09-21

The destination snapshot, split registry, bottleneck preparation, bottleneck
runtime profile, and all three salience candidate foundations have completed.
The original tier3 screen tail (`21741409` through `21741416`) was canceled by
exact job ID before it began scientific work. A fresh, source-pinned v2 screen
was then submitted to the SPORC `debug` partition because scheduler priority
was the limiting factor. Its scientific data, foundations, validation split,
seeds, model, and schedule are unchanged; only the scheduler partition and
bounded eight-hour request differ. The downstream production execution site
remains explicitly pinned to `tier3`.

| Job | Meaning | Observed state |
| --- | --- | --- |
| `21748718` | authenticate the exact reused foundations | running when this handoff was updated |
| `21748719` | genuine A100 screen preflight and validation firewall | depends on authentication |
| `21748720` | bottleneck-context U100 CE control | depends on preflight |
| `21748721` | linear-salience U100 CE fit | depends on preflight |
| `21748722` | quadratic-salience U100 CE fit | depends on preflight |
| `21748723` | quadratic-core25 U100 CE fit | depends on preflight |
| `21748724` | deterministic candidate selection | depends on all four fits |
| `21748725` | screen-completion lock and reuse boundary | depends on selection |

No winner is named in this document before `21748724` succeeds. The immutable
`selection_lock.json`, not a chat message or candidate path, is the authority.
Poor scientific metrics do not fail selection; invalid lineage, data, source,
or nonfinite required quantities do.

If `21748725` completes successfully, the matching and selection are ready for
reuse. Unlike the canceled continuation tail, this replacement screen does not
itself create the 30-task production dry run. A consumer dependent on
`21748725` must authenticate the completed screen and selected foundation,
then create its own source-pinned campaign and canonical dry run. If a parent
fails, downstream `afterok` jobs remain unsatisfied; do not replace the
dependency with `afterany`.

## Dataset identity and locations

Original producer location at CERN:

```text
/eos/user/l/llambrec/jetclass/output_jetclass2_10M_20260918_dzfix/jetclass2
```

Authenticated partial snapshot on SPORC:

```text
/home/ryreu/atlas/datasets/jetclass2_10M_20260918_dzfix_partial_v1/jetclass2
```

Inventory and split-registry artifacts:

```text
/home/ryreu/atlas/HLT_Classification/checkpoints/jc2_dzfix_partial_inventory_0b3ed523_r1/inventory.json
/home/ryreu/atlas/HLT_Classification/checkpoints/jc2_dzfix_partial_inventory_0b3ed523_r1/splits.json
/home/ryreu/atlas/datasets/jetclass2_10M_20260918_dzfix_partial_v1/jetclass2_delphes_scaling_splits_20260918_v1
/home/ryreu/atlas/datasets/jetclass2_10M_20260918_dzfix_partial_v1/jetclass2_delphes_scaling_splits_20260918_v1/profiles/TRAIN_500K.json
```

Destination verification recorded:

- 186 authenticated ROOT files;
- 17,273,042,926 bytes;
- 5,283,833 selected rows in the partial snapshot;
- outer reservoirs of 3,169,607 train, 1,057,457 validation, and 1,056,769
  final-test rows;
- registered `TRAIN_500K` membership of 500,000 train, 1,000,000 validation,
  and 1,000,000 sealed final-test rows.

The snapshot retains the producer's all-reconstructed-particle PUPPI
neighbourhood setting and fixes HLT `dz` to be relative to the primary vertex
rather than the origin. It is not the prospective no-PUPPI production. A
no-PUPPI or otherwise regenerated dataset is a new snapshot and requires new
inventory, split, assignments, candidate screen, and acceptance.

Treat the raw dataset and split registry as read-only. Absolute paths are
operational locations, not scientific identity; reusable artifacts bind file
hashes, tree cycles, entries, schema, and exact row membership.

## Canonical execution and artifact roots

Source commit used by the active replacement screen:

```text
0d25a4a53aafb1348c8279d86bac7dbac82c8841
```

Pinned checkout and active screen root:

```text
/home/ryreu/atlas/HLT_Classification_jc2_dzfix_debug_0d25a4a5
/home/ryreu/atlas/HLT_Classification/checkpoints/jc2_dzfix_salience_debug_0d25a4a5_r1
```

For commands below:

```bash
export PROJECT_DIR=/home/ryreu/atlas/HLT_Classification_jc2_dzfix_debug_0d25a4a5
export DATA_ROOT=/home/ryreu/atlas/datasets/jetclass2_10M_20260918_dzfix_partial_v1/jetclass2
export SCREEN_ROOT=/home/ryreu/atlas/HLT_Classification/checkpoints/jc2_dzfix_salience_debug_0d25a4a5_r1
export PYTHONPATH="${PROJECT_DIR}/src"
```

The relevant subtree is:

```text
SCREEN_ROOT/
  screen_spec.json
  command_plan.json
  dry_run_submission_ledger.json
  submission_ledger.json
  runtime_profile.json               # after preflight
  screen_split.json                  # after preflight
  selection_lock.json                # after selection
  screen_complete.json               # after completion
  tasks/...
  attempts/...
```

The active `screen_spec.json` is authoritative for the bottleneck root,
resource template, data root, and all three candidate foundation roots. Do not
reconstruct those paths from the old continuation-root name. That assumption
was tested during the debug transition and was false. Read the exact paths and
content hashes from the screen specification.

The preceding bottleneck readiness and its recovered profile remain part of
the screen lineage:

```text
/home/ryreu/atlas/HLT_Classification/checkpoints/jc2_dzfix_sporc_ready_500k_0b3ed523_r1
/home/ryreu/atlas/HLT_Classification/checkpoints/jc2_dzfix_sporc_ready_500k_0b3ed523_r1/recovery_assign_480m_r1
```

Do not substitute the older September 10 roots such as
`jc2_salience500k_linear_4f5e520e_r1`. Those belong to the previous dataset and
are not valid assignments for the dz-fixed snapshot.

## Matching semantics

For every jet, the matcher selects exactly
`min(n_HLT, n_offline)` unique pairs over the full rectangular pairing problem:

- when `n_HLT <= n_offline`, every HLT particle receives one unique offline
  partner and some offline particles may remain unused;
- when `n_HLT > n_offline`, every offline particle is used once and the excess
  HLT particles remain unmatched;
- no HLT or offline particle is reused within a jet;
- the durable map is HLT-oriented: one native offline index per HLT slot, with
  `-1` only for unavoidable excess HLT slots.

This is a forced training coordinate, not a statement that every pair is the
same reconstructed particle. Pair validity means only that a selected partner
exists. Match indices, validity, salience, and quality are forbidden model
features.

The primary objective maximizes total integer pair utility. Angular closeness
is based on float64, half-open-phi delta R, quantized at `1e-7`, and capped at
`0.8` in the primary closeness term. Each pair's closeness is weighted by the
sum of its HLT and offline endpoint saliences. The three registered definitions
are:

- `SALIENCE_PT_LINEAR`: floor plus linear endpoint pT share;
- `SALIENCE_PT_QUADRATIC`: floor plus squared endpoint pT share;
- `SALIENCE_PT_QUADRATIC_CORE25`: quadratic salience plus an at-most-25% own-axis
  central-core bonus.

The solver is global, not greedy. After primary utility it applies the frozen
integer-exact hierarchy: uncapped delta R, selected larger-side salience,
absolute log-pT response, category mismatch, valid-charge mismatch, and native
index tie-breaking. Small eligible cases are checked against exhaustive
enumeration.

## Persistent-HLT U/D views

The HLT skeleton never disappears:

| View | Matched HLT slots | Excess HLT slots | Extra offline particles |
| --- | --- | --- | --- |
| `U000` | offline features placed on HLT slots | native HLT | appended source-only tail |
| U progression | still offline on matched slots | native HLT | progressively removed |
| `U100` | still offline on matched slots | native HLT | absent |
| D progression | offline-to-HLT interpolation/switching | native HLT | absent |
| `D000` | exact native HLT | exact native HLT | absent |

Consequences that future ladders must preserve:

- If HLT has more particles than offline, `U000` has HLT cardinality and the
  extra HLT particles remain native HLT.
- If offline has more particles, `U000` has the HLT skeleton plus the unused
  offline tail; U removes only that tail.
- `U100` has HLT cardinality but is not exact HLT, because matched slots still
  carry offline features.
- Only `D000` is the exact HLT endpoint and can be built without offline input.
- D is allowed only after U support removal is complete.

The model interface remains 17 inputs, 11 output classes, capacity 240,
`trim=False`. Intermediate views use offline information only as declared
training-time supervision. A deployable model consumes HLT inputs only.

## Candidate screen and selection

The screen compares four seed-, data-, architecture-, and schedule-matched
CE-only U100 fits:

1. historical bottleneck context;
2. linear salience;
3. quadratic salience;
4. quadratic-core25 salience.

Only the three salience candidates are eligible to win. The validation role is
deterministically split by class and identity into 75% checkpoint selection
and a one-shot 25% matcher-selection partition. Selection maximizes macro OVR
AUC, treats candidates within `5e-5` as tied, then uses macro log R50,
accuracy, lower pT-weighted selected delta R, and frozen registry order.

The winner is authoritative only when all of these agree:

- `screen_spec.json`;
- `selection_lock.json`;
- `screen_complete.json`;
- the selected `foundation_spec.json` and `foundation_lock.json`;
- the selected foundation path and content hash recorded in the lock.

## Checking completion and discovering the winner

```bash
set -euo pipefail

source /home/ryreu/miniconda3/etc/profile.d/conda.sh
conda activate atlas_kd_sporc
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${PROJECT_DIR}/src"

test -f "${SCREEN_ROOT}/selection_lock.json"
test -f "${SCREEN_ROOT}/screen_complete.json"

python -s - "${SCREEN_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
spec = json.loads((root / "screen_spec.json").read_text())
selection = json.loads((root / "selection_lock.json").read_text())
complete = json.loads((root / "screen_complete.json").read_text())
ledger = json.loads((root / "submission_ledger.json").read_text())

assert selection["final_test_accessed"] is False
assert selection["screen_sha256"] == spec["content_hash"]
assert complete["screen_sha256"] == spec["content_hash"]
assert complete["selection_lock_sha256"] == selection["content_hash"]
assert ledger["dry_run"] is False
assert ledger["jobs"]["complete"] == "21748725"
assert spec["screen_execution_site"]["partition"] == "debug"
assert spec["production_execution_site"]["partition"] == "tier3"

print("Winner:          ", selection["winner"])
print("Foundation root: ", selection["winner_foundation_root"])
print("Foundation hash: ", selection["winner_foundation_sha256"])
print("Selection hash:  ", selection["content_hash"])
print("Reuse boundary:   afterok:" + ledger["jobs"]["complete"])
PY
```

## Authenticating and reusing the selected assignments

A consumer must validate content and lineage, not just test that directories
exist. The following check authenticates the screen and every compact payload
in the selected foundation without rerunning the Hungarian matcher:

```bash
python -s - "${SCREEN_ROOT}" "${DATA_ROOT}" <<'PY'
import sys
from pathlib import Path

from hlt_classification.data.cache_contracts import load_json
from hlt_classification.jetclass2_delphes.contracts import validate
from hlt_classification.jetclass2_delphes.salience_foundation import (
    authenticate_preparation,
    validate_foundation_spec,
)
from hlt_classification.jetclass2_delphes.salience_screen import validate_screen

screen_root = Path(sys.argv[1]).resolve()
data_root = Path(sys.argv[2]).resolve()
screen = load_json(screen_root / "screen_spec.json")
validate_screen(screen, deep=True)
selection = load_json(screen_root / "selection_lock.json")
validate(selection, "SALIENCE_SELECTION_LOCK")
complete = load_json(screen_root / "screen_complete.json")
validate(complete, "SALIENCE_SCREEN_COMPLETE")

foundation_root = Path(selection["winner_foundation_root"]).resolve()
foundation = load_json(foundation_root / "foundation_spec.json")
validate_foundation_spec(foundation)
lock = authenticate_preparation(foundation, foundation_root)

assert Path(screen["data_root"]).resolve() == data_root
assert selection["screen_sha256"] == screen["content_hash"]
assert selection["winner"] == foundation["candidate"]
assert selection["winner_foundation_sha256"] == foundation["content_hash"]
assert foundation_root == Path(selection["winner_foundation_root"]).resolve()
assert complete["screen_sha256"] == screen["content_hash"]
assert complete["selection_lock_sha256"] == selection["content_hash"]
assert lock["foundation_sha256"] == foundation["content_hash"]
assert selection["final_test_accessed"] is False

print("AUTHENTICATED DZ-FIX MATCHING")
print("candidate:", foundation["candidate"])
print("foundation:", foundation["content_hash"])
print("lock:", lock["content_hash"])
print("selection:", selection["content_hash"])
print("root:", foundation_root)
PY
```

Retain the winner foundation as one atomic unit: its spec, lock, sample and
assignment audits, every assignment payload, and every corresponding report.
Retain the screen spec, runtime profile, selection lock, completion lock, and
task reports. The existing deep validator also authenticates the nonwinning
candidate foundations and bottleneck context as evidence for the selection;
do not delete them while using the current screen contract.

## Using the view cache in a new ladder

The supported cache builder is
`hlt_classification.jetclass2_delphes.salience_cache.prepare_cache`. It joins
the exact compact maps by row identity and constructs views in RAM:

```python
from pathlib import Path

from hlt_classification.data.cache_contracts import load_json
from hlt_classification.jetclass2_delphes.salience_cache import prepare_cache

foundation_root = Path(selected_foundation_root)
foundation = load_json(foundation_root / "foundation_spec.json")

train = prepare_cache(
    foundation,
    data_root=Path(data_root),
    foundation_root=foundation_root,
    role="train",
    coordinate_name="D066",
    workers=runtime_profile["workers"],
    max_ram_bytes=runtime_profile["cache_budgets"]["train"],
)
validation = prepare_cache(
    foundation,
    data_root=Path(data_root),
    foundation_root=foundation_root,
    role="validation",
    coordinate_name="D066",
    workers=runtime_profile["workers"],
    max_ram_bytes=runtime_profile["cache_budgets"]["validation"],
)
```

Valid registered coordinate names use the existing `Uxxx`/`Dxxx` rational
coordinate parser. The cache is RAM-only. Do not serialize dense pairwise
matrices or materialized particle views. `D000` deliberately bypasses offline
data and assignment loading.

For the already registered DIRECT/COARSE/DENSE graph, create a new production
spec and canonical dry ledger from the completed active screen; the canceled
old continuation did not publish an authoritative production directory. For a
different ladder graph, create a new versioned campaign and source/import
contract. That contract must bind:

- the selected foundation spec and lock hashes;
- matcher-spec hash and selected candidate;
- screen spec, selection-lock, and completion hashes;
- inventory and exact `TRAIN_500K` role memberships;
- reader, 17-input schema, capacity/no-trimming policy, and assignment producer
  semantic source identities;
- `persistent_hlt_skeleton_offline_tail_v1` support semantics;
- runtime profile or separately justified new resource evidence;
- absence of final-test access.

A new source commit must not pretend to be the old screen source. Define and
test an explicit immutable import/reuse artifact that authenticates the old
selection and foundation from the new source. Do not weaken the existing
same-source check or copy the candidate name into a new spec without hashes.

## Which queued job another campaign should depend on

For the current execution, use:

```text
afterok:21748725
```

`21748725` is the active screen's completion job. It can succeed only after
authentication, preflight, all four matched fits, and deterministic selection.
Depending directly on `21748721`, `21748722`, or `21748723` would race the
selection step and is invalid. The superseded jobs `21741409` through
`21741416` were canceled and must not be used as parents.

The dependency means "the selected matching is published," not "a consumer's
campaign has been validated." The dependent worker must still deeply validate
the v2 screen, selection lock, completion lock, and selected foundation; create
its own immutable import/campaign specification; and materialize its canonical
dry run before any separately authorized live submission.

The recommended pattern is one small CPU post-selection launcher:

```bash
sbatch --parsable \
  --account=reu-aisocial \
  --partition=debug \
  --qos=qos_tier3 \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=1 \
  --mem=8G \
  --time=01:00:00 \
  --no-requeue \
  --dependency=afterok:21748725 \
  --job-name=MY_LADDER_AFTER_DZFIX_MATCHING \
  /absolute/path/to/source_pinned_post_selection_worker.sh
```

That worker should authenticate the artifacts, create the new campaign and
canonical dry run, and only then perform whatever separately authorized live
submission the new campaign defines. Do not pre-create a campaign against a
selection lock that does not yet exist, and do not attach the dependency only
to its first GPU job while skipping campaign authentication.

Do not permanently hardcode `21748725` into reusable source. For the current
execution, recover it from the active screen ledger:

```bash
python -s - "${SCREEN_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
ledger = json.loads((root / "submission_ledger.json").read_text())
assert ledger["dry_run"] is False
assert len(ledger["jobs"]) == 8
print(ledger["jobs"]["complete"])
PY
```

If that printed job is not `21748725`, stop and reconcile the execution before
submitting a dependent launcher. If it fails, use a source- and ledger-bound
recovery; never change the dependency to `afterany` and never infer completion
from a directory alone.

## Environment and execution rules

- Cluster: SPORC.
- Account: `reu-aisocial`.
- Active selection-screen partition: `debug`, explicitly recorded by the v2
  screen contract and authenticated at runtime.
- Production training partition: `tier3`; the v2 runtime profile deliberately
  keeps `execution_site=tier3` even though `screen_execution_site=debug`.
- QoS: `qos_tier3`.
- GPU: one A100 per fit/reducer.
- Environment: `/home/ryreu/miniconda3/envs/atlas_kd_sporc`.
- Every worker sets `PYTHONNOUSERSITE=1`, clears inherited Python/library
  paths through the common helper, and prepends `${CONDA_PREFIX}/lib`.
- Workers use an absolute pinned `${PROJECT_DIR}`; Slurm spool-relative imports
  are forbidden.
- Resource requests come from the authenticated runtime profile rather than
  being copied from an older snapshot.
- Training is restart-from-zero. There are no rolling optimizer checkpoints.

## What may reuse these assignments

No rematching is needed merely because a future study changes:

- ladder density or coordinates, provided the registered view parser supports
  them;
- immediate-parent teacher graph;
- CE/KD mixture, temperature, random seeds, learning-rate schedule, or pass
  budget;
- reporting, provided population and final-test boundaries remain unchanged.

Fresh matching and selection are required if any of these change:

- dataset bytes or producer version;
- inventory, outer split, or exact role membership;
- HLT/offline particle decoding or native index semantics;
- model input schema, capacity, or truncation policy;
- matcher candidate, constants, objective, or tie hierarchy;
- persistent-HLT endpoint/support semantics;
- treatment of empty/invalid particles or phi wrapping.

An authentication failure is not permission to recompute matches silently in
a training job. Publish a new foundation and, for a new scientific domain, run
the registered candidate screen.

## Storage and retention

Durable matching artifacts are compact identity/offset/index maps and reports.
Pairwise matrices, salience matrices, and particle views remain in RAM. Normal
training retains selected model weights, small reports, and compact class
probability banks; it does not retain optimizer or rolling-resume state.

For future reuse, preserve:

- the raw authenticated dz-fix partial snapshot;
- inventory and split-registry artifacts;
- bottleneck readiness foundation and runtime profile;
- all three candidate foundation directories named and hashed by the active
  `screen_spec.json`;
- `${SCREEN_ROOT}` including all four fit reports/checkpoints and its three
  lock/profile files;
- the active screen command plan, dry ledger, live ledger, task reports, and
  source-pinned v2 specification;
- the selected foundation in full.

Do not preserve only bare assignment NPZ files or only the winning candidate
name. Without reports, parent hashes, locks, and selection evidence, the maps
are not reusable scientific artifacts.

## Final-test and failure rules

- Ordinary matching, caching, training, selection, and validation may access
  only train and validation roles.
- Final-test particles, predictions, and metrics remain sealed.
- Scientific underperformance is a valid result and must not fail a job.
- Missing rows, reused indices, incomplete smaller-side coverage, corrupt
  payloads, nonfinite required values, source drift, parent-hash mismatch, or
  forbidden role access fail closed.
- Cancel or recover only exact campaign-bound job IDs. Never cancel by broad
  name matching.

## Handoff checklist for another ladder

Before queueing, the receiving setup should be able to answer yes to all of
the following:

- Did `21748725`, or the exact `complete` job recovered from the active screen
  ledger,
  complete successfully?
- Were screen completion, selection, selected foundation, and every compact
  assignment payload authenticated?
- Is the dataset exactly the September 18 dz-fixed partial snapshot and the
  split exactly `TRAIN_500K`?
- Does the new source contract bind hashes rather than paths or informal names?
- Does its view builder preserve unmatched HLT tokens and distinguish U100
  from exact-HLT D000?
- Are matching metadata excluded from deployable inputs?
- Are train and validation caches RAM-only and explicitly budgeted?
- Is final test still inaccessible?
- Does the new campaign have its own fresh root, exact pushed commit, complete
  dry run, measured resources, focused tests, and explicit live authorization?

If any answer is no, the new ladder is not yet a valid reuse of this matching
setup.
