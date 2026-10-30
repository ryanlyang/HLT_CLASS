# U100 multiview/readout debug campaign

See the [plan](plans/LUKA_U100_MULTIVIEW_READOUT_PLAN.md) for scientific meaning
and [versioned contracts](contracts/LUKA_U100_MULTIVIEW.md) for evidence boundaries.
The [frozen-readout CE handoff](LUKA_U100_FROZEN_READOUT_CE_HANDOFF.md) explains
`u100_readout_ce`, its code and how to adapt the idea in another setup.
The [retained-predictor handoff](LUKA_U100_RETAINED_PREDICTOR_HANDOFF.md) covers
`u100_use_prediction`, its matched capacity control and the FullStat port boundary.

This is eight fresh HLT student fits and two fresh oracle teachers, with the
existing regular COARSE_U100 reused. The new CONCAT teacher sees the full
untagged physical-particle multiset; FUSION has HLT-primary cross-attention to
native OFFLINE at blocks 2/4/6/8. Students use CE1 and total embedding weight5.
U100-only readout CE, readout KD, retained-predictor, and capacity-control arms
separate the proposed mechanisms. All jobs use SPORC **debug**.

## What must exist

- A clean, pushed worktree containing all files in `source.REQUIRED`.
- Original fixed30 science spec and retained M0HLT/OFFLINE/COARSE_U100 artifacts.
- Completed `luka_u100_direct_ed6d0af4_tier3_r1/campaign_spec.json`, original
  source worktree and its retained receipts, bank manifest, readback proof,
  selected model, reports and predictions. The old disposable U100 TRAIN array
  may be retired: it is not reused or needed for this result-only import.
- Read-only historical ALLJETS prepared caches and raw paired source files.
  D000/U100 payloads must be present. SPEC/v2 can reconstruct missing native
  OFFLINE payloads in RAM, with exact historical block hashes and input
  fingerprints; metadata/manifests and original raw files must still exist.
- At least four personally available persistent GiB plus filesystem headroom.
  Datasets and Fresh_check are never modified or copied.

## Two explicit submission stages

The plan requires new-architecture measurements. Existing U100 timing cannot
establish that a full-support union or fusion fits on debug. Therefore the
first helper invocation queues **one real acceptance job**, not the science
fits. It runs full-support stress and two-epoch miniature fits for every new
graph through the production code, then publishes measured science walltimes.
It also verifies the original U100 classifier and target/readout round trip.

Once that succeeds, the second invocation prints a concrete measured plan for
review and queues the science DAG. Nothing is automatically submitted while
waiting for acceptance. No scientific accuracy threshold gates any arm.

On SPORC, after committing/pushing the new files locally:

```bash
bash <<'BASH'
set -euo pipefail
read -r -p 'Pushed 40-character commit: ' COMMIT </dev/tty
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
MAIN=/home/ryreu/atlas/HLT_Classification
PROJECT="${MAIN}_u100multi_${COMMIT:0:8}"
git -C "${MAIN}" fetch origin main
git -C "${MAIN}" merge-base --is-ancestor "${COMMIT}" origin/main
if [[ -e "${PROJECT}" ]]; then
    test "$(git -C "${PROJECT}" rev-parse HEAD)" = "${COMMIT}"
    test -z "$(git -C "${PROJECT}" status --porcelain --untracked-files=all)"
else
    git -C "${MAIN}" worktree add --detach "${PROJECT}" "${COMMIT}"
fi
bash "${PROJECT}/scripts/queue_luka_u100_multiview.sh" accept
BASH
```

The helper prints the exact second command. After successful acceptance, in
that same worktree:

```bash
bash scripts/queue_luka_u100_multiview.sh science
```

Each stage requests its printed plan hash and `YES` only after preparing the
reviewable plan. Submission runs under `nohup`. After a completed exact
submission ledger exists, Slurm owns those jobs and logging off is safe.
Inspect `submit-*.log` if submission fails; do not rerun partial claims blindly.

Acceptance requests one A100, six CPUs, 120000M host RAM and twelve hours.
Scientific walltimes come from conservative real measurements, capped at 24h;
oversized projections require resource review, never silent cropping or batch
changes. The installed environment and GPU must match acceptance at execution.
This is an executable admission path, not a claim that remote acceptance has
already passed on your current data/hardware.

## Dependencies and monitoring

Science submits twelve jobs: two teacher fits (each exports its selected TRAIN
targets), one U100 target-reuse validation, eight students, and one report.
The `export_U100` task reuses the exact original-teacher bank made during
acceptance; it does not repeat teacher inference or copy that bank. Acceptance
is separate.
No student waits for another student's success. U100-only students depend on
U100 export; new-view students additionally wait for their required teachers.
The capacity control is independent. The scheduler decides actual start order;
submission order does not guarantee priority over other users or campaigns.

```bash
squeue --me -o "%.18i %.12P %.40j %.2t %.10M %R"
```

Every fit logs epoch/batch percent, validation AUC, running best and best epoch.
The report waits for terminal attempts and identifies incomplete rows honestly.
A failed new teacher does not invalidate U100-only students. Cancel only exact
IDs in this campaign's submission ledger.

For partial or completed results, with the same Conda environment activated:

```bash
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
python -u -B -s scripts/luka_u100_multiview.py results --spec /absolute/campaign_spec.json
```

Read the saved `training_report.json` for per-class metrics, QCD rejection,
gradient diagnostics and inference costs. Recovery always uses historical
M0HLT/OFFLINE anchors; new oracle AUCs do not change the denominator.
The validation subset called original50k is overlapping and report-only.

## Code and extension boundaries

All runtime additions are under
[`luka_u100_multiview`](../src/hlt_classification/luka_u100_multiview/__init__.py).
`contracts.py` registers arms/weights; `views.py` builds the source-blind union;
`teachers.py` taps exact classifier inputs; `model.py` implements student heads
and frozen readouts; `targets.py` handles compact TRAIN banks; `training.py`
contains the shared fixed30 loop. `acceptance.py`, `campaign.py`, `submission.py`
and `worker.py` implement measurements, immutable plans, exact journals and
validated publication. The CLI and Slurm scripts are thin adapters.

Selected files: `fit/selected_full.pt` contains all training heads;
`fit/selected.pt` is the deployed graph. Never strip the retained predictor.
Native-arm checkpoints load into `DelphesParticleTransformer`; retained/control
checkpoints load into `Student(arm)`, with `capture=False`, then `.eval()`.
No teacher or bank is needed for either deployed graph.

There is no resume mode. Preserve incomplete outputs. New loss weights,
population, views, seeds or resource policies require new registered semantics
and a fresh campaign; do not edit an existing `campaign_spec.json`.

## Recovery from acceptance job 21983825

The user-supplied traceback identifies a missing file at
`luka_alljets_9e591c7a_r1/prepared/inputs/train/OFFLINE/0000.npz`.
The CONCAT two-epoch miniature completed before FUSION input loading failed.
This is a missing-cache failure, not a scientific metric failure or a reported
OOM. The earlier CONCAT raw read shows those selected source files were
accessible then; corrected execution still verifies them afresh.

`offline_cache.py` now reconstructs absent OFFLINE blocks in memory using the
historical raw reader, physical conventions and feature transform. It requires
exact original serialized byte count/SHA256 and the complete prepared-input
fingerprint. Present corruption still fails. It writes no replacement cache
files and leaves datasets, Fresh_check and old campaigns untouched.

Commit/push the scoped repair, create a new clean worktree at that commit, and
use the `accept` block above. The commit-derived root will be new. Keep
`luka_u100_multiview_de79c729_debug_r1`, its claims and logs intact. Do not requeue
21983825, edit its spec, or run the corrected source against its SPEC/v1.
The repaired registration uses SPEC/v2; its scientific RECIPE remains v1.

Some U100/CONCAT technical outputs may exist in the failed root, but the full
acceptance resource evidence was never published. The current tool has no
authenticated partial-acceptance resume path, so the fresh acceptance repeats
the technical work. It does not retrain a completed scientific student.
An old automatic-submission watcher is tied to the failed job/root; do not
assume it will submit the new campaign. Built-in staging remains `accept`,
then review and `science` after a valid new acceptance receipt.

## Retire compact banks after success

After all tasks, including the report, have valid receipts, review the exact
five newly owned acceptance/science target payloads:

```bash
python -B -s scripts/luka_u100_multiview.py retire --spec /absolute/campaign_spec.json
```

Repeat with `--execute --plan-hash <printed-content_hash>` only to approve those
permanent deletions. The immutable retirement inventory permits later results
checks. It retains manifests, predictions, reports and selected models and
never deletes imported banks, datasets or Fresh_check. Incomplete/failed
consumers block retirement. Automatic deletion is not enabled.
