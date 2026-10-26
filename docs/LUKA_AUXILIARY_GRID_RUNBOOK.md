# Twenty DE000 auxiliary-loss fits on debug

Authority: [plan](plans/LUKA_AUXILIARY_GRID_PLAN.md) and
[v1 contract](contracts/LUKA_AUXILIARY_GRID.md). This is a trial-size follow-up,
not FullStat and not a new ladder. It uses the completed c50 teachers and a
fresh HLT-only DE000 model for every row. Nothing is submitted by installation.

## Registered experiment

All runs use main-head CE coefficient 0.5, four auxiliary logit heads and four
embedding-prediction heads. KD uses temperature 2 and T-squared exactly once.
K and E below are the TOTAL coefficients, divided equally among the four
teachers: OFFLINE and c50 UE100, DE066, DE033. Relational supervision uses the
same teacher targets with 2,048 TRAIN anchors, self masking, temperature 0.2
and a two-epoch linear coefficient ramp to 1. It is not additional input data.

Each row is a pair: relation enabled and disabled.

| Wave | Total auxiliary KD K | Total embedding E |
| --- | --- | --- |
| 1 | 5 | 10 |
| 2 | 5 | 20 |
| 3 | 2.5 | 10 |
| 4 | 2.5 | 20 |
| 5 | 1 | 10 |
| 6 | 1 | 20 |
| 7 | 5 | 40 |
| 8 | 2.5 | 40 |
| 9 | 1 | 40 |
| 10 | 10 | 20 |

The ordering is a hypothesis, not a prediction of measured performance. Both
fits in a wave must complete before the next wave becomes eligible; the order
within each pair is up to Slurm. This also caps this study at two concurrent
fits. Technical failures block later waves. Poor but finite scientific results
do not: every successful fit runs all 30 epochs without quality thresholds.

Matched initialization, data ordering, batch 128, AdamW and the fixed30 LR
schedule are unchanged. 299,234 TRAIN / 74,895 validation jets. Checkpoints
are selected using the full validation main-head AUC and existing tie rules.
Original50k is an overlapping report-only subset; there is NO independent test.
This is a one-seed exploratory screen, not evidence that its best row will
generalize. Auxiliary heads never enter saved native HLT inference checkpoints.

## Jobs, resources and reuse

22 jobs are submitted together: one shared targets job, twenty fits, one CPU
report. There is no separate preflight or login polling controller. The targets
job contains three disposable three-update checks (plain K5/E10, relational
K5/E40 and relational K10/E20), including native parity, main-head gradient
isolation and maximum TRAIN/validation support. Science reinitializes afterward.

GPU jobs use debug/A100, 6 CPUs, 90,000 MiB and at least 8 hours (or the larger
inherited c50 allowance, capped at 24 hours). Report uses 1 CPU, 16 GiB, 2 hours.
The original installed environment and c50 acceptance are authenticated in their
pinned source. New combined-objective full throughput has not been measured on
SPORC; local tests are not remote resource certification.

The helper defaults to these exact trial campaigns:

- Parent: `luka_trial_hybrid_72490a9d_debug_r1/campaign_spec.json`.
- K5/E5 reference and optional bank source:
  `luka_auxlogit_strong_234b029e_debug_r1/campaign_spec.json`.

Both are under `/home/ryreu/atlas/HLT_Classification/checkpoints`.
The latter must have an authenticated completed k5 fit, not just a log. Intact
four-teacher banks are checksum-checked, copied into newly owned directories
and semantically checked against TRAIN identities/labels. No hardlinks,
symlinks, or edits to the original campaign. If its normal retirement is
authenticated, regenerate all four banks once from the exact retained teachers.
Missing/corrupt payloads without valid retirement do NOT silently trigger export.
A concurrent old cleanup can fail copying; do not blindly retry partial work.

Reserve at least 4 GiB of genuinely available persistent quota for this study.
All twenty fits share the new banks. After all fits and saved results validate,
the report deletes only the four newly owned TRAIN payloads, with a retirement
record and no backup. Selected models, predictions, histories, manifests,
normalization moments, source and reports remain. Do not manually remove current
or parent files. Fresh_check, datasets, and unrelated jobs are never modified.

## Commit and push locally

Review only this implementation's files. Preserve unrelated staged/worktree
changes; do not use `git add .`, force-push, reset, or clean. New required paths:

```powershell
git add -- src/hlt_classification/luka_auxiliary_grid tests/test_luka_auxiliary_grid.py scripts/luka_auxiliary_grid.py scripts/queue_luka_auxiliary_grid.sh sbatch/run_luka_auxiliary_grid.sh docs/LUKA_AUXILIARY_GRID_RUNBOOK.md docs/contracts/LUKA_AUXILIARY_GRID.md docs/plans/LUKA_AUXILIARY_GRID_PLAN.md
if ($LASTEXITCODE -ne 0) { throw "Staging failed" }
git diff --cached --check
if ($LASTEXITCODE -ne 0) { throw "Check staged whitespace" }
git diff --cached --stat
```

Review/stage the relevant HANDOFF and LEGACY_SOURCE_MAP hunks separately if
those shared files contain other work. After reviewing the complete staged
diff, commit and push; stop on non-fast-forward, never force-push:

```powershell
git commit -m "Add twenty matched DE000 KD embedding relational fits"
if ($LASTEXITCODE -ne 0) { throw "Commit failed; inspect status" }
git push origin HEAD:main
if ($LASTEXITCODE -ne 0) { throw "Push failed; do not force-push" }
git rev-parse HEAD
```

## Queue from sporcsubmit

Use the newly pushed full commit, NOT a prior campaign commit. The helper
creates one immutable root and prints the entire 22-job dry plan before asking
for its 64-character hash and explicit authorization. It does not cancel or
change existing jobs. Nothing in this runbook has already queued jobs for you.

```bash
bash <<'BASH'
set -euo pipefail
trap 'echo "Stopped; preserve logs and journals. Do not blindly retry." >&2' ERR
read -r -p 'New pushed 40-character commit: ' COMMIT </dev/tty
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
MAIN=/home/ryreu/atlas/HLT_Classification
PROJECT="${MAIN}_auxgrid_${COMMIT:0:8}"
git -C "${MAIN}" fetch origin main
git -C "${MAIN}" merge-base --is-ancestor "${COMMIT}" origin/main
if [[ -e "${PROJECT}" ]]; then
    test "$(git -C "${PROJECT}" rev-parse HEAD)" = "${COMMIT}"
    test -z "$(git -C "${PROJECT}" status --porcelain --untracked-files=all)"
else
    git -C "${MAIN}" worktree add --detach "${PROJECT}" "${COMMIT}"
fi
bash "${PROJECT}/scripts/queue_luka_auxiliary_grid.sh"
BASH
```

Copy the actual hexadecimal PLAN_HASH value, not the word `PLAN_HASH`.
Confirm only if all 22 jobs, resources, paths and storage are correct. The helper
prints a submission log. Follow it until the exact journal records all 22 jobs;
a helper PID alone does not prove successful submission. All dependencies are
Slurm-owned afterward. An ambiguous sbatch response stops without retry; keep
the journal and inspect the exact accepted IDs before any recovery action.

## Watch and report

```bash
squeue --me -o '%.18i %.12P %.40j %.2t %.10M %R'
```

Epoch logs are `<root>/<arm>-<job>.out`, e.g. `k5_e10_rel-<job>.out`.
They print AUC, best epoch, raw CE, weighted KD/embedding, relation and beta.
Live values are provisional; use authenticated results for completed fits.

In the same pinned environment, run:

```bash
python -u -B -s "${PROJECT}/scripts/luka_auxiliary_grid.py" results \
    --spec "${ROOT}/campaign_spec.json"
```

Here `ROOT` is the printed `luka_auxgrid_<commit8>_debug_r1` directory; restore
the exact `PROJECT` and activate `atlas_kd_sporc` with its `lib` prepended to
`LD_LIBRARY_PATH`, `PYTHONNOUSERSITE=1`, and `PYTHONPATH=${PROJECT}/src`.
Results include M0HLT, OFFLINE, single-OFFLINE-w5, U100-w5, c50_DE000 and K5/E5;
each completed arm shows AUC gap recovery and delta versus K5/E5. Missing fits
are marked pending, never treated as zero. The report survives normal bank
retirement. It does not require old teacher weights for retained reporting.

## Implementation

`src/hlt_classification/luka_auxiliary_grid/` owns registration, model heads,
losses, bank acquisition, source replay, worker receipts, reporting and the DAG.
CLI/sbatch wrappers remain thin. `tests/test_luka_auxiliary_grid.py` exercises
the grid and production workers. See HANDOFF for exact validation evidence and
LEGACY_SOURCE_MAP for donors. No previous contract or scientific runtime is
redefined by this package.
