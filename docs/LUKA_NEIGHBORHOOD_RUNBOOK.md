# Luka offline-neighborhood screen: direct debug

Read the [plan](plans/LUKA_NEIGHBORHOOD_PLAN.md),
[contract](contracts/LUKA_NEIGHBORHOOD.md) and
[scientific handoff](LUKA_OFFLINE_NEIGHBORHOOD_STRATEGY_HANDOFF.md).

Exactly **16 jobs**, all on debug: shared neighborhood preparation, fourteen
fresh HLT fits, report. No standalone preflight, new teacher or baseline fit.
Prepared neighborhoods and bounded safety checks precede dependent fits.
Two GPU lanes; smooth_w3/smooth_w10 first, annular and HLT controls next.
This is an exploratory comparison, not a promise of recovery improvement.

## Local release

Commit only this implementation and its documentation after reviewing unrelated
work. Required files are listed by `luka_neighborhood.source.REQUIRED`.
Include the new strategy handoff and append-only HANDOFF/source-map entries.
Do not stage all untracked temporary directories. Run focused tests and
`git diff --cached --check`, commit, then push normally. If push is rejected,
integrate/review remote changes without force-push and recheck the source guard.
Use the resulting **full pushed 40-character commit**, not an old depth commit.

No local test claims a genuine new SPORC/A100 measurement. Inherited source,
environment and acceptance must replay on SPORC; new objective checks execute
inside the registered jobs, with no separate preflight allocation.

## SPORC queue

The original depth campaign's shared targets and repeated baseline must be
complete and retained. Other recovery arms need not finish. Leave their jobs,
outputs, source worktrees, Fresh_check and datasets alone.

Run on `sporcsubmit`:

```bash
bash <<'BASH'
set -euo pipefail
trap 'echo "Stopped; preserve logs/journals. Do not blindly retry." >&2' ERR
read -r -p 'New pushed 40-character commit: ' COMMIT </dev/tty
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
MAIN=/home/ryreu/atlas/HLT_Classification
PROJECT="${MAIN}_neighborhood_${COMMIT:0:8}"
git -C "${MAIN}" fetch origin main
git -C "${MAIN}" merge-base --is-ancestor "${COMMIT}" origin/main
if [[ -e "${PROJECT}" ]]; then
    test "$(git -C "${PROJECT}" rev-parse HEAD)" = "${COMMIT}"
    test -z "$(git -C "${PROJECT}" status --porcelain --untracked-files=all)"
else
    git -C "${MAIN}" worktree add --detach "${PROJECT}" "${COMMIT}"
fi
bash "${PROJECT}/scripts/queue_luka_neighborhood.sh" \
    "${MAIN}/checkpoints/luka_depth_9e90b7de_debug_r1/campaign_spec.json"
BASH
```

Declare **personally available** persistent space, allowing for other jobs.
Four GiB is only the minimum declaration, not a promise that four suffices.
Preparation counts actual TRAIN HLT particles before allocating: three raw banks
need432 bytes/particle, plus indexes, outputs and2GiB reserve. Insufficient
quota stops before scientific fits; do not delete dependencies to bypass it.
Borrowed four-teacher banks are not copied or regenerated.

The helper creates a fresh immutable study, prints the entire dry plan and
PLAN_HASH, and asks for that hash and `YES`. It then starts a nohup submitter.
After that point SSH disconnect does not stop that process, but login-node
reboot/site termination can. **Confirm the complete16-job ledger in its log**
before assuming all jobs are accepted. Once accepted, Slurm owns the graph and
no login helper is needed. Never repeat submission after ambiguous output;
preserve the exclusive claim and journal for exact-ID review.

No live commands here cancel other jobs or change their priorities. Submission
order/two lanes prioritize our choices, not all users' cluster scheduling.

## Progress and results

```bash
squeue --me -o "%.18i %.12P %.42j %.2t %.10M %R"
```

Job names start `lukanb_`. Each fit logs completed epoch AUC, best epoch/AUC,
delta/recovery relative to the fixed references, training seconds and current
epoch percentage (every128 batches); all30 epochs run. Saved histories include
accuracy, raw/weighted loss families and all36 category/scale error diagnostics.

With the exact printed study root and pinned project, in the established conda
environment, use:

```bash
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PROJECT}/src"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
python -u -B -s "${PROJECT}/scripts/luka_neighborhood.py" results \
    --spec "${ROOT}/campaign_spec.json"
```

`results` authenticates saved completed outputs and reports unfinished rows.
It never selects on original50k or auxiliary loss. Recovery is relative to
original M0HLT/OFFLINE; the primary AUC delta uses the repeated baseline.
Final reporting can succeed with incomplete rows; that does not authorize
retirement or declare incomplete fits successful.

## Optional space recovery after all results validate

There is no automatic cleanup. First dry-review:

```bash
python -u -B -s "${PROJECT}/scripts/luka_neighborhood.py" retirement \
    --spec "${ROOT}/campaign_spec.json"
```

Only after reviewing the exact three files and the loss of rerun targets,
repeat with `--execute --plan-hash THE_PRINTED_HASH` and
`--authorization DELETE_ONLY_COMPLETED_NEIGHBORHOOD_PAYLOADS_V1`.
Deletion is permanent with no backup. Manifest, index/moments, selected native
models, epoch histories, predictions, receipts and results remain. This command
cannot delete old banks, raw particles, Fresh_check or any source worktree.
