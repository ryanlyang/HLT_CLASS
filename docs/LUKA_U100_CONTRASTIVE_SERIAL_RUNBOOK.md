# Queue the serial contrastive continuation on SPORC/debug

This is the [authorized amendment](plans/LUKA_U100_CONTRASTIVE_SERIAL_PLAN.md)
to the running20-arm identification campaign. It preserves completed results
and target banks and replaces only pending old campaign jobs. Running old fits
finish normally so their best checkpoint is not lost. The new graph runs one
fit at a time, minimum15 epochs, patience5, maximum30, unchanged30-epoch LR.

There is **no new preflight, setup or target export**. Original successful A100
miniatures and full30 measured bounds are authenticated and reused. Independent
particle/pair correction work is not changed. Do not manually cancel KINMIX
before it has published a completed receipt.

## Commit and push locally

From PowerShell in the repository, review other staged work first. These are
the additive campaign files; do not stage unrelated README/HANDOFF changes.

```powershell
git diff --cached --stat
git add -- src/hlt_classification/luka_u100_contrastive_serial `
  scripts/luka_u100_contrastive_serial.py scripts/queue_luka_u100_contrastive_serial.sh `
  sbatch/run_luka_u100_contrastive_serial.sh tests/test_luka_u100_contrastive_serial.py `
  docs/plans/LUKA_U100_CONTRASTIVE_SERIAL_PLAN.md `
  docs/contracts/LUKA_U100_CONTRASTIVE_SERIAL.md docs/LUKA_U100_CONTRASTIVE_SERIAL_RUNBOOK.md
git diff --cached --check
if ($LASTEXITCODE -ne 0) { throw 'Review staged whitespace.' }
git diff --cached --stat
git commit -m "Fix serial campaign pending-job cancellation and verification"
if ($LASTEXITCODE -ne 0) { throw 'Commit failed.' }
git push origin main
if ($LASTEXITCODE -ne 0) { throw 'Push failed.' }
git rev-parse HEAD
```

The source guard freezes `c08f6df603d87a452a4814a9e954421d8731a4a7`, preserving
both original contrastive and particle/pair-correction runtimes. Subsequent
runtime changes must be limited to this additive campaign. Preserve old source
worktrees and target directories because the continuation references them.

## Launch on SPORC

```bash
bash <<'BASH'
set -euo pipefail
read -r -p 'New pushed 40-character commit: ' COMMIT </dev/tty
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
MAIN=/home/ryreu/atlas/HLT_Classification
PROJECT="${MAIN}_contrast_serial_${COMMIT:0:8}"
git -C "${MAIN}" fetch origin main
git -C "${MAIN}" merge-base --is-ancestor "${COMMIT}" origin/main
if [[ -e "${PROJECT}" ]]; then
    test "$(git -C "${PROJECT}" rev-parse HEAD)" = "${COMMIT}"
    test -z "$(git -C "${PROJECT}" status --porcelain --untracked-files=all)"
else
    git -C "${MAIN}" worktree add --detach "${PROJECT}" "${COMMIT}"
fi
bash "${PROJECT}/scripts/queue_luka_u100_contrastive_serial.sh"
BASH
```

The default parent is the successful
`luka_u100_contrastive_431341db_debug_r1/campaign_spec.json`; this is not either
failed setup root. An explicit alternate location can be passed with
`--parent /absolute/path/campaign_spec.json`, but the authenticated source and
registered path still must match431341db.

The wrapper requests at least8 personally available GiB, prints the read-only
review and PLAN_HASH, then asks for that hash and YES. This is the existing
repository exact-source/full-dry-review requirement, not an extra preflight.
The reviewed transition includes pending-only exact-ID cancellation and fresh
attempts for terminal incomplete rows in a separate root, with partials retained.

After the detached PID/log is printed, watch that log. Once it says
`Pending-job transition verified`, the dispatcher is running
and can wait after you log out. It does not cancel running jobs; a fit that
starts during the cancellation race is allowed to finish and reused if valid.
After terminal accounting and replay, it prints
`All remaining fits and the report are queued in one debug lane`.
Thereafter Slurm owns the whole sequence. No new manual queue step is needed.

Original `JobHeldUser` pending jobs are replaced as part of the explicit review;
new registrations are not held. Only this campaign's ledger IDs are affected.
The dispatcher may survive logout but not a submit-host reboot. If submission
fails, preserve request, claims, log and journals; do not rerun blindly inside
the existing root or remove claims. No automatic recovery or deletion occurs.

### Recovery from the f8f61ade cancellation error

The reported `submit.2OQbFi.log` stopped at `scancel`, before new scientific
submission. The earlier `sbatch: Job ... to start ...` lines came from
`check_site` running `sbatch --test-only`; they were resource checks, not accepted
replacement jobs. Some old pending jobs might have been cancelled before the
nonzero return. The log does not establish the exact reason for that return.

Commit/push this repair and use the launch block above with the **new commit**.
It creates a distinct worktree and output root, retaining the failed
`luka_u100_contrastive_serial_f8f61ade_debug_r1` request/claim/log for inspection.
Do not delete its claim or rerun its old dispatcher. The parent stays the
original `431341db` campaign: targets and completed fits are reused, including
any that finished during the failed transition. Do not manually cancel running
old fits, since their selected checkpoint is only published at completion.

The corrected dispatcher sends only controller-listed pending original IDs,
uses `--user=<owner>` and controller-side `--ctld --state=PENDING`, and saves
`transition_claim/cancellation.json` plus `cancellation_check.json`. No pending
jobs means no `scancel` call. Even a nonzero return can proceed if a fresh
controller query proves none remain pending; all original allocations must
still reach terminal accounting and completed receipts must authenticate.
Remaining pending jobs, query failures or unverified completions stop submission
with preserved diagnostics. No broad cancellation or silent retry is used.

This recovery applies to the reported cancellation-stage failure. If another
attempt reached live `sbatch`, inspect its exact submission journal first;
creating another campaign then could duplicate work.

```bash
squeue --me -o "%.18i %.12P %.40j %.2t %.10M %.12l %R"
```

New prefix: `lukaos_`. With the five original attempted fits completed, there
are15 new students and one report. Other original completions are reused too.
Every next job has an afterany dependency on its immediate predecessor, so a
failed fit does not block independent later rows. This is a one-active-fit
limit for this campaign; unrelated user jobs are untouched.

## Saved results

Use the new root printed by the queue helper. `campaign_spec.json` appears
after the old allocations finish; while waiting, inspect `request.json` and
the submit log instead.

```bash
bash <<'BASH'
set -euo pipefail
read -r -p 'New serial campaign root: ' ROOT </dev/tty
SPEC="${ROOT}/campaign_spec.json"
source /home/ryreu/miniconda3/etc/profile.d/conda.sh
conda activate /home/ryreu/miniconda3/envs/atlas_kd_sporc
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
PROJECT="$(python -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["project_dir"])' "${SPEC}")"
export PYTHONPATH="${PROJECT}/src"
python -u -B -s "${PROJECT}/scripts/luka_u100_contrastive_serial.py" results --spec "${SPEC}"
BASH
```

Results authenticate saved predictions without new inference. They label
`original30` reused fits separately from `patience5` new fits, print actual
epochs and selected epoch, and retain stop reasons in JSON. Missing completed
receipts remain pending/incomplete, not invented results. Original50k overlaps
full validation. No final test is opened.

## Code map

- [contracts.py](../src/hlt_classification/luka_u100_contrastive_serial/contracts.py): priority order, unchanged20 rows, policy replay.
- [training.py](../src/hlt_classification/luka_u100_contrastive_serial/training.py): original batch updates, per-best checkpoints, successful early stop, report/reload validation.
- [source.py](../src/hlt_classification/luka_u100_contrastive_serial/source.py): original-source authentication and external references.
- [campaign.py](../src/hlt_classification/luka_u100_contrastive_serial/campaign.py): review request, terminal inventory and new spec.
- [submission.py](../src/hlt_classification/luka_u100_contrastive_serial/submission.py): pending-only transition, detached wait, serial exact DAG.
- [worker.py](../src/hlt_classification/luka_u100_contrastive_serial/worker.py): native training and combined historical/new results.
- [tests](../tests/test_luka_u100_contrastive_serial.py): policy, CPU/CUDA parity, durable state and transition failures.

No scientific improvement is assumed. Local tests and historical resource
evidence do not imply this new campaign has already run remotely.
