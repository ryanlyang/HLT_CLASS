# Queue and work with teacher_m16 particle prediction on Hydra

This campaign trains12 fresh native HLT students, all with CE+U100-w5. Ten add
removable heads predicting the frozen privileged teacher's full particle states;
two are matched native U100-w5 controls. See the [plan](plans/LUKA_PARTICLE_TEACHER_HYDRA_PLAN.md)
for the exact grid and [contract](contracts/LUKA_PARTICLE_TEACHER_HYDRA.md) for
artifact rules. No new teacher training or SPORC transfer is needed.

## Existing prerequisites

Under the physical Brown home directory, `hlt_u100_pairs` already contains:

- `code_65952ed3`, the main Git clone used to fetch new commits;
- `runs/cross_message_199928da_3693bcda_a6000_r1/campaign_spec.json`;
- its original pinned worktree, completed setup144840 and teacher_m16/job145410;
- original input bundle, adapter/U100 encoder deltas and `environment_pair_hydra_v1/venv`.

These are authenticated, including original-source replay. Do not delete them.
Reserve4GiB for outputs plus1GiB filesystem headroom. Dense teacher targets use
allocation RAM and are rebuilt once per supervised fit; they are not saved to
home. Shared setup checks actual peak memory and runtime before releasing fits.

## Commit and push on Windows

The following stages only this campaign. Review pre-existing staged files first.
Leave unrelated runtime changes, including the earlier shared-message CPU fix,
out of this additive commit. The new campaign contains its own corrected guard.

```powershell
Set-Location 'C:\Users\22rya\ComputerScience\CERN\HLT_Classification'
git diff --cached --stat
git add -- src/hlt_classification/luka_particle_teacher_hydra scripts/luka_particle_teacher_hydra.py scripts/queue_luka_particle_teacher_hydra.sh sbatch/run_luka_particle_teacher_hydra.sh tests/test_luka_particle_teacher_hydra.py docs/plans/LUKA_PARTICLE_TEACHER_HYDRA_PLAN.md docs/contracts/LUKA_PARTICLE_TEACHER_HYDRA.md docs/LUKA_PARTICLE_TEACHER_HYDRA_RUNBOOK.md docs/LUKA_PARTICLE_TEACHER_HYDRA_VALIDATION.md
if ($LASTEXITCODE -ne 0) { throw 'Staging failed' }
git diff --cached --check
if ($LASTEXITCODE -ne 0) { throw 'Check staged whitespace' }
git diff --cached --stat
git commit -m 'Add HLT-aligned teacher particle prediction campaign on Hydra'
if ($LASTEXITCODE -ne 0) { throw 'Commit failed' }
git push origin main
if ($LASTEXITCODE -ne 0) { throw 'Push failed' }
git rev-parse HEAD
```

README/HANDOFF/source-map integration notes contain prior working-tree edits;
review them separately before staging. Do not use a broad `git add .`.

## Queue on Brown

Use the full40-character commit printed above. This makes a clean detached
worktree and a fresh root, reviews the resource envelope, then asks for `YES`
to submit setup and a durable CPU release. No login-host watcher is needed.

```bash
bash <<'BASH'
set -euo pipefail
BASE="$(cd "${HOME}" && pwd -P)/hlt_u100_pairs"
MAIN="${BASE}/code_65952ed3"
read -r -p 'New pushed 40-character particle-teacher commit: ' COMMIT </dev/tty
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
git -C "${MAIN}" fetch origin main
git -C "${MAIN}" merge-base --is-ancestor "${COMMIT}" origin/main
PROJECT="${BASE}/code_particle_teacher_${COMMIT:0:8}"
if [[ -e "${PROJECT}" ]]; then
    [[ "$(git -C "${PROJECT}" rev-parse HEAD)" == "${COMMIT}" ]]
    [[ -z "$(git -C "${PROJECT}" status --porcelain --untracked-files=all)" ]]
else
    git -C "${MAIN}" worktree add --detach "${PROJECT}" "${COMMIT}"
fi
bash "${PROJECT}/scripts/queue_luka_particle_teacher_hydra.sh"
BASH
```

Wait for both bootstrap job IDs before logging off. Prefix is `lukapt_`; root is
`runs/particle_teacher_COMMIT8_a6000_r1`. Setup builds real TRAIN teacher targets
and runs warmed full/partial production miniatures for all12 arms. Successful
release submits12 independent fits plus an afterany CPU report. One RTX A6000,
6CPUs,32000MiB per GPU job, no node pin or concurrency lanes, measured walltime
up to48h. This is15 jobs total,12 scientific fits. Existing campaigns are untouched.

```bash
squeue --me -o "%.18i %.38j %.12T %.12l %R"
```

Poor recovery does not stop the campaign. Technical rejection prints diagnostics
and preserves outputs. Do not edit immutable specs or delete claims to retry.

## Validation results and live progress

Replace COMMIT8 with the eight-character source prefix. `progress` prints each
published epoch's AUC/recovery, same-seed control and historical U100-w5, plus
best-so-far values. `results` authenticates completed selected checkpoints using
saved probabilities. These are read-only; neither requires a GPU or the report job.

```bash
bash <<'BASH'
set -euo pipefail
BASE="$(cd "${HOME}" && pwd -P)/hlt_u100_pairs"
SPEC="${BASE}/runs/particle_teacher_COMMIT8_a6000_r1/campaign_spec.json"
PROJECT="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["project_dir"])' "${SPEC}")"
ENV_DIR="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["environment_dir"])' "${SPEC}")"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="${ENV_DIR}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
"${ENV_DIR}/bin/python" -B -s "${PROJECT}/scripts/luka_particle_teacher_hydra.py" progress --spec "${SPEC}"
"${ENV_DIR}/bin/python" -B -s "${PROJECT}/scripts/luka_particle_teacher_hydra.py" results --spec "${SPEC}"
BASH
```

Recovery uses historical M0HLT/OFFLINE anchors. Full validation is exploratory;
original50k overlaps it and is report-only. Both controls still have U100-w5.
Use same-seed native-control deltas to assess additional particle supervision.

## Release recovery

If setup completed and the release job failed, inspect its exact log and the
`submission_science` journal. With the variables/exports above set, run:

```bash
"${ENV_DIR}/bin/python" -B -s "${PROJECT}/scripts/luka_particle_teacher_hydra.py" finish --spec "${SPEC}"
```

This verifies original authorization, setup receipt and measured evidence, then
performs the exact science dry/live submission. Complete submission is idempotent.
Active/partial claims fail closed for reconciliation; never remove a claim.

## Code and adaptation

All new runtime lives in `src/hlt_classification/luka_particle_teacher_hydra/`:

| File | Responsibility |
|---|---|
| contracts.py | Twelve arms, weights, schedule and scientific identities |
| model.py | Native student, final-token prediction heads, full/centered loss, stripped deployment |
| inputs.py | Authenticated teacher/encoder, active oracle extraction, TRAIN normalization and target fingerprints |
| training.py | Fresh30epoch fits, component diagnostics, native validation and replay |
| campaign.py | New source/specs and original-worktree teacher/setup proofs |
| resources.py | Warmup, fresh measured miniatures and resource-only acceptance |
| submission.py / worker.py | Exact DAGs, CPU release/recovery, ownership and publication |
| reporting.py | Saved results and epoch progress |

The CLI, queue script and sbatch wrapper are thin. Import scientific functions
from the package when adapting, rather than copying only the launcher. Changing
targets, normalization, weights, population or inference requires a new versioned
registration and fresh outputs. A teacher hook must wrap `teacher.oracle(...)`;
the ordinary native helper is only appropriate for the frozen U100 encoder.

This is queue-ready source with local validation. Real Hydra setup still runs
before science; no new remote fit or performance improvement is claimed.
