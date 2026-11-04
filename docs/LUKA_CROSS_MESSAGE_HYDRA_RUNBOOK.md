# Queue the cross-attention message campaign on Brown Hydra

This campaign trains **two privileged teachers and twelve fresh HLT students**.
Every student keeps CE+U100 embedding weight5. The recommended first comparison
is `m32_both_l0p3_s0` against `m32_ce_s0`: same HLT architecture, with and without
message supervision. The existing native U100-w5 reference is reused.

Read the [plan](plans/LUKA_CROSS_MESSAGE_HYDRA_PLAN.md) for the full ordered grid
and [contract](contracts/LUKA_CROSS_MESSAGE_HYDRA.md) for artifact rules. Runtime
lives in `src/hlt_classification/luka_cross_message_hydra/`:

| File | Responsibility |
|---|---|
| `contracts.py` | Immutable grid, objective, seeds and dependencies |
| `model.py` | U100 cross-attention teacher, HLT predictor, block4 injection |
| `inputs.py` | Small encoder export, authenticated inputs, bounded token cache |
| `training.py` | Shared training/evaluation, target RMS, selection, reload |
| `resources.py` | Real miniature fits and resource measurements |
| `campaign.py` | Source pin and campaign registration |
| `worker.py` | Shared setup, exact-job execution, authenticated results |
| `submission.py` | Reviewed DAG, submission journals, watcher recovery |
| `progress.py` | Read-only per-epoch AUC/recovery and batch progress |

There is no separate preflight job. One shared setup validates the environment,
replays the frozen U100 encoder and measures actual production miniatures before
submitting scientific fits. Each fit requests one L40 on `gpus/default/normal`.
Controls need no teacher; supervised students wait only for their teacher.
Submission order does not guarantee Slurm start order. Reserve8GiB for outputs
in addition to other campaigns, the existing environment/bundle and the small
encoder transfer. Available quota is not converted into a larger campaign budget.

## 1. Commit and push on Windows

Review the staged diff, particularly the shared documentation, before committing.
These paths stage this implementation; no `git add .` is required.

```powershell
Set-Location 'C:\Users\22rya\ComputerScience\CERN\HLT_Classification'
git add -- src/hlt_classification/luka_cross_message_hydra scripts/luka_cross_message_hydra.py scripts/queue_luka_cross_message_hydra.sh sbatch/run_luka_cross_message_hydra.sh tests/test_luka_cross_message_hydra.py docs/plans/LUKA_CROSS_MESSAGE_HYDRA_PLAN.md docs/contracts/LUKA_CROSS_MESSAGE_HYDRA.md docs/LUKA_CROSS_MESSAGE_HYDRA_RUNBOOK.md docs/LUKA_CROSS_MESSAGE_HYDRA_VALIDATION.md
if ($LASTEXITCODE -ne 0) { throw 'Staging failed' }
git add -p -- README.md docs/HANDOFF.md docs/LEGACY_SOURCE_MAP.md
git diff --cached --stat
git diff --cached --check
if ($LASTEXITCODE -ne 0) { throw 'Check the staged changes' }
git commit -m 'Add privileged cross-attention message distillation on Hydra'
if ($LASTEXITCODE -ne 0) { throw 'Commit failed' }
git push origin main
if ($LASTEXITCODE -ne 0) { throw 'Push failed' }
git rev-parse HEAD
```

The runtime source guard permits only the new package/CLI/worker changes relative
to the registered donor commit. Keep unrelated runtime changes in separate work.
It verifies clean committed source on research compute.

## 2. Export the missing U100 encoder on SPORC (CPU only)

Brown already has the particle arrays, embedding bank and validation delta.
Only the original selected COARSE_U100 checkpoint and its saved predictions need
copying. This authenticates the original campaign/control with its own source;
it does not train, run inference, rebuild banks or submit a SPORC GPU job.

```bash
bash <<'BASH'
set -euo pipefail
read -r -p 'New pushed 40-character commit: ' COMMIT </dev/tty
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
MAIN=/home/ryreu/atlas/HLT_Classification
PROJECT="${MAIN}_cross_message_${COMMIT:0:8}"
DELTA="/home/ryreu/atlas/cross_message_encoder_${COMMIT:0:8}"
git -C "${MAIN}" fetch origin main
git -C "${MAIN}" merge-base --is-ancestor "${COMMIT}" origin/main
if [[ -e "${PROJECT}" ]]; then
    test "$(git -C "${PROJECT}" rev-parse HEAD)" = "${COMMIT}"
    test -z "$(git -C "${PROJECT}" status --porcelain --untracked-files=all)"
else
    git -C "${MAIN}" worktree add --detach "${PROJECT}" "${COMMIT}"
fi
ENV_DIR=/home/ryreu/miniconda3/envs/atlas_kd_sporc
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="${ENV_DIR}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
"${ENV_DIR}/bin/python" -B -s "${PROJECT}/scripts/luka_cross_message_hydra.py" export \
    --spec "${MAIN}/checkpoints/luka_u100_pair_correction_c08f6df6_debug_r1/campaign_spec.json" \
    --output "${DELTA}" --expected-commit "${COMMIT}"
BASH
```

Save the printed `manifest_hash` (64 characters). Missing original source or
checkpoint evidence stops export; another teacher cannot silently replace it.
Do not delete a partially exported directory or blindly repeat this command.

## 3. Copy, verify and queue on Brown

Use the same pushed commit and the encoder hash printed on SPORC. The known
joint campaign spec is only used to locate the existing validation-U100 delta;
its training jobs and results are not prerequisites for this campaign.

```bash
bash <<'BASH'
set -euo pipefail
read -r -p 'Same pushed 40-character commit: ' COMMIT </dev/tty
read -r -p 'SPORC encoder manifest hash (64 characters): ' ENCODER_HASH </dev/tty
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
[[ "${ENCODER_HASH}" =~ ^[0-9a-f]{64}$ ]]
BASE="$(cd "${HOME}" && pwd -P)/hlt_u100_pairs"
MAIN="${BASE}/code_65952ed3"
PROJECT="${BASE}/code_cross_message_${COMMIT:0:8}"
ENCODER="${BASE}/cross_message_encoder_${COMMIT:0:8}"
OLD_SPEC="${BASE}/runs/joint_adapter_a40ae47b_3c0f1980_r1/campaign_spec.json"
test -s "${OLD_SPEC}"
git -C "${MAIN}" fetch origin main
git -C "${MAIN}" merge-base --is-ancestor "${COMMIT}" origin/main
if [[ -e "${PROJECT}" ]]; then
    test "$(git -C "${PROJECT}" rev-parse HEAD)" = "${COMMIT}"
    test -z "$(git -C "${PROJECT}" status --porcelain --untracked-files=all)"
else
    git -C "${MAIN}" worktree add --detach "${PROJECT}" "${COMMIT}"
fi
test ! -e "${ENCODER}"
scp -r "ryreu@sporcsubmit.rc.rit.edu:/home/ryreu/atlas/cross_message_encoder_${COMMIT:0:8}" "${BASE}/"
ENV_DIR="${BASE}/environment_pair_hydra_v1/venv"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="${ENV_DIR}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
DELTA="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["delta_root"])' "${OLD_SPEC}")"
DELTA_HASH="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["delta_hash"])' "${OLD_SPEC}")"
"${ENV_DIR}/bin/python" -B -s "${PROJECT}/scripts/luka_cross_message_hydra.py" verify \
    --bundle "${BASE}/bundle_65952ed3" --delta "${DELTA}" --delta-hash "${DELTA_HASH}" \
    --encoder "${ENCODER}" --encoder-hash "${ENCODER_HASH}"
bash "${PROJECT}/scripts/queue_luka_cross_message_hydra.sh" "${BASE}" "${OLD_SPEC}" "${ENCODER}"
BASH
```

The helper prints the campaign path and `tail -f` command. After the log says
`Setup submitted as ...`, the setup is owned by Slurm and the detached watcher
will submit the remaining15 jobs when it succeeds. You may log off then. A
login-node restart or cleanup can still stop the watcher; `finish` below recovers
after successful setup. The `submission_science/submission_ledger.json` records
all14 fits and report once accepted. Cancel only exact campaign-bound IDs.

If files have already been copied and verified, run only the final queue helper
with the same three paths. Do not rerun `scp` into an existing directory. Once a
campaign spec exists, use its status/diagnostics rather than creating it again.

## Results, progress and watcher recovery

Paste the exact campaign path printed by the helper when prompted. For completed
results use `results`; for epoch AUC/recovery versus the historical U100-w5 curve
and the last logged batch percentage use `progress`. Both are read-only.

```bash
bash <<'BASH'
set -euo pipefail
read -r -p 'Exact campaign directory: ' ROOT </dev/tty
read -r -p 'Command (results or progress): ' COMMAND </dev/tty
[[ "${COMMAND}" == results || "${COMMAND}" == progress ]]
SPEC="${ROOT}/campaign_spec.json"
PROJECT="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["project_dir"])' "${SPEC}")"
ENV_DIR="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["environment_dir"])' "${SPEC}")"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="${ENV_DIR}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
"${ENV_DIR}/bin/python" -B -s "${PROJECT}/scripts/luka_cross_message_hydra.py" "${COMMAND}" --spec "${SPEC}"
BASH
```

Recovery uses historical SPORC M0HLT/OFFLINE. Results additionally show differences
from the Hydra U100-w5 replay and the correct width/seed architecture control.
Seed1 versus historical U100 is context, not a matched-seed comparison. Teacher
message-off rows are explicitly diagnostic. No teacher-quality threshold gates
the students. This debug-sized population cannot establish high-data gains.

If setup completed but the watcher stopped before submitting science, run:

```bash
bash <<'BASH'
set -euo pipefail
read -r -p 'Exact campaign directory with completed setup: ' ROOT </dev/tty
SPEC="${ROOT}/campaign_spec.json"
PROJECT="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["project_dir"])' "${SPEC}")"
ENV_DIR="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["environment_dir"])' "${SPEC}")"
PLAN_HASH="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["content_hash"])' "${ROOT}/submission/review.json")"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="${ENV_DIR}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
"${ENV_DIR}/bin/python" -B -s "${PROJECT}/scripts/luka_cross_message_hydra.py" finish \
    --spec "${SPEC}" --plan-hash "${PLAN_HASH}" --authorization AUTHORIZE_CROSS_MESSAGE_HYDRA_V1
BASH
```

This never reruns setup or removes claims. A complete science ledger is returned
without duplicates; partial or active submission requires inspection. Keep this
short submission command connected until it returns the full ledger.
