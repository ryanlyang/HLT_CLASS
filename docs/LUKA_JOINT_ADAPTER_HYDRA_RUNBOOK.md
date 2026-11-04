# Joint privileged-adapter Hydra campaign

This additive campaign has two end-to-end U100 oracle teachers and twelve fresh
HLT students. All students keep U100 embedding weight5 and compare intermediate
injection, intermediate auxiliary supervision and final-token auxiliary
supervision at correction weights0.3/1. See the [scientific plan](plans/LUKA_JOINT_ADAPTER_HYDRA_PLAN.md),
[contract](contracts/LUKA_JOINT_ADAPTER_HYDRA.md) and [validation record](LUKA_JOINT_ADAPTER_HYDRA_VALIDATION.md).

## Code map

- `src/hlt_classification/luka_joint_adapter_hydra/contracts.py`: ordered grid,
  fixed seeds/recipe, artifact namespace and teacher dependencies.
- `model.py`: trainable native teacher, U100-only adapter input, block6 bias or
  edge correction, block7 final-token tap, learned student alignment, deployment.
- `training.py`: CE+embedding+correction updates, TRAIN RMS, 30-epoch selection,
  per-epoch publication, exact selected-model reload and validated reports.
- `inputs.py`: reuses the frozen campaign's authenticated Brown inputs/delta.
- `references.py`: original-source replay of existing completed controls.
- `resources.py`, `worker.py`: real shared setup miniatures, fit workers/results.
- `campaign.py`, `submission.py`: source pin, immutable spec, dry/live plans,
  exact journals and restartable post-setup science submission.
- `progress.py`: per-epoch AUC/recovery and last emitted batch progress.
- `scripts/luka_joint_adapter_hydra.py`: thin CLI; `sbatch/run_luka_joint_adapter_hydra.sh`:
  absolute-project worker; `scripts/queue_luka_joint_adapter_hydra.sh`: launcher.

## Commit and push on Windows

The original frozen, pair and contrastive runtime files must remain unchanged.
Existing unrelated work is not part of this commit. Stage the new implementation:

```powershell
Set-Location 'C:\Users\22rya\ComputerScience\CERN\HLT_Classification'
git add -- src/hlt_classification/luka_joint_adapter_hydra scripts/luka_joint_adapter_hydra.py scripts/queue_luka_joint_adapter_hydra.sh sbatch/run_luka_joint_adapter_hydra.sh tests/test_luka_joint_adapter_hydra.py docs/plans/LUKA_JOINT_ADAPTER_HYDRA_PLAN.md docs/contracts/LUKA_JOINT_ADAPTER_HYDRA.md docs/LUKA_JOINT_ADAPTER_HYDRA_RUNBOOK.md docs/LUKA_JOINT_ADAPTER_HYDRA_VALIDATION.md
if ($LASTEXITCODE -ne 0) { throw 'Stage failed' }
git add -p -- README.md docs/HANDOFF.md docs/LEGACY_SOURCE_MAP.md
git diff --cached --stat
git diff --cached --check
if ($LASTEXITCODE -ne 0) { throw 'Review staged changes' }
git commit -m "Add joint privileged adapter teachers and final-token supervision on Hydra"
if ($LASTEXITCODE -ne 0) { throw 'Commit failed' }
git push origin main
if ($LASTEXITCODE -ne 0) { throw 'Push failed' }
git rev-parse HEAD
```

Select only this campaign's additions in the interactive documentation staging.
Source admission requires the new commit to descend from
`ceb2146149d2b1508a36b2cf91b274e0b58a26e6`, with runtime changes restricted to this
new package/CLI/worker. Use the printed full40-character commit below.

## Queue from a Brown login terminal

No new SPORC export, data copy, environment install, target rebuild or reference
training is needed. The launcher discovers the existing delta from the original
frozen campaign spec. Keep that original checkout and all saved inputs/results.

```bash
bash <<'BASH'
set -euo pipefail
read -r -p 'New pushed joint-adapter commit (40 characters): ' COMMIT </dev/tty
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
BASE="$(cd "${HOME}" && pwd -P)/hlt_u100_pairs"
MAIN="${BASE}/code_frozen_adapter_4b9630fa"
PROJECT="${BASE}/code_joint_adapter_${COMMIT:0:8}"
CONTROL_SPEC="${BASE}/runs/frozen_adapter_4b9630fa_3c0f1980_r1/campaign_spec.json"
git -C "${MAIN}" fetch origin main
git -C "${MAIN}" merge-base --is-ancestor "${COMMIT}" origin/main
if [[ -e "${PROJECT}" ]]; then
    test "$(git -C "${PROJECT}" rev-parse HEAD)" = "${COMMIT}"
    test -z "$(git -C "${PROJECT}" status --porcelain --untracked-files=all)"
else
    git -C "${MAIN}" worktree add --detach "${PROJECT}" "${COMMIT}"
fi
bash "${PROJECT}/scripts/queue_luka_joint_adapter_hydra.sh" "${BASE}" "${CONTROL_SPEC}"
BASH
```

Reserve at least8GiB **additional output allowance**, leaving space for running
campaigns. The launcher uses an8GiB fixed campaign budget, not the entire number
typed. It authenticates controls that have finished; unfinished controls remain
pending and do not hold up setup or science.

The launcher prints the exact campaign directory and detached `setsid`/`nohup`
dispatcher log. Once its PID/log have been printed, submission can continue
after logout. Check the log for the actual setup ID and later the science ledger:
the process being detached is not proof that submission has succeeded.

Shared setup is one L40 job that performs real miniatures and resource acceptance.
It automatically releases the two teachers, twelve students and report after
technical success. Only each student's own teacher is an `afterok` dependency;
the report is `afterany`. There are no lanes or array caps. Resources are
`gpus/default/normal`, one `nvidia_l40`, six CPUs and120000MiB. Existing jobs and
user holds are untouched. This is16 Slurm jobs including setup/report.

## Results and live epoch comparisons

Use the exact ROOT printed by the launcher; replace the example suffix. This
block reads the pinned project/environment from that campaign, not another clone.

```bash
bash <<'BASH'
set -euo pipefail
read -r -p 'Exact new campaign directory: ' ROOT </dev/tty
SPEC="${ROOT}/campaign_spec.json"
PROJECT="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["project_dir"])' "${SPEC}")"
ENV_DIR="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["environment_dir"])' "${SPEC}")"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT}/src"
export LD_LIBRARY_PATH="${ENV_DIR}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
"${ENV_DIR}/bin/python" -u -B -s "${PROJECT}/scripts/luka_joint_adapter_hydra.py" results --spec "${SPEC}"
"${ENV_DIR}/bin/python" -u -B -s "${PROJECT}/scripts/luka_joint_adapter_hydra.py" progress --spec "${SPEC}"
BASH
```

`results` authenticates completed fits and replays original control evidence;
it reads saved arrays but performs no inference/training or writes. `progress`
reads published epoch JSONs and logs, includes unfinished fits, and prints
per-epoch AUC/recovery relative to the saved SPORC U100-w5 history. Teacher rows
are privileged oracles. Early negative recovery uses fixed best-model anchors.
Batch percentages exclude validation and describe the last logged marker.

## Resume automatic submission after completed setup

If setup has completed successfully but the dispatcher stopped before science,
use this command. It neither reruns setup nor removes old claims. If science is
already fully submitted it returns the same ledger. If a submission is active
or partial, it stops to preserve exact journals; do not blindly retry/delete.

```bash
bash <<'BASH'
set -euo pipefail
read -r -p 'Exact new campaign directory: ' ROOT </dev/tty
SPEC="${ROOT}/campaign_spec.json"
PROJECT="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["project_dir"])' "${SPEC}")"
ENV_DIR="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["environment_dir"])' "${SPEC}")"
PLAN_HASH="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["content_hash"])' "${ROOT}/submission/review.json")"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT}/src"
export LD_LIBRARY_PATH="${ENV_DIR}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
"${ENV_DIR}/bin/python" -u -B -s "${PROJECT}/scripts/luka_joint_adapter_hydra.py" finish \
    --spec "${SPEC}" --plan-hash "${PLAN_HASH}" --authorization AUTHORIZE_JOINT_ADAPTER_HYDRA_V1
BASH
```

The initial queue command authorizes the complete reviewed16-job campaign.
This recovery command completes that same submission, not a new campaign.
