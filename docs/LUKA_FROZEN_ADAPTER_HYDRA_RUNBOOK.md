# Frozen privileged adapters: implementation and Hydra runbook

The campaign fits **two privileged adapters on frozen M0HLT**, then **twelve
from-scratch HLT students**. Each adapter site (attention bias, edge messages)
has auxiliary-only, CE-only injected, and supervised-injected students, each
with embedding weight 0 or U100 weight 5. There is one shared setup and a report:
16 jobs total. The setup includes real production miniatures; there is no
separate preflight. Students have no scheduling lanes. Only the eight students
that consume teacher targets wait for their own selected teacher.

Hydra uses partition `gpus`, account `default`, QoS `normal`, and one L40 per job.
`debug` is the SPORC partition and is not used here. These scripts do not cancel,
release, modify, or resubmit existing jobs.

## Code map

The authoritative definitions are the [plan](plans/LUKA_FROZEN_ADAPTER_HYDRA_PLAN.md)
and [v1 contracts](contracts/LUKA_FROZEN_ADAPTER_HYDRA.md).

| File under `src/hlt_classification/luka_frozen_adapter_hydra/` | Responsibility |
| --- | --- |
| `contracts.py` | Fixed recipe, two oracles, twelve students and genuine dependencies |
| `model.py` | Native block-6 hooks, bounded correction predictor, row centering, edge contraction, head/message alignment, HLT-only forward |
| `training.py` | CE, optional embedding5, correction loss/ramp; frozen-buffer checks; selected deployment publication/reload; TRAIN target RMS |
| `inputs.py` | Minimal M0HLT/validation-U100 transfer, exact manifest binding, role/identity/particle-slot joins |
| `resources.py` | Production miniatures, worst-multiplicity samples, L40 memory/time envelope |
| `campaign.py` | Source pin, immutable spec, input and portability lineage |
| `submission.py` | Dry plan, detached dispatcher, exact-ID journals, no artificial dependencies |
| `worker.py` | Setup, fits, receipts and authenticated completed-results report |

Entry points are `scripts/luka_frozen_adapter_hydra.py`,
`scripts/queue_luka_frozen_adapter_hydra.sh`, and
`sbatch/run_luka_frozen_adapter_hydra.sh`.

The predictor is separate from the U100 jet-embedding head. It reads contextual
HLT tokens and the four native HLT pair features. The teacher additionally reads
U100 features in the same particle slots. It never gets matching indices as
model features. Frozen teacher weights make the target stable; the target still
varies for every jet. Targets are computed online, avoiding an N-squared disk
bank. The bias mixer and edge-message basis are learnable in fresh students.
CE can therefore adapt the correction's use without requiring new attention
heads to inherit the frozen teacher's meaning.

## Commit the new campaign locally

Run in Windows PowerShell. These paths stage only this campaign. Shared README,
handoff and donor-map files may contain unrelated edits; inspect and stage their
new sections separately if desired. Their complete standalone evidence is also
in [the validation record](LUKA_FROZEN_ADAPTER_HYDRA_VALIDATION.md).

```powershell
Set-Location 'C:\Users\22rya\ComputerScience\CERN\HLT_Classification'
git add -- src/hlt_classification/luka_frozen_adapter_hydra scripts/luka_frozen_adapter_hydra.py scripts/queue_luka_frozen_adapter_hydra.sh sbatch/run_luka_frozen_adapter_hydra.sh tests/test_luka_frozen_adapter_hydra.py docs/plans/LUKA_FROZEN_ADAPTER_HYDRA_PLAN.md docs/contracts/LUKA_FROZEN_ADAPTER_HYDRA.md docs/LUKA_FROZEN_ADAPTER_HYDRA_RUNBOOK.md docs/LUKA_FROZEN_ADAPTER_HYDRA_VALIDATION.md
if ($LASTEXITCODE -ne 0) { throw 'Staging failed' }
git diff --cached --stat
git diff --cached --check
if ($LASTEXITCODE -ne 0) { throw 'Check staged changes' }
git commit -m 'Add frozen privileged adapter distillation on Hydra'
if ($LASTEXITCODE -ne 0) { throw 'Commit failed' }
git push origin main
if ($LASTEXITCODE -ne 0) { throw 'Push failed' }
git rev-parse HEAD
```

The source guard requires additive runtime changes relative to
`54473f108c4b1b435f799ca2e41a9c69a4ebe332`. Do not include unrelated runtime
changes in this campaign commit; the queue reads the exact pushed checkout.

## Export the small missing delta on SPORC

Brown already has TRAIN D000/U100, validation D000 and the U100 bank. The only
additional payload is the authenticated original M0HLT checkpoint plus validation
U100 shards for oracle selection (roughly 0.3 GiB; actual bytes are printed).
No GPU or new inference is used for export. This does not depend on the separate
contrastive campaign's delta.

```bash
bash <<'BASH'
set -euo pipefail
read -r -p 'New pushed 40-character commit: ' COMMIT </dev/tty
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
MAIN=/home/ryreu/atlas/HLT_Classification
PROJECT="${MAIN}_frozen_adapter_${COMMIT:0:8}"
DELTA="/home/ryreu/atlas/frozen_adapter_delta_${COMMIT:0:8}"
git -C "${MAIN}" fetch origin main
git -C "${MAIN}" merge-base --is-ancestor "${COMMIT}" origin/main
if [[ -e "${PROJECT}" ]]; then
    test "$(git -C "${PROJECT}" rev-parse HEAD)" = "${COMMIT}"
    test -z "$(git -C "${PROJECT}" status --porcelain --untracked-files=all)"
else
    git -C "${MAIN}" worktree add --detach "${PROJECT}" "${COMMIT}"
fi
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
ENV_DIR=/home/ryreu/miniconda3/envs/atlas_kd_sporc
export LD_LIBRARY_PATH="${ENV_DIR}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
"${ENV_DIR}/bin/python" -B -s "${PROJECT}/scripts/luka_frozen_adapter_hydra.py" export \
    --spec "${MAIN}/checkpoints/luka_u100_pair_correction_c08f6df6_debug_r1/campaign_spec.json" \
    --output "${DELTA}" --expected-commit "${COMMIT}"
BASH
```

Keep the printed `manifest_hash`. Existing source checkpoints, arrays and jobs
are not changed. A missing checkpoint or shard stops export instead of silently
rebuilding data. Preserve a partial failed export; use a fresh destination for
an investigated retry.

## Copy and queue on Brown

Paste the same commit and the exported manifest hash. SCP prompts for your RIT
password/Duo. Enter credentials in the terminal only.

```bash
bash <<'BASH'
set -euo pipefail
read -r -p 'Same pushed 40-character commit: ' COMMIT </dev/tty
read -r -p 'SPORC delta manifest hash (64 characters): ' DELTA_HASH </dev/tty
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ && "${DELTA_HASH}" =~ ^[0-9a-f]{64}$ ]]
BASE="$(cd "${HOME}" && pwd -P)/hlt_u100_pairs"
PROJECT="${BASE}/code_frozen_adapter_${COMMIT:0:8}"
DELTA="${BASE}/frozen_adapter_delta_${COMMIT:0:8}"
REPO="${BASE}/code_65952ed3"
git -C "${REPO}" fetch origin main
git -C "${REPO}" merge-base --is-ancestor "${COMMIT}" origin/main
if [[ -e "${PROJECT}" ]]; then
    test "$(git -C "${PROJECT}" rev-parse HEAD)" = "${COMMIT}"
    test -z "$(git -C "${PROJECT}" status --porcelain --untracked-files=all)"
else
    git -C "${REPO}" worktree add --detach "${PROJECT}" "${COMMIT}"
fi
test ! -e "${DELTA}"
scp -r "ryreu@sporcsubmit.rc.rit.edu:/home/ryreu/atlas/frozen_adapter_delta_${COMMIT:0:8}" "${BASE}/"
ENV_DIR="${BASE}/environment_pair_hydra_v1/venv"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="${ENV_DIR}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
"${ENV_DIR}/bin/python" -B -s "${PROJECT}/scripts/luka_frozen_adapter_hydra.py" verify \
    --bundle "${BASE}/bundle_65952ed3" --delta "${DELTA}" --delta-hash "${DELTA_HASH}"
bash "${PROJECT}/scripts/queue_luka_frozen_adapter_hydra.sh" "${BASE}" "${DELTA}" "${DELTA_HASH}"
BASH
```

Reserve **8 additional GiB** for this campaign, separate from space reserved for
other running campaigns. The script uses an eight-GiB budget, not your entire
reported free-space total. It requires one extra GiB of actual free headroom.

The helper reviews setup and all fourteen fits before starting a detached
dispatcher. Once the submit log says `Setup submitted ... Automatic science
submission active; safe to log off`, logout will not interrupt submission.
Successful measured setup automatically queues the two teachers, twelve
students, and report. Finite poor teacher performance does not cancel students.
A technically failed teacher cancels only its four target-dependent students;
the other teacher family and all CE controls remain eligible. The report runs
after all accepted jobs finish or are canceled. No automatic retry occurs.

## Results and progress

Use the printed campaign directory, not a glob selecting an arbitrary run:

```bash
ROOT=/cs/home/rlyang/hlt_u100_pairs/runs/frozen_adapter_COMMIT8_DELTAHASH8_r1
SPEC="${ROOT}/campaign_spec.json"
PROJECT="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["project_dir"])' "${SPEC}")"
ENV_DIR="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["environment_dir"])' "${SPEC}")"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="${ENV_DIR}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
"${ENV_DIR}/bin/python" -B -s "${PROJECT}/scripts/luka_frozen_adapter_hydra.py" results --spec "${SPEC}"
squeue --me -o "%.18i %.12P %.38j %.2t %.10M %R"
```

Each fit saves `TASK/fit/epoch_01.json` through `epoch_30.json` as epochs finish,
and prints `FROZEN_ADAPTER TASK epoch=... auc=...` plus batch progress in its
`TASK-JOBID.out` log. Logs and incomplete fits are provisional. Completed rows
require both fit and exact-job receipts, verified predictions, and correct
parents. Recovery uses historical M0HLT/OFFLINE; this is not an independent test.

## Copying or extending this setup

Copy the campaign as an additive package or version its contracts; do not edit
an admitted checkout. `AdapterParT.forward(features,vectors,mask)` is the HLT-only
student path. `oracle(...,u_features=...,u_vectors=...)` is an explicitly different
interface. `training.deployment(settings,state,device)` strictly loads a selected
deployment artifact and removes training-only heads. `aux` deployments are
ordinary native ParT. `ce`/`both` retain the correction and alignment modules.

Do not interpret an oracle gain as evidence that the student can recover it.
The key comparisons are `both` versus its embedding-matched `ce` and `aux`, and
all three versus M0HLT or U100-w5. Start with the registered weights before
tuning based on this development set. Repeated seeds and a new untouched
evaluation population would be needed for stronger claims.
