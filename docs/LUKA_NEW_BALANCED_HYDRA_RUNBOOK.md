# Queue the six-model balanced campaign on the new Hydra dataset

See the [plan](plans/LUKA_NEW_BALANCED_HYDRA_PLAN.md),
[contract](contracts/LUKA_NEW_BALANCED_HYDRA.md) and
[validation record](LUKA_NEW_BALANCED_HYDRA_VALIDATION.md).
Dataset upload is already complete. The natural OFFLINE baseline jobs 174642
and 174643 belong to a separate experiment and must remain intact.

## Commit and push from local PowerShell

Stage only this campaign's files. Other chats have unrelated dirty work here.

```powershell
git add -- src/hlt_classification/luka_new_balanced_hydra `
  scripts/luka_new_balanced_hydra.py scripts/launch_luka_new_balanced_hydra.sh `
  sbatch/run_luka_new_balanced_hydra.sh tests/test_luka_new_balanced_hydra.py `
  docs/plans/LUKA_NEW_BALANCED_HYDRA_PLAN.md `
  docs/contracts/LUKA_NEW_BALANCED_HYDRA.md `
  docs/LUKA_NEW_BALANCED_HYDRA_RUNBOOK.md `
  docs/LUKA_NEW_BALANCED_HYDRA_VALIDATION.md

# Select this campaign's hunks only in these shared documents.
git add -p -- README.md docs/HANDOFF.md docs/LEGACY_SOURCE_MAP.md docs/plans/README.md
git diff --cached --stat
git commit -m "Add balanced six-model Hydra campaign for new FullSim data"
git push
git rev-parse HEAD
```

The complete source must be pushed to `origin/main`. Keep local large population
and miniature artifacts in `reports/luka_new_balanced_hydra_validation/` out of
the source commit; the registered population is deterministically rebuilt and
must reproduce its frozen hash on Hydra.

## Login

```powershell
ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=6 -i "$env:USERPROFILE\.ssh\id_rsa_brown_cs" rlyang@ssh.cs.brown.edu
```

## Paste this block in Hydra

The quota prompt means **current available home quota**, before subtracting the
natural baseline's reserve. The launcher calculates that reserve itself.
At the user's latest measurement, 29 GiB was available and the baseline cache was
3.7 GiB; do not assume those values remain current. The new result budget is 14 GiB,
plus up to 17 GiB minus the baseline's already-written cache/results. Other running
experiments can consume space; consult current availability before launching.
Recovery reserves space for unfinished outputs without treating abandoned
partial attempts as reusable. It never removes those files automatically.

```bash
bash <<'BASH'
set -euo pipefail
BASE=/cs/home/rlyang/hlt_u100_pairs
REPO="${BASE}/code_65952ed3"
DATASET="${BASE}/datasets/luka_qcdrun2com_higgs2p_sb_v1"
BASELINE="${BASE}/runs/new_fullstat_offline_a865e5b6_r1/campaign_spec.json"
df -h "${BASE}"
read -r -p 'Pushed 40-character balanced campaign commit: ' COMMIT </dev/tty
read -r -p 'Current free HOME quota, whole GiB: ' FREE_GIB </dev/tty
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ && "${FREE_GIB}" =~ ^[0-9]+$ ]]
git -C "${REPO}" fetch origin main
git -C "${REPO}" merge-base --is-ancestor "${COMMIT}" origin/main
SHORT="${COMMIT:0:8}"
PROJECT="${BASE}/code_new_balanced_${SHORT}"
ROOT="${BASE}/runs/new_balanced_${SHORT}_r1"
if [[ ! -e "${PROJECT}" ]]; then
    git -C "${REPO}" worktree add --detach "${PROJECT}" "${COMMIT}"
fi
test "$(git -C "${PROJECT}" rev-parse HEAD)" = "${COMMIT}"
bash "${PROJECT}/scripts/launch_luka_new_balanced_hydra.sh" detach \
    "${PROJECT}" "${DATASET}" "${ROOT}" "${COMMIT}" l40s "${FREE_GIB}" "${BASELINE}"
BASH
```

After `DETACHED`, disconnecting or Ctrl-C on the tail does not stop registration
or queue submission. `LAUNCHER_EXIT=0` and six `SUBMITTED` lines confirm successful
submission. Detached does not mean queued yet: population authentication/replay
and scheduler dry review happen first. Do not launch twice if they take time.
Failure logs remain under `ROOT.launch/launcher.log`; inspect before retrying.

Six GPU science jobs are submitted. OFFLINE, M0HLT and CONCAT are independent;
UE100 depends on all three, DE050 on its four teachers, DE000 on its four
teachers. Each requests one L40S, 192000 MiB, 6 CPUs, 48 h. No separate preflight job.
Actual acceptance and conservative timing checks occur inside those allocations.
An explicit GPU argument `l40` or `a6000` changes the registration; this is not
an automatic scheduler fallback and does not use multiple GPUs.

## Watch progress

Replace the eight-character suffix below with the pushed commit prefix:

```bash
BASE=/cs/home/rlyang/hlt_u100_pairs
SHORT=YOUR_COMMIT_PREFIX
PROJECT="${BASE}/code_new_balanced_${SHORT}"
ROOT="${BASE}/runs/new_balanced_${SHORT}_r1"
ENV_DIR="${BASE}/environment_pair_hydra_v1/venv"
squeue --me -o '%.18i %.40j %.2t %.10M %R'
tail -n 60 -f "${ROOT}.launch/launcher.log"
```

After Ctrl-C stops that tail, use:

```bash
"${ENV_DIR}/bin/python" -u -B -s "${PROJECT}/scripts/luka_new_balanced_hydra.py" progress \
    --spec "${ROOT}/campaign_spec.json" --stage DE000
```

Any of `OFFLINE M0HLT CONCAT UE100 DE050 DE000` is accepted. This fast command
reads saved metadata/logs, not prediction payloads. After all stages publish:

```bash
"${ENV_DIR}/bin/python" -u -B -s "${PROJECT}/scripts/luka_new_balanced_hydra.py" results \
    --spec "${ROOT}/campaign_spec.json"
```

Results authenticate saved outputs and report primary, per-bin and outside
validation AUC/accuracy/R50/R30 plus recovery from the new balanced anchors.
No new inference or final-test evaluation is performed.

## Explicit recovery

First inspect the failing job's log and scheduler accounting. All previous exact
campaign IDs must be terminal; no broad cancellation command is provided.
If a stage failed after its twentieth epoch, recovery repeats integrated checks
and publication without repeating scientific training updates.
Earlier failures resume from the latest committed epoch, including optimizer
and RNG. Published stages are authenticated and omitted from the new DAG.

With the variables above still set, and after refreshing current free quota:

```bash
COMMIT="$(git -C "${PROJECT}" rev-parse HEAD)"
BASELINE="${BASE}/runs/new_fullstat_offline_a865e5b6_r1/campaign_spec.json"
DATASET="${BASE}/datasets/luka_qcdrun2com_higgs2p_sb_v1"
read -r -p 'Current free HOME quota, whole GiB: ' FREE_GIB
bash "${PROJECT}/scripts/launch_luka_new_balanced_hydra.sh" resume-detach \
    "${PROJECT}" "${DATASET}" "${ROOT}" "${COMMIT}" l40s "${FREE_GIB}" "${BASELINE}"
```

The detached recovery launcher is single-use. A second interrupted recovery
requires inspecting its log/journal before a new named launch. Never delete
submission intents or claims to force a retry. An uncertain sbatch response
requires exact-ID reconciliation. This resume path does not change source,
scientific recipe, GPU or walltime.
