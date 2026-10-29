# Oscar: late fine-tuning of fine-tail DE000

One new job; no separate setup/preflight. It warm-starts the completed
K1/E40/plain fine-tail DE000 (reported AUC 0.96688543, 31.8% recovery), not the
historical split-head DE000 or a meta-embedding arm. Authority:
[active plan](plans/LUKA_FULLSTAT_FINETUNE_PLAN.md) and
[v1 contracts](contracts/LUKA_FULLSTAT_FINETUNE.md).

The original checkpoint lacks auxiliary heads and optimizer state. Therefore:

1. Restore its native HLT weights and buffers exactly.
2. Calibrate the recreated 14 auxiliary heads for three TRAIN-only passes on
   cached, frozen-native HLT features. No native parameter or buffer changes.
3. Fine-tune all parameters for ten additional epochs, cosine LR 1.5e-5 to
   1.5e-6, fresh AdamW, unchanged CE0.5/K1/E40 and all seven teachers.

This is a warm-start refinement, not an exact resume. The unchanged initializer
is eligible as epoch0; all ten new epochs run even if performance is worse.
Original models/banks stay read-only. Same 3M TRAIN/500k validation; test sealed.
The result remains adaptive single-seed validation, with a larger training budget.

## Commit and push locally

Stage only this task's files; do not use `git add .`. Shared HANDOFF and
LEGACY_SOURCE_MAP have unrelated work: selectively stage this task's sections
with `git add -p` if desired.

```powershell
git add -- src/hlt_classification/luka_fullstat_finetune
git add -- scripts/luka_fullstat_finetune.py scripts/queue_luka_fullstat_finetune.sh scripts/launch_luka_fullstat_finetune.sh sbatch/run_luka_fullstat_finetune.sh
git add -- tests/test_luka_fullstat_finetune.py tests/test_luka_fullstat_finetune_launch.py docs/plans/LUKA_FULLSTAT_FINETUNE_PLAN.md docs/contracts/LUKA_FULLSTAT_FINETUNE.md docs/LUKA_FULLSTAT_FINETUNE_RUNBOOK.md
git diff --cached --stat
git commit -m "Add low-LR FullStat DE000 fine-tuning with no separate preflight"
if ($LASTEXITCODE -ne 0) { throw "Inspect commit status before continuing" }
git push origin HEAD:main
if ($LASTEXITCODE -ne 0) { throw "Push failed" }
git rev-parse HEAD
```

The compatibility gate allows only this additive runtime namespace relative to
`a65caa78913e5e31a89ff7855b6f184605881d22`. Do not combine other runtime changes
into its release without explicitly reviewing/versioning that compatibility.

## One Oscar launch block

Reserve at least 4 GiB of personally available durable quota after other jobs.
The single L40S job requests 6 CPUs, 180000 MiB and a conservative 24-hour ceiling;
this is not a runtime prediction. Short parity, memory and timing admission runs
inside it before science, using disposable state and prior authenticated acceptance.

Paste after pushing; only commit and quota are requested before detachment:

```bash
(
set -euo pipefail
BASE=/oscar/home/rlyang/hlt_classification
REPO="${BASE}/source/HLT_Classification"
PARENT="${BASE}/checkpoints/luka_fullstat_fine_tail_k1_e40_04d1f0f5_r1/campaign_spec.json"
read -r -p 'Pushed fine-tuning commit (40 characters): ' COMMIT </dev/tty
checkquota
read -r -p 'Personally available result GiB (minimum 4): ' QUOTA </dev/tty
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ && "${QUOTA}" =~ ^[1-9][0-9]*$ && "${#QUOTA}" -le 6 ]]
(( QUOTA >= 4 ))
test -r "${PARENT}"
git -C "${REPO}" fetch origin main
git -C "${REPO}" cat-file -e "${COMMIT}^{commit}"
git -C "${REPO}" merge-base --is-ancestor "${COMMIT}" origin/main
PROJECT="${BASE}/source/HLT_Classification_finetune_${COMMIT:0:8}"
ROOT="${BASE}/checkpoints/luka_fullstat_finetune_${COMMIT:0:8}_r1"
if [[ ! -e "${PROJECT}" ]]; then
    git -C "${REPO}" worktree add --detach "${PROJECT}" "${COMMIT}"
fi
test "$(git -C "${PROJECT}" rev-parse HEAD)" = "${COMMIT}"
bash "${PROJECT}/scripts/launch_luka_fullstat_finetune.sh" \
    "${COMMIT}" "${ROOT}" "${PARENT}" "${QUOTA}" \
    --authorize AUTHORIZE_FULLSTAT_FINETUNE_AUTOSUBMIT_V1
)
```

After `DETACHED: launcher PID ...`, you may disconnect. The attached tail is
only a view; Ctrl-C stops that view, not authentication/submission. No further
prompt follows. Detachment protects against SSH hangup, not node reboot or
site process-cleanup policy. `DETACHED` is not confirmation of a submitted job:
look for `SUBMITTED`, `LAUNCH_COMPLETE` and `LAUNCHER_EXIT=0` later.

Full source/parent authentication and a saved dry plan happen before the one
authorized submission. This does not cancel, alter, or depend on other jobs.
On failure preserve `.launch/failure.json`, claims and submission journals;
do not rerun, delete claims, or manually submit in parallel.

## Progress and completed results

Use the exact ROOT/PROJECT printed at launch (set them again after reconnecting).

```bash
tail -n 60 -f "${ROOT}.launch/launcher.log"
squeue --me -o '%.18i %.12P %.40j %.2t %.10M %.12l %R'
```

Job name: `lukafsftune_finetune`. Worker log: `finetune-JOBID.out` under ROOT.
`FINETUNE frozen_features` and `head_calibration` precede added epoch1.
`FINETUNE_PROGRESS` records batch/epoch progress; `FINETUNE epoch=` records AUC.
Every completed epoch has a full-state snapshot plus receipt; selected native
state, report and full state are saved before final publication checks. There
is no automatic recovery command: partial output requires reviewed recovery.

Authenticated saved comparison, including R30/R50 in JSON:

```bash
(
set -euo pipefail
export PROJECT_DIR="${PROJECT}"
export JC2_SITE=oscar_l40s
source "${PROJECT_DIR}/sbatch/luka_splithead_finalize_common.sh"
python -u -B -s "${PROJECT_DIR}/scripts/luka_fullstat_finetune.py" \
    results --spec "${ROOT}/campaign_spec.json"
)
```

This reads saved results, not new inference/test data. Fine-tune epoch numbers
are additional epochs; epoch0 means the original checkpoint was retained.
The report also records best-new validation independently of that fallback.

Manual dry-review alternative (do not combine with detached launch):

```bash
bash "${PROJECT}/scripts/queue_luka_fullstat_finetune.sh" "${COMMIT}" "${ROOT}" "${PARENT}" "${QUOTA}"
read -r -p 'Reviewed PLAN_HASH: ' PLAN_HASH </dev/tty
bash "${PROJECT}/scripts/queue_luka_fullstat_finetune.sh" "${COMMIT}" "${ROOT}" "${PARENT}" "${QUOTA}" --execute "${PLAN_HASH}"
```

## Validation boundary

2026-10-29: 116 passed, 2 skipped across the two new and two original fine-tail
test suites (155.76 seconds). New tests: 51 passed, one Linux-only HUP skip.
Both skips concern genuine Linux hangup behavior; Windows Bash detached-flow
smoke tests passed. AST, shell syntax, CLI help and documentation links passed.

Local tests exercise installed Weaver, full native CUDA architecture, synthetic
TRAIN/validation data, ten added epochs, calibration and native/full readback.
Scheduler/source boundaries are mocked in the miniature; local CUDA is not
Oscar resource or scientific evidence. No jobs were submitted during implementation.
Real Oscar inline admission and the exact completed parent remain required.
