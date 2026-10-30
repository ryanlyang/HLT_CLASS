# Oscar: FullStat U100 frozen-readout CE

One fresh HLT student on 3M TRAIN / 500k validation, 20 epochs, batch 128.
Objective: native CE1 + U100 embedding SmoothL1 weight5 + frozen-readout CE,
ramped from zero to weight1 over the first three passes. Original FullStat
COARSE_U100 supplies both the embeddings and its exact frozen classifier.
The completed U100-w5 run is the historical reference; no new control is queued.
See [plan](plans/LUKA_FULLSTAT_READOUT_PLAN.md) and
[contracts](contracts/LUKA_FULLSTAT_READOUT.md). Final test remains sealed.

The only job is `lukafsro_train_readout`, on one L40S with 6 CPUs, 180000 MiB
and a 24-hour ceiling. Reserve 4 GiB of durable quota. Existing TRAIN bank is
reused read-only. Bounded parity, largest-support gradient/memory and timing
checks run inside this job before fresh training. There is no standalone
preflight, no teacher retraining and no full target re-export.

## Retry after job 8118177

The first job at `abe387791386507580b7e826850577c2b4cea6cf` failed in
`integrated_readout_checks` before any completed epochs. Its teacher replay
disabled TF32/deterministically changed the backend and regrouped rows into
different padded batches from the historical exporter. That is an invalid
numerical comparison. The correction replays complete original export batches
under the ordinary production backend with unchanged numerical tolerances.
It adds logged batch ranges, backend flags and error magnitudes. Strict checks
for fresh-vs-fresh native/installed-Weaver parity are retained.

Commit/push the correction and use the launch block below with that NEW commit.
It creates a fresh `luka_fullstat_readout_NEWCOMMIT_r1` root and submits one
training job with inline checks; no standalone preflight or target re-export.
Keep `luka_fullstat_readout_abe38779_r1` and its source/logs unchanged. Do not
requeue 8118177, replace its spec or reuse its output directory: it points at
the old code and SPEC/STARTUP v1. The correction uses SPEC/STARTUP v2 while
retaining the scientific recipe. No trained epochs were lost in this failure.

## Commit and push this implementation

Preserve unrelated work. Stage only these files; shared HANDOFF/LEGACY_SOURCE_MAP
and plans index can be selectively staged with `git add -p`.

```powershell
git add -- src/hlt_classification/luka_fullstat_readout
git add -- scripts/luka_fullstat_readout.py scripts/launch_luka_fullstat_readout.sh scripts/queue_luka_fullstat_readout.sh sbatch/run_luka_fullstat_readout.sh
git add -- tests/test_luka_fullstat_readout.py tests/test_luka_fullstat_readout_launch.py docs/plans/LUKA_FULLSTAT_READOUT_PLAN.md docs/contracts/LUKA_FULLSTAT_READOUT.md docs/LUKA_FULLSTAT_READOUT_RUNBOOK.md
git diff --cached --stat
git commit -m "Fix FullStat readout teacher replay backend and export batches"
if ($LASTEXITCODE -ne 0) { throw "Inspect commit status before continuing" }
git push origin HEAD:main
if ($LASTEXITCODE -ne 0) { throw "Push failed" }
git rev-parse HEAD
```

The exact pushed commit must add only this runtime namespace relative to
`de79c7298a54dcb5818b97772635837e0a769914`. The original U100-w5 source, teacher,
bank, receipts and completed scheduler evidence are authenticated separately.

## One Oscar launch block

Paste after pushing. Supply the commit and personally available quota before
detachment. The completed parent is U100-w5, not a meta or fine-tail run.

```bash
(
set -euo pipefail
BASE=/oscar/home/rlyang/hlt_classification
REPO="${BASE}/source/HLT_Classification"
PARENT="${BASE}/checkpoints/luka_fullstat_u100_embedding_w5_7ea8b0c6_r1/campaign_spec.json"
read -r -p 'Pushed readout commit (40 characters): ' COMMIT </dev/tty
checkquota
read -r -p 'Personally available result GiB (minimum 4): ' QUOTA </dev/tty
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ && "${QUOTA}" =~ ^[1-9][0-9]*$ && "${#QUOTA}" -le 6 ]]
(( QUOTA >= 4 ))
test -r "${PARENT}"
git -C "${REPO}" fetch origin main
git -C "${REPO}" cat-file -e "${COMMIT}^{commit}"
git -C "${REPO}" merge-base --is-ancestor "${COMMIT}" origin/main
PROJECT="${BASE}/source/HLT_Classification_readout_${COMMIT:0:8}"
ROOT="${BASE}/checkpoints/luka_fullstat_readout_${COMMIT:0:8}_r1"
if [[ ! -e "${PROJECT}" ]]; then
    git -C "${REPO}" worktree add --detach "${PROJECT}" "${COMMIT}"
fi
test "$(git -C "${PROJECT}" rev-parse HEAD)" = "${COMMIT}"
bash "${PROJECT}/scripts/launch_luka_fullstat_readout.sh" \
    "${COMMIT}" "${ROOT}" "${PARENT}" "${QUOTA}" \
    --authorize AUTHORIZE_FULLSTAT_READOUT_AUTOSUBMIT_V1
)
```

After `DETACHED: launcher PID ...` you can disconnect. The tail is only a log
view; Ctrl-C closes it without stopping the detached launcher. No later prompt
is needed. The launcher authenticates source/results, saves a complete dry plan,
checks its exact hash against the bounded one-job authorization and submits once.
Look later for `LAUNCH_COMPLETE` and `LAUNCHER_EXIT=0`; detachment alone is not
proof of submission. SSH hangup protection does not cover a login-node reboot.
On failure retain claims, journals and failure.json; do not launch twice.

## Progress and saved results

Set ROOT/PROJECT to the exact printed paths after reconnecting:

```bash
tail -n 60 -f "${ROOT}.launch/launcher.log"
squeue --me -o '%.18i %.12P %.40j %.2t %.10M %.12l %R'
```

The worker log is `train_readout-JOBID.out`. `FULLSTAT READOUT progress` includes
batch/epoch and estimated training time remaining; `FULLSTAT READOUT weight=`
includes native validation AUC. Estimates exclude validation/checkpointing.
Completed epochs persist under `train_readout/epochs/NN/`, including predictor,
optimizer, RNG and validation history. Selected native/full state is saved
before final readback. Partial work is preserved for reviewed recovery; there
is no automatic resume.

```bash
(
set -euo pipefail
export PROJECT_DIR="${PROJECT}" JC2_SITE=oscar_l40s
source "${PROJECT_DIR}/sbatch/luka_splithead_finalize_common.sh"
python -u -B -s "${PROJECT_DIR}/scripts/luka_fullstat_readout.py" \
    results --spec "${ROOT}/campaign_spec.json"
)
```

This authenticates saved native validation results and prints AUC/recovery;
JSON includes accuracy, R30/R50 and their recoveries against M0HLT/OFFLINE,
plus historical U100-w5. All use each model's AUC-selected checkpoint and the
same ordered validation population. No new inference or test access.

Manual dry/live alternative, never alongside the detached launcher:

```bash
bash "${PROJECT}/scripts/queue_luka_fullstat_readout.sh" "${COMMIT}" "${ROOT}" "${PARENT}" "${QUOTA}"
read -r -p 'Reviewed PLAN_HASH: ' PLAN_HASH </dev/tty
bash "${PROJECT}/scripts/queue_luka_fullstat_readout.sh" "${COMMIT}" "${ROOT}" "${PARENT}" "${QUOTA}" --execute "${PLAN_HASH}"
```

Local synthetic checks do not constitute Oscar acceptance or scientific results.
Real Oscar checks occur inline before the fit. No remote jobs are submitted by
the implementation or local tests.

2026-10-30 validation: 117 passed, 1 skipped in 98.39s across new readout and
historical U100/multiview tests. Includes installed-Weaver CUDA full architecture
and all20 synthetic epochs, exact historical-kernel parity with readout disabled,
durable failure preservation and detached Bash flow. The Linux-only real HUP
test is skipped on Windows. CLI/AST, shell syntax and documentation links pass.

2026-10-30 replay correction: **120 passed, 1 skipped in 97.31s** across the
same suites, now including ragged multi-batch historical export/replay on CPU
and full-architecture CUDA, corruption rejection and v1 acceptance rejection.
The complete CUDA worker still trains/publishes all20 synthetic epochs. New
replay keeps rtol=2e-4 / atol=2e-5. AST, documentation links and diff whitespace
checks pass. Actual corrected Oscar execution remains pending.
