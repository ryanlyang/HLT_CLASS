# Oscar FullStat U100-w5 meta-embedding continuation

Authority: [plan](plans/LUKA_FULLSTAT_META_EMBEDDING_PLAN.md),
[v1 contracts](contracts/LUKA_FULLSTAT_META_EMBEDDING.md).
This is NOT the SPORC OFFLINE-w5 30-epoch campaign.

One shared setup followed by two independent20-additional-epoch fits:
`fixed` versus `meta`. Both start from completed FullStat U100-w5, not scratch;
teacher is its original COARSE_U100. CE1/E5/no student logit KD. Same500k
validation; final test remains sealed. Setup retains95% of original TRAIN for
real updates and5% for meta-query feedback, calibrates one missing predictor
head, and measures native/five-step/full-epoch technical acceptance. No separate
preflight job, but useful setup can take hours. These are not free checks.

## Before queueing

Commit and push only the additive `luka_fullstat_meta_embedding` namespace,
its CLI, queue/launch helpers, Slurm wrapper, two test files and three documents.
Review scoped HANDOFF/LEGACY_SOURCE_MAP edits; do not stage unrelated work.
Do not modify historical runtime to force source compatibility. Local readiness
does not mean Oscar admission or a scientific improvement is already established.

Preserve the complete original U100-w5 campaign and its source/data lineage:

```text
/oscar/home/rlyang/hlt_classification/checkpoints/luka_fullstat_u100_embedding_w5_7ea8b0c6_r1/campaign_spec.json
```

Its selected native student, original U100 teacher, reports, predictions, bank,
prepared caches and old preflight/source dependencies are authenticated read-only.
No large target bank is duplicated. Need at least8GiB personally available home
headroom (filesystem free space alone does not establish personal quota).
Uses existing repaired runtime:
`/oscar/scratch/rlyang/hlt_classification/runtime_repair_v1/atlas_kd_oscar`.

## One detached launch on Oscar

Paste the following on an Oscar login node after pushing the implementation.
Enter the FULL40-character pushed commit. Confirm quota before authorizing.
This authorizes exactly setup + fixed + meta, after a complete saved dry review;
it does not authorize cancellation, cleanup, other fits or another retry.

```bash
bash <<'BASH'
set -euo pipefail
read -r -p 'Pushed 40-character FullStat meta commit: ' COMMIT </dev/tty
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
PROJECT="/oscar/home/rlyang/hlt_classification/source/HLT_Classification_fullstat_meta_${COMMIT:0:8}"
ROOT="/oscar/home/rlyang/hlt_classification/checkpoints/luka_fullstat_meta_u100_w5_${COMMIT:0:8}_r1"
PARENT=/oscar/home/rlyang/hlt_classification/checkpoints/luka_fullstat_u100_embedding_w5_7ea8b0c6_r1/campaign_spec.json
test -r "${PARENT}"
REPO="$(/usr/bin/python3 -I -B -c 'import json,sys; print(json.load(open(sys.argv[1]))["project_dir"])' "${PARENT}")"
git -C "${REPO}" fetch origin main
git -C "${REPO}" merge-base --is-ancestor "${COMMIT}" origin/main
if [[ -e "${PROJECT}" ]]; then
    test "$(git -C "${PROJECT}" rev-parse HEAD)" = "${COMMIT}"
    test -z "$(git -C "${PROJECT}" status --porcelain --untracked-files=all)"
else
    git -C "${REPO}" worktree add --detach "${PROJECT}" "${COMMIT}"
fi
checkquota
read -r -p 'Personally available result headroom in GiB (at least 8): ' QUOTA </dev/tty
[[ "${QUOTA}" =~ ^[1-9][0-9]*$ && "${#QUOTA}" -le 6 ]]
(( QUOTA >= 8 ))
read -r -p 'Authorize shared setup plus BOTH 20-epoch continuations? Type YES: ' YES </dev/tty
[[ "${YES}" == YES ]]
bash "${PROJECT}/scripts/launch_luka_fullstat_meta_embedding.sh" \
    "${COMMIT}" "${ROOT}" "${PARENT}" "${QUOTA}" \
    --authorize AUTHORIZE_FULLSTAT_META_EMBEDDING_AUTOSUBMIT_V1
BASH
```

Once `DETACHED: launcher PID ...` and the log path appear, you can disconnect.
The foreground `tail` is only a view. Authentication, saved dry review and
authorized submission continue without input. Successful log ends
`LAUNCH_COMPLETE`, with exact three job IDs. No login followup controller is
needed after submission. Disconnect protection does not cover host reboot,
administrative cleanup or node failure.

If `STOPPED`/`LAUNCHER_EXIT=1` appears, preserve `.launch/failure.json`, root and
submission journals. Do not repeat, delete claims or choose another root to
bypass an ambiguous submission. No automatic retry exists. Setup failure
prevents dependent fits. The two fits are independent: poor finite outcomes or
failure of one do not cancel the other. Resource ceilings:12h setup,24h fixed,
48h meta, each L40S/6CPU/180000MiB. Actual timings are measured in setup.

Manual alternative (do NOT mix with detached launch):

```bash
bash "${PROJECT}/scripts/queue_luka_fullstat_meta_embedding.sh" "${COMMIT}" "${ROOT}" "${PARENT}" "${QUOTA}"
# Inspect saved plan; only then use its printed PLAN_HASH:
bash "${PROJECT}/scripts/queue_luka_fullstat_meta_embedding.sh" "${COMMIT}" "${ROOT}" "${PARENT}" "${QUOTA}" --execute "${PLAN_HASH}"
```

## Progress and results

```bash
squeue --me -o '%.18i %.12P %.40j %.2t %.10M %R'
# ROOT is the exact path printed by the launcher:
tail -n 60 -f "${ROOT}.launch/launcher.log"
```

Fit logs are `fixed-<jobid>.out` and `meta-<jobid>.out`. Epoch markers are
`FULLSTAT_META_EMBED arm=... epoch=.../20 auc=...`; progress shows real batches,
not a live walltime estimate. Meta events add uneven extra work. Epoch numbers
are additional continuation epochs, not comparable with the historical model's
epoch1 starting from scratch.

Authenticated saved results, no new inference or test access:

```bash
bash <<'BASH'
set -euo pipefail
read -r -p 'Exact FullStat meta study root: ' ROOT </dev/tty
SPEC="${ROOT}/campaign_spec.json"
test -s "${SPEC}"
PROJECT_DIR="$(/usr/bin/python3 -I -B -c 'import json,sys; print(json.load(open(sys.argv[1]))["project_dir"])' "${SPEC}")"
export PROJECT_DIR JC2_SITE=oscar_l40s
source "${PROJECT_DIR}/sbatch/luka_splithead_finalize_common.sh"
python -u -B -s "${PROJECT_DIR}/scripts/luka_fullstat_meta_embedding.py" results --spec "${SPEC}"
BASH
```

Results include historical M0HLT/OFFLINE/U100-w5, each completed continuation,
recovery and delta versus matched FIXED. Meta-query labels were seen by the old
initializer; this is not independent holdout evidence. Main comparison is META
versus FIXED, with additional meta compute/query supervision disclosed.

Artifacts: `setup/{initializer,checks,measurement,receipt}.json`,
`initial/{student.pt,classifier.pt,statistics.npz}`, disposable `setup/probe_*`,
`{fixed,meta}/fit/{selected.pt,training_state.pt,completed_training.json,
training_report.json,validation_predictions.npz}`, final task receipts.
Selected native/full states and completed report are saved before publication
checks. Full state includes predictor, adapter, optimizers and RNG at the selected
epoch; this is not an automatic rolling restart checkpoint.

## Local tests

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONPATH='src;.tmp/fullstat-neighborhood-weaver-v1'
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
& 'C:\Users\22rya\miniconda3\envs\tagging-hlt\python.exe' -B -m pytest -q -p no:cacheprovider tests/test_luka_fullstat_meta_embedding.py tests/test_luka_fullstat_meta_embedding_launch.py tests/test_luka_meta_embedding.py tests/test_luka_fullstat_u100_embedding.py
```

Use a scientific environment with installed Weaver0.5.3; the workspace target
above is the local test installation, not an Oscar runtime path. CUDA miniature
uses production workers, smaller data/depth and mocked remote boundaries. It
does not certify L40S resource use. Linux-only real HUP test skips on Windows.
