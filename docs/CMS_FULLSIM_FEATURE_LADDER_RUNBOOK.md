# CMS FullSim 200k/50k feature-interface ladder

Scientific authority: [plan](plans/CMS_FULLSIM_SHARED_FEATURE_LADDER_200K_PLAN.md).
Contract: [v1](contracts/CMS_FULLSIM_FEATURE_LADDER.md).

This is a new campaign using the real paired CMS files, not either proxy dataset.
Default is **SHARED17 only**, nine fresh fits, five teacher reducers and two
closure jobs. `--arms both` adds the native-feature control (32 science jobs).
Do not opt into that extra compute without choosing it explicitly.

## Source and creation

First commit/push the new files listed in HANDOFF. Do not stage unrelated work.
On **sporcsubmit**, use a new detached worktree at that exact pushed commit.
Replace `COMMIT` with the actual full hash; do not use a historical donor hash.

```bash
COMMIT=YOUR_NEW_40_CHARACTER_COMMIT
MAIN=/home/ryreu/atlas/HLT_Classification
PROJECT="/home/ryreu/atlas/HLT_Classification_cms_feature_${COMMIT:0:8}"
ROOT="${MAIN}/checkpoints/cms_feature_200k50k_${COMMIT:0:8}_r1"

git -C "${MAIN}" fetch origin main
git -C "${MAIN}" merge-base --is-ancestor "${COMMIT}" origin/main
git -C "${MAIN}" worktree add --detach "${PROJECT}" "${COMMIT}"

export PROJECT_DIR="${PROJECT}" JC2_SITE=sporc_a100_debug
source "${PROJECT}/sbatch/jetclass2_delphes_common.sh"
```

The known original native-CMS preparation was
`${MAIN}/checkpoints/cms_salience_learned_dense_500k_f2e8a374_r1/campaign_spec.json`.
Use it **only to locate its existing raw root and split**; no fit, teacher,
matching, preflight, or 500k population is imported. Read these values:

```bash
CMS_SOURCE_SPEC="${MAIN}/checkpoints/cms_salience_learned_dense_500k_f2e8a374_r1/campaign_spec.json"
python -s - "${CMS_SOURCE_SPEC}" <<'PY'
import json, sys
from pathlib import Path
s = json.loads(Path(sys.argv[1]).read_text())
print('CMS data root:', s['data_root'])
print('Split manifest:', s['split_manifest']['path'])
PY
```

Set `CMS_DATA` and `CMS_SPLIT` to those **printed existing paths**. Do not make
a new split. The historical default raw root was
`/home/ryreu/cms/data/ScoutingAK8_native_compact/2024/train`; an existing source
spec and file checks take precedence if storage has moved. Then:

```bash
python -u -s "${PROJECT}/scripts/cms_fullsim_feature_ladder.py" create \
  --source-commit "${COMMIT}" --root "${ROOT}" \
  --data-root "${CMS_DATA}" --split-manifest "${CMS_SPLIT}" --arms shared
SPEC="${ROOT}/campaign_spec.json"
```

Creation is metadata-only, requires a fresh output root and clean pushed source,
and submits nothing. It freezes 200k/50k, not a user-adjustable population size.

## Gate: one reviewed submission, including preparation and GPU preflight

```bash
bash "${PROJECT}/scripts/queue_cms_fullsim_feature_ladder.sh" "${SPEC}" gate
```

Review the printed PLAN content_hash and complete commands. CPU tasks select
rows, match each source and lock the foundation; GPU preflight depends on that
lock. All jobs use debug; only preflight requests an A100. Submit the exact plan:

```bash
PLAN_HASH=THE_ACTUAL_64_CHARACTER_PLAN_CONTENT_HASH
bash "${PROJECT}/scripts/queue_cms_fullsim_feature_ladder.sh" "${SPEC}" gate "${PLAN_HASH}"
```

For SSH resilience, redirect this command into a campaign-local log and launch
it with `nohup ... >"${LOG}" 2>&1 </dev/null &`; use `mktemp` for the log. Do
not mistake a helper PID for a Slurm job: inspect `submission_gate/live.json`.
An incomplete `live.claim` requires inspection of the journal before any retry.
Never delete a claim or resubmit just because SSH disconnected.

GPU preflight builds complete RAM caches, checks installed-Weaver FP32 parity,
three batch-256 longest-U000 KD updates, one real full-population CE pass,
selected-teacher inference and a short true-probability KD miniature. It records
the missing-error/PID diagnostics, resources, environment and measured timing.
The new campaign cannot claim runtime acceptance from historical jobs.

## Science and current results

After the requested preflight(s) succeed:

```bash
bash "${PROJECT}/scripts/queue_cms_fullsim_feature_ladder.sh" "${SPEC}" science
```

Review its **new** plan hash, then run the same command with that hash as the
third argument. This queues the entire 16-job direct/coarse science DAG for one
arm (32 for both). No dense ladder or final-test job exists. There is no need
to queue individual rungs. Science checks saved acceptance, full source,
teacher joins and measured job limits. Failed jobs do not trigger blanket
retry, cleanup, cancellation, or source changes.

```bash
python -u -s "${PROJECT}/scripts/cms_fullsim_feature_ladder.py" status --spec "${SPEC}"
python -u -s "${PROJECT}/scripts/cms_fullsim_feature_ladder.py" results --spec "${SPEC}"
squeue --me -o "%.18i %.12P %.48j %.2t %.10M %R"
```

`status` reports authenticated output completion, not live scheduler state.
`results` works before aggregate finishes. It prints accuracy, AUC, AUC recovery,
Xbb/Xcc QCD rejection@50%, and coarse-minus-direct endpoint differences. Full
15-class statistics are in each selected training report. Zero background passes
are censored, never reported as a measured infinite rejection. Final test stays
sealed. A null or negative recovery is a scientific result, not a job failure.
