# Oscar Multi3 plus OFFLINE matching

One fresh HLT-only student,3M TRAIN/500k validation,20 epochs. Reuse completed
U100, CONCAT and FUSION embeddings (total SmoothL1 weight10) and add pure-OFFLINE
jet identification with mix negatives, weight0.3 ramped over3 epochs. No teacher
retraining and no separate preflight job. The native inference model stays the
same. See the [plan](plans/LUKA_FULLSTAT_MULTI3_CONTRASTIVE_PLAN.md) and
[contract](contracts/LUKA_FULLSTAT_MULTI3_CONTRASTIVE.md).

## Commit and push

The runtime must be additive relative to execution baseline
`73abecfd95281246cea8e50bba3887a2737bac82`. Its new Hydra package leaves the
scientific donors at `a40ae47ba87ee23020b482c17d467e2a42282e8f` unchanged.
Stage only this campaign; review shared documentation separately so unrelated
in-progress work is preserved.

```powershell
git add -- src/hlt_classification/luka_fullstat_multi3_contrastive scripts/luka_fullstat_multi3_contrastive.py scripts/launch_luka_fullstat_multi3_contrastive.sh sbatch/run_luka_fullstat_multi3_contrastive.sh tests/test_luka_fullstat_multi3_contrastive.py docs/plans/LUKA_FULLSTAT_MULTI3_CONTRASTIVE_PLAN.md docs/contracts/LUKA_FULLSTAT_MULTI3_CONTRASTIVE.md docs/LUKA_FULLSTAT_MULTI3_CONTRASTIVE_RUNBOOK.md docs/LUKA_FULLSTAT_MULTI3_CONTRASTIVE_VALIDATION.md
git diff --cached --stat
git commit -m "Add FullStat Multi3 OFFLINE contrastive student"
git push origin main
git rev-parse HEAD
```

## One paste on Oscar

Check current personal HOME quota; at least8GiB available is required. This
locates the completed parent using job8305743, rather than guessing its retry
suffix. It uses an isolated worktree. Authentication and scheduler dry review
continue after detachment, then exactly one job is submitted.

```bash
bash <<'BASH'
set -euo pipefail
checkquota
read -r -p 'Pushed 40-character campaign commit: ' COMMIT </dev/tty
read -r -p 'Available HOME quota GiB (minimum 8): ' QUOTA </dev/tty
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
[[ "${QUOTA}" =~ ^[1-9][0-9]*$ && "${#QUOTA}" -le 6 ]]
(( QUOTA >= 8 ))
BASE=/oscar/home/rlyang/hlt_classification
REPO="${BASE}/source/HLT_Classification_multiview_export_ceb21461"
PROJECT="${BASE}/source/HLT_Classification_multi3_contrastive_${COMMIT:0:8}"
ROOT="${BASE}/checkpoints/luka_fullstat_multi3_contrastive_${COMMIT:0:8}_r1"
PARENT="$(/usr/bin/python3 -I -B - "${BASE}/checkpoints" <<'PY'
import json,sys
from pathlib import Path
found=[]
for path in Path(sys.argv[1]).glob('luka_fullstat_multiview_export_*/campaign_spec.json'):
    receipt=path.parent/'train_multi3/task_receipt.json'
    if receipt.is_file() and str(json.loads(receipt.read_text()).get('job_id'))=='8305743':
        found.append(path)
if len(found)!=1:
    raise SystemExit('Expected exactly one completed Multi3 job8305743; inspect saved roots.')
print(found[0])
PY
)"
git -C "${REPO}" fetch origin main
git -C "${REPO}" merge-base --is-ancestor "${COMMIT}" origin/main
if [[ ! -e "${PROJECT}" ]]; then
    git -C "${REPO}" worktree add --detach "${PROJECT}" "${COMMIT}"
fi
bash "${PROJECT}/scripts/launch_luka_fullstat_multi3_contrastive.sh" \
    "${COMMIT}" "${ROOT}" "${PARENT}" "${QUOTA}" \
    --authorize AUTHORIZE_FULLSTAT_MULTI3_CONTRASTIVE_AUTOSUBMIT_V1
BASH
```

After **DETACHED**, you can disconnect. The foreground tail is only a view.
**LAUNCH_COMPLETE** and **LAUNCHER_EXIT=0** confirm submission; DETACHED alone
does not. Do not launch twice. Keep the original teacher/source/result roots.

The single `lukafsmc_train_multi3_mix` job requests48h,6CPUs,192000MiB and one
L40S. That is an allocation ceiling, not a runtime estimate. Integrated checks
measure the complete sampler/update on the actual GPU before the fresh fit.
Three persistent CPU workers and one-batch GPU lookahead reduce candidate overhead.
They save diagnostics and stop early if the measured workload cannot fit; they
never reduce negatives, tokens, precision or batch size. No guaranteed AUC gain
or Oscar runtime speedup is claimed from local checks.

## Watch and report

```bash
tail -n 60 -f "${ROOT}.launch/launcher.log"
squeue --me -o '%.18i %.45j %.2t %.10M %.12l %R'
grep -E '^MULTI3 MIX (epoch=|progress|measured)' "${ROOT}"/train_multi3_mix-*.out | tail -n 30
```

After completion:

```bash
export PROJECT_DIR="${PROJECT}" JC2_SITE=oscar_l40s
source "${PROJECT_DIR}/sbatch/luka_splithead_finalize_common.sh"
python -u -B -s "${PROJECT_DIR}/scripts/luka_fullstat_multi3_contrastive.py" \
    results --spec "${ROOT}/campaign_spec.json"
```

This reads saved validation predictions and reports AUC/accuracy/R50/R30 and
gap recoveries for M0HLT, OFFLINE, U100-w5, historical Multi3 and the new model.
It performs artifact verification, not new inference or test evaluation.

## Explicit recovery

Full states and receipts are saved at each completed epoch. For an owned
terminal failed/cancelled/timed-out job with at least one completed epoch, use
the **same commit/worktree**, a fresh ROOT, the same original Multi3 PARENT,
and append `--resume-spec /absolute/failed/root/campaign_spec.json` to the
launcher command. The new attempt authenticates/copies completed epochs and
restores model, optimizer, RNG, LR and negative-sampler position. At20 completed
epochs it only republishes the selected result after integrated checks.

Do not remove a claim or repeat an ambiguous submission. If failure happened
before any completed epoch, or requires different source, inspect its saved
diagnostics and journals first. This workflow does not silently retry.
