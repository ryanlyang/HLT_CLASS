# Recover the finished FullStat FUSION fit

This recovery uses the selected epoch14 FUSION model from failed job8220911,
exports its TRAIN embeddings in a fresh process, then trains the original
three-target student. It does not rerun either teacher. Job8220911 finished all
20epochs and selected validation readback, then received SIGBUS during export;
its42h15m elapsed time was below the72h limit. The cause remains unestablished.

See the [plan](plans/LUKA_FULLSTAT_MULTIVIEW_PLAN.md) and
[recovery contract](contracts/LUKA_FULLSTAT_MULTIVIEW_EXPORT.md). Keep original
a1eb1a74 and8ad0b381 worktrees and result roots. Their scientific dependencies
are loaded from original source; unrelated later edits cannot change this fit.
The only original-root write is an exclusive export-recovery claim after dry
review. Original fit/bank/report bytes remain untouched.

## Commit the adapter and push

Stage only the new package, entrypoints, test and new docs. Review and stage the
relevant hunks in the shared plan/handoff/source map separately; preserve other
work. Runtime changes outside this adapter are intentionally rejected relative
to4b9630fa00f137c4e48b749eb86714719b9d26cf. No new source commit has been created
by this implementation.

```powershell
git add -- src/hlt_classification/luka_fullstat_multiview_export scripts/luka_fullstat_multiview_export.py scripts/launch_luka_fullstat_multiview_export.sh sbatch/run_luka_fullstat_multiview_export.sh tests/test_luka_fullstat_multiview_export.py docs/LUKA_FULLSTAT_MULTIVIEW_EXPORT_RUNBOOK.md docs/contracts/LUKA_FULLSTAT_MULTIVIEW_EXPORT.md
git add -p -- docs/plans/LUKA_FULLSTAT_MULTIVIEW_PLAN.md docs/HANDOFF.md docs/LEGACY_SOURCE_MAP.md
git diff --cached --stat
git commit -m "Recover finished FullStat FUSION export without retraining"
git push origin main
git rev-parse HEAD
```

## Paste on Oscar

Use the full pushed40-character commit and at least22GiB of personally available
HOME quota. The old69.87GiB-used report suggested room, but check current quota.
The launcher detaches before expensive authentication, saves a full scheduler
dry review, then submits exactly two jobs without further prompts.

```bash
bash <<'BASH'
set -euo pipefail
checkquota
read -r -p 'Paste the pushed 40-character export-recovery commit: ' COMMIT </dev/tty
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
read -r -p 'Available HOME quota in whole GiB (minimum 22): ' QUOTA </dev/tty
[[ "${QUOTA}" =~ ^[1-9][0-9]*$ && "${#QUOTA}" -le 6 ]]
(( QUOTA >= 22 ))
REPO=/oscar/home/rlyang/hlt_classification/source/HLT_Classification_multiview_recovery_a1eb1a74
PROJECT=/oscar/home/rlyang/hlt_classification/source/HLT_Classification_multiview_export_${COMMIT:0:8}
ROOT=/oscar/home/rlyang/hlt_classification/checkpoints/luka_fullstat_multiview_export_${COMMIT:0:8}_r1
PARENT=/oscar/home/rlyang/hlt_classification/checkpoints/luka_fullstat_multiview_recovery_a1eb1a74_r1/campaign_spec.json
git -C "${REPO}" fetch origin main
git -C "${REPO}" merge-base --is-ancestor "${COMMIT}" origin/main
if [[ ! -e "${PROJECT}" ]]; then
    git -C "${REPO}" worktree add --detach "${PROJECT}" "${COMMIT}"
fi
bash "${PROJECT}/scripts/launch_luka_fullstat_multiview_export.sh" \
    "${COMMIT}" "${ROOT}" "${PARENT}" "${QUOTA}" \
    --authorize AUTHORIZE_FULLSTAT_MULTIVIEW_EXPORT_AUTOSUBMIT_V1
BASH
```

After **DETACHED**, SSH may disconnect; the foreground tail only displays the
log. **LAUNCH_COMPLETE** and **LAUNCHER_EXIT=0** confirm successful submission.
DETACHED alone does not mean jobs were submitted. Do not launch twice.

Jobs: lukafsmx_export_FUSION requests6h, then lukafsmx_train_multi3 requests24h
afterok. Both use6CPU/192000MiB/oneL40S. These are allocation ceilings, not
runtime promises. There is no separate preflight. The export checks the saved
fit, loads TRAIN views only and performs FP32 stress/timing admission. Student
retains its original inline checks and scientific recipe:
CE + (10/3)*(SmoothL1_U100 + SmoothL1_CONCAT + SmoothL1_FUSION),20epochs,batch128.

If authentication or scheduler dry review fails, no science is submitted. If
submission is ambiguous or partial, preserve `.launch/failure.json`, the
submission journals and `fusion_export_recovery_claim_v1` in the old root.
Do not remove claims, rerun the helper or manually resubmit the old teacher.
Technical export failure blocks the student. Finite low AUC never blocks it.

## Inspect and report

```bash
tail -n 60 -f "${ROOT}.launch/launcher.log"
squeue --me -o '%.18i %.12P %.45j %.2t %.10M %R'
```

Export phases are persisted under `export_FUSION/phases/`; its
`export_check.json` records timing, memory and selected checkpoint binding.
Workers enable Python fault handling for fatal-signal traces. This improves
diagnosis but does not guarantee every operating-system fault yields a trace.

After completion, use the new CLI (original multiview CLI does not know this
new execution contract):

```bash
export PROJECT_DIR="${PROJECT}" JC2_SITE=oscar_l40s
source "${PROJECT_DIR}/sbatch/luka_splithead_finalize_common.sh"
python -u -B -s "${PROJECT_DIR}/scripts/luka_fullstat_multiview_export.py" \
    results --spec "${ROOT}/campaign_spec.json"
```

It reports saved selected-checkpoint AUC, accuracy, R50/R30 and recoveries on
the same500k validation population, labeling CONCAT/FUSION privileged and U100-w5
historical. No new inference or test access. Full artifact authentication can
take several minutes. Original failed job remains failed in accounting; the
new export job has its own completion record.

## Validation and donor record

Local evidence (2026-11-03): **27 existing recovery tests passed**, then
**36 recovery/parity checks passed**. Coverage includes CPU/CUDA post-training
export failure recovery, unchanged originals, byte-identical target exports,
full20 student workers, full installed-Weaver FP32 export parity, native fusion
FP32/BF16 update parity, corrupted/mismatched artifacts and partial submissions.
Detached shell smoke passed; Windows uses a setsid fixture, not a real Linux
SSH hangup. Python/shell syntax, Bash snippets, links and whitespace passed.
Details are in [HANDOFF.md](HANDOFF.md).
No new remote Oscar/Tigris miniature or full-size recovery has run here.
Real fullstat export, scheduler acceptance and downstream training remain to
be established by the queued jobs. Local synthetic tests do not certify them.

Internal donor commit: a1eb1a746875a2697216d45a7c8ea4b6acdabc8a.
The adapter reuses `luka_fullstat_multiview/{model,training,targets,views,startup,
resources,worker,recovery,source,execution}.py` for fit validation, scientific
kernels, raw export, memory limits and scheduler/runtime conventions. Publication
and inspection orchestration in new worker.py is adapted from that worker;
submission/launch/shell patterns come from original multiview submission.py,
launch.py, scripts/launch_luka_fullstat_multiview_recovery.sh and
sbatch/run_luka_fullstat_multiview.sh. Supporting existing donors are
data/cache_contracts.py, cms_proxy_ladder/contracts.py, context_fusion/worker.py,
jetclass2_delphes/{model,reporting,execution}.py, luka_fullstat_readout/worker.py,
luka_fullstat_retained/worker.py, luka_fullstat_u100_embedding/banks.py,
luka_fullstat_fine_tail/worker.py, luka_fullsim/preflight_v2.py,
luka_u100_multiview/teachers.py, luka_fullstat_multiteacher/campaign.py,
luka_fullstat/submission.py, scouting/{hcwdl_exact_dag_submission,hcwdl_recovery}.py
and provenance.py. Tests adapt original tests/test_luka_fullstat_multiview*.py.
No external/Fresh_check code migration. Original scientific contracts stay v1;
the new execution contracts are separately versioned.
