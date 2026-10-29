# Oscar FullStat smooth-w10: one new DE000, no standalone preflight

One fresh HLT-only, 20-epoch fit with CE0.5/K1/E40 and neighborhood weight 10. Exact teachers
come from completed non-relational `luka_splithead_finalize_retry_5e704ef4_r1`.
No M0HLT/DE022/DE011 targets. Historical DE000 is a reference, not a matched control.

The two-job queue is CPU target construction → `afterok` GPU training. Both jobs
are submitted together after source/parent/quota authentication and a saved dry
review. In-worker checks happen automatically; you do not need to await or queue
a preflight. No current job is canceled or changed. The inherited final test
stays sealed.

## Publish only these changes locally

Review `git diff` and the new files. Do not use `git add .` in this workspace:
many unrelated documents and temporary files already exist. Stage the new
`luka_fullstat_neighborhood` namespace, its three scripts and Slurm wrapper,
two test files, plan/contract/runbook (and the referenced smooth-w10 handoff if
not already published), and only the matching HANDOFF/donor-map
hunks. Commit and push them. Record the full 40-character commit.

## One detached launch after checkout

On Oscar, use a clean detached worktree at that exact pushed commit. Do not modify
any old source snapshot. The paths below are explicit new outputs; change the
suffix to the short new commit when preparing the launch. Never overwrite a root
that exists. Confirm quota with `checkquota` immediately before launching.

Durable selected models/reports go under home. Put the new large, reproducible
neighborhood bank under your existing scratch `hlt_classification` directory.
No old data or model is moved. Example headroom declarations: 8 GiB for home,
100 GiB for scratch **only if personally available**. These are declarations, not
promises of actual usage: exact bank and RAM requirements are computed from
authenticated TRAIN support before jobs are submitted.

```bash
bash "${PROJECT}/scripts/launch_luka_fullstat_neighborhood.sh" \
  "${COMMIT}" \
  "/oscar/home/rlyang/hlt_classification/checkpoints/luka_fullstat_neighborhood_${COMMIT:0:8}_r1" \
  "/oscar/home/rlyang/hlt_classification/checkpoints/luka_splithead_finalize_retry_5e704ef4_r1/campaign_spec.json" \
  8 \
  "/oscar/scratch/rlyang/hlt_classification/luka_fullstat_neighborhood_${COMMIT:0:8}_r1" \
  100 \
  --authorize AUTHORIZE_FULLSTAT_NEIGHBORHOOD_AUTOSUBMIT_V1
```

`PROJECT` must be the new worktree's absolute path and `COMMIT` its exact pushed
40-character hash. The command immediately attaches a log tail while detached
authentication/submission continues. After `Detached launcher PID` and `Log`
appear, Ctrl-C or SSH disconnect stops only the log view. Submission can still
fail its checks: reconnect to inspect the saved log. `LAUNCH_COMPLETE` means
Slurm owns both accepted IDs. There is no continuing login controller.

Manual alternative: use `queue_luka_fullstat_neighborhood.sh` with the same first
six arguments (no authorize token). It prints `PLAN_HASH` without submission;
repeat with `--execute HASH` to submit exactly the reviewed two-job plan.

If launch fails, preserve `.launch/failure.json`, submission claims and journals.
Do not delete them or blindly relaunch. If target build fails, the dependent fit
does not run. Partial banks stay on disk for diagnosis. There is no automatic
retry or recovery that silently substitutes teachers or restarts a fit.

## Monitor and report

```bash
squeue --me -o '%.18i %.12P %.40j %.2t %.10M %R'
tail -n 40 -f "${ROOT}.launch/launcher.log"
```

Training emits `FULLSTAT_NEIGHBOR ... epoch=... auc=...` and progress every 128
batches, including elapsed seconds. Results are provisional until a completed
task receipt exists. CPU preparation logs show per-file row coverage/time.

To read completed AUC/recovery and R50/R30, from the pinned worktree:

```bash
export PROJECT_DIR="${PROJECT}" JC2_SITE=oscar_l40s
source "${PROJECT_DIR}/sbatch/luka_splithead_finalize_common.sh"
python -u -B -s "${PROJECT_DIR}/scripts/luka_fullstat_neighborhood.py" results \
  --spec "${ROOT}/campaign_spec.json"
```

Keep the new scratch bank, parent sources/receipts and borrowed teacher banks
while the campaign needs them. Scratch is not a backup. Selected native model,
completed-fit snapshot and validation outputs are durable in the home result root.

Local validation is not remote acceptance. No actual Oscar job or source push
is performed by implementing these files. The first real run must still pass
the exact runtime/allocation checks and integrated model checks.
