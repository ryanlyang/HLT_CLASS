# FullStat fast attention: commit, benchmark, then replace

This reuses the already exported D033 bank and trains **only ATTN_D000**.
It compares with the original COARSE_D000 and controls. No repeated logit fit.
Scientific recipe: unchanged3M/500k,20epochs,batch128,0.1attentionKL. Test stays
sealed. Real Oscar speedup is unmeasured until the new gate completes.

Keep the original62c9 attention worktree, original9bc FullStat worktree, cache,
export bank and results intact. Do not edit their specs/source or clear partial
outputs. The old running job has no epoch checkpoint; this replacement starts
fresh from the same initialization. New fast runs save every completed epoch.

## 1. Commit and push locally

Stage the additive implementation explicitly; do not `git add .`. Review the
HANDOFF/LEGACY_SOURCE_MAP changes separately because these files may contain
unrelated user edits.

```powershell
git add src/hlt_classification/luka_fullstat_attention_fast scripts/luka_fullstat_attention_fast.py scripts/queue_luka_fullstat_attention_fast.sh sbatch/run_luka_fullstat_attention_fast.sh tests/test_luka_fullstat_attention_fast.py docs/plans/LUKA_FULLSTAT_FAST_ATTENTION_PLAN.md docs/contracts/LUKA_FULLSTAT_FAST_ATTENTION.md docs/LUKA_FULLSTAT_FAST_ATTENTION_RUNBOOK.md
git diff --cached --stat
git commit -m "Add science-identical resident-bank FullStat attention recovery"
if ($LASTEXITCODE -ne 0) { throw "Inspect git status before continuing" }
git push origin HEAD:main
if ($LASTEXITCODE -ne 0) { throw "Push failed" }
git rev-parse HEAD
```

No runtime files outside the new fast-attention package/entrypoints may change
relative to donor`cfb3a1587a6739b1949be9e65ac80f7ba9f8ead9` in this execution.

## 2. Oscar: pin a fresh worktree and dry-review one GPU gate

Run on an Oscar login node. These variables are ordinary shell variables, not
environment identities. Check that you personally have at least8GiB spare
durable result space before declaring `QUOTA=8`.

```bash
read -r -p 'Pushed40-character fast-attention commit: ' COMMIT
MAIN=/oscar/home/rlyang/hlt_classification/source/HLT_Classification
PROJECT="/oscar/home/rlyang/hlt_classification/source/HLT_Classification_fullstat_fast_${COMMIT:0:8}"
ROOT="/oscar/home/rlyang/hlt_classification/checkpoints/luka_fullstat_fast_attention_${COMMIT:0:8}_r1"
ORIGINAL=/oscar/home/rlyang/hlt_classification/checkpoints/luka_fullstat_last_attention_62c9b2df_r1/science/campaign_spec.json
QUOTA=8

[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]] &&
git -C "${MAIN}" fetch origin main &&
git -C "${MAIN}" merge-base --is-ancestor "${COMMIT}" origin/main &&
git -C "${MAIN}" worktree add --detach "${PROJECT}" "${COMMIT}"

checkquota
bash "${PROJECT}/scripts/queue_luka_fullstat_attention_fast.sh" \
    "${COMMIT}" "${ROOT}" "${ORIGINAL}" "${QUOTA}" gate
```

If the worktree already exists, preserve it: check its exact HEAD and clean
status before reusing it. Never reset or delete it. Authentication can take a
few minutes. The dry review submits nothing.

Review the one L40S/6CPU/180000MiB/3h preflight job, then submit its printed
**plan** content_hash (not a spec/bank hash):

```bash
read -r -p 'Reviewed PLAN content_hash: ' PLAN_HASH
LOG="${ROOT}/gate-submit.log"
test ! -e "${LOG}" && nohup bash "${PROJECT}/scripts/queue_luka_fullstat_attention_fast.sh" \
    "${COMMIT}" "${ROOT}" "${ORIGINAL}" "${QUOTA}" gate --execute "${PLAN_HASH}" \
    >"${LOG}" 2>&1 </dev/null &
```

Read `tail -f "${ROOT}/gate-submit.log"` and confirm a submitted job ID. Ctrl-C
stops tail only. Do not launch a second helper if the first result is ambiguous.
The3h limit is a ceiling, not the expected benchmark duration.

The old180000MiB job may prevent simultaneous gate scheduling under your QOS.
If you choose to stop it before measurement to free resources, first verify
`scontrol show job 7618255` identifies the original62c9 ATTN job, then cancel
**only that ID**. This forfeits its unsaved live fit, not its exported bank.
The helper never cancels anything automatically.

## 3. Review measured speed, then dry-review and queue science

After the gate is `COMPLETED|0:0`, inspect `${ROOT}/preflight/receipt.json` and
the end of its preflight log. It reports resident bytes, old/resident target
loader timings, measured optimizer-step times, projected epoch-training time,
full500kvalidation time, exact parity and conservative walltime. A loader-only
ratio is **not** an end-to-end speedup. Keep/stop the old job based on this
evidence; do not assume a near-logit runtime beforehand.

```bash
bash "${PROJECT}/scripts/queue_luka_fullstat_attention_fast.sh" \
    "${COMMIT}" "${ROOT}" "${ORIGINAL}" "${QUOTA}" science
```

Review exactly two jobs: ATTN_D000 and dependent CPU aggregate. If the old job
is still running and you are replacing it, verify its exact ID/source as above
and cancel that one job. Do not cancel the completed export or old results.

```bash
read -r -p 'Reviewed science PLAN content_hash: ' PLAN_HASH
LOG="${ROOT}/science-submit.log"
test ! -e "${LOG}" && nohup bash "${PROJECT}/scripts/queue_luka_fullstat_attention_fast.sh" \
    "${COMMIT}" "${ROOT}" "${ORIGINAL}" "${QUOTA}" science --execute "${PLAN_HASH}" \
    >"${LOG}" 2>&1 </dev/null &
```

Inspect the submission log. No automatic follow-up, duplicate fits or restart.
Live training prints `FAST progress` every128updates and the original-format
epoch validation-AUC line after each full validation pass.

## Results and an explicitly requested restart

After completion, in the pinned environment (`PROJECT_DIR` and`JC2_SITE` set,
source the absolute `sbatch/jetclass2_delphes_common.sh`, export CUBLAS as in the
worker), run:

```bash
python -u -B -s "${PROJECT}/scripts/luka_fullstat_attention_fast.py" results \
    --spec "${ROOT}/science/campaign_spec.json"
```

For a future interrupted **fast** fit, preserve all attempt files. Inspect the
exact science jobs and cancel only its blocked dependent aggregate if needed.
Once all previous registered jobs are terminal, choose an explicit committed
`science/attempts/JOBID/checkpoints/epoch-NN.json` (not a bare`.pt`). Do not use
mtime to identify an epoch. Create a fresh reviewed recovery:

```bash
python -u -B -s "${PROJECT}/scripts/luka_fullstat_attention_fast.py" create-resume \
    --spec "${ROOT}/science/campaign_spec.json" \
    --epoch-receipt "${EPOCH_RECEIPT}" --root "${ROOT}/science/recoveries/r1"
python -u -B -s "${PROJECT}/scripts/luka_fullstat_attention_fast.py" submit \
    --spec "${ROOT}/science/recoveries/r1/resume_spec.json"
```

Review its two-job plan, then repeat`submit` with`--execute --plan-hash HASH`
and`--authorization 'AUTHORIZE FULLSTAT FAST ATTENTION EXACT PLAN'`.
Resumption requires the same source/environment/gate/bank and restores AdamW,
RNG, history and best state. A complete20epoch report is still mandatory.
