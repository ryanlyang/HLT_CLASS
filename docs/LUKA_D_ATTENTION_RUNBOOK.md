# Queue the full aligned-attention D ladder on SPORC debug

Authority: [plan](plans/LUKA_D_ATTENTION_PLAN.md),
[contract](contracts/LUKA_D_ATTENTION.md),
[scientific explanation](LUKA_D_LADDER_ATTENTION_DISTILLATION.md).

This is a new campaign. It leaves the running fusion/fixed30 jobs untouched.
The user requested the complete U100 -> D066 -> D033 -> D000 ladder, not a
one-step science pilot. There are six fresh fixed30/batch128 fits: ordinary KD
and attention KD, paired coordinate seeds and a common regular U100 anchor.
The auxiliary coefficient is frozen at 0.1, last particle block, head-mean
pre-dropout probabilities. No independent test exists for this trial ALLJETS
population (299,234 TRAIN / 74,895 validation); original 50k is secondary reporting.

## Prerequisites

- Original fixed30 source `74aa58c582ff6f497a2c6d91c503f5489fac3e9a`, its
  original pinned worktree, prepared caches and provenance must still exist.
- `COARSE_U100`, M0HLT, OFFLINE and DIRECT_D000 must have completed and published
  valid receipts. This does not wait for the fusion branch to finish. Gate
  creation explicitly stops if an anchor/reference is still pending.
- Keep the original installed SPORC environment. No upgrade/reinstall is part
  of these commands. Parent authentication runs in its original source.
- Review your personal remaining quota, not just `df` filesystem availability.
  Attention banks are dense on real tokens but do not store padding. Gate
  measures `4*sum(Npart**2)` and requires seven such bank equivalents plus
  probability/checkpoint overhead and 5 GiB reserve. This includes retained gate
  probes. No automatic cleanup occurs.
- New source must be clean, committed, pushed, fetched and pinned. No broad
  `git add .`, reset, force-push or editing an active execution worktree.

## 1. Commit and push on Windows

Review the new isolated files and documentation. Existing unrelated changes in
HANDOFF/LEGACY_SOURCE_MAP must be reviewed separately, not swept into this commit.
Use `git add -p -- docs/HANDOFF.md docs/LEGACY_SOURCE_MAP.md` to stage only the
new attention implementation/donor entries if those documents also contain
unrelated changes. Review the entire staged diff before committing.

```powershell
git status --short -- src/hlt_classification/luka_attention scripts/luka_attention.py scripts/queue_luka_attention.sh sbatch/run_luka_attention.sh tests/test_luka_attention.py docs/LUKA_D_LADDER_ATTENTION_DISTILLATION.md docs/LUKA_D_ATTENTION_RUNBOOK.md docs/plans/LUKA_D_ATTENTION_PLAN.md docs/contracts/LUKA_D_ATTENTION.md
git add -- src/hlt_classification/luka_attention scripts/luka_attention.py scripts/queue_luka_attention.sh sbatch/run_luka_attention.sh tests/test_luka_attention.py docs/LUKA_D_LADDER_ATTENTION_DISTILLATION.md docs/LUKA_D_ATTENTION_RUNBOOK.md docs/plans/LUKA_D_ATTENTION_PLAN.md docs/contracts/LUKA_D_ATTENTION.md
git diff --cached --stat
git diff --cached --check
if ($LASTEXITCODE -ne 0) { throw "Review staged changes before committing" }
git commit -m "Add matched aligned-attention D ladder and SPORC acceptance"
if ($LASTEXITCODE -ne 0) { throw "Commit did not complete; inspect before proceeding" }
git push origin HEAD:main
if ($LASTEXITCODE -ne 0) { throw "Push failed; do not force-push" }
git rev-parse HEAD
```

If the changes were already committed, inspect `git log -1 --stat` and use that
commit rather than making an empty commit. A non-fast-forward rejection needs
inspection/integration; it is not permission to overwrite remote history.

## 2. Create a pinned worktree on SPORC and review the gate

Run in Bash on **sporcsubmit**, not Tigris or Oscar. Paste the actual new commit
and an honest integer GiB estimate of personally available quota at the prompts.

```bash
read -r -p 'New pushed 40-character commit: ' COMMIT
read -r -p 'Personally available quota, integer GiB: ' QUOTA
MAIN=/home/ryreu/atlas/HLT_Classification
PROJECT="/home/ryreu/atlas/HLT_Classification_luka_attention_${COMMIT:0:8}"
ROOT="${MAIN}/checkpoints/luka_attention_${COMMIT:0:8}_r1"
PARENT="${MAIN}/checkpoints/luka_fixed30_74aa58c5_r1/science/campaign_spec.json"
(
set -euo pipefail
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ && "${QUOTA}" =~ ^[1-9][0-9]*$ ]]
test -f "${PARENT}"
git -C "${MAIN}" fetch origin main
git -C "${MAIN}" merge-base --is-ancestor "${COMMIT}" origin/main
if [[ -e "${PROJECT}" ]]; then
    test "$(git -C "${PROJECT}" rev-parse HEAD)" = "${COMMIT}"
    test -z "$(git -C "${PROJECT}" status --porcelain --untracked-files=all)"
else
    git -C "${MAIN}" worktree add --detach "${PROJECT}" "${COMMIT}"
fi
bash "${PROJECT}/scripts/queue_luka_attention.sh" "${COMMIT}" "${ROOT}" "${PARENT}" "${QUOTA}" gate
)
```

If the worktree already exists, do not remove it. Verify its HEAD equals COMMIT
and `git status --porcelain --untracked-files=all` is empty, then rerun just the
helper. If a campaign directory exists without a gate spec, preserve it and
inspect the failure; do not overwrite it or automatically choose an r2.

The dry plan contains **one GPU preflight**, 6 CPUs, 128,000 MiB, one A100,
8-hour ceiling, debug / account `reu-aisocial` / QoS `qos_tier3`.
This is a technical ceiling, not an ETA.
It performs installed native FP32/BF16 parity, real support stress in each role,
target-bank readbacks, and both complete three-rung full-population one-pass
technical rehearsals. These models never become scientific teachers.

## 3. Submit the reviewed gate

Use the plan's `content_hash`, not the gate/anchor hash. This helper is durable
under SSH disconnects; Ctrl-C on tail does not stop it.

```bash
read -r -p 'Reviewed gate PLAN content_hash: ' PLAN_HASH
(
set -euo pipefail
[[ "${PLAN_HASH}" =~ ^[0-9a-f]{64}$ ]]
LOG="$(mktemp "${MAIN}/checkpoints/luka-attention-gate.XXXXXX.log")"
nohup bash "${PROJECT}/scripts/queue_luka_attention.sh" \
  "${COMMIT}" "${ROOT}" "${PARENT}" "${QUOTA}" gate --execute "${PLAN_HASH}" \
  >"${LOG}" 2>&1 </dev/null &
printf 'Helper PID: %s\nLog: %s\n' "$!" "${LOG}"
tail -f "${LOG}"
)
```

Require a numeric preflight job ID and `QUEUE_HELPER_EXIT=0`. The helper PID is
not the Slurm job. A partial/ambiguous submit is intentionally blocked by its
exclusive claim and journal; do not launch another helper to bypass it.

## 4. After preflight COMPLETED/0:0, review and submit all science

```bash
bash "${PROJECT}/scripts/queue_luka_attention.sh" \
  "${COMMIT}" "${ROOT}" "${PARENT}" "${QUOTA}" science
```

Science creation authenticates exact gate accounting/journal, source, all
technical payloads and measured budgets. The plan has **12 jobs**: one common
U100 export, six fits, four intermediate exports, and aggregate. Dependencies
carry each arm's own teacher. Measured train/export ceilings must fit 24 hours; no
quality thresholds, automatic epoch reduction or constituent truncation.

```bash
read -r -p 'Reviewed science PLAN content_hash: ' PLAN_HASH
(
set -euo pipefail
[[ "${PLAN_HASH}" =~ ^[0-9a-f]{64}$ ]]
LOG="$(mktemp "${MAIN}/checkpoints/luka-attention-science.XXXXXX.log")"
nohup bash "${PROJECT}/scripts/queue_luka_attention.sh" \
  "${COMMIT}" "${ROOT}" "${PARENT}" "${QUOTA}" science --execute "${PLAN_HASH}" \
  >"${LOG}" 2>&1 </dev/null &
printf 'Helper PID: %s\nLog: %s\n' "$!" "${LOG}"
tail -f "${LOG}"
)
```

No automatic gate-to-science follow-up is armed by these instructions.

## Progress and results

```bash
squeue --me -o '%.18i %.12P %.38j %.2t %.10M %R'
```

Task logs live in ROOT (gate) or ROOT/science (science), named
`preflight-JOB.out`, `train_ATTN_D066-JOB.out`, etc. They show phase heartbeats,
epoch counts, validation AUC, training time and attention KL. Finite poor
metrics are retained and do not stop later registered rungs.

In the same shell (with PROJECT and ROOT set), inspect current completed fits
without new inference:

```bash
source /home/ryreu/miniconda3/etc/profile.d/conda.sh
conda activate /home/ryreu/miniconda3/envs/atlas_kd_sporc
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT}/src"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
python -u -B -s "${PROJECT}/scripts/luka_attention.py" results \
  --spec "${ROOT}/science/campaign_spec.json"
```

Authentication can take several minutes. Primary comparison: ATTN_D000 versus
LOGIT_D000, both single-HLT ParT. Full 74,895 selects checkpoints; original 50k
reports those exact checkpoints. Parent M0HLT/OFFLINE/direct/U100 are labelled
references. Results also retain per-class R50, fit cost, native parameter count
and a small warmed inference timing diagnostic (not a production throughput
benchmark). All comparisons are single-seed development results.
