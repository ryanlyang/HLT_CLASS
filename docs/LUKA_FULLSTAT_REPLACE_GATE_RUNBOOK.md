# FullStat final-rung cold replacement gate — Oscar

[Scientific plan](plans/LUKA_FULLSTAT_REPLACE_GATE_PLAN.md) ·
[contracts](contracts/LUKA_FULLSTAT_REPLACE_GATE.md)

Three NEW 20-epoch, batch128 fits on the existing 3M/500k split:
`COARSE_D033 -> PRIV_D000 -> {REPLACE_LOGIT, REPLACE_GATE}`.
The original D033 is reused, not retrained. PRIV is an offline-containing
oracle; both replacements deploy with HLT alone AND retain their self-gates.
All existing experiments remain untouched. No final-test access.

## 1. Commit and push on Windows

Inspect the staged diff; do not stage unrelated work with `git add .`.
The source guard requires an additive commit descended from donor
`1dff4e5767e66fb5f355cf5d592626ecffcf73ea` with no intervening changes to old
runtime code. Documentation changes are allowed; if other runtime changes
have landed, stop and review compatibility rather than bypassing the guard.

```powershell
git add -- src/hlt_classification/luka_fullstat_replace_gate scripts/luka_fullstat_replace_gate.py scripts/queue_luka_fullstat_replace_gate.sh sbatch/run_luka_fullstat_replace_gate.sh tests/test_luka_fullstat_replace_gate.py docs/plans/LUKA_FULLSTAT_REPLACE_GATE_PLAN.md docs/contracts/LUKA_FULLSTAT_REPLACE_GATE.md docs/LUKA_FULLSTAT_REPLACE_GATE_RUNBOOK.md docs/LUKA_REPLACE_GATE_STRATEGY_HANDOFF.md
git diff --cached --stat
git diff --cached --check
if ($LASTEXITCODE -ne 0) { throw "Inspect staged whitespace errors" }
git commit -m "Add FullStat cold privileged and self-gate final-rung comparison"
if ($LASTEXITCODE -ne 0) { throw "Inspect commit status before continuing" }
git push origin HEAD:main
if ($LASTEXITCODE -ne 0) { throw "Push failed" }
git rev-parse HEAD
```

Stage the new HANDOFF/LEGACY_SOURCE_MAP sections separately with `git add -p`
if their other local edits belong to another task.

## 2. Create a clean pinned worktree and dry-review one preflight

On Oscar, run this block. The dataset/cache/source parent are reused read-only.
The environment must be intact; no repair or environment changes are automatic.
`24` is a declaration of **personally available quota**, not filesystem free
space. Check quota first, and increase the declaration only if actually available.
The preflight additionally checks measured storage and host/GPU memory.

```bash
(
set -euo pipefail
read -r -p 'Paste the pushed 40-character commit: ' COMMIT
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
MAIN=/oscar/home/rlyang/hlt_classification/source/HLT_Classification
PROJECT="/oscar/home/rlyang/hlt_classification/source/HLT_Classification_fullstat_replace_gate_${COMMIT:0:8}"
ROOT="/oscar/home/rlyang/hlt_classification/checkpoints/luka_fullstat_replace_gate_${COMMIT:0:8}_r1"
ORIGINAL=/oscar/home/rlyang/hlt_classification/checkpoints/luka_fullstat_recovery_9bc3cae0_r1/science/campaign_spec.json
git -C "${MAIN}" fetch origin main
git -C "${MAIN}" cat-file -e "${COMMIT}^{commit}"
git -C "${MAIN}" merge-base --is-ancestor "${COMMIT}" origin/main
if [[ -e "${PROJECT}" ]]; then
    test "$(git -C "${PROJECT}" rev-parse HEAD)" = "${COMMIT}"
    test -z "$(git -C "${PROJECT}" status --porcelain --untracked-files=all)"
else
    git -C "${MAIN}" worktree add --detach "${PROJECT}" "${COMMIT}"
fi
checkquota
bash "${PROJECT}/scripts/queue_luka_fullstat_replace_gate.sh" \
    "${COMMIT}" "${ROOT}" "${ORIGINAL}" 24 gate
printf '\nStudy: %s\nProject: %s\n' "${ROOT}" "${PROJECT}"
)
```

Review exactly ONE `lukafsg_preflight`, L40S, 6 CPUs, 180000 MiB, 8h.
This performs three full-population technical epochs, not a full campaign.
No score threshold. A failed or incomplete gate cannot admit science.

## 3. Submit that reviewed preflight

Use the same commit, same root suffix, quota and returned **PLAN** hash.
No heredoc/TTY seeking or nested command substitution is needed.

```bash
(
set -euo pipefail
read -r -p 'Same full commit: ' COMMIT
read -r -p 'Reviewed gate PLAN content_hash: ' HASH
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ && "${HASH}" =~ ^[0-9a-f]{64}$ ]]
PROJECT="/oscar/home/rlyang/hlt_classification/source/HLT_Classification_fullstat_replace_gate_${COMMIT:0:8}"
ROOT="/oscar/home/rlyang/hlt_classification/checkpoints/luka_fullstat_replace_gate_${COMMIT:0:8}_r1"
ORIGINAL=/oscar/home/rlyang/hlt_classification/checkpoints/luka_fullstat_recovery_9bc3cae0_r1/science/campaign_spec.json
read -r -p 'Submit ONE reviewed GPU preflight? Type YES: ' CONFIRM
[[ "${CONFIRM}" == YES ]]
LOG="${ROOT}/gate-submit.log"
test ! -e "${LOG}"
nohup bash "${PROJECT}/scripts/queue_luka_fullstat_replace_gate.sh" \
    "${COMMIT}" "${ROOT}" "${ORIGINAL}" 24 gate --execute "${HASH}" \
    >"${LOG}" 2>&1 </dev/null &
printf 'Helper PID: %s\nLog: %s\n' "$!" "${LOG}"
)
```

Inspect that log for submission success, then `squeue --me`. A helper PID alone
does not prove a Slurm job was accepted. No automatic science follow-up.
Preserve partial outputs or uncertain submissions; do not blindly rerun.

## 4. After preflight COMPLETED/0:0, review and submit five science jobs

The helper authenticates exact preflight job accounting, all evidence and
measured per-task walltimes. Targets from the technical epoch are NOT reused.
Science trains a fresh PRIV teacher, exports its validation-selected weights
on TRAIN, then starts both fresh replacements from the same initialization.

```bash
(
set -euo pipefail
read -r -p 'Same full commit: ' COMMIT
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
PROJECT="/oscar/home/rlyang/hlt_classification/source/HLT_Classification_fullstat_replace_gate_${COMMIT:0:8}"
ROOT="/oscar/home/rlyang/hlt_classification/checkpoints/luka_fullstat_replace_gate_${COMMIT:0:8}_r1"
ORIGINAL=/oscar/home/rlyang/hlt_classification/checkpoints/luka_fullstat_recovery_9bc3cae0_r1/science/campaign_spec.json
bash "${PROJECT}/scripts/queue_luka_fullstat_replace_gate.sh" \
    "${COMMIT}" "${ROOT}" "${ORIGINAL}" 24 science
read -r -p 'Reviewed science PLAN content_hash: ' HASH
[[ "${HASH}" =~ ^[0-9a-f]{64}$ ]]
read -r -p 'Submit these FIVE reviewed science jobs? Type YES: ' CONFIRM
[[ "${CONFIRM}" == YES ]]
LOG="${ROOT}/science-submit.log"
test ! -e "${LOG}"
nohup bash "${PROJECT}/scripts/queue_luka_fullstat_replace_gate.sh" \
    "${COMMIT}" "${ROOT}" "${ORIGINAL}" 24 science --execute "${HASH}" \
    >"${LOG}" 2>&1 </dev/null &
printf 'Helper PID: %s\nLog: %s\n' "$!" "${LOG}"
)
```

No direct inputs from the earlier attention/embedding experiments. Do not
cancel anything by name. The new prefix is `lukafsg_`.

## Results and progress

After setting PROJECT/ROOT to the exact paths printed above:

```bash
export PROJECT_DIR="${PROJECT}"
export JC2_SITE=oscar_l40s
unset LD_PRELOAD
source "${PROJECT_DIR}/sbatch/jetclass2_delphes_common.sh"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -u -B -s "${PROJECT}/scripts/luka_fullstat_replace_gate.py" results \
    --spec "${ROOT}/science/campaign_spec.json"
```

This authenticates saved results, including partially completed campaigns.
Primary contrast is REPLACE_GATE vs REPLACE_LOGIT; PRIV is an oracle.
Each fit logs `FULLSTAT GATE progress` every128 batches and one validation AUC
per full epoch. Use `tail -f` on the exact task log under `science/`.
Progress ETA covers the rest of training in that epoch, not validation or queue
time. All20 epoch optimizer/RNG/full-gated states are retained under each task's
`*_attempt/epochs/`. Selected checkpoints are `science/train_*/selected.pt`.
No automatic checkpoint resumption, target cleanup or final-test evaluation.

Local tests are not evidence that this run fits an Oscar allocation. Only the
successful measured gate authorizes the science plan. Preserve its source,
environment and evidence with the results.
