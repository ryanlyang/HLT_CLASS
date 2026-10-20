# Oscar: final-rung D033 logit KD + fixed OFFLINE embedding prediction

One new HLT-only `EMBED_D000`, from scratch, with the completed D033 teacher.
The auxiliary training head predicts the **pure OFFLINE model's pre-classifier
embedding**, not individual particles or D033 features. It is removed from
the selected deployable checkpoint. No attention loss or fusion.

Unchanged FullStat 3M TRAIN / 500k validation, batch 128, 20 epochs,
warmup/cosine and original D000 seeds. Primary comparison: original
`COARSE_D000`; also report M0HLT, OFFLINE, DIRECT_D000 and COARSE_D033.
References are historical runs, not newly repeated controls. Final test stays
sealed. The trial ALLJETS/SPORC/30-epoch embedding campaign is separate.
Authority: [plan](plans/LUKA_FULLSTAT_LAST_EMBEDDING_PLAN.md) and
[contracts](contracts/LUKA_FULLSTAT_LAST_EMBEDDING.md).

Do not cancel the attention job or remove its files. This experiment does not
need that run. Preserve the original FullStat recovery worktree, completed
science artifacts (including `reduce_COARSE_D033/train`), foundation and caches.
The old Oscar environment previously lost files; verify its imports before
queueing, and do not repair or modify it while jobs are using it.

## 1. Commit and push from Windows

Only the new runtime package/entrypoints may differ from donor
`5ed50964085e9bf364030764bbbc755179c445ae`. Review unrelated staged work first.
Do not use `git add .`. Select only this task's HANDOFF/LEGACY_SOURCE_MAP hunks
with `git add -p` if those files contain other changes.

```powershell
git add src/hlt_classification/luka_fullstat_embedding scripts/luka_fullstat_embedding.py scripts/queue_luka_fullstat_embedding.sh sbatch/run_luka_fullstat_embedding.sh tests/test_luka_fullstat_embedding.py docs/plans/LUKA_FULLSTAT_LAST_EMBEDDING_PLAN.md docs/contracts/LUKA_FULLSTAT_LAST_EMBEDDING.md docs/LUKA_FULLSTAT_LAST_EMBEDDING_RUNBOOK.md
if ($LASTEXITCODE -ne 0) { throw "Stage failed" }
git diff --cached --stat
git commit -m "Add FullStat final-rung offline embedding supervision on Oscar"
if ($LASTEXITCODE -ne 0) { throw "Inspect git status; the changes may already be committed" }
git push origin HEAD:main
if ($LASTEXITCODE -ne 0) { throw "Push failed" }
git rev-parse HEAD
```

## 2. Oscar: pin source and dry-review one preflight

Use an Oscar login shell. Confirm at least **12 GiB of personal free durable
quota**, not just filesystem-wide free space. No dataset is regenerated.
The compact 3M-row embedding values alone occupy about 1.43 GiB; epoch states
and reports also need room. The gate checks physical space too.

```bash
read -r -p 'Pushed 40-character embedding commit: ' COMMIT
MAIN=/oscar/home/rlyang/hlt_classification/source/HLT_Classification
PROJECT="/oscar/home/rlyang/hlt_classification/source/HLT_Classification_fullstat_embedding_${COMMIT:0:8}"
ROOT="/oscar/home/rlyang/hlt_classification/checkpoints/luka_fullstat_last_embedding_${COMMIT:0:8}_r1"
ORIGINAL=/oscar/home/rlyang/hlt_classification/checkpoints/luka_fullstat_recovery_9bc3cae0_r1/science/campaign_spec.json
QUOTA=12

(
set -euo pipefail
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
test -f "${ORIGINAL}"
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
export PROJECT_DIR="${PROJECT}"
export JC2_SITE=oscar_l40s
unset LD_PRELOAD
source "${PROJECT_DIR}/sbatch/jetclass2_delphes_common.sh"
python -I -B -c \
    'import encodings, numpy, torch, awkward, uproot, sklearn, weaver.nn.model.ParticleTransformer; print("Imports OK")'
bash "${PROJECT}/scripts/queue_luka_fullstat_embedding.sh" \
    "${COMMIT}" "${ROOT}" "${ORIGINAL}" "${QUOTA}" gate
)
```

Review: **one** `lukafse_preflight`, Oscar `gpu`, L40S, 6 CPUs, 180000 MiB,
8-hour ceiling. No science jobs are submitted automatically.
It exports TRAIN-only OFFLINE targets, checks native/head/gradient parity,
tests maximum support, runs one full 3M/500k technical epoch, and verifies
head-free checkpoint inference. The probe is not a scientific result.

## 3. Submit that exact reviewed preflight

Keep the variables from step 2. Copy the printed **PLAN** `content_hash`,
not the gate hash.

```bash
read -r -p 'Reviewed gate PLAN content_hash: ' PLAN
read -r -p 'Submit ONE embedding preflight? Type YES: ' CONFIRM
if [[ "${CONFIRM}" == YES && "${PLAN}" =~ ^[0-9a-f]{64}$ ]]; then
    bash "${PROJECT}/scripts/queue_luka_fullstat_embedding.sh" \
        "${COMMIT}" "${ROOT}" "${ORIGINAL}" "${QUOTA}" gate --execute "${PLAN}"
fi
squeue --me -o "%.18i %.12P %.36j %.2t %.10M %R"
```

Wait for the helper to finish and print its submission ledger before
disconnecting. This avoids an ambiguous interrupted submission. An incomplete
live claim fails closed; do not delete claims or rerun with a new root to bypass
an unknown submission. Inspect the exact journal/job first.

## 4. After successful preflight, review and submit science

```bash
bash "${PROJECT}/scripts/queue_luka_fullstat_embedding.sh" \
    "${COMMIT}" "${ROOT}" "${ORIGINAL}" "${QUOTA}" science
```

This fails closed unless the accepted receipt, all bound evidence and exact
gate journal validate and Slurm reports the gate `COMPLETED` with `0:0`.
Failed/over-envelope preflight means **no science submission**.
Inspect the measured request and two jobs: `train_EMBED_D000` then dependent
CPU `aggregate`. The fit requests a conservative measured walltime, at least
12 hours and at most 48 hours.

```bash
read -r -p 'Reviewed science PLAN content_hash: ' PLAN
read -r -p 'Submit embedding fit and comparison? Type YES: ' CONFIRM
if [[ "${CONFIRM}" == YES && "${PLAN}" =~ ^[0-9a-f]{64}$ ]]; then
    bash "${PROJECT}/scripts/queue_luka_fullstat_embedding.sh" \
        "${COMMIT}" "${ROOT}" "${ORIGINAL}" "${QUOTA}" science --execute "${PLAN}"
fi
```

Scientific quality does not gate success. Every completed epoch has immutable
weights, optimizer/RNG and history under `science/training_attempt/checkpoints`.
There is no automatic resume/deletion. Preserve interrupted attempts for an
explicit reviewed recovery; do not assume rerunning `run` resumes them.

## 5. Progress and comparison

```bash
grep -hE '^FULLSTAT EMBED (progress|node=)' "${ROOT}"/science/train_EMBED_D000-*.out | tail -n 25
```

After completion, run the authenticated comparison in a sourced shell:

```bash
(
set -euo pipefail
export PROJECT_DIR="${PROJECT}"
export JC2_SITE=oscar_l40s
unset LD_PRELOAD
source "${PROJECT_DIR}/sbatch/jetclass2_delphes_common.sh"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -u -B -s "${PROJECT_DIR}/scripts/luka_fullstat_embedding.py" results \
    --spec "${ROOT}/science/campaign_spec.json"
)
```

This compares the same registered full validation population and reports AUC,
accuracy and recovery. The selected checkpoint is AUC-first across all 20
epochs. Do not use validation gains to claim test performance or multi-seed
significance. No test command exists in this namespace.
