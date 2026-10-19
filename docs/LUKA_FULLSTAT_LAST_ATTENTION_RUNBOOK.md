# FullStat last-rung attention: commit, review and queue on Oscar

Authority: [scientific plan](plans/LUKA_FULLSTAT_LAST_ATTENTION_PLAN.md) and
[v1 contracts](contracts/LUKA_FULLSTAT_LAST_ATTENTION.md).

This is **two fresh D000 fits**, not another complete ladder. Both use the
already-selected **COARSE_D033 epoch17** teacher. `LOGIT_D000` has the unchanged
logit-KD objective; `ATTN_D000` adds constant0.1 aligned-attention KL. Both keep
3M TRAIN/500k validation, batch128,20 epochs and identical initialization/sampler
seeds. Validation chooses each checkpoint. The1.5M final test and105325 unused
rows remain sealed. Event-level independence and overlap with the old trial
remain unverified. Nothing here predicts a performance improvement.

One new L40S preflight must pass before science is eligible. Science is four
jobs: one shared TRAIN-target export, two fits, then CPU aggregation. No
automatic submission, retry, cleanup or test evaluation. Selected checkpoints
and validation predictions are retained; teacher targets remain in the declared
scratch directory. Do not delete any parent source, evidence or prepared cache.

## 1. Commit and push the new code on Windows

From the repository PowerShell terminal, review and stage **only** these paths:

```powershell
git add -- src/hlt_classification/luka_fullstat_attention `
  scripts/luka_fullstat_attention.py scripts/queue_luka_fullstat_attention.sh `
  sbatch/run_luka_fullstat_attention.sh tests/test_luka_fullstat_attention.py `
  docs/plans/LUKA_FULLSTAT_LAST_ATTENTION_PLAN.md `
  docs/contracts/LUKA_FULLSTAT_LAST_ATTENTION.md `
  docs/LUKA_FULLSTAT_LAST_ATTENTION_RUNBOOK.md
if ($LASTEXITCODE -ne 0) { throw "Staging failed" }
git diff --cached --stat
git diff --cached --check
if ($LASTEXITCODE -ne 0) { throw "Review the staged whitespace errors" }
git commit -m "Add matched FullStat D033-to-D000 attention comparison"
if ($LASTEXITCODE -ne 0) { throw "Check whether these changes are already committed before continuing" }
git push origin HEAD:main
if ($LASTEXITCODE -ne 0) { throw "Push failed" }
git rev-parse HEAD
```

Review any previously staged changes before committing. `docs/HANDOFF.md` and
`docs/LEGACY_SOURCE_MAP.md` also have new continuation notes, but contained
unrelated edits before this work: review/stage their intended hunks separately.
Do not use `git add .`. The queue helper requires all runtime/new plan files
committed and a clean detached worktree; the main working copy may retain
unrelated edits. The existing runtime is frozen against donora1d3a2c6; changing
older runtime code in the same execution commit is deliberately rejected.

## 2. Create a detached worktree and dry-review the gate on Oscar

Run in the **Oscar login Bash terminal**. Paste the full40-character commit
printed above. The original completed campaign is authenticated in its original
worktree before it can supervise this comparison; this may take several minutes.

```bash
read -r -p 'Pushed 40-character attention commit: ' COMMIT
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]] || { echo 'Invalid commit'; false; }
MAIN=/oscar/home/rlyang/hlt_classification/source/HLT_Classification
SHORT="${COMMIT:0:8}"
PROJECT="/oscar/home/rlyang/hlt_classification/source/HLT_Classification_fullstat_attention_${SHORT}"
ROOT="/oscar/home/rlyang/hlt_classification/checkpoints/luka_fullstat_last_attention_${SHORT}_r1"
PARENT=/oscar/home/rlyang/hlt_classification/checkpoints/luka_fullstat_recovery_9bc3cae0_r1/science/campaign_spec.json
BANK_ROOT="/oscar/scratch/rlyang/hlt_classification/attention_banks/luka_fullstat_last_attention_${SHORT}_r1"

(
set -euo pipefail
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
git -C "${MAIN}" fetch origin main
git -C "${MAIN}" cat-file -e "${COMMIT}^{commit}"
git -C "${MAIN}" merge-base --is-ancestor "${COMMIT}" origin/main
if [[ -e "${PROJECT}" ]]; then
    test "$(git -C "${PROJECT}" rev-parse HEAD)" = "${COMMIT}"
    test -z "$(git -C "${PROJECT}" status --porcelain --untracked-files=all)"
else
    git -C "${MAIN}" worktree add --detach "${PROJECT}" "${COMMIT}"
fi
test -r "${PARENT}"
)
```

**Stop if that block reports an error.** Now check your own quota, not filesystem
capacity. Declare genuinely available headroom after allowing for other jobs:

```bash
checkquota
read -r -p 'Available scratch GiB reserved for the shared attention bank: ' BANK_GIB
read -r -p 'Available home GiB reserved for this new result directory (at least 8): ' RESULT_GIB
bash "${PROJECT}/scripts/queue_luka_fullstat_attention.sh" \
  "${COMMIT}" "${ROOT}" "${PARENT}" "${BANK_ROOT}" "${BANK_GIB}" "${RESULT_GIB}" gate
```

This creates immutable registration/dry-review records, not jobs. It does not
rebuild ROOT preprocessing. The real preflight measures the bank's support:
FP32 attention alone is `4 * sum(n_particles**2)` bytes over3M TRAIN jets;
the admission bound adds population overhead and reserve. No fabricated bank
size is assumed. Bank output is shared, not duplicated per student.

Existing roots are preserved. If the source, environment, parent completion,
teacher epoch, cache identity or personal quota cannot authenticate, stop and
inspect that error. Do not modify the old campaign to make the check pass.

## 3. Submit the one reviewed preflight

After reviewing the printed gate plan, paste **its**64-character `content_hash`:

```bash
read -r -p 'Reviewed gate PLAN content_hash: ' PLAN_HASH
bash "${PROJECT}/scripts/queue_luka_fullstat_attention.sh" \
  "${COMMIT}" "${ROOT}" "${PARENT}" "${BANK_ROOT}" "${BANK_GIB}" "${RESULT_GIB}" \
  gate --execute "${PLAN_HASH}"
squeue --me -o '%.18i %.12P %.40j %.2t %.10M %R'
```

This asks for one Oscar `gpu:l40s:1`,6CPU,180000MiB,8h technical job using
`default`/`gpu`/`norm-gpu`. Keep the connection until submission returns the
job ID/journal; after Slurm accepts it, the job does not require the SSH session.
If submission is interrupted/ambiguous, inspect the saved journal before any
retry; do not delete its `live_claim` guard.

Preflight logs: `${ROOT}/preflight-<jobid>.out`.
Successful evidence: `${ROOT}/preflight/receipt.json`.
Use `sacct -j <jobid> -X -P -o JobID,State,ExitCode,Elapsed` to confirm
`COMPLETED|0:0`. A file's mere existence does not admit science: the code verifies
the receipt, all bound payloads, exact gate journal and successful accounting.

Preflight includes separate maximum-length TRAIN/validation support, actual
selected-teacher target export, installed native FP32/BF16 parity, lambda-zero
training equality, both one-pass production fits and checkpoint readbacks.
These are technical miniature fits, **not** scientific checkpoints. Full-cache
support is audited, but20-epoch runtimes remain conservative projections.

## 4. After successful preflight, review and submit science

Use the same variables (or reconstruct their exact values after reconnecting):

```bash
bash "${PROJECT}/scripts/queue_luka_fullstat_attention.sh" \
  "${COMMIT}" "${ROOT}" "${PARENT}" "${BANK_ROOT}" "${BANK_GIB}" "${RESULT_GIB}" science
```

Review all **four** jobs and measured walltimes; the CPU aggregate has no GPU
request. Both fits depend on successful shared export. The primary comparison
is `ATTN_D000 - LOGIT_D000`, regardless of whether attention helps or hurts.

```bash
read -r -p 'Reviewed science PLAN content_hash: ' PLAN_HASH
bash "${PROJECT}/scripts/queue_luka_fullstat_attention.sh" \
  "${COMMIT}" "${ROOT}" "${PARENT}" "${BANK_ROOT}" "${BANK_GIB}" "${RESULT_GIB}" \
  science --execute "${PLAN_HASH}"
squeue --me -o '%.18i %.12P %.40j %.2t %.10M %R'
```

No parent training is resubmitted. No models are initialized from D033 weights:
D033 is an eval/frozen teacher only. Both students are ordinary single-HLT ParTs
at inference; attention capture is a training-only readout, not a new encoder.

## 5. Results and progress

```bash
(
set -euo pipefail
export PROJECT_DIR="${PROJECT}" JC2_SITE=oscar_l40s
unset LD_PRELOAD
source "${PROJECT_DIR}/sbatch/jetclass2_delphes_common.sh"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -u -B -s "${PROJECT_DIR}/scripts/luka_fullstat_attention.py" results \
  --spec "${ROOT}/science/campaign_spec.json"
)
```

This authenticates saved results and prints both new rows, recovery against
the same original M0HLT/OFFLINE controls, and QCD rejection@50%. Incomplete fits
are `PENDING`; they are not evaluated from partial checkpoints. Original
DIRECT_D000 and COARSE_D000 are historical references, not matched new repeats;
OFFLINE and D033 are privileged-input controls, not deployable HLT endpoints.

Each fit's Slurm log prints every epoch's AUC, train time and attention loss.
Selected states, histories, losses/entropy/gradient norms and predictions live
under `${ROOT}/science/train_LOGIT_D000` and `train_ATTN_D000`; combined results
are `${ROOT}/science/aggregate/results.json`. The attention bank is outside
those directories at `${BANK_ROOT}/D033_train`. No command in this runbook
evaluates final test or removes files.
