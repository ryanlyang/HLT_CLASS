# Fixed-OFFLINE embedding auxiliary + logit ladder on SPORC debug

Authority: [plan](plans/LUKA_EMBEDDING_LADDER_PLAN.md),
[v1 contracts](contracts/LUKA_EMBEDDING_LADDER.md).

This is the genuine FullSim **trial ALLJETS** split: 299,234 train / 74,895
validation, not Oscar FullStat. There is no independent test. The original
50k subset is report-only; all checkpoints use the full validation population.

Starting from the existing selected U000 teacher, train five fresh models:
U050 -> U100 -> D066 -> D033 -> D000. Each receives logit KD from the preceding
selected model. The EMBED arm additionally predicts the same frozen pure
OFFLINE embedding at every rung through the existing training-only head.
Its loss is 0.25 CE + 0.75 T^2 KL, T=2, plus 0.3 SmoothL1 embedding prediction.
The coefficient stays 0.3, but CE is now 0.25 rather than the single-stage
experiment's 1.0. This combination is a registered hypothesis, not an established
gain. No attention loss, warm start, fusion, or inference-time auxiliary head.

One preflight tests all five coordinates for both objectives. Science consists
of separately authorized EMBED and optional matched LOGIT-only ladder jobs,
each sequentially running five fixed30/batch128 fits. The older three-rung
LOGIT result is not an exact control for the five-rung experiment. Queue the
LOGIT arm as well for the matched comparison. No science is auto-submitted.

Local validation: 37 combined ladder/single-stage tests passed, including real
installed CUDA/ParT kernels with miniature populations, native KD parity,
both full fixed30 ladder fixtures and exact ROOT-view reconstruction. Shell
syntax and CLI checks passed. Remote source/data/resource acceptance is still
required; no new real SPORC campaign has been submitted by the implementation.

## 1. Commit and push on Windows

Review only the new files. Preserve unrelated work and do not use `git add .`.

```powershell
git add -- src/hlt_classification/luka_embedding_ladder `
  scripts/luka_embedding_ladder.py scripts/queue_luka_embedding_ladder.sh `
  sbatch/run_luka_embedding_ladder.sh tests/test_luka_embedding_ladder.py `
  docs/plans/LUKA_EMBEDDING_LADDER_PLAN.md `
  docs/contracts/LUKA_EMBEDDING_LADDER.md docs/LUKA_EMBEDDING_LADDER_RUNBOOK.md
if ($LASTEXITCODE -ne 0) { throw "Stage failed" }
git diff --cached --stat
# docs/HANDOFF.md and docs/LEGACY_SOURCE_MAP.md may contain unrelated changes.
# Stage only this experiment's sections with git add -p if desired.
git commit -m "Add fixed-offline embedding prediction to matched logit ladders"
if ($LASTEXITCODE -ne 0) { throw "Inspect status; do not assume a new commit exists" }
git push origin HEAD:main
if ($LASTEXITCODE -ne 0) { throw "Stop; do not force-push" }
git rev-parse HEAD
```

The runtime guard allows only this additive namespace relative to donor
`991c9d44161552c3ff8ed65eafd59933217f08aa`. Do not bypass a source mismatch.

## 2. Create a fresh campaign and review/submit ONE preflight

Run on `sporcsubmit`. Keep the completed single-stage embedding targets and its
original pinned worktree: this run authenticates and reuses them read-only.
Reserve at least **2 GiB of personally available persistent storage** for the
whole new experiment. This excludes already-existing shared inputs/targets.

```bash
bash <<'BASH'
set -euo pipefail
trap 'echo "Stopped; preserve artifacts and paste the error." >&2' ERR
read -r -p 'Paste the NEW pushed 40-character commit: ' COMMIT </dev/tty
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
MAIN=/home/ryreu/atlas/HLT_Classification
PROJECT="/home/ryreu/atlas/HLT_Classification_embedladder_${COMMIT:0:8}"
ROOT="${MAIN}/checkpoints/luka_embedding_ladder_${COMMIT:0:8}_debug_r1"
PARENT="${MAIN}/checkpoints/luka_fixed30_74aa58c5_r1/science/campaign_spec.json"
EMBED="${MAIN}/checkpoints/luka_embedding_cfb3a158_debug_r1/science/campaign_spec.json"
test -f "${PARENT}"
test -f "${EMBED}"
test ! -e "${ROOT}"
df -h /home/ryreu
read -r -p 'Personally available GiB allocated to this experiment (at least 2): ' QUOTA </dev/tty
[[ "${QUOTA}" =~ ^[1-9][0-9]*$ ]]
git -C "${MAIN}" fetch origin main
git -C "${MAIN}" merge-base --is-ancestor "${COMMIT}" origin/main
if [[ -e "${PROJECT}" ]]; then
    test "$(git -C "${PROJECT}" rev-parse HEAD)" = "${COMMIT}"
    test -z "$(git -C "${PROJECT}" status --porcelain --untracked-files=all)"
else
    git -C "${MAIN}" worktree add --detach "${PROJECT}" "${COMMIT}"
fi
source /home/ryreu/miniconda3/etc/profile.d/conda.sh
conda activate /home/ryreu/miniconda3/envs/atlas_kd_sporc
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT}/src"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -u -B -s "${PROJECT}/scripts/luka_embedding_ladder.py" create \
    --root "${ROOT}" --parent "${PARENT}" --embedding "${EMBED}" \
    --expected-commit "${COMMIT}" --quota-gib "${QUOTA}"
printf '\nKeep these paths:\nPROJECT=%s\nROOT=%s\n' "${PROJECT}" "${ROOT}"
bash "${PROJECT}/scripts/queue_luka_embedding_ladder.sh" "${PROJECT}" "${ROOT}" gate
BASH
```

The helper prints the dry plan, asks for its **64-character hash value** and
`YES`, then starts a durable submission helper. A PID is not proof of submission:
inspect its log for the exact job ID and confirm `lukael_preflight` in `squeue`.
If interrupted after creation, do not rerun `create` over the root; invoke the
queue helper on the saved paths. If submission is uncertain, inspect the journal;
do not clear claims or submit a duplicate.

The preflight uses one A100, 6 CPUs, 90,000 MiB, and an 8-hour maximum. It measures
ten one-epoch probes, both full populations, full support, native KD and auxiliary
parity, selected checkpoint/export roundtrips, resource headroom, and artifact
retirement. Science is admitted only with `COMPLETED/0:0`, authenticated outputs,
and measured whole-ladder time <=24h per arm. Poor AUC never blocks admission.

## 3. After successful preflight, queue the requested ladder

Set `PROJECT` and `ROOT` to the exact paths printed above (not the old embedding
campaign), then:

```bash
bash "${PROJECT}/scripts/queue_luka_embedding_ladder.sh" "${PROJECT}" "${ROOT}" EMBED
```

For the matched logit-only comparison, separately review and authorize:

```bash
bash "${PROJECT}/scripts/queue_luka_embedding_ladder.sh" "${PROJECT}" "${ROOT}" LOGIT
```

There is **one science job per arm**, not one job per rung. Names are
`lukael_embed_ladder` and `lukael_logit_ladder`. The selected immediate teacher's
temperature-2 targets remain in RAM; no temporary target bank is written to the
shared filesystem. Neither arm depends on the other's weights or results.
Do not change partition/resources of a queued job behind its reviewed plan.

## 4. Progress and results

```bash
squeue --me -o "%.18i %.12P %.32j %.2t %.10M %R"
```

Each arm's log is `ROOT/science/ARM/ladder-JOBID.out`. Completed epoch lines start
`EMBED_LADDER node=... epoch=... auc=...` and include CE, KD and auxiliary losses.
Each finished rung also has `ladder/ARM_COORD/training_report.json` containing
every validation epoch and its selected pass, plus `validation_predictions.npz`.
The full results command requires the arm to finish; no partial receipt is
misrepresented as a completed ladder.

In the activated SPORC environment with the same library-path setup:

```bash
python -u -B -s "${PROJECT}/scripts/luka_embedding_ladder.py" results \
    --spec "${ROOT}/science/EMBED/campaign_spec.json"
python -u -B -s "${PROJECT}/scripts/luka_embedding_ladder.py" results \
    --spec "${ROOT}/science/LOGIT/campaign_spec.json"
```

Compare `EMBED_D000` against `LOGIT_D000` for the incremental ladder auxiliary
effect. M0HLT, OFFLINE and SINGLE_EMBED are authenticated supplementary rows.
Intermediate U/D views contain offline information and are oracles, not HLT-only
deployable scores. Compare full-validation rows with full-validation rows.
No seed uncertainty or independent test claim is made.

## Storage and recoverability

Only completely missing U000/U050 input caches may be reconstructed in RAM;
existing assignments and prepared fingerprints must match exactly. No new
durable particle cache and no attention bank is created. All six registered
views are held in bounded RAM per role, then released when the job exits.

After each intermediate selected checkpoint is verified and its TRAIN logits
exported, that newly owned file is individually deleted with a retirement
receipt. This is permanent, with no trash backup. It cannot subsequently be
used for arbitrary inference or resumed training. Final D000 native checkpoints,
all histories, selected validation predictions, identities, labels, subset
indices, comparisons and provenance are retained. The auxiliary head is never
saved in deployable checkpoints. Shared original teacher files and the compact
OFFLINE target bank are not deleted. Fresh_check is never touched.

Interruption loses in-memory targets; there is deliberately no automatic resume.
Preserve partial evidence and register a new attempt if recovery is needed.
Do not delete gate/reference artifacts while either arm needs them.
