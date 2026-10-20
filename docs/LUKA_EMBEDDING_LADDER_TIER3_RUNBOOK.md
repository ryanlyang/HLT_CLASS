# Queue the accepted embedding ladder on tier3

Authority: [transfer plan](plans/LUKA_EMBEDDING_LADDER_TIER3_PLAN.md),
[transfer contracts](contracts/LUKA_EMBEDDING_LADDER_TIER3.md).

Reuse completed original debug preflight **21929903**, after the code verifies
its full evidence and exact accounting. Do not rerun the preflight, edit the
old gate, or change a submitted job's partition. Same five-rung fixed30/batch128
science, same measured resource request; actual tier3 GPU/environment must match.
This is the **0.3 embedding + logit-KD ladder**, not the 0.75/1.5/10 no-KD sweep.

Local validation: original ladder baseline **15 passed**; final transfer suite
**16 passed in 82.24s**, including actual local CUDA/Weaver miniature execution
of both complete five-rung arms. Shell blocks, CLI and documentation links
passed checks. These fixtures do not independently verify remote 21929903;
the queue helper performs that original-source evidence replay before submission.
No remote jobs were submitted by this implementation.

## 1. Commit/push locally

Only new transfer files are required. Keep unrelated edits unstaged.

```powershell
& {
    git diff --cached --quiet
    if ($LASTEXITCODE -ne 0) { throw "Inspect existing staged changes first." }
    git add -- src/hlt_classification/luka_embedding_ladder/tier3.py `
        scripts/luka_embedding_ladder_tier3.py scripts/queue_luka_embedding_ladder_tier3.sh `
        sbatch/run_luka_embedding_ladder_tier3.sh tests/test_luka_embedding_ladder_tier3.py `
        docs/plans/LUKA_EMBEDDING_LADDER_TIER3_PLAN.md `
        docs/contracts/LUKA_EMBEDDING_LADDER_TIER3.md docs/LUKA_EMBEDDING_LADDER_TIER3_RUNBOOK.md
    if ($LASTEXITCODE -ne 0) { throw "Stage failed" }
    git diff --cached --stat
    git commit -m "Reuse accepted embedding ladder preflight for tier3 science"
    if ($LASTEXITCODE -ne 0) { throw "Commit failed; stop" }
    git push origin HEAD:main
    if ($LASTEXITCODE -ne 0) { throw "Push failed; paste error, do not force-push" }
    git rev-parse HEAD
}
```

Optionally stage only this step's new HANDOFF/LEGACY_SOURCE_MAP sections with
`git add -p`. Original scientific modules are unchanged. Keep the no-KD sweep
on its own already-pushed `0342951d` worktree; do not recreate it with this source.

## 2. On sporcsubmit: locate the accepted gate and submit one ladder

Select `EMBED` for the requested ladder. `LOGIT` is a separately authorized
matched five-rung comparison and can use the same accepted preflight.

```bash
bash <<'BASH'
set -euo pipefail
trap 'echo "Stopped; preserve everything and paste the error." >&2' ERR
read -r -p 'Paste the NEW pushed 40-character transfer commit: ' COMMIT </dev/tty
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
read -r -p 'Arm (EMBED for embedding+KD; LOGIT for matched KD-only): ' ARM </dev/tty
[[ "${ARM}" == EMBED || "${ARM}" == LOGIT ]]
MAIN=/home/ryreu/atlas/HLT_Classification
PROJECT="${MAIN}_embedladder_tier3_${COMMIT:0:8}"
PYTHON=/home/ryreu/miniconda3/envs/atlas_kd_sporc/bin/python
test -x "${PYTHON}"
ROOT="$("${PYTHON}" -B -s - "${MAIN}/checkpoints" <<'PY'
import json, sys
from pathlib import Path
matches = []
for p in Path(sys.argv[1]).glob("luka_embedding_ladder_*/preflight/receipt.json"):
    report = json.loads(p.read_text())
    if str(report.get("job_id")) == "21929903":
        matches.append(p.parent.parent)
if len(matches) != 1:
    raise SystemExit(f"STOP: expected one gate for 21929903; found {len(matches)}")
print(matches[0])
PY
)"
test -f "${ROOT}/gate_spec.json"
git -C "${MAIN}" fetch origin main
git -C "${MAIN}" merge-base --is-ancestor "${COMMIT}" origin/main
if [[ -e "${PROJECT}" ]]; then
    test "$(git -C "${PROJECT}" rev-parse HEAD)" = "${COMMIT}"
    test -z "$(git -C "${PROJECT}" status --porcelain --untracked-files=all)"
else
    git -C "${MAIN}" worktree add --detach "${PROJECT}" "${COMMIT}"
fi
printf '\nPROJECT=%s\nGATE_ROOT=%s\nARM=%s\n' "${PROJECT}" "${ROOT}" "${ARM}"
bash "${PROJECT}/scripts/queue_luka_embedding_ladder_tier3.sh" \
    "${PROJECT}" "${ROOT}" "${ARM}"
BASH
```

Creation/review authenticates the old preflight and can take several minutes.
Inspect the printed command: partition tier3, one A100, 6 CPUs, 90000 MiB and
the preflight's unchanged measured arm walltime. Paste the **value** of PLAN_HASH
(64 hexadecimal characters), then YES. The helper survives disconnection.
Read its printed log until there is an actual Slurm job ID. A PID alone is not
proof of submission. If it errors, preserve the output; do not create another
root or bypass a claim.

Names: `lukaelt_embed_ladder` / `lukaelt_logit_ladder`. Each is **one job running
all five rungs**, not five separately queued jobs. No automatic submission of
the other arm. Existing debug science for the same arm causes a stop for review.

## 3. Progress and results

Keep the exact PROJECT and GATE_ROOT printed above; their shell variables do
not survive the Bash block. Set ROOT to GATE_ROOT and ARM to the queued arm.

```bash
squeue --me -o "%.18i %.12P %.35j %.2t %.10M %R"
```

Live logs: `GATE_ROOT/science_tier3/ARM/ladder-JOBID.out`. Per-rung histories:
`science_tier3/ARM/ladder/ARM_COORDINATE/training_report.json`.
The result command expects the entire arm to be complete:

```bash
source /home/ryreu/miniconda3/etc/profile.d/conda.sh
conda activate /home/ryreu/miniconda3/envs/atlas_kd_sporc
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PROJECT}/src"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -u -B -s "${PROJECT}/scripts/luka_embedding_ladder_tier3.py" results \
    --spec "${ROOT}/science_tier3/${ARM}/campaign_spec.json"
```

No independent test; full validation selects checkpoints and original50k is
report-only. Intermediate model retirement is unchanged; final native D000
weights, histories, predictions and receipts remain. Shared targets, original
preflight and worktrees must be preserved. No new particle/attention caches.
