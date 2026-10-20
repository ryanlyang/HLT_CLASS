# Queue the alternating-gate experiment on SPORC debug

Authority: [plan](plans/LUKA_ALTERNATING_GATE_PLAN.md) and
[v1 contract](contracts/LUKA_ALTERNATING_GATE.md).

Three independent six-fit chains: CONTROL, LOGIT and GATE. Each descends
U100 -> TEACH_D066 -> SELF_D066 -> TEACH_D033 -> SELF_D033 -> TEACH_D000 ->
SELF_D000. LOGIT/GATE TEACH may use the previous richer view only through the
extra gate; SELF uses its own view only. GATE adds 0.1 centered gate loss at
each SELF replacement. CONTROL has no privileged gate context and the same
six-fit budget. Cold initialization, batch 128, fixed 30, original seeds.

Same trial ALLJETS 299234/74895 split, original50k report-only. No independent
test. Only D000 self-contained endpoints are deployment comparisons.

No old job is cancelled, no dataset is rebuilt, and Fresh_check is untouched.
The required parent is the selected fixed30 COARSE_U100 in the 74aa58c5 campaign;
its pinned source, metadata, checkpoints and U100/D input caches must remain.

## 1. Review, commit and push locally

Do not use `git add .`. Review only these new files. HANDOFF and LEGACY_SOURCE_MAP
contain other work: stage just this campaign's entries with `git add -p` if desired.

```powershell
git add -- src/hlt_classification/luka_alternating_gate scripts/luka_alternating_gate.py scripts/queue_luka_alternating_gate.sh sbatch/run_luka_alternating_gate.sh tests/test_luka_alternating_gate.py docs/plans/LUKA_ALTERNATING_GATE_PLAN.md docs/contracts/LUKA_ALTERNATING_GATE.md docs/LUKA_ALTERNATING_GATE_RUNBOOK.md
if ($LASTEXITCODE -ne 0) { throw 'Staging failed' }
git diff --cached --stat
git diff --cached --check
if ($LASTEXITCODE -ne 0) { throw 'Review staged changes' }
git commit -m "Add stagewise privileged-to-self gate ladders on debug"
if ($LASTEXITCODE -ne 0) { throw 'Commit failed; inspect before continuing' }
git push origin HEAD:main
if ($LASTEXITCODE -ne 0) { throw 'Push failed; do not force-push' }
git rev-parse HEAD
```

Source compatibility is pinned to e95c0d83a60b75cfc9acca4aaf377740fde8930e.
Only this package's new src files may differ. If other scientific source changes
are merged concurrently, explicitly reconcile and test before changing the pin.

## 2. Queue only the new debug preflight

Paste on **sporcsubmit**, not Tigris. This creates a fresh immutable root and
pinned worktree. The gate reserves 8h, 6 CPUs, 128000 MiB, one A100. It measures all 18
technical fits and their memory/time before any science is admitted.

Declare at least 2 GiB personally available persistent space, not filesystem-wide
free space. Targets use verified node-local `/tmp`; if the site provides a
canonical `/local/scratch`, set `LUKA_ALTERNATING_SCRATCH_BASE=/local/scratch`
before BOTH helper calls. No home fallback. Owned scratch targets are deleted
after verified consumption; selected checkpoints/results stay for reproducibility.

```bash
bash <<'BASH'
set -euo pipefail
trap 'echo "Stopped; preserve everything and paste the error." >&2' ERR
read -r -p 'NEW pushed 40-character commit: ' COMMIT </dev/tty
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
MAIN=/home/ryreu/atlas/HLT_Classification
PROJECT="/home/ryreu/atlas/HLT_Classification_lukaalternating_${COMMIT:0:8}"
ROOT="${MAIN}/checkpoints/luka_alternating_${COMMIT:0:8}_r1"
PARENT="${MAIN}/checkpoints/luka_fixed30_74aa58c5_r1/science/campaign_spec.json"
test -f "${PARENT}"
test ! -e "${ROOT}"
git -C "${MAIN}" fetch origin main
git -C "${MAIN}" merge-base --is-ancestor "${COMMIT}" origin/main
if [[ -e "${PROJECT}" ]]; then
    test "$(git -C "${PROJECT}" rev-parse HEAD)" = "${COMMIT}"
    test -z "$(git -C "${PROJECT}" status --porcelain --untracked-files=all)"
else
    git -C "${MAIN}" worktree add --detach "${PROJECT}" "${COMMIT}"
fi
df -h /home/ryreu
read -r -p 'Personally available persistent GiB for this campaign (minimum 2): ' QUOTA </dev/tty
[[ "${QUOTA}" =~ ^[1-9][0-9]*$ && "${QUOTA}" -ge 2 ]]
HELPER="${PROJECT}/scripts/queue_luka_alternating_gate.sh"
bash "${HELPER}" "${COMMIT}" "${ROOT}" "${PARENT}" "${QUOTA}" gate
read -r -p 'Paste reviewed PLAN content_hash VALUE (64 hex): ' PLAN_HASH </dev/tty
[[ "${PLAN_HASH}" =~ ^[0-9a-f]{64}$ ]]
read -r -p 'Submit ONE new debug preflight? Type YES: ' CONFIRM </dev/tty
if [[ "${CONFIRM}" != YES ]]; then echo 'Stopped; no jobs submitted.'; exit 0; fi
LOG="$(mktemp "${ROOT}/gate-submit.XXXXXX.log")"
nohup bash "${HELPER}" "${COMMIT}" "${ROOT}" "${PARENT}" "${QUOTA}" gate \
    --execute "${PLAN_HASH}" >"${LOG}" 2>&1 </dev/null &
printf 'Helper PID: %s\nStudy: %s\nLog: %s\nWatch with:\ntail -f "%s"\n' "$!" "${ROOT}" "${LOG}" "${LOG}"
BASH
```

Wait for a submitted job ID and `QUEUE_HELPER_EXIT=0`; helper PID alone is not
submission proof. Ctrl-C stops tail only. Preserve failures; do not resubmit
ambiguous live claims. This helper does NOT automatically submit science.

## 3. After successful preflight, review and submit science

The exact gate journal, COMPLETED/0:0 accounting and authenticated resource
report must all pass. This submits 18 fits plus aggregate, all debug, with
afterok dependencies inside each arm and independently measured <=24h fit limits.
No result-quality threshold; no truncation, shortened schedule or retry on failure.

```bash
bash <<'BASH'
set -euo pipefail
read -r -p 'Paste exact completed alternating study root: ' ROOT </dev/tty
PYTHON=/home/ryreu/miniconda3/envs/atlas_kd_sporc/bin/python
test -f "${ROOT}/gate_spec.json"
mapfile -t VALUES < <("${PYTHON}" -B -s -c 'import json,sys; g=json.load(open(sys.argv[1])); print(g["source_commit"]); print(g["project_dir"]); print(g["anchor"]["spec"]["path"]); print(g["available_quota_gib"]); print(g["scratch_base"])' "${ROOT}/gate_spec.json")
[[ "${#VALUES[@]}" -eq 5 ]]
COMMIT="${VALUES[0]}"
PROJECT="${VALUES[1]}"
PARENT="${VALUES[2]}"
QUOTA="${VALUES[3]}"
export LUKA_ALTERNATING_SCRATCH_BASE="${VALUES[4]}"
HELPER="${PROJECT}/scripts/queue_luka_alternating_gate.sh"
bash "${HELPER}" "${COMMIT}" "${ROOT}" "${PARENT}" "${QUOTA}" science
read -r -p 'Paste reviewed SCIENCE PLAN content_hash VALUE: ' PLAN_HASH </dev/tty
[[ "${PLAN_HASH}" =~ ^[0-9a-f]{64}$ ]]
read -r -p 'Submit all 19 debug science jobs? Type YES: ' CONFIRM </dev/tty
if [[ "${CONFIRM}" != YES ]]; then echo 'Stopped; no jobs submitted.'; exit 0; fi
LOG="$(mktemp "${ROOT}/science-submit.XXXXXX.log")"
nohup bash "${HELPER}" "${COMMIT}" "${ROOT}" "${PARENT}" "${QUOTA}" science \
    --execute "${PLAN_HASH}" >"${LOG}" 2>&1 </dev/null &
printf 'Helper PID: %s\nLog: %s\nWatch with:\ntail -f "%s"\n' "$!" "${LOG}" "${LOG}"
BASH
```

## Results and live progress

Set PROJECT and ROOT to the printed exact paths. In the same environment as
the helper:

```bash
source /home/ryreu/miniconda3/etc/profile.d/conda.sh
conda activate /home/ryreu/miniconda3/envs/atlas_kd_sporc
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PROJECT}/src"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -u -B -s "${PROJECT}/scripts/luka_alternating_gate.py" results \
    --spec "${ROOT}/science/campaign_spec.json"
```

Completed nodes appear immediately; unfinished ones remain PENDING. Logs
`${ROOT}/science/train_NODE-JOBID.out` print each epoch's AUC, best AUC so far,
best epoch, gate loss and timing. For a particular exact log:

```bash
tail -f "${ROOT}/science/train_GATE_SELF_D066-JOBID.out"
```

Replace JOBID with the actual ID. Live numbers are provisional, not authenticated
selected results. Compare the three SELF_D000 endpoints and the M0HLT/OFFLINE/
DIRECT_D000 references; do not call a privileged TEACH row HLT-only. Retain every
row, including poor results. The preflight and all scientific training still need
real SPORC execution; local tests are not evidence of cluster acceptance.
