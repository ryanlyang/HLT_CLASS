# Queue cold-start privileged-gate ladders on SPORC tier3

Scientific authority: [plan](plans/LUKA_COLD_GATE_TIER3_PLAN.md) and
[contract](contracts/LUKA_COLD_GATE_TIER3.md). This is a NEW campaign, not a
restart of the running warm/debug campaign. No old jobs or data are changed.

Same U100 anchor, 299234 TRAIN / 74895 validation, fixed30/batch128. Three fresh
D066 -> D033 -> D000 chains (LOGIT, SELF, PRIV); cold REPLACE_LOGIT,
REPLACE_GATE, SELF_REDISTILL. Twelve fits plus aggregate. PRIV is an oracle;
only LOGIT_D000, SELF_D000 and the replacement endpoints are HLT-only.
Historical test is consumed; this is single-seed exploratory validation.

## What stays on disk

Every fit retains one selected checkpoint, all epoch metrics, validation
probabilities/identities/labels/original50k membership and provenance. No
per-epoch checkpoint, optimizer snapshot, dense attention bank, or persistent
teacher bank. A consumer exports its own teacher targets into private
node-local scratch, then removes ONLY those exact files after verifying its
saved selected outputs. Different Slurm nodes need not share scratch.
U100, prepared input caches and all existing campaigns are read-only.

Declare at least **2 GiB of personally available persistent space** for this
new campaign (not filesystem-wide free capacity). The gate also measures a
separate local scratch allowance: 36 bytes per TRAIN particle, 128 bytes per
TRAIN row, plus 3 GiB reserve. This is conservative even for logit-only jobs.
Default scratch is `/tmp`; set `LUKA_COLD_SCRATCH_BASE=/local/scratch` before
both dry/live helper invocations if that canonical local directory is provided
on the actual nodes. Shared filesystems and missing space fail closed, with
no home fallback. No existing files are cleaned up. Failed jobs preserve their
partial outputs; local scratch may independently be purged by the site.

## 1. Commit and push on Windows

Review `git status` and the staged diff. Do not `git add .` or include unrelated
changes. The exact source checker requires all these new files committed:

```powershell
git add -- src/hlt_classification/luka_gating_cold scripts/luka_gating_cold.py scripts/queue_luka_gating_cold.sh sbatch/run_luka_gating_cold.sh tests/test_luka_gating_cold.py tests/test_luka_gating_cold_followup.py docs/plans/LUKA_COLD_GATE_TIER3_PLAN.md docs/contracts/LUKA_COLD_GATE_TIER3.md docs/LUKA_COLD_GATE_TIER3_RUNBOOK.md
if ($LASTEXITCODE -ne 0) { throw 'Staging failed' }
git diff --cached --stat
git diff --cached --check
if ($LASTEXITCODE -ne 0) { throw 'Review staged changes' }
git commit -m "Add cold gate tier3 ladders with per-job scratch targets"
if ($LASTEXITCODE -ne 0) { throw 'Commit failed; inspect before continuing' }
git push origin HEAD:main
if ($LASTEXITCODE -ne 0) { throw 'Push failed; do not force-push' }
git rev-parse HEAD
```

Use `git add -p` separately for only this campaign's HANDOFF/LEGACY_SOURCE_MAP
entries if those files contain unrelated work. The implementation pins existing
scientific source to donor `84fcbf25529091a2f32daf96bf8b910dec50dc3f` and permits
only the new cold package's source additions. If concurrent work changes other
`src` modules, reconcile and retest before advancing that compatibility pin.

## 2. Dry review, then optional automatic gate-to-science on sporcsubmit

No direct SSH from an agent or automatic submission by local tests. Paste on
`sporcsubmit`, not Tigris. This block uses `/dev/tty` for prompts and keeps the
helper alive after disconnection. It defaults to a fresh `_r1` root; preserve
partial roots and inspect errors rather than replacing them or retrying blindly.

```bash
bash <<'BASH'
set -euo pipefail
trap 'echo "Stopped; preserve everything and paste the error." >&2' ERR
read -r -p 'Paste the NEW pushed 40-character commit: ' COMMIT </dev/tty
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
MAIN=/home/ryreu/atlas/HLT_Classification
PROJECT="/home/ryreu/atlas/HLT_Classification_lukacold_${COMMIT:0:8}"
ROOT="${MAIN}/checkpoints/luka_cold_gate_${COMMIT:0:8}_r1"
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
read -r -p 'Personally available persistent storage (whole GiB, minimum 2): ' QUOTA </dev/tty
[[ "${QUOTA}" =~ ^[1-9][0-9]*$ && "${QUOTA}" -ge 2 ]]
HELPER="${PROJECT}/scripts/queue_luka_gating_cold.sh"
bash "${HELPER}" "${COMMIT}" "${ROOT}" "${PARENT}" "${QUOTA}" auto
read -r -p 'Paste the printed FOLLOWUP_PLAN_HASH value (64 hex characters): ' PLAN_HASH </dev/tty
[[ "${PLAN_HASH}" =~ ^[0-9a-f]{64}$ ]]
read -r -p 'Queue the tier3 preflight and authorize 13 science jobs after verified success? Type YES: ' CONFIRM </dev/tty
if [[ "${CONFIRM}" != YES ]]; then echo 'Stopped; no jobs submitted.'; exit 0; fi
LOG="$(mktemp "${ROOT}/auto-followup.XXXXXX.log")"
nohup bash "${HELPER}" "${COMMIT}" "${ROOT}" "${PARENT}" "${QUOTA}" auto \
    --execute "${PLAN_HASH}" >"${LOG}" 2>&1 </dev/null &
printf 'Controller PID: %s\nLog: %s\nWatch with:\ntail -f "%s"\n' "$!" "${LOG}" "${LOG}"
BASH
```

Wait for `ARMED: preflight ...` in the log. `Ctrl-C` stops `tail`, not the
controller. Science is submitted only after successful exact preflight
accounting AND authenticated technical/resource evidence. A bad score never
blocks science. A failed job, corrupt artifact, stale source or insufficient
resources does. No automatic retry; do not also submit science manually.

Both phases: tier3/A100, 6 CPUs,128000MiB. Gate8h. Science fit walltimes are
separately measured, capped24h; aggregate1CPU/16384MiB/1h. This run may coexist
with debug jobs, but queue availability and account limits still apply.

## Manual alternative

Use helper `gate` instead of `auto` for dry review and `gate --execute HASH`
for only the one preflight. After it passes, use `science` for dry review,
then `science --execute HASH` to submit the measured13-job DAG. This is an
alternative to automatic follow-up, never an additional submitter.

## Results and failures

Activate the same environment as the queue helper, then:

```bash
source /home/ryreu/miniconda3/etc/profile.d/conda.sh
conda activate /home/ryreu/miniconda3/envs/atlas_kd_sporc
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PROJECT}/src"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -u -B -s "${PROJECT}/scripts/luka_gating_cold.py" results \
    --spec "${ROOT}/science/campaign_spec.json"
```

Set `PROJECT` and `ROOT` to the exact new paths first if opening a new shell;
the queue block's variables were local to its subshell. Results authenticate
retained artifacts without requiring retired scratch or live Slurm accounting.
Full74895 validation selects the checkpoint; original50000 is report-only.
No warm/cold claim of significance from one seed; compare identical populations
and architectures, including SELF_REDISTILL's extra training-stage control.

Inspect `${ROOT}/preflight-JOBID.out`, `${ROOT}/preflight/` and followup failure
receipts on any failure. Do not delete partials, create a second submitter,
raise limits, truncate particles or shorten the schedule to bypass admission.
Real A100 full-population acceptance remains required even after local tests.
