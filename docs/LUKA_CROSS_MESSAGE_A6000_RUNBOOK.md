# Move the cross-message campaign to Hydra A6000

Same two teachers and twelve students; one RTX A6000, six CPUs,32000MiB host
RAM per job. The scheduler snapshot on2026-11-04 showed six unallocated A6000s
and no unallocated L40s. That is an opportunity, not a guaranteed start time.
L40 setup measured13.3GiB host RSS and6.3GiB peak GPU memory. A6000 setup must
check the smaller envelope and measure its own runtime before releasing fits.

No new transfer, teacher selection, data population, batch size, loss, seed,
precision recipe or control training is introduced. The original L40 jobs and
artifacts are unchanged by these local code edits. `SPEC/v2` distinguishes
A6000 execution. The [scientific plan](plans/LUKA_CROSS_MESSAGE_HYDRA_PLAN.md)
and [contract](contracts/LUKA_CROSS_MESSAGE_HYDRA.md) define qualification.

The migration has two stages: queue **setup only** while retaining old pending
jobs, then cancel the exact old pending fits and release the new graph once
setup passes. This avoids losing the old queue position if qualification fails.
Do not run both scientific campaigns unintentionally.

## Commit and push on Windows

Stage only these paths; preserve unrelated work:

```powershell
Set-Location 'C:\Users\22rya\ComputerScience\CERN\HLT_Classification'
git add -- src/hlt_classification/luka_cross_message_hydra scripts/luka_cross_message_hydra.py scripts/queue_luka_cross_message_hydra.sh tests/test_luka_cross_message_hydra.py tests/test_luka_cross_message_a6000.py docs/LUKA_CROSS_MESSAGE_A6000_RUNBOOK.md docs/LUKA_CROSS_MESSAGE_HYDRA_RUNBOOK.md docs/LUKA_CROSS_MESSAGE_HYDRA_VALIDATION.md docs/plans/LUKA_CROSS_MESSAGE_HYDRA_PLAN.md docs/contracts/LUKA_CROSS_MESSAGE_HYDRA.md
if ($LASTEXITCODE -ne 0) { throw 'Staging failed' }
git diff --cached --stat
git diff --cached --check
if ($LASTEXITCODE -ne 0) { throw 'Check the staged changes' }
git commit -m 'Add qualified A6000 execution for cross-message campaign'
if ($LASTEXITCODE -ne 0) { throw 'Commit failed' }
git push origin main
if ($LASTEXITCODE -ne 0) { throw 'Push failed' }
git rev-parse HEAD
```

`docs/HANDOFF.md` and `docs/LEGACY_SOURCE_MAP.md` also contain additive evidence;
stage those additions selectively with `git add -p` if including shared docs.
Source validation permits changes only in the cross-message runtime relative
to committed base `a7249f49b6f522f7df3b150d27f53613452b7fb0`.

## Queue A6000 shared setup on Brown

Paste the new40-character commit. This reuses the exact existing encoder and
validation delta paths from the old spec; no SPORC access is needed.
Eight GiB is the output allowance, separate from environment/input storage.

```bash
bash <<'BASH'
set -euo pipefail
read -r -p 'New pushed A6000 commit (40 characters): ' COMMIT </dev/tty
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
BASE="$(cd "${HOME}" && pwd -P)/hlt_u100_pairs"
MAIN="${BASE}/code_65952ed3"
OLD_SPEC="${BASE}/runs/cross_message_73abecfd_3693bcda_r1/campaign_spec.json"
PROJECT="${BASE}/code_cross_message_${COMMIT:0:8}"
test -s "${OLD_SPEC}"
git -C "${MAIN}" fetch origin main
git -C "${MAIN}" merge-base --is-ancestor "${COMMIT}" origin/main
if [[ -e "${PROJECT}" ]]; then
    test "$(git -C "${PROJECT}" rev-parse HEAD)" = "${COMMIT}"
    test -z "$(git -C "${PROJECT}" status --porcelain --untracked-files=all)"
else
    git -C "${MAIN}" worktree add --detach "${PROJECT}" "${COMMIT}"
fi
ENCODER="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["encoder_root"])' "${OLD_SPEC}")"
bash "${PROJECT}/scripts/queue_luka_cross_message_hydra.sh" "${BASE}" "${OLD_SPEC}" "${ENCODER}" 8 a6000 setup-only
BASH
```

After `Setup submitted as ... Setup only`, it is safe to log off. No watcher
needs to stay alive and no new scientific jobs will auto-submit in this mode.
The campaign directory ends `_a6000_r1`. Setup includes full native-control
replay, CPU/GPU FP32 and BF16 checks, U100 encoder replay, and all14 production
miniatures. It does not retrain the reference. The memory ceiling is85% of
32000MiB; the2x fit-time safety factor is retained and remeasured on A6000.

For a genuinely new campaign with no old duplicate jobs, replace `setup-only`
by `all` to enable the existing detached automatic science dispatcher.

## After A6000 setup succeeds: replace pending jobs and release science

Supply the exact new directory. This first authenticates A6000 setup, then
authenticates the old source/ledger in its original worktree. Cancellation is
limited to all15 exact old science/report IDs, only if all are still pending,
owned by you, and carry the correct task names and campaign comments. The old
completed setup is excluded. If any old fit started, stop and inspect it;
this command does not discard its work or create a mixed partial campaign.

```bash
bash <<'BASH'
set -euo pipefail
read -r -p 'Exact NEW A6000 campaign directory: ' ROOT </dev/tty
SPEC="${ROOT}/campaign_spec.json"
BASE="$(cd "${HOME}" && pwd -P)/hlt_u100_pairs"
OLD_SPEC="${BASE}/runs/cross_message_73abecfd_3693bcda_r1/campaign_spec.json"
PROJECT="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["project_dir"])' "${SPEC}")"
ENV_DIR="${BASE}/environment_pair_hydra_v1/venv"
PYTHON="${ENV_DIR}/bin/python"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="${ENV_DIR}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
"${PYTHON}" -B -s "${PROJECT}/scripts/luka_cross_message_hydra.py" check-setup --spec "${SPEC}"
"${PYTHON}" -B -s - "${OLD_SPEC}" "${SPEC}" <<'PY'
import getpass, json, subprocess, sys, time
from pathlib import Path
old = json.loads(Path(sys.argv[1]).read_text())
new = json.loads(Path(sys.argv[2]).read_text())
if new.get('gpu_profile') != 'a6000' or old['root'] == new['root']:
    raise SystemExit('Expected a separate qualified A6000 campaign')
sys.path.insert(0, str(Path(old['project_dir']) / 'src'))
from hlt_classification.luka_cross_message_hydra import campaign, submission
from hlt_classification.luka_cross_message_hydra.contracts import TASKS
campaign.check(old, check_storage=False)
jobs = submission.accepted_jobs(old, 'science')
if set(jobs) != {*TASKS, 'report'} or len(set(jobs.values())) != 15:
    raise SystemExit('Old ledger is incomplete or ambiguous; nothing cancelled')
ids = ','.join([jobs['report']] + [jobs[t] for t in TASKS if not t.startswith('teacher_')] +
               [jobs[t] for t in TASKS if t.startswith('teacher_')])
def query(fmt):
    output = subprocess.run(['squeue', '-h', '-u', getpass.getuser(), '-o', fmt],
                            check=True, capture_output=True, text=True).stdout
    return '\n'.join(r for r in output.splitlines() if r.split('|', 1)[0].strip() in set(jobs.values()))
rows = [[field.strip() for field in r.split('|')] for r in query('%i|%u|%j|%T|%k').splitlines()]
expected = {job:task for task,job in jobs.items()}
if len(rows) != 15 or {r[0] for r in rows} != set(expected):
    raise SystemExit('Not all old jobs are pending; inspect before replacing')
for job, user, name, state, comment in rows:
    task = expected[job]
    if (user != getpass.getuser() or state != 'PENDING' or name != 'lukacm_'+task or
            comment != 'lukacm:'+old['content_hash']+':'+task):
        raise SystemExit('Old job ownership/state differs; nothing cancelled')
print('Cancelling only old pending campaign jobs: '+ids, flush=True)
subprocess.run(['scancel', '--state=PENDING', *ids.split(',')], check=True)
for _ in range(12):
    if not query('%i|%T'):
        break
    time.sleep(5)
else:
    raise SystemExit('Old jobs still present; do not release duplicate science')
print('Old pending jobs removed. Saved artifacts preserved.')
PY
PLAN_HASH="$("${PYTHON}" -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["content_hash"])' "${ROOT}/submission/review.json")"
"${PYTHON}" -B -s "${PROJECT}/scripts/luka_cross_message_hydra.py" finish --spec "${SPEC}" \
    --plan-hash "${PLAN_HASH}" --authorization AUTHORIZE_CROSS_MESSAGE_HYDRA_V1
BASH
```

Keep `finish` connected until it returns the ledger. Once all15 jobs are recorded,
Slurm owns them. Teachers and three architecture controls can start independently;
nine supervised fits wait only for their teacher. Results/progress commands in
the [original runbook](LUKA_CROSS_MESSAGE_HYDRA_RUNBOOK.md) take the new spec.
Results compare to the newly replayed A6000 control. Epoch control curves remain
the historical SPORC training curve, clearly labelled. No independent test exists.

If cancellation or submission stops, preserve journals and inspect current
states; do not repeat the whole migration blindly. `finish` can safely return an
already complete ledger, but partial submissions retain claims for inspection.

Implementation: `execution.py` supplies A6000 allocation checks and numerical
replay, `campaign.py` registers resources and version2, `submission.py` retains
the exact DAG and adds setup-only dispatch. Original scientific `model.py`,
`training.py` and the resource time formula are unchanged.
