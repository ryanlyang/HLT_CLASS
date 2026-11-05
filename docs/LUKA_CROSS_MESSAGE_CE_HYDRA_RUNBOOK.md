# Two CE-only cross-message controls on Hydra

This adds `m32_ce_only_s0` and `m16_ce_only_s0` alongside the existing A6000
campaign. Each is the same HLT message architecture/seed as its CE+U100-w5
control, with the embedding head removed and CE as its only loss. See the
[plan](plans/LUKA_CROSS_MESSAGE_CE_HYDRA_PLAN.md) and
[contract](contracts/LUKA_CROSS_MESSAGE_CE_HYDRA.md).

There are exactly two new independent jobs, named `lukac0_*`, with no teacher
dependencies or new report/setup job. Existing jobs and specs are untouched.
Reuse the authenticated parent setup144840 and its per-control A6000 timings.
Each new allocation runs a disposable warmup and then a fresh measured CE-only
miniature before starting fresh full training. Both request one RTX A6000,6CPUs/32000MiB; thirty epochs,
batch128, identical TRAIN299234/validation74895. Reserve two additional GiB
for outputs, beyond space reserved for the active parent campaign.

## Retry after jobs145718 and145719

Both jobs completed a technical miniature but failed the runtime projection:
924 versus421 minutes for width32,896 versus425 for width16. These were
first-process, three-batch estimates, not actual thirty-epoch runtime or model
quality failures. No full CE-only fit started.

Execution v2 warms the CE-only path before timing a second fresh miniature.
Warmup overhead counts once, rather than once per projected batch. The2x
runtime margin, parent-derived walltime requests and85% memory limits remain.
All three models (warmup, measured miniature, full fit) start from identical
matched weights; no technical updates carry into science.

Commit/push the fix and use the normal queue block below with the **new commit**.
It creates a new worktree and campaign root. Keep
`cross_message_ce_5a3509eb_b29184c0_r1` and its diagnostics. Neither failed job
needs cancellation. Do not change the active parent campaign or delete claims.

## Commit and push on Windows

Stage only the additive files below. Shared README/HANDOFF/source-map additions
can be staged selectively; other existing edits must remain untouched.

```powershell
Set-Location 'C:\Users\22rya\ComputerScience\CERN\HLT_Classification'
git add -- src/hlt_classification/luka_cross_message_ce_hydra scripts/luka_cross_message_ce_hydra.py scripts/queue_luka_cross_message_ce_hydra.sh sbatch/run_luka_cross_message_ce_hydra.sh tests/test_luka_cross_message_ce_hydra.py docs/plans/LUKA_CROSS_MESSAGE_CE_HYDRA_PLAN.md docs/contracts/LUKA_CROSS_MESSAGE_CE_HYDRA.md docs/LUKA_CROSS_MESSAGE_CE_HYDRA_RUNBOOK.md
if ($LASTEXITCODE -ne 0) { throw 'Staging failed' }
git diff --cached --stat
git diff --cached --check
if ($LASTEXITCODE -ne 0) { throw 'Check the staged changes' }
git commit -m 'Warm CE-only miniature before projecting Hydra runtime'
if ($LASTEXITCODE -ne 0) { throw 'Commit failed' }
git push origin main
if ($LASTEXITCODE -ne 0) { throw 'Push failed' }
git rev-parse HEAD
```

## Queue on Brown

Paste the new40-character commit when prompted. Use the existing environment,
bundle and parent source worktree; no transfer or environment installation.

```bash
bash <<'BASH'
set -euo pipefail
read -r -p 'New pushed CE-only commit (40 characters): ' COMMIT </dev/tty
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
BASE="$(cd "${HOME}" && pwd -P)/hlt_u100_pairs"
MAIN="${BASE}/code_65952ed3"
PARENT_SPEC="${BASE}/runs/cross_message_199928da_3693bcda_a6000_r1/campaign_spec.json"
PROJECT="${BASE}/code_cross_message_ce_${COMMIT:0:8}"
git -C "${MAIN}" fetch origin main
git -C "${MAIN}" merge-base --is-ancestor "${COMMIT}" origin/main
if [[ -e "${PROJECT}" ]]; then
    test "$(git -C "${PROJECT}" rev-parse HEAD)" = "${COMMIT}"
    test -z "$(git -C "${PROJECT}" status --porcelain --untracked-files=all)"
else
    git -C "${MAIN}" worktree add --detach "${PROJECT}" "${COMMIT}"
fi
bash "${PROJECT}/scripts/queue_luka_cross_message_ce_hydra.sh" "${PARENT_SPEC}" 2
BASH
```

The launcher authenticates the original setup in its original source, creates
a separate spec, runs a two-job scheduler dry review and starts a detached
submission. Watch the printed log until it says **Both CE-only controls submitted
independently** and displays both IDs. Then logoff requires no watcher.
If a submit stops, keep claims/journals. A complete ledger is idempotent; a
partial/active claim requires inspection, not deleting files or blind retry.

## Results and epoch progress

Use the exact new root printed by the launcher. `progress` shows each epoch
against the matching parent architecture+embedding control and historical
U100-w5, with M0HLT/OFFLINE recovery and last logged batch percentage. `results`
authenticates only completed fits and recomputes saved metrics. Neither runs
inference nor changes files/jobs.

```bash
bash <<'BASH'
set -euo pipefail
read -r -p 'Exact CE-only campaign directory: ' ROOT </dev/tty
read -r -p 'Command (progress or results): ' COMMAND </dev/tty
[[ "${COMMAND}" == progress || "${COMMAND}" == results ]]
SPEC="${ROOT}/campaign_spec.json"
PROJECT="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["project_dir"])' "${SPEC}")"
ENV_DIR="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["environment_dir"])' "${SPEC}")"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="${ENV_DIR}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
"${ENV_DIR}/bin/python" -B -s "${PROJECT}/scripts/luka_cross_message_ce_hydra.py" "${COMMAND}" --spec "${SPEC}"
BASH
```

Full validation is exploratory, original50k overlaps it, and no independent test
exists. Historical M0HLT is a SPORC reference rather than a newly trained A6000
CE baseline. Compare matched seeds and architecture before attributing gains.

## Code and validation

`src/hlt_classification/luka_cross_message_ce_hydra/`: contracts register the two
tasks; model subclasses the unchanged message architecture; training adapts the
donor fit/publication loop; campaign authenticates parent/source/storage;
worker performs the integrated miniature; submission journals two independent
IDs; reporting reads epochs and authenticated final predictions. CLIs and
Slurm/queue wrappers are thin. Donor commit is
`199928da991f8a4194dcf42df009130cfcf84118`; old runtime is not edited.

Local verification and the remaining real-allocation check are recorded in
`docs/HANDOFF.md`. Local tests do not assert that the remote jobs were submitted
or that either new scientific control has completed.

Timing-fix validation: pre-edit24 passed/30.39s; final28 passed/36.85s, including
installed-Weaver CUDA warmup/measured full and partial batches, fresh weights
across all three phases, authenticated warmup provenance, simulated924/896-minute
regressions, and retained memory/runtime/nonfinite checks. The suite also covers
synthetic full30 fits, independent submission and read-only results. The original
addon implementation separately passed all60 cross-message/A6000 regressions;
those donor workers remain unchanged. Real Hydra acceptance of the corrected
timing path is still required inside the two new allocations.
