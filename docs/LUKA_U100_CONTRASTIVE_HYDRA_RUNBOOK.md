# Run the remaining contrastive fits independently on Hydra

This is the original CE1/U1005/OFFLINE-identification twenty-row campaign,
with the original thirty-epoch budget. Completed SPORC fits are reused; all
remaining students are submitted together after shared setup, without student
dependencies, lanes or a concurrency cap. Only the final report waits for fits.
Hydra scheduling/account quotas still determine actual starts.

Brown already has the HLT/U100 inputs and pinned environment. The one new
transfer is the OFFLINE key bank plus completed result metadata/predictions.
The exporter does no inference or scheduler mutation. It leaves SPORC jobs
alone; active/pending SPORC attempts can still run there, independently of new
Hydra attempts. The transfer output lists reused and remaining rows and records
the original accounting snapshot. Do not run the serial recovery helper for
this Hydra execution.

## 1. Commit and push from Windows PowerShell

```powershell
& {
    Set-Location C:\Users\22rya\ComputerScience\CERN\HLT_Classification
    if ((git branch --show-current) -ne 'main') { throw 'Expected main branch.' }
    if (@(git diff --cached --name-only).Count) { throw 'Review existing staged changes first.' }
    $contrastHydraFiles = @(
        'src/hlt_classification/luka_u100_contrastive_hydra',
        'scripts/luka_u100_contrastive_hydra.py',
        'scripts/queue_luka_u100_contrastive_hydra.sh',
        'sbatch/run_luka_u100_contrastive_hydra.sh',
        'tests/test_luka_u100_contrastive_hydra.py',
        'docs/plans/LUKA_U100_CONTRASTIVE_HYDRA_PLAN.md',
        'docs/contracts/LUKA_U100_CONTRASTIVE_HYDRA.md',
        'docs/LUKA_U100_CONTRASTIVE_HYDRA_RUNBOOK.md',
        'docs/LUKA_U100_CONTRASTIVE_HYDRA_VALIDATION.md'
    )
    git add -- @contrastHydraFiles
    if ($LASTEXITCODE -ne 0) { throw 'Staging failed.' }
    git diff --cached --check
    if ($LASTEXITCODE -ne 0) { throw 'Check staged changes.' }
    git diff --cached --stat
    git commit -m 'Run remaining contrastive fits independently on Hydra'
    if ($LASTEXITCODE -ne 0) { throw 'Commit failed.' }
    git push origin main
    if ($LASTEXITCODE -ne 0) { throw 'Push failed.' }
    git rev-parse HEAD
}
```

## 2. Export the small additional input directory on SPORC

No GPU is required. Use the original successful `431341db` campaign. Keep the
new 40-character commit and the printed 64-character `manifest_hash` for Brown.

```bash
bash <<'BASH'
set -euo pipefail
read -r -p 'New pushed contrastive Hydra commit: ' COMMIT </dev/tty
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
MAIN=/home/ryreu/atlas/HLT_Classification
PROJECT="${MAIN}_contrast_hydra_${COMMIT:0:8}"
OUT="/home/ryreu/atlas/luka_contrast_delta_${COMMIT:0:8}"
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
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
python -u -B -s "${PROJECT}/scripts/luka_u100_contrastive_hydra.py" export \
    --spec "${MAIN}/checkpoints/luka_u100_contrastive_431341db_debug_r1/campaign_spec.json" \
    --output "${OUT}" --expected-commit "${COMMIT}"
BASH
```

The final manifest is the publication marker; partial export directories are
not reusable. If a newly COMPLETED job lacks a published receipt, the exporter
stops rather than assuming it can be skipped or reused. Preserve that output
for diagnosis. No targets or controls are retrained.

## 3. Copy the delta and queue on Brown

Paste with newlines intact. The transfer login uses your normal SPORC password
and Duo flow. Enter the manifest hash printed by the SPORC exporter, not a hash
invented from the destination files.

```bash
bash <<'BASH'
set -euo pipefail
read -r -p 'Same pushed contrastive Hydra commit: ' COMMIT </dev/tty
read -r -p 'SPORC exporter manifest_hash (64 characters): ' DELTA_HASH </dev/tty
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ && "${DELTA_HASH}" =~ ^[0-9a-f]{64}$ ]]
BASE="$(cd "${HOME}" && pwd -P)/hlt_u100_pairs"
PROJECT="${BASE}/code_contrast_hydra_${COMMIT:0:8}"
DELTA="${BASE}/contrast_delta_${COMMIT:0:8}"
git -C "${BASE}/code_65952ed3" fetch origin main
git -C "${BASE}/code_65952ed3" merge-base --is-ancestor "${COMMIT}" origin/main
if [[ -e "${PROJECT}" ]]; then
    test "$(git -C "${PROJECT}" rev-parse HEAD)" = "${COMMIT}"
    test -z "$(git -C "${PROJECT}" status --porcelain --untracked-files=all)"
else
    git -C "${BASE}/code_65952ed3" worktree add --detach "${PROJECT}" "${COMMIT}"
fi
if [[ ! -e "${DELTA}" ]]; then
    scp -r "ryreu@sporcsubmit.rc.rit.edu:/home/ryreu/atlas/luka_contrast_delta_${COMMIT:0:8}" "${DELTA}"
fi
ENV_DIR="${BASE}/environment_pair_hydra_v1/venv"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="${ENV_DIR}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
"${ENV_DIR}/bin/python" -B -s "${PROJECT}/scripts/luka_u100_contrastive_hydra.py" verify \
    --delta "${DELTA}" --delta-hash "${DELTA_HASH}"
bash "${PROJECT}/scripts/queue_luka_u100_contrastive_hydra.sh" "${BASE}" "${DELTA}" "${DELTA_HASH}"
BASH
```

Reserve at least 8 GiB **additional** outputs, after other campaign reservations
and the transfer. The helper checks actual free space with 1 GiB headroom, prints
the full dry plan and launches a detached dispatcher under the user's campaign
authorization. No additional confirmation or separate preflight is queued.

Use the printed `tail -f` command. Once it says `Setup submitted as ... Automatic
independent-fit submission is active; safe to log off.`, the detached dispatcher
waits for that exact successful setup and automatically submits every remaining
student plus report. All student jobs then have no Slurm dependencies. A failure
in one student cannot hold other students. A failed setup releases no students.

The shared setup reuses the completed numerical diagnostic 131484, verifies
local targets and runs the six original real production miniatures/resource
checks on one L40. It does not perform teacher export or repeat control inference.
That measured acceptance is still required; local tests are not its replacement.

## Results and code

Run with the new checkout and existing environment:

```bash
"${ENV_DIR}/bin/python" -B -s "${PROJECT}/scripts/luka_u100_contrastive_hydra.py" \
    results --spec "${BASE}/runs/contrast_hydra_${COMMIT:0:8}_${DELTA_HASH:0:8}_r1/campaign_spec.json"
```

Variables in the previous `bash` block are scoped to it. For later use, set them
again or use the absolute campaign path printed at submission. Results include
all reused completions and available new completions, site labels, selected
epochs, matched-seed deltas and historical gap recovery. JSON additionally holds
the Hydra seed0 replay delta. This is mixed-site developmental validation, with
overlapping original50k report-only. No independent-test access.

Code is under `src/hlt_classification/luka_u100_contrastive_hydra/`: `transfer.py`
authenticates and copies the delta, `inputs.py` binds relocated keys, `campaign.py`
owns immutable registration, `submission.py` owns the independent graph, and
`worker.py` calls unchanged original contrastive training/resources and reports
combined results. The pair-Hydra environment/site/input/precision components are
reused. [Plan](plans/LUKA_U100_CONTRASTIVE_HYDRA_PLAN.md),
[contract](contracts/LUKA_U100_CONTRASTIVE_HYDRA.md),
[validation and donor record](LUKA_U100_CONTRASTIVE_HYDRA_VALIDATION.md).
