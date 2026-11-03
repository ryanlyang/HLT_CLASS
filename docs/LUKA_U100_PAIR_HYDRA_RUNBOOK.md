# Run the four U100 correction fits on Brown CS Hydra

**Current next step after job 131410:** run the separate
[Hydra precision diagnostic](LUKA_U100_PAIR_HYDRA_PRECISION_RUNBOOK.md).
Allocation now passes; numerical portability is unresolved. That runbook has
the current scoped commit/queue commands for a single inference job. The older
setup retry instructions below do not bypass this outstanding check.

The additive adapter is `src/hlt_classification/luka_u100_pair_hydra/`.
It reuses the original four-fit scientific runtime. This is Brown CS **Hydra**,
partition **gpus**, account **default**, job QoS **normal**, one L40 per job.
It is not Oscar and does not request a Hydra debug partition.

The user has already transferred and verified the 2.913 GiB bundle at:

```text
$HOME/hlt_u100_pairs/inputs_65952ed3/campaign-inputs.tar
```

The original code checkout `code_65952ed3` contains only the transfer helper.
Commit/push the additive Hydra implementation before using the commands below.
Keep the verified archive. No further SPORC data transfer is needed.

## Retry after setup allocation rejection

Job **131385** provides the exact cause for its rejection: all controller
resource fields match, including `NumTasks=1`, but the job environment exports
`SLURM_CLUSTER_NAME="(null)"`. The previous repair handled only an absent export.
The current repair also resolves this literal placeholder through the controller
and requires its configured cluster to be `hydra`. The regression reproduces
the supplied error before the change and passes afterward. Real L40 shared
setup still needs to run; the failed job never reached training inputs.

Job 131366 failed in the allocation guard, before loading training inputs or
running the setup miniature. Its controller record has expired. Accounting
confirmed hydra/gpus/default/normal, six CPUs, 120000 MiB and one GPU, but left
`NTasks` blank. That suggests a metadata difference; it does not identify the
exact field that failed. The repair cross-checks controller and batch evidence,
rejects contradictory values, and logs all relevant fields before admission.

Preserve both `runs/pair_hydra_63f31c2f_r1` and
`runs/pair_hydra_17a1719e_r1`, including their journals. Reuse
`environment_pair_hydra_v1` and `bundle_65952ed3`; do not reinstall or extract
again. Commit and push the repair from Windows, staging only these paths:

```powershell
Set-Location C:\Users\22rya\ComputerScience\CERN\HLT_Classification
if ((git branch --show-current) -ne 'main') { throw 'Use the intended main branch.' }
if (@(git diff --cached --name-only).Count -ne 0) { throw 'Review existing staged changes first.' }
$hydraRepairFiles = @(
    'src/hlt_classification/luka_u100_pair_hydra/site.py',
    'tests/test_luka_u100_pair_hydra.py',
    'docs/plans/LUKA_U100_PAIR_HYDRA_PLAN.md',
    'docs/contracts/LUKA_U100_PAIR_HYDRA.md',
    'docs/LUKA_U100_PAIR_HYDRA_RUNBOOK.md',
    'docs/LUKA_U100_PAIR_HYDRA_VALIDATION.md'
)
git add -- @hydraRepairFiles
if ($LASTEXITCODE -ne 0) { throw 'Staging failed.' }
git diff --cached --check
if ($LASTEXITCODE -ne 0) { throw 'Check the staged changes.' }
git diff --cached --stat
git commit -m 'Resolve null Hydra cluster exports using controller evidence'
if ($LASTEXITCODE -ne 0) { throw 'Commit failed.' }
git push origin main
if ($LASTEXITCODE -ne 0) { throw 'Push failed.' }
git rev-parse HEAD
```

Then paste this block into Brown with line breaks intact. It creates a new
checkout and campaign root for the new commit, using the existing inputs/env:

```bash
bash <<'BASH'
set -euo pipefail
read -r -p 'New pushed repair commit (40 characters): ' COMMIT </dev/tty
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
BASE="$(cd "${HOME}" && pwd -P)/hlt_u100_pairs"
PROJECT="${BASE}/code_hydra_${COMMIT:0:8}"
git -C "${BASE}/code_65952ed3" fetch origin main
git -C "${BASE}/code_65952ed3" merge-base --is-ancestor "${COMMIT}" origin/main
if [[ -e "${PROJECT}" ]]; then
    test "$(git -C "${PROJECT}" rev-parse HEAD)" = "${COMMIT}"
    test -z "$(git -C "${PROJECT}" status --porcelain --untracked-files=all)"
else
    git -C "${BASE}/code_65952ed3" worktree add --detach "${PROJECT}" "${COMMIT}"
fi
bash "${PROJECT}/scripts/queue_luka_u100_pair_hydra.sh" "${BASE}"
BASH
```

Complete the existing quota/plan review prompts. The detached dispatcher runs
one new shared setup; only successful numerical parity and miniature checks
automatically release the four fits. Real L40 setup remains unvalidated by
this local repair. If admission fails, preserve the log's `Hydra allocation
evidence` and detailed mismatch message; no extra diagnostic job is needed.

## 0. Commit/push the additive implementation from Windows

These explicit paths stage the new Hydra files, leaving existing edits to shared
documentation and unrelated code unstaged. The standalone validation file records
the donor and test evidence for this commit. Review any pre-existing staged work
separately before running this block.

```powershell
Set-Location C:\Users\22rya\ComputerScience\CERN\HLT_Classification
if ((git branch --show-current) -ne 'main') { throw 'Use the intended main branch before publishing.' }
if (@(git diff --cached --name-only).Count -ne 0) { throw 'Review the existing staged changes first.' }
$hydraFiles = @(
    'src/hlt_classification/luka_u100_pair_hydra',
    'scripts/luka_u100_pair_hydra.py',
    'scripts/setup_luka_u100_pair_hydra.sh',
    'scripts/queue_luka_u100_pair_hydra.sh',
    'sbatch/run_luka_u100_pair_hydra.sh',
    'tests/test_luka_u100_pair_hydra.py',
    'docs/plans/LUKA_U100_PAIR_HYDRA_PLAN.md',
    'docs/contracts/LUKA_U100_PAIR_HYDRA.md',
    'docs/LUKA_U100_PAIR_HYDRA_RUNBOOK.md',
    'docs/LUKA_U100_PAIR_HYDRA_VALIDATION.md'
)
git add -- @hydraFiles
if ($LASTEXITCODE -ne 0) { throw 'Staging failed.' }
git diff --cached --check
if ($LASTEXITCODE -ne 0) { throw 'Check the staged changes.' }
git diff --cached --stat
git commit -m 'Add portable Hydra execution for four U100 correction fits'
if ($LASTEXITCODE -ne 0) { throw 'Commit failed.' }
git push origin main
if ($LASTEXITCODE -ne 0) { throw 'Push failed; do not queue an unpublished commit.' }
git rev-parse HEAD
```

## 1. Check out the new pushed implementation on Brown

Run in the Brown terminal. Paste the block with its line breaks intact; `BASH`
must finish on a line by itself.

```bash
bash <<'BASH'
set -euo pipefail
read -r -p 'New pushed Hydra implementation commit (40 characters): ' COMMIT </dev/tty
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
BASE="$(cd "${HOME}" && pwd -P)/hlt_u100_pairs"
PROJECT="${BASE}/code_hydra_${COMMIT:0:8}"
git -C "${BASE}/code_65952ed3" fetch origin main
git -C "${BASE}/code_65952ed3" merge-base --is-ancestor "${COMMIT}" origin/main
if [[ -e "${PROJECT}" ]]; then
    test "$(git -C "${PROJECT}" rev-parse HEAD)" = "${COMMIT}"
    test -z "$(git -C "${PROJECT}" status --porcelain --untracked-files=all)"
else
    git -C "${BASE}/code_65952ed3" worktree add --detach "${PROJECT}" "${COMMIT}"
fi
printf 'PROJECT=%s\n' "${PROJECT}"
BASH
```

## 2. Create the separate HLT environment

Set PROJECT to the path printed above. Run once in a stable terminal or tmux;
this installation step is not a detached Slurm submission.

```bash
BASE="$(cd "${HOME}" && pwd -P)/hlt_u100_pairs"
PROJECT="${BASE}/code_hydra_REPLACE_WITH_EIGHT_CHARACTER_COMMIT"
bash "${PROJECT}/scripts/setup_luka_u100_pair_hydra.sh" \
    "${BASE}" "${BASE}/inputs_65952ed3/campaign-inputs.tar"
```

The helper installs unmanaged uv and Python3.10.21 under a new
`environment_pair_hydra_v1` directory, then an isolated venv with the recorded
packages. It does not edit shell startup files or other environments. It uses
the official [uv installer options](https://docs.astral.sh/uv/reference/installer/)
and [PyTorch2.5.1 CUDA11.8 wheel index](https://pytorch.org/get-started/previous-versions/).
The final check compares the complete recorded software environment, including
Weaver source hashes. The login host need not have a visible GPU for that check.
If it fails, preserve the directory and output; do not substitute newer packages
or erase the environment. Its exact executable is:

```text
$BASE/environment_pair_hydra_v1/venv/bin/python
```

## 3. Extract the verified inputs once

This creates a new ~2.913 GiB extracted copy. It preserves the original archive
and original metadata bytes. It does not reconstruct arrays or train models.

```bash
PYTHON="${BASE}/environment_pair_hydra_v1/venv/bin/python"
"${PYTHON}" -u -B -s "${PROJECT}/scripts/luka_u100_pair_hydra.py" unpack \
    --archive "${BASE}/inputs_65952ed3/campaign-inputs.tar" \
    --output "${BASE}/bundle_65952ed3"
```

An existing destination is refused, including a partial failed extraction.
Successful extraction publishes `extraction_receipt.json`; subsequent campaigns
reuse this shared bundle without extracting again.

## 4. Review and queue

After environment and extraction, reserve at least eight personally available
GiB for outputs, with an additional one GiB filesystem headroom. The fixed
campaign budget remains eight GiB even if more space is available.

```bash
bash "${PROJECT}/scripts/queue_luka_u100_pair_hydra.sh" "${BASE}"
```

The helper checks clean pushed source, authenticates inputs, creates a fresh
execution spec, and performs `sbatch --test-only` checks for the entire resource
envelope. Review its plan, paste PLAN_HASH and type YES. These final inputs
authorize the concrete automatic graph shown by the helper.

The detached dispatcher prints its PID, campaign directory and log. It survives
logout. Once its log says setup was submitted, Slurm owns that job; the dispatcher
remains responsible for automatically releasing the fits after successful setup.
Only setup appears in squeue initially. It validates numerical portability and
measures all four real production variants. There is no separate preflight job.

The two lanes are `p4_pair_u100` then `dual_u100`, and `pair_u100` then `p4_u100`.
Report follows all four. Dependencies between students are ordering-only
`afterany`; unsuccessful scientific results do not cancel later rows.
All four are fresh CE1/U100 embedding5 students with retained corrections.
The old U100-w5 reference and embedding bank are reused without retraining.

```bash
squeue --me -o "%.18i %.12P %.35j %.2t %.10M %R"
tail -f /absolute/campaign/path/submit.XXXXXX.log
```

Do not rerun the queue helper against an interrupted or partly submitted root.
Keep claims and journals for exact-ID diagnosis. Nothing in this adapter cancels
the original SPORC campaign or any unrelated Brown jobs.

## Results and code map

Use the same pinned checkout/environment as the execution:

```bash
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export LD_LIBRARY_PATH="${BASE}/environment_pair_hydra_v1/venv/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
"${PYTHON}" -u -B -s "${PROJECT}/scripts/luka_u100_pair_hydra.py" results \
    --spec /absolute/campaign/path/campaign_spec.json
```

- `inputs.py`: verified extraction, portable metadata/views, reused embedding
  bank, matched particle targets. No old absolute input paths are followed.
- `campaign.py`, `contracts.py`: pinned transfer/source, original recipe,
  Hydra site, quota and numerical parity limits.
- `site.py`: software equality, allocation checks, read-only Slurm dry review.
- `submission.py`: exact-job journals and automatic two-lane dispatch.
- `worker.py`: saved control parity, shared setup, four fits and results.
- Original `luka_u100_pair_correction/{model,geometry,losses,training,resources}.py`:
  unchanged scientific runtime and real miniature measurements.
- `tests/test_luka_u100_pair_hydra.py`: extraction integrity, portable real
  synthetic arrays/targets, control reload, science reuse and submission failures.

The comparison remains exploratory validation of 74,895 jets. Original50k
overlaps that validation and is report-only. Local synthetic tests do not
establish GPU feasibility or improved recovery. Actual Brown environment and
shared setup must pass before scientific fits are released.
